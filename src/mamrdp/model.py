from scipy.signal import firwin, minimum_phase
import torch
from torch import nn
from torch.nn import functional as F

IMPLEMENTATION_REVISION = 3


def require_current_revision(config):
    if config.get("implementation_revision") not in (2, IMPLEMENTATION_REVISION):
        raise ValueError()
    if config.get("implementation_revision") == 3:
        required = {"filter_initialization", "contrastive_normalize", "gate_initialization"}
        if not required.issubset(config):
            raise ValueError("Requires explicit initialization and normalization settings")


class FrequencyBranch(nn.Module):
    def __init__(self, channels, length, band, sample_rate, initialization="linear_phase"):
        super().__init__()
       
        taps = firwin(length, band, pass_zero=False, fs=sample_rate)
        if initialization == "minimum_phase":
            taps = minimum_phase(taps, method="homomorphic", half=False)
        elif initialization != "linear_phase":
            raise ValueError("filter_initialization must be linear_phase or minimum_phase")
        self.weight = nn.Parameter(torch.tensor(taps, dtype=torch.float32)[None, None])      
        self.bias = nn.Parameter(torch.zeros(1))
        self.channels = channels

    def forward(self, x):
        weights = self.weight.flip(-1).expand(self.channels, -1, -1)
        padded = F.pad(x, (weights.shape[-1] - 1, 0))
        return F.conv1d(padded, weights, self.bias.expand(self.channels), groups=self.channels)


def conv_block(cin, cout, kernel, stride=1):
    return nn.Sequential(nn.Conv1d(cin, cout, kernel, stride=stride, padding=kernel // 2,
                                  bias=False), nn.BatchNorm1d(cout), nn.ELU())


class MAMRDPNet(nn.Module):
    def __init__(self, channels, sample_rate, config):
        super().__init__()
        require_current_revision(config)
        self.config = config
        self.channels = channels
        self.sample_rate = sample_rate
        width, dim = config["branch_channels"], config["feature_dim"]
        drop = config["dropout"]
        self.filters = nn.ModuleList([
            FrequencyBranch(channels, length, band, sample_rate,
                            config.get("filter_initialization", "linear_phase"))
            for length, band in zip(config["filter_lengths"], config["bands"])])
        self.frequency_fusion = nn.Sequential(
            conv_block(channels * len(self.filters), width, 1),
            nn.AvgPool1d(config["pool_size"]), nn.Dropout(drop))
        k1, k2 = config["temporal_kernels"]
        self.temporal = nn.Sequential(conv_block(channels, width, k1),
                                      conv_block(width, width, k2),
                                      nn.AvgPool1d(config["pool_size"]), nn.Dropout(drop))
        self.resolutions = nn.ModuleList([
            nn.Sequential(conv_block(2 * width, dim, k, stride), nn.Dropout(drop),
                          nn.AdaptiveAvgPool1d(1))
            for k, stride in zip(config["resolution_kernels"], config["resolution_strides"])])

        self.attention = nn.MultiheadAttention(dim, config["attention_heads"],
                                              dropout=0.0, bias=False, batch_first=True)
        self.gate = nn.Linear(2 * dim, dim)
        if config.get("gate_initialization", "pytorch") == "balanced":
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
        self.projection = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.enhance = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.classifier = nn.Linear(dim, 2)
        freqs = torch.fft.rfftfreq(config["frequency_nfft"], d=1 / sample_rate)
        masks = []
        for index, (low, high) in enumerate(config["bands"]):
            mask = (freqs >= low) & ((freqs <= high) if index == len(config["bands"]) - 1 else (freqs < high))
            if not mask.any():
                raise ValueError("FFT grid has no in-band bins; increase frequency_nfft")
            masks.append(mask)
        self.register_buffer("band_masks", torch.stack(masks))

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != self.channels:
            raise ValueError(f"Expected batch x {self.channels} x samples, got {tuple(x.shape)}")
        frequency = self.frequency_fusion(torch.cat([branch(x) for branch in self.filters], dim=1))
        temporal = self.temporal(x)
        tf = torch.cat([temporal, frequency], dim=1)
        resolutions = torch.stack([branch(tf).squeeze(-1) for branch in self.resolutions], dim=1)
        attended, _ = self.attention(resolutions, resolutions, resolutions, need_weights=False)
        original, fused = resolutions.mean(dim=1), attended.mean(dim=1)
        gate = torch.sigmoid(self.gate(torch.cat([original, fused], dim=-1)))
        projected = self.projection(gate * fused + (1 - gate) * original)
        
        z = F.normalize(projected, dim=-1) if self.config.get("contrastive_normalize", False) else projected
        logits = self.classifier(self.enhance(projected))
        return logits, z

    def frequency_regularization(self):
        loss = self.filters[0].weight.new_zeros(())
        for branch, mask in zip(self.filters, self.band_masks):
            response = torch.fft.rfft(branch.weight.flatten(), n=self.config["frequency_nfft"]).abs()
            loss = loss + (1 - response[mask]).square().sum()
            loss = loss + self.config["out_of_band_weight"] * response[~mask].square().sum()
        return loss


class MemoryQueue(nn.Module):
    def __init__(self, capacity, dim, temperature):
        super().__init__()
        self.capacity = capacity
        self.temperature = temperature
        self.register_buffer("features", torch.zeros(capacity, dim))
        self.register_buffer("labels", torch.full((capacity,), -1, dtype=torch.long))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))
        self.register_buffer("pointer", torch.zeros((), dtype=torch.long))

    def loss(self, features, labels):
        count = int(self.count.item())
        if count == 0:
            return features.sum() * 0
        keys = self.features[:count].detach().clone()
        key_labels = self.labels[:count].detach().clone()
        similarity = features @ keys.T / self.temperature
        positives = labels[:, None] == key_labels[None, :]
        valid = positives.any(dim=1)
        if not valid.any():
            return features.sum() * 0
        log_prob = F.log_softmax(similarity[valid], dim=1)
        mask = positives[valid]
        return -((log_prob * mask).sum(dim=1) / mask.sum(dim=1)).mean()

    @torch.no_grad()
    def enqueue(self, features, labels):
        features = features.detach()
        labels = labels.detach()
        if len(labels) >= self.capacity:
            self.features.copy_(features[-self.capacity:])
            self.labels.copy_(labels[-self.capacity:])
            self.count.fill_(self.capacity)
            self.pointer.zero_()
            return
        n, pointer = len(labels), int(self.pointer.item())
        indices = (torch.arange(n, device=self.labels.device) + pointer) % self.capacity
        self.features[indices] = features
        self.labels[indices] = labels
        self.pointer.fill_((pointer + n) % self.capacity)
        self.count.fill_(min(self.capacity, int(self.count.item()) + n))
