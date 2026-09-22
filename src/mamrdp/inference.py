import numpy as np
import torch

from .engine import get_device, load_checkpoint
from .model import MAMRDPNet


class Predictor:
    def __init__(self, checkpoint_path, device="auto"):
        saved = load_checkpoint(checkpoint_path)
        self.config = saved["config"]
        self.device = get_device(device)
        self.channels = saved["channels"]
        self.samples = round(self.config["data"]["window_s"] * self.config["data"]["target_sample_rate"])
        self.mean = saved["normalizer"]["mean"].numpy()[None, :, None]
        self.std = saved["normalizer"]["std"].numpy()[None, :, None]
        self.model = MAMRDPNet(self.channels, self.config["data"]["target_sample_rate"], self.config["model"]).to(self.device)
        self.model.load_state_dict(saved["model"])
        self.model.eval()

    @torch.no_grad()
    def predict_proba(self, eeg_windows, batch_size=16):
        x = np.asarray(eeg_windows, dtype=np.float32)
        if x.ndim == 2:
            x = x[None]
        if x.ndim != 3 or x.shape[1:] != (self.channels, self.samples) or not len(x):
            raise ValueError(f"Expected N x {self.channels} x {self.samples}, got {x.shape}")
        if not np.isfinite(x).all() or batch_size < 1:
            raise ValueError("Nonfinite EEG or invalid batch_size")
        output = []
        for start in range(0, len(x), batch_size):
            batch = (x[start:start + batch_size] - self.mean) / self.std
            logits, _ = self.model(torch.from_numpy(batch).to(self.device))
            output.append(logits.softmax(-1).cpu().numpy())
        return np.concatenate(output)
