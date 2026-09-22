import copy
import json
from pathlib import Path


def load_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    for key, value in cfg["paths"].items():
        p = Path(value).expanduser()
        cfg["paths"][key] = str(p if p.is_absolute() else path.parent / p)
    validate(cfg)
    return cfg


def validate(cfg):
    d, m, t, e = (cfg[k] for k in ("data", "model", "training", "evaluation"))
    if m.get("implementation_revision") not in (2, 3):
        raise ValueError()
    if m.get("implementation_revision") == 3:
        if m.get("filter_initialization") not in ("minimum_phase", "linear_phase"):
            raise ValueError("Invalid filter_initialization")
        if not isinstance(m.get("contrastive_normalize"), bool):
            raise ValueError("contrastive_normalize must be an explicit boolean")
        if m.get("gate_initialization") not in ("balanced", "pytorch"):
            raise ValueError("Invalid gate_initialization")
    if min(d["sample_rate"], d["target_sample_rate"], d["window_s"], d["stride_s"]) <= 0:
        raise ValueError("Sampling rates and window/stride durations must be positive")
    for key in ("window_s", "stride_s"):
        n = d[key] * d["target_sample_rate"]
        if abs(n - round(n)) > 1e-6:
            raise ValueError(f"{key} must correspond to an integer number of samples")
    if m["feature_dim"] % m["attention_heads"]:
        raise ValueError("feature_dim must be divisible by attention_heads")
    if len(m["bands"]) != len(m["filter_lengths"]) or not m["bands"]:
        raise ValueError("Specify one filter length per band")
    if any(k < 3 or k % 2 != 1 for k in m["filter_lengths"] + m["temporal_kernels"]):
        raise ValueError("Filter and temporal kernel lengths must be odd and >= 3")
    if m["frequency_nfft"] < max(m["filter_lengths"]):
        raise ValueError("frequency_nfft must cover every filter")
    if len(m["resolution_kernels"]) != len(m["resolution_strides"]):
        raise ValueError("Resolution kernels and strides must have equal lengths")
    if any(not 0 < low < high < d["target_sample_rate"] / 2 for low, high in m["bands"]):
        raise ValueError("Bands must lie strictly below target Nyquist frequency")
    if round(d["window_s"] * d["target_sample_rate"]) < m["pool_size"]:
        raise ValueError("Window too short for pooling")
    if min(t["epochs"], t["batch_size"], t["queue_size"], t["temperature"]) <= 0:
        raise ValueError("Epochs, batch size, queue size, temperature must be positive")
    if t["num_workers"] < 0:
        raise ValueError("num_workers must be non-negative")
    if not 0 < t["validation_fraction_loso"] < 1:
        raise ValueError("validation_fraction_loso must be in (0,1)")
    if e.get("recursive_time_constant_s", e.get("recursive_n", 0)) <= 0 or e["match_tolerance_s"] < 0:
        raise ValueError("Invalid postprocessing parameters")
    if e["event_time"] not in ("end", "center"):
        raise ValueError("event_time must be end or center")

def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def clone(cfg):
    return copy.deepcopy(cfg)
