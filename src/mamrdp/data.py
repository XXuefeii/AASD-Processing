import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import butter, resample_poly, sosfiltfilt
import soundfile as sf
import torch
from torch.utils.data import Dataset

from .config import save_json

CACHE_VERSION = 1


def scalar(value):
    while isinstance(value, np.ndarray):
        if value.size != 1:
            if value.dtype.kind in "US":
                return "".join(value.reshape(-1).tolist())
            raise ValueError(f"Expected scalar, got {value.shape}")
        value = value.reshape(-1)[0]
    return value.item() if isinstance(value, np.generic) else value


def integer(value):
    number = float(scalar(value))
    if not np.isfinite(number) or number != int(number):
        raise ValueError(f"Expected integer event code/epoch, got {value!r}")
    return int(number)


def field(value, name):
    while isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(-1)[0]
    if isinstance(value, dict):
        return value[name]
    if isinstance(value, np.void) and value.dtype.names:
        return value[name]
    return getattr(value, name)


def event_rows(events):
    a = np.asarray(events)
    if a.dtype.names:
        return [(integer(row["type"]), float(scalar(row["latency"])),
                 integer(row["epoch"])) for row in a.reshape(-1)]
    if a.ndim == 2 and a.shape[1] == 5:
        return [(integer(r[0]), float(scalar(r[1])), integer(r[4])) for r in a]
    rows = []
    for r in a.reshape(-1):
        rows.append((integer(field(r, "type")), float(scalar(field(r, "latency"))),
                     integer(field(r, "epoch"))))
    return rows


def _load_hdf5(path, key):
    import h5py
    with h5py.File(path, "r") as f:
        def read(obj):
            if isinstance(obj, h5py.Group):
                return {k: read(v) for k, v in obj.items()}
            data = obj[()]
            if h5py.check_dtype(ref=obj.dtype) is not None:
                out = np.empty(data.shape, dtype=object)
                for index in np.ndindex(data.shape):
                    out[index] = read(f[data[index]]) if data[index] else None
                return out.T
            if obj.attrs.get("MATLAB_class") == b"char":
                return "".join(chr(int(c)) for c in data.T.reshape(-1))
            return data.T
        eeg = read(f[key])
        if isinstance(eeg, np.ndarray):
            eeg = scalar(eeg)
        ev = eeg["event"]
        if isinstance(ev, dict):
            eeg["event"] = np.array([
                [v, l, 0, 0, ep] for v, l, ep in zip(
                    np.asarray(ev["type"]).reshape(-1),
                    np.asarray(ev["latency"]).reshape(-1),
                    np.asarray(ev["epoch"]).reshape(-1))], dtype=object)
        return eeg


def load_eeg(path, key="EEG_new"):
    import h5py
    if h5py.is_hdf5(path):
        eeg = _load_hdf5(path, key)
    else:
        raw = scipy.io.loadmat(path, variable_names=[key])
        if key not in raw:
            raise KeyError(f"{path}: missing {key}; change data.mat_key if needed")
        eeg = raw[key]
    x = np.asarray(field(eeg, "data"), dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"{path}: expected channels x samples x trials, got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"{path}: EEG contains NaN or infinity")
    return x, event_rows(field(eeg, "event"))


def trial_events(rows, n_samples, n_trials, fs):
    result = []
    if any(ep < 1 or ep > n_trials for _, _, ep in rows):
        raise ValueError("Event epoch is outside the EEG trial range")
    for ep in range(1, n_trials + 1):
        selected = sorted((lat - 1 - (ep - 1) * n_samples, code)
                          for code, lat, epoch in rows if epoch == ep)
        if any(not np.isfinite(lat) or lat < -1e-5 or lat >= n_samples for lat, _ in selected):
            raise ValueError(f"Epoch {ep}: latency must be 1-based and globally concatenated")
        starts = [(lat, code) for lat, code in selected if 11 <= code <= 70]
        if len(starts) != 1 or abs(starts[0][0]) > 1e-3:
            raise ValueError(f"Epoch {ep}: expected exactly one trial-start trigger 11..70 at its start")
        directions = [(lat / fs, int(code == 184)) for lat, code in selected if code in (179, 184)]
        if not directions:
            raise ValueError(f"Epoch {ep}: no 179/184 event; cannot invent its attention label")
        for a, b in zip(directions, directions[1:]):
            if a[0] == b[0] and a[1] != b[1]:
                raise ValueError(f"Epoch {ep}: conflicting directions at one timestamp")
        times, labels = [0.0], [directions[0][1]]
        for time, label in directions:
            if label != labels[-1]:
                times.append(time)
                labels.append(label)
        result.append({"epoch": ep, "trigger": starts[0][1],
                       "state_times": times, "state_labels": labels,
                       "duration_s": n_samples / fs})
    return result


def labels_at(trial, times):
    indexes = np.searchsorted(trial["state_times"], times, side="right") - 1
    if np.any(indexes < 0):
        raise ValueError("Label requested before trial start")
    return np.asarray(trial["state_labels"], dtype=np.int64)[indexes]


def file_signature(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def stimulus_map(cfg):
    frame = pd.read_excel(cfg["paths"]["stimlist"])
    if not {"Trigger", "MixFile"}.issubset(frame.columns):
        raise ValueError("StimList.xlsx must contain Trigger and MixFile columns")
    result = {}
    root = Path(cfg["paths"]["audio_root"])
    audio_index = None
    for _, row in frame.iterrows():
        if pd.isna(row["Trigger"]):
            continue
        trigger = integer(row["Trigger"])
        if not 11 <= trigger <= 70:
            continue
        if trigger in result:
            raise ValueError(f"Duplicate stimulus trigger {trigger}")
        if pd.isna(row["MixFile"]):
            raise ValueError(f"Trigger {trigger}: missing MixFile")
        mix = str(row["MixFile"]).strip()
        path = root / mix
        item = {"mix_file": mix, "audio_path": str(path)}
        if cfg["data"]["verify_audio"]:
            if not path.is_file():
                if audio_index is None:
                    audio_index = {}
                    for p in root.rglob("*"):
                        if p.is_file() and p.suffix.lower() == ".wav":
                            audio_index.setdefault(p.name.lower(), []).append(p)
                choices = audio_index.get(Path(mix).name.lower(), [])
                if len(choices) != 1:
                    raise FileNotFoundError(f"Trigger {trigger}: cannot uniquely resolve {path}")
                path = choices[0]
            info = sf.info(str(path))
            item.update(audio_path=str(path), audio_duration_s=info.duration,
                        audio_sample_rate=info.samplerate, audio_channels=info.channels,
                        audio_signature=file_signature(path))
        result[trigger] = item
    return result


def prepare_subject(cfg, subject, stimuli, force=False):
    source = Path(cfg["paths"]["dataset_root"]) / f"S{subject}.mat"
    dc = cfg["data"]
    settings = {k: v for k, v in dc.items() if k not in ("window_s", "stride_s")}
    signature = {"version": CACHE_VERSION, "source": file_signature(source),
                 "stimlist": file_signature(cfg["paths"]["stimlist"]),
                 "settings": settings, "stimuli": stimuli}
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
    folder = Path(cfg["paths"]["cache_root"]) / f"S{subject}_{digest}"
    manifest = folder / "manifest.json"
    if manifest.exists() and (folder / "eeg.npy").exists() and not force:
        return load_subject(folder)
    x, rows = load_eeg(source, dc["mat_key"])
    trials = trial_events(rows, x.shape[1], x.shape[2], dc["sample_rate"])
    if dc["reference_channels"] is not None:
        refs = dc["reference_channels"]
        if not refs or min(refs) < 0 or max(refs) >= x.shape[0]:
            raise ValueError("reference_channels must be valid zero-based original channel indexes")
        x = x - x[refs].mean(axis=0, keepdims=True)
    channels = dc["channels"] if dc["channels"] is not None else list(range(x.shape[0]))
    if not channels or len(set(channels)) != len(channels) or min(channels) < 0 or max(channels) >= x.shape[0]:
        raise ValueError(f"Invalid channel selection for {x.shape[0]} recorded channels")
    x = np.ascontiguousarray(x[channels].transpose(2, 0, 1))
    if dc["expected_channels"] is not None and x.shape[1] != dc["expected_channels"]:
        raise ValueError(f"Expected {dc['expected_channels']} channels, got {x.shape[1]}")
    if dc["bandpass"] is not None:
        sos = butter(dc["filter_order"], dc["bandpass"], btype="bandpass",
                     fs=dc["sample_rate"], output="sos")
        x = sosfiltfilt(sos, x, axis=-1).astype(np.float32)
    if dc["target_sample_rate"] != dc["sample_rate"]:
        ratio = Fraction(dc["target_sample_rate"] / dc["sample_rate"]).limit_denominator(10000)
        x = resample_poly(x, ratio.numerator, ratio.denominator, axis=-1).astype(np.float32)
    for index, trial in enumerate(trials):
        trigger = trial["trigger"]
        if trigger not in stimuli:
            raise ValueError(f"S{subject}, trigger {trigger}: missing from StimList.xlsx")
        trial.update(stimuli[trigger])
        trial.update(subject=subject, trial_index=index)
        if dc["verify_audio"] and abs(trial["duration_s"] - trial["audio_duration_s"]) > dc["audio_duration_tolerance_s"]:
            raise ValueError(f"S{subject}, trigger {trigger}: EEG/audio durations disagree")
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / "eeg.tmp.npy"
    np.save(temporary, x)
    temporary.replace(folder / "eeg.npy")
    save_json(manifest, {"subject": subject, "shape": list(x.shape),
                        "fs": dc["target_sample_rate"], "channels": channels,
                        "signature": signature, "trials": trials})
    print(f"Prepared S{subject}: {tuple(x.shape)}, {len(trials)} trials", flush=True)
    return load_subject(folder)


def load_subject(folder):
    folder = Path(folder)
    result = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    result["array_path"] = str((folder / "eeg.npy").resolve())
    occurrences = {}
    for trial in result["trials"]:
        occurrences.setdefault(str(trial["trigger"]), []).append(trial["epoch"])
    result["audit"] = {
        "unique_trigger_count": len(occurrences),
        "repeated_triggers": {k: epochs for k, epochs in occurrences.items() if len(epochs) > 1},
        "missing_triggers_11_70": sorted(set(range(11, 71)) - {int(k) for k in occurrences}),
        "mapping_policy": "Preserve original trigger-to-MixFile mapping; trial identity is subject+epoch."
    }
    return result


def prepare(cfg, subjects, force=False):
    stimuli = stimulus_map(cfg)
    result = {s: prepare_subject(cfg, s, stimuli, force) for s in subjects}
    layouts = {tuple(v["channels"]) for v in result.values()}
    if len(layouts) != 1:
        raise ValueError("All subjects must have the same channel count and order")
    return result


def split_trials(subjects, protocol, heldout, seed, validation_fraction=0.1):
    train, val, test = [], [], []
    for sid in sorted(subjects):
        if protocol == "cross-trial" and sid != heldout:
            continue
        n = len(subjects[sid]["trials"])
        rng = np.random.default_rng(seed + sid)
        indices = rng.permutation(n).tolist()
        if protocol == "loso" and sid == heldout:
            test.extend((sid, i) for i in range(n))
            continue
        if n < 3:
            raise ValueError("At least three trials per training subject are required")
        if protocol == "cross-trial":
            n_test = max(1, round(n * 0.1))
            n_val = max(1, round(n * 0.1))
            test.extend((sid, i) for i in indices[:n_test])
            val.extend((sid, i) for i in indices[n_test:n_test + n_val])
            train.extend((sid, i) for i in indices[n_test + n_val:])
        elif protocol == "loso":
            n_val = min(n - 1, max(1, round(n * validation_fraction)))
            val.extend((sid, i) for i in indices[:n_val])
            train.extend((sid, i) for i in indices[n_val:])
        else:
            raise ValueError(f"Unknown protocol {protocol}")
    if not train or not val or not test:
        raise ValueError("Empty split; LOSO requires at least two subjects and a valid held-out subject")
    assert not set(train) & set(val) and not set(train) & set(test) and not set(val) & set(test)
    return {"train": train, "val": val, "test": test}


def fit_normalizer(subjects, keys):
    """Stable channelwise population moments from TRAINING TRIALS ONLY."""
    arrays = {}
    count, mean, m2 = 0, None, None
    for sid, index in keys:
        if sid not in arrays:
            arrays[sid] = np.load(subjects[sid]["array_path"], mmap_mode="r")
        x = np.asarray(arrays[sid][index], dtype=np.float64)
        n = x.shape[-1]
        batch_mean = x.mean(axis=-1)
        batch_m2 = np.square(x - batch_mean[:, None]).sum(axis=-1)
        if mean is None:
            mean, m2, count = batch_mean, batch_m2, n
        else:
            delta = batch_mean - mean
            m2 += batch_m2 + delta ** 2 * (count * n / (count + n))
            mean += delta * (n / (count + n))
            count += n
    if count == 0:
        raise ValueError("Cannot normalize an empty training set")
    std = np.sqrt(m2 / count)
    return {"mean": mean.astype(np.float32), "std": np.maximum(std, 1e-6).astype(np.float32)}


class EEGWindows(Dataset):
    def __init__(self, subjects, keys, normalizer, window_s, stride_s):
        self.subjects = subjects
        self.mean = np.asarray(normalizer["mean"], dtype=np.float32)[:, None]
        self.std = np.asarray(normalizer["std"], dtype=np.float32)[:, None]
        self.arrays = {}
        self.records = []
        for sid, index in sorted(keys):
            meta = subjects[sid]
            trial = meta["trials"][index]
            fs = meta["fs"]
            width, step = round(window_s * fs), round(stride_s * fs)
            for start in range(0, meta["shape"][-1] - width + 1, step):
                center = (start + width / 2) / fs
                self.records.append({"subject": sid, "trial_index": index,
                                     "epoch": trial["epoch"], "trigger": trial["trigger"],
                                     "mix_file": trial["mix_file"], "start_sample": start,
                                     "stop_sample": start + width, "center_s": center,
                                     "end_s": (start + width) / fs,
                                     "label": int(labels_at(trial, center))})
        if not self.records:
            raise ValueError("No complete windows in this split")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        rec = self.records[index]
        sid = rec["subject"]
        if sid not in self.arrays:
            self.arrays[sid] = np.load(self.subjects[sid]["array_path"], mmap_mode="r")
        x = self.arrays[sid][rec["trial_index"], :, rec["start_sample"]:rec["stop_sample"]]
        x = np.ascontiguousarray((x - self.mean) / self.std, dtype=np.float32)
        return torch.from_numpy(x), rec["label"], index

    def __getstate__(self):
        state = self.__dict__.copy()
        state["arrays"] = {} 
        return state
