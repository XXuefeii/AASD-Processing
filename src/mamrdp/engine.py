import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .config import save_json
from .data import EEGWindows, fit_normalizer, split_trials
from .metrics import (aggregate_events, bounded_recursive, classification_metrics,
                      match_events, state_transitions, resolve_recursive_settings)
from .model import MAMRDPNet, MemoryQueue, require_current_revision


def seed_all(seed, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices: cpu, auto, cuda or cuda:<index>")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable. Check the NVIDIA driver and CUDA-enabled "
                "PyTorch installation with python check_gpu.py. For CPU use --device cpu explicitly.")
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA index {index} unavailable; visible GPU count={torch.cuda.device_count()}")
        device = torch.device("cuda", index)
        torch.cuda.set_device(device)
    return device


def device_info(device):
    result = {"device": str(device), "torch_version": str(torch.__version__),
              "torch_cuda_version": torch.version.cuda, "precision": "float32"}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result.update(gpu_name=properties.name, gpu_total_memory_gib=properties.total_memory / 2**30,
                      cuda_visible_device_count=torch.cuda.device_count())
    return result


def loader(dataset, cfg, device, shuffle=False, generator=None):
    return DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=shuffle,
                      num_workers=cfg["training"]["num_workers"],
                      pin_memory=device.type == "cuda", generator=generator,
                      persistent_workers=cfg["training"]["num_workers"] > 0,
                      drop_last=False)


def atomic_checkpoint(path, payload):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    require_current_revision(checkpoint["config"]["model"])
    return checkpoint


@torch.no_grad()
def infer(model, dataset, cfg, device):
    model.eval()
    probabilities = np.zeros((len(dataset), 2), dtype=np.float32)
    truth = np.zeros(len(dataset), dtype=np.int64)
    ce_sum = 0.0
    for x, y, indices in loader(dataset, cfg, device):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits, _ = model(x)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite prediction")
        ce_sum += F.cross_entropy(logits, y, reduction="sum").item()
        probabilities[indices.numpy()] = logits.softmax(dim=-1).cpu().numpy()
        truth[indices.numpy()] = y.cpu().numpy()
    metrics = classification_metrics(truth, probabilities.argmax(axis=1))
    metrics["cross_entropy"] = ce_sum / len(dataset)
    return metrics, probabilities


def export_evaluation(folder, dataset, probabilities, window_metrics, cfg):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    ec = resolve_recursive_settings(cfg["evaluation"], cfg["data"]["stride_s"])
    records = [dict(r) for r in dataset.records]
    groups = {}
    for index, rec in enumerate(records):
        groups.setdefault((rec["subject"], rec["trial_index"]), []).append(index)
    events = []
    for (sid, trial_index), indices in groups.items():
        trial = dataset.subjects[sid]["trials"][trial_index]
        p = probabilities[indices]
        states, gains = bounded_recursive(p, ec["recursive_n"], ec["recursive_bias"], ec["initial_gain"])
        times = [records[i][ec["event_time"] + "_s"] for i in indices]
        truth = list(zip(trial["state_times"][1:], trial["state_labels"][1:]))
        raw_switches = state_transitions(p.argmax(axis=1), times)
        filtered_switches = state_transitions(states, times)
        events.append({"subject": sid, "epoch": trial["epoch"], "trigger": trial["trigger"],
                       "mix_file": trial["mix_file"], "true_switches": truth,
                       "raw_switches": raw_switches, "recursive_switches": filtered_switches,
                       "raw": match_events(truth, raw_switches, ec["match_tolerance_s"]),
                       "recursive": match_events(truth, filtered_switches, ec["match_tolerance_s"])})
        for local, index in enumerate(indices):
            records[index].update(p_left=float(p[local, 0]), p_right=float(p[local, 1]),
                                  predicted_label=int(p[local].argmax()),
                                  recursive_label=int(states[local]),
                                  gain_left=float(gains[local, 0]), gain_right=float(gains[local, 1]))
    with (folder / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    output = {"window": window_metrics,
              "raw_events": aggregate_events([v["raw"] for v in events]),
              "recursive_events": aggregate_events([v["recursive"] for v in events]),
              "evaluation_settings": ec,
              "label_map": {"0": "left (179)", "1": "right (184)"}}
    save_json(folder / "events.json", events)
    save_json(folder / "metrics.json", output)
    return output


def dataset_from_keys(subjects, keys, normalizer, cfg):
    return EEGWindows(subjects, keys, normalizer, cfg["data"]["window_s"], cfg["data"]["stride_s"])


def train_fold(cfg, subjects, protocol, heldout, run_name, resume=None):
    tc = cfg["training"]
    seed_all(tc["seed"] + heldout, tc["cpu_threads"])
    device = get_device(tc["device"])
    folder = Path(cfg["paths"]["output_root"]) / run_name / protocol / f"S{heldout:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "last.pt").exists() and resume is None:
        raise FileExistsError(f"Run exists: {folder}. Use --resume or a different --run-name.")
    runtime = device_info(device)
    save_json(folder / "device.json", runtime)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        print(f"GPU: {runtime['gpu_name']} ({device}), "
              f"{runtime['gpu_total_memory_gib']:.1f} GiB, float32", flush=True)
    split = split_trials(subjects, protocol, heldout, tc["seed"], tc["validation_fraction_loso"])
    normalizer = fit_normalizer(subjects, split["train"])
    channels = len(normalizer["mean"])
    datasets = {k: dataset_from_keys(subjects, v, normalizer, cfg) for k, v in split.items()}
    model = MAMRDPNet(channels, cfg["data"]["target_sample_rate"], cfg["model"]).to(device)
    queue = MemoryQueue(tc["queue_size"], cfg["model"]["feature_dim"], tc["temperature"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tc["learning_rate"], weight_decay=tc["weight_decay"])
    generator = torch.Generator().manual_seed(tc["seed"] + heldout)
    history, start_epoch, best_ce = [], 0, float("inf")
    fingerprints = {str(s): v["signature"] for s, v in subjects.items()}
    if resume:
        old = load_checkpoint(resume)
        if old["protocol"] != protocol or old["heldout"] != heldout:
            raise ValueError("Resume checkpoint protocol/heldout does not match this fold")
        if old["split"] != split or old["fingerprints"] != fingerprints:
            raise ValueError("Data/split changed since checkpoint; start a new run")
        for part in ("data", "model", "evaluation"):
            if old["config"][part] != cfg[part]:
                raise ValueError(f"Resume configuration changed: {part}")
        ignored = {"epochs", "device", "num_workers", "cpu_threads"}
        if any(old["config"]["training"][k] != v for k, v in tc.items() if k not in ignored):
            raise ValueError("Training settings changed; only epochs/device/workers/threads may change on resume")
        if Path(resume).resolve().parent != folder.resolve():
            raise ValueError("Resume in the original run directory (same --run-name/output-root)")
        model.load_state_dict(old["model"])
        queue.load_state_dict(old["queue"])
        optimizer.load_state_dict(old["optimizer"])
        for state in optimizer.state.values():
            for key, val in state.items():
                if isinstance(val, torch.Tensor) and key != "step":
                    state[key] = val.to(device)
        history, start_epoch, best_ce = old["history"], old["epoch"], old["best_val_ce"]
        torch.set_rng_state(old["torch_rng"])
        if device.type == "cuda" and old["cuda_rng"]:
            torch.cuda.set_rng_state_all(old["cuda_rng"])
        generator.set_state(old["loader_rng"])
    save_json(folder / "config.json", cfg)
    split_report = {name: [{"subject": s, "trial_index": i,
                           "trigger": subjects[s]["trials"][i]["trigger"]} for s, i in keys]
                    for name, keys in split.items()}
    save_json(folder / "split.json", split_report)
    audio_sets = {name: {subjects[s]["trials"][i]["mix_file"] for s, i in keys}
                  for name, keys in split.items()}
    save_json(folder / "data_audit.json", {
        "subjects": {str(s): v["audit"] for s, v in subjects.items()},
        "shared_audio_between_splits": {
            a + "_" + b: sorted(audio_sets[a] & audio_sets[b])
            for a, b in (("train", "val"), ("train", "test"), ("val", "test"))},
        "split_unit": "EEG trial (subject, epoch); not audio identity"})
    np.savez(folder / "normalizer.npz", **normalizer)
    train_loader = loader(datasets["train"], cfg, device, shuffle=True, generator=generator)
    print(f"{protocol} S{heldout}: {channels} channels, device={device}, "
          f"windows=" + str({k: len(v) for k, v in datasets.items()}), flush=True)
    for epoch in range(start_epoch, tc["epochs"]):
        model.train()
        sums = np.zeros(4, dtype=float)
        correct, seen = 0, 0
        grad_norm_sum, clipped_batches, batches = 0.0, 0, 0
        started = time.perf_counter()
        for x, y, _ in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, z = model(x)
            ce = F.cross_entropy(logits, y)
            contrast = queue.loss(z, y)
            regularization = model.frequency_regularization()
            loss = ce + tc["contrastive_weight"] * contrast + tc["frequency_weight"] * regularization
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["gradient_clip"], error_if_nonfinite=True)
            grad_norm_sum += float(grad_norm.item())
            clipped_batches += int(grad_norm.item() > tc["gradient_clip"])
            batches += 1
            optimizer.step()
            queue.enqueue(z, y)
            n = len(y)
            sums += np.array([loss.item(), ce.item(), contrast.item(), regularization.item()]) * n
            correct += int((logits.argmax(dim=1) == y).sum().item())
            seen += n
        val_metrics, _ = infer(model, datasets["val"], cfg, device)
        entry = {"epoch": epoch + 1, "train_loss": float(sums[0] / seen),
                 "train_ce": float(sums[1] / seen), "train_contrastive": float(sums[2] / seen),
                 "train_frequency_regularization": float(sums[3] / seen),
                 "train_gradient_norm_mean": grad_norm_sum / batches,
                 "train_gradient_clip_fraction": clipped_batches / batches,
                 "train_accuracy": correct / seen, "val_ce": val_metrics["cross_entropy"],
                 "val_accuracy": val_metrics["accuracy"], "val_f1_macro": val_metrics["f1_macro"],
                 "seconds": time.perf_counter() - started}
        if device.type == "cuda":
            entry["cuda_peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
        history.append(entry)
        improved = entry["val_ce"] < best_ce
        if improved:
            best_ce = entry["val_ce"]
        payload = {"format_version": 2, "model": model.state_dict(), "queue": queue.state_dict(),
                   "optimizer": optimizer.state_dict(), "epoch": epoch + 1,
                   "best_val_ce": best_ce, "config": cfg, "protocol": protocol, "heldout": heldout,
                   "channels": channels, "split": split, "fingerprints": fingerprints,
                   "normalizer": {k: torch.from_numpy(v) for k, v in normalizer.items()},
                   "history": history, "torch_rng": torch.get_rng_state(),
                   "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                   "loader_rng": generator.get_state()}
        if improved:
            atomic_checkpoint(folder / "best.pt", payload)
        atomic_checkpoint(folder / "last.pt", payload)
        save_json(folder / "history.json", history)
        print(f"Epoch {epoch + 1:03d}/{tc['epochs']}: loss={entry['train_loss']:.4f}, "
              f"train_ACC={entry['train_accuracy']:.3f}, val_ACC={entry['val_accuracy']:.3f}, "
              f"val_CE={entry['val_ce']:.4f}, {entry['seconds']:.1f}s", flush=True)
    best = load_checkpoint(folder / "best.pt")
    model.load_state_dict(best["model"])
    test_metrics, probabilities = infer(model, datasets["test"], cfg, device)
    result = export_evaluation(folder / "test", datasets["test"], probabilities, test_metrics, cfg)
    result.update(subject=heldout, best_epoch=best["epoch"], checkpoint=str(folder / "best.pt"))
    save_json(folder / "result.json", result)
    print(f"TEST S{heldout}: ACC={test_metrics['accuracy']:.4f}, "
          f"F1(right)={test_metrics['f1_right']:.4f}, F1(macro)={test_metrics['f1_macro']:.4f}", flush=True)
    return result


def test_checkpoint(path, cfg, subjects, output, new_subjects=False):
    checkpoint = load_checkpoint(path)
    if checkpoint["config"]["data"] != cfg["data"] or checkpoint["config"]["model"] != cfg["model"]:
        raise ValueError("Test must use the checkpoint's preprocessing/window/model settings")
    seed_all(cfg["training"]["seed"], cfg["training"]["cpu_threads"])
    device = get_device(cfg["training"]["device"])
    save_json(Path(output) / "device.json", device_info(device))
    if new_subjects:
        keys = [(s, i) for s, sub in subjects.items() for i in range(len(sub["trials"]))]
    else:
        keys = checkpoint["split"]["test"]
        for sid in {s for s, _ in keys}:
            if subjects[sid]["signature"] != checkpoint["fingerprints"][str(sid)]:
                raise ValueError(f"S{sid}: source data changed since training")
    normalizer = {k: v.numpy() for k, v in checkpoint["normalizer"].items()}
    if any(v["shape"][1] != checkpoint["channels"] for v in subjects.values()):
        raise ValueError("Checkpoint and input data have different channel counts")
    dataset = dataset_from_keys(subjects, keys, normalizer, cfg)
    model = MAMRDPNet(checkpoint["channels"], cfg["data"]["target_sample_rate"], cfg["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    metrics, probabilities = infer(model, dataset, cfg, device)
    return export_evaluation(output, dataset, probabilities, metrics, cfg)


def summarize(results, destination):
    rows = [{"subject": r["subject"], "accuracy": r["window"]["accuracy"],
             "f1_right": r["window"]["f1_right"], "f1_macro": r["window"]["f1_macro"],
             "sdl_s": r["recursive_events"]["sdl_s"],
             "fp_per_trial": r["recursive_events"]["false_positive_per_trial"],
             "false_negative_rate": r["recursive_events"]["false_negative_rate"]} for r in results]
    stats = {}
    for name in rows[0]:
        if name == "subject":
            continue
        values = [r[name] for r in rows if r[name] is not None]
        stats[name] = {"mean": float(np.mean(values)) if values else None,
                       "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                       "n_subjects": len(values)}
    save_json(destination, {"per_subject": rows, "subject_mean_sd": stats})
