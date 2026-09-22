import argparse
import json
from pathlib import Path

from mamrdp.config import clone, load_config, save_json, validate
from mamrdp.data import prepare
from mamrdp.engine import get_device, load_checkpoint, summarize, test_checkpoint, train_fold


def subject_list(value):
    result = set()
    for piece in value.split(","):
        if "-" in piece:
            start, end = map(int, piece.split("-"))
            if start > end:
                raise argparse.ArgumentTypeError("Subject range must be ascending")
            result.update(range(start, end + 1))
        else:
            result.add(int(piece))
    if not result or min(result) < 1:
        raise argparse.ArgumentTypeError("Subjects must be positive integers, e.g. 1-18 or 1,3,5")
    return sorted(result)


def main():
    parser = argparse.ArgumentParser(description="MAMR-DPNet EEG training and evaluation")
    parser.add_argument("command", choices=["prepare", "train", "test", "predict"])
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--subjects", type=subject_list, help="Default: 1-18 for training; checkpoint test subjects for test")
    parser.add_argument("--protocol", choices=["cross-trial", "loso"], default="cross-trial")
    parser.add_argument("--folds", type=subject_list, help="Held-out subject IDs to run; default: all selected subjects")
    parser.add_argument("--run-name", default="mamrdp_1s")
    parser.add_argument("--checkpoint", help="best.pt/last.pt for test or predict")
    parser.add_argument("--resume", help="last.pt; requires exactly one matching fold and original run-name")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--window", type=float, help="Decision window seconds")
    parser.add_argument("--stride", type=float)
    parser.add_argument("--device", help="auto, cpu, cuda or cuda:0")
    parser.add_argument("--num-workers", type=int, help="DataLoader workers; default 0 is portable on Windows")
    parser.add_argument("--dataset-root")
    parser.add_argument("--stimlist")
    parser.add_argument("--audio-root")
    parser.add_argument("--output-root")
    parser.add_argument("--output", help="Independent evaluation output directory")
    parser.add_argument("--skip-audio-check", action="store_true", help="Audio is not a model input")
    parser.add_argument("--force", action="store_true", help="Rebuild prepared cache")
    args = parser.parse_args()

    if args.command in ("test", "predict"):
        if not args.checkpoint:
            parser.error("--checkpoint is required")
        checkpoint = load_checkpoint(args.checkpoint)
        cfg = clone(checkpoint["config"])
        if args.window is not None or args.stride is not None:
            parser.error("Test/predict uses the saved training window/stride")
        if args.command == "test" and args.subjects is not None:
            parser.error("test uses the saved held-out trials; use predict --subjects for all trials of other subjects")
        if args.command == "predict" and not args.subjects:
            parser.error("predict requires --subjects")
        subjects = args.subjects or sorted({s for s, _ in checkpoint["split"]["test"]})
    else:
        cfg = load_config(args.config)
        subjects = args.subjects or list(range(1, 19))
    for key in ("dataset_root", "stimlist", "audio_root", "output_root"):
        value = getattr(args, key)
        if value:
            cfg["paths"][key] = str(Path(value).resolve())
    for key in ("epochs", "batch_size", "device", "num_workers"):
        value = getattr(args, key)
        if value is not None:
            cfg["training"][key] = value
    if args.window is not None:
        cfg["data"]["window_s"] = args.window
        cfg["data"]["stride_s"] = args.window
    if args.stride is not None:
        cfg["data"]["stride_s"] = args.stride
    if args.skip_audio_check:
        cfg["data"]["verify_audio"] = False
    validate(cfg)
    if args.command != "prepare":
        get_device(cfg["training"]["device"]) 
    data = prepare(cfg, subjects, force=args.force)
    if args.command == "prepare":
        report = [{"subject": s, "shape": v["shape"], "sample_rate": v["fs"],
                   "trial_count": len(v["trials"]),
                   "switch_count": sum(len(t["state_times"]) - 1 for t in v["trials"]),
                   "cache": v["array_path"], "audit": v["audit"]} for s, v in data.items()]
        save_json(Path(cfg["paths"]["cache_root"]) / "prepare_report.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "train":
        folds = args.folds or subjects
        if not set(folds).issubset(subjects):
            parser.error("--folds must be a subset of --subjects")
        if args.resume and len(folds) != 1:
            parser.error("--resume requires a single --folds value")
        results = []
        destination = Path(cfg["paths"]["output_root"]) / args.run_name / args.protocol / "summary.json"
        for fold in folds:
            active = {fold: data[fold]} if args.protocol == "cross-trial" else data
            result = train_fold(cfg, active, args.protocol, fold, args.run_name, args.resume)
            results.append(result)
            all_results = {}
            for path in destination.parent.glob("S*/result.json"):
                r = json.loads(path.read_text(encoding="utf-8"))
                all_results[r["subject"]] = r
            all_results.update({r["subject"]: r for r in results})
            summarize([all_results[s] for s in sorted(all_results)], destination)
        print(f"Summary: {destination}")
    else:
        destination = args.output or str(Path(args.checkpoint).resolve().parent / ("retest" if args.command == "test" else "predict"))
        result = test_checkpoint(args.checkpoint, cfg, data, destination, args.command == "predict")
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
