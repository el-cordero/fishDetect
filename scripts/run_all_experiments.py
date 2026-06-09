#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fishdetect.config import load_config
from fishdetect.experiment import command_string, init_experiment_dir, write_metrics_artifacts
from fishdetect.models.segmentation import train_segmentation_model
from fishdetect.models.torchvision_detectors import train_torchvision_detector
from fishdetect.models.yolo import train_yolo
from fishdetect.pipeline import prepare_dataset_pipeline
from fishdetect.utils.files import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all configured FishDetect experiments or dependency-aware smoke checks.")
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-downloads", action="store_true", help="Allow remote weights in smoke mode.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    run_config = copy.deepcopy(config)
    if args.smoke_test:
        smoke_root = Path(run_config["dataset"]["prepared_root"]).with_name("prepared_smoke")
        summary = prepare_dataset_pipeline(run_config, max_images=args.max_images or 50, prepared_root_override=smoke_root)
        run_config["dataset"]["prepared_root"] = summary["prepared_root"]
    results = []
    for exp in run_config.get("experiments", []):
        out = init_experiment_dir(args.output_root, exp, run_config, command_string())
        family = exp.get("family")
        if family == "yolo":
            metrics = train_yolo(
                run_config,
                exp,
                out,
                epochs=args.epochs or (1 if args.smoke_test else None),
                device=args.device,
                smoke_test=args.smoke_test,
                allow_downloads=args.allow_downloads,
            )
        elif family == "torchvision_detection":
            metrics = train_torchvision_detector(
                run_config,
                exp,
                out,
                epochs=args.epochs or (1 if args.smoke_test else None),
                device=args.device,
                smoke_test=args.smoke_test,
            )
        elif family == "segmentation":
            metrics = train_segmentation_model(run_config, exp, out, smoke_test=args.smoke_test)
        else:
            metrics = {"status": "skipped", "reason": f"Unknown experiment family: {family}", "experiment": exp["name"]}
        write_metrics_artifacts(out, exp, metrics, metrics.get("reason", "Experiment artifact."))
        results.append({"experiment": exp["name"], "status": metrics.get("status", "completed"), "reason": metrics.get("reason", "")})
        print(f"{exp['name']}: {metrics.get('status', 'completed')}")
    write_json(Path(args.output_root) / "run_all_summary.json", {"smoke_test": args.smoke_test, "results": results})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
