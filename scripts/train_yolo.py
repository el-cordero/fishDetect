#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fishdetect.config import experiments_for_group, find_experiment, load_config
from fishdetect.experiment import command_string, init_experiment_dir, skipped_metrics, write_metrics_artifacts
from fishdetect.models.yolo import train_yolo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or smoke-check an Ultralytics YOLO/RT-DETR detector.")
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--group", default=None, help="Run a configured experiment group instead of one experiment.")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda:0, or Ultralytics-compatible device.")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--allow-downloads", action="store_true", help="Allow remote weight downloads in smoke mode.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.experiment and not args.group:
        raise ValueError("Use --experiment <name> or --group <group_name>.")
    config = load_config(args.config)
    experiments = experiments_for_group(config, args.group) if args.group else [find_experiment(config, args.experiment)]
    for exp in experiments:
        if exp.get("family") != "yolo":
            print(f"Skipping {exp.get('name')}: family={exp.get('family')} is not yolo.")
            continue
        run_one(args, config, exp)
    return 0


def run_one(args: argparse.Namespace, config: dict, exp: dict) -> None:
    out = init_experiment_dir(args.output_root, exp, config, command_string())
    try:
        metrics = train_yolo(
            config,
            exp,
            out,
            epochs=args.epochs,
            device=args.device,
            smoke_test=args.smoke_test,
            allow_downloads=args.allow_downloads,
        )
    except Exception as exc:
        if args.smoke_test:
            metrics = skipped_metrics(f"Smoke training failed cleanly: {exc}", exp, smoke_test=True)
        else:
            raise
    write_metrics_artifacts(out, exp, metrics, metrics.get("reason", "YOLO training/evaluation artifact."))
    print(f"Wrote experiment artifacts to {out} with status={metrics.get('status', 'completed')}.")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
