#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fishdetect.config import find_experiment, load_config
from fishdetect.experiment import command_string, init_experiment_dir, write_metrics_artifacts
from fishdetect.models.segmentation import train_segmentation_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segmentation entry point. The current FishDetect merge is box-only.")
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--experiment", default="segmentation_disabled")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--allow-weak-box-masks", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    try:
        exp = find_experiment(config, args.experiment)
    except Exception:
        if args.experiment != "segmentation_disabled":
            raise
        exp = {
            "name": "segmentation_disabled",
            "family": "segmentation",
            "model": "none",
            "task": "none",
        }
    out = init_experiment_dir(args.output_root, exp, config, command_string())
    metrics = train_segmentation_model(
        config,
        exp,
        out,
        smoke_test=args.smoke_test,
        allow_weak_box_masks=args.allow_weak_box_masks,
    )
    write_metrics_artifacts(out, exp, metrics, metrics.get("reason", "Segmentation artifact."))
    print(f"Wrote segmentation status to {out} with status={metrics.get('status', 'completed')}.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
