#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fishdetect.utils.files import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect or recompute experiment evaluation metrics.")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--experiment", default=None, help="Experiment directory or experiment name.")
    parser.add_argument("--all", action="store_true", help="Evaluate/collect all experiment metrics.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    if args.all:
        experiment_dirs = sorted((output_root / "experiments").glob("*"))
    elif args.experiment:
        candidate = Path(args.experiment)
        experiment_dirs = [candidate if candidate.exists() else output_root / "experiments" / args.experiment]
    else:
        raise ValueError("Use --all or --experiment.")

    summaries = []
    for exp_dir in experiment_dirs:
        metrics_path = exp_dir / "metrics.json"
        if not metrics_path.exists():
            summaries.append({"experiment": exp_dir.name, "status": "missing_metrics"})
            continue
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        metrics["evaluated_from"] = str(metrics_path)
        write_json(exp_dir / "evaluation_summary.json", metrics)
        summaries.append(
            {
                "experiment": exp_dir.name,
                "status": metrics.get("status", "completed"),
                "mAP50": metrics.get("mAP50"),
                "mAP50_95": metrics.get("mAP50_95"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
            }
        )
    ensure_dir(output_root)
    write_json(output_root / "evaluation_summary.json", {"experiments": summaries})
    print(f"Wrote evaluation summary for {len(summaries)} experiment(s).")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
