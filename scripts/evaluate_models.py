#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fishdetect.config import load_config
from fishdetect.evaluation.detection_metrics import evaluate_detections, object_size_bin
from fishdetect.models.yolo import export_yolo_predictions
from fishdetect.predictions import (
    load_ground_truth,
    load_predictions,
    match_predictions,
    save_error_tables,
)
from fishdetect.utils.files import ensure_dir, write_csv_dicts, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect or recompute experiment evaluation metrics.")
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--experiment", default=None, help="Experiment directory or experiment name.")
    parser.add_argument("--all", action="store_true", help="Evaluate/collect all experiment metrics.")
    parser.add_argument("--split", default="test", choices=["val", "test", "train"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conf", type=float, default=0.001)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
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
        evaluated = evaluate_experiment(exp_dir, metrics, config, args.split, args.device, args.conf)
        write_json(exp_dir / "evaluation_summary.json", evaluated)
        summaries.append(
            {
                "experiment": exp_dir.name,
                "status": evaluated.get("status", "completed"),
                "mAP50": evaluated.get("mAP50"),
                "mAP50_95": evaluated.get("mAP50_95"),
                "precision": evaluated.get("precision"),
                "recall": evaluated.get("recall"),
            }
        )
    ensure_dir(output_root)
    write_json(output_root / "evaluation_summary.json", {"experiments": summaries})
    print(f"Wrote evaluation summary for {len(summaries)} experiment(s).")
    return 0


def evaluate_experiment(
    exp_dir: Path,
    metrics: dict,
    config: dict,
    split: str,
    device: str,
    conf: float,
) -> dict:
    prepared_root = _prepared_root(exp_dir, config)
    gt = load_ground_truth(prepared_root, split=split)
    predictions_path = exp_dir / "predictions" / "predictions.json"
    if not predictions_path.exists() and metrics.get("best_weights_path"):
        try:
            export_yolo_predictions(
                model_path=metrics["best_weights_path"],
                prepared_root=prepared_root,
                split=split,
                output_json=predictions_path,
                device=device,
                conf=conf,
            )
        except Exception as exc:
            metrics["evaluation_status"] = "prediction_generation_failed"
            metrics["evaluation_warning"] = str(exc)
            write_json(exp_dir / "metrics.json", metrics)
            return metrics
    preds = load_predictions(predictions_path)
    if not preds:
        metrics["evaluation_status"] = "no_predictions"
        metrics.setdefault("reason", metrics.get("reason", "No prediction JSON or trained weights available."))
        return metrics

    class_names = sorted({row["class_name"] for row in gt} | {row["class_name"] for row in preds})
    eval_metrics = evaluate_detections(gt, preds, class_names=class_names)
    matches, false_positives, false_negatives = match_predictions(gt, preds)
    pred_dir = ensure_dir(exp_dir / "predictions")
    save_error_tables(pred_dir, matches, false_positives, false_negatives)
    per_class_rows = []
    for class_name, row in eval_metrics.get("per_class", {}).items():
        row = {"class_name": class_name, **row}
        per_class_rows.append(row)
    write_csv_dicts(
        exp_dir / "per_class_metrics.csv",
        per_class_rows,
        ["class_name", "ap", "precision", "recall", "f1", "tp", "fp", "fn", "support"],
    )
    write_csv_dicts(
        pred_dir / "object_size_performance.csv",
        _object_size_summary(gt, preds),
        ["size_bin", "support", "predictions"],
    )
    metrics.update(eval_metrics)
    metrics.update(
        {
            "status": "completed",
            "evaluation_status": "completed",
            "evaluated_split": split,
            "prediction_count": len(preds),
            "ground_truth_count": len(gt),
            "false_positive_count": len(false_positives),
            "false_negative_count": len(false_negatives),
            "evaluated_from": str(predictions_path),
        }
    )
    write_json(exp_dir / "metrics.json", metrics)
    return metrics


def _prepared_root(exp_dir: Path, config: dict) -> str:
    config_used = exp_dir / "config_used.yaml"
    if config_used.exists():
        try:
            from fishdetect.config import load_yaml

            return load_yaml(config_used)["dataset"]["prepared_root"]
        except Exception:
            pass
    return config["dataset"]["prepared_root"]


def _object_size_summary(gt: list[dict], preds: list[dict]) -> list[dict]:
    support = {}
    pred_counts = {}
    for row in gt:
        key = object_size_bin(row)
        support[key] = support.get(key, 0) + 1
    for row in preds:
        if "width" not in row:
            continue
        key = object_size_bin(row)
        pred_counts[key] = pred_counts.get(key, 0) + 1
    return [
        {"size_bin": key, "support": support.get(key, 0), "predictions": pred_counts.get(key, 0)}
        for key in ["small", "medium", "large"]
    ]


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
