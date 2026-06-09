#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fishdetect.evaluation.plots import (
    save_barplot,
    save_per_class_heatmap,
    save_precision_recall_scatter,
    save_rare_class_plot,
)
from fishdetect.utils.files import ensure_dir, write_csv_dicts, write_json, write_text


FIELDS = [
    "experiment",
    "family",
    "model",
    "status",
    "mAP50",
    "mAP50_95",
    "precision",
    "recall",
    "f1",
    "inference_speed_ms",
    "model_size_mb",
    "training_time_seconds",
    "reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare FishDetect experiment metrics.")
    parser.add_argument("--output-root", default="outputs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    comparison = ensure_dir(output_root / "comparison")
    rows = []
    for metrics_path in sorted((output_root / "experiments").glob("*/metrics.json")):
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        row = {field: metrics.get(field) for field in FIELDS}
        row["experiment"] = row.get("experiment") or metrics_path.parent.name
        rows.append(row)
    write_csv_dicts(comparison / "model_comparison.csv", rows, FIELDS)
    write_text(comparison / "model_comparison.md", _markdown_table(rows))
    write_text(comparison / "model_comparison.html", _html_table(rows))
    best = _best_row(rows)
    write_json(comparison / "best_model_summary.json", best or {})
    plots = ensure_dir(comparison / "plots")
    save_barplot(plots / "map50_barplot.png", rows, "experiment", "mAP50", "mAP50")
    save_barplot(plots / "map5095_barplot.png", rows, "experiment", "mAP50_95", "mAP50-95")
    save_precision_recall_scatter(plots / "precision_recall_scatter.png", rows)
    per_class_rows = _load_per_class_rows(output_root)
    save_per_class_heatmap(plots / "per_class_performance_heatmap.png", per_class_rows)
    save_rare_class_plot(plots / "rare_class_performance.png", per_class_rows)
    print(f"Wrote comparison for {len(rows)} experiment(s) to {comparison}.")
    return 0


def _num(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _best_row(rows):
    scored = [(row, _num(row.get("mAP50_95")) or _num(row.get("mAP50")) or -1.0) for row in rows]
    scored = [item for item in scored if item[1] >= 0]
    return max(scored, key=lambda item: item[1])[0] if scored else None


def _markdown_table(rows):
    lines = ["# Model Comparison", ""]
    if not rows:
        return "# Model Comparison\n\nNo metrics found.\n"
    lines.append("| " + " | ".join(FIELDS) + " |")
    lines.append("| " + " | ".join(["---"] * len(FIELDS)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in FIELDS) + " |")
    return "\n".join(lines) + "\n"


def _html_table(rows):
    header = "".join(f"<th>{field}</th>" for field in FIELDS)
    body = "\n".join("<tr>" + "".join(f"<td>{row.get(field, '')}</td>" for field in FIELDS) + "</tr>" for row in rows)
    return f"<html><body><h1>Model Comparison</h1><table border='1'><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></body></html>\n"


def _load_per_class_rows(output_root: Path):
    import csv

    rows = []
    for path in sorted((output_root / "experiments").glob("*/per_class_metrics.csv")):
        experiment = path.parent.name
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("class_name"):
                    row["experiment"] = experiment
                    rows.append(row)
    return rows


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
