from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from fishdetect.config import dump_yaml
from fishdetect.evaluation.plots import save_confusion_matrix_placeholder
from fishdetect.utils.files import ensure_dir, write_csv_dicts, write_json, write_text
from fishdetect.utils.seed import detect_device, package_versions


def init_experiment_dir(
    output_root: str | Path,
    experiment: dict[str, Any],
    config: dict[str, Any],
    command: str,
) -> Path:
    out = ensure_dir(Path(output_root) / "experiments" / experiment["name"])
    ensure_dir(out / "predictions" / "sample_predictions")
    ensure_dir(out / "predictions" / "false_positives")
    ensure_dir(out / "predictions" / "false_negatives")
    ensure_dir(out / "weights")
    write_text(out / "config_used.yaml", dump_yaml(config))
    env = {
        "command": command,
        "created_at_unix": time.time(),
        "git_commit": git_commit(),
        "versions": package_versions(),
        "device_auto": detect_device("auto"),
    }
    write_json(out / "environment.json", env)
    return out


def write_metrics_artifacts(
    out: str | Path,
    experiment: dict[str, Any],
    metrics: dict[str, Any],
    status_note: str,
) -> None:
    out = Path(out)
    metrics.setdefault("experiment", experiment["name"])
    metrics.setdefault("family", experiment.get("family", ""))
    metrics.setdefault("model", experiment.get("model", ""))
    write_json(out / "metrics.json", metrics)
    per_class = metrics.get("per_class", {})
    rows = []
    if isinstance(per_class, dict):
        for class_name, values in per_class.items():
            row = {"class_name": class_name}
            if isinstance(values, dict):
                row.update(values)
            rows.append(row)
    write_csv_dicts(
        out / "per_class_metrics.csv",
        rows,
        ["class_name", "ap", "precision", "recall", "f1", "tp", "fp", "fn", "support"],
    )
    write_csv_dicts(out / "train_log.csv", metrics.get("train_log", []), ["epoch", "metric", "value"])
    save_confusion_matrix_placeholder(out / "confusion_matrix.png")
    save_confusion_matrix_placeholder(out / "pr_curve.png", title="Precision-Recall Curve")
    save_confusion_matrix_placeholder(out / "f1_curve.png", title="F1 Curve")
    write_model_card(out / "model_card.md", experiment, metrics, status_note)


def skipped_metrics(reason: str, experiment: dict[str, Any], smoke_test: bool = False) -> dict[str, Any]:
    return {
        "status": "skipped",
        "reason": reason,
        "smoke_test": smoke_test,
        "mAP50": None,
        "mAP50_95": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "per_class": {},
        "inference_speed_ms": None,
        "model_size_mb": None,
        "training_time_seconds": 0.0,
        "experiment": experiment["name"],
        "family": experiment.get("family", ""),
        "model": experiment.get("model", ""),
    }


def write_model_card(path: str | Path, experiment: dict[str, Any], metrics: dict[str, Any], status_note: str) -> None:
    lines = [
        f"# {experiment['name']}",
        "",
        f"family: {experiment.get('family', '')}",
        f"model: {experiment.get('model', '')}",
        f"task: {experiment.get('task', 'detect')}",
        f"status: {metrics.get('status', 'completed')}",
        f"note: {status_note}",
        "",
        "## Metrics",
        "",
        f"- mAP50: {metrics.get('mAP50')}",
        f"- mAP50-95: {metrics.get('mAP50_95')}",
        f"- Precision: {metrics.get('precision')}",
        f"- Recall: {metrics.get('recall')}",
        f"- F1: {metrics.get('f1')}",
        "",
        "## Notes",
        "",
        "This dataset is box-only. Do not treat DIVE geometry as a mask.",
        "Review rare classes and look-alike species with image galleries before using a model outside this dataset.",
    ]
    write_text(path, "\n".join(lines) + "\n")


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def command_string(argv: list[str] | None = None) -> str:
    return " ".join(argv or sys.argv)


def file_size_mb(path: str | Path) -> float | None:
    path = Path(path)
    if not path.exists():
        return None
    return round(path.stat().st_size / (1024 * 1024), 4)
