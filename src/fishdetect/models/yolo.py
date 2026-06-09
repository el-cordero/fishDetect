from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fishdetect.experiment import file_size_mb, skipped_metrics
from fishdetect.transforms import yolo_augmentation_kwargs
from fishdetect.utils.seed import detect_device


def ultralytics_available() -> bool:
    try:
        import ultralytics  # noqa: F401

        return True
    except Exception:
        return False


def train_yolo(
    config: dict[str, Any],
    experiment: dict[str, Any],
    output_dir: str | Path,
    epochs: int | None = None,
    device: str = "auto",
    smoke_test: bool = False,
    allow_downloads: bool = False,
) -> dict[str, Any]:
    training_cfg = config.get("training", {})
    dataset_cfg = config["dataset"]
    data_yaml = Path(dataset_cfg["prepared_root"]) / "yolo_det" / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"YOLO data.yaml not found. Run prepare_dataset first: {data_yaml}")

    model_name = experiment["model"]
    model_path = Path(model_name)
    skip_downloads = bool(training_cfg.get("skip_network_weight_downloads_in_smoke", True))
    if smoke_test and skip_downloads and not model_path.exists() and not Path(output_dir, model_name).exists():
        return skipped_metrics(
            f"Smoke mode skipped remote weight '{model_name}' to avoid network/download side effects.",
            experiment,
            smoke_test=True,
        )
    if not ultralytics_available():
        return skipped_metrics("Ultralytics is not installed.", experiment, smoke_test=smoke_test)

    from ultralytics import YOLO

    start = time.time()
    resolved_device = detect_device(device)
    model = YOLO(model_name)
    train_kwargs = {
        "data": str(data_yaml),
        "epochs": int(epochs or training_cfg.get("epochs", 100)),
        "imgsz": int(training_cfg.get("imgsz", 1280)),
        "batch": training_cfg.get("batch", "auto"),
        "patience": int(training_cfg.get("patience", 25)),
        "workers": int(training_cfg.get("workers", 4)),
        "device": _ultralytics_device(resolved_device),
        "amp": bool(training_cfg.get("amp", True)),
        "project": str(Path(output_dir)),
        "name": "ultralytics",
        "exist_ok": True,
        "plots": True,
    }
    train_kwargs.update(yolo_augmentation_kwargs(config.get("augmentation", {})))
    model.train(**train_kwargs)
    val_metrics = model.val(data=str(data_yaml), split="test", device=_ultralytics_device(resolved_device))
    metrics = _extract_ultralytics_metrics(val_metrics)
    metrics.update(
        {
            "status": "completed",
            "smoke_test": smoke_test,
            "training_time_seconds": round(time.time() - start, 3),
            "device": resolved_device,
            "model_size_mb": _best_weight_size(Path(output_dir)),
        }
    )
    return metrics


def _ultralytics_device(device: str) -> str:
    if device.startswith("cuda:"):
        return device.split(":", 1)[1]
    return device


def _extract_ultralytics_metrics(result: Any) -> dict[str, Any]:
    box = getattr(result, "box", None)
    speed = getattr(result, "speed", {}) or {}
    return {
        "mAP50": float(getattr(box, "map50", 0.0) or 0.0) if box else 0.0,
        "mAP50_95": float(getattr(box, "map", 0.0) or 0.0) if box else 0.0,
        "precision": float(getattr(box, "mp", 0.0) or 0.0) if box else 0.0,
        "recall": float(getattr(box, "mr", 0.0) or 0.0) if box else 0.0,
        "f1": None,
        "per_class": {},
        "inference_speed_ms": speed.get("inference"),
    }


def _best_weight_size(output_dir: Path) -> float | None:
    candidates = sorted(output_dir.rglob("best.pt"))
    if not candidates:
        candidates = sorted(output_dir.rglob("last.pt"))
    return file_size_mb(candidates[0]) if candidates else None
