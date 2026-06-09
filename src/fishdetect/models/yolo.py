from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fishdetect.experiment import file_size_mb, skipped_metrics
from fishdetect.predictions import save_predictions
from fishdetect.transforms import yolo_augmentation_kwargs
from fishdetect.utils.files import ensure_dir
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
    skip_downloads = bool(training_cfg.get("skip_network_weight_downloads_in_smoke", True))
    if _would_require_download(model_name) and not allow_downloads and (smoke_test or not training_cfg.get("allow_remote_weight_downloads", False)):
        return skipped_metrics(
            f"Skipped '{model_name}' because weights are not local and remote downloads are disabled.",
            experiment,
            smoke_test=smoke_test,
        )
    if not ultralytics_available():
        return skipped_metrics("Ultralytics is not installed.", experiment, smoke_test=smoke_test)

    from ultralytics import YOLO

    start = time.time()
    resolved_device = detect_device(device)
    try:
        model = YOLO(model_name)
    except Exception as exc:
        return skipped_metrics(f"Ultralytics could not load '{model_name}': {exc}", experiment, smoke_test=smoke_test)
    train_kwargs = {
        "data": str(data_yaml),
        "epochs": int(epochs or training_cfg.get("epochs", 100)),
        "imgsz": int(training_cfg.get("imgsz", 1280)),
        "batch": _resolve_batch(training_cfg.get("batch", 4), smoke_test=smoke_test),
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
    try:
        model.train(**train_kwargs)
    except Exception as exc:
        if resolved_device == "mps" and device == "auto":
            train_kwargs["device"] = "cpu"
            resolved_device = "cpu"
            model.train(**train_kwargs)
        else:
            return skipped_metrics(f"Training failed for '{model_name}': {exc}", experiment, smoke_test=smoke_test)
    val_metrics = model.val(data=str(data_yaml), split="test", device=_ultralytics_device(resolved_device))
    metrics = _extract_ultralytics_metrics(val_metrics)
    best_weight = _best_weight_path(Path(output_dir))
    if best_weight:
        predictions_path = Path(output_dir) / "predictions" / "predictions.json"
        try:
            export_yolo_predictions(
                model_path=best_weight,
                prepared_root=dataset_cfg["prepared_root"],
                split="test",
                output_json=predictions_path,
                device=resolved_device,
            )
        except Exception as exc:
            metrics["prediction_export_warning"] = str(exc)
    metrics.update(
        {
            "status": "completed",
            "smoke_test": smoke_test,
            "training_time_seconds": round(time.time() - start, 3),
            "device": resolved_device,
            "model_size_mb": file_size_mb(best_weight) if best_weight else None,
            "best_weights_path": str(best_weight) if best_weight else None,
        }
    )
    return metrics


def export_yolo_predictions(
    model_path: str | Path,
    prepared_root: str | Path,
    split: str,
    output_json: str | Path,
    device: str = "auto",
    conf: float = 0.001,
) -> list[dict[str, Any]]:
    if not ultralytics_available():
        raise RuntimeError("Ultralytics is not installed.")
    from ultralytics import YOLO

    prepared = Path(prepared_root)
    image_dir = prepared / "yolo_det" / "images" / split
    if not image_dir.exists():
        raise FileNotFoundError(f"Image split directory not found: {image_dir}")
    split_rows = _load_split_rows(prepared, split)
    row_by_filename = {row["filename"]: row for row in split_rows}
    model = YOLO(str(model_path))
    results = model.predict(
        source=str(image_dir),
        conf=conf,
        device=_ultralytics_device(detect_device(device)),
        verbose=False,
        stream=False,
    )
    names = getattr(model, "names", {}) or {}
    predictions: list[dict[str, Any]] = []
    for result in results:
        filename = Path(getattr(result, "path", "")).name
        split_row = row_by_filename.get(filename, {"image_id": Path(filename).stem, "filename": filename})
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xyxy = boxes.xyxy.cpu().tolist() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy.tolist()
        cls = boxes.cls.cpu().tolist() if hasattr(boxes.cls, "cpu") else boxes.cls.tolist()
        confs = boxes.conf.cpu().tolist() if hasattr(boxes.conf, "cpu") else boxes.conf.tolist()
        for box, class_id, score in zip(xyxy, cls, confs):
            class_id = int(class_id)
            predictions.append(
                {
                    "image_id": str(split_row["image_id"]),
                    "filename": filename,
                    "class_name": names.get(class_id, str(class_id)),
                    "score": float(score),
                    "bbox_x1": float(box[0]),
                    "bbox_y1": float(box[1]),
                    "bbox_x2": float(box[2]),
                    "bbox_y2": float(box[3]),
                }
            )
    save_predictions(
        output_json,
        predictions,
        metadata={"model_path": str(model_path), "prepared_root": str(prepared), "split": split, "conf": conf},
    )
    return predictions


def _ultralytics_device(device: str) -> str:
    if device.startswith("cuda:"):
        return device.split(":", 1)[1]
    return device


def _resolve_batch(batch: Any, smoke_test: bool = False) -> int | float:
    """Convert config batch values into Ultralytics-compatible numeric values."""
    if isinstance(batch, str):
        value = batch.strip().lower()
        if value == "auto":
            return 2 if smoke_test else 4
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid batch value: {batch!r}. Use an integer like 4, "
                    "a float like 0.5, or 'auto'."
                ) from exc
    return batch


def _would_require_download(model_name: str) -> bool:
    path = Path(model_name)
    if path.exists():
        return False
    if Path.cwd().joinpath(model_name).exists():
        return False
    return model_name.endswith(".pt")


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


def _best_weight_path(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.rglob("best.pt"))
    if not candidates:
        candidates = sorted(output_dir.rglob("last.pt"))
    return candidates[0] if candidates else None


def _load_split_rows(prepared_root: Path, split: str) -> list[dict[str, Any]]:
    import csv

    path = prepared_root / "metadata" / "split_manifest.csv"
    with path.open("r", encoding="utf-8", newline="") as f:
        return [row for row in csv.DictReader(f) if row["split"] == split]
