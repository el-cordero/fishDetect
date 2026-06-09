from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from fishdetect.evaluation.detection_metrics import bbox_iou
from fishdetect.utils.files import read_csv_dicts, read_json, write_csv_dicts, write_json


PREDICTION_FIELDS = [
    "image_id",
    "filename",
    "class_name",
    "score",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
]


def load_ground_truth(prepared_root: str | Path, split: str = "test") -> list[dict[str, Any]]:
    prepared = Path(prepared_root)
    annotations = read_csv_dicts(prepared / "annotations_common.csv")
    split_rows = read_csv_dicts(prepared / "metadata" / "split_manifest.csv")
    split_ids = {str(row["image_id"]) for row in split_rows if row["split"] == split}
    out = []
    for row in annotations:
        if str(row["image_id"]) in split_ids:
            out.append(_coerce_box_row(row))
    return out


def load_split_images(prepared_root: str | Path, split: str = "test") -> list[dict[str, Any]]:
    rows = read_csv_dicts(Path(prepared_root) / "metadata" / "split_manifest.csv")
    return [row for row in rows if row["split"] == split]


def load_predictions(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    if path.suffix.lower() == ".json":
        data = read_json(path)
        rows = data.get("predictions", data if isinstance(data, list) else [])
    else:
        rows = read_csv_dicts(path)
    return [_coerce_prediction_row(row) for row in rows]


def save_predictions(path: str | Path, predictions: list[dict[str, Any]], metadata: dict[str, Any] | None = None) -> None:
    payload = {
        "schema": "fishdetect.detection_predictions.v1",
        "metadata": metadata or {},
        "predictions": [_prediction_json_row(row) for row in predictions],
    }
    write_json(path, payload)
    write_csv_dicts(Path(path).with_suffix(".csv"), payload["predictions"], PREDICTION_FIELDS)


def match_predictions(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    gt_by_image_class: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for gt in ground_truth:
        gt_by_image_class[(str(gt["image_id"]), gt["class_name"])].append(gt)
    matched_gt = set()
    matches = []
    false_positives = []
    for pred_index, pred in enumerate(sorted(predictions, key=lambda row: float(row.get("score", 0)), reverse=True)):
        key = (str(pred["image_id"]), pred["class_name"])
        best_iou = 0.0
        best_index = None
        for gt_index, gt in enumerate(gt_by_image_class.get(key, [])):
            match_key = (key, gt_index)
            if match_key in matched_gt:
                continue
            iou = bbox_iou(_box(gt), _box(pred))
            if iou > best_iou:
                best_iou = iou
                best_index = gt_index
        if best_index is not None and best_iou >= iou_threshold:
            matched_gt.add((key, best_index))
            matches.append({**pred, "match_status": "tp", "iou": best_iou})
        else:
            false_positives.append({**pred, "match_status": "fp", "iou": best_iou})

    false_negatives = []
    for key, rows in gt_by_image_class.items():
        for gt_index, gt in enumerate(rows):
            if (key, gt_index) not in matched_gt:
                false_negatives.append({**gt, "match_status": "fn", "iou": 0.0, "score": ""})
    return matches, false_positives, false_negatives


def save_error_tables(
    output_dir: str | Path,
    matches: list[dict[str, Any]],
    false_positives: list[dict[str, Any]],
    false_negatives: list[dict[str, Any]],
) -> None:
    out = Path(output_dir)
    fields = PREDICTION_FIELDS + ["match_status", "iou"]
    write_csv_dicts(out / "true_positives.csv", matches, fields)
    write_csv_dicts(out / "false_positives.csv", false_positives, fields)
    write_csv_dicts(out / "false_negatives.csv", false_negatives, fields)


def _coerce_box_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for field in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]:
        out[field] = float(out[field])
    out["image_id"] = str(out["image_id"])
    out["filename"] = str(out.get("filename", ""))
    out["class_name"] = str(out["class_name"])
    return out


def _coerce_prediction_row(row: dict[str, Any]) -> dict[str, Any]:
    out = _coerce_box_row(row)
    out["score"] = float(out.get("score", 0.0) or 0.0)
    return out


def _prediction_json_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "image_id": str(row["image_id"]),
        "filename": row.get("filename", ""),
        "class_name": row["class_name"],
        "score": float(row.get("score", 0.0) or 0.0),
        "bbox_x1": float(row["bbox_x1"]),
        "bbox_y1": float(row["bbox_y1"]),
        "bbox_x2": float(row["bbox_x2"]),
        "bbox_y2": float(row["bbox_y2"]),
    }


def _box(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return float(row["bbox_x1"]), float(row["bbox_y1"]), float(row["bbox_x2"]), float(row["bbox_y2"])
