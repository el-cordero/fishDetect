from __future__ import annotations

from collections import defaultdict
from typing import Any


def bbox_iou(box_a: list[float] | tuple[float, float, float, float], box_b: list[float] | tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def evaluate_detections(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    class_names: list[str] | None = None,
    iou_thresholds: list[float] | None = None,
) -> dict[str, Any]:
    iou_thresholds = iou_thresholds or [round(0.5 + i * 0.05, 2) for i in range(10)]
    class_names = class_names or sorted({row["class_name"] for row in ground_truth} | {row["class_name"] for row in predictions})
    per_threshold = {
        threshold: _evaluate_at_threshold(ground_truth, predictions, class_names, threshold)
        for threshold in iou_thresholds
    }
    at50 = per_threshold[iou_thresholds[0]]
    map50_95 = sum(item["mAP"] for item in per_threshold.values()) / len(per_threshold) if per_threshold else 0.0
    return {
        "mAP50": at50["mAP"],
        "mAP50_95": map50_95,
        "precision": at50["precision"],
        "recall": at50["recall"],
        "f1": at50["f1"],
        "per_class": at50["per_class"],
        "thresholds": per_threshold,
    }


def _evaluate_at_threshold(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    class_names: list[str],
    iou_threshold: float,
) -> dict[str, Any]:
    gt_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for gt in ground_truth:
        gt_by_key[(str(gt["image_id"]), gt["class_name"])].append(gt)
    pred_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pred in predictions:
        pred_by_class[pred["class_name"]].append(pred)
    per_class = {}
    total_tp = total_fp = total_fn = 0
    aps = []
    for class_name in class_names:
        gt_count = sum(len(v) for key, v in gt_by_key.items() if key[1] == class_name)
        matched = set()
        preds = sorted(pred_by_class.get(class_name, []), key=lambda row: float(row.get("score", 1.0)), reverse=True)
        tp_flags: list[int] = []
        fp_flags: list[int] = []
        for pred in preds:
            key = (str(pred["image_id"]), class_name)
            candidates = gt_by_key.get(key, [])
            best_iou = 0.0
            best_index = None
            for index, gt in enumerate(candidates):
                match_key = (key, index)
                if match_key in matched:
                    continue
                iou = bbox_iou(_box(gt), _box(pred))
                if iou > best_iou:
                    best_iou = iou
                    best_index = index
            if best_index is not None and best_iou >= iou_threshold:
                matched.add((key, best_index))
                tp_flags.append(1)
                fp_flags.append(0)
            else:
                tp_flags.append(0)
                fp_flags.append(1)
        tp = sum(tp_flags)
        fp = sum(fp_flags)
        fn = max(0, gt_count - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        ap = _average_precision(tp_flags, fp_flags, gt_count)
        aps.append(ap)
        per_class[class_name] = {
            "ap": ap,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": gt_count,
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "mAP": sum(aps) / len(aps) if aps else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_class": per_class,
    }


def _average_precision(tp_flags: list[int], fp_flags: list[int], gt_count: int) -> float:
    if gt_count == 0:
        return 0.0
    cum_tp = 0
    cum_fp = 0
    precisions = []
    recalls = []
    for tp, fp in zip(tp_flags, fp_flags):
        cum_tp += tp
        cum_fp += fp
        precisions.append(cum_tp / (cum_tp + cum_fp) if cum_tp + cum_fp else 0.0)
        recalls.append(cum_tp / gt_count)
    ap = 0.0
    for threshold in [i / 10 for i in range(11)]:
        precision_at_recall = max((p for p, r in zip(precisions, recalls) if r >= threshold), default=0.0)
        ap += precision_at_recall
    return ap / 11.0


def _box(row: dict[str, Any]) -> tuple[float, float, float, float]:
    if "bbox" in row:
        bbox = row["bbox"]
        if len(bbox) == 4:
            return float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    return (
        float(row["bbox_x1"]),
        float(row["bbox_y1"]),
        float(row["bbox_x2"]),
        float(row["bbox_y2"]),
    )


def object_size_bin(row: dict[str, Any]) -> str:
    area = (float(row["bbox_x2"]) - float(row["bbox_x1"])) * (float(row["bbox_y2"]) - float(row["bbox_y1"]))
    image_area = max(float(row.get("width", 0)) * float(row.get("height", 0)), 1.0)
    ratio = area / image_area
    if ratio < 0.01:
        return "small"
    if ratio < 0.05:
        return "medium"
    return "large"
