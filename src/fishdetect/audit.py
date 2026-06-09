from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from fishdetect.dataset import build_annotation_table, class_count_summary, validate_dataset
from fishdetect.evaluation.detection_metrics import object_size_bin
from fishdetect.utils.files import ensure_dir, write_csv_dicts, write_json, write_text


def audit_dataset(dataset_root: str | Path, out_dir: str | Path, rare_threshold: int = 5) -> dict[str, Any]:
    out = ensure_dir(out_dir)
    annotations, images, meta = build_annotation_table(dataset_root)
    validation = validate_dataset(dataset_root, annotations, images)

    class_rows = class_count_summary(annotations)
    rare_classes = [row for row in class_rows if int(row["annotation_count"]) < rare_threshold]

    image_ann_counts = Counter(str(row["image_id"]) for row in annotations)
    image_classes: dict[str, set[str]] = defaultdict(set)
    size_counts = Counter()
    areas = []
    for row in annotations:
        image_classes[str(row["image_id"])].add(row["class_name"])
        size_counts[object_size_bin(row)] += 1
        area = (float(row["bbox_x2"]) - float(row["bbox_x1"])) * (float(row["bbox_y2"]) - float(row["bbox_y1"]))
        image_area = max(float(row["width"]) * float(row["height"]), 1.0)
        areas.append(area / image_area)

    multi_ann = sum(1 for count in image_ann_counts.values() if count > 1)
    multi_class = sum(1 for classes in image_classes.values() if len(classes) > 1)
    area_summary = _numeric_summary(areas)
    summary = {
        "dataset_root": str(dataset_root),
        "box_only": True,
        "image_count": len(images),
        "annotation_count": len(annotations),
        "class_count": len(class_rows),
        "rare_class_count": len(rare_classes),
        "images_with_annotations": len(image_ann_counts),
        "images_without_annotations": validation.get("image_without_annotations", 0),
        "images_with_multiple_annotations": multi_ann,
        "images_with_multiple_classes": multi_class,
        "bbox_relative_area": area_summary,
        "bbox_size_bins": dict(size_counts),
        "validation": validation,
        "source_meta": meta,
    }
    write_json(out / "dataset_audit_summary.json", summary)
    write_csv_dicts(out / "class_frequency.csv", class_rows, ["class_name", "annotation_count", "count_bin"])
    write_csv_dicts(out / "rare_classes.csv", rare_classes, ["class_name", "annotation_count", "count_bin"])
    write_csv_dicts(
        out / "bbox_size_bins.csv",
        [{"size_bin": key, "annotation_count": value} for key, value in sorted(size_counts.items())],
        ["size_bin", "annotation_count"],
    )
    write_text(out / "dataset_audit.md", _summary_markdown(summary, rare_classes))
    return summary


def _numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    ordered = sorted(values)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return {
        "count": len(values),
        "min": round(ordered[0], 6),
        "median": round(median, 6),
        "mean": round(sum(values) / len(values), 6),
        "max": round(ordered[-1], 6),
    }


def _summary_markdown(summary: dict[str, Any], rare_classes: list[dict[str, Any]]) -> str:
    lines = [
        "# Dataset Audit",
        "",
        "Annotation type: bounding boxes only.",
        "",
        f"- Images: {summary['image_count']}",
        f"- Annotations: {summary['annotation_count']}",
        f"- Classes: {summary['class_count']}",
        f"- Rare classes below threshold: {summary['rare_class_count']}",
        f"- Images with multiple annotations: {summary['images_with_multiple_annotations']}",
        f"- Images with multiple classes: {summary['images_with_multiple_classes']}",
        "",
        "## Rare Classes",
        "",
    ]
    if rare_classes:
        for row in rare_classes:
            lines.append(f"- {row['class_name']}: {row['annotation_count']}")
    else:
        lines.append("None")
    lines.append("")
    return "\n".join(lines)
