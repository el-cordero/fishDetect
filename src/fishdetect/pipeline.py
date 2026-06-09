from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from fishdetect.converters import (
    class_names_from_annotations,
    export_annotation_csv,
    export_coco_detection,
    export_yolo_detection,
)
from fishdetect.dataset import build_annotation_table, class_count_summary, save_dataset_tables, validate_dataset
from fishdetect.splits import assert_no_leakage, load_split_manifest, make_split_manifest, save_split_manifest
from fishdetect.utils.files import ensure_dir, write_csv_dicts, write_json


def prepare_dataset_pipeline(
    config: dict[str, Any],
    max_images: int | None = None,
    prepared_root_override: str | Path | None = None,
    reuse_split: str | Path | None = None,
) -> dict[str, Any]:
    dataset_cfg = config["dataset"]
    dataset_root = Path(dataset_cfg["root"])
    prepared_root = Path(prepared_root_override or dataset_cfg["prepared_root"])
    ensure_dir(prepared_root)

    annotations, images, meta = build_annotation_table(dataset_root)
    validation = validate_dataset(dataset_root, annotations, images)
    if not validation["passed"]:
        message = "\n".join(validation["errors"][:20])
        raise ValueError(f"Dataset validation failed:\n{message}")

    metadata_dir = ensure_dir(prepared_root / "metadata")
    save_dataset_tables(metadata_dir, annotations, images, validation)
    class_summary = class_count_summary(annotations)
    write_csv_dicts(metadata_dir / "class_summary.csv", class_summary, ["class_name", "annotation_count", "count_bin"])

    split_path = Path(reuse_split) if reuse_split else metadata_dir / "split_manifest.csv"
    if reuse_split:
        split_manifest = load_split_manifest(split_path)
    else:
        split_manifest = make_split_manifest(
            images,
            annotations,
            train=float(dataset_cfg.get("train", 0.70)),
            val=float(dataset_cfg.get("val", 0.15)),
            test=float(dataset_cfg.get("test", 0.15)),
            seed=int(dataset_cfg.get("split_seed", 23401)),
            group_by=_normalize_group_by(dataset_cfg.get("group_by", "sha256")),
            stratify=bool(dataset_cfg.get("stratify", True)),
            max_images=max_images,
        )
        save_split_manifest(split_path, split_manifest)
    assert_no_leakage(split_manifest)

    split_image_ids = {str(row["image_id"]) for row in split_manifest}
    export_annotations = [row for row in annotations if str(row["image_id"]) in split_image_ids]
    export_images = [row for row in images if str(row["image_id"]) in split_image_ids]
    class_names = class_names_from_annotations(annotations)
    split_summary_rows = _split_summary(split_manifest, export_annotations)

    export_annotation_csv(prepared_root, export_annotations)
    write_csv_dicts(metadata_dir / "split_summary.csv", split_summary_rows, ["split", "image_count", "annotation_count", "class_count"])
    write_csv_dicts(
        metadata_dir / "class_counts_by_split.csv",
        _class_counts_by_split(split_manifest, export_annotations),
        ["split", "class_name", "annotation_count"],
    )
    yolo_root = export_yolo_detection(
        prepared_root,
        export_annotations,
        export_images,
        split_manifest,
        class_names=class_names,
        link_images=bool(dataset_cfg.get("link_images", True)),
    )
    coco_root = export_coco_detection(
        prepared_root,
        export_annotations,
        export_images,
        split_manifest,
        class_names=class_names,
        link_images=bool(dataset_cfg.get("link_images", True)),
    )
    summary = {
        "dataset_root": str(dataset_root),
        "prepared_root": str(prepared_root),
        "max_images": max_images,
        "image_count_total": len(images),
        "annotation_count_total": len(annotations),
        "image_count_exported": len(export_images),
        "annotation_count_exported": len(export_annotations),
        "class_count": len(class_names),
        "dive_geometry_annotation_count_exported": sum(1 for row in export_annotations if row.get("has_dive_geometry")),
        "split_manifest": str(split_path),
        "yolo_data_yaml": str(yolo_root / "data.yaml"),
        "coco_root": str(coco_root),
        "mask_export_supported": False,
        "mask_export_note": "Bounding-box dataset only; no mask exports are created.",
        "validation": validation,
        "source_meta": meta,
        "split_summary": split_summary_rows,
    }
    write_json(metadata_dir / "prepare_summary.json", summary)
    return summary


def _normalize_group_by(value: str) -> str:
    aliases = {"sha256": "sha256_if_available", "source": "source_file_or_dataset_if_available"}
    return aliases.get(value, value)


def _split_summary(split_manifest: list[dict[str, Any]], annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    split_by_image = {str(row["image_id"]): row["split"] for row in split_manifest}
    image_counts = Counter(row["split"] for row in split_manifest)
    ann_counts = Counter()
    classes = defaultdict(set)
    for ann in annotations:
        split = split_by_image.get(str(ann["image_id"]), "unknown")
        ann_counts[split] += 1
        classes[split].add(ann["class_name"])
    return [
        {
            "split": split,
            "image_count": image_counts.get(split, 0),
            "annotation_count": ann_counts.get(split, 0),
            "class_count": len(classes.get(split, set())),
        }
        for split in ["train", "val", "test"]
    ]


def _class_counts_by_split(split_manifest: list[dict[str, Any]], annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    split_by_image = {str(row["image_id"]): row["split"] for row in split_manifest}
    counts = Counter((split_by_image.get(str(ann["image_id"]), "unknown"), ann["class_name"]) for ann in annotations)
    return [
        {"split": split, "class_name": class_name, "annotation_count": count}
        for (split, class_name), count in sorted(counts.items())
    ]
