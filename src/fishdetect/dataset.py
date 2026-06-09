from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from fishdetect.dive import load_meta_image_map, parse_dive_json
from fishdetect.utils.files import image_size, read_csv_dicts, write_csv_dicts, write_json
from fishdetect.viame import parse_viame_csv


ANNOTATION_FIELDS = [
    "annotation_id",
    "image_id",
    "frame",
    "filename",
    "image_path",
    "width",
    "height",
    "class_name",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "track_id",
    "source_file_or_dataset_if_available",
    "sha256_if_available",
    "has_dive_geometry",
    "dive_geometry_if_available",
]

IMAGE_FIELDS = [
    "image_id",
    "frame",
    "filename",
    "image_path",
    "width",
    "height",
    "sha256_if_available",
    "source_file_or_dataset_if_available",
    "source_filename",
    "manifest_class",
    "annotation_count",
]


def dataset_file(dataset_root: str | Path, name: str) -> Path:
    return Path(dataset_root) / name


def load_image_manifest(dataset_root: str | Path) -> list[dict[str, Any]]:
    path = dataset_file(dataset_root, "image_manifest.csv")
    if not path.exists():
        raise FileNotFoundError(f"Image manifest not found: {path}")
    rows = read_csv_dicts(path)
    by_filename: dict[str, dict[str, Any]] = {}
    provenance: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        filename = row.get("output_filename", "")
        if not filename:
            continue
        frame = int(row.get("global_frame") or row.get("source_frame") or 0)
        image_path = Path(dataset_root) / filename
        if filename not in by_filename:
            by_filename[filename] = {
                "image_id": str(frame),
                "frame": frame,
                "filename": filename,
                "image_path": str(image_path),
                "sha256_if_available": row.get("sha256", ""),
                "source_file_or_dataset_if_available": row.get("source_dataset", "") or row.get("source_path", ""),
                "source_filename": row.get("source_filename", ""),
                "manifest_class": row.get("class", ""),
                "duplicate_of_output_filename": row.get("duplicate_of_output_filename", ""),
                "duplicate_reference_count": 1,
            }
        else:
            by_filename[filename]["duplicate_reference_count"] = int(by_filename[filename].get("duplicate_reference_count", 1)) + 1
        for field in ["source_dataset", "source_path", "source_filename", "class"]:
            if row.get(field):
                provenance[filename][field].add(_public_provenance_value(row[field]))
    out: list[dict[str, Any]] = []
    for filename, item in by_filename.items():
        item["source_file_or_dataset_if_available"] = "|".join(sorted(provenance[filename]["source_dataset"] or provenance[filename]["source_path"]))
        item["source_filename"] = "|".join(sorted(provenance[filename]["source_filename"]))
        item["manifest_class"] = "|".join(sorted(provenance[filename]["class"]))
        out.append(item)
    return sorted(out, key=lambda row: int(row["frame"]))


def _public_provenance_value(value: str) -> str:
    if not value:
        return ""
    normalized = value.replace("\\", "/")
    if "/" in normalized:
        return Path(normalized).name
    return value


def load_class_counts(dataset_root: str | Path) -> list[dict[str, Any]]:
    path = dataset_file(dataset_root, "class_counts.csv")
    if not path.exists():
        return []
    rows = read_csv_dicts(path)
    for row in rows:
        for col in ["dive_feature_count", "viame_row_count"]:
            if col in row and row[col] != "":
                row[col] = int(row[col])
    return rows


def build_annotation_table(dataset_root: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    root = Path(dataset_root)
    viame_rows = parse_viame_csv(root / "annotations.viame.csv", root)
    dive_rows = parse_dive_json(root / "annotations.dive.json", root / "meta.json", root)
    manifest_rows = load_image_manifest(root)
    frame_to_filename = load_meta_image_map(root / "meta.json")

    image_by_filename = {row["filename"]: row for row in manifest_rows}
    image_by_frame = {int(row["frame"]): row for row in manifest_rows}
    dive_by_track_frame = defaultdict(list)
    dive_by_bbox_key = defaultdict(list)
    for row in dive_rows:
        dive_by_track_frame[(str(row["track_id"]), int(row["frame"]))].append(row)
        dive_by_bbox_key[_bbox_match_key(row)].append(row)

    dim_cache: dict[str, tuple[int, int]] = {}
    images: list[dict[str, Any]] = []
    for row in manifest_rows:
        width, height = _cached_image_size(row["image_path"], dim_cache)
        row["width"] = width
        row["height"] = height
        row["annotation_count"] = 0
        images.append(row)

    annotations: list[dict[str, Any]] = []
    for index, row in enumerate(viame_rows):
        frame = int(row["frame"])
        manifest = image_by_filename.get(row["filename"]) or image_by_frame.get(frame) or {}
        width, height = _cached_image_size(row["image_path"], dim_cache)
        dive_match = _match_dive_row(row, dive_by_track_frame, dive_by_bbox_key)
        if manifest:
            manifest["annotation_count"] = int(manifest.get("annotation_count", 0)) + 1
        annotation = {
            "annotation_id": str(index),
            "image_id": str(manifest.get("image_id", frame)),
            "frame": frame,
            "filename": row["filename"],
            "image_path": row["image_path"],
            "width": width,
            "height": height,
            "class_name": row["class_name"],
            "bbox_x1": float(row["bbox_x1"]),
            "bbox_y1": float(row["bbox_y1"]),
            "bbox_x2": float(row["bbox_x2"]),
            "bbox_y2": float(row["bbox_y2"]),
            "track_id": str(row["track_id"]),
            "source_file_or_dataset_if_available": manifest.get("source_file_or_dataset_if_available", ""),
            "sha256_if_available": manifest.get("sha256_if_available", ""),
            "has_dive_geometry": bool(dive_match and dive_match.get("has_dive_geometry")),
            "dive_geometry_if_available": dive_match.get("dive_geometry_if_available", "") if dive_match else "",
        }
        annotations.append(annotation)

    meta = {
        "viame_rows": len(viame_rows),
        "dive_rows": len(dive_rows),
        "image_count": len(images),
        "frame_to_filename_count": len(frame_to_filename),
        "dive_geometry_annotation_count": sum(1 for row in annotations if row.get("has_dive_geometry")),
    }
    return annotations, images, meta


def validate_dataset(
    dataset_root: str | Path,
    annotations: list[dict[str, Any]],
    images: list[dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    root = Path(dataset_root)
    frame_to_filename = load_meta_image_map(root / "meta.json")
    class_counts = load_class_counts(root)
    expected_classes = {row["class"] for row in class_counts if row.get("class")}
    observed_classes = {row["class_name"] for row in annotations}

    for image in images:
        if not Path(image["image_path"]).exists():
            errors.append(f"Missing image: {image['image_path']}")
        frame = int(image["frame"])
        if frame in frame_to_filename and frame_to_filename[frame] != image["filename"]:
            errors.append(
                f"Frame mapping mismatch for frame {frame}: meta={frame_to_filename[frame]} manifest={image['filename']}"
            )

    duplicate_keys = Counter()
    for row in annotations:
        x1, y1, x2, y2 = [float(row[k]) for k in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]]
        width, height = int(row["width"]), int(row["height"])
        if not Path(row["image_path"]).exists():
            errors.append(f"Annotation image missing: {row['image_path']}")
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            errors.append(f"Invalid bbox in annotation {row['annotation_id']}: {(x1, y1, x2, y2)} for {(width, height)}")
        duplicate_keys[
            (
                row["filename"],
                row["class_name"],
                round(x1, 2),
                round(y1, 2),
                round(x2, 2),
                round(y2, 2),
                row["track_id"],
            )
        ] += 1
    duplicate_count = sum(count - 1 for count in duplicate_keys.values() if count > 1)
    if duplicate_count:
        warnings.append(f"Found {duplicate_count} duplicate-like annotation rows.")
    missing_classes = observed_classes - expected_classes if expected_classes else set()
    if missing_classes:
        warnings.append(f"Classes observed in annotations but absent from class_counts.csv: {sorted(missing_classes)}")

    class_counter = Counter(row["class_name"] for row in annotations)
    image_without_annotations = sum(1 for row in images if int(row.get("annotation_count", 0)) == 0)
    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "image_count": len(images),
        "annotation_count": len(annotations),
        "class_count": len(class_counter),
        "class_counts": dict(sorted(class_counter.items())),
        "image_without_annotations": image_without_annotations,
        "duplicate_like_annotation_count": duplicate_count,
        "dive_geometry_annotation_count": sum(1 for row in annotations if row.get("has_dive_geometry")),
    }


def save_dataset_tables(
    out_dir: str | Path,
    annotations: list[dict[str, Any]],
    images: list[dict[str, Any]],
    validation: dict[str, Any],
) -> None:
    out = Path(out_dir)
    write_csv_dicts(out / "annotations.csv", annotations, ANNOTATION_FIELDS)
    write_csv_dicts(out / "images.csv", images, IMAGE_FIELDS)
    write_json(out / "validation_report.json", validation)


def class_count_summary(annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(row["class_name"] for row in annotations)
    rows = []
    for class_name, count in sorted(counts.items()):
        if count < 5:
            bin_name = "rare"
        elif count < 25:
            bin_name = "medium"
        else:
            bin_name = "common"
        rows.append({"class_name": class_name, "annotation_count": count, "count_bin": bin_name})
    return rows


def _cached_image_size(path: str | Path, cache: dict[str, tuple[int, int]]) -> tuple[int, int]:
    key = str(path)
    if key not in cache:
        cache[key] = image_size(path)
    return cache[key]


def _bbox_match_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["frame"]),
        row["class_name"],
        round(float(row["bbox_x1"]), 1),
        round(float(row["bbox_y1"]), 1),
        round(float(row["bbox_x2"]), 1),
        round(float(row["bbox_y2"]), 1),
    )


def _match_dive_row(
    viame_row: dict[str, Any],
    dive_by_track_frame: dict[Any, list[dict[str, Any]]],
    dive_by_bbox_key: dict[Any, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    candidates = dive_by_track_frame.get((str(viame_row["track_id"]), int(viame_row["frame"])), [])
    if candidates:
        return candidates.pop(0)
    key = _bbox_match_key(viame_row)
    candidates = dive_by_bbox_key.get(key, [])
    if candidates:
        return candidates.pop(0)
    return None
