from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from fishdetect.utils.files import read_csv_dicts, write_csv_dicts


SPLIT_FIELDS = [
    "image_id",
    "frame",
    "filename",
    "image_path",
    "width",
    "height",
    "sha256_if_available",
    "source_file_or_dataset_if_available",
    "split",
    "annotation_count",
    "class_names",
]


def make_split_manifest(
    images: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 23401,
    group_by: str = "sha256_if_available",
    stratify: bool = True,
    max_images: int | None = None,
) -> list[dict[str, Any]]:
    if round(train + val + test, 6) != 1.0:
        raise ValueError("Split fractions must sum to 1.0")
    image_classes: dict[str, set[str]] = defaultdict(set)
    for ann in annotations:
        image_classes[str(ann["image_id"])].add(ann["class_name"])

    selected_images = sorted(images, key=lambda row: int(row["frame"]))
    if max_images is not None:
        annotated_ids = {str(row["image_id"]) for row in annotations}
        selected_images = [row for row in selected_images if str(row["image_id"]) in annotated_ids][:max_images]

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for image in selected_images:
        group_value = image.get(group_by) or image.get("sha256_if_available") or image["filename"]
        groups[str(group_value)].append(image)

    rng = random.Random(seed)
    group_items = list(groups.items())
    if stratify:
        buckets: dict[str, list[tuple[str, list[dict[str, Any]]]]] = defaultdict(list)
        for group_key, members in group_items:
            classes = sorted({cls for member in members for cls in image_classes.get(str(member["image_id"]), set())})
            label = classes[0] if classes else members[0].get("manifest_class", "__unannotated__") or "__unannotated__"
            buckets[label].append((group_key, members))
        ordered_groups: list[tuple[str, list[dict[str, Any]]]] = []
        for label in sorted(buckets):
            rng.shuffle(buckets[label])
            ordered_groups.extend(buckets[label])
    else:
        rng.shuffle(group_items)
        ordered_groups = group_items

    n_images = sum(len(members) for _, members in ordered_groups)
    targets = {"train": train * n_images, "val": val * n_images, "test": test * n_images}
    counts = {"train": 0, "val": 0, "test": 0}
    split_for_group: dict[str, str] = {}
    for group_key, members in ordered_groups:
        deficits = {split: targets[split] - counts[split] for split in targets}
        split = max(deficits, key=deficits.get)
        split_for_group[group_key] = split
        counts[split] += len(members)

    rows: list[dict[str, Any]] = []
    for group_key, members in groups.items():
        split = split_for_group[group_key]
        for image in members:
            classes = sorted(image_classes.get(str(image["image_id"]), []))
            rows.append(
                {
                    "image_id": image["image_id"],
                    "frame": image["frame"],
                    "filename": image["filename"],
                    "image_path": image["image_path"],
                    "width": image.get("width", ""),
                    "height": image.get("height", ""),
                    "sha256_if_available": image.get("sha256_if_available", ""),
                    "source_file_or_dataset_if_available": image.get("source_file_or_dataset_if_available", ""),
                    "split": split,
                    "annotation_count": image.get("annotation_count", 0),
                    "class_names": "|".join(classes),
                }
            )
    return sorted(rows, key=lambda row: int(row["frame"]))


def save_split_manifest(path: str | Path, rows: list[dict[str, Any]]) -> None:
    write_csv_dicts(path, rows, SPLIT_FIELDS)


def load_split_manifest(path: str | Path) -> list[dict[str, Any]]:
    rows = read_csv_dicts(path)
    for row in rows:
        row["frame"] = int(row["frame"])
        row["width"] = int(float(row["width"])) if row.get("width") else ""
        row["height"] = int(float(row["height"])) if row.get("height") else ""
        row["annotation_count"] = int(float(row["annotation_count"])) if row.get("annotation_count") else 0
    return rows


def assert_no_leakage(split_rows: list[dict[str, Any]], group_field: str = "sha256_if_available") -> None:
    seen: dict[str, str] = {}
    for row in split_rows:
        key = str(row.get(group_field) or row["filename"])
        split = row["split"]
        if key in seen and seen[key] != split:
            raise ValueError(f"Leakage detected for group {key}: {seen[key]} and {split}")
        seen[key] = split


def make_kfold_manifests(
    images: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    folds: int = 5,
    seed: int = 23401,
    group_by: str = "sha256_if_available",
) -> list[list[dict[str, Any]]]:
    if folds < 2:
        raise ValueError("folds must be at least 2")
    base = make_split_manifest(images, annotations, train=1.0, val=0.0, test=0.0, seed=seed, group_by=group_by)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in base:
        groups[str(row.get(group_by) or row["filename"])].append(row)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    manifests: list[list[dict[str, Any]]] = []
    for fold in range(folds):
        rows = []
        for index, key in enumerate(keys):
            split = "test" if index % folds == fold else "train"
            for row in groups[key]:
                new_row = dict(row)
                new_row["split"] = split
                rows.append(new_row)
        manifests.append(sorted(rows, key=lambda row: int(row["frame"])))
    return manifests
