from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from fishdetect.dataset import ANNOTATION_FIELDS
from fishdetect.utils.files import ensure_dir, safe_link_or_copy, write_csv_dicts, write_json, write_text


def bbox_to_yolo(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[float, float, float, float]:
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive.")
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return cx, cy, bw, bh


def bbox_to_coco(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
    return x1, y1, x2 - x1, y2 - y1


def class_names_from_annotations(annotations: list[dict[str, Any]]) -> list[str]:
    return sorted({row["class_name"] for row in annotations})


def export_yolo_detection(
    prepared_root: str | Path,
    annotations: list[dict[str, Any]],
    images: list[dict[str, Any]],
    split_manifest: list[dict[str, Any]],
    class_names: list[str] | None = None,
    link_images: bool = True,
) -> Path:
    root = Path(prepared_root) / "yolo_det"
    class_names = class_names or class_names_from_annotations(annotations)
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    ann_by_image = _annotations_by_image(annotations)
    split_by_image = {str(row["image_id"]): row["split"] for row in split_manifest}
    image_by_id = {str(row["image_id"]): row for row in images}

    for split in ["train", "val", "test"]:
        ensure_dir(root / "images" / split)
        ensure_dir(root / "labels" / split)

    exported_images = []
    for split_row in split_manifest:
        image_id = str(split_row["image_id"])
        image = image_by_id.get(image_id, split_row)
        split = split_row["split"]
        src = Path(image["image_path"])
        dst = root / "images" / split / image["filename"]
        safe_link_or_copy(src, dst, link=link_images)
        label_path = root / "labels" / split / f"{Path(image['filename']).stem}.txt"
        lines = []
        for ann in ann_by_image.get(image_id, []):
            if ann["class_name"] not in class_to_id:
                continue
            cx, cy, bw, bh = bbox_to_yolo(
                float(ann["bbox_x1"]),
                float(ann["bbox_y1"]),
                float(ann["bbox_x2"]),
                float(ann["bbox_y2"]),
                int(ann["width"]),
                int(ann["height"]),
            )
            lines.append(f"{class_to_id[ann['class_name']]} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")
        write_text(label_path, "\n".join(lines) + ("\n" if lines else ""))
        exported_images.append({"image_id": image_id, "split": split, "filename": image["filename"]})

    names_block = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(class_names))
    yaml_text = (
        f"path: {root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(class_names)}\n"
        "names:\n"
        f"{names_block}\n"
    )
    write_text(root / "data.yaml", yaml_text)
    write_csv_dicts(root / "export_manifest.csv", exported_images, ["image_id", "split", "filename"])
    return root


def export_coco_detection(
    prepared_root: str | Path,
    annotations: list[dict[str, Any]],
    images: list[dict[str, Any]],
    split_manifest: list[dict[str, Any]],
    class_names: list[str] | None = None,
    link_images: bool = True,
) -> Path:
    root = Path(prepared_root) / "coco_det"
    image_dir = ensure_dir(root / "images")
    class_names = class_names or class_names_from_annotations(annotations)
    class_to_id = {name: idx + 1 for idx, name in enumerate(class_names)}
    categories = [{"id": class_to_id[name], "name": name, "supercategory": "fish"} for name in class_names]
    ann_by_image = _annotations_by_image(annotations)
    image_by_id = {str(row["image_id"]): row for row in images}

    for split in ["train", "val", "test"]:
        coco_images = []
        coco_annotations = []
        ann_id = 1
        for split_row in split_manifest:
            if split_row["split"] != split:
                continue
            image_id = str(split_row["image_id"])
            image = image_by_id.get(image_id, split_row)
            safe_link_or_copy(image["image_path"], image_dir / image["filename"], link=link_images)
            coco_image_id = int(image["image_id"])
            coco_images.append(
                {
                    "id": coco_image_id,
                    "file_name": image["filename"],
                    "width": int(image["width"]),
                    "height": int(image["height"]),
                }
            )
            for ann in ann_by_image.get(image_id, []):
                x, y, w, h = bbox_to_coco(
                    float(ann["bbox_x1"]),
                    float(ann["bbox_y1"]),
                    float(ann["bbox_x2"]),
                    float(ann["bbox_y2"]),
                )
                coco_ann = {
                    "id": ann_id,
                    "image_id": coco_image_id,
                    "category_id": class_to_id[ann["class_name"]],
                    "bbox": [x, y, w, h],
                    "area": max(w, 0.0) * max(h, 0.0),
                    "iscrowd": 0,
                }
                coco_annotations.append(coco_ann)
                ann_id += 1
        write_json(root / f"{split}.json", {"images": coco_images, "annotations": coco_annotations, "categories": categories})
    return root


def export_annotation_csv(
    prepared_root: str | Path,
    annotations: list[dict[str, Any]],
) -> Path:
    path = Path(prepared_root) / "annotations_common.csv"
    write_csv_dicts(path, annotations, ANNOTATION_FIELDS)
    return path


def _annotations_by_image(annotations: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ann in annotations:
        grouped[str(ann["image_id"])].append(ann)
    return grouped
