import json

from fishdetect.converters import bbox_to_coco, bbox_to_yolo, export_coco_detection, export_yolo_detection
from fishdetect.dataset import build_annotation_table, validate_dataset
from fishdetect.splits import make_split_manifest


def test_dataset_table_and_exports(synthetic_dataset, tmp_path):
    annotations, images, meta = build_annotation_table(synthetic_dataset)
    validation = validate_dataset(synthetic_dataset, annotations, images)
    assert validation["passed"] is True
    assert meta["dive_geometry_annotation_count"] == 1
    assert len(images) == 4
    assert len(annotations) == 3

    assert bbox_to_yolo(10, 20, 50, 60, 100, 100) == (0.3, 0.4, 0.4, 0.4)
    assert bbox_to_coco(10, 20, 50, 60) == (10, 20, 40, 40)

    split = make_split_manifest(images, annotations, seed=7, stratify=False)
    yolo_root = export_yolo_detection(tmp_path, annotations, images, split)
    coco_root = export_coco_detection(tmp_path, annotations, images, split)
    assert (yolo_root / "data.yaml").exists()
    with (coco_root / "train.json").open("r", encoding="utf-8") as f:
        coco = json.load(f)
    assert "categories" in coco
    for ann in coco["annotations"]:
        assert "segmentation" not in ann
