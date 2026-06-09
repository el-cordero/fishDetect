from fishdetect.evaluation.detection_metrics import bbox_iou, evaluate_detections
from fishdetect.evaluation.segmentation_metrics import binary_iou, dice_score


def test_detection_metrics_on_toy_example():
    assert round(bbox_iou([0, 0, 10, 10], [5, 5, 15, 15]), 4) == 0.1429
    gt = [{"image_id": "1", "class_name": "fish", "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 10, "bbox_y2": 10}]
    pred = [{"image_id": "1", "class_name": "fish", "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 10, "bbox_y2": 10, "score": 0.9}]
    metrics = evaluate_detections(gt, pred, class_names=["fish"])
    assert metrics["precision"] == 1
    assert metrics["recall"] == 1
    assert metrics["mAP50"] > 0.9


def test_segmentation_metrics_on_toy_masks():
    true = [[1, 1], [0, 0]]
    pred = [[1, 0], [1, 0]]
    assert binary_iou(pred, true) == 1 / 3
    assert dice_score(pred, true) == 0.5
