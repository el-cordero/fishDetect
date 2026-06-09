# Experiment Plan

This project is set up for bounding-box object detection. The current dataset should not be used for segmentation metrics.

## Dataset Split

Default split:

```text
train: 70%
val:   15%
test:  15%
```

Splits are made at the image/hash group level. The same image hash cannot appear in more than one split. This matters because the merge collapses duplicate image references by SHA256.

For this dataset, class imbalance is expected. Do not use oversampling or weighted sampling by default. Start with the plain split, inspect failures, then add balancing strategies only if they improve validation and test behavior.

## Model Families

### YOLOv8

Configured models:

```text
yolov8n.pt
yolov8s.pt
yolov8m.pt
```

Use these as the first production candidates.

- `yolov8n`: fastest and smallest; useful for smoke tests, edge deployment checks, and early sanity runs.
- `yolov8s`: good first real baseline; usually a better accuracy/speed tradeoff than nano.
- `yolov8m`: larger and slower; useful if small fish and look-alike classes need more capacity.

### YOLOv11

Configured models:

```text
yolo11n.pt
yolo11s.pt
yolo11m.pt
```

Use these to compare against YOLOv8 under the same data split and image size.

- `yolo11n`: fast baseline.
- `yolo11s`: likely main comparison point against `yolov8s`.
- `yolo11m`: capacity check against `yolov8m`.

Availability depends on the installed Ultralytics version. If a weight name is unsupported, the runner records a skipped experiment rather than failing silently.

### RT-DETR

Configured model:

```text
rtdetr-l.pt
```

RT-DETR gives a transformer-style detector comparison through Ultralytics. It may be slower and heavier than YOLO. Include it when there is enough GPU time and when detection quality is more important than real-time speed.

### Torchvision Baselines

Configured models:

```text
fasterrcnn_resnet50_fpn
retinanet_resnet50_fpn
fcos_resnet50_fpn
```

These are reference baselines, not the first path for production training.

- Faster R-CNN: two-stage detector; often strong on moderate datasets, slower at inference.
- RetinaNet: one-stage detector with focal loss; useful for class imbalance comparisons.
- FCOS: anchor-free detector; useful as a different detection formulation.

The full Torchvision training loop is intentionally minimal in this repository. Use these after the YOLO baseline results are available.

## Metrics To Report

Primary metrics:

```text
mAP50
mAP50-95
precision
recall
F1
per-class AP
per-class precision/recall
```

Secondary checks:

```text
small / medium / large object performance
rare / medium / common class bins
false positives by class
false negatives by class
inference speed
model size
training time
```

Report test metrics only after selecting models based on training and validation behavior. Do not tune directly on the test set.

## Recommended Order

1. Prepare dataset and audit classes.
2. Run `yolov8n` for a fast end-to-end check.
3. Train `yolov8s` and `yolo11s` as the main comparison.
4. Add `yolov8m` and `yolo11m` if the small models underfit.
5. Add RT-DETR if higher capacity is needed.
6. Run visual review on false positives, false negatives, rare classes, and crowded images.
7. Only then decide whether class balancing or augmentation changes are justified.

## Segmentation

Not part of the current experiment plan. The dataset is box-only.

DIVE geometry fields are preserved as metadata for traceability. They are not mask labels.
