# fishdetect-noaa-ml

Training and evaluation code for the NOAA Fish Detect image merge.

This dataset is treated as **bounding boxes only**. Some DIVE records contain geometry-looking fields, but for this merge they are not trusted as masks. COCO export writes detection boxes. YOLO export writes normalized box labels. Segmentation is off unless a future dataset has validated masks.

## Dataset

Input is configured with environment variables:

```bash
export FISHDETECT_DATASET_ROOT=/path/to/cleaned_merged_viame_dive_dataset
export FISHDETECT_PREPARED_ROOT=/path/to/prepared_outputs
```

Expected files:

```text
annotations.viame.csv
annotations.dive.json
meta.json
image_manifest.csv
class_counts.csv
source_dataset_manifest.csv
qc_report.txt
qc_summary.json
merge_summary.json
```

The scripts do not edit the dataset directory. Images are symlinked into the prepared output directory when possible.

## Setup

```bash
make setup
```

For quick checks, the repo works with normal Python data packages. Real training needs Ultralytics for YOLO or Torch/Torchvision for baselines.

## Usual Run

```bash
make audit
make prepare
make smoke-test

python scripts/train_yolo.py --experiment yolo8n_det
python scripts/train_yolo.py --experiment yolo8s_det
python scripts/evaluate_models.py --all
python scripts/compare_experiments.py
python scripts/visualize_predictions.py --best --split test --n 100
```

Smoke mode uses a small prepared subset and avoids remote YOLO weight downloads unless `--allow-downloads` is passed.

## Prepare Data

```bash
python scripts/prepare_dataset.py --config configs/experiments.yaml
```

Writes:

```text
prepared_outputs/
  metadata/
    annotations.csv
    images.csv
    split_manifest.csv
    validation_report.json
    prepare_summary.json
  annotations_common.csv
  yolo_det/
  coco_det/
  segmentation_optional/
```

Default split is 70/15/15 train/val/test with a fixed seed. Splitting groups by SHA256 so duplicate image references do not cross splits.

## Train

```bash
python scripts/train_yolo.py --experiment yolo8n_det --device auto
```

Configured detectors:

```text
yolov8n/s/m
yolo11n/s/m
rtdetr-l
fasterrcnn_resnet50_fpn
retinanet_resnet50_fpn
fcos_resnet50_fpn
```

Torchvision baselines are registered, but the full training loop is intentionally thin. YOLO is the main path to start with.

## Evaluate And Compare

```bash
python scripts/evaluate_models.py --all
python scripts/compare_experiments.py
```

Each run gets:

```text
outputs/experiments/<name>/
  config_used.yaml
  train_log.csv
  metrics.json
  per_class_metrics.csv
  confusion_matrix.png
  pr_curve.png
  f1_curve.png
  predictions/
  weights/
  model_card.md
```

Comparison files land in `outputs/comparison/`.

## Visual Checks

```bash
python scripts/visualize_predictions.py \
  --experiment outputs/experiments/yolo8s_det \
  --split test \
  --n 100 \
  --mode mixed
```

Green boxes are ground truth. Red boxes are predictions. Labels include class, confidence, and IoU status when prediction JSON is present.

Useful modes:

```text
mixed
random
high_confidence
low_confidence
crowded
rare
false_positives
false_negatives
```

## Class Names

```bash
python scripts/audit_classes.py \
  --dataset "$FISHDETECT_DATASET_ROOT" \
  --out outputs/class_audit
```

The audit only flags names. It does not rename anything.

Known checks include:

```text
Acantharus -> Acanthurus
Archarhinus -> Carcharhinus
bivitatus -> bivittatus
Lactophyrys -> Lactophrys
Sparisoma viridae -> Sparisoma viride
```

Keep juvenile and phase labels separate unless a taxonomy review says otherwise.

## Segmentation

Off for this merge.

DIVE geometry is kept as metadata, but it is not exported as mask annotations and is not used for mask metrics. Box masks, if someone adds them later, should stay in a separate weak baseline and should not be reported as segmentation.

## More Documentation

- [Experiment plan](docs/EXPERIMENTS.md)
- [Results template](docs/RESULTS.md)
- [Data and privacy notes](docs/DATA_PRIVACY.md)

## Field Notes

Expect small fish, haze, reef occlusion, color shifts, class imbalance, and look-alike species. Start with conservative underwater augmentations: brightness/contrast/color jitter, mild blur, light noise/haze, and scale changes. Vertical flips are off by default.
