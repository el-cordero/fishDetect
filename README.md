# FishDetect

Reproducible machine learning workflow for Caribbean fish detection using cleaned VIAME/DIVE bounding-box annotations.

This repository is designed for local validation first, then model training and client-facing reporting. It does not contain the image dataset, prepared training exports, prediction galleries, or model weights.

## Scope

This project supports:

- local environment checks
- dataset audit and class-name review
- train/validation/test split creation with duplicate-image leakage prevention
- YOLO detection export
- COCO detection export
- small MacBook smoke tests
- opt-in YOLO model training
- test-set prediction export
- detection metrics
- visual prediction review
- model comparison reports

This project does **not** support segmentation for the current dataset. The annotations are bounding boxes only. DIVE geometry fields, when present, are retained as metadata and are not treated as semantic masks, instance masks, pseudo-masks, or Mask R-CNN inputs.

## Dataset Assumptions

Expected cleaned dataset files:

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

Set paths with environment variables:

```bash
export FISHDETECT_DATASET_ROOT=/path/to/cleaned_merged_viame_dive_dataset
export FISHDETECT_PREPARED_ROOT=/path/to/prepared_outputs
```

The preparation scripts never modify the dataset root. Prepared outputs are written to `FISHDETECT_PREPARED_ROOT`.

## First Run On A MacBook

```bash
git clone https://github.com/el-cordero/fishDetect.git
cd fishDetect

python -m venv .venv
source .venv/bin/activate

make setup
make check-local
make audit
make prepare
make smoke-mac
make test
```

Use `configs/macbook_smoke.yaml` for the first local test. It uses a small subset, 640 px images, one epoch, conservative workers, and automatic CPU/MPS/CUDA detection. If MPS is unavailable or unstable, the workflow can fall back to CPU.

## Notebook Order

Run these notebooks in order for a step-by-step workflow:

1. `00_local_setup_check.ipynb`
2. `01_dataset_audit.ipynb`
3. `02_prepare_detection_dataset.ipynb`
4. `03_visualize_ground_truth.ipynb`
5. `04_yolo_smoke_test_macbook.ipynb`
6. `05_train_yolo_models.ipynb`
7. `06_evaluate_models.ipynb`
8. `07_visual_review_predictions.ipynb`
9. `08_model_comparison_report.ipynb`

The first five notebooks do not require full model training.

## Configuration

Main configs:

```text
configs/local.example.yaml
configs/macbook_smoke.yaml
configs/experiments.yaml
configs/yolo_baselines.yaml
configs/class_aliases.yaml
```

`configs/experiments.yaml` uses environment variables for cross-computer reproducibility. `configs/macbook_smoke.yaml` contains local MacBook defaults and can be copied or edited for a first workstation run.

## Model Groups

Full model comparisons are opt-in. The default group is `smoke`.

```bash
python scripts/run_all_experiments.py --group smoke
python scripts/run_all_experiments.py --group first_pass --run-full
python scripts/run_all_experiments.py --group main_comparison --run-full
python scripts/run_all_experiments.py --group capacity_check --run-full
python scripts/run_all_experiments.py --group extended --run-full
```

Recommended order:

1. Run `yolov8n` smoke test first.
2. Run first-pass nano/small models.
3. Run small-model comparison across YOLOv8, YOLOv9, YOLOv10, YOLO11, and YOLOv12.
4. Run medium models only if results justify the compute.
5. Keep RT-DETR as an optional high-capacity comparison.

YOLOv9, YOLOv10, YOLO11, YOLOv12, and RT-DETR support depends on the installed Ultralytics version. Unsupported or unavailable models are skipped with the reason recorded in `metrics.json`.

## Common Commands

```bash
make check-local
make audit
make prepare
make visualize-gt
make smoke-mac
make train-first-pass
make evaluate
make visualize-predictions
make compare
make test
```

Train one model:

```bash
python scripts/train_yolo.py --experiment yolo8s_det --config configs/experiments.yaml
```

If model weights are not already local, add `--allow-downloads` intentionally:

```bash
python scripts/train_yolo.py --experiment yolo8s_det --allow-downloads
```

Evaluate all experiment directories:

```bash
python scripts/evaluate_models.py --all --split test
```

## Outputs

Prepared data:

```text
prepared_outputs/
  metadata/
  yolo_det/
  coco_det/
```

Experiment outputs:

```text
outputs/experiments/<experiment_name>/
  config_used.yaml
  environment.json
  train_log.csv
  metrics.json
  per_class_metrics.csv
  predictions/
    predictions.json
    false_positives.csv
    false_negatives.csv
    galleries/
  weights/
  model_card.md
```

Comparison outputs:

```text
outputs/comparison/
  model_comparison.csv
  model_comparison.md
  model_comparison.html
  best_model_summary.json
  plots/
```

Generated outputs are ignored by git.

## Class-Name Review

Class aliases are audit-only unless explicitly applied in downstream code.

```bash
python scripts/audit_classes.py --dataset "$FISHDETECT_DATASET_ROOT" --out outputs/class_audit
```

Known review flags include likely spelling or taxonomy issues such as `Acantharus`, `Archarhinus`, `bivitatus`, `Lactophyrys`, and `Sparisoma viridae`. Juvenile and phase labels should remain separate unless a domain review says otherwise.

## Troubleshooting

- `Dataset root not found`: set `FISHDETECT_DATASET_ROOT`.
- `Prepared dataset not found`: run `make prepare`.
- `Ultralytics is not installed`: run `make setup` inside the active environment.
- `Skipped remote weight`: pass `--allow-downloads` only when you intentionally want Ultralytics to fetch weights.
- MPS issues on Apple Silicon: rerun with `--device cpu`.
- Empty prediction galleries: run evaluation first so `predictions.json` and FP/FN tables exist.

## Public Repository Notes

Do not commit raw data, prepared data, model weights, prediction images, generated outputs, or local filesystem paths. See [docs/DATA_PRIVACY.md](docs/DATA_PRIVACY.md) before publishing release updates.
