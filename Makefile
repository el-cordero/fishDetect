PYTHON ?= python
CONFIG ?= configs/experiments.yaml
SMOKE_CONFIG ?= configs/macbook_smoke.yaml
FISHDETECT_DATASET_ROOT ?= data/input
FISHDETECT_PREPARED_ROOT ?= data/prepared
ALLOW_DOWNLOADS ?=
export FISHDETECT_DATASET_ROOT
export FISHDETECT_PREPARED_ROOT

.PHONY: setup check-local audit prepare visualize-gt smoke-mac train-yolo train-first-pass train-main-comparison evaluate visualize-predictions compare test

setup:
	$(PYTHON) -m pip install --no-build-isolation -e ".[training,dev]"

check-local:
	$(PYTHON) scripts/check_local_setup.py --config $(CONFIG)

audit:
	$(PYTHON) scripts/audit_dataset.py --config $(CONFIG) --out outputs/dataset_audit
	$(PYTHON) scripts/audit_classes.py --dataset $(FISHDETECT_DATASET_ROOT) --out outputs/class_audit

prepare:
	$(PYTHON) scripts/prepare_dataset.py --config $(CONFIG)

visualize-gt:
	$(PYTHON) scripts/visualize_ground_truth.py --config $(CONFIG) --mode random --n 50
	$(PYTHON) scripts/visualize_ground_truth.py --config $(CONFIG) --mode rare --n 50
	$(PYTHON) scripts/visualize_ground_truth.py --config $(CONFIG) --mode crowded --n 50

smoke-mac:
	$(PYTHON) scripts/run_all_experiments.py --config $(SMOKE_CONFIG) --group smoke --smoke-test --max-images 50 --epochs 1 $(ALLOW_DOWNLOADS)

train-yolo:
	$(PYTHON) scripts/train_yolo.py --config $(CONFIG) --experiment yolo8n_det $(ALLOW_DOWNLOADS)

train-first-pass:
	$(PYTHON) scripts/run_all_experiments.py --config $(CONFIG) --group first_pass --run-full $(ALLOW_DOWNLOADS)

train-main-comparison:
	$(PYTHON) scripts/run_all_experiments.py --config $(CONFIG) --group main_comparison --run-full $(ALLOW_DOWNLOADS)

evaluate:
	$(PYTHON) scripts/evaluate_models.py --config $(CONFIG) --all --split test

visualize-predictions:
	$(PYTHON) scripts/visualize_predictions.py --best --split test --n 100 --mode mixed

compare:
	$(PYTHON) scripts/compare_experiments.py

test:
	pytest
