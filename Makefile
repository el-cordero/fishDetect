PYTHON ?= python
CONFIG ?= configs/experiments.yaml
FISHDETECT_DATASET_ROOT ?= data/input
FISHDETECT_PREPARED_ROOT ?= data/prepared
export FISHDETECT_DATASET_ROOT
export FISHDETECT_PREPARED_ROOT

.PHONY: setup audit prepare train-yolo train-baselines evaluate compare smoke-test test

setup:
	$(PYTHON) -m pip install -e ".[training,dev]"

audit:
	$(PYTHON) scripts/audit_classes.py --dataset $(FISHDETECT_DATASET_ROOT) --out outputs/class_audit

prepare:
	$(PYTHON) scripts/prepare_dataset.py --config $(CONFIG)

train-yolo:
	$(PYTHON) scripts/train_yolo.py --config $(CONFIG) --experiment yolo8n_det

train-baselines:
	$(PYTHON) scripts/train_torchvision_detector.py --config $(CONFIG) --experiment fasterrcnn_resnet50

evaluate:
	$(PYTHON) scripts/evaluate_models.py --all

compare:
	$(PYTHON) scripts/compare_experiments.py

smoke-test:
	$(PYTHON) scripts/run_all_experiments.py --smoke-test --max-images 50 --epochs 1

test:
	pytest
