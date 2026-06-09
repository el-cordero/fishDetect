#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fishdetect.audit import audit_dataset
from fishdetect.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a FishDetect VIAME/DIVE bounding-box dataset.")
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--dataset", default=None, help="Dataset root. Overrides config.")
    parser.add_argument("--out", default="outputs/dataset_audit")
    parser.add_argument("--rare-threshold", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dataset:
        dataset_root = args.dataset
    else:
        config = load_config(args.config)
        dataset_root = config["dataset"]["root"]
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {root}. Set FISHDETECT_DATASET_ROOT or pass --dataset."
        )
    summary = audit_dataset(root, args.out, rare_threshold=args.rare_threshold)
    print(f"Dataset audit written to {args.out}")
    print(f"Images: {summary['image_count']}")
    print(f"Annotations: {summary['annotation_count']}")
    print(f"Classes: {summary['class_count']}")
    print("Annotation type: bounding boxes only")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
