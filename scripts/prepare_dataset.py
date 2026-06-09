#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fishdetect.config import load_config
from fishdetect.pipeline import prepare_dataset_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and export the FishDetect dataset to YOLO and COCO formats.")
    parser.add_argument("--config", default="configs/experiments.yaml", help="Experiment YAML config.")
    parser.add_argument("--prepared-root", default=None, help="Override prepared output root.")
    parser.add_argument("--reuse-split", default=None, help="Reuse an existing split manifest CSV.")
    parser.add_argument("--max-images", type=int, default=None, help="Prepare only a tiny annotated subset, useful for smoke tests.")
    parser.add_argument("--weak-box-masks", action="store_true", help="Enable weak box-mask metadata. Does not mark boxes as true masks.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    summary = prepare_dataset_pipeline(
        config,
        max_images=args.max_images,
        prepared_root_override=args.prepared_root,
        reuse_split=args.reuse_split,
        allow_weak_box_masks=args.weak_box_masks,
    )
    print("Prepared dataset:")
    print(f"  root: {summary['prepared_root']}")
    print(f"  exported images: {summary['image_count_exported']}")
    print(f"  exported annotations: {summary['annotation_count_exported']}")
    print(f"  DIVE geometry records preserved as metadata: {summary['dive_geometry_annotation_count_exported']}")
    print("  segmentation masks: none")
    print(f"  split manifest: {summary['split_manifest']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
