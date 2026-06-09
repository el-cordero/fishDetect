#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fishdetect.config import load_config
from fishdetect.utils.files import ensure_dir, read_csv_dicts, write_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create ground-truth bounding-box review galleries.")
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--prepared-root", default=None)
    parser.add_argument("--out", default="outputs/ground_truth_review")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--mode", choices=["random", "rare", "crowded"], default="random")
    parser.add_argument("--seed", type=int, default=23401)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prepared_root = Path(args.prepared_root or load_config(args.config)["dataset"]["prepared_root"])
    annotations_path = prepared_root / "annotations_common.csv"
    split_path = prepared_root / "metadata" / "split_manifest.csv"
    if not annotations_path.exists() or not split_path.exists():
        raise FileNotFoundError(f"Prepared dataset not found under {prepared_root}. Run prepare_dataset.py first.")
    annotations = read_csv_dicts(annotations_path)
    images = read_csv_dicts(split_path)
    out = ensure_dir(Path(args.out) / args.mode)
    selected = _select(images, annotations, args.mode, args.n, args.seed)
    rendered = _render(out, selected, annotations)
    _write_html(out / "index.html", rendered)
    print(f"Wrote {len(rendered)} ground-truth review image(s) to {out}.")
    return 0


def _select(images, annotations, mode, n, seed):
    ann_by_image = defaultdict(list)
    for ann in annotations:
        ann_by_image[str(ann["image_id"])].append(ann)
    rows = list(images)
    if mode == "crowded":
        rows.sort(key=lambda row: len(ann_by_image[str(row["image_id"])]), reverse=True)
    elif mode == "rare":
        counts = defaultdict(int)
        for ann in annotations:
            counts[ann["class_name"]] += 1
        rows.sort(key=lambda row: min([counts[a["class_name"]] for a in ann_by_image[str(row["image_id"])]] or [999999]))
    else:
        random.Random(seed).shuffle(rows)
    return rows[:n]


def _render(out: Path, selected, annotations):
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        write_text(out / "README.txt", f"Pillow is required for image rendering: {exc}\n")
        return []
    ann_by_image = defaultdict(list)
    for ann in annotations:
        ann_by_image[str(ann["image_id"])].append(ann)
    rendered = []
    for row in selected:
        path = Path(row["image_path"])
        if not path.exists():
            continue
        with Image.open(path) as im:
            im = im.convert("RGB")
            draw = ImageDraw.Draw(im)
            for ann in ann_by_image[str(row["image_id"])]:
                box = tuple(float(ann[k]) for k in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"])
                draw.rectangle(box, outline=(39, 174, 96), width=4)
                draw.text((box[0], max(0, box[1] - 14)), ann["class_name"], fill=(39, 174, 96))
            im.thumbnail((1600, 1600))
            dst = out / f"{Path(row['filename']).stem}_gt.jpg"
            im.save(dst, quality=92)
            rendered.append({"file": dst.name, "source": row["filename"], "annotations": len(ann_by_image[str(row["image_id"])])})
    return rendered


def _write_html(path: Path, rendered):
    lines = ["<html><body><h1>Ground Truth Review</h1>", "<p>Green boxes are ground-truth annotations.</p>"]
    for item in rendered:
        lines.append(f"<h3>{item['source']} | boxes: {item['annotations']}</h3>")
        lines.append(f"<img src='{item['file']}' style='max-width:100%;height:auto;'>")
    lines.append("</body></html>")
    write_text(path, "\n".join(lines) + "\n")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
