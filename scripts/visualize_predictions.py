#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fishdetect.config import load_config, load_yaml
from fishdetect.evaluation.detection_metrics import bbox_iou
from fishdetect.utils.files import ensure_dir, read_csv_dicts, write_text


GT_COLOR = (39, 174, 96)
PRED_COLOR = (231, 76, 60)
TEXT_COLOR = (255, 255, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create visual galleries of ground truth and predicted fish boxes.")
    parser.add_argument("--experiment", default=None, help="Experiment directory or name.")
    parser.add_argument("--best", action="store_true", help="Use best experiment from outputs/comparison/best_model_summary.json.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--mode", default="mixed", choices=["mixed", "random", "high_confidence", "false_positives", "false_negatives", "low_confidence", "crowded", "rare"])
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--prepared-root", default=None)
    parser.add_argument("--seed", type=int, default=23401)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exp_dir = _resolve_experiment(args)
    prepared_from_experiment = _prepared_root_from_experiment(exp_dir)
    if args.prepared_root or prepared_from_experiment:
        prepared_root = Path(args.prepared_root or prepared_from_experiment)
    else:
        config = load_config(args.config)
        prepared_root = Path(config["dataset"]["prepared_root"])
    annotations_path = prepared_root / "annotations_common.csv"
    split_path = prepared_root / "metadata" / "split_manifest.csv"
    if not annotations_path.exists() or not split_path.exists():
        raise FileNotFoundError(f"Prepared annotations/splits not found under {prepared_root}. Run prepare_dataset first.")
    annotations = read_csv_dicts(annotations_path)
    split_rows = [row for row in read_csv_dicts(split_path) if row["split"] == args.split]
    predictions = _load_predictions(exp_dir)
    review_ids = _review_image_ids(exp_dir, args.mode)

    gallery_root = ensure_dir(exp_dir / "predictions" / "galleries" / f"{args.split}_{args.mode}")
    selected = _select_images(split_rows, annotations, predictions, args.mode, args.n, args.seed, review_ids)
    rendered = _render_gallery(gallery_root, selected, annotations, predictions)
    _write_html(gallery_root / "index.html", rendered, predictions_available=bool(predictions))
    print(f"Wrote {len(rendered)} rendered review image(s) to {gallery_root}.")
    return 0


def _resolve_experiment(args: argparse.Namespace) -> Path:
    if args.best:
        best_path = Path("outputs/comparison/best_model_summary.json")
        if not best_path.exists():
            raise FileNotFoundError("Best model summary not found. Run compare_experiments.py first.")
        with best_path.open("r", encoding="utf-8") as f:
            best = json.load(f)
        name = best.get("experiment")
        if not name:
            raise ValueError("Best model summary does not contain an experiment.")
        return Path("outputs") / "experiments" / name
    if not args.experiment:
        raise ValueError("Use --experiment or --best.")
    path = Path(args.experiment)
    return path if path.exists() else Path("outputs") / "experiments" / args.experiment


def _prepared_root_from_experiment(exp_dir: Path) -> str | None:
    config_path = exp_dir / "config_used.yaml"
    if not config_path.exists():
        return None
    try:
        return load_yaml(config_path)["dataset"]["prepared_root"]
    except Exception:
        return None


def _load_predictions(exp_dir: Path) -> list[dict]:
    for path in [exp_dir / "predictions" / "predictions.json", exp_dir / "predictions.json"]:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data.get("predictions", [])
            if isinstance(data, list):
                return data
    return []


def _review_image_ids(exp_dir: Path, mode: str) -> set[str]:
    if mode not in {"false_positives", "false_negatives"}:
        return set()
    table = exp_dir / "predictions" / f"{mode}.csv"
    if not table.exists():
        return set()
    import csv

    with table.open("r", encoding="utf-8", newline="") as f:
        return {str(row["image_id"]) for row in csv.DictReader(f) if row.get("image_id")}


def _select_images(split_rows, annotations, predictions, mode, n, seed, review_ids=None):
    ann_by_image = defaultdict(list)
    for ann in annotations:
        ann_by_image[str(ann["image_id"])].append(ann)
    pred_by_image = defaultdict(list)
    for pred in predictions:
        pred_by_image[str(pred["image_id"])].append(pred)
    rows = list(split_rows)
    if review_ids:
        rows = [row for row in rows if str(row["image_id"]) in review_ids]
    if mode == "crowded":
        rows.sort(key=lambda row: len(ann_by_image[str(row["image_id"])]), reverse=True)
    elif mode == "rare":
        class_counts = defaultdict(int)
        for ann in annotations:
            class_counts[ann["class_name"]] += 1
        rows.sort(key=lambda row: min([class_counts[a["class_name"]] for a in ann_by_image[str(row["image_id"])]] or [999999]))
    elif mode in {"high_confidence", "low_confidence"} and predictions:
        reverse = mode == "high_confidence"
        rows.sort(key=lambda row: max([float(p.get("score", 0)) for p in pred_by_image[str(row["image_id"])]] or [0]), reverse=reverse)
    else:
        rng = random.Random(seed)
        rng.shuffle(rows)
    return rows[:n]


def _render_gallery(gallery_root: Path, selected, annotations, predictions):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        write_text(gallery_root / "README.txt", f"Pillow is required for rendering galleries: {exc}\n")
        return []
    ann_by_image = defaultdict(list)
    for ann in annotations:
        ann_by_image[str(ann["image_id"])].append(ann)
    pred_by_image = defaultdict(list)
    for pred in predictions:
        pred_by_image[str(pred["image_id"])].append(pred)
    rendered = []
    for row in selected:
        image_id = str(row["image_id"])
        src = Path(row["image_path"])
        if not src.exists():
            continue
        with Image.open(src) as im:
            im = im.convert("RGB")
            draw = ImageDraw.Draw(im)
            for ann in ann_by_image.get(image_id, []):
                box = _box(ann)
                draw.rectangle(box, outline=GT_COLOR, width=4)
                _label(draw, box[0], box[1], f"GT {ann['class_name']}", GT_COLOR)
            for pred in pred_by_image.get(image_id, []):
                box = _box(pred)
                best_iou = max([bbox_iou(box, _box(gt)) for gt in ann_by_image.get(image_id, [])] or [0.0])
                status = "match" if best_iou >= 0.5 else "unmatched"
                label = f"P {pred.get('class_name', '')} {float(pred.get('score', 0)):.2f} IoU {best_iou:.2f} {status}"
                draw.rectangle(box, outline=PRED_COLOR, width=3)
                _label(draw, box[0], max(0, box[1] - 18), label, PRED_COLOR)
            dst = gallery_root / f"{Path(row['filename']).stem}_review.jpg"
            im.thumbnail((1600, 1600))
            im.save(dst, quality=92)
            rendered.append({"filename": dst.name, "source": row["filename"], "gt": len(ann_by_image.get(image_id, [])), "pred": len(pred_by_image.get(image_id, []))})
    return rendered


def _box(row):
    if "bbox" in row:
        return tuple(float(v) for v in row["bbox"])
    return (
        float(row["bbox_x1"]),
        float(row["bbox_y1"]),
        float(row["bbox_x2"]),
        float(row["bbox_y2"]),
    )


def _label(draw, x, y, text, color):
    padding = 3
    bbox = draw.textbbox((x, y), text)
    draw.rectangle((bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding), fill=color)
    draw.text((x, y), text, fill=TEXT_COLOR)


def _write_html(path: Path, rendered, predictions_available: bool):
    lines = [
        "<html><body>",
        "<h1>FishDetect Prediction Review</h1>",
        f"<p>Predictions available: {predictions_available}</p>",
        "<p>Green boxes are ground truth. Red boxes are predictions.</p>",
    ]
    for item in rendered:
        lines.append(f"<h3>{item['source']} | GT {item['gt']} | Pred {item['pred']}</h3>")
        lines.append(f"<img src='{item['filename']}' style='max-width: 100%; height: auto;'>")
    lines.append("</body></html>")
    write_text(path, "\n".join(lines) + "\n")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
