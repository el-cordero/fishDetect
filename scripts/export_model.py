#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fishdetect.utils.files import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the best available model weights from an experiment.")
    parser.add_argument("--experiment", required=True, help="Experiment directory.")
    parser.add_argument("--out", default=None, help="Export directory. Defaults to experiment/exported_model.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exp_dir = Path(args.experiment)
    if not exp_dir.exists():
        exp_dir = Path("outputs") / "experiments" / args.experiment
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment not found: {args.experiment}")
    out = ensure_dir(args.out or exp_dir / "exported_model")
    candidates = sorted(exp_dir.rglob("best.pt")) + sorted(exp_dir.rglob("last.pt")) + sorted((exp_dir / "weights").glob("*"))
    copied = []
    for src in candidates[:1]:
        if src.is_file():
            dst = out / src.name
            shutil.copy2(src, dst)
            copied.append(str(dst))
    write_json(out / "export_summary.json", {"experiment": str(exp_dir), "copied": copied})
    print(f"Exported {len(copied)} file(s) to {out}.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
