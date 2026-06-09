#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fishdetect_mplconfig"))

from fishdetect.config import ConfigError, load_config
from fishdetect.models.yolo import _resolve_batch
from fishdetect.utils.files import ensure_dir, write_json
from fishdetect.utils.seed import detect_device, package_versions

DEFAULT_DATASET_ROOT = "/Users/ec/Documents/Data/FishDetectNOAA/_data/merged_viame_v2"
DEFAULT_PREPARED_ROOT = "/Users/ec/Documents/Data/FishDetectNOAA/_data/prepared_ml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check local FishDetect setup before running data preparation or training.")
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--out", default="outputs/local_setup")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("FISHDETECT_DATASET_ROOT", DEFAULT_DATASET_ROOT)
    os.environ.setdefault("FISHDETECT_PREPARED_ROOT", DEFAULT_PREPARED_ROOT)

    out = ensure_dir(args.out)
    checks = []
    ready = True

    try:
        config = load_config(args.config)
        checks.append(_check("config", True, args.config))
    except Exception as exc:
        config = None
        ready = False
        checks.append(_check("config", False, f"{exc}"))

    repo_root = Path(__file__).resolve().parents[1]
    checks.append(_check("repo_root", repo_root.exists(), str(repo_root)))
    checks.append(_check("python", sys.version_info >= (3, 9), platform.python_version()))

    for package in ["numpy", "pandas", "PIL", "matplotlib", "sklearn"]:
        checks.append(_import_check(package))
    checks.append(_import_check("torch"))
    checks.append(_import_check("ultralytics"))

    device = detect_device("auto")
    checks.append(_check("device_auto", True, device))

    if config:
        dataset_root = Path(config["dataset"]["root"])
        prepared_root = Path(config["dataset"]["prepared_root"])
        dataset_ok = dataset_root.exists()
        checks.append(_check("dataset_root", dataset_ok, str(dataset_root)))
        if not dataset_ok:
            ready = False
        try:
            ensure_dir(prepared_root)
            checks.append(_check("prepared_root", True, str(prepared_root)))
            usage = shutil.disk_usage(prepared_root)
            checks.append(_check("prepared_root_free_gb", usage.free > 5 * 1024**3, f"{usage.free / 1024**3:.1f} GB free"))
        except Exception as exc:
            ready = False
            checks.append(_check("prepared_root", False, str(exc)))

        training = config.get("training", {})
        try:
            resolved_batch = _resolve_batch(training.get("batch", 4), smoke_test=bool(training.get("smoke_test", False)))
            checks.append(_check("training_batch", True, str(resolved_batch)))
        except ValueError as exc:
            ready = False
            checks.append(_check("training_batch", False, str(exc)))
        macbook_notes = {
            "imgsz": training.get("imgsz"),
            "epochs": training.get("epochs"),
            "batch": training.get("batch"),
            "workers": training.get("workers"),
            "device": training.get("device"),
            "smoke_test": training.get("smoke_test", False),
            "max_images": training.get("max_images"),
            "old_script_patterns_reflected": "small smoke config, fixed seed, conservative workers, CPU/MPS/CUDA fallback",
        }
    else:
        macbook_notes = {}

    smoke_help = shutil.which("python") is not None and (repo_root / "scripts" / "run_all_experiments.py").exists()
    checks.append(_check("smoke_command_available", smoke_help, "python scripts/run_all_experiments.py --group smoke --smoke-test"))

    summary = {
        "ready": ready and all(item["ok"] for item in checks if item["name"] not in {"ultralytics"}),
        "checks": checks,
        "device_auto": device,
        "versions": package_versions(),
        "macbook_smoke_assumptions": macbook_notes,
    }
    write_json(out / "local_setup_summary.json", summary)
    for item in checks:
        status = "OK" if item["ok"] else "MISSING"
        print(f"{status:8} {item['name']}: {item['detail']}")
    print(f"\nReady: {'yes' if summary['ready'] else 'no'}")
    if not summary["ready"]:
        print("Set FISHDETECT_DATASET_ROOT and FISHDETECT_PREPARED_ROOT, then rerun this check.")
    return 0 if summary["ready"] else 1


def _check(name: str, ok: bool, detail: str) -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail}


def _import_check(package: str) -> dict:
    try:
        module = importlib.import_module(package)
        return _check(package, True, getattr(module, "__version__", "installed"))
    except Exception as exc:
        return _check(package, False, str(exc))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
