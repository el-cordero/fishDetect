from __future__ import annotations

from pathlib import Path
from typing import Any

from fishdetect.experiment import skipped_metrics
from fishdetect.utils.files import read_json


def segmentation_status(prepared_root: str | Path) -> dict[str, Any]:
    path = Path(prepared_root) / "segmentation_optional" / "segmentation_status.json"
    if not path.exists():
        return {"true_segmentation_supported": False, "real_polygon_annotations": 0, "dive_geometry_annotation_count": 0}
    return read_json(path)


def train_segmentation_model(
    config: dict[str, Any],
    experiment: dict[str, Any],
    output_dir: str | Path,
    smoke_test: bool = False,
    allow_weak_box_masks: bool = False,
) -> dict[str, Any]:
    status = segmentation_status(config["dataset"]["prepared_root"])
    dive_geometry_count = int(status.get("dive_geometry_annotation_count", 0) or 0)
    if experiment.get("requires_real_polygons", False):
        return skipped_metrics("Dataset is box-only; no validated segmentation masks are available.", experiment, smoke_test)
    if allow_weak_box_masks:
        return skipped_metrics(
            "Weak box masks are disabled by default and must be handled as a separate box baseline, not segmentation.",
            experiment,
            smoke_test,
        )
    return skipped_metrics(
        f"Dataset is box-only. Preserved DIVE geometry records: {dive_geometry_count}; trained masks: none.",
        experiment,
        smoke_test,
    )
