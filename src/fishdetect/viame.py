from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


VIAME_FIELDS = [
    "track_id",
    "filename",
    "frame",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "confidence",
    "target_length",
    "class_name",
    "class_confidence",
]


def parse_viame_csv(path: str | Path, dataset_root: str | Path | None = None) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"VIAME CSV not found: {path}")
    root = Path(dataset_root) if dataset_root else path.parent
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for line_number, raw in enumerate(reader, start=1):
            if not raw or raw[0].startswith("#"):
                continue
            if len(raw) < 10:
                raise ValueError(f"VIAME row {line_number} has fewer than 10 columns.")
            try:
                track_id = str(raw[0])
                filename = raw[1]
                frame = int(float(raw[2]))
                x1, y1, x2, y2 = [float(raw[i]) for i in range(3, 7)]
                confidence = float(raw[7])
                target_length = float(raw[8])
                class_name = raw[9].strip()
                class_confidence = float(raw[10]) if len(raw) > 10 and _is_float(raw[10]) else None
            except Exception as exc:
                raise ValueError(f"Could not parse VIAME row {line_number}: {raw}") from exc
            rows.append(
                {
                    "track_id": track_id,
                    "filename": filename,
                    "frame": frame,
                    "image_path": str(root / filename),
                    "bbox_x1": min(x1, x2),
                    "bbox_y1": min(y1, y2),
                    "bbox_x2": max(x1, x2),
                    "bbox_y2": max(y1, y2),
                    "confidence": confidence,
                    "target_length": target_length,
                    "class_name": class_name,
                    "class_confidence": class_confidence,
                    "source_format": "viame",
                    "line_number": line_number,
                }
            )
    return rows


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False
