from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fishdetect.utils.files import read_json


def load_meta_image_map(path: str | Path) -> dict[int, str]:
    data = read_json(path)
    image_data = data.get("imageData", [])
    mapping: dict[int, str] = {}
    for index, item in enumerate(image_data):
        filename = item.get("filename")
        if filename:
            mapping[index] = filename
    return mapping


def parse_dive_json(
    path: str | Path,
    meta_path: str | Path | None = None,
    dataset_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DIVE JSON not found: {path}")
    root = Path(dataset_root) if dataset_root else path.parent
    frame_to_filename = load_meta_image_map(meta_path) if meta_path else {}
    data = read_json(path)
    tracks = data.get("tracks", {})
    rows: list[dict[str, Any]] = []
    for track_key, track in tracks.items():
        class_name, class_conf = _best_confidence_pair(track.get("confidencePairs", []))
        track_id = str(track.get("id", track_key))
        for feature_index, feature in enumerate(track.get("features", [])):
            if "bounds" not in feature:
                continue
            bounds = feature["bounds"]
            if len(bounds) < 4:
                raise ValueError(f"DIVE track {track_id} feature {feature_index} has invalid bounds.")
            frame = int(feature.get("frame", track.get("begin", 0)))
            filename = frame_to_filename.get(frame, "")
            geometry = feature.get("geometry")
            dive_geometry = _extract_polygon_geometry(geometry)
            x1, y1, x2, y2 = [float(v) for v in bounds[:4]]
            rows.append(
                {
                    "track_id": track_id,
                    "filename": filename,
                    "frame": frame,
                    "image_path": str(root / filename) if filename else "",
                    "bbox_x1": min(x1, x2),
                    "bbox_y1": min(y1, y2),
                    "bbox_x2": max(x1, x2),
                    "bbox_y2": max(y1, y2),
                    "class_name": class_name,
                    "class_confidence": class_conf,
                    "has_dive_geometry": dive_geometry is not None,
                    "dive_geometry_if_available": json.dumps(dive_geometry) if dive_geometry else "",
                    "has_polygon": False,
                    "polygon_geometry_if_available": "",
                    "source_format": "dive",
                    "feature_index": feature_index,
                }
            )
    return rows


def _best_confidence_pair(pairs: list[Any]) -> tuple[str, float | None]:
    best_name = ""
    best_conf: float | None = None
    for pair in pairs or []:
        if not isinstance(pair, list) or len(pair) < 2:
            continue
        name = str(pair[0]).strip()
        try:
            conf = float(pair[1])
        except Exception:
            conf = None
        if best_conf is None or (conf is not None and conf > best_conf):
            best_name = name
            best_conf = conf
    return best_name, best_conf


def _extract_polygon_geometry(geometry: Any) -> Any | None:
    if not geometry:
        return None
    if isinstance(geometry, dict) and geometry.get("type") == "Polygon":
        return geometry
    if isinstance(geometry, dict) and geometry.get("type") == "Feature":
        return _extract_polygon_geometry(geometry.get("geometry"))
    if isinstance(geometry, dict) and geometry.get("type") == "FeatureCollection":
        polygons = []
        for feature in geometry.get("features", []):
            polygon = _extract_polygon_geometry(feature)
            if polygon:
                polygons.append(polygon)
        if len(polygons) == 1:
            return polygons[0]
        if polygons:
            return {"type": "MultiPolygon", "polygons": polygons}
    if isinstance(geometry, dict) and geometry.get("type") == "MultiPolygon":
        return geometry
    return None
