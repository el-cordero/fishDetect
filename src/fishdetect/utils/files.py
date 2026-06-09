from __future__ import annotations

import csv
import json
import os
import shutil
import struct
from pathlib import Path
from typing import Any, Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, sort_keys=False)
        f.write("\n")


def read_csv_dicts(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_dicts(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def list_images(path: str | Path) -> list[Path]:
    root = Path(path)
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def safe_link_or_copy(src: str | Path, dst: str | Path, link: bool = True) -> None:
    src = Path(src)
    dst = Path(dst)
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        return
    if link:
        try:
            os.symlink(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def image_size(path: str | Path) -> tuple[int, int]:
    """Return width, height using Pillow when available and header parsing otherwise."""
    path = Path(path)
    try:
        from PIL import Image

        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:
        pass

    with path.open("rb") as f:
        header = f.read(32)
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            width, height = struct.unpack(">II", header[16:24])
            return int(width), int(height)
        if header[:2] == b"\xff\xd8":
            f.seek(2)
            while True:
                marker_start = f.read(1)
                if not marker_start:
                    break
                if marker_start != b"\xff":
                    continue
                marker = f.read(1)
                while marker == b"\xff":
                    marker = f.read(1)
                if marker in {b"\xc0", b"\xc1", b"\xc2", b"\xc3", b"\xc5", b"\xc6", b"\xc7", b"\xc9", b"\xca", b"\xcb", b"\xcd", b"\xce", b"\xcf"}:
                    f.read(3)
                    height, width = struct.unpack(">HH", f.read(4))
                    return int(width), int(height)
                length_bytes = f.read(2)
                if len(length_bytes) != 2:
                    break
                length = struct.unpack(">H", length_bytes)[0]
                f.seek(max(length - 2, 0), os.SEEK_CUR)
    raise ValueError(f"Could not determine image dimensions for {path}")


def package_root() -> Path:
    return Path(__file__).resolve().parents[3]


def relpath_or_abs(path: str | Path, start: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(start).resolve()))
    except ValueError:
        return str(path)
