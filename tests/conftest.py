from __future__ import annotations

import shutil
import struct
import zlib
from pathlib import Path

import pytest


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "synthetic"


def write_png(path: Path, width: int = 100, height: int = 100) -> None:
    raw = b"".join(b"\x00" + b"\x80\x90\xa0" * width for _ in range(height))
    compressor = zlib.compressobj()
    data = compressor.compress(raw) + compressor.flush()

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", data)
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


@pytest.fixture()
def synthetic_dataset(tmp_path: Path) -> Path:
    dst = tmp_path / "synthetic"
    shutil.copytree(FIXTURE_ROOT, dst)
    for name in ["000000.png", "000001.png", "000002.png", "000003.png"]:
        write_png(dst / name)
    return dst
