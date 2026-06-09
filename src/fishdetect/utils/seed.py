from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def detect_device(preferred: str = "auto") -> str:
    if preferred and preferred != "auto":
        return preferred
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def package_versions() -> dict[str, Any]:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fishdetect_mplconfig"))
    versions: dict[str, Any] = {"python": None}
    try:
        import platform

        versions["python"] = platform.python_version()
        versions["platform"] = platform.platform()
    except Exception:
        pass
    for package in ["numpy", "pandas", "PIL", "torch", "torchvision", "ultralytics", "matplotlib"]:
        try:
            module = __import__(package)
            versions[package] = getattr(module, "__version__", "installed")
        except Exception:
            versions[package] = None
    return versions
