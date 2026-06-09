from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import tempfile

from fishdetect.utils.files import ensure_dir, write_text


os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fishdetect_mplconfig"))


def save_barplot(path: str | Path, rows: list[dict[str, Any]], x_key: str, y_key: str, title: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [str(row.get(x_key, "")) for row in rows]
        values = [float(row.get(y_key, 0) or 0) for row in rows]
        fig_width = max(6, min(20, len(labels) * 0.6))
        plt.figure(figsize=(fig_width, 4))
        plt.bar(labels, values, color="#267c8f")
        plt.xticks(rotation=45, ha="right")
        plt.ylim(0, max(1.0, max(values, default=0.0) * 1.1))
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
    except Exception as exc:
        write_text(path.with_suffix(".txt"), f"Plot unavailable: {exc}\n")


def save_precision_recall_scatter(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(5, 5))
        for row in rows:
            plt.scatter(float(row.get("precision", 0) or 0), float(row.get("recall", 0) or 0), label=row.get("experiment", ""))
        plt.xlabel("Precision")
        plt.ylabel("Recall")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        if len(rows) <= 12:
            plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
    except Exception as exc:
        write_text(path.with_suffix(".txt"), f"Plot unavailable: {exc}\n")


def save_confusion_matrix_placeholder(path: str | Path, title: str = "Confusion Matrix") -> None:
    path = Path(path)
    ensure_dir(path.parent)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(4, 4))
        plt.text(0.5, 0.5, "Generated after predictions are available", ha="center", va="center", wrap=True)
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
    except Exception as exc:
        write_text(path.with_suffix(".txt"), f"Plot unavailable: {exc}\n")
