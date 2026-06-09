from __future__ import annotations

from pathlib import Path
from typing import Any

from fishdetect.experiment import skipped_metrics


def torchvision_available() -> bool:
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401

        return True
    except Exception:
        return False


def available_models() -> set[str]:
    return {"fasterrcnn_resnet50_fpn", "retinanet_resnet50_fpn", "fcos_resnet50_fpn"}


def train_torchvision_detector(
    config: dict[str, Any],
    experiment: dict[str, Any],
    output_dir: str | Path,
    epochs: int | None = None,
    device: str = "auto",
    smoke_test: bool = False,
    train_in_smoke: bool = False,
) -> dict[str, Any]:
    model_name = experiment["model"]
    if model_name not in available_models():
        return skipped_metrics(f"Torchvision model '{model_name}' is not registered.", experiment, smoke_test=smoke_test)
    if not torchvision_available():
        return skipped_metrics("Torch/Torchvision is not installed.", experiment, smoke_test=smoke_test)
    if smoke_test and not train_in_smoke:
        return skipped_metrics(
            "Torchvision smoke mode validated configuration but skipped training to keep the smoke run fast.",
            experiment,
            smoke_test=True,
        )
    raise NotImplementedError(
        "Full Torchvision training is scaffolded but intentionally not launched by default. "
        "Use YOLO for first production training, or extend this runner with project-specific batching."
    )


def build_model(model_name: str, num_classes: int, weights: str | None = None) -> Any:
    import torchvision

    if model_name == "fasterrcnn_resnet50_fpn":
        return torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
    if model_name == "retinanet_resnet50_fpn":
        return torchvision.models.detection.retinanet_resnet50_fpn(weights=None, num_classes=num_classes)
    if model_name == "fcos_resnet50_fpn":
        return torchvision.models.detection.fcos_resnet50_fpn(weights=None, num_classes=num_classes)
    raise ValueError(f"Unsupported Torchvision detector: {model_name}")
