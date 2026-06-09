from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AugmentationPolicy:
    hsv: bool = True
    scale: bool = True
    mosaic: bool = True
    flipud: bool = False
    fliplr: bool = True
    underwater_color_jitter: bool = True
    mild_blur: bool = True
    haze_noise: bool = True


def yolo_augmentation_kwargs(policy: dict) -> dict:
    """Translate project augmentation flags to Ultralytics training kwargs."""
    kwargs = {}
    if not policy.get("hsv", True):
        kwargs.update({"hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0})
    if not policy.get("scale", True):
        kwargs["scale"] = 0.0
    if not policy.get("mosaic", True):
        kwargs["mosaic"] = 0.0
    kwargs["flipud"] = 0.5 if policy.get("flipud", False) else 0.0
    kwargs["fliplr"] = 0.5 if policy.get("fliplr", True) else 0.0
    return kwargs


def marine_augmentation_notes() -> list[str]:
    return [
        "Use mild brightness, contrast, color jitter, haze/noise, blur, and scale variation.",
        "Avoid vertical flips by default because fish orientation and benthic context may be meaningful.",
        "Keep juvenile, initial phase, and terminal phase labels separate unless taxonomy review explicitly merges them.",
    ]
