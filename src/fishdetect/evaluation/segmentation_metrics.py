from __future__ import annotations

from typing import Sequence


def binary_iou(pred_mask: Sequence[Sequence[int]], true_mask: Sequence[Sequence[int]]) -> float:
    inter = 0
    union = 0
    for pred_row, true_row in zip(pred_mask, true_mask):
        for pred, true in zip(pred_row, true_row):
            pred_bool = bool(pred)
            true_bool = bool(true)
            inter += int(pred_bool and true_bool)
            union += int(pred_bool or true_bool)
    return inter / union if union else 0.0


def dice_score(pred_mask: Sequence[Sequence[int]], true_mask: Sequence[Sequence[int]]) -> float:
    inter = 0
    pred_count = 0
    true_count = 0
    for pred_row, true_row in zip(pred_mask, true_mask):
        for pred, true in zip(pred_row, true_row):
            pred_bool = bool(pred)
            true_bool = bool(true)
            inter += int(pred_bool and true_bool)
            pred_count += int(pred_bool)
            true_count += int(true_bool)
    denom = pred_count + true_count
    return (2 * inter / denom) if denom else 0.0
