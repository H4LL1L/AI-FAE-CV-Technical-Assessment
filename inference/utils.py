"""
Utility helpers for inference, box math, and timing.

Designed to be backend-agnostic and NumPy-friendly so the same helpers can be
used for PyTorch, ONNX Runtime, and TensorRT outputs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class Detection:
    """Lightweight detection container."""

    xyxy: np.ndarray  # shape (4,) in [x1, y1, x2, y2]
    score: float
    cls: int
    label: str | None = None
    track_id: int | None = None


def now_ms() -> float:
    """Return wall-clock time in milliseconds."""
    return time.perf_counter() * 1000.0


def xyxy_to_xywh(xyxy: Sequence[float]) -> np.ndarray:
    """Convert [x1, y1, x2, y2] -> [x, y, w, h]."""
    x1, y1, x2, y2 = xyxy
    return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)


def xywh_to_xyxy(xywh: Sequence[float]) -> np.ndarray:
    """Convert [x, y, w, h] -> [x1, y1, x2, y2]."""
    x, y, w, h = xywh
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute IoU matrix between two sets of boxes.
    a: (N,4) [x1,y1,x2,y2], b: (M,4)
    """
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    a = a.astype(np.float32)
    b = b.astype(np.float32)

    # Areas
    area_a = (a[:, 2] - a[:, 0]).clip(min=0) * (a[:, 3] - a[:, 1]).clip(min=0)
    area_b = (b[:, 2] - b[:, 0]).clip(min=0) * (b[:, 3] - b[:, 1]).clip(min=0)

    # Broadcast intersections
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter_w = (x2 - x1).clip(min=0)
    inter_h = (y2 - y1).clip(min=0)
    inter = inter_w * inter_h

    union = area_a[:, None] + area_b[None, :] - inter
    union = np.clip(union, a_min=1e-6, a_max=None)
    return inter / union


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.45) -> List[int]:
    """
    Pure NumPy NMS. Returns indices of kept boxes.
    Assumes boxes are [x1, y1, x2, y2].
    """
    if len(boxes) == 0:
        return []
    boxes = boxes.astype(np.float32)
    scores = scores.astype(np.float32)

    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        ious = iou_matrix(boxes[i : i + 1], boxes[order[1:]])[0]
        remain = np.where(ious <= iou_thr)[0]
        order = order[remain + 1]
    return keep


def classwise_nms(
    boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_thr: float = 0.45
) -> List[int]:
    """
    Run NMS per class and return kept indices (global indices into inputs).
    """
    if len(boxes) == 0:
        return []

    keep: List[int] = []
    classes = classes.astype(int)
    for c in np.unique(classes):
        idx = np.where(classes == c)[0]
        if len(idx) == 0:
            continue
        keep_cls = nms(boxes[idx], scores[idx], iou_thr=iou_thr)
        keep.extend(idx[ki] for ki in keep_cls)
    return keep


def clip_boxes(boxes: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Clip boxes to image shape (h, w)."""
    h, w = shape
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h)
    return boxes


def scale_boxes(
    boxes: np.ndarray, src_shape: Tuple[int, int], dst_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Scale boxes from src_shape (h,w) to dst_shape (h,w).
    """
    src_h, src_w = src_shape
    dst_h, dst_w = dst_shape
    gain_w = dst_w / src_w
    gain_h = dst_h / src_h
    scaled = boxes.copy().astype(np.float32)
    scaled[:, [0, 2]] *= gain_w
    scaled[:, [1, 3]] *= gain_h
    return scaled




