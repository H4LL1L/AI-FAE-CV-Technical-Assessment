"""Shared Pydantic schemas for the API."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class DetectionResponse(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    cls: int
    label: Optional[str] = None
    track_id: Optional[int] = None


class DetectResult(BaseModel):
    backend: str
    detections: List[DetectionResponse]
    timings_ms: dict
    analytics: Optional[dict] = None


class Metrics(BaseModel):
    fps: float
    avg_latency_ms: float
    p50: float
    p90: float
    p95: float
    backend: str
    gpu: Optional[dict] = None
    detections_total: int = 0
    zones: Optional[dict] = None
