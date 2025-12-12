"""
FastAPI service exposing detection and metrics endpoints.

Endpoints:
- GET /health  -> {"status": "ok"}
- POST /detect -> accept image file, return detections + timings
- GET /metrics -> basic runtime metrics (fps, avg latency, gpu info if available)
"""

from __future__ import annotations

import asyncio
import time
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool
try:
    from ultralytics import __version__ as yolo_version
except Exception:  # pragma: no cover - optional dependency
    yolo_version = "not-installed"

from api.schemas import DetectResult, DetectionResponse, Metrics
from inference.detector import ByteTrackLite, Detector
from inference.fusion import FusionEngine, QueueArea, Zone
from monitoring.fps_meter import FPSMeter
from monitoring.logger import GpuMonitor, JsonlLogger
from monitoring.dashboard import HTML_PAGE


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return raw if raw is not None and raw != "" else default


@dataclass(frozen=True)
class AppConfig:
    model_path: Path
    device: str
    imgsz: int
    backend: str
    startup_warmup_iters: int
    stream_ttl_s: int
    max_streams: int

    @staticmethod
    def from_env(
        *,
        model_path_default: str | Path,
        device_default: str,
        imgsz_default: int,
    ) -> "AppConfig":
        return AppConfig(
            model_path=Path(_env_str("MODEL_PATH", str(model_path_default))),
            device=_env_str("DEVICE", device_default),
            imgsz=_env_int("IMGSZ", imgsz_default),
            backend=_env_str("BACKEND", "auto"),
            startup_warmup_iters=_env_int("STARTUP_WARMUP_ITERS", 2),
            stream_ttl_s=_env_int("STREAM_TTL_S", 300),
            max_streams=_env_int("MAX_STREAMS", 32),
        )


@dataclass
class StreamState:
    tracker: ByteTrackLite
    fusion: FusionEngine
    last_seen_ts: float
    lock: asyncio.Lock


def load_image_to_bgr(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image bytes")
    return img


def _resolve_model_path(path_like: str | Path) -> Path:
    """
    Resolve model path with fallbacks:
      1) Explicit path if exists
      2) models/last.onnx if default path missing
    """
    p = Path(path_like)
    if p.exists():
        return p
    fallback = Path("models/last.onnx")
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Model not found: {p} (also tried {fallback})")


def _default_zones() -> list[Zone]:
    # Default ROIs (normalized) – can be tuned per deployment
    return [
        Zone(id="zone_entry", name="Entry", polygon=[(0.0, 0.0), (0.35, 0.0), (0.35, 1.0), (0.0, 1.0)]),
        Zone(id="zone_checkout", name="Checkout", polygon=[(0.35, 0.0), (0.7, 0.0), (0.7, 0.6), (0.35, 0.6)]),
        Zone(id="zone_shelf", name="Shelves", polygon=[(0.35, 0.6), (1.0, 0.6), (1.0, 1.0), (0.35, 1.0)]),
    ]


def _default_queues() -> list[QueueArea]:
    return [
        QueueArea(
            id="queue_checkout",
            name="Checkout Queue",
            polygon=[(0.4, 0.55), (0.95, 0.55), (0.95, 1.0), (0.4, 1.0)],
        )
    ]


def _new_tracker() -> ByteTrackLite:
    return ByteTrackLite(
        high_thresh=0.5,
        low_thresh=0.1,
        iou_thresh_high=0.5,
        iou_thresh_low=0.3,
        max_age=30,
        min_hits=1,
    )


def _new_fusion() -> FusionEngine:
    return FusionEngine(
        zones=_default_zones(),
        queues=_default_queues(),
        dwell_threshold_s=25.0,
        target_class_ids=[0],
    )


async def _get_stream_state(app: FastAPI, stream_id: str) -> StreamState:
    cfg: AppConfig = app.state.cfg
    now = time.time()
    async with app.state.streams_lock:
        streams: Dict[str, StreamState] = app.state.streams
        if cfg.stream_ttl_s > 0:
            stale = [k for k, v in streams.items() if now - v.last_seen_ts > cfg.stream_ttl_s]
            for k in stale:
                del streams[k]

        st = streams.get(stream_id)
        if st is None:
            if len(streams) >= cfg.max_streams:
                oldest_key = min(streams.items(), key=lambda kv: kv[1].last_seen_ts)[0]
                del streams[oldest_key]
            st = StreamState(
                tracker=_new_tracker(),
                fusion=_new_fusion(),
                last_seen_ts=now,
                lock=asyncio.Lock(),
            )
            streams[stream_id] = st
        else:
            st.last_seen_ts = now
        return st


def create_app(
    model_path: str | Path = "models/model.onnx",  # CPU-friendly default
    device: str = "cpu",
    imgsz: int = 640,
) -> FastAPI:
    cfg = AppConfig.from_env(model_path_default=model_path, device_default=device, imgsz_default=imgsz)
    app = FastAPI(title="CV Advanced Assessment API", version="1.0.0")
    app.state.cfg = cfg
    app.state.detector = None
    app.state.detector_error = None
    app.state.streams = {}
    app.state.streams_lock = asyncio.Lock()
    app.state.fps_meter = FPSMeter(window=200)
    app.state.gpu_monitor = GpuMonitor()
    app.state.metrics_logger = JsonlLogger(Path("monitoring/logs/metrics.jsonl"))
    app.state.detection_counter = {"count": 0}
    app.state.fusion_config = _new_fusion().config()

    @app.on_event("startup")
    async def _startup() -> None:
        try:
            resolved_model = _resolve_model_path(cfg.model_path)
            detector = Detector(
                model_path=resolved_model,
                device=cfg.device,
                imgsz=cfg.imgsz,
                conf=0.25,
                backend=cfg.backend,
                warmup_iters=0,
            )
            app.state.detector = detector
            app.state.detector_error = None
        except Exception as exc:
            app.state.detector = None
            app.state.detector_error = str(exc)
            return

        warmup_iters = cfg.startup_warmup_iters
        if warmup_iters > 0:
            dummy = np.zeros((cfg.imgsz, cfg.imgsz, 3), dtype=np.uint8)
            for _ in range(warmup_iters):
                try:
                    await run_in_threadpool(detector, dummy)
                except Exception:
                    break

    @app.get("/health")
    def health():
        detector: Optional[Detector] = app.state.detector
        return {
            "status": "ok",
            "model_loaded": detector is not None,
            "backend": detector.backend if detector is not None else "not-ready",
            "yolo": yolo_version,
            "model_path": str(cfg.model_path),
            "device": cfg.device,
            "imgsz": cfg.imgsz,
            "error": app.state.detector_error,
        }

    @app.get("/dashboard")
    def dashboard():
        return HTMLResponse(content=HTML_PAGE, status_code=200)

    @app.post("/detect", response_model=DetectResult)
    async def detect(file: UploadFile = File(...), stream_id: str = "default"):
        detector: Optional[Detector] = app.state.detector
        if detector is None:
            detail = app.state.detector_error or "model not loaded"
            raise HTTPException(status_code=503, detail=detail)

        try:
            data = await file.read()
            frame = await run_in_threadpool(load_image_to_bgr, data)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        try:
            t0 = time.perf_counter()
            out = await run_in_threadpool(detector, frame)
            detections = out["detections"]

            # Track/fusion state is per-stream to avoid cross-client mixing.
            stream = await _get_stream_state(app, stream_id)
            async with stream.lock:
                tracked = await run_in_threadpool(stream.tracker.update, detections, detector.names)
                analytics = await run_in_threadpool(stream.fusion.update, tracked, frame.shape[:2])
            t1 = time.perf_counter()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"inference failed: {exc}")

        latency_ms = (t1 - t0) * 1000.0
        app.state.fps_meter.tick()
        app.state.fps_meter.observe(latency_ms)
        app.state.detection_counter["count"] += 1
        resp = DetectResult(
            backend=out["backend"],
            detections=[
                DetectionResponse(
                    x1=float(d.xyxy[0]),
                    y1=float(d.xyxy[1]),
                    x2=float(d.xyxy[2]),
                    y2=float(d.xyxy[3]),
                    score=d.score,
                    cls=d.cls,
                    label=d.label,
                    track_id=d.track_id,
                )
                for d in tracked
            ],
            timings_ms={
                "inference": out["timings_ms"]["inference"],
                "total": latency_ms,
            },
            analytics=analytics,
        )
        try:
            await run_in_threadpool(
                app.state.metrics_logger.log,
                {
                    "latency_ms": latency_ms,
                    "backend": out["backend"],
                    "fps": app.state.fps_meter.fps(),
                    **app.state.fps_meter.percentiles(),
                    "ts": time.time(),
                    "detections_total": app.state.detection_counter["count"],
                    "stream_id": stream_id,
                    "analytics": analytics,
                }
            )
        except Exception:
            pass
        return resp

    @app.get("/metrics", response_model=Metrics)
    def metrics():
        detector: Optional[Detector] = app.state.detector
        backend = detector.backend if detector is not None else "not-ready"
        gpu_info = app.state.gpu_monitor.read()
        pct = app.state.fps_meter.percentiles()
        return Metrics(
            fps=app.state.fps_meter.fps(),
            avg_latency_ms=app.state.fps_meter.avg_latency_ms(),
            p50=pct["p50"],
            p90=pct["p90"],
            p95=pct["p95"],
            backend=backend,
            gpu=gpu_info,
            detections_total=app.state.detection_counter["count"],
            zones=app.state.fusion_config,
        )

    return app


app = create_app()

# To run: uvicorn api.server:app --host 0.0.0.0 --port 8000
