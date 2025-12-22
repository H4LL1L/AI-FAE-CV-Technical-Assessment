# CV Advanced Assessment — Real‑Time YOLO + Tracking + Zone/Queue Analytics

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-111827)](https://docs.ultralytics.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-CPU%2FGPU-005CED)](https://onnxruntime.ai/)

An end-to-end sample project for real-time **object detection + tracking + area analytics**:
YOLO-based detection (multi-backend) + lightweight tracker + zone/queue analytics + FastAPI service + live metrics dashboard.

## What’s Inside?

- **Multi-backend inference**: `.pt` (PyTorch/Ultralytics), `.onnx` (ONNX Runtime), `.engine` (TensorRT) + `backend=auto`
- **Tracking**: `ByteTrackLite` + IoU-based drift guard
- **Fusion / Analytics**: zone occupancy, queue counting, dwell-time (loitering proxy)
- **API**: `/detect`, `/metrics`, `/health`, `/dashboard` (single-page modern panel)
- **Monitoring**: FPS/latency p50/p90/p95 + optional GPU (NVML) + JSONL logging
- **Optimization**: ONNX export, TensorRT FP16/INT8 engine build, INT8 calibration, benchmark scripts
- **Training**: Ultralytics YOLOv8 fine-tuning pipeline + optional Albumentations

## Quickstart

### 1) Model Artifacts

By default the API expects `MODEL_PATH=models/model.onnx`; if it doesn’t exist it falls back to `models/last.onnx`.

- `models/latest.pt`: PyTorch checkpoint (training output)
- `models/model.onnx` or `models/last.onnx`: ONNX model
- `models/last_fp16.engine`, `models/last_int8.engine`: TensorRT engines (optional)

### 2) Local Setup (CPU / incl. macOS)

This repo does not include a `pyproject.toml`; for a practical setup you can start from the Docker requirements list.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

Quick path on Linux (CPU/GPU):

```bash
python -m pip install -r api/docker/requirements.txt
```

For macOS/CPU (recommended): use `onnxruntime` instead of `onnxruntime-gpu`, and `opencv-python` instead of `opencv-python-headless`.

```bash
python -m pip install fastapi uvicorn[standard] python-multipart pydantic numpy opencv-python PyYAML ultralytics onnxruntime requests
```

### 3) Run the API

```bash
MODEL_PATH="models/model.onnx" \
DEVICE="cpu" \
IMGSZ="640" \
BACKEND="auto" \
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

- Health: `http://127.0.0.1:8000/health`
- Dashboard: `http://127.0.0.1:8000/dashboard`
- Metrics (JSON): `http://127.0.0.1:8000/metrics`

## API Usage

### `/detect` (single frame)

```bash
curl -sS -F "file=@/path/to/image.jpg" "http://127.0.0.1:8000/detect?stream_id=cam-1" | jq .
```

Response (high level):

- `backend`: active inference backend (`ort`, `tensorrt`, `pytorch`, …)
- `detections[]`: bbox + score + class + optional `track_id`
- `analytics`: zone/queue stats, dwell-time, `suspicious` track list
- `timings_ms`: inference and total

### Populate the Dashboard (send video frames)

```bash
python scripts/send_frames_api.py \
  --video video.mp4 \
  --url http://127.0.0.1:8000/detect \
  --stride 2 \
  --limit 200 \
  --stream-id demo
```

## Video Pipeline (local)

Runs detector + tracker + (optional) fusion analytics in a single run.

```bash
python scripts/run_pipeline.py \
  --model models/last.onnx \
  --backend auto \
  --device cpu \
  --imgsz 640 \
  --detect-every 2 \
  --source video.mp4 \
  --visualize
```

Multi-thread (more stable FPS):

```bash
python scripts/run_pipeline.py --threaded --visualize --source video.mp4
```

## Configuration

Environment variables for the FastAPI service:

- `MODEL_PATH`: model path (`.onnx/.pt/.engine`)
- `DEVICE`: `"cpu"` or something like `"0"` for GPU
- `IMGSZ`: inference input size (e.g. `640`)
- `BACKEND`: `auto | ort | ort-trt | tensorrt | ultralytics`
- `STARTUP_WARMUP_ITERS`: startup warmup iterations (default `2`)
- `STREAM_TTL_S`: stream state TTL (default `300`)
- `MAX_STREAMS`: max number of stream states to keep (default `32`)

## Architecture (High Level)

```mermaid
flowchart LR
  A[Frame (BGR)] --> B[Detector\n(PT / ORT / TRT)]
  B --> C[Postprocess + NMS]
  C --> D[Tracker\n(ByteTrackLite)]
  D --> E[FusionEngine\nZones & Queues]
  E --> F[FastAPI Response]
  F --> G[JsonlLogger + FPSMeter]
  G --> H[/metrics & /dashboard]
```

## Repository Layout

- `api/`: FastAPI service (`api/server.py`) + schemas
- `inference/`: detector, tracker, video engine, fusion analytics
- `monitoring/`: FPS/latency meter, GPU monitor, HTML dashboard, logs
- `optimization/`: ONNX export, TRT engine build, INT8 calibration, benchmarks
- `training/`: Ultralytics fine-tuning + augmentations
- `tests/`: pytest tests (with optional dependency guards)

## Benchmarks / Results

This repo includes sample measurements:

- CPU ONNX pipeline: `optimization/before_quantization_benchmark_results.json`
  - `throughput_fps`: `13.65`
  - `wall_ms p50/p95`: `72.10 / 80.03`
- TensorRT engine benchmark: `benchmark_results.json`
  - FP16: `~117 FPS` (GPU avg `~8.55 ms`)
  - INT8: `~160 FPS` (GPU avg `~6.25 ms`)

Note: Numbers vary by hardware, input size, video source, and postprocess load; rerun benchmarks in your own environment for reliable results.

## Optimization Flow

1) **PyTorch → ONNX**: `optimization/export_to_onnx.py`
2) **INT8 calibration cache**: `optimization/calibrate_int8.py`
3) **TensorRT FP16/INT8 engine**: `optimization/build_trt_engine.py`
4) **Benchmarks**: `optimization/benchmarks.py` (video-based), `optimization/benchmarks.py` (TRT engine-based)

## Docker (recommended for GPU/TensorRT)

`api/docker/Dockerfile` uses an NVIDIA TensorRT base image.

```bash
docker build -f api/docker/Dockerfile -t cv-assessment:latest .
docker run --rm -p 8000:8000 --gpus all cv-assessment:latest
```

## Tests

```bash
pytest -q
```

Tests automatically `skip` when certain backends or model files are missing (e.g. `onnxruntime`, `torch`, models).


