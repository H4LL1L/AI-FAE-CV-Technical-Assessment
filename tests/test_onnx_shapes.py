import os
import numpy as np
import pytest

try:
    from inference.detector import Detector
    _detector_available = True
except Exception as exc:  # pragma: no cover - optional dependency guard
    Detector = None  # type: ignore
    _detector_available = False
    _detector_import_error = exc


def _skip_if_no_onnxruntime():
    try:
        import onnxruntime  # noqa: F401
    except Exception as exc:
        pytest.skip(f"onnxruntime missing: {exc}")


def _skip_if_detector_unavailable():
    if not _detector_available:
        pytest.skip(f"detector backend unavailable: {_detector_import_error}")


@pytest.mark.skipif(
    not os.path.exists("models/model.onnx"),
    reason="model.onnx missing; place an ONNX file to run this test",
)
def test_detector_backend_label_names():
    _skip_if_no_onnxruntime()
    _skip_if_detector_unavailable()
    det = Detector("models/model.onnx", device="cpu", imgsz=320, conf=0.1, warmup_iters=0)
    assert det.backend in {"ort", "ort-trt"}
    assert isinstance(det.names, dict)
    assert len(det.names) >= 1
    for k, v in list(det.names.items())[:3]:
        assert isinstance(k, int)
        assert isinstance(v, str)
        assert v


@pytest.mark.skipif(
    not os.path.exists("models/model.onnx"),
    reason="model.onnx missing; place an ONNX file to run this test",
)
def test_detector_preprocess_handles_varied_input_frames():
    _skip_if_no_onnxruntime()
    _skip_if_detector_unavailable()
    det = Detector("models/model.onnx", device="cpu", imgsz=320, conf=0.1, warmup_iters=0)
    # Preprocess (letterbox) should accept different frame sizes without crashing.
    for h, w in [(320, 320), (480, 640)]:
        img = np.zeros((h, w, 3), dtype=np.uint8)
        out = det(img)
        assert "detections" in out


@pytest.mark.skipif(
    not os.path.exists("models/model.onnx"),
    reason="model.onnx missing; place an ONNX file to run this test",
)
def test_ort_model_accepts_multiple_imgsz_when_dynamic():
    _skip_if_no_onnxruntime()
    _skip_if_detector_unavailable()

    det = Detector("models/model.onnx", device="cpu", imgsz=320, conf=0.1, warmup_iters=0)
    runner = getattr(det, "_runner", None)
    session = getattr(runner, "session", None)
    if session is None:
        pytest.skip("ORT session not accessible; cannot inspect input shape")

    input_shape = session.get_inputs()[0].shape

    def _is_dynamic_dim(dim: object) -> bool:
        return dim is None or isinstance(dim, str)

    if len(input_shape) < 4 or not (_is_dynamic_dim(input_shape[2]) or _is_dynamic_dim(input_shape[3])):
        pytest.skip(f"model input appears static (H/W): {input_shape}")

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for imgsz in (320, 640):
        det_i = Detector("models/model.onnx", device="cpu", imgsz=imgsz, conf=0.1, warmup_iters=0)
        out = det_i(frame)
        assert "detections" in out

