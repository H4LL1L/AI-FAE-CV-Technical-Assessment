import os

import numpy as np
import pytest

try:
    from inference.detector import Detector
    from inference.tracker import Tracker
    from inference.utils import Detection
    from inference.video_engine import VideoEngine

    _detector_available = True
except Exception as exc:  # pragma: no cover - optional dependency guard
    Detector = None  # type: ignore
    Tracker = None  # type: ignore
    Detection = None  # type: ignore
    VideoEngine = None  # type: ignore
    _detector_available = False
    _detector_import_error = exc


def _skip_if_torch_no_numpy():
    """
    .pt inference typically requires torch. Also skip if torch was built without
    NumPy interop (common in minimal builds).
    """
    try:
        import torch
    except Exception as exc:
        pytest.skip(f"torch missing: {exc}")

    try:
        torch.from_numpy(np.zeros((1,), dtype=np.float32))
    except Exception as exc:
        pytest.skip(f"torch<->numpy interop unavailable: {exc}")


def _skip_if_detector_unavailable():
    if not _detector_available:
        pytest.skip(f"detector backend unavailable: {_detector_import_error}")


def _get_pt_model_path() -> str:
    env_path = os.getenv("MODEL_PT")
    if env_path:
        if os.path.exists(env_path):
            return env_path
        pytest.skip(f"MODEL_PT is set but missing: {env_path}")

    for candidate in ("models/latest.pt", "models/last.pt", "latest.pt", "last.pt"):
        if os.path.exists(candidate):
            return candidate

    pytest.skip("No .pt model found; set MODEL_PT=... or add models/latest.pt (or models/last.pt)")


class DummyDetector:
    def __init__(self) -> None:
        self.names = {0: "person"}
        self.backend = "dummy"

    def __call__(self, frame):  # pragma: no cover - should not be called in this test
        raise AssertionError("DummyDetector should not be called in this test")


@pytest.mark.skipif(Detector is None, reason="detector backend unavailable")
def test_detector_runs_on_dummy_image():
    _skip_if_torch_no_numpy()
    _skip_if_detector_unavailable()
    model_path = _get_pt_model_path()
    det = Detector(model_path, device="cpu", imgsz=320, conf=0.1, warmup_iters=0)
    img = np.zeros((320, 320, 3), dtype=np.uint8)
    out = det(img)
    assert "detections" in out
    assert "timings_ms" in out


def test_video_engine_drift_guard_resets_tracks():
    _skip_if_detector_unavailable()
    # No model needed; operate on the helper directly
    tracker = Tracker(iou_thresh=0.3, max_age=5, min_hits=1)
    dummy_detector = DummyDetector()
    engine = VideoEngine(detector=dummy_detector, tracker=tracker, detect_every=1, drift_iou=0.5, visualize=False)

    # Seed tracker with existing tracks; drift guard should drop these and rebuild.
    seed_dets = [
        Detection(xyxy=np.array([50, 50, 60, 60], dtype=np.float32), score=0.9, cls=0),
        Detection(xyxy=np.array([70, 70, 80, 80], dtype=np.float32), score=0.8, cls=0),
    ]
    tracker.update(seed_dets, label_lookup=dummy_detector.names)
    assert len(tracker.tracks) == 2
    old_ids = {t.track_id for t in tracker.tracks}

    dets = [Detection(xyxy=np.array([0, 0, 10, 10], dtype=np.float32), score=0.9, cls=0)]
    # Provide tracked boxes far away to force drift reset.
    bad_tracked = [
        Detection(
            xyxy=np.array([100, 100, 110, 110], dtype=np.float32),
            score=0.5,
            cls=0,
            track_id=999,
        )
    ]

    tracked_after = engine._apply_drift_guard(dets, bad_tracked)
    assert len(tracker.tracks) == 1
    assert {t.track_id for t in tracker.tracks}.isdisjoint(old_ids)
    assert len(tracked_after) == 1
    assert tracked_after[0].track_id is not None


@pytest.mark.skipif(Detector is None, reason="detector backend unavailable")
def test_detector_batch_mode():
    _skip_if_torch_no_numpy()
    _skip_if_detector_unavailable()
    model_path = _get_pt_model_path()
    det = Detector(model_path, device="cpu", imgsz=320, conf=0.1, warmup_iters=0)
    imgs = np.zeros((2, 320, 320, 3), dtype=np.uint8)
    out = det(imgs)
    assert isinstance(out["detections"], list)
    assert len(out["detections"]) == 2
