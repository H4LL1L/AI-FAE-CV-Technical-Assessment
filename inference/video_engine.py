"""
Real-time video pipelines:
    - Single-threaded detector+tracker loop
    - Multi-threaded queue-based pipeline for smoother FPS
    - Optional fusion analytics (zone occupancy, queue length)
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Generator, Iterable, List, Optional, Protocol, Tuple

import cv2
import numpy as np

from .detector import ByteTrackLite, Detector
from .fusion import FusionEngine
from .tracker import Tracker
from .utils import Detection, iou_matrix


class TrackerLike(Protocol):
    tracks: list

    def update(self, detections: List[Detection], label_lookup: dict | None = None) -> List[Detection]: ...

    def step_without_detections(self) -> List[Detection]: ...


class VideoEngine:
    def __init__(
        self,
        detector: Detector,
        tracker: TrackerLike,
        detect_every: int = 3,
        drift_iou: float = 0.5,
        visualize: bool = False,
        fusion: Optional[FusionEngine] = None,
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.detect_every = max(1, detect_every)
        self.drift_iou = drift_iou
        self.visualize = visualize
        self.fusion = fusion
        self.last_analytics: Optional[dict] = None

    def run(self, source: str | int | Path) -> None:
        """
        Run loop. Press 'q' to exit if visualize=True.
        """
        cap = cv2.VideoCapture(int(source) if isinstance(source, int) else str(source))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            t_start = time.perf_counter()
            do_detect = frame_idx % self.detect_every == 0
            if do_detect:
                det_out = self.detector(frame)
                detections = det_out["detections"]  # type: ignore[assignment]
                tracked = self.tracker.update(detections, label_lookup=self.detector.names)
                tracked = self._apply_drift_guard(detections, tracked)
            else:
                tracked = self.tracker.step_without_detections()

            if self.fusion:
                self.last_analytics = self.fusion.update(tracked, frame.shape[:2])

            latency_ms = (time.perf_counter() - t_start) * 1000.0
            if self.visualize:
                vis = self._draw(frame.copy(), tracked)
                cv2.putText(
                    vis,
                    f"Frame {frame_idx} | detect {do_detect} | {latency_ms:.1f} ms",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("video_engine", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1

        cap.release()
        if self.visualize:
            cv2.destroyAllWindows()

    def stream(
        self, source: str | int | Path
    ) -> Generator[Tuple[np.ndarray, Iterable[Detection]], None, None]:
        """
        Generator version: yields (frame, tracked_detections) for downstream use.
        """
        cap = cv2.VideoCapture(int(source) if isinstance(source, int) else str(source))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            do_detect = frame_idx % self.detect_every == 0
            if do_detect:
                det_out = self.detector(frame)
                detections = det_out["detections"]  # type: ignore[assignment]
                tracked = self.tracker.update(detections, label_lookup=self.detector.names)
                tracked = self._apply_drift_guard(detections, tracked)
            else:
                tracked = self.tracker.step_without_detections()

            if self.fusion:
                self.last_analytics = self.fusion.update(tracked, frame.shape[:2])

            yield frame, tracked
            frame_idx += 1

        cap.release()

    @staticmethod
    def _draw(frame: np.ndarray, detections: Iterable[Detection]) -> np.ndarray:
        for det in detections:
            x1, y1, x2, y2 = det.xyxy.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = det.label or "obj"
            tid = f" id:{det.track_id}" if det.track_id else ""
            text = f"{label} {det.score:.2f}{tid}"
            cv2.putText(
                frame,
                text,
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        return frame

    def _apply_drift_guard(
        self, detections: Iterable[Detection], tracked: Iterable[Detection]
    ) -> list[Detection]:
        """
        If detections and tracks disagree strongly (IoU below drift_iou),
        reset tracks to current detections to avoid drift.

        Note: This is a hard reset. Existing tracks are dropped and new track IDs
        will be assigned (ID continuity is intentionally broken to recover).
        """
        detections = list(detections)
        tracked = list(tracked)
        if self.drift_iou <= 0 or not detections or not tracked:
            return tracked

        det_boxes = np.stack([d.xyxy for d in detections], axis=0)
        trk_boxes = np.stack([t.xyxy for t in tracked], axis=0)
        ious = iou_matrix(trk_boxes, det_boxes)
        if np.max(ious) < self.drift_iou:
            # Hard reset: rebuild tracker from current detections
            self.tracker.tracks = []
            tracked = self.tracker.update(detections, label_lookup=self.detector.names)
        return tracked


class MultiThreadVideoEngine:
    """
    Queue-based multi-threaded video pipeline:
      - Capture thread reads frames
      - Inference thread runs detector + tracker
      - Optional visualization thread
    """

    def __init__(
        self,
        detector: Detector,
        tracker: TrackerLike,
        detect_every: int = 3,
        drift_iou: float = 0.5,
        visualize: bool = False,
        fusion: Optional[FusionEngine] = None,
        max_queue: int = 12,
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.detect_every = max(1, detect_every)
        self.drift_iou = drift_iou
        self.visualize = visualize
        self.fusion = fusion
        self.max_queue = max_queue
        self.stop_event = threading.Event()
        self.last_analytics: Optional[dict] = None

    def _process_frame(self, frame_idx: int, frame: np.ndarray) -> Tuple[np.ndarray, Iterable[Detection]]:
        do_detect = frame_idx % self.detect_every == 0
        if do_detect:
            det_out = self.detector(frame)
            detections = det_out["detections"]  # type: ignore[assignment]
            tracked = self.tracker.update(detections, label_lookup=self.detector.names)
            tracked = self._apply_drift_guard(detections, tracked)
        else:
            tracked = self.tracker.step_without_detections()

        if self.fusion:
            self.last_analytics = self.fusion.update(tracked, frame.shape[:2])
        return frame, tracked

    def _apply_drift_guard(
        self, detections: Iterable[Detection], tracked: Iterable[Detection]
    ) -> list[Detection]:
        """
        Same drift guard as VideoEngine, applied in the inference thread.

        Note: This is a hard reset. Existing tracks are dropped and new track IDs
        will be assigned (ID continuity is intentionally broken to recover).
        """
        detections = list(detections)
        tracked = list(tracked)
        if self.drift_iou <= 0 or not detections or not tracked:
            return tracked
        det_boxes = np.stack([d.xyxy for d in detections], axis=0)
        trk_boxes = np.stack([t.xyxy for t in tracked], axis=0)
        ious = iou_matrix(trk_boxes, det_boxes)
        if np.max(ious) < self.drift_iou:
            self.tracker.tracks = []
            tracked = self.tracker.update(detections, label_lookup=self.detector.names)
        return tracked

    def run(self, source: str | int | Path) -> None:
        frame_q: queue.Queue = queue.Queue(maxsize=self.max_queue)  # items: (capture_idx, frame)
        vis_q: queue.Queue = queue.Queue(maxsize=self.max_queue) if self.visualize else None

        def capture_worker() -> None:
            cap = cv2.VideoCapture(int(source) if isinstance(source, int) else str(source))
            if not cap.isOpened():
                self.stop_event.set()
                return
            capture_idx = 0
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                try:
                    frame_q.put((capture_idx, frame), timeout=0.5)
                except queue.Full:
                    # Drop frame under backpressure, but keep a "real" capture index.
                    pass
                finally:
                    capture_idx += 1
            cap.release()
            self.stop_event.set()

        def inference_worker() -> None:
            while not self.stop_event.is_set():
                try:
                    frame_idx, frame = frame_q.get(timeout=0.5)
                except queue.Empty:
                    continue
                _, tracked = self._process_frame(frame_idx, frame)
                if vis_q is not None:
                    try:
                        vis_q.put((frame, list(tracked), frame_idx), timeout=0.2)
                    except queue.Full:
                        pass

        def visualize_worker() -> None:
            if vis_q is None:
                return
            while not self.stop_event.is_set():
                try:
                    frame, tracked, frame_idx = vis_q.get(timeout=0.5)
                except queue.Empty:
                    continue
                vis = VideoEngine._draw(frame.copy(), tracked)
                cv2.putText(
                    vis,
                    f"Frame {frame_idx}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("video_engine_mt", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.stop_event.set()
                    break
            cv2.destroyAllWindows()

        threads = [
            threading.Thread(target=capture_worker, daemon=True),
            threading.Thread(target=inference_worker, daemon=True),
        ]
        if self.visualize:
            threads.append(threading.Thread(target=visualize_worker, daemon=True))

        for t in threads:
            t.start()
        try:
            while not self.stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop_event.set()
        for t in threads:
            t.join(timeout=1.0)
