"""
Lightweight IoU-based multi-object tracker (SORT-style without Kalman).

Tracks are updated greedily by IoU matching. If no detection step is run
(e.g., between detector intervals), call `step_without_detections` to age tracks.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import List

import numpy as np

from .utils import Detection, iou_matrix


_id_gen = itertools.count(1)


@dataclass
class Track:
    track_id: int
    xyxy: np.ndarray
    cls: int
    score: float
    age: int = 0  # total frames since birth
    time_since_update: int = 0  # frames since last matched detection
    hits: int = 1  # matched count

    def to_detection(self, label_lookup: dict | None = None) -> Detection:
        label = None
        if label_lookup:
            label = label_lookup.get(self.cls)
        return Detection(
            xyxy=self.xyxy.copy(),
            score=self.score,
            cls=self.cls,
            label=label,
            track_id=self.track_id,
        )


class Tracker:
    def __init__(self, iou_thresh: float = 0.3, max_age: int = 30, min_hits: int = 1):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: List[Track] = []

    def _create_track(self, det: Detection) -> Track:
        return Track(
            track_id=next(_id_gen),
            xyxy=det.xyxy.astype(np.float32),
            cls=det.cls,
            score=det.score,
        )

    def step_without_detections(self) -> List[Detection]:
        """Age existing tracks when detector is skipped."""
        surviving: List[Track] = []
        for trk in self.tracks:
            trk.age += 1
            trk.time_since_update += 1
            if trk.time_since_update <= self.max_age:
                surviving.append(trk)
        self.tracks = surviving
        return self.get_active_tracks()

    def update(self, detections: List[Detection], label_lookup: dict | None = None) -> List[Detection]:
        """
        Update tracks with current detections.
        Returns list of detections augmented with track_id for active tracks.
        """
        if len(detections) == 0:
            return self.step_without_detections()

        det_boxes = np.stack([d.xyxy for d in detections], axis=0)
        trk_boxes = np.stack([t.xyxy for t in self.tracks], axis=0) if self.tracks else np.zeros((0, 4))

        if len(trk_boxes) and len(det_boxes):
            ious = iou_matrix(trk_boxes, det_boxes)
        else:
            ious = np.zeros((len(trk_boxes), len(det_boxes)), dtype=np.float32)

        matched_trk = set()
        matched_det = set()

        # Greedy matching by IoU
        while True:
            if ious.size == 0:
                break
            idx = np.unravel_index(np.argmax(ious), ious.shape)
            max_iou = ious[idx]
            if max_iou < self.iou_thresh:
                break
            trk_idx, det_idx = int(idx[0]), int(idx[1])
            if trk_idx in matched_trk or det_idx in matched_det:
                ious[trk_idx, det_idx] = -1
                continue

            trk = self.tracks[trk_idx]
            det = detections[det_idx]

            # Update track with detection
            trk.xyxy = det.xyxy.astype(np.float32)
            trk.score = det.score
            trk.cls = det.cls
            trk.hits += 1
            trk.time_since_update = 0
            trk.age += 1

            matched_trk.add(trk_idx)
            matched_det.add(det_idx)
            ious[trk_idx, :] = -1
            ious[:, det_idx] = -1

        # Unmatched tracks -> age
        updated_tracks: List[Track] = []
        for idx, trk in enumerate(self.tracks):
            if idx not in matched_trk:
                trk.age += 1
                trk.time_since_update += 1
            if trk.time_since_update <= self.max_age:
                updated_tracks.append(trk)
        self.tracks = updated_tracks

        # New tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx in matched_det:
                continue
            self.tracks.append(self._create_track(det))

        # Filter by min_hits
        active = [t for t in self.tracks if t.hits >= self.min_hits and t.time_since_update == 0]
        return [t.to_detection(label_lookup) for t in active]

    def get_active_tracks(self, label_lookup: dict | None = None) -> List[Detection]:
        return [t.to_detection(label_lookup) for t in self.tracks if t.time_since_update == 0 and t.hits >= self.min_hits]




