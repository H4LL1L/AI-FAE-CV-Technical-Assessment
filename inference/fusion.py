"""
Fusion/analytics utilities for zone/area monitoring:
- Zone occupancy counting
- Optional "queue/holding-area" analytics
- Dwell-time / loitering proxy

All coordinates are normalized (0..1) relative to frame width/height so they
work across resolutions. The engine keeps lightweight per-track state using
`track_id` produced by the tracker.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from .utils import Detection


Point = Tuple[float, float]


@dataclass
class Zone:
    id: str
    name: str
    polygon: List[Point]  # normalized [(x,y), ...]


@dataclass
class QueueArea:
    id: str
    name: str
    polygon: List[Point]  # normalized


def _normalize_polygon_to_pixels(
    polygon: List[Point], frame_shape: Tuple[int, int]
) -> np.ndarray:
    """Convert normalized polygon to absolute pixel coordinates (float32)."""
    h, w = frame_shape
    arr = np.array(polygon, dtype=np.float32)
    arr[:, 0] *= w
    arr[:, 1] *= h
    return arr


def _point_in_polygon(point: Point, polygon_px: np.ndarray) -> bool:
    """
    Ray casting algorithm for point-in-polygon.
    polygon_px: (N,2) float32 in pixel coordinates.
    """
    x, y = point
    poly = polygon_px
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


class FusionEngine:
    """
    Maintains zone/queue analytics and dwell-time heuristics using tracker IDs.
    """

    def __init__(
        self,
        zones: List[Zone],
        queues: Optional[List[QueueArea]] = None,
        dwell_threshold_s: float = 25.0,
        target_class_ids: Optional[List[int]] = None,
        state_ttl_s: float = 600.0,
    ) -> None:
        self.zones = zones
        self.queues = queues or []
        self.dwell_threshold_s = dwell_threshold_s
        # Which classes count as "people" for occupancy; default=all if None
        self.target_class_ids = target_class_ids
        self.state_ttl_s = state_ttl_s
        # track_id -> state dict
        self._track_state: Dict[int, Dict[str, object]] = {}

    def config(self) -> Dict[str, object]:
        """
        Return static (side-effect free) configuration for dashboards/metrics.
        """
        return {
            "zones": [{"id": z.id, "name": z.name, "polygon": z.polygon} for z in self.zones],
            "queues": [{"id": q.id, "name": q.name, "polygon": q.polygon} for q in self.queues],
            "dwell_threshold_s": self.dwell_threshold_s,
            "target_class_ids": self.target_class_ids,
            "state_ttl_s": self.state_ttl_s,
        }

    def _is_target(self, det: Detection) -> bool:
        if self.target_class_ids is None:
            return True
        return det.cls in self.target_class_ids

    def _cleanup_state(self, ts: float) -> None:
        if self.state_ttl_s <= 0:
            return
        stale = []
        for tid, st in self._track_state.items():
            last_ts = st.get("last_ts")
            if isinstance(last_ts, (int, float)) and (ts - float(last_ts) > self.state_ttl_s):
                stale.append(tid)
        for tid in stale:
            self._track_state.pop(tid, None)

    def _update_track_state(self, track_id: int, zone_id: str | None, queue_id: str | None, ts: float) -> None:
        state = self._track_state.get(track_id)
        if state is None:
            self._track_state[track_id] = {
                "zone_id": zone_id,
                "zone_enter_ts": ts,
                "queue_id": queue_id,
                "queue_enter_ts": ts,
                "last_ts": ts,
            }
            return

        prev_zone = state.get("zone_id")
        if prev_zone != zone_id:
            state["zone_id"] = zone_id
            state["zone_enter_ts"] = ts

        prev_queue = state.get("queue_id")
        if prev_queue != queue_id:
            state["queue_id"] = queue_id
            state["queue_enter_ts"] = ts

        state["last_ts"] = ts

    def update(
        self,
        detections: Iterable[Detection],
        frame_shape: Tuple[int, int],
        ts: Optional[float] = None,
    ) -> Dict[str, object]:
        """
        Update analytics with current tracked detections.

        Returns:
            {
              "zones": [{id,name,count,dwell_seconds_avg}],
              "queues": [{id,name,count,avg_wait_s,max_wait_s}],
              "suspicious": [track_id, ...],
              "total_people": int
            }
        """
        ts = ts if ts is not None else time.time()
        self._cleanup_state(ts)
        h, w = frame_shape[:2]
        detections = [d for d in detections if d.track_id is not None and self._is_target(d)]

        # Precompute polygons in pixels
        zones_px = {z.id: _normalize_polygon_to_pixels(z.polygon, (h, w)) for z in self.zones}
        queues_px = {q.id: _normalize_polygon_to_pixels(q.polygon, (h, w)) for q in self.queues}

        zone_counts: Dict[str, Set[int]] = {z.id: set() for z in self.zones}
        zone_dwells: Dict[str, List[float]] = {z.id: [] for z in self.zones}
        queue_counts: Dict[str, Set[int]] = {q.id: set() for q in self.queues}
        queue_dwells: Dict[str, List[float]] = {q.id: [] for q in self.queues}
        suspicious: List[int] = []

        for det in detections:
            track_id = int(det.track_id)  # type: ignore[arg-type]
            cx = float((det.xyxy[0] + det.xyxy[2]) * 0.5)
            cy = float((det.xyxy[1] + det.xyxy[3]) * 0.5)
            point = (cx, cy)

            # Zone occupancy
            in_zone_id = None
            for z in self.zones:
                if _point_in_polygon(point, zones_px[z.id]):
                    in_zone_id = z.id
                    break

            # Queue/holding area occupancy (first matching)
            in_queue_id = None
            for q in self.queues:
                if _point_in_polygon(point, queues_px[q.id]):
                    in_queue_id = q.id
                    break

            self._update_track_state(track_id, in_zone_id, in_queue_id, ts)

            if in_zone_id:
                zone_counts[in_zone_id].add(track_id)
                state = self._track_state.get(track_id)
                if state:
                    dwell_s = ts - float(state.get("zone_enter_ts", ts))
                    zone_dwells[in_zone_id].append(dwell_s)
                    if dwell_s >= self.dwell_threshold_s and track_id not in suspicious:
                        suspicious.append(track_id)  # long dwell proxy

            if in_queue_id:
                queue_counts[in_queue_id].add(track_id)
                state = self._track_state.get(track_id)
                if state:
                    dwell_s = ts - float(state.get("queue_enter_ts", ts))
                    queue_dwells[in_queue_id].append(dwell_s)

        zone_out = []
        for z in self.zones:
            dwells = zone_dwells[z.id]
            zone_out.append(
                {
                    "id": z.id,
                    "name": z.name,
                    "count": len(zone_counts[z.id]),
                    "dwell_seconds_avg": float(np.mean(dwells)) if dwells else 0.0,
                }
            )

        queue_out = []
        for q in self.queues:
            dwells = queue_dwells[q.id]
            queue_out.append(
                {
                    "id": q.id,
                    "name": q.name,
                    "count": len(queue_counts[q.id]),
                    "avg_wait_s": float(np.mean(dwells)) if dwells else 0.0,
                    "max_wait_s": float(np.max(dwells)) if dwells else 0.0,
                }
            )

        return {
            "zones": zone_out,
            "queues": queue_out,
            "suspicious": suspicious,
            "total_people": len({int(d.track_id) for d in detections if d.track_id is not None}),
        }


