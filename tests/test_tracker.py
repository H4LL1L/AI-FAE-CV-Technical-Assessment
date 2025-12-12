import numpy as np

from inference.detector import ByteTrackLite
from inference.utils import Detection


def test_bytetracklite_updates_and_ages_out():
    trk = ByteTrackLite(
        high_thresh=0.5,
        low_thresh=0.1,
        iou_thresh_high=0.3,
        iou_thresh_low=0.1,
        max_age=2,
        min_hits=1,
    )
    det = Detection(xyxy=np.array([0, 0, 10, 10], dtype=np.float32), score=0.9, cls=0)
    out1 = trk.update([det])
    assert len(out1) == 1
    tid = out1[0].track_id
    assert tid is not None

    # With max_age=2: after update time_since_update=0, then 1 and 2 survive,
    # and the 3rd step prunes the track (time_since_update=3 > max_age).
    trk.step_without_detections()  # 1
    trk.step_without_detections()  # 2
    trk.step_without_detections()  # 3
    assert tid not in [t.track_id for t in trk.tracks]


