
from datetime import datetime, timezone

import numpy as np

from core.perception.models import (
    BoundingBox,
    Detection,
    DetectionBatch,
)
from core.reid.color_histogram import HSVColorHistogramExtractor
from core.sensors.models import SensorFrame
from core.tracking.engine import TrackingEngine
from core.tracking.models import TrackBatch
from core.tracking.pipeline import PerceptionTrackingPipeline
from core.tracking.reid_tracker import ReIDTracker


class FakePerception:
    """Returns detections at different locations across frames."""

    def process(self, frame: SensorFrame) -> DetectionBatch:
        x1 = 10 if frame.frame_id == 1 else 400

        detection = Detection(
            class_id=0,
            label="person",
            confidence=0.95,
            bbox=BoundingBox(
                x1=x1,
                y1=10,
                x2=x1 + 90,
                y2=100,
            ),
        )

        return DetectionBatch(
            frame_id=frame.frame_id,
            sensor_id=frame.sensor_id,
            timestamp=frame.timestamp,
            model_name="fake-detector",
            inference_ms=2.0,
            image_width=640,
            image_height=480,
            detections=(detection,),
        )


def create_frame(frame_id: int) -> SensorFrame:
    # Same visual appearance at both locations.
    image = np.full(
        (480, 640, 3),
        (0, 0, 255),
        dtype=np.uint8,
    )

    return SensorFrame(
        frame_id=frame_id,
        sensor_id="test-camera",
        source="test",
        timestamp=datetime.now(timezone.utc),
        image=image,
    )


def test_reid_pipeline_extracts_features_and_recovers_identity():
    tracker = ReIDTracker(
        iou_threshold=0.3,
        reid_similarity_threshold=0.85,
    )

    pipeline = PerceptionTrackingPipeline(
        perception=FakePerception(),
        tracking=TrackingEngine(tracker),
        appearance_extractor=HSVColorHistogramExtractor(),
    )

    first = pipeline.process(create_frame(1))
    second = pipeline.process(create_frame(2))

    assert isinstance(first, TrackBatch)
    assert isinstance(second, TrackBatch)

    assert len(first.tracks) == 1
    assert len(second.tracks) == 1

    assert first.tracks[0].track_id == 1
    assert second.tracks[0].track_id == 1

    assert tracker.gallery.identity_count == 1
