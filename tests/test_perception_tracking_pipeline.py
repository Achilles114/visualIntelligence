
from datetime import datetime, timezone

import numpy as np

from core.perception.engine import PerceptionEngine
from core.perception.models import (
    BoundingBox,
    Detection,
    DetectionBatch,
)

from core.sensors.models import SensorFrame
from core.tracking.engine import TrackingEngine
from core.tracking.iou_tracker import IoUTracker
from core.tracking.pipeline import PerceptionTrackingPipeline


class FakeDetector:
    """Deterministic detector for integration testing."""

    model_name = "fake-detector"

    def detect(self, frame: SensorFrame) -> DetectionBatch:
        detection = Detection(
            class_id=0,
            label="person",
            confidence=0.95,
            bbox=BoundingBox(
                x1=100 + frame.frame_id * 5,
                y1=100,
                x2=200 + frame.frame_id * 5,
                y2=300,
            ),
        )

        return DetectionBatch(
            frame_id=frame.frame_id,
            sensor_id=frame.sensor_id,
            timestamp=frame.timestamp,
            model_name=self.model_name,
            inference_ms=2.0,
            image_width=frame.width,
            image_height=frame.height,
            detections=(detection,),
        )


def create_frame(frame_id: int) -> SensorFrame:
    return SensorFrame(
        frame_id=frame_id,
        sensor_id="test-camera",
        source="test",
        timestamp=datetime.now(timezone.utc),
        image=np.zeros((480, 640, 3), dtype=np.uint8),
    )


def create_pipeline() -> PerceptionTrackingPipeline:
    perception = PerceptionEngine(FakeDetector())
    tracking = TrackingEngine(IoUTracker())

    return PerceptionTrackingPipeline(
        perception=perception,
        tracking=tracking,
    )


def test_pipeline_processes_sensor_frame():
    pipeline = create_pipeline()

    result = pipeline.process(create_frame(1))

    assert result.frame_id == 1
    assert result.sensor_id == "test-camera"
    assert result.tracker_name == "iou-tracker"
    assert len(result.tracks) == 1
    assert result.tracks[0].label == "person"


def test_pipeline_preserves_track_id_across_frames():
    pipeline = create_pipeline()

    first = pipeline.process(create_frame(1))
    second = pipeline.process(create_frame(2))

    assert first.tracks[0].track_id == 1
    assert second.tracks[0].track_id == 1
    assert second.tracks[0].age == 2
    assert second.tracks[0].hits == 2


def test_pipeline_confirms_track_across_frames():
    pipeline = create_pipeline()

    first = pipeline.process(create_frame(1))
    second = pipeline.process(create_frame(2))

    assert first.tracks[0].is_confirmed is False
    assert second.tracks[0].is_confirmed is True


def test_pipeline_reset_clears_tracking_state():
    pipeline = create_pipeline()

    first = pipeline.process(create_frame(1))
    assert first.tracks[0].track_id == 1

    pipeline.reset()

    second = pipeline.process(create_frame(1))
    assert second.tracks[0].track_id == 1
