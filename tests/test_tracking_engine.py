
from datetime import datetime, timezone

from core.perception.models import (
    BoundingBox,
    Detection,
    DetectionBatch,
)
from core.tracking.engine import TrackingEngine
from core.tracking.iou_tracker import IoUTracker
from core.tracking.models import TrackBatch, TrackingInput


def create_batch(frame_id: int = 1) -> DetectionBatch:
    detection = Detection(
        class_id=0,
        label="person",
        confidence=0.95,
        bbox=BoundingBox(
            x1=100,
            y1=100,
            x2=200,
            y2=300,
        ),
    )

    return DetectionBatch(
        frame_id=frame_id,
        sensor_id="test-camera",
        timestamp=datetime.now(timezone.utc),
        model_name="fake-detector",
        inference_ms=2.0,
        image_width=640,
        image_height=480,
        detections=(detection,),
    )


def update_engine(
    engine: TrackingEngine,
    batch: DetectionBatch,
) -> TrackBatch:
    """Pass a DetectionBatch through the standardized tracking input."""

    return engine.process(
        TrackingInput(detections=batch)
    )


def test_tracking_engine_processes_detections():
    engine = TrackingEngine(IoUTracker())

    result = update_engine(engine, create_batch(1))

    assert isinstance(result, TrackBatch)
    assert result.frame_id == 1
    assert result.sensor_id == "test-camera"
    assert result.tracker_name == "iou-tracker"
    assert len(result.tracks) == 1
    assert result.tracks[0].track_id == 1


def test_tracking_engine_exposes_tracker_name():
    engine = TrackingEngine(IoUTracker())

    assert engine.tracker_name == "iou-tracker"


def test_tracking_engine_preserves_track_ids():
    engine = TrackingEngine(IoUTracker())

    first = update_engine(engine, create_batch(1))
    second = update_engine(engine, create_batch(2))

    assert first.tracks[0].track_id == 1
    assert second.tracks[0].track_id == 1


def test_tracking_engine_resets_tracker():
    engine = TrackingEngine(IoUTracker())

    first = update_engine(engine, create_batch(1))
    assert first.tracks[0].track_id == 1

    engine.reset()

    second = update_engine(engine, create_batch(1))
    assert second.tracks[0].track_id == 1
