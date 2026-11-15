
from datetime import datetime, timezone

import pytest

from core.perception.models import (
    BoundingBox,
    Detection,
    DetectionBatch,
)


def test_bounding_box_dimensions():
    box = BoundingBox(10, 20, 110, 120)

    assert box.width == 100
    assert box.height == 100


def test_invalid_bounding_box():
    with pytest.raises(ValueError):
        BoundingBox(100, 100, 50, 50)


def test_detection_confidence_validation():
    with pytest.raises(ValueError):
        Detection(
            class_id=0,
            label="person",
            confidence=1.5,
            bbox=BoundingBox(10, 20, 100, 200),
        )


def test_detection_batch_creation():
    detection = Detection(
        class_id=0,
        label="person",
        confidence=0.95,
        bbox=BoundingBox(10, 20, 100, 200),
    )

    batch = DetectionBatch(
        frame_id=1,
        sensor_id="webcam-001",
        timestamp=datetime.now(timezone.utc),
        model_name="test-detector",
        inference_ms=12.5,
        image_width=640,
        image_height=480,
        detections=(detection,),
    )

    assert batch.frame_id == 1
    assert len(batch.detections) == 1
    assert batch.detections[0].label == "person"
