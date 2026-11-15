
from datetime import datetime, timezone

import numpy as np

from core.perception.engine import PerceptionEngine
from core.perception.models import DetectionBatch
from core.sensors.models import SensorFrame


class FakeDetector:
    """Mock detector used to test the perception engine independently."""

    model_name = "fake-detector"

    def detect(self, frame: SensorFrame) -> DetectionBatch:
        return DetectionBatch(
            frame_id=frame.frame_id,
            sensor_id=frame.sensor_id,
            timestamp=frame.timestamp,
            model_name=self.model_name,
            inference_ms=2.0,
            image_width=frame.width,
            image_height=frame.height,
            detections=(),
        )


def test_perception_engine_processes_frame():
    # Arrange: create a sample sensor frame.
    frame = SensorFrame(
        frame_id=1,
        sensor_id="test-camera",
        source="test",
        timestamp=datetime.now(timezone.utc),
        image=np.zeros((480, 640, 3), dtype=np.uint8),
    )

    detector = FakeDetector()
    engine = PerceptionEngine(detector)

    # Act: process the frame through the perception engine.
    result = engine.process(frame)

    # Assert: verify the returned detection batch.
    assert isinstance(result, DetectionBatch)
    assert result.frame_id == 1
    assert result.sensor_id == "test-camera"
    assert result.model_name == "fake-detector"
    assert result.inference_ms == 2.0
    assert result.image_width == 640
    assert result.image_height == 480
    assert result.detections == ()


def test_perception_engine_exposes_detector_model_name():
    engine = PerceptionEngine(FakeDetector())

    assert engine.model_name == "fake-detector"
