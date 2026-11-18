
from datetime import datetime, timezone

import numpy as np

from core.perception.models import BoundingBox, Detection, DetectionBatch
from core.perception.engine import PerceptionEngine
from core.scene.pipeline import ScenePipeline
from core.sensors.models import SensorFrame
from core.spatial.depth.models import (
    DepthMap,
    DepthRepresentation,
    DepthUnit,
)
from core.tracking.engine import TrackingEngine
from core.tracking.iou_tracker import IoUTracker


class FakeDetector:
    @property
    def model_name(self):
        return "fake-detector"

    def detect(self, frame):
        return DetectionBatch(
            frame_id=frame.frame_id,
            sensor_id=frame.sensor_id,
            timestamp=frame.timestamp,
            model_name=self.model_name,
            inference_ms=1.0,
            image_width=frame.width,
            image_height=frame.height,
            detections=(
                Detection(
                    class_id=0,
                    label="person",
                    confidence=0.95,
                    bbox=BoundingBox(100, 50, 300, 250),
                ),
            ),
        )


class FakeDepthEstimator:
    @property
    def estimator_name(self):
        return "fake-depth"

    def estimate(self, frame):
        return DepthMap(
            frame_id=frame.frame_id,
            sensor_id=frame.sensor_id,
            timestamp=frame.timestamp,
            values=np.ones(
                (frame.height, frame.width),
                dtype=np.float32,
            ),
            representation=DepthRepresentation.RELATIVE,
            unit=DepthUnit.RELATIVE,
            source=self.estimator_name,
        )


def create_frame():
    return SensorFrame(
        frame_id=1,
        sensor_id="test-camera",
        source="test",
        timestamp=datetime.now(timezone.utc),
        image=np.zeros((480, 640, 3), dtype=np.uint8),
    )


def create_pipeline(depth_estimator=None):
    perception = PerceptionEngine(FakeDetector())
    tracking = TrackingEngine(
        IoUTracker(
            iou_threshold=0.3,
            max_age=5,
            min_hits=1,
        )
    )

    return ScenePipeline(
        perception=perception,
        tracking=tracking,
        depth_estimator=depth_estimator,
    )


def test_scene_pipeline_works_without_depth_estimator():
    pipeline = create_pipeline()

    result = pipeline.process(create_frame())

    assert result.spatial is not None
    assert result.spatial.depth_map is None
    assert len(result.tracks) == 1


def test_scene_pipeline_integrates_depth_estimator():
    pipeline = create_pipeline(FakeDepthEstimator())

    result = pipeline.process(create_frame())

    assert result.spatial is not None
    assert result.spatial.depth_map is not None
    assert result.spatial.depth_map.source == "fake-depth"
    assert result.spatial.depth_map.frame_id == result.frame_id
    assert len(result.spatial.objects) == 1
