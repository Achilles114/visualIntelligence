
from datetime import datetime, timezone

import numpy as np

from core.sensors.models import SensorFrame
from core.spatial.depth.base import (
    DepthSensor,
    MonocularDepthEstimator,
    StereoDepthEstimator,
)
from core.spatial.depth.models import (
    DepthMap,
    DepthRepresentation,
    DepthUnit,
)


class FakeMonocularEstimator:
    estimator_name = "fake-monocular"

    def estimate(self, frame: SensorFrame) -> DepthMap:
        return create_depth_map(frame.frame_id, frame.sensor_id)


class FakeStereoEstimator:
    estimator_name = "fake-stereo"

    def estimate(
        self,
        left_frame: SensorFrame,
        right_frame: SensorFrame,
    ) -> DepthMap:
        return create_depth_map(
            left_frame.frame_id,
            left_frame.sensor_id,
        )


class FakeDepthSensor:
    sensor_name = "fake-depth-sensor"

    def read_depth(self) -> DepthMap | None:
        return create_depth_map(0, self.sensor_name)


def create_frame(frame_id: int, sensor_id: str) -> SensorFrame:
    return SensorFrame(
        frame_id=frame_id,
        sensor_id=sensor_id,
        source="test",
        timestamp=datetime.now(timezone.utc),
        image=np.zeros((480, 640, 3), dtype=np.uint8),
    )


def create_depth_map(frame_id: int, sensor_id: str) -> DepthMap:
    return DepthMap(
        frame_id=frame_id,
        sensor_id=sensor_id,
        timestamp=datetime.now(timezone.utc),
        values=np.ones((480, 640), dtype=np.float32),
        representation=DepthRepresentation.RELATIVE,
        unit=DepthUnit.RELATIVE,
        source="test",
    )


def test_monocular_interface():
    estimator: MonocularDepthEstimator = FakeMonocularEstimator()

    result = estimator.estimate(create_frame(1, "camera-left"))

    assert result.frame_id == 1
    assert result.source == "test"


def test_stereo_interface():
    estimator: StereoDepthEstimator = FakeStereoEstimator()

    result = estimator.estimate(
        create_frame(1, "camera-left"),
        create_frame(1, "camera-right"),
    )

    assert result.frame_id == 1


def test_depth_sensor_interface():
    sensor: DepthSensor = FakeDepthSensor()

    result = sensor.read_depth()

    assert result is not None
    assert result.representation == DepthRepresentation.RELATIVE
