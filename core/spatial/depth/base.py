
from typing import Protocol

from core.sensors.models import SensorFrame
from core.spatial.depth.models import DepthMap


class MonocularDepthEstimator(Protocol):
    """Estimates relative or metric depth from a single RGB frame."""

    @property
    def estimator_name(self) -> str:
        ...

    def estimate(self, frame: SensorFrame) -> DepthMap:
        ...


class StereoDepthEstimator(Protocol):
    """Estimates depth using a synchronized pair of camera frames."""

    @property
    def estimator_name(self) -> str:
        ...

    def estimate(
        self,
        left_frame: SensorFrame,
        right_frame: SensorFrame,
    ) -> DepthMap:
        ...


class DepthSensor(Protocol):
    """Interface for hardware that directly supplies depth measurements."""

    @property
    def sensor_name(self) -> str:
        ...

    def read_depth(self) -> DepthMap | None:
        ...
