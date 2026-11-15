from typing import Protocol

from core.perception.models import DetectionBatch
from core.sensors.models import SensorFrame


class ObjectDetector(Protocol):
    """Common interface for all VIE object detection models."""

    @property
    def model_name(self) -> str:
        ...

    def detect(self, frame: SensorFrame) -> DetectionBatch:
        ...

class PerceptionService(Protocol):
    """Common interface for VIE perception services."""

    def process(self, frame: SensorFrame) -> DetectionBatch:
        ...
