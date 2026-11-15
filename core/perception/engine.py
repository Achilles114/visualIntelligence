
from core.perception.base import ObjectDetector
from core.perception.models import DetectionBatch
from core.sensors.models import SensorFrame


class PerceptionEngine:
    """Coordinates perception using replaceable detection models."""

    def __init__(self, detector: ObjectDetector) -> None:
        self.detector = detector

    @property
    def model_name(self) -> str:
        return self.detector.model_name

    def process(self, frame: SensorFrame) -> DetectionBatch:
        return self.detector.detect(frame)
