
from core.perception.engine import PerceptionEngine
from core.reid.base import AppearanceExtractor
from core.sensors.models import SensorFrame
from core.tracking.engine import TrackingEngine
from core.tracking.models import TrackBatch, TrackingInput


class PerceptionTrackingPipeline:
    """
    Combines object detection, appearance extraction and tracking.

    SensorFrame
        -> PerceptionEngine
        -> DetectionBatch
        -> Optional AppearanceExtractor
        -> TrackingInput
        -> TrackingEngine
        -> TrackBatch
    """

    def __init__(
        self,
        perception: PerceptionEngine,
        tracking: TrackingEngine,
        appearance_extractor: AppearanceExtractor | None = None,
    ) -> None:
        self.perception = perception
        self.tracking = tracking
        self.appearance_extractor = appearance_extractor

    def process(self, frame: SensorFrame) -> TrackBatch:
        """Run perception, optional appearance extraction and tracking."""

        detections = self.perception.process(frame)

        appearance_features = None

        if self.appearance_extractor is not None:
            appearance_features = tuple(
                self.appearance_extractor.extract(
                    frame.image,
                    detection.bbox,
                )
                for detection in detections.detections
            )

        tracking_input = TrackingInput(
            detections=detections,
            appearance_features=appearance_features,
        )

        return self.tracking.process(tracking_input)

    def reset(self) -> None:
        """Reset tracking state for a new processing session."""
        self.tracking.reset()
