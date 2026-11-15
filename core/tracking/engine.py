
from core.tracking.base import ObjectTracker
from core.tracking.models import TrackBatch, TrackingInput


class TrackingEngine:
    """Coordinates object tracking using a replaceable tracker."""

    def __init__(self, tracker: ObjectTracker) -> None:
        self.tracker = tracker

    @property
    def tracker_name(self) -> str:
        return self.tracker.tracker_name

    def process(self, data: TrackingInput) -> TrackBatch:
        """Process tracking input and return tracked objects."""
        return self.tracker.update(data)

    def reset(self) -> None:
        """Reset the tracking state."""
        self.tracker.reset()
