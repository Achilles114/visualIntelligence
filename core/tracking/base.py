from typing import Protocol

from core.tracking.models import TrackBatch, TrackingInput


class ObjectTracker(Protocol):
    """Common interface for interchangeable VIE tracking algorithms."""

    @property
    def tracker_name(self) -> str:
        ...

    def update(self, data: TrackingInput) -> TrackBatch:
        """Associate detections and optional appearance features."""
        ...

    def reset(self) -> None:
        """Clear tracking state."""
        ...