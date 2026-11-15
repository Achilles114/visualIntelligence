
from dataclasses import dataclass
from datetime import datetime

from core.perception.models import BoundingBox
from core.reid.models import AppearanceFeature
from core.perception.models import DetectionBatch


@dataclass(frozen=True, slots=True)
class Track:
    """Represents an object tracked across multiple video frames."""

    track_id: int
    class_id: int
    label: str
    confidence: float
    bbox: BoundingBox
    age: int
    hits: int
    is_confirmed: bool

    def __post_init__(self) -> None:
        if self.track_id < 0:
            raise ValueError("track_id cannot be negative")

        if self.class_id < 0:
            raise ValueError("class_id cannot be negative")

        if not self.label:
            raise ValueError("label cannot be empty")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

        if self.age < 1:
            raise ValueError("age must be at least 1")

        if self.hits < 1:
            raise ValueError("hits must be at least 1")

        if self.hits > self.age:
            raise ValueError("hits cannot exceed age")


@dataclass(frozen=True, slots=True)
class TrackBatch:
    """Contains all tracked objects for a particular frame."""

    frame_id: int
    sensor_id: str
    timestamp: datetime
    tracker_name: str
    image_width: int
    image_height: int
    tracks: tuple[Track, ...]

    def __post_init__(self) -> None:
        if self.frame_id < 0:
            raise ValueError("frame_id cannot be negative")

        if not self.sensor_id:
            raise ValueError("sensor_id cannot be empty")

        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")

        if not self.tracker_name:
            raise ValueError("tracker_name cannot be empty")

        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("Image dimensions must be positive")

        track_ids = [track.track_id for track in self.tracks]

        if len(track_ids) != len(set(track_ids)):
            raise ValueError("Track IDs must be unique within a batch")

@dataclass(frozen=True, slots=True)
class TrackingInput:
    """Standardized input consumed by tracking algorithms."""

    detections: DetectionBatch
    appearance_features: tuple[AppearanceFeature, ...] | None = None

    def __post_init__(self) -> None:
        if self.appearance_features is not None:
            if len(self.appearance_features) != len(
                self.detections.detections
            ):
                raise ValueError(
                    "Appearance features must correspond "
                    "to every detection"
                )

            if self.appearance_features:
                extractor_names = {
                    feature.extractor_name
                    for feature in self.appearance_features
                }

                if len(extractor_names) != 1:
                    raise ValueError(
                        "All features must use the same extractor"
                    )
