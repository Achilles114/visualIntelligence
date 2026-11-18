
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from core.tracking.models import Track
from core.spatial.processor import SpatialBatch
from core.spatial.motion import MotionBatch

class SceneEventType(str, Enum):
    TRACK_STARTED = "track_started"
    TRACK_LOST = "track_lost"


@dataclass(frozen=True, slots=True)
class SceneEvent:
    """An event observed during scene processing."""

    event_type: SceneEventType
    frame_id: int
    timestamp: datetime
    track_id: int
    label: str

    def __post_init__(self) -> None:
        if self.frame_id < 0:
            raise ValueError("frame_id cannot be negative")

        if self.track_id < 0:
            raise ValueError("track_id cannot be negative")

        if not self.label:
            raise ValueError("label cannot be empty")

        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")


@dataclass(frozen=True, slots=True)
class SceneSnapshot:
    """Structured representation of the observed scene."""

    frame_id: int
    sensor_id: str
    timestamp: datetime
    tracks: tuple[Track, ...]
    object_counts: tuple[tuple[str, int], ...]
    events: tuple[SceneEvent, ...]
    spatial: SpatialBatch | None = None
    motion: MotionBatch | None = None

    def __post_init__(self) -> None:
        if self.frame_id < 0:
            raise ValueError("frame_id cannot be negative")

        if not self.sensor_id:
            raise ValueError("sensor_id cannot be empty")

        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")

        if any(count < 0 for _, count in self.object_counts):
            raise ValueError("Object counts cannot be negative")

    @property
    def total_objects(self) -> int:
        return len(self.tracks)
