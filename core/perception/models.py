
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """
    Object location in image pixel coordinates.

    (x1, y1) = top-left
    (x2, y2) = bottom-right
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x1 < 0 or self.y1 < 0:
            raise ValueError("Bounding box coordinates cannot be negative")

        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("Bounding box must have positive width and height")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


@dataclass(frozen=True, slots=True)
class Detection:
    """One detected object."""

    class_id: int
    label: str
    confidence: float
    bbox: BoundingBox

    def __post_init__(self) -> None:
        if self.class_id < 0:
            raise ValueError("class_id cannot be negative")

        if not self.label:
            raise ValueError("label cannot be empty")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class DetectionBatch:
    """All detections produced for one camera frame."""

    frame_id: int
    sensor_id: str
    timestamp: datetime
    model_name: str
    inference_ms: float
    image_width: int
    image_height: int
    detections: tuple[Detection, ...]

    def __post_init__(self) -> None:
        if self.frame_id < 0:
            raise ValueError("frame_id cannot be negative")

        if not self.sensor_id or not self.model_name:
            raise ValueError("sensor_id and model_name are required")

        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")

        if self.inference_ms < 0:
            raise ValueError("inference_ms cannot be negative")

        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("Image dimensions must be positive")
