
from dataclasses import dataclass
from enum import Enum
from math import isfinite


class CoordinateSystem(str, Enum):
    IMAGE = "image"
    NORMALIZED_IMAGE = "normalized_image"
    CAMERA = "camera"
    WORLD = "world"


@dataclass(frozen=True, slots=True)
class Point2D:
    x: float
    y: float
    coordinate_system: CoordinateSystem

    def __post_init__(self) -> None:
        if not isfinite(self.x) or not isfinite(self.y):
            raise ValueError("Point coordinates must be finite")


@dataclass(frozen=True, slots=True)
class Point3D:
    x: float
    y: float
    z: float
    coordinate_system: CoordinateSystem

    def __post_init__(self) -> None:
        if not all(isfinite(v) for v in (self.x, self.y, self.z)):
            raise ValueError("Point coordinates must be finite")

        if self.coordinate_system in (
            CoordinateSystem.IMAGE,
            CoordinateSystem.NORMALIZED_IMAGE,
        ):
            raise ValueError(
                "Point3D requires a 3D coordinate system"
            )


@dataclass(frozen=True, slots=True)
class SpatialGeometry:
    """Image-space geometry associated with a tracked object."""

    center: Point2D
    normalized_center: Point2D
    width: float
    height: float
    normalized_width: float
    normalized_height: float
    area: float
    normalized_area: float
    aspect_ratio: float

    def __post_init__(self) -> None:
        if self.center.coordinate_system != CoordinateSystem.IMAGE:
            raise ValueError("center must use image coordinates")

        if (
            self.normalized_center.coordinate_system
            != CoordinateSystem.NORMALIZED_IMAGE
        ):
            raise ValueError(
                "normalized_center must use normalized coordinates"
            )

        if any(
            not isfinite(value) or value < 0
            for value in (
                self.width,
                self.height,
                self.normalized_width,
                self.normalized_height,
                self.area,
                self.normalized_area,
            )
        ):
            raise ValueError("Geometry dimensions must be finite and non-negative")

        if not isfinite(self.aspect_ratio) or self.aspect_ratio <= 0:
            raise ValueError("aspect_ratio must be finite and positive")


@dataclass(frozen=True, slots=True)
class SpatialMeasurement:
    """A spatial estimate with an explicit coordinate system."""

    position: Point3D
    confidence: float
    source: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

        if not self.source:
            raise ValueError("source cannot be empty")
