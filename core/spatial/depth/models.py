
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import numpy as np


class DepthRepresentation(str, Enum):
    """Describes whether depth values are relative or real-world distances."""

    RELATIVE = "relative"
    METRIC = "metric"


class DepthUnit(str, Enum):
    RELATIVE = "relative"
    METERS = "meters"
    MILLIMETERS = "millimeters"


@dataclass(frozen=True, slots=True)
class DepthMap:
    """Depth information aligned with a particular sensor frame."""

    frame_id: int
    sensor_id: str
    timestamp: datetime
    values: np.ndarray
    representation: DepthRepresentation
    unit: DepthUnit
    source: str
    valid_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.frame_id < 0:
            raise ValueError("frame_id cannot be negative")

        if not np.isfinite(self.values).all():
            raise ValueError("Depth map values must be finite")

        if not self.sensor_id or not self.source:
            raise ValueError("sensor_id and source are required")

        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")

        if not isinstance(self.values, np.ndarray):
            raise TypeError("values must be a NumPy array")

        if self.values.ndim != 2 or self.values.size == 0:
            raise ValueError("Depth values must be a non-empty 2D array")

        if not np.issubdtype(self.values.dtype, np.number):
            raise TypeError("Depth values must have a numeric dtype")

        if self.representation == DepthRepresentation.RELATIVE:
            if self.unit != DepthUnit.RELATIVE:
                raise ValueError("Relative depth must use the relative unit")

        elif self.representation == DepthRepresentation.METRIC:
            if self.unit == DepthUnit.RELATIVE:
                raise ValueError("Metric depth requires a physical unit")

        if self.valid_mask is not None:
            if not isinstance(self.valid_mask, np.ndarray):
                raise TypeError("valid_mask must be a NumPy array")

            if self.valid_mask.shape != self.values.shape:
                raise ValueError("valid_mask must match the depth map dimensions")

            if self.valid_mask.dtype != np.bool_:
                raise TypeError("valid_mask must contain boolean values")

            valid_values = self.values[self.valid_mask]
        else:
            valid_values = self.values[np.isfinite(self.values)]

        if not np.all(np.isfinite(valid_values)):
            raise ValueError("Valid depth values must be finite")

        if np.any(valid_values <= 0):
            raise ValueError("Valid depth values must be positive")

    @property
    def width(self) -> int:
        return self.values.shape[1]

    @property
    def height(self) -> int:
        return self.values.shape[0]
