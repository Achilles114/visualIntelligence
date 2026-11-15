
from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass(frozen=True, slots=True)
class SensorFrame:
    """
    Standard frame representation used throughout VIE.

    Every visual sensor should produce frames in this format.
    """

    frame_id: int
    sensor_id: str
    source: str
    timestamp: datetime
    image: np.ndarray

    def __post_init__(self) -> None:
        if self.frame_id < 0:
            raise ValueError("frame_id cannot be negative")

        if not self.sensor_id:
            raise ValueError("sensor_id cannot be empty")

        if not self.source:
            raise ValueError("source cannot be empty")

        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")

        if not isinstance(self.image, np.ndarray):
            raise TypeError("image must be a NumPy array")

        if self.image.size == 0:
            raise ValueError("image cannot be empty")

        if self.image.ndim not in (2, 3):
            raise ValueError("image must be grayscale or multi-channel")

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def height(self) -> int:
        return self.image.shape[0]
