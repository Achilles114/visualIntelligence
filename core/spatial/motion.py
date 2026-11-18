
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from math import atan2, degrees, hypot, isfinite

from core.spatial.models import CoordinateSystem, Point2D
from core.tracking.models import TrackBatch


class MotionStatus(str, Enum):
    UNKNOWN = "unknown"
    STATIONARY = "stationary"
    MOVING = "moving"


@dataclass(frozen=True, slots=True)
class MotionObject:
    """Image-space movement information for a tracked object."""

    track_id: int
    label: str
    position: Point2D
    displacement_x: float | None
    displacement_y: float | None
    velocity_x: float | None
    velocity_y: float | None
    speed_px_per_second: float | None
    direction_degrees: float | None
    status: MotionStatus


@dataclass(frozen=True, slots=True)
class MotionBatch:
    """Motion estimates for all currently tracked objects."""

    frame_id: int
    sensor_id: str
    timestamp: datetime
    objects: tuple[MotionObject, ...]


@dataclass(frozen=True, slots=True)
class _PreviousObservation:
    x: float
    y: float
    timestamp: datetime


class MotionProcessor:
    """
    Estimates object movement using consecutive tracked positions.

    Velocity and speed are expressed in pixels per second.
    Direction is measured clockwise in image coordinates:
    0 degrees = right, 90 degrees = down.
    """

    def __init__(
        self,
        stationary_threshold_px_per_second: float = 8.0,
    ) -> None:
        if (
            not isfinite(stationary_threshold_px_per_second)
            or stationary_threshold_px_per_second < 0
        ):
            raise ValueError(
                "stationary_threshold_px_per_second "
                "must be finite and non-negative"
            )

        self.stationary_threshold = (
            stationary_threshold_px_per_second
        )

        self._previous: dict[int, _PreviousObservation] = {}

    def reset(self) -> None:
        """Clear all previous movement observations."""
        self._previous.clear()

    def process(self, batch: TrackBatch) -> MotionBatch:
        objects: list[MotionObject] = []
        current_observations: dict[int, _PreviousObservation] = {}

        for track in batch.tracks:
            x = (track.bbox.x1 + track.bbox.x2) / 2
            y = (track.bbox.y1 + track.bbox.y2) / 2

            position = Point2D(
                x=x,
                y=y,
                coordinate_system=CoordinateSystem.IMAGE,
            )

            previous = self._previous.get(track.track_id)

            displacement_x = None
            displacement_y = None
            velocity_x = None
            velocity_y = None
            speed = None
            direction = None
            status = MotionStatus.UNKNOWN

            if previous is not None:
                elapsed = (
                    batch.timestamp - previous.timestamp
                ).total_seconds()

                if elapsed > 0:
                    displacement_x = x - previous.x
                    displacement_y = y - previous.y

                    velocity_x = displacement_x / elapsed
                    velocity_y = displacement_y / elapsed

                    speed = hypot(velocity_x, velocity_y)

                    direction = (
                        degrees(atan2(displacement_y, displacement_x))
                        % 360.0
                    )

                    status = (
                        MotionStatus.MOVING
                        if speed > self.stationary_threshold
                        else MotionStatus.STATIONARY
                    )

            current_observations[track.track_id] = (
                _PreviousObservation(
                    x=x,
                    y=y,
                    timestamp=batch.timestamp,
                )
            )

            objects.append(
                MotionObject(
                    track_id=track.track_id,
                    label=track.label,
                    position=position,
                    displacement_x=displacement_x,
                    displacement_y=displacement_y,
                    velocity_x=velocity_x,
                    velocity_y=velocity_y,
                    speed_px_per_second=speed,
                    direction_degrees=direction,
                    status=status,
                )
            )

        # Only retain objects observed in this frame.
        # A returning object starts with an unknown motion estimate.
        self._previous = current_observations

        return MotionBatch(
            frame_id=batch.frame_id,
            sensor_id=batch.sensor_id,
            timestamp=batch.timestamp,
            objects=tuple(objects),
        )
