
from dataclasses import dataclass

from core.spatial.depth.models import DepthMap
from core.spatial.geometry import calculate_geometry
from core.spatial.models import SpatialGeometry
from core.tracking.models import TrackBatch


@dataclass(frozen=True, slots=True)
class SpatialObject:
    track_id: int
    label: str
    geometry: SpatialGeometry


@dataclass(frozen=True, slots=True)
class SpatialBatch:
    frame_id: int
    sensor_id: str
    objects: tuple[SpatialObject, ...]
    depth_map: DepthMap | None = None

    def __post_init__(self) -> None:
        if self.frame_id < 0:
            raise ValueError("frame_id cannot be negative")

        if not self.sensor_id:
            raise ValueError("sensor_id cannot be empty")

        if self.depth_map is not None:
            if self.depth_map.frame_id != self.frame_id:
                raise ValueError("Depth map frame_id does not match")

            if self.depth_map.sensor_id != self.sensor_id:
                raise ValueError("Depth map sensor_id does not match")


class SpatialProcessor:
    """Processes image geometry and optional depth information."""

    def process(
        self,
        batch: TrackBatch,
        depth_map: DepthMap | None = None,
    ) -> SpatialBatch:

        if depth_map is not None:
            if depth_map.frame_id != batch.frame_id:
                raise ValueError(
                    "Depth map and tracking batch frame IDs must match"
                )

            if depth_map.sensor_id != batch.sensor_id:
                raise ValueError(
                    "Depth map and tracking batch sensor IDs must match"
                )

            expected_shape = (
                batch.image_height,
                batch.image_width,
            )

            if depth_map.values.shape != expected_shape:
                raise ValueError(
                    "Depth map dimensions must match the tracking image"
                )

        objects = tuple(
            SpatialObject(
                track_id=track.track_id,
                label=track.label,
                geometry=calculate_geometry(
                    track.bbox,
                    batch.image_width,
                    batch.image_height,
                ),
            )
            for track in batch.tracks
        )

        return SpatialBatch(
            frame_id=batch.frame_id,
            sensor_id=batch.sensor_id,
            objects=objects,
            depth_map=depth_map,
        )
