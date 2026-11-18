
from datetime import datetime, timezone

import numpy as np
import pytest

from core.perception.models import BoundingBox
from core.spatial.depth.models import (
    DepthMap,
    DepthRepresentation,
    DepthUnit,
)
from core.spatial.processor import SpatialProcessor
from core.tracking.models import Track, TrackBatch


def create_track_batch(
    frame_id: int = 1,
    image_width: int = 640,
    image_height: int = 480,
) -> TrackBatch:
    track = Track(
        track_id=1,
        class_id=0,
        label="person",
        confidence=0.95,
        bbox=BoundingBox(100, 50, 300, 250),
        age=3,
        hits=3,
        is_confirmed=True,
    )

    return TrackBatch(
        frame_id=frame_id,
        sensor_id="test-camera",
        timestamp=datetime.now(timezone.utc),
        tracker_name="test-tracker",
        image_width=image_width,
        image_height=image_height,
        tracks=(track,),
    )


def create_depth_map(
    batch: TrackBatch,
    frame_id: int | None = None,
    values: np.ndarray | None = None,
) -> DepthMap:
    if values is None:
        values = np.ones(
            (batch.image_height, batch.image_width),
            dtype=np.float32,
        )

    return DepthMap(
        frame_id=batch.frame_id if frame_id is None else frame_id,
        sensor_id=batch.sensor_id,
        timestamp=batch.timestamp,
        values=values,
        representation=DepthRepresentation.RELATIVE,
        unit=DepthUnit.RELATIVE,
        source="test",
    )


def test_spatial_processor_calculates_geometry_for_tracks():
    batch = create_track_batch()

    result = SpatialProcessor().process(batch)

    assert result.frame_id == 1
    assert len(result.objects) == 1

    spatial_object = result.objects[0]

    assert spatial_object.track_id == 1
    assert spatial_object.label == "person"
    assert spatial_object.geometry.center.x == 200
    assert spatial_object.geometry.center.y == 150


def test_spatial_processor_handles_empty_scene():
    batch = TrackBatch(
        frame_id=1,
        sensor_id="test-camera",
        timestamp=datetime.now(timezone.utc),
        tracker_name="test-tracker",
        image_width=640,
        image_height=480,
        tracks=(),
    )

    result = SpatialProcessor().process(batch)

    assert result.objects == ()
    assert result.depth_map is None


def test_spatial_processor_attaches_depth_map():
    batch = create_track_batch()
    depth_map = create_depth_map(batch)

    result = SpatialProcessor().process(batch, depth_map)

    assert result.depth_map is depth_map
    assert result.frame_id == depth_map.frame_id
    assert result.sensor_id == depth_map.sensor_id


def test_spatial_processor_rejects_mismatched_depth_frame():
    batch = create_track_batch()
    depth_map = create_depth_map(
        batch,
        frame_id=batch.frame_id + 1,
    )

    with pytest.raises(ValueError, match="frame IDs"):
        SpatialProcessor().process(batch, depth_map)


def test_spatial_processor_rejects_mismatched_depth_sensor():
    batch = create_track_batch()
    depth_map = DepthMap(
        frame_id=batch.frame_id,
        sensor_id="different-camera",
        timestamp=batch.timestamp,
        values=np.ones(
            (batch.image_height, batch.image_width),
            dtype=np.float32,
        ),
        representation=DepthRepresentation.RELATIVE,
        unit=DepthUnit.RELATIVE,
        source="test",
    )

    with pytest.raises(ValueError, match="sensor IDs"):
        SpatialProcessor().process(batch, depth_map)


def test_spatial_processor_rejects_mismatched_depth_dimensions():
    batch = create_track_batch()
    depth_map = create_depth_map(
        batch,
        values=np.ones((10, 10), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="dimensions"):
        SpatialProcessor().process(batch, depth_map)
