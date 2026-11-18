
from datetime import datetime, timezone

import numpy as np
import pytest

from core.spatial.depth.models import (
    DepthMap,
    DepthRepresentation,
    DepthUnit,
)


def create_depth_map(
    values=None,
    representation=DepthRepresentation.RELATIVE,
    unit=DepthUnit.RELATIVE,
    valid_mask=None,
):
    if values is None:
        values = np.array(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=np.float32,
        )

    return DepthMap(
        frame_id=1,
        sensor_id="camera-001",
        timestamp=datetime.now(timezone.utc),
        values=values,
        representation=representation,
        unit=unit,
        source="test-depth-provider",
        valid_mask=valid_mask,
    )


def test_depth_map_dimensions():
    depth = create_depth_map()

    assert depth.width == 2
    assert depth.height == 2


def test_relative_depth_map():
    depth = create_depth_map()

    assert depth.representation == DepthRepresentation.RELATIVE
    assert depth.unit == DepthUnit.RELATIVE


def test_metric_depth_map():
    depth = create_depth_map(
        representation=DepthRepresentation.METRIC,
        unit=DepthUnit.METERS,
    )

    assert depth.representation == DepthRepresentation.METRIC
    assert depth.unit == DepthUnit.METERS


def test_depth_map_rejects_invalid_dimensions():
    with pytest.raises(ValueError):
        create_depth_map(values=np.array([1.0, 2.0]))


def test_depth_map_rejects_negative_values():
    with pytest.raises(ValueError):
        create_depth_map(
            values=np.array([[1.0, -2.0]], dtype=np.float32)
        )


def test_depth_map_rejects_metric_depth_with_relative_unit():
    with pytest.raises(ValueError):
        create_depth_map(
            representation=DepthRepresentation.METRIC,
            unit=DepthUnit.RELATIVE,
        )


def test_depth_map_validates_mask_dimensions():
    with pytest.raises(ValueError):
        create_depth_map(
            valid_mask=np.array([[True, False]], dtype=bool)
        )


def test_depth_map_allows_invalid_pixels_with_mask():
    values = np.array(
        [[1.0, 0.0], [2.0, 3.0]],
        dtype=np.float32,
    )
    mask = np.array(
        [[True, False], [True, True]],
        dtype=bool,
    )

    depth = create_depth_map(values=values, valid_mask=mask)

    assert depth.width == 2
    assert depth.height == 2


def test_depth_map_rejects_non_finite_valid_values():
    values = np.array(
        [[1.0, np.nan]],
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        create_depth_map(values=values)
