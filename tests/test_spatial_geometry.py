
import pytest

from core.perception.models import BoundingBox
from core.spatial.geometry import calculate_geometry
from core.spatial.models import CoordinateSystem


def test_calculate_geometry():
    bbox = BoundingBox(100, 50, 300, 250)

    result = calculate_geometry(bbox, 640, 480)

    assert result.center.x == 200
    assert result.center.y == 150

    assert result.normalized_center.x == pytest.approx(0.3125)
    assert result.normalized_center.y == pytest.approx(0.3125)

    assert result.width == 200
    assert result.height == 200
    assert result.area == 40000
    assert result.aspect_ratio == 1.0

    assert result.center.coordinate_system == CoordinateSystem.IMAGE
    assert (
        result.normalized_center.coordinate_system
        == CoordinateSystem.NORMALIZED_IMAGE
    )


def test_calculate_geometry_rejects_invalid_image_dimensions():
    bbox = BoundingBox(10, 10, 100, 100)

    with pytest.raises(ValueError):
        calculate_geometry(bbox, 0, 480)


def test_calculate_geometry_rejects_bbox_outside_image():
    bbox = BoundingBox(10, 10, 700, 100)

    with pytest.raises(ValueError):
        calculate_geometry(bbox, 640, 480)
