
from core.perception.models import BoundingBox
from core.spatial.models import (
    CoordinateSystem,
    Point2D,
    SpatialGeometry,
)


def calculate_geometry(
    bbox: BoundingBox,
    image_width: int,
    image_height: int,
) -> SpatialGeometry:
    """Calculate image-space and normalized object geometry."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")

    if bbox.x2 > image_width or bbox.y2 > image_height:
        raise ValueError("Bounding box exceeds image dimensions")

    width = bbox.width
    height = bbox.height

    center_x = (bbox.x1 + bbox.x2) / 2
    center_y = (bbox.y1 + bbox.y2) / 2

    normalized_width = width / image_width
    normalized_height = height / image_height

    return SpatialGeometry(
        center=Point2D(
            x=center_x,
            y=center_y,
            coordinate_system=CoordinateSystem.IMAGE,
        ),
        normalized_center=Point2D(
            x=center_x / image_width,
            y=center_y / image_height,
            coordinate_system=CoordinateSystem.NORMALIZED_IMAGE,
        ),
        width=width,
        height=height,
        normalized_width=normalized_width,
        normalized_height=normalized_height,
        area=width * height,
        normalized_area=normalized_width * normalized_height,
        aspect_ratio=width / height,
    )
