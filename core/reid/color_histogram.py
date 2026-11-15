
import cv2
import numpy as np

from core.perception.models import BoundingBox
from core.reid.models import AppearanceFeature


class HSVColorHistogramExtractor:
    """Extracts lightweight HSV colour histogram features."""

    def __init__(
        self,
        bins: tuple[int, int, int] = (16, 8, 8),
    ) -> None:
        if len(bins) != 3 or any(value < 1 for value in bins):
            raise ValueError("bins must contain three positive integers")

        self.bins = bins

    @property
    def extractor_name(self) -> str:
        return "hsv-color-histogram"

    def extract(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
    ) -> AppearanceFeature:

        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("image cannot be empty")

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected grayscale, BGR or BGRA image")

        height, width = image.shape[:2]

        x1 = max(0, min(width, int(bbox.x1)))
        y1 = max(0, min(height, int(bbox.y1)))
        x2 = max(0, min(width, int(np.ceil(bbox.x2))))
        y2 = max(0, min(height, int(np.ceil(bbox.y2))))

        if x2 <= x1 or y2 <= y1:
            raise ValueError("Bounding box does not contain a valid image crop")

        crop = image[y1:y2, x1:x2]

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        histogram = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            None,
            list(self.bins),
            [0, 180, 0, 256, 0, 256],
        )

        histogram = cv2.normalize(
            histogram,
            None,
            alpha=1.0,
            norm_type=cv2.NORM_L1,
        )

        values = tuple(
            float(value)
            for value in histogram.flatten()
        )

        return AppearanceFeature(
            values=values,
            extractor_name=self.extractor_name,
        )
