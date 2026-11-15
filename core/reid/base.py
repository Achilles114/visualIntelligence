
from typing import Protocol

import numpy as np

from core.perception.models import BoundingBox
from core.reid.models import AppearanceFeature


class AppearanceExtractor(Protocol):
    """Common interface for interchangeable appearance extractors."""

    @property
    def extractor_name(self) -> str:
        ...

    def extract(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
    ) -> AppearanceFeature:
        """Extract an appearance descriptor from an object crop."""
        ...
