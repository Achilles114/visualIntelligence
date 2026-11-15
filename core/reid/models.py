
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class AppearanceFeature:
    """Immutable visual descriptor extracted from an object."""

    values: tuple[float, ...]
    extractor_name: str

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("Appearance feature cannot be empty")

        if not self.extractor_name:
            raise ValueError("extractor_name cannot be empty")

        if not all(isfinite(value) for value in self.values):
            raise ValueError("Feature values must be finite")
