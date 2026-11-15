
from dataclasses import dataclass

from core.reid.models import AppearanceFeature
from core.reid.similarity import cosine_similarity


@dataclass(frozen=True, slots=True)
class ReIDMatch:
    """A candidate identity returned by the appearance gallery."""

    track_id: int
    similarity: float


class ReIDGallery:
    """
    Stores appearance observations for previously tracked identities.

    The gallery does not assign IDs. It only suggests matching identities.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        max_samples_per_identity: int = 5,
        max_identities: int = 1000,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1")

        if max_samples_per_identity < 1:
            raise ValueError("max_samples_per_identity must be at least 1")

        if max_identities < 1:
            raise ValueError("max_identities must be at least 1")

        self.similarity_threshold = similarity_threshold
        self.max_samples_per_identity = max_samples_per_identity
        self.max_identities = max_identities

        self._features: dict[int, list[AppearanceFeature]] = {}

    def add(
        self,
        track_id: int,
        feature: AppearanceFeature,
    ) -> None:
        """Store a new appearance observation for an identity."""

        if track_id < 0:
            raise ValueError("track_id cannot be negative")

        samples = self._features.get(track_id)

        if samples is None:
            if len(self._features) >= self.max_identities:
                oldest_track_id = next(iter(self._features))
                del self._features[oldest_track_id]

            samples = []
            self._features[track_id] = samples

        elif samples[0].extractor_name != feature.extractor_name:
            raise ValueError(
                "Cannot mix different extractors for one identity"
            )

        samples.append(feature)

        if len(samples) > self.max_samples_per_identity:
            samples.pop(0)

    def match(
        self,
        feature: AppearanceFeature,
        exclude_track_ids: set[int] | None = None,
    ) -> ReIDMatch | None:
        """
        Find the most similar stored identity.

        Returns None when no candidate reaches the threshold.
        """

        excluded = exclude_track_ids or set()

        best_match: ReIDMatch | None = None

        for track_id, samples in self._features.items():
            if track_id in excluded:
                continue

            best_identity_similarity = max(
                (
                    cosine_similarity(feature, stored_feature)
                    for stored_feature in samples
                    if (
                        feature.extractor_name
                        == stored_feature.extractor_name
                        and len(feature.values) == len(stored_feature.values)
                    )
                ),
                default=-1.0,
            )

            if best_identity_similarity < self.similarity_threshold:
                continue

            if (
                best_match is None
                or best_identity_similarity > best_match.similarity
            ):
                best_match = ReIDMatch(
                    track_id=track_id,
                    similarity=best_identity_similarity,
                )

        return best_match

    def remove(self, track_id: int) -> None:
        """Remove one identity from the gallery."""

        self._features.pop(track_id, None)

    def clear(self) -> None:
        """Remove all stored identities."""

        self._features.clear()

    @property
    def identity_count(self) -> int:
        return len(self._features)
