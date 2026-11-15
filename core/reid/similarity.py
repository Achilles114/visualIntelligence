
import numpy as np

from core.reid.models import AppearanceFeature


def cosine_similarity(
    feature_a: AppearanceFeature,
    feature_b: AppearanceFeature,
) -> float:
    """Calculate cosine similarity between two appearance features."""

    if feature_a.extractor_name != feature_b.extractor_name:
        raise ValueError(
            "Cannot compare features from different extractors"
        )

    if len(feature_a.values) != len(feature_b.values):
        raise ValueError("Feature dimensions must match")

    vector_a = np.asarray(feature_a.values, dtype=np.float32)
    vector_b = np.asarray(feature_b.values, dtype=np.float32)

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    similarity = np.dot(vector_a, vector_b) / (norm_a * norm_b)

    return float(np.clip(similarity, -1.0, 1.0))
