
import pytest

from core.reid.models import AppearanceFeature
from core.reid.similarity import cosine_similarity


def test_identical_features_have_similarity_one():
    feature = AppearanceFeature(
        values=(1.0, 2.0, 3.0),
        extractor_name="test",
    )

    assert cosine_similarity(feature, feature) == pytest.approx(1.0)


def test_different_features_have_expected_similarity():
    feature_a = AppearanceFeature(
        values=(1.0, 0.0),
        extractor_name="test",
    )

    feature_b = AppearanceFeature(
        values=(0.0, 1.0),
        extractor_name="test",
    )

    assert cosine_similarity(feature_a, feature_b) == pytest.approx(0.0)


def test_incompatible_extractors_raise_error():
    feature_a = AppearanceFeature(
        values=(1.0, 2.0),
        extractor_name="model-a",
    )

    feature_b = AppearanceFeature(
        values=(1.0, 2.0),
        extractor_name="model-b",
    )

    with pytest.raises(ValueError):
        cosine_similarity(feature_a, feature_b)


def test_different_dimensions_raise_error():
    feature_a = AppearanceFeature(
        values=(1.0, 2.0),
        extractor_name="test",
    )

    feature_b = AppearanceFeature(
        values=(1.0, 2.0, 3.0),
        extractor_name="test",
    )

    with pytest.raises(ValueError):
        cosine_similarity(feature_a, feature_b)


def test_zero_vectors_return_zero():
    feature_a = AppearanceFeature(
        values=(0.0, 0.0),
        extractor_name="test",
    )

    feature_b = AppearanceFeature(
        values=(1.0, 0.0),
        extractor_name="test",
    )

    assert cosine_similarity(feature_a, feature_b) == 0.0
