
import pytest

from core.reid.gallery import ReIDGallery
from core.reid.models import AppearanceFeature


def make_feature(values):
    return AppearanceFeature(
        values=tuple(values),
        extractor_name="test-extractor",
    )


def test_gallery_matches_existing_identity():
    gallery = ReIDGallery(similarity_threshold=0.8)

    gallery.add(1, make_feature([1.0, 0.0]))

    result = gallery.match(make_feature([0.99, 0.01]))

    assert result is not None
    assert result.track_id == 1
    assert result.similarity > 0.99


def test_gallery_rejects_dissimilar_identity():
    gallery = ReIDGallery(similarity_threshold=0.8)

    gallery.add(1, make_feature([1.0, 0.0]))

    result = gallery.match(make_feature([0.0, 1.0]))

    assert result is None


def test_gallery_returns_best_matching_identity():
    gallery = ReIDGallery(similarity_threshold=0.5)

    gallery.add(1, make_feature([1.0, 0.0]))
    gallery.add(2, make_feature([0.8, 0.6]))

    result = gallery.match(make_feature([1.0, 0.0]))

    assert result is not None
    assert result.track_id == 1


def test_gallery_excludes_specified_identities():
    gallery = ReIDGallery(similarity_threshold=0.5)

    gallery.add(1, make_feature([1.0, 0.0]))
    gallery.add(2, make_feature([0.0, 1.0]))

    result = gallery.match(
        make_feature([1.0, 0.0]),
        exclude_track_ids={1},
    )

    assert result is None


def test_gallery_limits_samples_per_identity():
    gallery = ReIDGallery(max_samples_per_identity=2)

    gallery.add(1, make_feature([1.0, 0.0]))
    gallery.add(1, make_feature([0.0, 1.0]))
    gallery.add(1, make_feature([-1.0, 0.0]))

    assert gallery.identity_count == 1
    assert len(gallery._features[1]) == 2


def test_gallery_limits_number_of_identities():
    gallery = ReIDGallery(max_identities=2)

    gallery.add(1, make_feature([1.0, 0.0]))
    gallery.add(2, make_feature([0.0, 1.0]))
    gallery.add(3, make_feature([-1.0, 0.0]))

    assert gallery.identity_count == 2
    assert 1 not in gallery._features
    assert 2 in gallery._features
    assert 3 in gallery._features


def test_gallery_removes_identity():
    gallery = ReIDGallery()

    gallery.add(1, make_feature([1.0, 0.0]))
    gallery.remove(1)

    assert gallery.identity_count == 0


def test_gallery_clear_removes_everything():
    gallery = ReIDGallery()

    gallery.add(1, make_feature([1.0, 0.0]))
    gallery.add(2, make_feature([0.0, 1.0]))

    gallery.clear()

    assert gallery.identity_count == 0


def test_gallery_rejects_mixed_extractors_for_same_identity():
    gallery = ReIDGallery()

    gallery.add(
        1,
        AppearanceFeature((1.0, 0.0), "extractor-a"),
    )

    with pytest.raises(ValueError):
        gallery.add(
            1,
            AppearanceFeature((1.0, 0.0), "extractor-b"),
        )


def test_gallery_rejects_invalid_track_id():
    gallery = ReIDGallery()

    with pytest.raises(ValueError):
        gallery.add(-1, make_feature([1.0, 0.0]))
