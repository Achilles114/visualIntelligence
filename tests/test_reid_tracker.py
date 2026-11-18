
from datetime import datetime, timezone

from core.perception.models import (
    BoundingBox,
    Detection,
    DetectionBatch,
)
from core.reid.models import AppearanceFeature
from core.tracking.models import TrackingInput
from core.tracking.reid_tracker import ReIDTracker


def create_detection(
    x1: float = 100,
    y1: float = 100,
    x2: float = 200,
    y2: float = 300,
    class_id: int = 0,
    label: str = "person",
) -> Detection:
    return Detection(
        class_id=class_id,
        label=label,
        confidence=0.95,
        bbox=BoundingBox(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        ),
    )


def create_batch(
    frame_id: int,
    detections: tuple[Detection, ...] = (),
) -> DetectionBatch:
    return DetectionBatch(
        frame_id=frame_id,
        sensor_id="test-camera",
        timestamp=datetime.now(timezone.utc),
        model_name="fake-detector",
        inference_ms=2.0,
        image_width=640,
        image_height=480,
        detections=detections,
    )


def create_feature(
    values: tuple[float, ...] = (1.0, 0.0, 0.0),
) -> AppearanceFeature:
    return AppearanceFeature(
        values=values,
        extractor_name="test-extractor",
    )


def update_tracker(
    tracker: ReIDTracker,
    frame_id: int,
    detections: tuple[Detection, ...] = (),
    features: tuple[AppearanceFeature, ...] | None = None,
):
    return tracker.update(
        TrackingInput(
            detections=create_batch(frame_id, detections),
            appearance_features=features,
        )
    )


def test_reid_tracker_assigns_id_to_new_object():
    tracker = ReIDTracker()

    result = update_tracker(
        tracker,
        1,
        (create_detection(),),
        (create_feature(),),
    )

    assert len(result.tracks) == 1
    assert result.tracks[0].track_id == 1


def test_reid_tracker_preserves_id_across_frames():
    tracker = ReIDTracker()

    first = update_tracker(
        tracker,
        1,
        (create_detection(),),
        (create_feature(),),
    )

    second = update_tracker(
        tracker,
        2,
        (
            create_detection(
                x1=105,
                y1=105,
                x2=205,
                y2=305,
            ),
        ),
        (create_feature(),),
    )

    assert first.tracks[0].track_id == 1
    assert second.tracks[0].track_id == 1
    assert second.tracks[0].hits == 2


def test_reid_tracker_recovers_identity_after_track_expires():
    tracker = ReIDTracker(
        max_age=1,
        reid_similarity_threshold=0.85,
    )

    first = update_tracker(
        tracker,
        1,
        (create_detection(),),
        (create_feature(),),
    )

    assert first.tracks[0].track_id == 1

    # Object disappears.
    missing_1 = update_tracker(tracker, 2)
    assert missing_1.tracks == ()

    # Active track expires after exceeding max_age.
    missing_2 = update_tracker(tracker, 3)
    assert missing_2.tracks == ()
    assert 1 not in tracker._tracks

    # Same object reappears at a different location.
    recovered = update_tracker(
        tracker,
        4,
        (
            create_detection(
                x1=400,
                y1=100,
                x2=500,
                y2=300,
            ),
        ),
        (create_feature(),),
    )

    assert len(recovered.tracks) == 1
    assert recovered.tracks[0].track_id == 1


def test_reid_tracker_does_not_match_different_classes():
    tracker = ReIDTracker()

    first = update_tracker(
        tracker,
        1,
        (create_detection(),),
        (create_feature(),),
    )

    second = update_tracker(
        tracker,
        2,
        (
            create_detection(
                x1=400,
                y1=100,
                x2=500,
                y2=300,
                class_id=1,
                label="bicycle",
            ),
        ),
        (create_feature(),),
    )

    assert first.tracks[0].track_id == 1
    assert second.tracks[0].track_id == 2
    assert second.tracks[0].label == "bicycle"


def test_reid_tracker_prevents_duplicate_identity_in_same_frame():
    tracker = ReIDTracker()

    first = update_tracker(
        tracker,
        1,
        (create_detection(),),
        (create_feature(),),
    )

    assert first.tracks[0].track_id == 1

    # Two spatially separate detections have the same appearance.
    second = update_tracker(
        tracker,
        2,
        (
            create_detection(
                x1=100,
                y1=100,
                x2=200,
                y2=300,
            ),
            create_detection(
                x1=400,
                y1=100,
                x2=500,
                y2=300,
            ),
        ),
        (
            create_feature(),
            create_feature(),
        ),
    )

    track_ids = {
        track.track_id
        for track in second.tracks
    }

    assert len(second.tracks) == 2
    assert len(track_ids) == 2
    assert 1 in track_ids


def test_reid_tracker_works_without_appearance_features():
    tracker = ReIDTracker()

    first = update_tracker(
        tracker,
        1,
        (create_detection(),),
    )

    second = update_tracker(
        tracker,
        2,
        (
            create_detection(
                x1=105,
                y1=105,
                x2=205,
                y2=305,
            ),
        ),
    )

    assert first.tracks[0].track_id == 1
    assert second.tracks[0].track_id == 1


def test_reid_tracker_reset_clears_tracking_and_gallery():
    tracker = ReIDTracker()

    first = update_tracker(
        tracker,
        1,
        (create_detection(),),
        (create_feature(),),
    )

    assert first.tracks[0].track_id == 1
    assert tracker.gallery.identity_count == 1

    tracker.reset()

    assert tracker.gallery.identity_count == 0
    assert len(tracker._tracks) == 0

    second = update_tracker(
        tracker,
        1,
        (create_detection(),),
        (create_feature((0.0, 1.0, 0.0)),),
    )

    assert second.tracks[0].track_id == 1
