
from datetime import datetime, timezone

from core.perception.models import (
    BoundingBox,
    Detection,
    DetectionBatch,
)
from core.tracking.iou_tracker import IoUTracker
from core.tracking.models import TrackingInput


def create_detection(
    x1: float = 100,
    y1: float = 100,
    x2: float = 200,
    y2: float = 300,
    class_id: int = 0,
    label: str = "person",
    confidence: float = 0.95,
) -> Detection:
    return Detection(
        class_id=class_id,
        label=label,
        confidence=confidence,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
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


def update_tracker(
    tracker: IoUTracker,
    batch: DetectionBatch,
):
    return tracker.update(
        TrackingInput(detections=batch)
    )


def test_tracker_assigns_id_to_new_object():
    tracker = IoUTracker()

    result = update_tracker(
        tracker,
        create_batch(1, (create_detection(),)),
    )

    assert len(result.tracks) == 1
    assert result.tracks[0].track_id == 1
    assert result.tracks[0].label == "person"


def test_tracker_preserves_id_across_frames():
    tracker = IoUTracker()

    first = update_tracker(
        tracker,
        create_batch(1, (create_detection(),)),
    )

    second = update_tracker(
        tracker,
        create_batch(
            2,
            (
                create_detection(
                    x1=105,
                    y1=105,
                    x2=205,
                    y2=305,
                ),
            ),
        ),
    )

    assert first.tracks[0].track_id == 1
    assert second.tracks[0].track_id == 1
    assert second.tracks[0].age == 2
    assert second.tracks[0].hits == 2


def test_tracker_assigns_different_ids_to_new_objects():
    tracker = IoUTracker()

    first = update_tracker(
        tracker,
        create_batch(1, (create_detection(),)),
    )

    second = update_tracker(
        tracker,
        create_batch(
            2,
            (
                create_detection(),
                create_detection(
                    x1=350,
                    y1=100,
                    x2=450,
                    y2=300,
                ),
            ),
        ),
    )

    assert len(first.tracks) == 1
    assert len(second.tracks) == 2

    track_ids = {track.track_id for track in second.tracks}

    assert track_ids == {1, 2}


def test_tracker_does_not_match_different_classes():
    tracker = IoUTracker()

    first = update_tracker(
        tracker,
        create_batch(1, (create_detection(),)),
    )

    second = update_tracker(
        tracker,
        create_batch(
            2,
            (
                create_detection(
                    class_id=1,
                    label="bicycle",
                ),
            ),
        ),
    )

    assert first.tracks[0].track_id == 1
    assert len(second.tracks) == 1
    assert second.tracks[0].track_id == 2
    assert second.tracks[0].label == "bicycle"


def test_tracker_handles_temporarily_missing_object():
    tracker = IoUTracker(max_age=2)

    update_tracker(
        tracker,
        create_batch(1, (create_detection(),)),
    )

    missing = update_tracker(tracker, create_batch(2))

    assert len(missing.tracks) == 0

    recovered = update_tracker(
        tracker,
        create_batch(3, (create_detection(),)),
    )

    assert len(recovered.tracks) == 1
    assert recovered.tracks[0].track_id == 1
    assert recovered.tracks[0].hits == 2


def test_tracker_expires_old_tracks():
    tracker = IoUTracker(max_age=2)

    update_tracker(
        tracker,
        create_batch(1, (create_detection(),)),
    )

    update_tracker(tracker, create_batch(2))
    update_tracker(tracker, create_batch(3))
    update_tracker(tracker, create_batch(4))

    result = update_tracker(
        tracker,
        create_batch(5, (create_detection(),)),
    )

    assert len(result.tracks) == 1
    assert result.tracks[0].track_id == 2


def test_tracker_confirms_object_after_minimum_hits():
    tracker = IoUTracker(min_hits=2)

    first = update_tracker(
        tracker,
        create_batch(1, (create_detection(),)),
    )

    second = update_tracker(
        tracker,
        create_batch(2, (create_detection(),)),
    )

    assert first.tracks[0].is_confirmed is False
    assert second.tracks[0].is_confirmed is True


def test_tracker_reset_clears_tracking_state():
    tracker = IoUTracker()

    first = update_tracker(
        tracker,
        create_batch(1, (create_detection(),)),
    )

    assert first.tracks[0].track_id == 1

    tracker.reset()

    second = update_tracker(
        tracker,
        create_batch(1, (create_detection(),)),
    )

    assert second.tracks[0].track_id == 1


def test_tracker_returns_empty_batch_when_no_objects_exist():
    tracker = IoUTracker()

    result = update_tracker(tracker, create_batch(1))

    assert result.frame_id == 1
    assert result.sensor_id == "test-camera"
    assert result.tracks == ()
