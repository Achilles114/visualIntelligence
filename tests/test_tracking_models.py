
from datetime import datetime, timezone

import pytest

from core.perception.models import BoundingBox
from core.tracking.models import Track, TrackBatch


def create_track(
    track_id: int = 1,
    age: int = 3,
    hits: int = 2,
) -> Track:
    return Track(
        track_id=track_id,
        class_id=0,
        label="person",
        confidence=0.95,
        bbox=BoundingBox(
            x1=100,
            y1=100,
            x2=200,
            y2=300,
        ),
        age=age,
        hits=hits,
        is_confirmed=True,
    )


def create_track_batch(
    tracks: tuple[Track, ...] = (),
) -> TrackBatch:
    return TrackBatch(
        frame_id=1,
        sensor_id="test-camera",
        timestamp=datetime.now(timezone.utc),
        tracker_name="test-tracker",
        image_width=640,
        image_height=480,
        tracks=tracks,
    )


def test_track_creation():
    track = create_track()

    assert track.track_id == 1
    assert track.label == "person"
    assert track.confidence == 0.95
    assert track.age == 3
    assert track.hits == 2
    assert track.is_confirmed is True


def test_track_rejects_negative_id():
    with pytest.raises(ValueError, match="track_id"):
        create_track(track_id=-1)


def test_track_rejects_invalid_age():
    with pytest.raises(ValueError, match="age"):
        create_track(age=0)


def test_track_rejects_hits_greater_than_age():
    with pytest.raises(ValueError, match="hits cannot exceed age"):
        create_track(age=2, hits=3)


def test_track_batch_creation():
    track = create_track()
    batch = create_track_batch(tracks=(track,))

    assert batch.frame_id == 1
    assert batch.sensor_id == "test-camera"
    assert batch.tracker_name == "test-tracker"
    assert len(batch.tracks) == 1
    assert batch.tracks[0].track_id == 1


def test_track_batch_rejects_duplicate_track_ids():
    track1 = create_track(track_id=1)
    track2 = create_track(track_id=1)

    with pytest.raises(ValueError, match="Track IDs must be unique"):
        create_track_batch(tracks=(track1, track2))


def test_track_batch_rejects_naive_timestamp():
    with pytest.raises(ValueError, match="timezone-aware"):
        TrackBatch(
            frame_id=1,
            sensor_id="test-camera",
            timestamp=datetime.now(),
            tracker_name="test-tracker",
            image_width=640,
            image_height=480,
            tracks=(),
        )
