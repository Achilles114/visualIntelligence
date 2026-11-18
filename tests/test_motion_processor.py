
from datetime import datetime, timedelta, timezone

import pytest

from core.perception.models import BoundingBox
from core.spatial.motion import MotionProcessor, MotionStatus
from core.tracking.models import Track, TrackBatch


def create_batch(
    frame_id: int,
    timestamp: datetime,
    x: float = 100,
    y: float = 100,
) -> TrackBatch:
    track = Track(
        track_id=1,
        class_id=0,
        label="person",
        confidence=0.95,
        bbox=BoundingBox(
            x1=x,
            y1=y,
            x2=x + 100,
            y2=y + 200,
        ),
        age=frame_id,
        hits=frame_id,
        is_confirmed=True,
    )

    return TrackBatch(
        frame_id=frame_id,
        sensor_id="test-camera",
        timestamp=timestamp,
        tracker_name="test-tracker",
        image_width=640,
        image_height=480,
        tracks=(track,),
    )


def test_first_observation_has_unknown_motion():
    processor = MotionProcessor()

    batch = create_batch(
        1,
        datetime.now(timezone.utc),
    )

    result = processor.process(batch)

    motion = result.objects[0]

    assert motion.status == MotionStatus.UNKNOWN
    assert motion.speed_px_per_second is None
    assert motion.direction_degrees is None


def test_motion_processor_calculates_horizontal_velocity():
    processor = MotionProcessor()

    timestamp = datetime.now(timezone.utc)

    processor.process(create_batch(1, timestamp, x=100))

    result = processor.process(
        create_batch(
            2,
            timestamp + timedelta(seconds=0.5),
            x=110,
        )
    )

    motion = result.objects[0]

    assert motion.displacement_x == pytest.approx(10)
    assert motion.displacement_y == pytest.approx(0)
    assert motion.velocity_x == pytest.approx(20)
    assert motion.velocity_y == pytest.approx(0)
    assert motion.speed_px_per_second == pytest.approx(20)
    assert motion.direction_degrees == pytest.approx(0)
    assert motion.status == MotionStatus.MOVING


def test_motion_processor_calculates_vertical_direction():
    processor = MotionProcessor()

    timestamp = datetime.now(timezone.utc)

    processor.process(create_batch(1, timestamp, y=100))

    result = processor.process(
        create_batch(
            2,
            timestamp + timedelta(seconds=1),
            y=110,
        )
    )

    motion = result.objects[0]

    assert motion.direction_degrees == pytest.approx(90)


def test_motion_processor_detects_stationary_object():
    processor = MotionProcessor(
        stationary_threshold_px_per_second=8,
    )

    timestamp = datetime.now(timezone.utc)

    processor.process(create_batch(1, timestamp, x=100))

    result = processor.process(
        create_batch(
            2,
            timestamp + timedelta(seconds=1),
            x=102,
        )
    )

    assert result.objects[0].status == MotionStatus.STATIONARY


def test_motion_processor_resets_state():
    processor = MotionProcessor()

    timestamp = datetime.now(timezone.utc)

    processor.process(create_batch(1, timestamp))

    processor.reset()

    result = processor.process(
        create_batch(
            2,
            timestamp + timedelta(seconds=1),
            x=200,
        )
    )

    assert result.objects[0].status == MotionStatus.UNKNOWN


def test_motion_processor_rejects_negative_threshold():
    with pytest.raises(ValueError):
        MotionProcessor(
            stationary_threshold_px_per_second=-1,
        )
