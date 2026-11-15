
from datetime import datetime, timezone

import numpy as np

from core.sensors.models import SensorFrame
from runtime.frame_pipeline import FramePipeline, PipelineOutput


class FakeCamera:
    def __init__(self):
        self.frame_id = 0
        self.released = False

    def open(self):
        pass

    def read_frame(self):
        frame = SensorFrame(
            frame_id=self.frame_id,
            sensor_id="test-camera",
            source="test",
            timestamp=datetime.now(timezone.utc),
            image=np.zeros((480, 640, 3), dtype=np.uint8),
        )

        self.frame_id += 1
        return frame

    def release(self):
        self.released = True


def test_pipeline_processes_frames():
    camera = FakeCamera()
    received_outputs = []

    def processor(frame):
        return f"processed-{frame.frame_id}"

    pipeline = FramePipeline(
        camera=camera,
        processor=processor,
        result_handler=received_outputs.append,
    )

    stats = pipeline.run(max_frames=5)

    assert stats.frames_processed == 5
    assert len(received_outputs) == 5
    assert camera.released is True

    assert all(
        isinstance(output, PipelineOutput)
        for output in received_outputs
    )

    assert received_outputs[0].frame.frame_id == 0
    assert received_outputs[0].result == "processed-0"


def test_pipeline_stops_at_max_frames():
    camera = FakeCamera()
    processed_frames = []

    pipeline = FramePipeline(
        camera=camera,
        processor=lambda frame: processed_frames.append(frame.frame_id),
    )

    stats = pipeline.run(max_frames=3)

    assert stats.frames_processed == 3
    assert processed_frames == [0, 1, 2]
    assert camera.released is True


def test_pipeline_handler_can_stop_pipeline():
    camera = FakeCamera()
    received_outputs = []

    def handler(output):
        received_outputs.append(output)
        return len(received_outputs) >= 3

    pipeline = FramePipeline(
        camera=camera,
        processor=lambda frame: frame.frame_id,
        result_handler=handler,
    )

    stats = pipeline.run()

    assert stats.frames_processed == 3
    assert len(received_outputs) == 3
    assert camera.released is True


def test_pipeline_calculates_processing_time():
    camera = FakeCamera()

    pipeline = FramePipeline(
        camera=camera,
        processor=lambda frame: frame.frame_id,
    )

    stats = pipeline.run(max_frames=3)

    assert stats.processing_ms >= 0
    assert stats.elapsed_seconds > 0
    assert stats.fps > 0
