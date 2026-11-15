
from collections.abc import Callable
from dataclasses import dataclass
from threading import Event
from time import perf_counter
from typing import Generic, TypeVar

from core.sensors.camera import CameraSensor
from core.sensors.models import SensorFrame


ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class PipelineOutput(Generic[ResultT]):
    """Pairs an input frame with its corresponding processing result."""

    frame: SensorFrame
    result: ResultT
    processing_ms: float


@dataclass(frozen=True, slots=True)
class PipelineStats:
    frames_processed: int
    elapsed_seconds: float
    processing_ms: float

    @property
    def fps(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.frames_processed / self.elapsed_seconds


class FramePipeline(Generic[ResultT]):
    """Generic pipeline independent of perception models and UI."""

    def __init__(
        self,
        camera: CameraSensor,
        processor: Callable[[SensorFrame], ResultT],
        result_handler: (
            Callable[[PipelineOutput[ResultT]], bool | None] | None
        ) = None,
    ) -> None:
        self.camera = camera
        self.processor = processor
        self.result_handler = result_handler

        self._stop_event = Event()
        self._frames_processed = 0
        self._processing_seconds = 0.0

    def stop(self) -> None:
        self._stop_event.set()

    def run(self, max_frames: int | None = None) -> PipelineStats:
        if max_frames is not None and max_frames < 1:
            raise ValueError("max_frames must be at least 1")

        self._stop_event.clear()
        self._frames_processed = 0
        self._processing_seconds = 0.0

        start_time = perf_counter()

        try:
            self.camera.open()

            while not self._stop_event.is_set():
                if (
                    max_frames is not None
                    and self._frames_processed >= max_frames
                ):
                    break

                frame = self.camera.read_frame()

                if frame is None:
                    break

                processing_start = perf_counter()
                result = self.processor(frame)
                processing_time = perf_counter() - processing_start

                self._processing_seconds += processing_time

                output = PipelineOutput(
                    frame=frame,
                    result=result,
                    processing_ms=processing_time * 1000,
                )

                self._frames_processed += 1

                if self.result_handler is not None:
                    should_stop = self.result_handler(output)
                    if should_stop:
                        self.stop()

        finally:
            self.camera.release()

        elapsed = perf_counter() - start_time

        return PipelineStats(
            frames_processed=self._frames_processed,
            elapsed_seconds=elapsed,
            processing_ms=(
                self._processing_seconds * 1000 / self._frames_processed
                if self._frames_processed
                else 0.0
            ),
        )
