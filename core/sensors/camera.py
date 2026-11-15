
from datetime import datetime, timezone

import cv2

from core.sensors.models import SensorFrame


class CameraError(Exception):
    """Raised when the camera cannot be accessed or initialized."""


class CameraSensor:
    """
    Captures frames from a webcam or video source.

    Produces standardized SensorFrame objects for VIE.
    """

    def __init__(
        self,
        source: int | str = 0,
        sensor_id: str = "camera-001",
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        self.source = source
        self.sensor_id = sensor_id
        self.width = width
        self.height = height

        self._capture: cv2.VideoCapture | None = None
        self._frame_id = 0

    def open(self) -> None:
        """Initialize the camera connection."""

        if self.is_open:
            return

        self._capture = cv2.VideoCapture(self.source)

        if not self._capture.isOpened():
            self._capture.release()
            self._capture = None
            raise CameraError(f"Could not open camera source: {self.source}")

        if self.width is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)

        if self.height is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    @property
    def is_open(self) -> bool:
        return (
            self._capture is not None
            and self._capture.isOpened()
        )

    def read_frame(self) -> SensorFrame | None:
        """
        Capture one frame.

        Returns None if a frame cannot be read.
        """

        if not self.is_open:
            raise CameraError("Camera is not open. Call open() first.")

        success, image = self._capture.read()

        if not success or image is None:
            return None

        frame = SensorFrame(
            frame_id=self._frame_id,
            sensor_id=self.sensor_id,
            source=str(self.source),
            timestamp=datetime.now(timezone.utc),
            image=image,
        )

        self._frame_id += 1

        return frame

    def release(self) -> None:
        """Release camera resources."""

        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
