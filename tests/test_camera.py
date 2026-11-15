
import pytest

from core.sensors.camera import CameraSensor, CameraError


def test_camera_requires_open_before_reading():
    camera = CameraSensor(source=0)

    with pytest.raises(CameraError):
        camera.read_frame()


def test_camera_release_is_safe():
    camera = CameraSensor(source=0)

    camera.release()

    assert camera.is_open is False
