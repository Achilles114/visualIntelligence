
from datetime import datetime, timezone

import numpy as np
import pytest
import torch

from core.sensors.models import SensorFrame
from core.spatial.depth.midas import MiDaSSmallEstimator
from core.spatial.depth.models import DepthRepresentation, DepthUnit


class FakeModel(torch.nn.Module):
    def forward(self, tensor):
        return torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]]],
            device=tensor.device,
        )


def fake_transform(image):
    return torch.zeros((1, 3, 8, 8), dtype=torch.float32)


def create_frame(image=None):
    if image is None:
        image = np.zeros((32, 48, 3), dtype=np.uint8)

    return SensorFrame(
        frame_id=5,
        sensor_id="camera-001",
        source="test",
        timestamp=datetime.now(timezone.utc),
        image=image,
    )


@pytest.fixture
def estimator():
    return MiDaSSmallEstimator(
        device="cpu",
        model=FakeModel(),
        transform=fake_transform,
    )


def test_estimator_returns_correct_depth_dimensions(estimator):
    result = estimator.estimate(create_frame())

    assert result.values.shape == (32, 48)


def test_estimator_returns_normalized_relative_depth(estimator):
    result = estimator.estimate(create_frame())

    assert result.values.min() >= 0.0
    assert result.values.max() <= 1.0
    assert result.representation == DepthRepresentation.RELATIVE
    assert result.unit == DepthUnit.RELATIVE


def test_estimator_preserves_frame_metadata(estimator):
    frame = create_frame()
    result = estimator.estimate(frame)

    assert result.frame_id == frame.frame_id
    assert result.sensor_id == frame.sensor_id
    assert result.timestamp == frame.timestamp
    assert result.source == "midas-small"


def test_estimator_handles_grayscale_frames(estimator):
    frame = create_frame(
        np.zeros((32, 48), dtype=np.uint8)
    )

    result = estimator.estimate(frame)

    assert result.values.shape == (32, 48)


def test_estimator_handles_constant_predictions():
    class ConstantModel(torch.nn.Module):
        def forward(self, tensor):
            return torch.ones((1, 2, 2), device=tensor.device)

    estimator = MiDaSSmallEstimator(
        device="cpu",
        model=ConstantModel(),
        transform=fake_transform,
    )

    result = estimator.estimate(create_frame())

    assert np.isfinite(result.values).all()
    assert np.all(result.values == 1.0)
    assert np.all(result.values > 0.0)


def test_estimator_requires_model_and_transform_together():
    with pytest.raises(ValueError):
        MiDaSSmallEstimator(
            device="cpu",
            model=FakeModel(),
        )
