
from collections.abc import Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as functional

from core.sensors.models import SensorFrame
from core.spatial.depth.models import (
    DepthMap,
    DepthRepresentation,
    DepthUnit,
)


class MiDaSSmallEstimator:
    """Monocular relative-depth estimator using MiDaS Small."""

    def __init__(
        self,
        device: str | None = None,
        model: torch.nn.Module | None = None,
        transform: Callable | None = None,
    ) -> None:
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if (model is None) != (transform is None):
            raise ValueError(
                "Provide both model and transform, or neither"
            )

        if model is None:
            model, transform = self._load_model()

        self.model = model.to(self.device)
        self.model.eval()
        self.transform = transform

    @property
    def estimator_name(self) -> str:
        return "midas-small"

    def _load_model(self):
        model = torch.hub.load(
            "intel-isl/MiDaS",
            "MiDaS_small",
            trust_repo=True,
        )

        transforms = torch.hub.load(
            "intel-isl/MiDaS",
            "transforms",
            trust_repo=True,
        )

        return model, transforms.small_transform

    def estimate(self, frame: SensorFrame) -> DepthMap:
        image = frame.image

        if image.ndim == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        input_tensor = self.transform(rgb).to(self.device)

        with torch.inference_mode():
            prediction = self.model(input_tensor)

            if prediction.ndim == 3:
                prediction = prediction.unsqueeze(1)

            # Detect constant predictions before interpolation.
            is_constant = (
                prediction.max() - prediction.min()
            ).item() <= 1e-8

            if is_constant:
                values = np.ones(
                    (frame.height, frame.width),
                    dtype=np.float32,
                )
            else:
                prediction = functional.interpolate(
                    prediction,
                    size=(frame.height, frame.width),
                    mode="bicubic",
                    align_corners=False,
                )

                values = (
                    prediction.squeeze()
                    .float()
                    .cpu()
                    .numpy()
                )

                minimum = float(values.min())
                maximum = float(values.max())

                values = (values - minimum) / (
                    maximum - minimum
                )

                # Keep every depth value positive and <= 1.0.
                values = (
                    values * (1.0 - 1e-6) + 1e-6
                )

        values = np.asarray(values, dtype=np.float32)

        return DepthMap(
            frame_id=frame.frame_id,
            sensor_id=frame.sensor_id,
            timestamp=frame.timestamp,
            values=values,
            representation=DepthRepresentation.RELATIVE,
            unit=DepthUnit.RELATIVE,
            source=self.estimator_name,
        )
