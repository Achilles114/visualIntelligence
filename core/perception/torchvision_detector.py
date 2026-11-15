
from time import perf_counter

import cv2
import torch
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large,
    SSDLite320_MobileNet_V3_Large_Weights,
)

from core.perception.base import ObjectDetector
from core.perception.models import (
    BoundingBox,
    Detection,
    DetectionBatch,
)
from core.sensors.models import SensorFrame


class TorchVisionDetector:
    """Object detector powered by TorchVision SSD MobileNet."""

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        device: str | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")

        self.confidence_threshold = confidence_threshold
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.categories = self.weights.meta["categories"]

        self.model = ssdlite320_mobilenet_v3_large(
            weights=self.weights
        )
        self.model.to(self.device)
        self.model.eval()

        self.transforms = self.weights.transforms()

    @property
    def model_name(self) -> str:
        return "torchvision-ssdlite320-mobilenet-v3-large"

    def detect(self, frame: SensorFrame) -> DetectionBatch:
        # OpenCV camera frames use BGR; TorchVision expects RGB.
        image = frame.image

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert HWC NumPy image to CHW PyTorch tensor.
        image_tensor = torch.from_numpy(image.copy())
        image_tensor = image_tensor.permute(2, 0, 1).float() / 255.0

        image_tensor = self.transforms(image_tensor)
        image_tensor = image_tensor.to(self.device)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        start_time = perf_counter()

        with torch.inference_mode():
            predictions = self.model([image_tensor])

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        inference_ms = (perf_counter() - start_time) * 1000

        prediction = predictions[0]

        detections = []

        for box, label_id, score in zip(
            prediction["boxes"],
            prediction["labels"],
            prediction["scores"],
        ):
            confidence = float(score.item())

            if confidence < self.confidence_threshold:
                continue

            class_id = int(label_id.item())
            label = self.categories[class_id]

            x1, y1, x2, y2 = box.tolist()

            # Keep coordinates within the original frame.
            x1 = max(0.0, min(x1, float(frame.width)))
            y1 = max(0.0, min(y1, float(frame.height)))
            x2 = max(0.0, min(x2, float(frame.width)))
            y2 = max(0.0, min(y2, float(frame.height)))

            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                Detection(
                    class_id=class_id,
                    label=label,
                    confidence=confidence,
                    bbox=BoundingBox(x1, y1, x2, y2),
                )
            )

        return DetectionBatch(
            frame_id=frame.frame_id,
            sensor_id=frame.sensor_id,
            timestamp=frame.timestamp,
            model_name=self.model_name,
            inference_ms=inference_ms,
            image_width=frame.width,
            image_height=frame.height,
            detections=tuple(detections),
        )
