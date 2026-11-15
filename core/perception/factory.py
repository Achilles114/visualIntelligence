from core.perception.base import ObjectDetector


def create_detector(
    model: str = "torchvision-ssd",
    confidence_threshold: float = 0.5,
    device: str | None = None,
) -> ObjectDetector:
    """Create a detector without exposing model-specific details."""

    if model == "torchvision-ssd":
        from core.perception.torchvision_detector import TorchVisionDetector

        return TorchVisionDetector(
            confidence_threshold=confidence_threshold,
            device=device,
        )

    raise ValueError(f"Unsupported detection model: {model}")