
from apps.edge_runtime.handlers.opencv_preview import OpenCVPreview
from core.perception.engine import PerceptionEngine
from core.perception.factory import create_detector
from core.sensors.camera import CameraSensor
from core.tracking.engine import TrackingEngine
from core.tracking.iou_tracker import IoUTracker
from core.tracking.pipeline import PerceptionTrackingPipeline
from runtime.frame_pipeline import FramePipeline


def main() -> None:
    # Initialize the object detection model.
    detector = create_detector(
        model="torchvision-ssd",
        confidence_threshold=0.5,
    )

    # Initialize perception and tracking engines.
    perception = PerceptionEngine(detector)
    tracker = IoUTracker(
        iou_threshold=0.3,
        max_age=5,
        min_hits=2,
    )
    tracking = TrackingEngine(tracker)

    # Combine perception and tracking.
    processor = PerceptionTrackingPipeline(
        perception=perception,
        tracking=tracking,
    )

    # Initialize camera and visualization.
    camera = CameraSensor(source=0)
    preview = OpenCVPreview()

    # Create the generic frame pipeline.
    pipeline = FramePipeline(
        camera=camera,
        processor=processor.process,
        result_handler=preview,
    )

    try:
        stats = pipeline.run()
    finally:
        preview.close()

    print("\nVIE pipeline completed")
    print("Frames:", stats.frames_processed)
    print(
        "Average processing time:",
        f"{stats.processing_ms:.2f} ms",
    )
    print("Average FPS:", f"{stats.fps:.2f}")


if __name__ == "__main__":
    main()
