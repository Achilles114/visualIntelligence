
from apps.edge_runtime.handlers.opencv_preview import OpenCVPreview
from core.perception.engine import PerceptionEngine
from core.perception.factory import create_detector
from core.scene.pipeline import ScenePipeline
from core.sensors.camera import CameraSensor
from core.tracking.engine import TrackingEngine
from core.tracking.iou_tracker import IoUTracker
from runtime.frame_pipeline import FramePipeline
from core.spatial.depth.midas import MiDaSSmallEstimator

depth_estimator = MiDaSSmallEstimator()


def main() -> None:
    detector = create_detector(
        model="torchvision-ssd",
        confidence_threshold=0.5,
    )

    perception = PerceptionEngine(detector)

    tracker = IoUTracker(
        iou_threshold=0.3,
        max_age=5,
        min_hits=2,
    )

    tracking = TrackingEngine(tracker)

    scene_pipeline = ScenePipeline(
        perception=perception,
        tracking=tracking,
        depth_estimator=depth_estimator,
    )

    camera = CameraSensor(source=0)
    preview = OpenCVPreview()

    pipeline = FramePipeline(
        camera=camera,
        processor=scene_pipeline.process,
        result_handler=preview,
    )

    try:
        stats = pipeline.run()
    finally:
        preview.close()

    print("\nVIE scene pipeline completed")
    print("Detector:", perception.model_name)
    print("Tracker:", tracking.tracker_name)
    print("Frames:", stats.frames_processed)
    print(
        "Average processing time:",
        f"{stats.processing_ms:.2f} ms",
    )
    print("Average FPS:", f"{stats.fps:.2f}")


if __name__ == "__main__":
    main()
