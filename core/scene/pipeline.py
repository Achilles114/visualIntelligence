
from core.perception.engine import PerceptionEngine
from core.reid.base import AppearanceExtractor
from core.scene.engine import SceneEngine
from core.scene.models import SceneSnapshot
from core.sensors.models import SensorFrame
from core.spatial.depth.base import MonocularDepthEstimator
from core.spatial.motion import MotionProcessor
from core.spatial.processor import SpatialProcessor
from core.tracking.engine import TrackingEngine
from core.tracking.pipeline import PerceptionTrackingPipeline


class ScenePipeline:
    """Coordinates perception, tracking, scene and spatial understanding."""

    def __init__(
        self,
        perception: PerceptionEngine,
        tracking: TrackingEngine,
        scene: SceneEngine | None = None,
        spatial: SpatialProcessor | None = None,
        motion: MotionProcessor | None = None,
        appearance_extractor: AppearanceExtractor | None = None,
        depth_estimator: MonocularDepthEstimator | None = None,
    ) -> None:
        self.perception_tracking = PerceptionTrackingPipeline(
            perception=perception,
            tracking=tracking,
            appearance_extractor=appearance_extractor,
        )

        self.scene = scene or SceneEngine()
        self.spatial = spatial or SpatialProcessor()
        self.motion = motion or MotionProcessor()
        self.depth_estimator = depth_estimator

    def process(self, frame: SensorFrame) -> SceneSnapshot:
        track_batch = self.perception_tracking.process(frame)

        snapshot = self.scene.process(track_batch)

        depth_map = None
        if self.depth_estimator is not None:
            depth_map = self.depth_estimator.estimate(frame)

        spatial_batch = self.spatial.process(
            track_batch,
            depth_map=depth_map,
        )

        motion_batch = self.motion.process(track_batch)

        return SceneSnapshot(
            frame_id=snapshot.frame_id,
            sensor_id=snapshot.sensor_id,
            timestamp=snapshot.timestamp,
            tracks=snapshot.tracks,
            object_counts=snapshot.object_counts,
            events=snapshot.events,
            spatial=spatial_batch,
            motion=motion_batch,
        )

    def reset(self) -> None:
        self.perception_tracking.reset()
        self.scene.reset()
        self.motion.reset()
