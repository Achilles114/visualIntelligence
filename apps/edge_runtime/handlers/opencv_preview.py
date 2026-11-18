
import cv2

from core.scene.models import SceneSnapshot
from runtime.frame_pipeline import PipelineOutput


class OpenCVPreview:
    """Displays the current scene with object and event information."""

    def __init__(
        self,
        window_name: str = "VIE - Visual Intelligence Preview",
    ) -> None:
        self.window_name = window_name

    def __call__(
        self,
        output: PipelineOutput[SceneSnapshot],
    ) -> bool:
        display = output.frame.image.copy()
        scene = output.result

        for track in scene.tracks:
            box = track.bbox

            x1, y1 = int(box.x1), int(box.y1)
            x2, y2 = int(box.x2), int(box.y2)

            label = (
                f"ID: {track.track_id} | "
                f"{track.label} | "
                f"{track.confidence:.2f}"
            )

            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                display,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

        object_summary = ", ".join(
            f"{label}: {count}"
            for label, count in scene.object_counts
        )

        cv2.putText(
            display,
            f"Objects: {scene.total_objects} | {object_summary}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )

        cv2.putText(
            display,
            f"Frame: {scene.frame_id} | Processing: "
            f"{output.processing_ms:.1f} ms",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
        )

        # Show the latest scene events.
        for index, event in enumerate(scene.events[-3:]):
            message = (
                f"{event.event_type.value}: "
                f"{event.label} (ID {event.track_id})"
            )

            cv2.putText(
                display,
                message,
                (10, 95 + index * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 200, 0),
                1,
            )

        cv2.imshow(self.window_name, display)

        return (cv2.waitKey(1) & 0xFF) == ord("q")

    def close(self) -> None:
        cv2.destroyAllWindows()
