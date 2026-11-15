
import cv2

from core.tracking.models import TrackBatch
from runtime.frame_pipeline import PipelineOutput


class OpenCVPreview:
    """Displays video frames with tracked objects and persistent IDs."""

    def __init__(
        self,
        window_name: str = "VIE - Visual Intelligence Preview",
    ) -> None:
        self.window_name = window_name

    def __call__(
        self,
        output: PipelineOutput[TrackBatch],
    ) -> bool:
        display = output.frame.image.copy()

        for track in output.result.tracks:
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

        # Display tracking and performance information.
        cv2.putText(
            display,
            (
                f"Tracked objects: {len(output.result.tracks)} | "
                f"Processing: {output.processing_ms:.1f} ms"
            ),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

        cv2.putText(
            display,
            f"Frame: {output.frame.frame_id}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

        cv2.imshow(self.window_name, display)

        return (cv2.waitKey(1) & 0xFF) == ord("q")

    def close(self) -> None:
        cv2.destroyAllWindows()
