
import cv2

from core.sensors.camera import CameraSensor, CameraError
from runtime.frame_pipeline import FramePipeline


def display_frame(frame):
    cv2.imshow("VIE - Real-Time Frame Pipeline", frame.image)

    print(
        f"Frame: {frame.frame_id} | "
        f"Resolution: {frame.width}x{frame.height}"
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        pipeline.stop()


camera = CameraSensor(
    source=0,
    sensor_id="webcam-001",
)

pipeline = FramePipeline(
    camera=camera,
    on_frame=display_frame,
)

try:
    stats = pipeline.run()

    print("\nPipeline stopped.")
    print(f"Frames processed: {stats.frames_processed}")
    print(f"Average FPS: {stats.fps:.2f}")

except CameraError as error:
    print(f"Camera error: {error}")

finally:
    cv2.destroyAllWindows()
