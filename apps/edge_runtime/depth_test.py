
import time

import cv2
import numpy as np

from core.sensors.camera import CameraSensor
from core.spatial.depth.midas import MiDaSSmallEstimator


def main() -> None:
    estimator = MiDaSSmallEstimator()
    camera = CameraSensor(source=0)

    print(f"Depth estimator: {estimator.estimator_name}")
    print(f"Device: {estimator.device}")
    print("Press Q to exit.")

    try:
        camera.open()

        while True:
            frame = camera.read_frame()

            if frame is None:
                print("Could not read camera frame.")
                break

            start = time.perf_counter()
            depth_map = estimator.estimate(frame)
            inference_ms = (time.perf_counter() - start) * 1000

            # Convert relative depth into a displayable image.
            depth_visual = np.clip(
                depth_map.values * 255,
                0,
                255,
            ).astype(np.uint8)

            depth_colormap = cv2.applyColorMap(
                depth_visual,
                cv2.COLORMAP_INFERNO,
            )

            original = frame.image.copy()

            cv2.putText(
                original,
                f"Frame: {frame.frame_id}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                depth_colormap,
                f"Inference: {inference_ms:.1f} ms",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            combined = np.hstack((original, depth_colormap))

            cv2.imshow("VIE - Monocular Depth Estimation", combined)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
