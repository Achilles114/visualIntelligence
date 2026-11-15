
import cv2

from core.sensors.camera import CameraSensor, CameraError


try:
    with CameraSensor(source=0) as camera:
        print("Camera connected. Press Q to exit.")

        while True:
            frame = camera.read_frame()

            if frame is None:
                print("Unable to read frame.")
                break

            print(
                f"Frame: {frame.frame_id} | "
                f"Size: {frame.width}x{frame.height}"
            )

            cv2.imshow("VIE Camera Preview", frame.image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

except CameraError as error:
    print(f"Camera error: {error}")

finally:
    cv2.destroyAllWindows()
