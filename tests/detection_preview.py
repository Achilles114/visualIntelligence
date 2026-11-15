import cv2

from core.perception.torchvision_detector import TorchVisionDetector
from core.sensors.camera import CameraSensor


def main() -> None:
    detector = TorchVisionDetector(confidence_threshold=0.5)
    camera = CameraSensor(source=0)

    print(f"Model: {detector.model_name}")
    print(f"Device: {detector.device}")
    print("Starting camera. Press Q to quit.")

    try:
        camera.open()

        while True:
            frame = camera.read_frame()

            if frame is None:
                break

            result = detector.detect(frame)

            # Draw detected objects.
            display = frame.image.copy()

            for detection in result.detections:
                box = detection.bbox

                x1, y1 = int(box.x1), int(box.y1)
                x2, y2 = int(box.x2), int(box.y2)

                label = (
                    f"{detection.label} "
                    f"{detection.confidence:.2f}"
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
                    0.6,
                    (0, 255, 0),
                    2,
                )

            cv2.putText(
                display,
                f"Objects: {len(result.detections)} | "
                f"Inference: {result.inference_ms:.1f} ms",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

            cv2.imshow("VIE - Object Detection", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()