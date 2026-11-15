
from dataclasses import dataclass

from core.perception.models import BoundingBox, Detection, DetectionBatch

from core.tracking.models import Track, TrackBatch, TrackingInput


@dataclass(slots=True)
class _TrackState:
    """Internal mutable state maintained for each tracked object."""

    track_id: int
    class_id: int
    label: str
    confidence: float
    bbox: BoundingBox
    age: int = 1
    hits: int = 1
    missed: int = 0


class IoUTracker:
    """
    Lightweight object tracker using Intersection over Union (IoU).

    Associates detections between consecutive frames using bounding-box
    overlap and object class.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 5,
        min_hits: int = 2,
    ) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")

        if max_age < 0:
            raise ValueError("max_age cannot be negative")

        if min_hits < 1:
            raise ValueError("min_hits must be at least 1")

        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits

        self._tracks: dict[int, _TrackState] = {}
        self._next_track_id = 1

    @property
    def tracker_name(self) -> str:
        return "iou-tracker"

    def reset(self) -> None:
        """Clear all tracking state."""
        self._tracks.clear()
        self._next_track_id = 1

    @staticmethod
    def _calculate_iou(
        box_a: BoundingBox,
        box_b: BoundingBox,
    ) -> float:
        """Calculate Intersection over Union between two bounding boxes."""

        intersection_x1 = max(box_a.x1, box_b.x1)
        intersection_y1 = max(box_a.y1, box_b.y1)
        intersection_x2 = min(box_a.x2, box_b.x2)
        intersection_y2 = min(box_a.y2, box_b.y2)

        intersection_width = max(
            0.0,
            intersection_x2 - intersection_x1,
        )
        intersection_height = max(
            0.0,
            intersection_y2 - intersection_y1,
        )

        intersection_area = intersection_width * intersection_height

        area_a = box_a.width * box_a.height
        area_b = box_b.width * box_b.height

        union_area = area_a + area_b - intersection_area

        if union_area <= 0:
            return 0.0

        return intersection_area / union_area

    def _create_track(self, detection: Detection) -> _TrackState:
        """Create a new track for an unmatched detection."""

        track = _TrackState(
            track_id=self._next_track_id,
            class_id=detection.class_id,
            label=detection.label,
            confidence=detection.confidence,
            bbox=detection.bbox,
        )

        self._tracks[self._next_track_id] = track
        self._next_track_id += 1

        return track

    def update(self, data: TrackingInput) -> TrackBatch:
        """Associate detections with existing tracks."""

        detections = data.detections

        existing_tracks = list(self._tracks.values())

        # Generate possible track-to-detection matches.
        candidates: list[tuple[float, int, int]] = []

        for track in existing_tracks:
            for detection_index, detection in enumerate(
                detections.detections
            ):
                # Never associate different object classes.
                if track.class_id != detection.class_id:
                    continue

                iou = self._calculate_iou(
                    track.bbox,
                    detection.bbox,
                )

                if iou >= self.iou_threshold:
                    candidates.append(
                        (iou, track.track_id, detection_index)
                    )

        # Prefer the highest-overlap matches.
        candidates.sort(key=lambda item: item[0], reverse=True)

        matched_track_ids: set[int] = set()
        matched_detection_indices: set[int] = set()

        matched_tracks: dict[int, Detection] = {}

        # Greedy one-to-one matching.
        for _, track_id, detection_index in candidates:
            if track_id in matched_track_ids:
                continue

            if detection_index in matched_detection_indices:
                continue

            matched_track_ids.add(track_id)
            matched_detection_indices.add(detection_index)

            matched_tracks[track_id] = detections.detections[
                detection_index
            ]

        # Update existing tracks.
        for track in existing_tracks:
            track.age += 1

            if track.track_id in matched_tracks:
                detection = matched_tracks[track.track_id]

                track.bbox = detection.bbox
                track.confidence = detection.confidence
                track.label = detection.label
                track.hits += 1
                track.missed = 0

            else:
                track.missed += 1

        # Create tracks for unmatched detections.
        for detection_index, detection in enumerate(
            detections.detections
        ):
            if detection_index not in matched_detection_indices:
                self._create_track(detection)

        # Remove tracks that have been missing for too long.
        expired_track_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if track.missed > self.max_age
        ]

        for track_id in expired_track_ids:
            del self._tracks[track_id]

        # Return tracks associated with detections in this frame.
        output_tracks = tuple(
            Track(
                track_id=track.track_id,
                class_id=track.class_id,
                label=track.label,
                confidence=track.confidence,
                bbox=track.bbox,
                age=track.age,
                hits=track.hits,
                is_confirmed=track.hits >= self.min_hits,
            )
            for track in sorted(
                self._tracks.values(),
                key=lambda item: item.track_id,
            )
            if track.missed == 0
        )

        return TrackBatch(
            frame_id=detections.frame_id,
            sensor_id=detections.sensor_id,
            timestamp=detections.timestamp,
            tracker_name=self.tracker_name,
            image_width=detections.image_width,
            image_height=detections.image_height,
            tracks=output_tracks,
        )
