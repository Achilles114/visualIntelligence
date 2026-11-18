
from dataclasses import dataclass

from core.perception.models import BoundingBox, Detection
from core.reid.gallery import ReIDGallery
from core.reid.models import AppearanceFeature
from core.tracking.models import Track, TrackBatch, TrackingInput


@dataclass(slots=True)
class _TrackState:
    """Mutable state for an active object identity."""

    track_id: int
    class_id: int
    label: str
    confidence: float
    bbox: BoundingBox
    age: int = 1
    hits: int = 1
    missed: int = 0


class ReIDTracker:
    """
    Object tracker combining spatial association and appearance matching.

    Matching priority:
        1. IoU matching for active tracks.
        2. Appearance matching for unmatched detections.
        3. Allocate a new identity if no match is found.

    The ReID gallery retains appearance records even after active tracks
    expire, allowing previously seen objects to recover their IDs.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        reid_similarity_threshold: float = 0.85,
        max_age: int = 5,
        min_hits: int = 2,
        max_samples_per_identity: int = 5,
        max_identities: int = 1000,
    ) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")

        if not 0.0 <= reid_similarity_threshold <= 1.0:
            raise ValueError(
                "reid_similarity_threshold must be between 0 and 1"
            )

        if max_age < 0:
            raise ValueError("max_age cannot be negative")

        if min_hits < 1:
            raise ValueError("min_hits must be at least 1")

        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits

        self.gallery = ReIDGallery(
            similarity_threshold=reid_similarity_threshold,
            max_samples_per_identity=max_samples_per_identity,
            max_identities=max_identities,
        )

        self._tracks: dict[int, _TrackState] = {}
        self._identity_classes: dict[int, int] = {}
        self._next_track_id = 1

    @property
    def tracker_name(self) -> str:
        return "reid-tracker"

    def reset(self) -> None:
        """Clear active tracks, appearance records and identity state."""

        self._tracks.clear()
        self._identity_classes.clear()
        self.gallery.clear()
        self._next_track_id = 1

    @staticmethod
    def _calculate_iou(
        box_a: BoundingBox,
        box_b: BoundingBox,
    ) -> float:
        """Calculate Intersection over Union."""

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

    def _create_track(
        self,
        detection: Detection,
        track_id: int | None = None,
    ) -> _TrackState:
        """Create a new track, optionally restoring an existing identity."""

        if track_id is None:
            track_id = self._next_track_id
            self._next_track_id += 1

        track = _TrackState(
            track_id=track_id,
            class_id=detection.class_id,
            label=detection.label,
            confidence=detection.confidence,
            bbox=detection.bbox,
        )

        self._tracks[track_id] = track

        return track

    @staticmethod
    def _update_track(
        track: _TrackState,
        detection: Detection,
    ) -> None:
        """Update an existing track using a matched detection."""

        track.bbox = detection.bbox
        track.confidence = detection.confidence
        track.label = detection.label
        track.hits += 1
        track.missed = 0

    def _store_feature(
        self,
        track_id: int,
        class_id: int,
        feature: AppearanceFeature,
    ) -> None:
        """Store appearance information and maintain identity metadata."""

        self.gallery.add(track_id, feature)
        self._identity_classes[track_id] = class_id

        # Keep class metadata bounded by the gallery's identity capacity.
        while len(self._identity_classes) > self.gallery.identity_count:
            oldest_track_id = next(iter(self._identity_classes))
            del self._identity_classes[oldest_track_id]

    def update(self, data: TrackingInput) -> TrackBatch:
        """Associate detections using IoU and optional appearance features."""

        detections = data.detections.detections
        features = data.appearance_features

        # Step 1: Find possible IoU matches with active tracks.
        existing_tracks = list(self._tracks.values())

        candidates: list[tuple[float, int, int]] = []

        for track in existing_tracks:
            for detection_index, detection in enumerate(detections):
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

        candidates.sort(key=lambda item: item[0], reverse=True)

        matched_track_ids: set[int] = set()
        matched_detection_indices: set[int] = set()

        detection_to_track: dict[int, int] = {}

        # Step 2: Greedy one-to-one IoU association.
        for _, track_id, detection_index in candidates:
            if track_id in matched_track_ids:
                continue

            if detection_index in matched_detection_indices:
                continue

            matched_track_ids.add(track_id)
            matched_detection_indices.add(detection_index)

            detection_to_track[detection_index] = track_id

        # Step 3: Age existing tracks and update IoU matches.
        for track in existing_tracks:
            track.age += 1

            if track.track_id in matched_track_ids:
                detection_index = next(
                    index
                    for index, matched_id in detection_to_track.items()
                    if matched_id == track.track_id
                )

                detection = detections[detection_index]
                self._update_track(track, detection)

            else:
                track.missed += 1

        # Step 4: Update appearance records for IoU matches.
        if features is not None:
            for detection_index, track_id in detection_to_track.items():
                self._store_feature(
                    track_id=track_id,
                    class_id=detections[detection_index].class_id,
                    feature=features[detection_index],
                )

        # Step 5: Re-identify unmatched detections.
        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detection_indices:
                continue

            feature = (
                features[detection_index]
                if features is not None
                else None
            )

            reidentified_track_id: int | None = None

            if feature is not None:
                incompatible_track_ids = {
                    track_id
                    for track_id, class_id in self._identity_classes.items()
                    if class_id != detection.class_id
                }

                excluded_track_ids = (
                    matched_track_ids | incompatible_track_ids
                )

                match = self.gallery.match(
                    feature,
                    exclude_track_ids=excluded_track_ids,
                )

                if match is not None:
                    reidentified_track_id = match.track_id

            if reidentified_track_id is not None:
                existing_track = self._tracks.get(
                    reidentified_track_id
                )

                if (
                    existing_track is not None
                    and existing_track.missed <= self.max_age
                ):
                    # Restore an active but temporarily missing track.
                    self._update_track(
                        existing_track,
                        detection,
                    )
                else:
                    # Restore an identity whose active track expired.
                    self._create_track(
                        detection,
                        track_id=reidentified_track_id,
                    )

                track_id = reidentified_track_id

            else:
                # No suitable appearance match: create a new identity.
                new_track = self._create_track(detection)
                track_id = new_track.track_id

            matched_track_ids.add(track_id)
            matched_detection_indices.add(detection_index)

            detection_to_track[detection_index] = track_id

            if feature is not None:
                self._store_feature(
                    track_id=track_id,
                    class_id=detection.class_id,
                    feature=feature,
                )

        # Step 6: Expire old active tracks.
        expired_track_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if track.missed > self.max_age
        ]

        for track_id in expired_track_ids:
            del self._tracks[track_id]

        # Step 7: Produce output for objects detected in this frame.
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

        batch = data.detections

        return TrackBatch(
            frame_id=batch.frame_id,
            sensor_id=batch.sensor_id,
            timestamp=batch.timestamp,
            tracker_name=self.tracker_name,
            image_width=batch.image_width,
            image_height=batch.image_height,
            tracks=output_tracks,
        )
