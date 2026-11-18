
from core.scene.models import (
    SceneEvent,
    SceneEventType,
    SceneSnapshot,
)
from core.tracking.models import TrackBatch


class SceneEngine:
    """
    Converts tracking results into scene snapshots and events.

    Tracks appearing for the first time generate TRACK_STARTED.
    Previously visible tracks that disappear generate TRACK_LOST.
    """

    def __init__(self) -> None:
        self._previous_tracks: dict[int, str] = {}

    def reset(self) -> None:
        """Clear scene history."""
        self._previous_tracks.clear()

    def process(self, batch: TrackBatch) -> SceneSnapshot:
        current_tracks = {
            track.track_id: track
            for track in batch.tracks
        }

        current_track_ids = set(current_tracks)
        previous_track_ids = set(self._previous_tracks)

        events: list[SceneEvent] = []

        # Detect newly observed tracks.
        for track_id in current_track_ids - previous_track_ids:
            track = current_tracks[track_id]

            events.append(
                SceneEvent(
                    event_type=SceneEventType.TRACK_STARTED,
                    frame_id=batch.frame_id,
                    timestamp=batch.timestamp,
                    track_id=track_id,
                    label=track.label,
                )
            )

        # Detect tracks that disappeared.
        for track_id in previous_track_ids - current_track_ids:
            events.append(
                SceneEvent(
                    event_type=SceneEventType.TRACK_LOST,
                    frame_id=batch.frame_id,
                    timestamp=batch.timestamp,
                    track_id=track_id,
                    label=self._previous_tracks[track_id],
                )
            )

        # Count currently visible objects by class label.
        counts: dict[str, int] = {}

        for track in current_tracks.values():
            counts[track.label] = counts.get(track.label, 0) + 1

        snapshot = SceneSnapshot(
            frame_id=batch.frame_id,
            sensor_id=batch.sensor_id,
            timestamp=batch.timestamp,
            tracks=batch.tracks,
            object_counts=tuple(sorted(counts.items())),
            events=tuple(events),
        )

        self._previous_tracks = {
            track_id: track.label
            for track_id, track in current_tracks.items()
        }

        return snapshot
