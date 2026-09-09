"""Dynamic-object track dataclasses and JSON (de)serialisation.

A :class:`Track` represents a single dynamic object observed over multiple
frames inside the scene, modeled on alpasim's ``sequence_tracks.json`` layout
but transposed to a per-track, frames-as-list form for ergonomics.

* :class:`TrackFrame` — one timestamped pose sample (translation + xyzw
  quaternion).
* :class:`Track` — track id + class label + constant box size + sorted list
  of :class:`TrackFrame` + optional per-track flag string.

The on-disk schema is ``splatsim.sequence_tracks/v2``::

    {
      "schema": "splatsim.sequence_tracks/v2",
      "frame": "world",
      "frame_convention": {...},
      "tracks": [
        {
          "track_id": "100",
          "class_name": "automobile",
          "size": [3.99, 1.80, 1.48],
          "flag": "NONE",
          "frames": [
            {
              "timestamp_us": 27567868848,
              "translation": [113.62, -58.55, 1.92],
              "rotation":    [-0.0005, -0.0113, 0.6645, 0.7472]
            },
            ...
          ],
          "metadata": {}
        },
        ...
      ]
    }

No legacy columnar or NRE representation is accepted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .frame_convention import (
    FRAME_CONVENTION,
    validate_frame_convention,
    validate_rotation,
    validate_timestamps,
)

__all__ = [
    "TRACKS_SCHEMA",
    "Track",
    "TrackFrame",
    "parse_tracks",
    "serialize_tracks",
]


TRACKS_SCHEMA = "splatsim.sequence_tracks/v2"

# Tracks live in the same declared world frame as the embedded SPZ chunks.
_FRAME = "world"


@dataclass
class TrackFrame:
    """One pose sample for a dynamic object at a specific moment."""

    timestamp_us: int
    translation: tuple[float, float, float]
    rotation: tuple[float, float, float, float]  # xyzw, unit-norm

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_us": int(self.timestamp_us),
            "translation": [float(v) for v in self.translation],
            "rotation": [float(v) for v in self.rotation],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrackFrame:
        tr = d["translation"]
        ro = d["rotation"]
        if len(tr) != 3:
            raise ValueError(f"frame.translation must have 3 elements, got {len(tr)}")
        if len(ro) != 4:
            raise ValueError(f"frame.rotation must have 4 elements (xyzw), got {len(ro)}")
        validate_timestamps([d["timestamp_us"]], where="track frame")
        return cls(
            timestamp_us=int(d["timestamp_us"]),
            translation=(float(tr[0]), float(tr[1]), float(tr[2])),
            rotation=(float(ro[0]), float(ro[1]), float(ro[2]), float(ro[3])),
        )


@dataclass
class Track:
    """A dynamic object tracked across frames in root-local coordinates."""

    track_id: str
    class_name: str
    size: tuple[float, float, float]  # (dx, dy, dz) — constant across frames
    frames: list[TrackFrame] = field(default_factory=list)
    flag: str = "NONE"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": str(self.track_id),
            "class_name": str(self.class_name),
            "size": [float(v) for v in self.size],
            "flag": str(self.flag),
            "frames": [f.to_dict() for f in self.frames],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Track:
        size = d["size"]
        if len(size) != 3:
            raise ValueError(f"track.size must have 3 elements, got {len(size)}")
        return cls(
            track_id=str(d["track_id"]),
            class_name=str(d["class_name"]),
            size=(float(size[0]), float(size[1]), float(size[2])),
            frames=[TrackFrame.from_dict(f) for f in d.get("frames") or []],
            flag=str(d.get("flag", "NONE")),
            metadata=dict(d.get("metadata") or {}),
        )


# ---------------------------------------------------------------------------
# Collection-level (de)serialisation
# ---------------------------------------------------------------------------


def serialize_tracks(tracks: list[Track]) -> dict[str, Any]:
    """Build a frame-explicit world-coordinate track document."""
    seen: set[str] = set()
    out_list: list[dict[str, Any]] = []
    for t in tracks:
        if t.track_id in seen:
            raise ValueError(f"duplicate track_id: {t.track_id!r}")
        seen.add(t.track_id)
        timestamps = [frame.timestamp_us for frame in t.frames]
        validate_timestamps(timestamps, where=f"track {t.track_id!r}")
        for i, frame in enumerate(t.frames):
            validate_rotation(frame.rotation, where=f"track {t.track_id!r} frame {i} rotation")
        out_list.append(t.to_dict())
    return {
        "schema": TRACKS_SCHEMA,
        "frame": _FRAME,
        "frame_convention": FRAME_CONVENTION,
        "tracks": out_list,
    }


def parse_tracks(doc: dict[str, Any]) -> list[Track]:
    """Inverse of :func:`serialize_tracks`.

    Rejects duplicate ``track_id`` so the load path enforces the same
    invariant as :func:`serialize_tracks`.
    """
    schema = doc.get("schema")
    if schema != TRACKS_SCHEMA:
        raise ValueError(f"unexpected tracks schema {schema!r}; expected {TRACKS_SCHEMA!r}")
    if doc.get("frame") != _FRAME:
        raise ValueError(f"tracks frame must be {_FRAME!r}")
    validate_frame_convention(doc.get("frame_convention"))
    raw = doc.get("tracks")
    if not isinstance(raw, list):
        raise ValueError("tracks document is missing the 'tracks' list")
    out: list[Track] = []
    seen: set[str] = set()
    for entry in raw:
        track = Track.from_dict(entry)
        if track.track_id in seen:
            raise ValueError(f"duplicate track_id: {track.track_id!r}")
        seen.add(track.track_id)
        validate_timestamps(
            [frame.timestamp_us for frame in track.frames],
            where=f"track {track.track_id!r}",
        )
        for i, frame in enumerate(track.frames):
            validate_rotation(frame.rotation, where=f"track {track.track_id!r} frame {i} rotation")
        out.append(track)
    return out
