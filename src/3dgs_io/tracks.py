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

This is **not** byte-compatible with alpasim's ``sequence_tracks.json`` — the
alpasim layout uses parallel columnar arrays (``tracks_id`` / ``tracks_poses``
/ ``tracks_timestamps_us`` / ``tracks_label_class`` / ``tracks_flags`` +
``cuboidtracks_data.cuboids_dims``); we transpose it once per Track. Use
:func:`parse_alpasim_sequence_tracks` to ingest an alpasim document into our
schema.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .frame_convention import (
    FRAME_CONVENTION,
    validate_frame_convention,
    validate_rotation,
    validate_timestamps,
)

_log = logging.getLogger(__name__)

_ALPASIM_KNOWN_TRACKS_KEYS = frozenset(
    {"tracks_id", "tracks_poses", "tracks_timestamps_us", "tracks_label_class", "tracks_flags"}
)
_ALPASIM_KNOWN_CUBOID_KEYS = frozenset({"cuboids_dims"})

__all__ = [
    "TRACKS_SCHEMA",
    "Track",
    "TrackFrame",
    "dump_alpasim_sequence_tracks",
    "parse_alpasim_sequence_tracks",
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


# ---------------------------------------------------------------------------
# alpasim sequence_tracks.json ingestion
# ---------------------------------------------------------------------------


def parse_alpasim_sequence_tracks(doc: dict[str, Any]) -> list[Track]:
    """Convert an alpasim ``sequence_tracks.json`` document into our schema.

    alpasim stores tracks as parallel columnar arrays under a chunk key
    (typically ``"dummy_chunk_id"``)::

        {
          "<chunk_id>": {
            "tracks_data": {
              "tracks_id":            [str, ...],
              "tracks_poses":         [[[tx, ty, tz, qx, qy, qz, qw], ...], ...],
              "tracks_timestamps_us": [[int, ...], ...],
              "tracks_label_class":   [str, ...],
              "tracks_flags":         [str, ...]
            },
            "cuboidtracks_data": {
              "cuboids_dims": [[dx, dy, dz], ...]
            }
          }
        }

    The pose tuple is ``(tx, ty, tz, qx, qy, qz, qw)`` in the NRE-local frame,
    which equals our root-local frame after the bundle's ``root.transform`` is
    applied.  Multi-chunk inputs are concatenated; track_ids are namespaced
    with the chunk id when more than one chunk is present, to avoid collisions.
    """
    if not isinstance(doc, dict):
        raise ValueError("alpasim sequence_tracks document must be a dict at the top level")

    chunks = list(doc.items())
    if not chunks:
        return []
    multi = len(chunks) > 1
    out: list[Track] = []

    for chunk_id, chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        td = chunk.get("tracks_data") or {}
        cd = chunk.get("cuboidtracks_data") or {}
        unknown_tracks_keys = set(td) - _ALPASIM_KNOWN_TRACKS_KEYS
        if unknown_tracks_keys:
            _log.warning(
                "alpasim chunk %r: dropping unknown tracks_data keys %s",
                chunk_id,
                sorted(unknown_tracks_keys),
            )
        unknown_cuboid_keys = set(cd) - _ALPASIM_KNOWN_CUBOID_KEYS
        if unknown_cuboid_keys:
            _log.warning(
                "alpasim chunk %r: dropping unknown cuboidtracks_data keys %s",
                chunk_id,
                sorted(unknown_cuboid_keys),
            )
        ids = td.get("tracks_id") or []
        poses = td.get("tracks_poses") or []
        tss = td.get("tracks_timestamps_us") or []
        classes = td.get("tracks_label_class") or []
        flags = td.get("tracks_flags") or []
        dims = cd.get("cuboids_dims") or []

        n = len(ids)
        if not (len(poses) == len(tss) == len(classes) == n):
            raise ValueError(
                f"alpasim chunk {chunk_id!r}: column-array lengths disagree "
                f"(ids={n}, poses={len(poses)}, ts={len(tss)}, classes={len(classes)})"
            )
        if dims and len(dims) != n:
            raise ValueError(
                f"alpasim chunk {chunk_id!r}: cuboids_dims length {len(dims)} != n_tracks {n}"
            )
        if flags and len(flags) != n:
            raise ValueError(
                f"alpasim chunk {chunk_id!r}: tracks_flags length {len(flags)} != n_tracks {n}"
            )

        for i in range(n):
            track_pose_list = poses[i]
            track_ts = tss[i]
            if len(track_pose_list) != len(track_ts):
                raise ValueError(
                    f"alpasim chunk {chunk_id!r} track {ids[i]!r}: "
                    f"pose count {len(track_pose_list)} != timestamp count {len(track_ts)}"
                )
            frames = [
                TrackFrame(
                    timestamp_us=int(ts),
                    translation=(float(p[0]), float(p[1]), float(p[2])),
                    rotation=(float(p[3]), float(p[4]), float(p[5]), float(p[6])),
                )
                for p, ts in zip(track_pose_list, track_ts, strict=True)
            ]
            size_xyz = dims[i] if dims else (0.0, 0.0, 0.0)
            track_id = f"{chunk_id}/{ids[i]}" if multi else str(ids[i])
            out.append(
                Track(
                    track_id=track_id,
                    class_name=str(classes[i]),
                    size=(float(size_xyz[0]), float(size_xyz[1]), float(size_xyz[2])),
                    frames=frames,
                    flag=str(flags[i]) if flags else "NONE",
                )
            )
    return out


def dump_alpasim_sequence_tracks(
    tracks: list[Track],
    *,
    chunk_id: str = "ego",
) -> dict[str, Any]:
    """Serialize :class:`Track` list into an alpasim ``sequence_tracks.json`` document.

    Inverse of :func:`parse_alpasim_sequence_tracks`. The output is the
    columnar shape alpasim's ``TrafficObjects.load_from_json`` expects::

        {
          "<chunk_id>": {
            "tracks_data": {
              "tracks_id":            [str, ...],
              "tracks_poses":         [[[tx, ty, tz, qx, qy, qz, qw], ...], ...],
              "tracks_timestamps_us": [[int, ...], ...],
              "tracks_label_class":   [str, ...],
              "tracks_flags":         [str, ...]
            },
            "cuboidtracks_data": {
              "cuboids_dims": [[dx, dy, dz], ...]
            }
          }
        }

    Even when ``tracks`` is empty, every columnar sub-key is emitted with an
    empty list — alpasim's ``TrafficObjects.load_from_json`` unconditionally
    indexes these keys and would ``KeyError`` on a bare
    ``{"tracks_data": {}, "cuboidtracks_data": {}}``.

    Parameters
    ----------
    tracks:
        Tracks in the v1 in-memory shape (root-local frames). Duplicate
        ``track_id`` values raise ``ValueError`` — same invariant as
        :func:`serialize_tracks`.
    chunk_id:
        The single chunk key to write the columnar block under. Multi-chunk
        output is not modelled; callers who need multi-chunk output should
        partition their tracks and merge multiple dumps.
    """
    tracks_id: list[str] = []
    tracks_poses: list[list[list[float]]] = []
    tracks_timestamps_us: list[list[int]] = []
    tracks_label_class: list[str] = []
    tracks_flags: list[str] = []
    cuboids_dims: list[list[float]] = []

    seen: set[str] = set()
    for track in tracks:
        if track.track_id in seen:
            raise ValueError(f"duplicate track_id: {track.track_id!r}")
        seen.add(track.track_id)

        pose_rows: list[list[float]] = []
        ts_row: list[int] = []
        for frame in track.frames:
            tx, ty, tz = (float(v) for v in frame.translation)
            qx, qy, qz, qw = (float(v) for v in frame.rotation)
            pose_rows.append([tx, ty, tz, qx, qy, qz, qw])
            ts_row.append(int(frame.timestamp_us))

        tracks_id.append(str(track.track_id))
        tracks_poses.append(pose_rows)
        tracks_timestamps_us.append(ts_row)
        tracks_label_class.append(str(track.class_name))
        tracks_flags.append(str(track.flag))
        cuboids_dims.append([float(v) for v in track.size])

    return {
        str(chunk_id): {
            "tracks_data": {
                "tracks_id": tracks_id,
                "tracks_label_class": tracks_label_class,
                "tracks_flags": tracks_flags,
                "tracks_timestamps_us": tracks_timestamps_us,
                "tracks_poses": tracks_poses,
            },
            "cuboidtracks_data": {
                "cuboids_dims": cuboids_dims,
            },
        }
    }
