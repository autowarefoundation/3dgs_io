"""Sensor-rig (ego / multi-rig) trajectory dataclasses + JSON (de)serialisation.

A :class:`RigTrajectory` is a time-series of :class:`RigPose` samples for a
single sensor rig (typically the ego vehicle). Poses live in the bundle's
**root-local frame** (the same coordinate system as the embedded SPZ chunks,
the cameras and the dynamic-object tracks), so applying the output USDZ's
``tileset.json.root.transform`` after the rig pose lifts it into world space.

On-disk schema (``rig_trajectories.json`` inside the USDZ; also accepted as a
standalone JSON file by the CLI) — ``splatsim.rig_trajectories/v1``::

    {
      "schema": "splatsim.rig_trajectories/v1",
      "frame": "root_local",
      "rigs": [
        {
          "rig_id": "ego",
          "poses": [
            {
              "timestamp_us": 27567868848,
              "translation": [113.62, -58.55, 1.92],
              "rotation":    [-0.0005, -0.0113, 0.6645, 0.7472]
            },
            ...
          ],
          "metadata": {"sequence_id": "..."}
        },
        ...
      ]
    }

This is **not** byte-compatible with alpasim's ``rig_trajectories.json`` —
alpasim stores per-rig ``T_rig_worlds`` (a list of 4×4 matrices, relative to
the top-level ``T_world_base`` and ``world_to_nre``). Use
:func:`parse_alpasim_rig_trajectories` to ingest alpasim documents into our
schema: the alpasim ingester composes the global frames and extracts a clean
(translation, xyzw quaternion) tuple per timestamp.
"""

from __future__ import annotations

import json
import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "RIG_TRAJECTORIES_SCHEMA",
    "RigPose",
    "RigTrajectory",
    "load_rig_trajectories_from_usdz",
    "parse_alpasim_rig_trajectories",
    "parse_rig_trajectories",
    "serialize_rig_trajectories",
]


RIG_TRAJECTORIES_SCHEMA = "splatsim.rig_trajectories/v1"
_FRAME = "root_local"

_log = logging.getLogger(__name__)


@dataclass
class RigPose:
    """One timestamped rig-to-scene pose."""

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
    def from_dict(cls, d: dict[str, Any]) -> RigPose:
        tr = d["translation"]
        ro = d["rotation"]
        if len(tr) != 3:
            raise ValueError(f"pose.translation must have 3 elements, got {len(tr)}")
        if len(ro) != 4:
            raise ValueError(f"pose.rotation must have 4 elements (xyzw), got {len(ro)}")
        return cls(
            timestamp_us=int(d["timestamp_us"]),
            translation=(float(tr[0]), float(tr[1]), float(tr[2])),
            rotation=(float(ro[0]), float(ro[1]), float(ro[2]), float(ro[3])),
        )


@dataclass
class RigTrajectory:
    """A sensor rig (ego or other) and its sequence of pose samples."""

    rig_id: str
    poses: list[RigPose] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rig_id": str(self.rig_id),
            "poses": [p.to_dict() for p in self.poses],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RigTrajectory:
        return cls(
            rig_id=str(d["rig_id"]),
            poses=[RigPose.from_dict(p) for p in d.get("poses") or []],
            metadata=dict(d.get("metadata") or {}),
        )


# ---------------------------------------------------------------------------
# Collection-level (de)serialisation
# ---------------------------------------------------------------------------


def serialize_rig_trajectories(rigs: list[RigTrajectory]) -> dict[str, Any]:
    """Build the JSON-ready ``splatsim.rig_trajectories/v1`` document."""
    seen: set[str] = set()
    out_list: list[dict[str, Any]] = []
    for rig in rigs:
        if rig.rig_id in seen:
            raise ValueError(f"duplicate rig_id: {rig.rig_id!r}")
        seen.add(rig.rig_id)
        out_list.append(rig.to_dict())
    return {
        "schema": RIG_TRAJECTORIES_SCHEMA,
        "frame": _FRAME,
        "rigs": out_list,
    }


def parse_rig_trajectories(doc: dict[str, Any]) -> list[RigTrajectory]:
    """Inverse of :func:`serialize_rig_trajectories`."""
    schema = doc.get("schema")
    if schema != RIG_TRAJECTORIES_SCHEMA:
        raise ValueError(
            f"unexpected rig_trajectories schema {schema!r}; expected {RIG_TRAJECTORIES_SCHEMA!r}"
        )
    raw = doc.get("rigs")
    if not isinstance(raw, list):
        raise ValueError("rig_trajectories document is missing the 'rigs' list")
    out: list[RigTrajectory] = []
    seen: set[str] = set()
    for entry in raw:
        rig = RigTrajectory.from_dict(entry)
        if rig.rig_id in seen:
            raise ValueError(f"duplicate rig_id: {rig.rig_id!r}")
        seen.add(rig.rig_id)
        out.append(rig)
    return out


def load_rig_trajectories_from_usdz(path: str | Path) -> list[RigTrajectory]:
    """Read ``rig_trajectories.json`` from a USDZ produced by :func:`save_scene_usdz`."""
    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        if "rig_trajectories.json" not in zf.namelist():
            raise FileNotFoundError(f"{path}: no rig_trajectories.json entry in archive")
        doc = json.loads(zf.read("rig_trajectories.json").decode("utf-8-sig"))
    return parse_rig_trajectories(doc)


# ---------------------------------------------------------------------------
# alpasim rig_trajectories.json ingestion
# ---------------------------------------------------------------------------

# Top-level alpasim fields we read. Anything else gets a debug warning.
_ALPASIM_KNOWN_TOP_KEYS = frozenset(
    {
        "T_world_base",
        "world_to_nre",
        "rig_trajectories",
        "camera_calibrations",
        "lidar_calibrations",
    }
)
_ALPASIM_KNOWN_RIG_KEYS = frozenset(
    {
        "sequence_id",
        "rig_bbox",
        "cameras_linear_start_frame_indices",
        "lidars_linear_start_frame_indices",
        "cameras_frame_timestamps_us",
        "lidars_frame_timestamps_us",
        "T_rig_worlds",
        "T_rig_world_timestamps_us",
        "cameras_frame_T_rig_worlds",
    }
)


def _as_4x4(m: Any) -> np.ndarray:
    arr = np.array(m, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"expected 4x4 matrix, got shape {arr.shape}")
    return arr


_Translation = tuple[float, float, float]
_Quaternion = tuple[float, float, float, float]


def _pose_from_matrix(m: np.ndarray) -> tuple[_Translation, _Quaternion]:
    """Extract translation + xyzw quaternion from a 4×4 row-major rigid transform."""
    r = m[:3, :3]
    t = m[:3, 3]
    # Shepperd's method (same algorithm as CameraExtrinsics.from_matrix).
    trace = float(r[0, 0] + r[1, 1] + r[2, 2])
    if trace > 0.0:
        s = 2.0 * np.sqrt(1.0 + trace)
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    return (float(t[0]), float(t[1]), float(t[2])), (float(x), float(y), float(z), float(w))


def parse_alpasim_rig_trajectories(doc: dict[str, Any]) -> list[RigTrajectory]:
    """Ingest an alpasim ``rig_trajectories.json`` document.

    The alpasim layout is::

        {
          "T_world_base":  [[...], [...], [...], [...]],      # base → world
          "world_to_nre":  {"matrix": [[...], ...]},          # world → NRE
          "rig_trajectories": [
            {
              "sequence_id": str,
              "T_rig_worlds":            [4x4, 4x4, ...],     # rig → "world" (= base, in practice)
              "T_rig_world_timestamps_us": [int, int, ...],
              "rig_bbox": {...},
              "cameras_frame_T_rig_worlds": {camera: [4x4, ...], ...},
              "cameras_frame_timestamps_us": {camera: [int, ...], ...},
              ...
            }, ...
          ],
          "camera_calibrations": {...},
          "lidar_calibrations":  {...},
        }

    Per-frame poses are returned in our **root-local** frame, computed as

        M_root_local = world_to_nre @ T_rig_worlds[i]

    ``T_world_base`` is the global ECEF anchor of the recording (the same
    role as Cesium's ``tileset.json.root.transform``) and is **not** part of
    the per-frame composition: alpasim's ``T_rig_worlds`` is already
    expressed in the recording's base frame, and ``world_to_nre`` shifts
    that base into the NRE-local (= root-local) frame used by the SPZ
    payload. ``T_world_base`` is exposed via the returned trajectory's
    ``metadata["T_world_base"]`` for callers that want to recover the ECEF
    anchor explicitly.

    Camera / LiDAR calibrations and per-sensor frame timestamps are out of
    scope of this ingester — use the :mod:`3dgs_io.cameras` module for
    camera intrinsics + extrinsics.
    """
    if not isinstance(doc, dict):
        raise ValueError("alpasim rig_trajectories document must be a dict at the top level")

    unknown = set(doc) - _ALPASIM_KNOWN_TOP_KEYS
    if unknown:
        _log.warning(
            "alpasim rig_trajectories: dropping unknown top-level keys %s",
            sorted(unknown),
        )

    T_world_base_raw = doc.get("T_world_base")
    w2n_raw = doc.get("world_to_nre")
    if isinstance(w2n_raw, dict) and "matrix" in w2n_raw:
        world_to_nre = _as_4x4(w2n_raw["matrix"])
    elif w2n_raw is None:
        world_to_nre = np.eye(4)
    else:
        world_to_nre = _as_4x4(w2n_raw)

    rigs_in = doc.get("rig_trajectories") or []
    if not isinstance(rigs_in, list):
        raise ValueError("alpasim rig_trajectories: 'rig_trajectories' must be a list")

    out: list[RigTrajectory] = []
    auto_ids: set[str] = set()
    for idx, rig in enumerate(rigs_in):
        if not isinstance(rig, dict):
            continue
        unknown_rig = set(rig) - _ALPASIM_KNOWN_RIG_KEYS
        if unknown_rig:
            _log.warning("alpasim rig[%d]: dropping unknown keys %s", idx, sorted(unknown_rig))
        seq_id = rig.get("sequence_id")
        # Use sequence_id when present, otherwise synthesise a stable id.
        rig_id = str(seq_id) if seq_id else f"rig_{idx}"
        if rig_id in auto_ids:
            rig_id = f"{rig_id}#{idx}"
        auto_ids.add(rig_id)

        mats = rig.get("T_rig_worlds") or []
        ts = rig.get("T_rig_world_timestamps_us") or []
        if len(mats) != len(ts):
            raise ValueError(
                f"alpasim rig {rig_id!r}: T_rig_worlds length {len(mats)} != "
                f"timestamps length {len(ts)}"
            )

        poses: list[RigPose] = []
        for mat, t_us in zip(mats, ts, strict=True):
            # ``T_rig_worlds[i]`` is the rig-to-base transform (despite the
            # name, it is NOT in the ECEF "world" frame); ``world_to_nre``
            # shifts the base into NRE-local. See the function docstring.
            T_rig_in_base = _as_4x4(mat)
            M_root_local = world_to_nre @ T_rig_in_base
            translation, rotation = _pose_from_matrix(M_root_local)
            poses.append(
                RigPose(timestamp_us=int(t_us), translation=translation, rotation=rotation)
            )

        metadata: dict[str, Any] = {}
        if seq_id:
            metadata["sequence_id"] = str(seq_id)
        bbox = rig.get("rig_bbox")
        if bbox is not None:
            metadata["rig_bbox"] = bbox
        if T_world_base_raw is not None:
            # Preserve the ECEF anchor for callers that need to recover
            # absolute world coordinates from the NRE-local rig pose.
            metadata["T_world_base"] = T_world_base_raw
        out.append(RigTrajectory(rig_id=rig_id, poses=poses, metadata=metadata))

    return out
