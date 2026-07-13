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

import logging
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .cameras import Camera, CameraExtrinsics, CameraModel

__all__ = [
    "RIG_TRAJECTORIES_SCHEMA",
    "RigPose",
    "RigTrajectory",
    "dump_alpasim_rig_trajectories",
    "load_rig_trajectories_doc",
    "parse_alpasim_rig_trajectories",
    "parse_rig_trajectories",
    "serialize_rig_trajectories",
    "update_camera_intrinsics",
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
    """A sensor rig (ego or other) with its pose time-series and its cameras.

    ``cameras`` lists the cameras physically mounted on this rig. Each
    camera's :attr:`Camera.extrinsics` is **rig-relative** (``T_sensor_rig``);
    compose with the rig's pose at a given timestamp to get the camera pose
    in scene coordinates.
    """

    rig_id: str
    poses: list[RigPose] = field(default_factory=list)
    cameras: list[Camera] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rig_id": str(self.rig_id),
            "poses": [p.to_dict() for p in self.poses],
            "cameras": [c.to_dict() for c in self.cameras],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RigTrajectory:
        return cls(
            rig_id=str(d["rig_id"]),
            poses=[RigPose.from_dict(p) for p in d.get("poses") or []],
            cameras=[Camera.from_dict(c) for c in d.get("cameras") or []],
            metadata=dict(d.get("metadata") or {}),
        )


# ---------------------------------------------------------------------------
# Collection-level (de)serialisation
# ---------------------------------------------------------------------------


def serialize_rig_trajectories(rigs: list[RigTrajectory]) -> dict[str, Any]:
    """Build the JSON-ready ``splatsim.rig_trajectories/v1`` document.

    Enforces unique ``rig_id`` across rigs and unique ``name`` for cameras
    within each rig — duplicates would silently collapse on round-trip when
    downstream tooling keys by name.
    """
    seen_rigs: set[str] = set()
    out_list: list[dict[str, Any]] = []
    for rig in rigs:
        if rig.rig_id in seen_rigs:
            raise ValueError(f"duplicate rig_id: {rig.rig_id!r}")
        seen_rigs.add(rig.rig_id)
        seen_cams: set[str] = set()
        for cam in rig.cameras:
            if cam.name in seen_cams:
                raise ValueError(f"duplicate camera name {cam.name!r} in rig {rig.rig_id!r}")
            seen_cams.add(cam.name)
        out_list.append(rig.to_dict())
    return {
        "schema": RIG_TRAJECTORIES_SCHEMA,
        "frame": _FRAME,
        "rigs": out_list,
    }


def update_camera_intrinsics(
    rigs: list[RigTrajectory],
    *,
    camera_name: str,
    rig_id: str | None = None,
    **updates: Any,
) -> Camera:
    """Replace the intrinsics of a camera mounted on one of ``rigs`` in place.

    Looks up the camera by ``camera_name`` (optionally scoped to ``rig_id``)
    and rebuilds its :attr:`Camera.camera_model` via
    :meth:`CameraModel.with_intrinsics`. The matched :class:`Camera` is returned
    for callers that want to inspect the result.

    Raises ``ValueError`` if no camera matches, if ``rig_id`` is given but does
    not name any rig, or if the camera name is ambiguous (multiple rigs, or
    multiple entries on the same rig, an in-memory state the parser/serializer
    would otherwise reject).
    """
    if not updates:
        raise ValueError("update_camera_intrinsics requires at least one intrinsic update")

    if rig_id is not None and not any(r.rig_id == rig_id for r in rigs):
        raise ValueError(f"rig {rig_id!r} not found")

    matches: list[tuple[RigTrajectory, int]] = []
    for rig in rigs:
        if rig_id is not None and rig.rig_id != rig_id:
            continue
        for i, cam in enumerate(rig.cameras):
            if cam.name == camera_name:
                matches.append((rig, i))

    if not matches:
        scope = f" on rig {rig_id!r}" if rig_id is not None else ""
        raise ValueError(f"camera {camera_name!r} not found{scope}")
    if len(matches) > 1:
        match_rig_ids = sorted({rig.rig_id for rig, _ in matches})
        if len(match_rig_ids) == 1:
            raise ValueError(
                f"camera {camera_name!r} appears multiple times on rig "
                f"{match_rig_ids[0]!r}; camera names must be unique within a rig"
            )
        raise ValueError(
            f"camera {camera_name!r} is mounted on multiple rigs {match_rig_ids}; "
            f"pass rig_id= to disambiguate"
        )

    rig, idx = matches[0]
    cam = rig.cameras[idx]
    rig.cameras[idx] = replace(
        cam,
        camera_model=cam.camera_model.with_intrinsics(**updates),
    )
    return rig.cameras[idx]


def load_rig_trajectories_doc(doc: dict[str, Any]) -> list[RigTrajectory]:
    """Parse either a ``splatsim.rig_trajectories/v1`` doc or an alpasim one.

    Single dispatcher used by CLIs and library callers that accept both
    schemas. Dispatch is by the top-level ``schema`` key: matches the
    splatsim constant → :func:`parse_rig_trajectories`; otherwise falls
    through to :func:`parse_alpasim_rig_trajectories` (which raises with an
    alpasim-shaped error if the document is neither).
    """
    if not isinstance(doc, dict):
        raise ValueError(
            f"rig_trajectories document must be a JSON object at top level, "
            f"got {type(doc).__name__}"
        )
    if doc.get("schema") == RIG_TRAJECTORIES_SCHEMA:
        return parse_rig_trajectories(doc)
    return parse_alpasim_rig_trajectories(doc)


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
        seen_cams: set[str] = set()
        for cam in rig.cameras:
            if cam.name in seen_cams:
                raise ValueError(f"duplicate camera name {cam.name!r} in rig {rig.rig_id!r}")
            seen_cams.add(cam.name)
        out.append(rig)
    return out


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
    t = m[:3, 3]
    x, y, z, w = Rotation.from_matrix(m[:3, :3]).as_quat()
    return (
        (float(t[0]), float(t[1]), float(t[2])),
        (float(x), float(y), float(z), float(w)),
    )


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

    Cameras are pulled out of the top-level ``camera_calibrations`` dict and
    attached to each rig under :attr:`RigTrajectory.cameras`. Membership is
    decided per rig from the keys of ``cameras_frame_timestamps_us`` (or
    ``cameras_frame_T_rig_worlds`` as a fallback): a camera name appearing in
    a rig's per-frame data belongs to that rig. If neither field is present
    and only one rig exists, every top-level camera is attached to it.

    LiDAR calibrations and per-sensor frame timestamps are out of scope.
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
    if w2n_raw is None:
        world_to_nre = np.eye(4)
    elif isinstance(w2n_raw, dict):
        if "matrix" not in w2n_raw:
            raise ValueError(
                "alpasim world_to_nre is a dict but is missing the 'matrix' key; "
                f"got keys {sorted(w2n_raw)}"
            )
        world_to_nre = _as_4x4(w2n_raw["matrix"])
    else:
        world_to_nre = _as_4x4(w2n_raw)

    rigs_in = doc.get("rig_trajectories") or []
    if not isinstance(rigs_in, list):
        raise ValueError("alpasim rig_trajectories: 'rig_trajectories' must be a list")

    cam_calibs_raw = doc.get("camera_calibrations") or {}
    if not isinstance(cam_calibs_raw, dict):
        raise ValueError("alpasim rig_trajectories: 'camera_calibrations' must be a dict")

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

        cameras_for_rig = _attach_alpasim_cameras_to_rig(
            rig=rig,
            cam_calibs_raw=cam_calibs_raw,
            is_single_rig=(len(rigs_in) == 1),
            rig_log_id=rig_id,
        )
        out.append(
            RigTrajectory(
                rig_id=rig_id,
                poses=poses,
                cameras=cameras_for_rig,
                metadata=metadata,
            )
        )

    return out


def _attach_alpasim_cameras_to_rig(
    *,
    rig: dict[str, Any],
    cam_calibs_raw: dict[str, Any],
    is_single_rig: bool,
    rig_log_id: str,
) -> list[Camera]:
    """Pick the cameras belonging to ``rig`` out of alpasim's flat top-level dict.

    Membership comes from the rig's ``cameras_frame_timestamps_us`` /
    ``cameras_frame_T_rig_worlds`` keys. If neither field is present and the
    document only has one rig, every top-level camera is assigned to it.
    """
    member_names: set[str] = set()
    for key in ("cameras_frame_timestamps_us", "cameras_frame_T_rig_worlds"):
        v = rig.get(key)
        if isinstance(v, dict):
            member_names.update(v.keys())
    if not member_names and is_single_rig:
        member_names = set(cam_calibs_raw.keys())

    cameras: list[Camera] = []
    for cam_name in sorted(member_names):
        entry = cam_calibs_raw.get(cam_name)
        if not isinstance(entry, dict):
            _log.warning(
                "alpasim rig %r: camera %r referenced by per-frame data but "
                "missing from camera_calibrations; skipping",
                rig_log_id,
                cam_name,
            )
            continue
        try:
            extrinsics = CameraExtrinsics.from_t_sensor_rig(entry["T_sensor_rig"])
        except (KeyError, ValueError) as e:
            _log.warning(
                "alpasim camera %r: cannot parse T_sensor_rig (%s); skipping",
                cam_name,
                e,
            )
            continue
        cam_model_raw = entry.get("camera_model") or {}
        if not isinstance(cam_model_raw, dict) or "type" not in cam_model_raw:
            _log.warning(
                "alpasim camera %r: missing or malformed camera_model; skipping",
                cam_name,
            )
            continue
        try:
            cam_model = CameraModel.from_dict(cam_model_raw)
        except ValueError as e:
            _log.warning(
                "alpasim camera %r: invalid camera_model intrinsics (%s); skipping",
                cam_name,
                e,
            )
            continue
        meta: dict[str, Any] = {}
        for k in ("sequence_id", "logical_sensor_name", "unique_sensor_idx"):
            if k in entry:
                meta[k] = entry[k]
        cameras.append(
            Camera(
                name=str(cam_name),
                camera_model=cam_model,
                extrinsics=extrinsics,
                metadata=meta,
            )
        )
    return cameras


def _matrix_from_rigpose(pose: RigPose) -> np.ndarray:
    """Inverse of :func:`_pose_from_matrix`: (t, quat_xyzw) → 4×4."""
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = Rotation.from_quat(list(pose.rotation)).as_matrix()
    m[:3, 3] = np.array(pose.translation, dtype=np.float64)
    return m


def dump_alpasim_rig_trajectories(
    rigs: list[RigTrajectory],
    *,
    world_to_nre: Any | None = None,
    t_world_base: Any | None = None,
) -> dict[str, Any]:
    """Serialize :class:`RigTrajectory` list into an alpasim ``rig_trajectories.json`` document.

    The parse direction (:func:`parse_alpasim_rig_trajectories`) composes each
    rig pose as ``M_root_local = world_to_nre @ T_rig_in_base`` and stores
    ``M_root_local`` on :class:`RigPose`. Here we reverse that: we recover
    ``T_rig_in_base = inv(world_to_nre) @ M_root_local`` and emit it as
    ``T_rig_worlds[i]`` (alpasim's legacy naming — despite "world" the field
    is in the base frame).

    Parameters
    ----------
    rigs:
        Rigs (already in the v1 in-memory shape, root-local poses).
    world_to_nre:
        4×4 world→NRE transform to emit. If ``None`` (typical for USDZs whose
        v1 rig never stored this), an identity matrix is used, and the poses
        are written verbatim (equivalent, since parse would compose them back
        to the same root-local frame).
    t_world_base:
        4×4 base→ECEF transform. If ``None`` the value is pulled from the
        first rig's ``metadata['T_world_base']`` if present, otherwise it is
        omitted (alpasim runtime treats a missing value as identity).
    """
    if world_to_nre is None:
        w2n = np.eye(4, dtype=np.float64)
    else:
        w2n = _as_4x4(world_to_nre)
    inv_w2n = np.linalg.inv(w2n)

    if t_world_base is None:
        for rig in rigs:
            candidate = rig.metadata.get("T_world_base")
            if candidate is not None:
                t_world_base = candidate
                break
    twb_matrix: np.ndarray | None = None
    if t_world_base is not None:
        twb_matrix = _as_4x4(t_world_base)

    camera_calibrations: dict[str, dict[str, Any]] = {}
    for rig in rigs:
        for cam in rig.cameras:
            if cam.name in camera_calibrations:
                raise ValueError(
                    f"camera name {cam.name!r} appears in multiple rigs; "
                    "alpasim camera_calibrations is a flat dict and requires unique names"
                )
            entry: dict[str, Any] = {
                "T_sensor_rig": cam.extrinsics.to_t_sensor_rig(),
                "camera_model": cam.camera_model.to_dict(),
                "logical_sensor_name": cam.metadata.get("logical_sensor_name", cam.name),
            }
            unique_idx = cam.metadata.get("unique_sensor_idx")
            if unique_idx is not None:
                entry["unique_sensor_idx"] = unique_idx
            camera_calibrations[cam.name] = entry

    rig_trajectories_out: list[dict[str, Any]] = []
    for rig in rigs:
        poses_sorted = sorted(rig.poses, key=lambda p: p.timestamp_us)
        ts_list = [int(p.timestamp_us) for p in poses_sorted]
        t_rig_worlds: list[list[list[float]]] = []
        for pose in poses_sorted:
            m_root_local = _matrix_from_rigpose(pose)
            t_rig_in_base = inv_w2n @ m_root_local
            t_rig_worlds.append(t_rig_in_base.tolist())

        cameras_frame_ts: dict[str, list[list[int]]] = {}
        if ts_list and rig.cameras:
            deltas = [ts_list[i + 1] - ts_list[i] for i in range(len(ts_list) - 1)]
            positive = [d for d in deltas if d > 0]
            median_dt = sorted(positive)[len(positive) // 2] if positive else 1
            ranges: list[list[int]] = [
                [ts_list[i], ts_list[i + 1]] for i in range(len(ts_list) - 1)
            ]
            ranges.append([ts_list[-1], ts_list[-1] + int(median_dt)])
            for cam in rig.cameras:
                cameras_frame_ts[cam.name] = [list(r) for r in ranges]

        sequence_id = rig.metadata.get("sequence_id") or rig.rig_id
        rig_dict: dict[str, Any] = {
            "sequence_id": str(sequence_id),
            "T_rig_world_timestamps_us": ts_list,
            "T_rig_worlds": t_rig_worlds,
        }
        if cameras_frame_ts:
            rig_dict["cameras_frame_timestamps_us"] = cameras_frame_ts
        rig_bbox = rig.metadata.get("rig_bbox")
        if rig_bbox is not None:
            rig_dict["rig_bbox"] = rig_bbox
        rig_trajectories_out.append(rig_dict)

    doc: dict[str, Any] = {}
    if twb_matrix is not None:
        doc["T_world_base"] = twb_matrix.tolist()
    doc["world_to_nre"] = {"matrix": w2n.tolist()}
    doc["rig_trajectories"] = rig_trajectories_out
    doc["camera_calibrations"] = camera_calibrations
    return doc
