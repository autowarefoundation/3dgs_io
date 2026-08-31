"""Connect multiple scene USDZ bundles into one continuous scene.

Each input bundle carries chunks, rig trajectories and tracks in its own Z-up
ENU world frame, anchored to the Earth by ``scene.json``'s ``ecef_anchor``
(row-major 4×4 world→ECEF). Because every anchor is geodetic, connecting
scenes is pure coordinate algebra — no registration is involved::

    T_i  = inv(A_ref) @ A_i          # world_i → world_ref
    p_ref = T_i @ p_i                # bake every scene into the ref frame
    c    = bbox_centre(all points)   # one recentring shift, applied to
                                     # gaussians, rig poses and tracks alike
    A_out = A_ref @ Translate(c)     # the shift moves INTO the anchor

Getting ``A_out`` wrong is exactly the failure mode this module exists to
prevent: an earlier external pipeline subtracted the anchor's MGRS map
coordinates instead, producing a bundle that rendered fine standalone but
whose anchor decoded ~95 km away — silently breaking every geodetic consumer
(CARLA bridge, Autoware map alignment). The writer additionally cross-checks
the output anchor against the bundled Lanelet2 ``map.osm``
(:func:`~3dgs_io._geodesy.validate_anchor_against_lanelet2`).

Both ``splatsim.scene/v2`` (sidecar LiDAR files) and ``splatsim.scene/v3``
(SPZ-embedded LiDAR extension records) bundles are accepted as input; the
output is always the current v3 layout.
"""

from __future__ import annotations

import json
import logging
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import spz
from scipy.spatial.transform import Rotation

from .ext_attributes import (
    LIDAR_MASK_KEY,
    decode_lidar_extension,
    extract_lidar_extension,
)
from .frame_convention import validate_rigid_transform
from .rig_trajectories import RigPose, RigTrajectory, parse_rig_trajectories
from .scene_usdz import (
    SceneUsdzOptions,
    SceneUsdzResult,
    _apply_transform_to_cloud,
    _concat_clouds,
    _concat_ext_attrs,
    _save_bundle,
)
from .spz_io import load_spz_world
from .tracks import Track, TrackFrame, parse_tracks
from .usdz_metadata import (
    USDZ_METADATA_ARCHIVE_PATH,
    UsdzMetadata,
    load_usdz_metadata,
)

__all__ = [
    "SceneBundle",
    "connect_scene_usdzs",
    "load_scene_bundle",
]

_log = logging.getLogger(__name__)

_SUPPORTED_SCHEMAS = ("splatsim.scene/v2", "splatsim.scene/v3")

# Archive entries the connect writer rebuilds (or that describe the source
# bundle itself); everything else in the reference bundle is carried verbatim.
_REBUILT_ENTRIES = frozenset(
    {
        "default.usda",
        "scene.json",
        USDZ_METADATA_ARCHIVE_PATH,
        "tileset.json",  # v2 bundles carried a Cesium tileset; v3 dropped it
        "rig_trajectories.json",
        "sequence_tracks.json",
    }
)
# Per-frame appearance data is tied to one sequence's timestamps/cameras and
# cannot be merged meaningfully — dropped with a warning.
_DROPPED_ENTRIES = frozenset({"ppisp.json"})

# SPZ quantises positions as 24-bit fixed point with 12 fractional bits, so
# world coordinates wrap at ±4096 m. Refuse to write chunks that would wrap.
_SPZ_COORD_LIMIT_M = 4096.0


# ---------------------------------------------------------------------------
# Bundle loading
# ---------------------------------------------------------------------------


@dataclass
class SceneBundle:
    """In-memory view of one scene USDZ, in its own ENU world frame."""

    path: Path
    schema: str
    cloud: spz.GaussianCloud
    ext_attrs: dict[str, np.ndarray]
    ecef_anchor: np.ndarray  # (4, 4) float64, row-major world→ECEF
    rigs: list[RigTrajectory] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    render_defaults: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    extras: list[tuple[str, bytes]] = field(default_factory=list)
    """Non-rebuilt archive entries (``map.osm``, CARLA world, ...) verbatim."""


def _load_chunk_cloud(zf: zipfile.ZipFile, name: str) -> tuple[spz.GaussianCloud, bytes]:
    data = zf.read(name)
    with tempfile.NamedTemporaryFile(suffix=".spz") as tmp:
        tmp.write(data)
        tmp.flush()
        cloud = load_spz_world(tmp.name)
    return cloud, data


def _load_chunks_v3(
    zf: zipfile.ZipFile, gaussians: dict[str, Any], where: str
) -> tuple[list[spz.GaussianCloud], list[dict[str, np.ndarray]]]:
    chunks = gaussians.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError(f"{where}: scene/v3 bundle has no gaussians.chunks index")
    has_ext = gaussians.get("ext_attributes") is not None
    clouds: list[spz.GaussianCloud] = []
    exts: list[dict[str, np.ndarray]] = []
    for entry in chunks:
        cloud, data = _load_chunk_cloud(zf, entry["uri"])
        if cloud.num_points != int(entry["n_points"]):
            raise ValueError(
                f"{where}: chunk {entry['uri']} has {cloud.num_points} points, "
                f"index says {entry['n_points']}"
            )
        ext = extract_lidar_extension(data) if has_ext else None
        if has_ext and ext is None:
            raise ValueError(f"{where}: chunk {entry['uri']} is missing its LiDAR extension record")
        clouds.append(cloud)
        exts.append(ext or {})
    return clouds, exts


def _load_chunks_v2(
    zf: zipfile.ZipFile, gaussians: dict[str, Any], where: str
) -> tuple[list[spz.GaussianCloud], list[dict[str, np.ndarray]]]:
    names = sorted(n for n in zf.namelist() if n.startswith("chunks/") and n.endswith(".spz"))
    if not names:
        raise ValueError(f"{where}: no SPZ chunks found under chunks/")
    ext_meta = gaussians.get("ext_attributes")
    sidecar_suffix: str | None = None
    if ext_meta is not None:
        sidecar_suffix = ext_meta.get("sidecar_suffix")
        if not isinstance(sidecar_suffix, str) or not sidecar_suffix.startswith("."):
            raise ValueError(f"{where}: invalid gaussian extension sidecar_suffix")
    clouds: list[spz.GaussianCloud] = []
    exts: list[dict[str, np.ndarray]] = []
    for name in names:
        cloud, _data = _load_chunk_cloud(zf, name)
        ext: dict[str, np.ndarray] = {}
        if sidecar_suffix is not None:
            sidecar_name = str(Path(name).with_suffix(sidecar_suffix))
            ext = decode_lidar_extension(zf.read(sidecar_name))
        clouds.append(cloud)
        exts.append(ext)
    return clouds, exts


def load_scene_bundle(usdz_path: str | Path) -> SceneBundle:
    """Load a scene USDZ (v2 or v3) into memory, world-frame values verbatim."""
    usdz_path = Path(usdz_path)
    with zipfile.ZipFile(usdz_path) as zf:
        names = set(zf.namelist())
        scene = json.loads(zf.read("scene.json"))
        schema = scene.get("schema")
        if schema not in _SUPPORTED_SCHEMAS:
            raise ValueError(
                f"{usdz_path}: unsupported scene schema {schema!r}; "
                f"expected one of {_SUPPORTED_SCHEMAS}"
            )
        anchor = np.asarray(scene["world"]["ecef_anchor"], dtype=np.float64)
        validate_rigid_transform(anchor, where=f"{usdz_path} world.ecef_anchor")

        gaussians = scene["gaussians"]
        if schema == "splatsim.scene/v3":
            clouds, exts = _load_chunks_v3(zf, gaussians, str(usdz_path))
        else:
            clouds, exts = _load_chunks_v2(zf, gaussians, str(usdz_path))
        counts = [c.num_points for c in clouds]
        cloud = clouds[0] if len(clouds) == 1 else _concat_clouds(clouds)
        ext_attrs = _concat_ext_attrs(exts, counts) if any(exts) else {}

        rigs: list[RigTrajectory] = []
        if "rig_trajectories.json" in names:
            rigs = parse_rig_trajectories(json.loads(zf.read("rig_trajectories.json")))
        tracks: list[Track] = []
        if "sequence_tracks.json" in names:
            tracks = parse_tracks(json.loads(zf.read("sequence_tracks.json")))

        extras: list[tuple[str, bytes]] = []
        for name in sorted(names):
            if name in _REBUILT_ENTRIES or name.startswith("chunks/"):
                continue
            if name in _DROPPED_ENTRIES:
                _log.warning(
                    "%s: dropping %s (per-frame appearance data cannot be merged)",
                    usdz_path,
                    name,
                )
                continue
            extras.append((name, zf.read(name)))

        metadata: dict[str, Any] = {}
        if USDZ_METADATA_ARCHIVE_PATH in names:
            metadata = load_usdz_metadata(zf.read(USDZ_METADATA_ARCHIVE_PATH)).to_dict()

    return SceneBundle(
        path=usdz_path,
        schema=str(schema),
        cloud=cloud,
        ext_attrs=ext_attrs,
        ecef_anchor=anchor,
        rigs=rigs,
        tracks=tracks,
        render_defaults=dict(scene.get("render_defaults") or {}),
        metadata=metadata,
        extras=extras,
    )


# ---------------------------------------------------------------------------
# Rigid-transform helpers (row-major, child-to-parent, column vectors)
# ---------------------------------------------------------------------------


def _transform_pose(
    transform: np.ndarray, translation: tuple[float, ...], rotation_xyzw: tuple[float, ...]
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    r = transform[:3, :3]
    t = transform[:3, 3]
    new_t = r @ np.asarray(translation, dtype=np.float64) + t
    new_q = (Rotation.from_matrix(r) * Rotation.from_quat(rotation_xyzw)).as_quat()
    new_q = new_q / np.linalg.norm(new_q)
    return tuple(float(v) for v in new_t), tuple(float(v) for v in new_q)


def _harmonise_lidar_masks(bundles: list[SceneBundle]) -> None:
    """Give mask-less scenes an all-ones mask when any other scene has one.

    ``_concat_ext_attrs`` zero-fills missing keys, but a zero ``lidar_mask``
    means "appearance only, exclude from LiDAR" — the opposite of the absent
    mask's meaning ("all participate"). Pre-fill with ones so semantics
    survive the merge.
    """
    if not any(LIDAR_MASK_KEY in b.ext_attrs for b in bundles):
        return
    for b in bundles:
        if b.ext_attrs and LIDAR_MASK_KEY not in b.ext_attrs:
            b.ext_attrs[LIDAR_MASK_KEY] = np.ones(b.cloud.num_points, dtype=np.float32)
            _log.info("%s: no lidar_mask; filling with ones (all participate)", b.path.name)


def _merge_rigs(bundles: list[SceneBundle], transforms: list[np.ndarray]) -> list[RigTrajectory]:
    """Concatenate per-scene rig trajectories in the connected world frame.

    Poses of rigs sharing a ``rig_id`` are merged (sorted by timestamp);
    sensor calibrations are taken from the first scene that defines the rig,
    with a warning when a later scene disagrees on the sensor set.
    """
    merged: dict[str, RigTrajectory] = {}
    for bundle, transform in zip(bundles, transforms, strict=True):
        for rig in bundle.rigs:
            poses = [
                RigPose(
                    timestamp_us=pose.timestamp_us,
                    **_transform_pose_as_kwargs(transform, pose),
                )
                for pose in rig.poses
            ]
            existing = merged.get(rig.rig_id)
            if existing is None:
                merged[rig.rig_id] = RigTrajectory(
                    rig_id=rig.rig_id,
                    poses=poses,
                    cameras=rig.cameras,
                    lidars=rig.lidars,
                    metadata=dict(rig.metadata),
                )
                continue
            own = {s.name for s in rig.cameras} | {s.name for s in rig.lidars}
            ref = {s.name for s in existing.cameras} | {s.name for s in existing.lidars}
            if own != ref:
                _log.warning(
                    "rig %r: sensor sets differ between scenes (%s vs %s); "
                    "keeping the first scene's calibration",
                    rig.rig_id,
                    sorted(ref),
                    sorted(own),
                )
            existing.poses.extend(poses)
    for rig in merged.values():
        rig.poses.sort(key=lambda p: p.timestamp_us)
    return list(merged.values())


def _transform_pose_as_kwargs(transform: np.ndarray, pose: RigPose | TrackFrame) -> dict[str, Any]:
    translation, rotation = _transform_pose(transform, pose.translation, pose.rotation)
    return {"translation": translation, "rotation": rotation}


def _merge_tracks(bundles: list[SceneBundle], transforms: list[np.ndarray]) -> list[Track]:
    """Concatenate tracks in the connected frame, de-duplicating track ids."""
    merged: list[Track] = []
    seen_ids: set[str] = set()
    for scene_idx, (bundle, transform) in enumerate(zip(bundles, transforms, strict=True)):
        for track in bundle.tracks:
            frames = [
                TrackFrame(
                    timestamp_us=f.timestamp_us,
                    **_transform_pose_as_kwargs(transform, f),
                )
                for f in track.frames
            ]
            track_id = track.track_id
            if track_id in seen_ids:
                track_id = f"scene{scene_idx}/{track_id}"
            seen_ids.add(track_id)
            merged.append(
                Track(
                    track_id=track_id,
                    class_name=track.class_name,
                    size=track.size,
                    frames=frames,
                    flag=track.flag,
                    metadata=dict(track.metadata),
                )
            )
    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def connect_scene_usdzs(
    inputs: list[str | Path],
    out_path: str | Path,
    *,
    chunk_size: float = 50.0,
    max_points_per_chunk: int = 200_000,
    metadata: UsdzMetadata | None = None,
) -> SceneUsdzResult:
    """Merge multiple scene USDZ bundles into one connected bundle.

    The first input is the *reference* scene: its world frame orientation,
    ``map.osm`` / CARLA world / other extras are carried into the output.
    Every other scene is baked into the reference frame through the geodetic
    anchors (``inv(A_ref) @ A_i``); the merged geometry is then recentred at
    its bounding-box centre and that shift is folded into the output
    ``ecef_anchor`` — so the connected bundle stays exactly as geodetically
    aligned as its inputs.

    Parameters
    ----------
    inputs:
        Two or more scene USDZ paths (``splatsim.scene/v2`` or ``v3``). The
        first entry is the reference scene.
    out_path:
        Destination ``.usdz`` (always written in the v3 layout).
    chunk_size / max_points_per_chunk:
        Spatial re-chunking of the merged cloud. Geometry itself passes
        through unfiltered — no scale clamping, no opacity or bbox filtering
        (inputs were already filtered when they were built) — and render
        defaults are taken from the reference scene.
    metadata:
        Output ``metadata.yaml`` identity. Defaults to a fresh UUID with
        ``scene_id`` set to the output filename stem.
    """
    if len(inputs) < 2:
        raise ValueError("connect_scene_usdzs needs at least two input bundles")
    out_path = Path(out_path)

    bundles = [load_scene_bundle(p) for p in inputs]
    reference = bundles[0]

    rd = reference.render_defaults
    options = SceneUsdzOptions(
        chunk_size=chunk_size,
        max_points_per_chunk=max_points_per_chunk,
        min_scale=0.0,
        max_aspect_ratio=float("inf"),
        opacity_threshold=0.0,
        bbox_radius=float("inf"),
        exposure=float(rd.get("exposure", 1.6)),
        near_plane=float(rd.get("near_plane", 0.5)),
        far_plane=float(rd.get("far_plane", 300.0)),
    )

    # world_i → world_ref transforms through the geodetic anchors.
    anchor_ref = reference.ecef_anchor
    transforms = [np.linalg.solve(anchor_ref, b.ecef_anchor) for b in bundles]
    for b, t in zip(bundles, transforms, strict=True):
        validate_rigid_transform(t, where=f"{b.path.name} world→reference transform")
        offset = float(np.linalg.norm(t[:3, 3]))
        _log.info("connect: %s sits %.1f m from the reference frame origin", b.path.name, offset)

    _harmonise_lidar_masks(bundles)
    required = {k for b in bundles for k in b.ext_attrs}
    for b in bundles:
        if required and not b.ext_attrs:
            raise ValueError(
                f"{b.path.name} carries no LiDAR ext attributes but other inputs do; "
                "connecting them would zero-fill semantic channels"
            )

    clouds: list[spz.GaussianCloud] = []
    counts: list[int] = []
    for b, t in zip(bundles, transforms, strict=True):
        cloud = b.cloud if np.allclose(t, np.eye(4)) else _apply_transform_to_cloud(b.cloud, t)
        clouds.append(cloud)
        counts.append(cloud.num_points)
    cloud = _concat_clouds(clouds)
    ext_attrs = _concat_ext_attrs([b.ext_attrs for b in bundles], counts)

    # One recentring shift for gaussians, rig poses and tracks alike; the
    # shift moves INTO the anchor so geodetic placement is preserved.
    n = cloud.num_points
    positions = np.array(cloud.positions, dtype=np.float64).reshape(n, 3)
    centre = (positions.min(axis=0) + positions.max(axis=0)) / 2.0
    positions -= centre
    half_extent = float(np.abs(positions).max())
    if half_extent >= _SPZ_COORD_LIMIT_M:
        raise ValueError(
            f"connected scene spans ±{half_extent:.0f} m after recentring, beyond "
            f"the ±{_SPZ_COORD_LIMIT_M:.0f} m SPZ fixed-point range — the inputs "
            "are too far apart to share one bundle"
        )
    cloud.positions = positions.astype(np.float32).reshape(-1)

    recentre = np.eye(4, dtype=np.float64)
    recentre[:3, 3] = -centre
    ecef_anchor = anchor_ref @ np.linalg.inv(recentre)
    validate_rigid_transform(ecef_anchor, where="connected ecef_anchor")

    pose_transforms = [recentre @ t for t in transforms]
    rigs = _merge_rigs(bundles, pose_transforms)
    tracks = _merge_tracks(bundles, pose_transforms)

    producer_source = {
        "source_scenes": [
            {
                "path": b.path.name,
                "uuid": b.metadata.get("uuid"),
                "scene_id": b.metadata.get("scene_id"),
            }
            for b in bundles
        ]
    }

    return _save_bundle(
        cloud=cloud,
        ext_attrs=ext_attrs,
        ecef_anchor=ecef_anchor.tolist(),
        out_path=out_path,
        options=options,
        metadata=metadata,
        producer_source=producer_source,
        extra_payloads=reference.extras,
        tracks=tracks or None,
        rig_trajectories=rigs or None,
    )
