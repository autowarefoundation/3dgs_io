"""Single-file, frame-explicit USDZ scene writer.

Input glTF/RUB gaussians are baked into the bundle's Z-up ENU world frame.
The Cesium root anchor is reconciled by the inverse rotation, preserving the
same ECEF placement without leaving an implicit axis conversion to consumers.

Output USDZ layout (one ``ZIP_STORED`` archive, all entries uncompressed)::

    default.usda                 # USDZ root stage (asset reference to scene.json)
    metadata.yaml                # identity card (uuid / scene_id / version_string)
    scene.json                   # splatsim.scene/v3 bundle index; carries the
                                 # chunk list (uri / n_points / world-frame AABB)
    chunks/chunk_NNNNNN.spz      # Niantic SPZ chunks (spatially split), stored
                                 # directly in the alpasim ENU world frame;
                                 # per-Gaussian LiDAR attrs ride inside each SPZ
                                 # as an extension record (no sidecar files)
    <user-supplied extras>       # verbatim files / dirs at user-chosen paths

The bundle itself contains no Cesium 3D Tiles structures; the Cesium
``tileset.json`` format is accepted only as *input* to :func:`save_scene_usdz`.

Recognised "well-known" extras paths get auto-recorded in ``scene.json``'s
``extras`` block so downstream tooling can resolve them without scanning:

================================  =================================
archive path                      scene.json key
================================  =================================
``map.osm``                       ``extras.map_lanelet2``
``map.xodr``                      ``extras.map_opendrive``
``carla_world/manifest.json``     ``extras.carla_world``
``tracks.parquet``                ``extras.tracks``
``trajectory.parquet``            ``extras.trajectory``
``sequence_tracks.json``          ``extras.sequence_tracks``
``rig_trajectories.json``         ``extras.rig_trajectories``
``ppisp.json``                    ``extras.ppisp``
================================  =================================
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import math
import tempfile
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import spz

from ._geodesy import validate_anchor_against_lanelet2
from .ext_attributes import (
    EXT_GAUSSIAN_LIDAR_NAME,
    RAYDROP_SH_KEY,
    SPZ_EXT_TYPE_TIER4_GAUSSIAN_LIDAR_HEX,
    embed_lidar_extension,
    raydrop_sh_degree_from_coefs,
)
from .frame_convention import FRAME_CONVENTION, RUB_TO_ENU, validate_rigid_transform
from .gltf_io import load_gltf_with_metadata
from .ppisp import Ppisp, serialize_ppisp
from .rig_trajectories import RigTrajectory, serialize_rig_trajectories
from .spz_io import save_spz_world
from .tiles_export import _assign_cell_keys
from .tiles_io import _apply_rotation_to_quats
from .tracks import Track, serialize_tracks
from .usdz_metadata import (
    USDZ_METADATA_ARCHIVE_PATH,
    UsdzMetadata,
    encode_usdz_metadata,
    make_default_metadata,
)

__all__ = [
    "SceneUsdzOptions",
    "SceneUsdzResult",
    "save_scene_usdz",
]

_log = logging.getLogger(__name__)

_TOOL_NAME = "3dgs_io.scene_usdz"
_TOOL_VERSION = "0.1.0"
_SCENE_SCHEMA = "splatsim.scene/v3"

# Archive entries owned by the writer; user extras must not collide.
_RESERVED_PATHS = frozenset({"default.usda", "scene.json", USDZ_METADATA_ARCHIVE_PATH})
_RESERVED_PREFIXES: tuple[str, ...] = ("chunks/",)

_IDENTITY_16: tuple[float, ...] = (
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
)

_DEFAULT_USDA = """#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{
    custom asset sceneIndex = @./scene.json@
}
"""

# Recognised archive paths that get auto-recorded in scene.json's extras.
_KNOWN_EXTRAS: dict[str, str] = {
    "map.osm": "map_lanelet2",
    "map.xodr": "map_opendrive",
    "carla_world/manifest.json": "carla_world",
    "tracks.parquet": "tracks",
    "trajectory.parquet": "trajectory",
    "sequence_tracks.json": "sequence_tracks",
    "rig_trajectories.json": "rig_trajectories",
    "ppisp.json": "ppisp",
}


# ---------------------------------------------------------------------------
# Options + Result
# ---------------------------------------------------------------------------


@dataclass
class SceneUsdzOptions:
    """Tunables matching the public CLI flags."""

    chunk_size: float = 50.0
    max_points_per_chunk: int = 200_000
    min_scale: float = 0.05
    max_aspect_ratio: float = 5.0
    opacity_threshold: float = 0.0
    bbox_radius: float = math.inf

    exposure: float = 1.6
    near_plane: float = 0.5
    far_plane: float = 300.0

    validate_geo_anchor: bool = True
    """Cross-check ``ecef_anchor`` against an embedded Lanelet2 ``map.osm``
    (when one is bundled) and refuse to write a bundle whose anchor decodes
    tens of kilometres away from the map. Skipped when no usable map nodes
    are found."""


@dataclass
class SceneUsdzResult:
    """Summary of what was packed into the output USDZ."""

    out_path: Path
    n_gaussians: int = 0
    sh_degree: int = 0
    n_chunks: int = 0
    extras: dict[str, str | None] = field(default_factory=dict)
    ecef_anchor: list[list[float]] = field(default_factory=lambda: np.eye(4).tolist())
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Cesium input adapter: produces a cloud and its ECEF anchor.
# ---------------------------------------------------------------------------


def _walk_leaves(
    tile: Mapping[str, Any], parent_transform: np.ndarray
) -> Iterator[tuple[str, np.ndarray]]:
    """Walk a tile subtree and yield ``(content_uri, cumulative_transform)``
    for each leaf that holds tile content.

    The root tile's own Cesium transform is stripped by the caller. Only
    sub-root transforms are baked into payload values here.
    """
    local = tile.get("transform")
    if local is not None:
        # 3D Tiles transforms are column-major; reshape and transpose to row-major.
        # Cesium 3D Tiles: each tile's transform maps tile-local space to its
        # PARENT's space. For a leaf at depth k the cumulative world-bound
        # transform is M_root @ M_1 @ ... @ M_k, so walking top-down the new
        # cumulative is `parent_cumulative @ local` (parent on the LEFT).
        local_mat = np.array(local, dtype=np.float64).reshape(4, 4).T
        transform = parent_transform @ local_mat
    else:
        transform = parent_transform
    children = tile.get("children", [])
    is_leaf = not children
    if is_leaf:
        contents: list[Mapping[str, Any]] = list(tile.get("contents") or [])
        if "content" in tile and tile["content"] is not None:
            contents.append(tile["content"])
        for entry in contents:
            uri = entry.get("uri") or entry.get("url")
            if uri:
                yield uri, transform
    for child in children:
        yield from _walk_leaves(child, transform)


def _apply_transform_to_cloud(gc: spz.GaussianCloud, transform: np.ndarray) -> spz.GaussianCloud:
    """Apply a 4×4 row-major rigid transform to positions + quaternions."""
    n = gc.num_points
    positions = np.array(gc.positions, dtype=np.float32).reshape(n, 3)
    rotations = np.array(gc.rotations, dtype=np.float32).reshape(n, 4)
    r = transform[:3, :3].astype(np.float32)
    det = float(np.linalg.det(r))
    if not np.isfinite(det) or det <= 0.0:
        raise ValueError(
            "tile transform contains a reflection or singular rotation "
            f"(det={det:.6g}); positions and orientations cannot be baked consistently"
        )
    t = transform[:3, 3].astype(np.float32)
    new_positions = positions @ r.T + t
    new_rotations = _apply_rotation_to_quats(r, rotations)

    out = spz.GaussianCloud()
    out.antialiased = gc.antialiased
    out.positions = np.ascontiguousarray(new_positions, dtype=np.float32).reshape(-1)
    out.rotations = np.ascontiguousarray(new_rotations, dtype=np.float32).reshape(-1)
    out.scales = np.array(gc.scales, dtype=np.float32)
    out.colors = np.array(gc.colors, dtype=np.float32)
    out.alphas = np.array(gc.alphas, dtype=np.float32)
    out.sh_degree = gc.sh_degree
    if gc.sh_degree > 0:
        out.sh = np.array(gc.sh, dtype=np.float32)
    else:
        out.sh = np.zeros(0, dtype=np.float32)
    return out


def _concat_clouds(clouds: list[spz.GaussianCloud]) -> spz.GaussianCloud:
    sh_deg = clouds[0].sh_degree
    for c in clouds[1:]:
        if c.sh_degree != sh_deg:
            raise ValueError(f"Cannot merge tiles with mixed sh_degree: {sh_deg} vs {c.sh_degree}")
    out = spz.GaussianCloud()
    out.antialiased = clouds[0].antialiased
    out.sh_degree = sh_deg
    out.positions = np.concatenate([np.array(c.positions, dtype=np.float32) for c in clouds])
    out.rotations = np.concatenate([np.array(c.rotations, dtype=np.float32) for c in clouds])
    out.scales = np.concatenate([np.array(c.scales, dtype=np.float32) for c in clouds])
    out.colors = np.concatenate([np.array(c.colors, dtype=np.float32) for c in clouds])
    out.alphas = np.concatenate([np.array(c.alphas, dtype=np.float32) for c in clouds])
    out.sh = np.concatenate([np.array(c.sh, dtype=np.float32) for c in clouds])
    return out


def _raydrop_sh_degree(ext: dict[str, np.ndarray]) -> int:
    """SH degree implied by an ext dict's ``raydrop_sh`` array, or ``0`` if absent."""
    sh = ext.get(RAYDROP_SH_KEY)
    if sh is None:
        return 0
    return raydrop_sh_degree_from_coefs(int(np.asarray(sh).shape[1]))


def _concat_ext_attrs(
    ext_per_leaf: list[dict[str, np.ndarray]],
    counts: list[int],
) -> dict[str, np.ndarray]:
    """Concatenate per-leaf ext_attribute dicts in leaf-walk order.

    Leaves missing a key contribute zeros for the matching count, so the
    output arrays always align positionally with the concatenated cloud.
    Returns an empty dict if no leaf carries any ext attribute.
    """
    keys: set[str] = set()
    for ext in ext_per_leaf:
        keys.update(ext.keys())
    if not keys:
        return {}
    out: dict[str, np.ndarray] = {}
    for key in keys:
        # ``raydrop_sh`` is 2-D ``(count, coefs)``; scalar attrs are 1-D.
        # The trailing coefficient count is fixed across leaves, so infer it
        # from the first leaf that carries the key (used to zero-fill leaves
        # that lack it).
        trailing: tuple[int, ...] = ()
        for ext in ext_per_leaf:
            arr = ext.get(key)
            if arr is not None:
                trailing = np.asarray(arr).shape[1:]
                break
        parts: list[np.ndarray] = []
        for ext, count in zip(ext_per_leaf, counts, strict=True):
            arr = ext.get(key)
            if arr is None:
                parts.append(np.zeros((count, *trailing), dtype=np.float32))
            else:
                arr = np.asarray(arr, dtype=np.float32)
                if arr.shape[0] != count or arr.shape[1:] != trailing:
                    raise ValueError(
                        f"ext attribute {key!r} has shape {arr.shape}, "
                        f"expected ({count}, *{trailing})"
                    )
                parts.append(arr)
        # Every part is already float32, so the concatenation is too.
        out[key] = np.concatenate(parts, axis=0)
    return out


def _load_from_tileset(
    tileset_path: Path,
) -> tuple[spz.GaussianCloud, list[float], dict[str, np.ndarray]]:
    """Parse ``tileset.json`` and return ``(cloud, root_transform, ext_attrs)``.

    ``ext_attrs`` aggregates per-Gaussian ``EXT_gaussian_lidar`` arrays from
    every input GLB, threaded through concatenation in the same leaf-walk
    order as the gaussians (so ``attr[i] ↔ gaussian[i]`` after concat).
    Empty dict if no input GLB carries any ext attribute.
    """
    base = tileset_path.parent
    # ``utf-8-sig`` tolerates the optional BOM some editors prepend.
    doc = json.loads(tileset_path.read_text(encoding="utf-8-sig"))
    root = doc.get("root")
    if root is None:
        raise ValueError(f"{tileset_path}: missing 'root' tile")

    root_transform_list = root.get("transform")
    if root_transform_list is None:
        root_transform_list = list(_IDENTITY_16)
    else:
        if len(root_transform_list) != 16:
            raise ValueError(
                f"{tileset_path}: root.transform must have 16 elements, "
                f"got {len(root_transform_list)}"
            )
        root_transform_list = [float(v) for v in root_transform_list]

    # Walk leaves with the root's own transform stripped so positions accumulate
    # only the sub-root local transforms.
    root_without_transform: dict[str, Any] = {k: v for k, v in root.items() if k != "transform"}
    leaves = list(_walk_leaves(root_without_transform, np.eye(4, dtype=np.float64)))
    if not leaves:
        raise ValueError(f"{tileset_path}: no tile content found")

    clouds: list[spz.GaussianCloud] = []
    ext_per_leaf: list[dict[str, np.ndarray]] = []
    counts: list[int] = []
    for uri, transform in leaves:
        if "://" in uri:
            raise ValueError(f"Remote tile content not supported: {uri!r}")
        content_path = (base / uri).resolve()
        suffix = content_path.suffix.lower()
        if suffix not in (".glb", ".gltf"):
            raise ValueError(f"Only glTF tile content is supported; got {content_path}")
        gc, _, leaf_ext = load_gltf_with_metadata(content_path)
        if not np.allclose(transform, np.eye(4)):
            gc = _apply_transform_to_cloud(gc, transform)
        clouds.append(gc)
        ext_per_leaf.append(leaf_ext)
        counts.append(gc.num_points)

    cloud = clouds[0] if len(clouds) == 1 else _concat_clouds(clouds)
    ext_attrs = _concat_ext_attrs(ext_per_leaf, counts)
    return cloud, root_transform_list, ext_attrs


# ---------------------------------------------------------------------------
# Cloud filtering & spatial chunking
# ---------------------------------------------------------------------------


class _CloudArrays(NamedTuple):
    """Working numpy view of a (filtered, clamped) gaussian cloud."""

    positions: np.ndarray  # (n, 3) float32
    rotations: np.ndarray  # (n, 4) float32, unit norm
    scales: np.ndarray  # (n, 3) float32, log-space
    colors: np.ndarray  # (n, 3) float32 (SH DC)
    alphas: np.ndarray  # (n,)  float32, logit
    sh: np.ndarray | None  # (n, per_ch, 3) float32 or None
    sh_degree: int
    antialiased: bool
    # {name: (n,) float32} scalars — parallel to positions. The 2-D
    # ``raydrop_sh`` key, when present, is ``(n, coefs) float32``.
    ext_attrs: dict[str, np.ndarray]

    @property
    def n(self) -> int:
        return int(self.positions.shape[0])


# Saturating opacity logit: sigmoid(±40) is exactly 1.0 / 0.0 in float32 and
# the SPZ uint8 quantiser maps anything this large to 255 / 0 anyway.
_ALPHA_LOGIT_LIMIT = 40.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _filter_and_clamp(
    gc: spz.GaussianCloud,
    options: SceneUsdzOptions,
    ext_attrs: dict[str, np.ndarray] | None = None,
) -> _CloudArrays:
    """Drop non-finite / out-of-bbox / low-opacity gaussians and clamp scales.

    ``ext_attrs`` (if given) is sliced by the same ``keep`` mask so that
    ``attr[i] ↔ gaussian[i]`` continues to hold after filtering.
    """
    n = gc.num_points
    if n == 0:
        raise ValueError("Input GaussianCloud is empty")

    positions = np.array(gc.positions, dtype=np.float32).reshape(n, 3)
    rotations = np.array(gc.rotations, dtype=np.float32).reshape(n, 4)
    scales_log = np.array(gc.scales, dtype=np.float32).reshape(n, 3)
    alphas = np.array(gc.alphas, dtype=np.float32).reshape(n)
    # SPZ stores opacity sigmoid-quantised to uint8, so a fully opaque
    # gaussian (255) decodes to a +inf logit. That is a valid value — clamp
    # ±inf to a saturating finite logit instead of letting the finite mask
    # below drop the point (np.clip keeps NaN as NaN, which IS dropped).
    alphas = np.clip(alphas, -_ALPHA_LOGIT_LIMIT, _ALPHA_LOGIT_LIMIT)
    colors = np.array(gc.colors, dtype=np.float32).reshape(n, 3)
    sh_flat = np.array(gc.sh, dtype=np.float32)
    per_ch = sh_flat.size // (n * 3) if sh_flat.size > 0 else 0
    sh = sh_flat.reshape(n, per_ch, 3) if per_ch > 0 else None

    finite = (
        np.isfinite(positions).all(axis=1)
        & np.isfinite(rotations).all(axis=1)
        & np.isfinite(scales_log).all(axis=1)
        & np.isfinite(alphas)
    )
    keep = finite
    if math.isfinite(options.bbox_radius) and options.bbox_radius > 0:
        if finite.any():
            median = np.median(positions[finite], axis=0)
            dist = np.linalg.norm(positions - median, axis=1)
            keep = keep & (dist <= options.bbox_radius)
    if options.opacity_threshold > 0.0:
        opacity = _sigmoid(alphas)
        keep = keep & (opacity >= options.opacity_threshold)

    ext_in: dict[str, np.ndarray] = {}
    for name, arr in (ext_attrs or {}).items():
        # Scalars are 1-D ``(n,)``; ``raydrop_sh`` is 2-D ``(n, coefs)``. Only the
        # leading dimension is validated, so both flow through unchanged.
        arr = np.asarray(arr, dtype=np.float32)
        if arr.shape[0] != n:
            raise ValueError(f"ext attribute {name!r} has {arr.shape[0]} entries, expected {n}")
        ext_in[name] = arr

    if not keep.all():
        positions = positions[keep]
        rotations = rotations[keep]
        scales_log = scales_log[keep]
        alphas = alphas[keep]
        colors = colors[keep]
        if sh is not None:
            sh = sh[keep]
        ext_in = {k: v[keep] for k, v in ext_in.items()}
    if positions.shape[0] == 0:
        raise ValueError("All gaussians were filtered out — try relaxing options")

    # Re-normalise quaternions, replacing degenerate rows with identity.
    norm = np.linalg.norm(rotations, axis=1, keepdims=True)
    degenerate = (norm <= 1e-12).squeeze(-1)
    safe_norm = np.where(norm > 1e-12, norm, 1.0)
    rotations = (rotations / safe_norm).astype(np.float32)
    if degenerate.any():
        rotations[degenerate] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    # Scale clamp: floor each axis at min_scale, then cap by min_axis * max_aspect_ratio.
    scales_lin = np.exp(scales_log).astype(np.float32)
    scales_lin = np.maximum(scales_lin, float(options.min_scale))
    cap = scales_lin.min(axis=1, keepdims=True) * float(options.max_aspect_ratio)
    scales_lin = np.minimum(scales_lin, cap)
    scales_log = np.log(scales_lin).astype(np.float32)

    return _CloudArrays(
        positions=positions,
        rotations=rotations,
        scales=scales_log,
        colors=colors,
        alphas=alphas,
        sh=sh,
        sh_degree=int(gc.sh_degree) if sh is not None else 0,
        antialiased=bool(gc.antialiased),
        ext_attrs=ext_in,
    )


def _split_oversized_chunk(member_idx: np.ndarray, max_n: int) -> list[np.ndarray]:
    if max_n <= 0 or member_idx.size <= max_n:
        return [member_idx]
    n_splits = math.ceil(member_idx.size / max_n)
    return [a for a in np.array_split(member_idx, n_splits) if a.size > 0]


def _split_cloud_into_chunks(
    arrays: _CloudArrays, options: SceneUsdzOptions
) -> tuple[
    list[spz.GaussianCloud],
    list[tuple[np.ndarray, np.ndarray]],
    list[dict[str, np.ndarray]],
]:
    positions = arrays.positions
    bbox_min = positions.min(axis=0)
    bbox_max = positions.max(axis=0)
    if options.chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    cell_keys = _assign_cell_keys(positions, bbox_min, bbox_max, options.chunk_size)
    order = np.argsort(cell_keys, kind="stable")
    sorted_keys = cell_keys[order]
    splits = np.flatnonzero(np.diff(sorted_keys)) + 1
    groups = np.split(order, splits)

    has_sh = arrays.sh is not None and arrays.sh.shape[1] > 0
    chunks: list[spz.GaussianCloud] = []
    bounds: list[tuple[np.ndarray, np.ndarray]] = []
    chunk_ext: list[dict[str, np.ndarray]] = []
    for group in groups:
        for sub_idx in _split_oversized_chunk(group, options.max_points_per_chunk):
            c = spz.GaussianCloud()
            c.antialiased = arrays.antialiased
            c.positions = np.ascontiguousarray(positions[sub_idx]).reshape(-1)
            c.rotations = np.ascontiguousarray(arrays.rotations[sub_idx]).reshape(-1)
            c.scales = np.ascontiguousarray(arrays.scales[sub_idx]).reshape(-1)
            c.colors = np.ascontiguousarray(arrays.colors[sub_idx]).reshape(-1)
            c.alphas = np.ascontiguousarray(arrays.alphas[sub_idx]).reshape(-1)
            if has_sh:
                c.sh_degree = arrays.sh_degree
                c.sh = np.ascontiguousarray(arrays.sh[sub_idx]).reshape(-1)
            else:
                c.sh_degree = 0
                c.sh = np.zeros(0, dtype=np.float32)
            chunks.append(c)
            p = positions[sub_idx]
            bounds.append((p.min(axis=0), p.max(axis=0)))
            chunk_ext.append(
                {name: arr[sub_idx].astype(np.float32) for name, arr in arrays.ext_attrs.items()}
            )
    return chunks, bounds, chunk_ext


def _build_chunk_index(
    sub_clouds: list[spz.GaussianCloud],
    bounds: list[tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, Any]]:
    """Plain chunk index for ``scene.json``: URI, point count, world-frame AABB.

    All values are in the alpasim ENU world frame — there is no Cesium
    bounding-volume or geometric-error indirection.
    """
    entries: list[dict[str, Any]] = []
    for i, (sub, (bmin, bmax)) in enumerate(zip(sub_clouds, bounds, strict=True)):
        entries.append(
            {
                "uri": f"chunks/chunk_{i:06d}.spz",
                "n_points": int(sub.num_points),
                "bbox_min": [float(v) for v in bmin],
                "bbox_max": [float(v) for v in bmax],
            }
        )
    return entries


# ---------------------------------------------------------------------------
# Extras handling
# ---------------------------------------------------------------------------


def _normalise_arc_path(p: str) -> str:
    # Strip leading slashes (`/scene.json` → `scene.json`), unify separators,
    # and trim trailing slashes so reserved-path collision checks aren't
    # bypassed by spellings like ``scene.json/``.
    return p.lstrip("/").rstrip("/").replace("\\", "/")


def _collect_extras_entries(
    extras: Mapping[str, str | Path] | None,
) -> list[tuple[str, Path]]:
    """Expand directory sources into per-file (archive_path, src_path) entries."""
    if not extras:
        return []
    out: list[tuple[str, Path]] = []
    for raw_key, raw_src in extras.items():
        arc_key = _normalise_arc_path(raw_key)
        if not arc_key:
            raise ValueError("extras key must not be empty")
        if arc_key in _RESERVED_PATHS or any(
            arc_key == p.rstrip("/") or arc_key.startswith(p) for p in _RESERVED_PREFIXES
        ):
            raise ValueError(f"extras key {raw_key!r} collides with a reserved scene-bundle path")
        src = Path(raw_src)
        if not src.exists():
            raise FileNotFoundError(f"extras source not found: {src}")
        if src.is_dir():
            for sub in sorted(src.rglob("*")):
                if sub.is_file():
                    rel = sub.relative_to(src).as_posix()
                    out.append((f"{arc_key}/{rel}", sub))
        else:
            out.append((arc_key, src))
    return out


def _detect_known_extras(archive_paths: set[str]) -> dict[str, str | None]:
    detected: dict[str, str | None] = {key: None for key in _KNOWN_EXTRAS.values()}
    for path, scene_key in _KNOWN_EXTRAS.items():
        if path in archive_paths:
            detected[scene_key] = path
    return detected


# ---------------------------------------------------------------------------
# scene.json
# ---------------------------------------------------------------------------


def _compose_scene_json(
    *,
    arrays: _CloudArrays,
    options: SceneUsdzOptions,
    extras: dict[str, str | None],
    ecef_anchor: list[list[float]],
    producer_source: Mapping[str, Any],
    chunk_index: list[dict[str, Any]],
) -> dict[str, Any]:
    created_at = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    gaussians: dict[str, Any] = {
        "chunk_format": "spz/ngsp-v4",
        "chunks": chunk_index,
        "n_gaussians": arrays.n,
        "sh_degree": arrays.sh_degree,
        "filter": {
            "min_scale": float(options.min_scale),
            "max_aspect_ratio": float(options.max_aspect_ratio),
            "opacity_threshold": float(options.opacity_threshold),
            "bbox_radius": (
                None if math.isinf(options.bbox_radius) else float(options.bbox_radius)
            ),
        },
    }
    if arrays.ext_attrs:
        ext_attributes_block: dict[str, Any] = {
            "extension": EXT_GAUSSIAN_LIDAR_NAME,
            "container": "spz_extension",
            "spz_extension_type": SPZ_EXT_TYPE_TIER4_GAUSSIAN_LIDAR_HEX,
            "attributes": sorted(arrays.ext_attrs.keys()),
        }
        sh_degree = _raydrop_sh_degree(arrays.ext_attrs)
        if sh_degree > 0:
            ext_attributes_block["raydrop_sh_degree"] = sh_degree
        gaussians["ext_attributes"] = ext_attributes_block
    return {
        "schema": _SCENE_SCHEMA,
        "producer": {
            "tool": _TOOL_NAME,
            "tool_version": _TOOL_VERSION,
            "created_at": created_at,
            **producer_source,
        },
        "world": {
            "frame_convention": FRAME_CONVENTION,
            "ecef_anchor": ecef_anchor,
        },
        "gaussians": {**gaussians, "frame": "world"},
        "extras": extras,
        "render_defaults": {
            "exposure": float(options.exposure),
            "near_plane": float(options.near_plane),
            "far_plane": float(options.far_plane),
        },
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_scene_usdz(
    tileset_path: str | Path,
    out_path: str | Path,
    *,
    extras: Mapping[str, str | Path] | None = None,
    tracks: list[Track] | None = None,
    rig_trajectories: list[RigTrajectory] | None = None,
    ppisp: Ppisp | None = None,
    metadata: UsdzMetadata | None = None,
    options: SceneUsdzOptions | None = None,
) -> SceneUsdzResult:
    """Pack a Cesium ``tileset.json`` (+ extras + tracks + rigs) into a USDZ.

    The input Cesium anchor is converted once into the row-major
    ``scene.json.world.ecef_anchor``; per-tile transforms are baked into ENU
    payload values. The output bundle carries no Cesium structures — chunks
    are listed in ``scene.json.gaussians.chunks`` with world-frame AABBs.

    Parameters
    ----------
    tileset_path:
        Path to a Cesium 3D Tiles ``tileset.json``. Its ``root`` must have a
        ``content`` (or descend through ``children`` to leaves with content),
        each pointing at a glTF ``.glb`` / ``.gltf``. Remote URIs are not
        supported.
    out_path:
        Destination ``.usdz`` path.
    extras:
        Mapping of archive-relative path → file or directory on disk. Files
        are added verbatim; directories are recursively zipped under the key
        prefix. Reserved paths (``default.usda`` / ``scene.json`` /
        ``chunks/*``) are rejected with ``ValueError``.
    tracks:
        Optional list of dynamic-object :class:`Track` objects. When given
        they are serialised into ``sequence_tracks.json`` inside the archive
        (schema ``splatsim.sequence_tracks/v2``) and recorded under
        ``scene.json.extras.sequence_tracks``. Track poses live in the same
        declared world frame as the SPZ payload.
    rig_trajectories:
        Optional list of sensor-rig :class:`RigTrajectory` objects (typically
        an ego trajectory). When given they are serialised into
        ``rig_trajectories.json`` inside the archive using the frame-explicit
        ``splatsim.rig_trajectories/v2`` schema. Poses and sensor extrinsics
        are directly consumable as child-to-parent translation + xyzw rotation.
    ppisp:
        Optional :class:`Ppisp` describing per-camera / per-frame PPISP
        appearance-correction parameters (exposure / vignetting / colour /
        CRF). When given the parameters are serialised into ``ppisp.json``
        inside the archive (schema ``splatsim.ppisp/v1``) and recorded under
        ``scene.json.extras.ppisp``. Cameras are keyed by name (matching
        ``rig_trajectories.json`` cameras) and frames by ``timestamp_us``
        (matching rig poses).
    metadata:
        Identity card written to ``metadata.yaml`` at the archive root
        (``uuid`` / ``scene_id`` / ``version_string``). When ``None`` a
        default :class:`~3dgs_io.UsdzMetadata` is generated with a random
        UUID4, ``scene_id`` set to ``out_path.stem`` and ``version_string``
        set to ``"3dgs_io/<installed-package-version>"``.
    options:
        Filtering, scale clamping, chunk size, and render defaults.
    """
    tileset_path = Path(tileset_path)

    cloud, root_transform, source_ext_attrs = _load_from_tileset(tileset_path)
    source_root_matrix = np.array(root_transform, dtype=np.float64).reshape(4, 4).T
    source_to_world = np.eye(4, dtype=np.float64)
    source_to_world[:3, :3] = RUB_TO_ENU
    cloud = _apply_transform_to_cloud(cloud, source_to_world)
    root_matrix = source_root_matrix @ np.linalg.inv(source_to_world)
    validate_rigid_transform(root_matrix, where="tileset root.transform")

    return _save_bundle(
        cloud=cloud,
        ext_attrs=source_ext_attrs,
        ecef_anchor=root_matrix.tolist(),
        out_path=out_path,
        options=options,
        metadata=metadata,
        producer_source={"source_tileset": tileset_path.name},
        extras=extras,
        tracks=tracks,
        rig_trajectories=rig_trajectories,
        ppisp=ppisp,
    )


def _save_bundle(
    *,
    cloud: spz.GaussianCloud,
    ext_attrs: dict[str, np.ndarray],
    ecef_anchor: list[list[float]],
    out_path: str | Path,
    options: SceneUsdzOptions | None,
    metadata: UsdzMetadata | None,
    producer_source: Mapping[str, Any],
    extras: Mapping[str, str | Path] | None = None,
    extra_payloads: Sequence[tuple[str, bytes]] = (),
    tracks: list[Track] | None = None,
    rig_trajectories: list[RigTrajectory] | None = None,
    ppisp: Ppisp | None = None,
) -> SceneUsdzResult:
    """Core bundle writer shared by the tileset path and the scene-connect path.

    ``cloud`` / ``ext_attrs`` are already in the Z-up ENU world frame implied
    by ``ecef_anchor``. ``extra_payloads`` carries in-memory archive entries
    (``(archive_path, bytes)``) written verbatim — used by connect to carry
    extras straight from an input bundle without touching the filesystem.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if options is None:
        options = SceneUsdzOptions()
    if metadata is None:
        metadata = make_default_metadata(out_path=out_path)
    metadata_payload = encode_usdz_metadata(metadata)

    arrays = _filter_and_clamp(cloud, options, ext_attrs)
    sub_clouds, bounds, chunk_ext = _split_cloud_into_chunks(arrays, options)
    chunk_index = _build_chunk_index(sub_clouds, bounds)

    extras_entries = _collect_extras_entries(extras)
    payload_entries: list[tuple[str, bytes]] = []
    for raw_arc, payload in extra_payloads:
        arc = _normalise_arc_path(raw_arc)
        if arc in _RESERVED_PATHS or any(arc.startswith(p) for p in _RESERVED_PREFIXES):
            raise ValueError(f"extra payload {raw_arc!r} collides with a reserved path")
        payload_entries.append((arc, payload))
    archive_paths = {arc for arc, _ in extras_entries}
    dup = archive_paths & {arc for arc, _ in payload_entries}
    if dup:
        raise ValueError(f"extras and extra_payloads both provide {sorted(dup)}")
    archive_paths |= {arc for arc, _ in payload_entries}

    tracks_payload: bytes | None = None
    if tracks is not None:
        if "sequence_tracks.json" in archive_paths:
            raise ValueError(
                "tracks=... was passed but 'sequence_tracks.json' is also present in "
                "extras; pick one of the two"
            )
        tracks_payload = json.dumps(serialize_tracks(tracks), indent=2).encode("utf-8")
        archive_paths.add("sequence_tracks.json")

    rig_trajectories_payload: bytes | None = None
    if rig_trajectories:
        if "rig_trajectories.json" in archive_paths:
            raise ValueError(
                "rig_trajectories=... was passed but 'rig_trajectories.json' is also "
                "present in extras; pick one of the two"
            )
        rig_doc = serialize_rig_trajectories(rig_trajectories)
        rig_trajectories_payload = json.dumps(rig_doc, indent=2).encode("utf-8")
        archive_paths.add("rig_trajectories.json")

    ppisp_payload: bytes | None = None
    if ppisp is not None:
        if "ppisp.json" in archive_paths:
            raise ValueError(
                "ppisp=... was passed but 'ppisp.json' is also present in "
                "extras; pick one of the two"
            )
        ppisp_payload = json.dumps(serialize_ppisp(ppisp), indent=2).encode("utf-8")
        archive_paths.add("ppisp.json")

    extras_meta = _detect_known_extras(archive_paths)

    if options.validate_geo_anchor:
        map_osm = _find_map_osm(extras_entries, payload_entries)
        if map_osm is not None:
            validate_anchor_against_lanelet2(np.asarray(ecef_anchor, dtype=np.float64), map_osm)

    scene_doc = _compose_scene_json(
        arrays=arrays,
        options=options,
        extras=extras_meta,
        ecef_anchor=ecef_anchor,
        producer_source=producer_source,
        chunk_index=chunk_index,
    )

    # Materialise chunks to a tempdir, then assemble the USDZ from disk so
    # large extras (multi-GB CARLA trees, etc.) never get loaded into RAM.
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        chunk_paths: list[Path] = []
        for i, sub in enumerate(sub_clouds):
            cp = td_path / f"chunk_{i:06d}.spz"
            save_spz_world(sub, cp)
            chunk_paths.append(cp)

        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
            _zip_write_str(zf, "default.usda", _DEFAULT_USDA)
            _zip_write_bytes(zf, USDZ_METADATA_ARCHIVE_PATH, metadata_payload)
            _zip_write_str(zf, "scene.json", json.dumps(scene_doc, indent=2))
            if tracks_payload is not None:
                _zip_write_bytes(zf, "sequence_tracks.json", tracks_payload)
            if rig_trajectories_payload is not None:
                _zip_write_bytes(zf, "rig_trajectories.json", rig_trajectories_payload)
            if ppisp_payload is not None:
                _zip_write_bytes(zf, "ppisp.json", ppisp_payload)
            for entry, src, ext in zip(chunk_index, chunk_paths, chunk_ext, strict=True):
                if ext:
                    # Splice the LiDAR extension in memory on the way into the
                    # archive — one read of the chunk, no rewrite on disk.
                    _zip_write_bytes(
                        zf,
                        entry["uri"],
                        embed_lidar_extension(src.read_bytes(), ext, count=entry["n_points"]),
                    )
                else:
                    zf.write(src, entry["uri"], compress_type=zipfile.ZIP_STORED)
            for arc, src in extras_entries:
                zf.write(src, arc, compress_type=zipfile.ZIP_STORED)
            for arc, payload in payload_entries:
                _zip_write_bytes(zf, arc, payload)

    return SceneUsdzResult(
        out_path=out_path,
        n_gaussians=arrays.n,
        sh_degree=arrays.sh_degree,
        n_chunks=len(sub_clouds),
        extras=extras_meta,
        ecef_anchor=ecef_anchor,
        metadata=metadata.to_dict(),
    )


def _find_map_osm(
    extras_entries: list[tuple[str, Path]],
    payload_entries: list[tuple[str, bytes]],
) -> bytes | None:
    """Return the bytes of the ``map.osm`` entry about to be bundled, if any."""
    for arc, payload in payload_entries:
        if arc == "map.osm":
            return payload
    for arc, src in extras_entries:
        if arc == "map.osm":
            return src.read_bytes()
    return None


def _zip_write_str(zf: zipfile.ZipFile, name: str, content: str) -> None:
    zi = zipfile.ZipInfo(name)
    zi.compress_type = zipfile.ZIP_STORED
    zf.writestr(zi, content.encode("utf-8"))


def _zip_write_bytes(zf: zipfile.ZipFile, name: str, content: bytes) -> None:
    zi = zipfile.ZipInfo(name)
    zi.compress_type = zipfile.ZIP_STORED
    zf.writestr(zi, content)


def _result_summary(result: SceneUsdzResult) -> dict[str, Any]:
    """Stringified summary used by the CLI."""
    d = asdict(result)
    d["out_path"] = str(result.out_path)
    return d
