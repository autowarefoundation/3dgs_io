"""Splatsim scene-bundle writer (USDZ → directory bundle).

The bundle this module produces is the **3D-GS portion** of a splatsim scene:

* ``scene.json``             — bundle index + producer info
* ``tileset.json``           — Cesium 3D Tiles v1.0 + ``EXT_3dgs_spz`` extension
* ``chunks/chunk_NNNNNN.spz`` — Niantic SPZ tiles produced by spatially
  splitting the input cloud

All other sidecars referenced from the splatsim spec (``map.osm`` /
``map.xodr`` / ``carla_world/`` / ``tracks.parquet`` / ``trajectory.parquet``)
are assumed to be produced by external tooling. If they happen to exist next
to the bundle at write time, :func:`save_scene_bundle` surfaces them through
the ``scene.json`` ``extras`` block (pass-through); otherwise those fields
are ``null``.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import spz

from .spz_io import save_spz
from .tiles_export import _assign_cell_keys
from .usdz_io import load_usdz

__all__ = [
    "SceneBundleOptions",
    "SceneBundleResult",
    "save_scene_bundle",
]

_log = logging.getLogger(__name__)

_TOOL_NAME = "splatsim-import-usdz"
_TOOL_VERSION = "0.1.0"
_SCENE_SCHEMA = "splatsim.scene/v1"

# spz / 3D Tiles tile content extension key.
_EXT_3DGS_SPZ = "EXT_3dgs_spz"


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


@dataclass
class SceneBundleOptions:
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

    geometric_error: float = 100.0
    """Root ``geometricError`` written into ``tileset.json``."""


@dataclass
class SceneBundleResult:
    """Summary of files produced by :func:`save_scene_bundle`."""

    scene_json: Path
    tileset_json: Path
    chunks: list[Path] = field(default_factory=list)
    n_gaussians: int = 0
    sh_degree: int = 0


# ---------------------------------------------------------------------------
# spz.GaussianCloud filtering — returns the working numpy arrays directly so
# the chunk writer can reuse them without a second extraction pass.
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

    @property
    def n(self) -> int:
        return int(self.positions.shape[0])


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _filter_and_clamp(gc: spz.GaussianCloud, options: SceneBundleOptions) -> _CloudArrays:
    """Drop non-finite / out-of-bbox / low-opacity gaussians and clamp scales.

    Returns the working numpy arrays directly; the caller (chunk writer) reuses
    them without re-extracting from a temporary :class:`spz.GaussianCloud`.
    """
    n = gc.num_points
    if n == 0:
        raise ValueError("Input GaussianCloud is empty")

    positions = np.array(gc.positions, dtype=np.float32).reshape(n, 3)
    rotations = np.array(gc.rotations, dtype=np.float32).reshape(n, 4)
    scales_log = np.array(gc.scales, dtype=np.float32).reshape(n, 3)
    alphas = np.array(gc.alphas, dtype=np.float32).reshape(n)
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

    if not keep.all():
        positions = positions[keep]
        rotations = rotations[keep]
        scales_log = scales_log[keep]
        alphas = alphas[keep]
        colors = colors[keep]
        if sh is not None:
            sh = sh[keep]
    if positions.shape[0] == 0:
        raise ValueError("All gaussians were filtered out — try relaxing options")

    # Re-normalise quaternions, replacing degenerate (zero-norm) rows with identity.
    norm = np.linalg.norm(rotations, axis=1, keepdims=True)
    degenerate = (norm <= 1e-12).squeeze(-1)
    safe_norm = np.where(norm > 1e-12, norm, 1.0)
    rotations = (rotations / safe_norm).astype(np.float32)
    if degenerate.any():
        # xyzw identity quaternion
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
    )


# ---------------------------------------------------------------------------
# Spatial chunking + per-tile spz writer + tileset.json
# ---------------------------------------------------------------------------


def _split_oversized_chunk(member_idx: np.ndarray, max_n: int) -> list[np.ndarray]:
    """Split a single cell into <= ``max_n`` sub-chunks by simple slicing."""
    if max_n <= 0 or member_idx.size <= max_n:
        return [member_idx]
    n_splits = math.ceil(member_idx.size / max_n)
    return [a for a in np.array_split(member_idx, n_splits) if a.size > 0]


def _aabb_to_3dtiles_box(bbox_min: np.ndarray, bbox_max: np.ndarray) -> list[float]:
    center = ((bbox_min + bbox_max) / 2).astype(np.float64)
    half = ((bbox_max - bbox_min) / 2).astype(np.float64)
    # Guard against zero-extent boxes (single-point chunks).
    half = np.where(half > 0, half, 1e-6)
    return [
        float(center[0]),
        float(center[1]),
        float(center[2]),
        float(half[0]),
        0.0,
        0.0,
        0.0,
        float(half[1]),
        0.0,
        0.0,
        0.0,
        float(half[2]),
    ]


def _split_cloud_into_chunks(
    arrays: _CloudArrays, options: SceneBundleOptions
) -> tuple[list[spz.GaussianCloud], list[tuple[np.ndarray, np.ndarray]]]:
    """Return per-cell sub-clouds + per-cell (bbox_min, bbox_max)."""
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
    return chunks, bounds


def _write_chunks_and_tileset(
    arrays: _CloudArrays,
    out_dir: Path,
    options: SceneBundleOptions,
) -> tuple[Path, list[Path]]:
    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    sub_clouds, bounds = _split_cloud_into_chunks(arrays, options)
    chunk_paths: list[Path] = []
    children: list[dict[str, Any]] = []
    bbox_min_all = bounds[0][0].copy()
    bbox_max_all = bounds[0][1].copy()

    for i, (sub, (bmin, bmax)) in enumerate(zip(sub_clouds, bounds, strict=True)):
        name = f"chunk_{i:06d}.spz"
        path = chunks_dir / name
        save_spz(sub, path)
        chunk_paths.append(path)
        bbox_min_all = np.minimum(bbox_min_all, bmin)
        bbox_max_all = np.maximum(bbox_max_all, bmax)
        children.append(
            {
                "boundingVolume": {"box": _aabb_to_3dtiles_box(bmin, bmax)},
                "geometricError": 0.0,
                "content": {
                    "uri": f"chunks/{name}",
                    "extensions": {
                        _EXT_3DGS_SPZ: {"format": "spz/1", "n_points": int(sub.num_points)},
                    },
                },
            }
        )

    root: dict[str, Any] = {
        "boundingVolume": {"box": _aabb_to_3dtiles_box(bbox_min_all, bbox_max_all)},
        "geometricError": float(options.geometric_error),
        "refine": "ADD",
        # Row-major identity (Cesium 3D Tiles transforms are column-major, but
        # identity is symmetric so this is unambiguous).
        "transform": [
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
        ],
        "children": children,
    }

    tileset = {
        "asset": {"version": "1.0", "tilesetVersion": "splatsim-spz/1.0"},
        "extensionsRequired": [_EXT_3DGS_SPZ],
        "extensionsUsed": [_EXT_3DGS_SPZ],
        "geometricError": float(options.geometric_error),
        "root": root,
    }
    tileset_path = out_dir / "tileset.json"
    tileset_path.write_text(json.dumps(tileset, indent=2))
    return tileset_path, chunk_paths


# ---------------------------------------------------------------------------
# scene.json
# ---------------------------------------------------------------------------


# Sidecars that may exist next to the bundle directory (produced by external
# tooling). We never create or convert these; we only record their presence in
# ``scene.json`` ``extras`` when they happen to be in place.
_EXTRA_PATHS: dict[str, str] = {
    "map_lanelet2": "map.osm",
    "map_opendrive": "map.xodr",
    "carla_world": "carla_world/manifest.json",
    "tracks": "tracks.parquet",
    "trajectory": "trajectory.parquet",
}


def _detect_existing_extras(out_dir: Path) -> dict[str, str | None]:
    """Return an ``extras`` dict pointing at any pre-existing sidecars in ``out_dir``."""
    return {key: rel if (out_dir / rel).exists() else None for key, rel in _EXTRA_PATHS.items()}


def _compose_scene_json(
    *,
    arrays: _CloudArrays,
    options: SceneBundleOptions,
    extras: dict[str, str | None],
) -> dict[str, Any]:
    created_at = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "schema": _SCENE_SCHEMA,
        "producer": {
            "tool": _TOOL_NAME,
            "tool_version": _TOOL_VERSION,
            "created_at": created_at,
        },
        "world": {"up_axis": "z", "units": "meters"},
        "gaussians": {
            "tileset": "tileset.json",
            "tile_content_format": "spz/1",
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
        },
        "extras": extras,
        "render_defaults": {
            "exposure": float(options.exposure),
            "near_plane": float(options.near_plane),
            "far_plane": float(options.far_plane),
        },
    }


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def save_scene_bundle(
    usdz_path: str | Path,
    out_dir: str | Path,
    options: SceneBundleOptions | None = None,
) -> SceneBundleResult:
    """Convert a USDZ-wrapped :class:`spz.GaussianCloud` into a scene bundle.

    Produces ``scene.json`` / ``tileset.json`` / ``chunks/*.spz``. Non-gaussian
    sidecars (``map.osm`` / ``carla_world/`` / ``tracks.parquet`` / etc.) are
    expected to be provided by upstream tooling — if any already exist in
    ``out_dir`` they are recorded in ``scene.json``'s ``extras`` block.
    """
    usdz_path = Path(usdz_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if options is None:
        options = SceneBundleOptions()

    arrays = _filter_and_clamp(load_usdz(usdz_path), options)
    tileset_path, chunk_paths = _write_chunks_and_tileset(arrays, out_dir, options)

    extras = _detect_existing_extras(out_dir)
    scene = _compose_scene_json(arrays=arrays, options=options, extras=extras)
    scene_path = out_dir / "scene.json"
    scene_path.write_text(json.dumps(scene, indent=2))

    return SceneBundleResult(
        scene_json=scene_path,
        tileset_json=tileset_path,
        chunks=chunk_paths,
        n_gaussians=arrays.n,
        sh_degree=arrays.sh_degree,
    )


# Used by the CLI to log a structured summary.
def _result_summary(result: SceneBundleResult) -> dict[str, Any]:
    d = asdict(result)
    d["chunks"] = [str(p) for p in result.chunks]
    for k in ("scene_json", "tileset_json"):
        v = d.get(k)
        if v is not None:
            d[k] = str(v)
    return d
