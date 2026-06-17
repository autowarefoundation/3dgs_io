"""Splatsim scene-bundle writer (alpasim USDZ → directory bundle).

This module is responsible only for the **3D-GS-related** outputs of the
splatsim scene bundle:

* ``scene.json``             — bundle index + producer info
* ``tileset.json``           — Cesium 3D Tiles v1.0 + ``EXT_3dgs_spz`` extension
* ``chunks/chunk_NNNN.spz``  — Niantic spz tiles produced from the alpasim
  background gaussians
* ``sky/manifest.json`` + ``sky/{px,nx,py,ny,pz,nz}.png`` — cubemap faces
  (sRGB 8-bit) reconstructed from ``volume.nurec``

All other sidecars referenced from the spec (``map.osm`` / ``map.xodr`` /
``carla_world/`` / ``tracks.parquet`` / ``trajectory.parquet``) are assumed
to be produced by external tooling. If they happen to already exist next to
the bundle at write time, :func:`save_scene_bundle` will surface them through
the ``scene.json`` ``extras`` block (pass-through); otherwise those fields
are left as ``null``.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import spz

from .spz_io import save_spz
from .tiles_export import _assign_cell_keys
from .usdz_io import AlpasimGaussianCloud, AlpasimSkyCubemap, load_usdz

__all__ = [
    "SceneBundleOptions",
    "SceneBundleResult",
    "alpasim_to_spz",
    "save_scene_bundle",
]

_log = logging.getLogger(__name__)

_TOOL_NAME = "splatsim-import-usdz"
_TOOL_VERSION = "0.1.0"
_SCENE_SCHEMA = "splatsim.scene/v1"
_SKY_SCHEMA = "splatsim.sky_cubemap/v1"

# spz / 3D Tiles tile content extension key.
_EXT_3DGS_SPZ = "EXT_3dgs_spz"

# Cubemap face order required by scene.json sky manifest.
_CUBE_FACES = ("px", "nx", "py", "ny", "pz", "nz")


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


@dataclass
class SceneBundleOptions:
    """Tunables matching the public CLI flags (see ``splatsim-import-usdz``)."""

    chunk_size: float = 50.0
    max_points_per_chunk: int = 200_000
    min_scale: float = 0.05
    max_aspect_ratio: float = 5.0
    opacity_threshold: float = 0.0
    bbox_radius: float = math.inf

    sky_format: str = "png"  # "png" (sRGB 8) / "exr" (linear fp16)

    sky_intensity: float = 0.4
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
    sky_dir: Path | None = None
    n_gaussians: int = 0
    sh_degree: int = 0


# ---------------------------------------------------------------------------
# NuRec → spz.GaussianCloud
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def alpasim_to_spz(
    cloud: AlpasimGaussianCloud,
    options: SceneBundleOptions | None = None,
) -> spz.GaussianCloud:
    """Convert :class:`AlpasimGaussianCloud` to :class:`spz.GaussianCloud`.

    Time-varying albedo Fourier coefficients (``features_albedo[:, 1:, :]``),
    extra signals and training statistics are dropped. Scales are clamped per
    the streak-avoidance heuristic described in the format spec (table ⑦).
    """
    if options is None:
        options = SceneBundleOptions()

    positions = np.asarray(cloud.positions, dtype=np.float32)
    rotations = np.asarray(cloud.rotations, dtype=np.float32)
    scales_log = np.asarray(cloud.scales, dtype=np.float32)
    densities = np.asarray(cloud.densities, dtype=np.float32).reshape(-1)
    albedo = np.asarray(cloud.features_albedo, dtype=np.float32)  # (N, F, 3)
    specular = np.asarray(cloud.features_specular, dtype=np.float32)  # (N, K)

    n = positions.shape[0]
    if n == 0:
        raise ValueError("AlpasimGaussianCloud is empty")
    if albedo.ndim == 3 and albedo.shape[1] >= 1:
        sh_dc = albedo[:, 0, :]
    elif albedo.ndim == 2 and albedo.shape[1] == 3:
        sh_dc = albedo
    else:
        raise ValueError(
            f"features_albedo must be (N, F>=1, 3) or (N, 3); got shape {albedo.shape}"
        )

    # ---- 1. drop non-finite & out-of-bbox -----------------------------
    finite = (
        np.isfinite(positions).all(axis=1)
        & np.isfinite(rotations).all(axis=1)
        & np.isfinite(scales_log).all(axis=1)
        & np.isfinite(densities)
    )
    keep = finite
    if math.isfinite(options.bbox_radius) and options.bbox_radius > 0:
        median = np.median(positions[finite], axis=0)
        dist = np.linalg.norm(positions - median, axis=1)
        keep = keep & (dist <= options.bbox_radius)
    if options.opacity_threshold > 0.0:
        opacity = _sigmoid(densities)
        keep = keep & (opacity >= options.opacity_threshold)
    if not keep.all():
        positions = positions[keep]
        rotations = rotations[keep]
        scales_log = scales_log[keep]
        densities = densities[keep]
        sh_dc = sh_dc[keep]
        specular = specular[keep]
    n = positions.shape[0]
    if n == 0:
        raise ValueError("All gaussians were filtered out — try relaxing options")

    # ---- 2. quaternion normalisation (xyzw stays xyzw) ----------------
    norm = np.linalg.norm(rotations, axis=1, keepdims=True)
    norm = np.where(norm > 1e-12, norm, 1.0)
    rotations = (rotations / norm).astype(np.float32)

    # ---- 3. scale clamp (streak avoidance) ----------------------------
    scales_lin = np.exp(scales_log, dtype=np.float32)
    scales_lin = np.maximum(scales_lin, float(options.min_scale))
    min_axis = scales_lin.min(axis=1, keepdims=True)
    cap = min_axis * float(options.max_aspect_ratio)
    scales_lin = np.minimum(scales_lin, cap)
    scales_log = np.log(scales_lin).astype(np.float32)

    # ---- 4. assemble spz cloud ---------------------------------------
    # SH layout: (N, per_ch, 3) -> flat (N*per_ch*3,).  NuRec features_specular
    # already follows the coeff-major / RGB-inner convention used by spz.
    per_ch = specular.shape[1] // 3
    sh_deg = int(round(math.sqrt(per_ch + 1))) - 1  # (d+1)^2 - 1 == per_ch (incl. DC)

    gc = spz.GaussianCloud()
    gc.antialiased = False
    gc.positions = np.ascontiguousarray(positions, dtype=np.float32).reshape(-1)
    gc.rotations = np.ascontiguousarray(rotations, dtype=np.float32).reshape(-1)
    gc.scales = np.ascontiguousarray(scales_log, dtype=np.float32).reshape(-1)
    gc.colors = np.ascontiguousarray(sh_dc, dtype=np.float32).reshape(-1)
    gc.alphas = np.ascontiguousarray(densities, dtype=np.float32).reshape(-1)
    if per_ch > 0 and sh_deg >= 1:
        gc.sh_degree = sh_deg
        gc.sh = np.ascontiguousarray(specular, dtype=np.float32).reshape(-1)
    else:
        gc.sh_degree = 0
        gc.sh = np.zeros(0, dtype=np.float32)
    return gc


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
    gc: spz.GaussianCloud, options: SceneBundleOptions
) -> tuple[list[spz.GaussianCloud], list[tuple[np.ndarray, np.ndarray]]]:
    """Return per-cell sub-clouds + per-cell (bbox_min, bbox_max)."""
    n = gc.num_points
    positions = np.array(gc.positions, dtype=np.float32).reshape(n, 3)
    rotations = np.array(gc.rotations, dtype=np.float32).reshape(n, 4)
    scales = np.array(gc.scales, dtype=np.float32).reshape(n, 3)
    colors = np.array(gc.colors, dtype=np.float32).reshape(n, 3)
    alphas = np.array(gc.alphas, dtype=np.float32).reshape(n)
    sh_flat = np.array(gc.sh, dtype=np.float32)
    has_sh = sh_flat.size > 0
    per_ch = sh_flat.size // (n * 3) if has_sh else 0
    sh = sh_flat.reshape(n, per_ch, 3) if has_sh else None

    bbox_min = positions.min(axis=0)
    bbox_max = positions.max(axis=0)
    if options.chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    cell_keys = _assign_cell_keys(positions, bbox_min, bbox_max, options.chunk_size)
    order = np.argsort(cell_keys, kind="stable")
    sorted_keys = cell_keys[order]
    splits = np.flatnonzero(np.diff(sorted_keys)) + 1
    groups = np.split(order, splits)

    chunks: list[spz.GaussianCloud] = []
    bounds: list[tuple[np.ndarray, np.ndarray]] = []
    for group in groups:
        for sub_idx in _split_oversized_chunk(group, options.max_points_per_chunk):
            c = spz.GaussianCloud()
            c.antialiased = gc.antialiased
            c.positions = np.ascontiguousarray(positions[sub_idx]).reshape(-1)
            c.rotations = np.ascontiguousarray(rotations[sub_idx]).reshape(-1)
            c.scales = np.ascontiguousarray(scales[sub_idx]).reshape(-1)
            c.colors = np.ascontiguousarray(colors[sub_idx]).reshape(-1)
            c.alphas = np.ascontiguousarray(alphas[sub_idx]).reshape(-1)
            if sh is not None and sh.shape[1] > 0:
                c.sh_degree = gc.sh_degree
                c.sh = np.ascontiguousarray(sh[sub_idx]).reshape(-1)
            else:
                c.sh_degree = 0
                c.sh = np.zeros(0, dtype=np.float32)
            chunks.append(c)
            p = positions[sub_idx]
            bounds.append((p.min(axis=0), p.max(axis=0)))
    return chunks, bounds


def _write_chunks_and_tileset(
    gc: spz.GaussianCloud,
    out_dir: Path,
    options: SceneBundleOptions,
) -> tuple[Path, list[Path]]:
    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    sub_clouds, bounds = _split_cloud_into_chunks(gc, options)
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
# Sky cubemap
# ---------------------------------------------------------------------------


def _save_sky_cubemap(sky: AlpasimSkyCubemap, out_dir: Path, sky_format: str) -> tuple[Path, Path]:
    sky_dir = out_dir / "sky"
    sky_dir.mkdir(parents=True, exist_ok=True)

    # textures: (1, 6, H, W, 3) fp16, RGB in linear-ish space.
    tex = np.asarray(sky.textures, dtype=np.float32)
    if tex.ndim != 5 or tex.shape[0] != 1 or tex.shape[1] != 6 or tex.shape[-1] != 3:
        raise ValueError(f"Unexpected sky.textures shape: {tex.shape}")
    faces = tex[0]
    height, width = int(faces.shape[1]), int(faces.shape[2])

    fmt = sky_format.lower()
    if fmt == "png":
        from PIL import Image  # noqa: PLC0415

        for i, face_name in enumerate(_CUBE_FACES):
            arr = np.clip(faces[i], 0.0, 1.0)
            arr = (arr * 255.0 + 0.5).astype(np.uint8)
            Image.fromarray(arr, mode="RGB").save(sky_dir / f"{face_name}.png")
        encoding = "sRGB_8"
    elif fmt == "exr":
        try:
            import Imath  # noqa: F401, PLC0415
            import OpenEXR  # noqa: F401, PLC0415
        except ImportError as e:
            raise ImportError(
                "sky_format='exr' requires the `OpenEXR` and `Imath` Python bindings."
            ) from e
        for i, face_name in enumerate(_CUBE_FACES):
            arr = faces[i].astype(np.float16)
            _write_exr(sky_dir / f"{face_name}.exr", arr)
        encoding = "linear_fp16"
    else:
        raise ValueError(f"Unsupported sky_format: {sky_format!r}")

    manifest = {
        "schema": _SKY_SCHEMA,
        "face_order": list(_CUBE_FACES),
        "frame": "y_up",
        "resolution": [width, height],
        "encoding": encoding,
        "n_grad_updates": int(sky.n_grad_updates),
    }
    manifest_path = sky_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path, sky_dir


def _write_exr(path: Path, arr: np.ndarray) -> None:  # pragma: no cover - optional
    import Imath  # noqa: PLC0415
    import OpenEXR  # noqa: PLC0415

    height, width, _ = arr.shape
    header = OpenEXR.Header(width, height)
    half = Imath.PixelType(Imath.PixelType.HALF)
    header["channels"] = {
        "R": Imath.Channel(half),
        "G": Imath.Channel(half),
        "B": Imath.Channel(half),
    }
    out = OpenEXR.OutputFile(str(path), header)
    r = arr[..., 0].tobytes()
    g = arr[..., 1].tobytes()
    b = arr[..., 2].tobytes()
    out.writePixels({"R": r, "G": g, "B": b})
    out.close()


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
    """Return ``extras`` dict pointing at any pre-existing sidecars in ``out_dir``.

    Missing files map to ``None`` so the scene.json schema stays stable.
    """
    return {key: rel if (out_dir / rel).exists() else None for key, rel in _EXTRA_PATHS.items()}


def _compose_scene_json(
    *,
    cloud: AlpasimGaussianCloud,
    gc: spz.GaussianCloud,
    options: SceneBundleOptions,
    has_sky: bool,
    ground_z: float | None,
    extras: dict[str, str | None],
) -> dict[str, Any]:
    created_at = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "schema": _SCENE_SCHEMA,
        "producer": {
            "tool": _TOOL_NAME,
            "tool_version": _TOOL_VERSION,
            "created_at": created_at,
            "source_alpasim_version": cloud.version,
        },
        "world": {
            "up_axis": "z",
            "units": "meters",
            "nre_offset": [float(v) for v in cloud.nre_offset],
            "ground_z": float(ground_z) if ground_z is not None else 0.0,
        },
        "gaussians": {
            "tileset": "tileset.json",
            "tile_content_format": "spz/1",
            "n_gaussians": int(gc.num_points),
            "sh_degree": int(gc.sh_degree),
            "filter": {
                "min_scale": float(options.min_scale),
                "max_aspect_ratio": float(options.max_aspect_ratio),
                "opacity_threshold": float(options.opacity_threshold),
                "bbox_radius": (
                    None if math.isinf(options.bbox_radius) else float(options.bbox_radius)
                ),
            },
        },
        "sky": (
            {
                "manifest": "sky/manifest.json",
                "default_intensity": float(options.sky_intensity),
            }
            if has_sky
            else None
        ),
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


def _estimate_ground_z(positions_flat: np.ndarray) -> float | None:
    if positions_flat.size == 0:
        return None
    z = positions_flat.reshape(-1, 3)[:, 2]
    z = z[np.isfinite(z)]
    if z.size == 0:
        return None
    return float(np.percentile(z, 1.0))


def save_scene_bundle(
    usdz_path: str | Path,
    out_dir: str | Path,
    options: SceneBundleOptions | None = None,
) -> SceneBundleResult:
    """Convert an alpasim USDZ into the 3D-GS portion of a splatsim scene bundle.

    Produces ``scene.json`` / ``tileset.json`` / ``chunks/*.spz`` and, when the
    source USDZ carries a sky cubemap, the ``sky/`` directory. Non-gaussian
    sidecars (``map.osm`` / ``carla_world/`` / ``tracks.parquet`` / etc.) are
    expected to be provided by upstream tooling — if they already exist in
    ``out_dir`` they are recorded in ``scene.json``'s ``extras`` block.
    """
    usdz_path = Path(usdz_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if options is None:
        options = SceneBundleOptions()

    cloud = load_usdz(usdz_path)
    gc = alpasim_to_spz(cloud, options)

    tileset_path, chunk_paths = _write_chunks_and_tileset(gc, out_dir, options)

    sky_dir: Path | None = None
    has_sky = False
    if cloud.sky is not None and tuple(cloud.sky.textures.shape[2:4]) != (1, 1):
        _, sky_dir = _save_sky_cubemap(cloud.sky, out_dir, options.sky_format)
        has_sky = True

    extras = _detect_existing_extras(out_dir)

    ground_z = _estimate_ground_z(np.array(gc.positions, dtype=np.float32))
    scene = _compose_scene_json(
        cloud=cloud,
        gc=gc,
        options=options,
        has_sky=has_sky,
        ground_z=ground_z,
        extras=extras,
    )
    scene_path = out_dir / "scene.json"
    scene_path.write_text(json.dumps(scene, indent=2))

    return SceneBundleResult(
        scene_json=scene_path,
        tileset_json=tileset_path,
        chunks=chunk_paths,
        sky_dir=sky_dir,
        n_gaussians=int(gc.num_points),
        sh_degree=int(gc.sh_degree),
    )


# Used by the CLI to log a structured summary.
def _result_summary(result: SceneBundleResult) -> dict[str, Any]:
    d = asdict(result)
    d["chunks"] = [str(p) for p in result.chunks]
    for k in ("scene_json", "tileset_json", "sky_dir"):
        v = d.get(k)
        if v is not None:
            d[k] = str(v)
    return d
