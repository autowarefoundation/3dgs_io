"""3D Tiles (OGC) writer.

:func:`save_tileset` accepts:

* an in-memory :class:`~spz.GaussianCloud` — spatially chunked,
* a path / URL to an existing ``tileset.json`` — streamed and re-chunked,
* a ``list[Tile3DContent]`` — written as-is (one GLB per tile) with each
  tile's :attr:`~Tile3DContent.bounding_volume` preserved.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import spz

from .ext_attributes import EXT_GAUSSIAN_LIDAR_NAME
from .gltf_io import GltfSaveOptions, load_gltf_with_metadata, save_gltf
from .tiles_io import (
    BoundingVolume,
    BoundingVolumeBox,
    BoundingVolumeRegion,
    BoundingVolumeSphere,
    Tile3DContent,
    _apply_rotation_to_quats,
    _degree_from_coef_count,
    _fetch_json,
    _load_tile_content,
    _resolve_uri,
)


@dataclass
class TilesetSaveOptions:
    """Options for exporting a GaussianCloud as a 3D Tiles tileset."""

    chunk_size: float = 10.0
    """Side length of each cubic grid cell.  The bounding box is divided into
    a regular 3D grid where each cell has this edge length.  Points that fall
    into the same cell are written as a single GLB tile."""

    geometric_error: float = 100.0
    """Root ``geometricError`` written to the tileset.  Controls at what
    screen-space error the viewer decides to load tiles."""

    save_options: GltfSaveOptions = field(default_factory=GltfSaveOptions)
    """Options forwarded to :func:`~3dgs_io.gltf_io.save_gltf` for each
    chunk GLB (e.g. SPZ compression, metadata)."""

    max_workers: int | None = None
    """Maximum number of threads used to write GLB tiles in parallel.
    ``None`` (the default) lets :class:`~concurrent.futures.ThreadPoolExecutor`
    choose automatically based on available CPUs."""


@dataclass
class _CellAccumulator:
    """Per-cell accumulator for streaming tile distribution."""

    positions: list[np.ndarray] = field(default_factory=list)
    rotations: list[np.ndarray] = field(default_factory=list)
    scales: list[np.ndarray] = field(default_factory=list)
    colors: list[np.ndarray] = field(default_factory=list)
    alphas: list[np.ndarray] = field(default_factory=list)
    sh: list[np.ndarray] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_bounding_volume(
    camera_positions: np.ndarray,
) -> BoundingVolumeBox:
    """Compute an axis-aligned :class:`BoundingVolumeBox` from camera positions.

    Parameters
    ----------
    camera_positions:
        *(N, 3)* array of camera positions used during training.

    Returns
    -------
    A :class:`BoundingVolumeBox` whose ``center`` is the midpoint and
    ``half_axes`` is a diagonal matrix of per-axis half-extents.
    """
    cam = np.asarray(camera_positions, dtype=np.float64)
    if cam.ndim != 2 or cam.shape[1] != 3:
        raise ValueError(f"camera_positions must be (N, 3), got shape {cam.shape}")
    if cam.shape[0] == 0:
        raise ValueError("Cannot compute bounding volume from empty camera_positions")

    bbox_min = cam.min(axis=0)
    bbox_max = cam.max(axis=0)
    center = (bbox_min + bbox_max) / 2
    half_axes = np.diag((bbox_max - bbox_min) / 2)

    return BoundingVolumeBox(center=center, half_axes=half_axes)


def save_tileset(
    source: spz.GaussianCloud | str | Path | list[Tile3DContent],
    output_dir: str | Path,
    options: TilesetSaveOptions | None = None,
    *,
    root_transform: np.ndarray | None = None,
    ext_attributes: dict[str, np.ndarray] | None = None,
) -> Path:
    """Write a 3D Tiles tileset from one of several source types.

    Parameters
    ----------
    source:
        * :class:`~spz.GaussianCloud` — split into spatial chunks.
        * ``str`` / :class:`~pathlib.Path` — path or URL to an existing
          ``tileset.json`` to re-chunk (memory-efficient streaming).
        * ``list[Tile3DContent]`` — each tile is saved as-is (one GLB per
          entry) with its :attr:`~Tile3DContent.bounding_volume` written
          into the child node.
    output_dir:
        Directory where ``tileset.json`` and GLB files are written.
        Created if it does not exist.
    options:
        Export options.  See :class:`TilesetSaveOptions`.
    root_transform:
        Optional 4×4 column-major ECEF transform written to the root tile.
        Defaults to identity (omitted from JSON).

    Returns
    -------
    Path to the generated ``tileset.json``.
    """
    if options is None:
        options = TilesetSaveOptions()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(source, list):
        if ext_attributes is not None:
            raise ValueError(
                "ext_attributes is only supported for GaussianCloud sources; "
                "pre-built Tile3DContent lists carry their own GLBs."
            )
        return _save_from_tiles(source, output_dir, options, root_transform)

    cs = float(options.chunk_size)
    if cs <= 0:
        raise ValueError(f"chunk_size must be positive, got {cs}")

    if isinstance(source, spz.GaussianCloud):
        return _save_from_cloud(
            source, output_dir, cs, options, ext_attributes, root_transform=root_transform
        )
    if ext_attributes is not None:
        raise ValueError(
            "ext_attributes is only supported for GaussianCloud sources; "
            "tileset sources should keep ext on their input GLBs."
        )
    return _save_from_tileset(source, output_dir, cs, options)


# ---------------------------------------------------------------------------
# Tile3DContent list path (pre-built tiles written as-is)
# ---------------------------------------------------------------------------


def _save_from_tiles(
    tiles: list[Tile3DContent],
    output_dir: Path,
    options: TilesetSaveOptions,
    root_transform: np.ndarray | None,
) -> Path:
    if not tiles:
        raise ValueError("Cannot export an empty list of tiles")

    children: list[dict[str, Any]] = []
    bbox_min: np.ndarray | None = None
    bbox_max: np.ndarray | None = None
    save_tasks: list[tuple[spz.GaussianCloud, Path, dict[str, np.ndarray] | None]] = []

    for i, tile in enumerate(tiles):
        filename = f"tile_{i}.glb"
        save_tasks.append((tile.cloud, output_dir / filename, None))

        child: dict[str, Any] = {
            "geometricError": tile.geometric_error,
            "content": {"uri": filename},
        }

        transform = tile.transform.reshape(4, 4)
        if not np.allclose(transform, np.eye(4)):
            child["transform"] = [float(v) for v in transform.ravel()]

        if tile.bounding_volume is not None:
            child["boundingVolume"] = tile.bounding_volume.to_dict()
            bmin, bmax = _bounding_volume_to_aabb(tile.bounding_volume)
        else:
            n = tile.cloud.num_points
            positions = np.array(tile.cloud.positions, dtype=np.float32).reshape(n, 3)
            bmin = positions.min(axis=0).astype(np.float64)
            bmax = positions.max(axis=0).astype(np.float64)
            child["boundingVolume"] = {"box": _aabb_to_3dtiles_box(bmin, bmax)}

        bbox_min = bmin if bbox_min is None else np.minimum(bbox_min, bmin)
        bbox_max = bmax if bbox_max is None else np.maximum(bbox_max, bmax)
        children.append(child)

    assert bbox_min is not None and bbox_max is not None  # guaranteed by non-empty check

    _save_gltf_parallel(save_tasks, options)

    return _write_tileset_json(output_dir, bbox_min, bbox_max, children, options, root_transform)


# ---------------------------------------------------------------------------
# GaussianCloud path (all data already in memory)
# ---------------------------------------------------------------------------


def _save_from_cloud(
    gc: spz.GaussianCloud,
    output_dir: Path,
    cs: float,
    options: TilesetSaveOptions,
    ext_attributes: dict[str, np.ndarray] | None = None,
    *,
    root_transform: np.ndarray | None = None,
) -> Path:
    n = gc.num_points
    if n == 0:
        raise ValueError("Cannot export an empty GaussianCloud")

    ext_attrs: dict[str, np.ndarray] = {}
    if ext_attributes is not None:
        for name, arr in ext_attributes.items():
            arr = np.asarray(arr, dtype=np.float32).reshape(-1)
            if arr.shape[0] != n:
                raise ValueError(f"ext attribute {name!r} has {arr.shape[0]} entries, expected {n}")
            ext_attrs[name] = arr

    positions = np.array(gc.positions, dtype=np.float32).reshape(n, 3)
    rotations = np.array(gc.rotations, dtype=np.float32).reshape(n, 4)
    scales = np.array(gc.scales, dtype=np.float32).reshape(n, 3)
    colors = np.array(gc.colors, dtype=np.float32).reshape(n, 3)
    alphas = np.array(gc.alphas, dtype=np.float32)
    sh = np.array(gc.sh, dtype=np.float32)
    has_sh = sh.size > 0
    sh_per_point = sh.size // (n * 3) if has_sh else 0

    bbox_min = positions.min(axis=0)
    bbox_max = positions.max(axis=0)

    cell_keys = _assign_cell_keys(positions, bbox_min, bbox_max, cs)
    unique_keys, inverse = np.unique(cell_keys, return_inverse=True)

    children: list[dict[str, Any]] = []
    save_tasks: list[tuple[spz.GaussianCloud, Path, dict[str, np.ndarray] | None]] = []

    for chunk_idx, _key in enumerate(unique_keys):
        mask = inverse == chunk_idx
        if not mask.any():
            continue

        chunk_gc = spz.GaussianCloud()
        chunk_gc.positions = positions[mask].reshape(-1).astype(np.float32)
        chunk_gc.rotations = rotations[mask].reshape(-1).astype(np.float32)
        chunk_gc.scales = scales[mask].reshape(-1).astype(np.float32)
        chunk_gc.colors = colors[mask].reshape(-1).astype(np.float32)
        chunk_gc.alphas = alphas[mask].astype(np.float32)
        if has_sh:
            chunk_gc.sh_degree = gc.sh_degree
            sh_reshaped = sh.reshape(n, sh_per_point, 3)
            chunk_gc.sh = sh_reshaped[mask].reshape(-1).astype(np.float32)

        chunk_ext = {name: arr[mask] for name, arr in ext_attrs.items()}

        filename = f"chunk_{chunk_idx}.glb"
        save_tasks.append((chunk_gc, output_dir / filename, chunk_ext))

        chunk_positions = positions[mask]
        bounding_box = _aabb_to_3dtiles_box(
            chunk_positions.min(axis=0), chunk_positions.max(axis=0)
        )

        children.append(
            {
                "boundingVolume": {"box": bounding_box},
                "geometricError": 0.0,
                "content": {"uri": filename},
            }
        )

    _save_gltf_parallel(save_tasks, options)

    return _write_tileset_json(
        output_dir,
        bbox_min,
        bbox_max,
        children,
        options,
        root_transform,
        has_ext_attributes=bool(ext_attrs),
    )


# ---------------------------------------------------------------------------
# Tileset path (streaming re-chunk)
# ---------------------------------------------------------------------------


def _save_from_tileset(
    source: str | Path,
    output_dir: Path,
    cs: float,
    options: TilesetSaveOptions,
) -> Path:
    base_url, tileset = _fetch_json(source)
    root = tileset.get("root")
    if root is None:
        raise ValueError("Tileset missing 'root' tile")

    bbox_min, bbox_max = _root_aabb(root)

    cells: dict[int, _CellAccumulator] = {}
    sh_degree_seen: int | None = None

    for uri, transform in _walk_tile_uris(root, base_url):

        def _loader(p):
            cloud, _meta, ext = load_gltf_with_metadata(p)
            if ext:
                raise NotImplementedError(
                    "loading EXT_gaussian_lidar attributes from a tileset source is "
                    "not yet supported; pass a tile list instead"
                )
            return cloud

        gc = _load_tile_content(uri, _loader)
        n = gc.num_points
        if n == 0:
            del gc
            continue

        positions = np.array(gc.positions, dtype=np.float32).reshape(n, 3)
        rotations = np.array(gc.rotations, dtype=np.float32).reshape(n, 4)
        scales = np.array(gc.scales, dtype=np.float32).reshape(n, 3)
        colors = np.array(gc.colors, dtype=np.float32).reshape(n, 3)
        alphas = np.array(gc.alphas, dtype=np.float32)
        sh = np.array(gc.sh, dtype=np.float32)

        # 3D Tiles stores transforms column-major; convert to row-major.
        m = transform.reshape(4, 4).T
        r = m[:3, :3]
        t = m[:3, 3]
        if not np.allclose(m, np.eye(4)):
            positions = (positions @ r.T + t).astype(np.float32)
            rotations = _apply_rotation_to_quats(r, rotations).astype(np.float32)

        has_sh = sh.size > 0
        if has_sh:
            sh_per_point = sh.size // (n * 3)
            sh_reshaped = sh.reshape(n, sh_per_point, 3)
            if sh_degree_seen is None:
                sh_degree_seen = gc.sh_degree
        else:
            sh_reshaped = None

        del gc

        cell_keys = _assign_cell_keys(positions, bbox_min, bbox_max, cs)

        # Distribute points to cells via argsort (O(n log n) vs O(n*k)).
        order = np.argsort(cell_keys)
        sorted_keys = cell_keys[order]
        split_indices = np.flatnonzero(np.diff(sorted_keys)) + 1
        groups = np.split(order, split_indices)
        unique_keys = sorted_keys[np.r_[0, split_indices]]

        for key, group_idx in zip(unique_keys, groups, strict=True):
            cell = cells.setdefault(int(key), _CellAccumulator())
            cell.positions.append(positions[group_idx])
            cell.rotations.append(rotations[group_idx])
            cell.scales.append(scales[group_idx])
            cell.colors.append(colors[group_idx])
            cell.alphas.append(alphas[group_idx])
            if sh_reshaped is not None:
                cell.sh.append(sh_reshaped[group_idx])

    if not cells:
        raise ValueError("Source tileset contains no loadable camera 3DGS tiles")

    children: list[dict[str, Any]] = []
    futures: list[Future[None]] = []

    with ThreadPoolExecutor(max_workers=options.max_workers) as executor:
        for chunk_idx, key in enumerate(sorted(cells)):
            cell = cells[key]
            pos = np.concatenate(cell.positions)
            chunk_gc = spz.GaussianCloud()
            chunk_gc.positions = pos.reshape(-1)
            chunk_gc.rotations = np.concatenate(cell.rotations).reshape(-1)
            chunk_gc.scales = np.concatenate(cell.scales).reshape(-1)
            chunk_gc.colors = np.concatenate(cell.colors).reshape(-1)
            chunk_gc.alphas = np.concatenate(cell.alphas)
            if cell.sh:
                chunk_gc.sh = np.concatenate(cell.sh).reshape(-1)
                if sh_degree_seen is not None:
                    chunk_gc.sh_degree = sh_degree_seen
                else:
                    chunk_gc.sh_degree = _degree_from_coef_count(cell.sh[0].shape[1])

            del cells[key]

            filename = f"chunk_{chunk_idx}.glb"
            futures.append(
                executor.submit(save_gltf, chunk_gc, output_dir / filename, options.save_options)
            )

            bounding_box = _aabb_to_3dtiles_box(pos.min(axis=0), pos.max(axis=0))
            children.append(
                {
                    "boundingVolume": {"box": bounding_box},
                    "geometricError": 0.0,
                    "content": {"uri": filename},
                }
            )

    for future in futures:
        future.result()

    return _write_tileset_json(output_dir, bbox_min, bbox_max, children, options)


# ---------------------------------------------------------------------------
# Parallel GLB writer
# ---------------------------------------------------------------------------


def _save_gltf_parallel(
    tasks: list[tuple[spz.GaussianCloud, Path, dict[str, np.ndarray] | None]],
    options: TilesetSaveOptions,
) -> None:
    """Save multiple GaussianClouds to GLB files in parallel.

    Each task is ``(gc, path, ext_attributes)``; pass ``None`` for
    ``ext_attributes`` when the tile has no extension arrays.
    """
    with ThreadPoolExecutor(max_workers=options.max_workers) as executor:
        futures = [
            executor.submit(save_gltf, gc, path, options.save_options, ext_attributes=ext)
            for gc, path, ext in tasks
        ]
        for future in futures:
            future.result()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _assign_cell_keys(
    positions: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    chunk_size: float,
) -> np.ndarray:
    """Return linearized grid-cell keys for each point."""
    grid_dims = np.maximum(np.ceil((bbox_max - bbox_min) / chunk_size), 1).astype(np.int32)
    cell_indices = np.floor((positions - bbox_min) / chunk_size).astype(np.int32)
    cell_indices = np.clip(cell_indices, 0, grid_dims - 1)
    return (
        cell_indices[:, 0] * grid_dims[1] * grid_dims[2]
        + cell_indices[:, 1] * grid_dims[2]
        + cell_indices[:, 2]
    )


def _write_tileset_json(
    output_dir: Path,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    children: list[dict[str, Any]],
    options: TilesetSaveOptions,
    root_transform: np.ndarray | None = None,
    *,
    has_ext_attributes: bool = False,
) -> Path:
    root_box = _aabb_to_3dtiles_box(bbox_min, bbox_max)

    # Declare glTF extensions used by tile content so CesiumJS routes
    # GLBs through the Gaussian Splatting pipeline instead of the
    # standard Model pipeline (which fails on SPZ virtual accessors).
    gltf_exts_used = ["KHR_gaussian_splatting"]
    gltf_exts_required = ["KHR_gaussian_splatting"]
    if options.save_options.spz_compression:
        spz_ext = "KHR_gaussian_splatting_compression_spz_2"
        gltf_exts_used.append(spz_ext)
        gltf_exts_required.append(spz_ext)
    if has_ext_attributes:
        gltf_exts_used.append(EXT_GAUSSIAN_LIDAR_NAME)

    root: dict[str, Any] = {
        "boundingVolume": {"box": root_box},
        "geometricError": options.geometric_error,
        "refine": "ADD",
        "children": children,
    }

    if root_transform is not None:
        root["transform"] = [float(v) for v in np.asarray(root_transform, dtype=np.float64).ravel()]

    tileset: dict[str, Any] = {
        "asset": {"version": "1.1", "generator": "3dgs-io"},
        "geometricError": options.geometric_error,
        "extensionsUsed": ["3DTILES_content_gltf"],
        "extensions": {
            "3DTILES_content_gltf": {
                "extensionsUsed": gltf_exts_used,
                "extensionsRequired": gltf_exts_required,
            }
        },
        "root": root,
    }
    tileset_path = output_dir / "tileset.json"
    tileset_path.write_text(json.dumps(tileset, indent=2))
    return tileset_path


def _aabb_to_3dtiles_box(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> list[float]:
    """Convert an axis-aligned bounding box to 3D Tiles box format.

    Returns the 12-element array ``[cx, cy, cz, hx, 0, 0, 0, hy, 0, 0, 0, hz]``.
    """
    center = ((bbox_min + bbox_max) / 2).astype(np.float64)
    half = ((bbox_max - bbox_min) / 2).astype(np.float64)
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


def _bounding_volume_to_aabb(bv: BoundingVolume) -> tuple[np.ndarray, np.ndarray]:
    """Compute an axis-aligned bounding box ``(min, max)`` from a typed volume."""
    if isinstance(bv, BoundingVolumeBox):
        half_extent = np.abs(bv.half_axes).sum(axis=0)
        return bv.center - half_extent, bv.center + half_extent
    if isinstance(bv, BoundingVolumeSphere):
        r = np.full(3, bv.radius, dtype=np.float64)
        return bv.center - r, bv.center + r
    if isinstance(bv, BoundingVolumeRegion):
        raise TypeError("Cannot compute a Cartesian AABB from a geographic BoundingVolumeRegion")
    raise TypeError(f"Unknown bounding volume type: {type(bv)}")  # pragma: no cover


def _root_aabb(root: dict) -> tuple[np.ndarray, np.ndarray]:
    """Extract an AABB from the root tile's bounding volume."""
    bv = root.get("boundingVolume", {})
    box = bv.get("box")
    if box is not None and len(box) == 12:
        return _bounding_volume_to_aabb(BoundingVolumeBox.from_list(box))

    raise ValueError("Root bounding volume must be a 'box'; found keys: " + ", ".join(bv.keys()))


def _walk_tile_uris(
    tile: dict,
    base_url: str,
    parent_transform: np.ndarray | None = None,
) -> Iterator[tuple[str, np.ndarray]]:
    """Walk the tile tree and yield ``(resolved_uri, cumulative_transform)``
    for each leaf tile that has content, without loading the content itself.
    """
    if parent_transform is None:
        parent_transform = np.eye(4, dtype=np.float64)

    transform = parent_transform
    local = tile.get("transform")
    if local is not None:
        local_mat = np.array(local, dtype=np.float64).reshape(4, 4)
        parent_mat = parent_transform.reshape(4, 4)
        transform = (local_mat @ parent_mat).astype(np.float64)

    children = tile.get("children", [])
    is_leaf = not children

    contents = tile.get("contents")
    if contents is None:
        content = tile.get("content")
        contents = [content] if content is not None else []

    if is_leaf:
        for entry in contents:
            if entry is None:
                continue
            uri = entry.get("uri") or entry.get("url")
            if uri is None:
                continue
            yield _resolve_uri(base_url, uri), transform

    for child in children:
        yield from _walk_tile_uris(child, base_url, transform)
