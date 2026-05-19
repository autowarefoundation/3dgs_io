"""3D Tiles (OGC) writer with spatial chunk splitting.

Provides two public functions:

* :func:`save_tileset` — split a single :class:`~spz.GaussianCloud` that is
  already in memory.
* :func:`export_tileset` — read an existing ``tileset.json`` tile-by-tile and
  re-chunk it, keeping only one input tile in memory at a time.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import spz

from .gltf_io import GltfSaveOptions, save_gltf
from .tiles_io import (
    _apply_rotation_to_quats,
    _degree_from_coef_count,
    _fetch_json,
    _load_tile_content,
    _resolve_uri,
    load_gltf,
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
# In-memory export (GaussianCloud already loaded)
# ---------------------------------------------------------------------------


def save_tileset(
    gc: spz.GaussianCloud,
    output_dir: str | Path,
    options: TilesetSaveOptions | None = None,
) -> Path:
    """Split a GaussianCloud into spatial chunks and write a 3D Tiles tileset.

    Parameters
    ----------
    gc:
        The Gaussian cloud to export.
    output_dir:
        Directory where ``tileset.json`` and chunk GLB files are written.
        Created if it does not exist.
    options:
        Export options.  See :class:`TilesetSaveOptions`.

    Returns
    -------
    Path to the generated ``tileset.json``.
    """
    if options is None:
        options = TilesetSaveOptions()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = gc.num_points
    if n == 0:
        raise ValueError("Cannot export an empty GaussianCloud")

    cs = float(options.chunk_size)
    if cs <= 0:
        raise ValueError(f"chunk_size must be positive, got {cs}")

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
            sh_reshaped = sh.reshape(n, sh_per_point, 3)
            chunk_gc.sh = sh_reshaped[mask].reshape(-1).astype(np.float32)
            chunk_gc.sh_degree = gc.sh_degree

        filename = f"chunk_{chunk_idx}.glb"
        save_gltf(chunk_gc, output_dir / filename, options.save_options)

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

    return _write_tileset_json(output_dir, bbox_min, bbox_max, children, options)


# ---------------------------------------------------------------------------
# Streaming export (reads tiles one-by-one from an existing tileset)
# ---------------------------------------------------------------------------


def export_tileset(
    source: str | Path,
    output_dir: str | Path,
    options: TilesetSaveOptions | None = None,
) -> Path:
    """Re-chunk an existing 3D Tiles tileset, loading one tile at a time.

    This is the memory-efficient counterpart to loading the entire tileset
    with :func:`~3dgs_io.load_tileset`, merging it, and calling
    :func:`save_tileset`.  Only one input tile is in memory at any point;
    the output cell accumulators grow incrementally.

    Parameters
    ----------
    source:
        Path or URL to the source ``tileset.json``.
    output_dir:
        Directory where the re-chunked ``tileset.json`` and GLB files are
        written.  Created if it does not exist.
    options:
        Export options.  See :class:`TilesetSaveOptions`.

    Returns
    -------
    Path to the generated ``tileset.json``.
    """
    if options is None:
        options = TilesetSaveOptions()

    cs = float(options.chunk_size)
    if cs <= 0:
        raise ValueError(f"chunk_size must be positive, got {cs}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_url, tileset = _fetch_json(source)
    root = tileset.get("root")
    if root is None:
        raise ValueError("Tileset missing 'root' tile")

    bbox_min, bbox_max = _root_aabb(root)

    cells: dict[int, _CellAccumulator] = {}
    sh_degree_seen: int | None = None

    for uri, transform in _walk_tile_uris(root, base_url):
        gc = _load_tile_content(uri, load_gltf)
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
        save_gltf(chunk_gc, output_dir / filename, options.save_options)

        bounding_box = _aabb_to_3dtiles_box(pos.min(axis=0), pos.max(axis=0))
        children.append(
            {
                "boundingVolume": {"box": bounding_box},
                "geometricError": 0.0,
                "content": {"uri": filename},
            }
        )

    return _write_tileset_json(output_dir, bbox_min, bbox_max, children, options)


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
) -> Path:
    root_box = _aabb_to_3dtiles_box(bbox_min, bbox_max)
    tileset: dict[str, Any] = {
        "asset": {"version": "1.1", "generator": "3dgs-io"},
        "geometricError": options.geometric_error,
        "root": {
            "boundingVolume": {"box": root_box},
            "geometricError": options.geometric_error,
            "refine": "ADD",
            "children": children,
        },
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


def _root_aabb(root: dict) -> tuple[np.ndarray, np.ndarray]:
    """Extract an AABB from the root tile's bounding volume.

    Supports the ``box`` bounding volume type.  The box format is
    ``[cx, cy, cz, ax0, ax1, ax2, ay0, ay1, ay2, az0, az1, az2]``
    where the last 9 values are 3 half-axis vectors.
    """
    bv = root.get("boundingVolume", {})
    box = bv.get("box")
    if box is not None and len(box) == 12:
        center = np.array(box[0:3], dtype=np.float64)
        axes = np.array(
            [[box[3], box[4], box[5]], [box[6], box[7], box[8]], [box[9], box[10], box[11]]],
            dtype=np.float64,
        )
        # For each world axis, the half-extent is the sum of absolute
        # projections of all half-axis vectors onto that axis.
        half_extent = np.abs(axes).sum(axis=0)
        return center - half_extent, center + half_extent

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
