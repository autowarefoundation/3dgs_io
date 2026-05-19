"""3D Tiles (OGC) writer with spatial chunk splitting.

Splits a :class:`~spz.GaussianCloud` into a regular 3D grid and writes each
non-empty cell as a separate GLB tile, producing a valid ``tileset.json`` that
can be loaded by CesiumJS or any 3D Tiles 1.1 viewer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import spz

from .gltf_io import GltfSaveOptions, save_gltf


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
    bbox_extent = bbox_max - bbox_min

    cs = float(options.chunk_size)
    if cs <= 0:
        raise ValueError(f"chunk_size must be positive, got {cs}")

    grid_dims = np.maximum(np.ceil(bbox_extent / cs), 1).astype(np.int32)

    cell_indices = np.floor((positions - bbox_min) / cs).astype(np.int32)
    cell_indices = np.clip(cell_indices, 0, grid_dims - 1)

    cell_keys = (
        cell_indices[:, 0] * grid_dims[1] * grid_dims[2]
        + cell_indices[:, 1] * grid_dims[2]
        + cell_indices[:, 2]
    )

    unique_keys, inverse = np.unique(cell_keys, return_inverse=True)

    children: list[dict[str, Any]] = []

    for chunk_idx, _key in enumerate(unique_keys):
        mask = inverse == chunk_idx
        chunk_n = int(mask.sum())
        if chunk_n == 0:
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
        c_min = chunk_positions.min(axis=0)
        c_max = chunk_positions.max(axis=0)
        bounding_box = _aabb_to_3dtiles_box(c_min, c_max)

        children.append(
            {
                "boundingVolume": {"box": bounding_box},
                "geometricError": 0.0,
                "content": {"uri": filename},
            }
        )

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
