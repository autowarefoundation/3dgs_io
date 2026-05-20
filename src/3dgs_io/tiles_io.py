"""3D Tiles (OGC) reader for tilesets containing Gaussian-splatting content.

Traverses a ``tileset.json`` (local file or HTTP/HTTPS URL), fetches each tile's
glTF/GLB content, and returns one or more cloud objects.

Supports:

* **Single content** -- ``content.uri`` (legacy / simple tilesets).
* **Multiple contents** -- ``contents`` array with ``group`` indices
  (3D Tiles 1.1) for mixed Camera 3DGS + LiDAR 2DGS tilesets.

Only explicit (non-implicit) tiling and cumulative ``transform`` matrices are
implemented.
"""

from __future__ import annotations

import json
import tempfile
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TypeVar

import numpy as np
import spz

from .gltf_io import load_gltf
from .lidar_2dgs import LidarGaussianCloud, load_lidar_gltf


class LayerType(str, Enum):
    """Content layer types for multi-content tilesets."""

    CAMERA_3DGS = "camera_3dgs"
    LIDAR_2DGS = "lidar_2dgs"


@dataclass
class Tile3DContent:
    """A single loaded camera 3DGS tile from a 3D Tileset."""

    cloud: spz.GaussianCloud
    """Gaussian cloud decoded from the tile's glTF content (tile-local coords)."""

    transform: np.ndarray
    """Cumulative 4x4 world transform (column-major, as stored in 3D Tiles)."""

    content_uri: str
    """Resolved URI/path of the tile content that was loaded."""

    geometric_error: float = 0.0
    """Tile ``geometricError`` (screen-space error threshold)."""

    refine: str = "REPLACE"
    """Refinement strategy inherited from the nearest ancestor (``ADD``/``REPLACE``)."""


@dataclass
class LidarTile3DContent:
    """A single loaded LiDAR 2DGS tile from a 3D Tileset."""

    cloud: LidarGaussianCloud
    """LiDAR Gaussian cloud decoded from the tile's glTF content (tile-local coords)."""

    transform: np.ndarray
    """Cumulative 4x4 world transform (column-major, as stored in 3D Tiles)."""

    content_uri: str
    """Resolved URI/path of the tile content that was loaded."""

    geometric_error: float = 0.0
    """Tile ``geometricError`` (screen-space error threshold)."""

    refine: str = "REPLACE"
    """Refinement strategy inherited from the nearest ancestor (``ADD``/``REPLACE``)."""


def load_tileset(
    source: str | Path,
    *,
    layer: str | LayerType = LayerType.CAMERA_3DGS,
    max_tiles: int | None = None,
    leaves_only: bool = True,
) -> list[Tile3DContent] | list[LidarTile3DContent]:
    """Load a 3D Tileset and return tiles filtered by layer.

    Parameters
    ----------
    source:
        Local path or HTTP(S) URL to a ``tileset.json`` file.
    layer:
        Which content layer to return.  Defaults to ``"camera_3dgs"``
        (backward-compatible with previous behaviour).  Use ``"lidar_2dgs"``
        for LiDAR content.
    max_tiles:
        Optional cap on the number of tiles loaded (useful for large tilesets).
    leaves_only:
        When ``True`` (default), only tiles without children are loaded. When
        ``False``, every tile that has a ``content`` entry is loaded.

    Returns
    -------
    A list of :class:`Tile3DContent` or :class:`LidarTile3DContent` depending
    on the *layer* parameter.
    """
    layer = LayerType(layer)

    base_url, tileset = _fetch_json(source)

    root = tileset.get("root")
    if root is None:
        raise ValueError("Tileset missing 'root' tile")

    results: list[Tile3DContent | LidarTile3DContent] = []
    identity = np.eye(4, dtype=np.float64)
    group_types = _parse_group_types(tileset)
    _traverse(
        root,
        base_url=base_url,
        parent_transform=identity,
        parent_refine="REPLACE",
        results=results,
        max_tiles=max_tiles,
        leaves_only=leaves_only,
        group_types=group_types,
        target_layer=layer,
    )
    return results


def merge_tileset(tiles: list[Tile3DContent]) -> spz.GaussianCloud:
    """Merge loaded tiles into a single GaussianCloud with world-space positions.

    Applies each tile's cumulative ``transform`` to positions and rotations.
    Scales and other attributes are copied as-is.
    """
    if not tiles:
        raise ValueError("Cannot merge an empty list of tiles")

    positions_list: list[np.ndarray] = []
    rotations_list: list[np.ndarray] = []
    scales_list: list[np.ndarray] = []
    colors_list: list[np.ndarray] = []
    alphas_list: list[np.ndarray] = []
    sh_list: list[np.ndarray] = []

    sh_coefs_per_point: int | None = None

    for tile in tiles:
        gc = tile.cloud
        n = gc.num_points
        if n == 0:
            continue

        positions = np.array(gc.positions, dtype=np.float32).reshape(n, 3)
        rotations = np.array(gc.rotations, dtype=np.float32).reshape(n, 4)
        scales = np.array(gc.scales, dtype=np.float32).reshape(n, 3)
        colors = np.array(gc.colors, dtype=np.float32).reshape(n, 3)
        alphas = np.array(gc.alphas, dtype=np.float32)
        sh = np.array(gc.sh, dtype=np.float32)

        # Apply transform to positions/rotations.
        # 3D Tiles transforms are column-major 4x4 matrices.
        m = tile.transform.reshape(4, 4).T  # to row-major
        r = m[:3, :3]
        t = m[:3, 3]
        positions = (positions @ r.T + t).astype(np.float32)
        rotations = _apply_rotation_to_quats(r, rotations).astype(np.float32)

        positions_list.append(positions)
        rotations_list.append(rotations)
        scales_list.append(scales)
        colors_list.append(colors)
        alphas_list.append(alphas)

        if sh.size:
            per_point = sh.size // (n * 3)
            if sh_coefs_per_point is None:
                sh_coefs_per_point = per_point
            elif sh_coefs_per_point != per_point:
                raise ValueError(
                    "Cannot merge tiles with different SH degrees "
                    f"({sh_coefs_per_point} vs {per_point} coefficients/point)"
                )
            sh_list.append(sh.reshape(n, per_point, 3))
        elif sh_coefs_per_point not in (None, 0):
            raise ValueError("Cannot merge: tile has no SH data but others do")
        else:
            sh_coefs_per_point = 0

    merged = spz.GaussianCloud()
    merged.positions = np.concatenate(positions_list).reshape(-1)
    merged.rotations = np.concatenate(rotations_list).reshape(-1)
    merged.scales = np.concatenate(scales_list).reshape(-1)
    merged.colors = np.concatenate(colors_list).reshape(-1)
    merged.alphas = np.concatenate(alphas_list)
    if sh_list:
        sh_cat = np.concatenate(sh_list)  # (N_total, per_point, 3)
        merged.sh_degree = _degree_from_coef_count(sh_cat.shape[1])
        merged.sh = sh_cat.reshape(-1).astype(np.float32)
    return merged


def _degree_from_coef_count(n_coefs: int) -> int:
    if n_coefs >= 15:
        return 3
    if n_coefs >= 8:
        return 2
    if n_coefs >= 3:
        return 1
    return 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_group_types(tileset: dict) -> dict[int, LayerType]:
    """Extract group index -> content type mapping from tileset metadata."""
    groups = tileset.get("groups", [])
    result: dict[int, LayerType] = {}
    for i, group in enumerate(groups):
        props = group.get("properties", {})
        raw_type = props.get("type", LayerType.CAMERA_3DGS.value)
        try:
            result[i] = LayerType(raw_type)
        except ValueError:
            result[i] = LayerType.CAMERA_3DGS
    return result


def _traverse(
    tile: dict,
    *,
    base_url: str,
    parent_transform: np.ndarray,
    parent_refine: str,
    results: list[Tile3DContent | LidarTile3DContent],
    max_tiles: int | None,
    leaves_only: bool,
    group_types: dict[int, LayerType],
    target_layer: LayerType,
) -> None:
    transform = parent_transform
    local = tile.get("transform")
    if local is not None:
        local_mat = np.array(local, dtype=np.float64).reshape(4, 4)
        parent_mat = parent_transform.reshape(4, 4)
        # Column-major multiplication: world = parent @ local
        # With row-major numpy storage of column-major data, equivalent op is local @ parent
        transform = (local_mat @ parent_mat).astype(np.float64)

    refine = tile.get("refine", parent_refine).upper()
    children = tile.get("children", [])

    is_leaf = not children

    # 3D Tiles 1.1: ``contents`` (plural) takes precedence over ``content``.
    contents = tile.get("contents")
    if contents is not None:
        has_content = True
    else:
        content = tile.get("content")
        contents = [content] if content is not None else []
        has_content = content is not None

    should_load = has_content and (is_leaf or not leaves_only)

    if should_load:
        for entry in contents:
            if entry is None:
                continue
            if max_tiles is not None and len(results) >= max_tiles:
                return
            uri = entry.get("uri") or entry.get("url")
            if uri is None:
                continue
            group_idx = entry.get("group")
            content_type = (
                group_types.get(group_idx, LayerType.CAMERA_3DGS)
                if group_idx is not None
                else LayerType.CAMERA_3DGS
            )

            # Skip content that does not match the requested layer.
            if content_type != target_layer:
                continue

            resolved = _resolve_uri(base_url, uri)
            geo_error = float(tile.get("geometricError", 0.0))

            if content_type == LayerType.LIDAR_2DGS:
                cloud = _load_tile_content(resolved, load_lidar_gltf)
                results.append(
                    LidarTile3DContent(
                        cloud=cloud,
                        transform=transform,
                        content_uri=resolved,
                        geometric_error=geo_error,
                        refine=refine,
                    )
                )
            else:
                cloud = _load_tile_content(resolved, load_gltf)
                results.append(
                    Tile3DContent(
                        cloud=cloud,
                        transform=transform,
                        content_uri=resolved,
                        geometric_error=geo_error,
                        refine=refine,
                    )
                )

    for child in children:
        if max_tiles is not None and len(results) >= max_tiles:
            return
        _traverse(
            child,
            base_url=base_url,
            parent_transform=transform,
            parent_refine=refine,
            results=results,
            max_tiles=max_tiles,
            leaves_only=leaves_only,
            group_types=group_types,
            target_layer=target_layer,
        )


def _fetch_json(source: str | Path) -> tuple[str, dict]:
    """Return (base_url, tileset_dict).

    ``base_url`` is the URL/path that relative content URIs should resolve
    against (the directory containing the tileset.json, with trailing slash).
    """
    src = str(source)
    if isinstance(source, Path) or urllib.parse.urlparse(src).scheme in ("", "file"):
        path = Path(source)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        base = path.resolve().parent.as_uri() + "/"
        return base, data

    with urllib.request.urlopen(src) as resp:  # noqa: S310 - intentional HTTP fetch
        data = json.loads(resp.read().decode("utf-8"))
    parsed = urllib.parse.urlparse(src)
    dir_path = parsed.path.rsplit("/", 1)[0] + "/"
    base = urllib.parse.urlunparse(parsed._replace(path=dir_path, query="", fragment=""))
    return base, data


_T = TypeVar("_T")


def _load_tile_content(uri: str, loader: Callable[[Path], _T]) -> _T:
    """Fetch a .glb/.gltf content and decode it with the given loader."""
    parsed = urllib.parse.urlparse(uri)
    suffix = Path(parsed.path).suffix.lower() or ".glb"

    if parsed.scheme in ("", "file"):
        path = parsed.path if parsed.scheme == "file" else uri
        return loader(Path(urllib.parse.unquote(path)))

    with urllib.request.urlopen(uri) as resp:  # noqa: S310 - intentional HTTP fetch
        data = resp.read()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        return loader(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _resolve_uri(base: str, uri: str) -> str:
    """Resolve ``uri`` relative to ``base`` for both URL and file paths."""
    if urllib.parse.urlparse(uri).scheme in ("http", "https", "file"):
        return uri
    if Path(uri).is_absolute():
        return str(Path(uri))
    return urllib.parse.urljoin(base, uri)


def _apply_rotation_to_quats(r: np.ndarray, quats: np.ndarray) -> np.ndarray:
    """Compose the 3x3 rotation ``r`` with each quaternion in ``quats``.

    ``quats`` uses glTF convention ``(x, y, z, w)``. The rotation part of
    ``r`` is extracted via a polar-like step: we orthonormalise columns and
    take the rotation factor so that scale does not pollute the quaternion.
    """
    u, _, vt = np.linalg.svd(r)
    rot = (u @ vt).astype(np.float64)
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1
        rot = u @ vt

    rq = _matrix_to_quat(rot)
    return _quat_multiply(rq, quats)


def _matrix_to_quat(m: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a quaternion (x, y, z, w)."""
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float64)


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply quaternion ``q1`` (shape (4,)) with each quaternion in ``q2``
    (shape (N, 4)). Both use (x, y, z, w) order. Returns (N, 4)."""
    if q2.ndim == 1:
        q2 = q2.reshape(1, 4)
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return np.stack([x, y, z, w], axis=1)
