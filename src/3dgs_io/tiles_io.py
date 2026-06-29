"""3D Tiles (OGC) reader for tilesets containing Gaussian-splatting content.

Traverses a ``tileset.json`` (local file or HTTP/HTTPS URL), fetches each tile's
glTF/GLB content, and returns :class:`Tile3DContent` objects.

Supports both ``content.uri`` (single content) and ``contents`` (3D Tiles 1.1
multi-content). Only explicit (non-implicit) tiling and cumulative
``transform`` matrices are implemented.
"""

from __future__ import annotations

import json
import tempfile
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np
import spz
from scipy.spatial.transform import Rotation

from .gltf_io import load_gltf


@dataclass
class BoundingVolumeBox:
    """A bounding volume defined as an oriented box."""

    center: np.ndarray
    """Centre of the box as a 3-element float64 array ``(cx, cy, cz)``."""

    half_axes: np.ndarray
    """Three half-axis column vectors as a 3x3 float64 array (columns = axes)."""

    @staticmethod
    def from_list(values: list[float]) -> BoundingVolumeBox:
        """Construct from the 12-element array used in 3D Tiles JSON."""
        arr = np.array(values, dtype=np.float64)
        return BoundingVolumeBox(
            center=arr[:3],
            half_axes=arr[3:].reshape(3, 3),
        )

    def to_dict(self) -> dict[str, list[float]]:
        """Serialize to the 3D Tiles JSON ``boundingVolume`` dict."""
        return {"box": self.center.tolist() + self.half_axes.ravel().tolist()}


@dataclass
class BoundingVolumeRegion:
    """A bounding volume defined as a geographic region.

    Longitude and latitude values are in **radians**; heights are in metres
    above the WGS 84 ellipsoid.
    """

    west: float
    """Western longitude (radians)."""

    south: float
    """Southern latitude (radians)."""

    east: float
    """Eastern longitude (radians)."""

    north: float
    """Northern latitude (radians)."""

    min_height: float
    """Minimum height above the ellipsoid (metres)."""

    max_height: float
    """Maximum height above the ellipsoid (metres)."""

    @staticmethod
    def from_list(values: list[float]) -> BoundingVolumeRegion:
        """Construct from the 6-element array used in 3D Tiles JSON."""
        return BoundingVolumeRegion(
            west=values[0],
            south=values[1],
            east=values[2],
            north=values[3],
            min_height=values[4],
            max_height=values[5],
        )

    def to_dict(self) -> dict[str, list[float]]:
        """Serialize to the 3D Tiles JSON ``boundingVolume`` dict."""
        return {
            "region": [
                float(self.west),
                float(self.south),
                float(self.east),
                float(self.north),
                float(self.min_height),
                float(self.max_height),
            ]
        }


@dataclass
class BoundingVolumeSphere:
    """A bounding volume defined as a sphere."""

    center: np.ndarray
    """Centre of the sphere as a 3-element float64 array ``(cx, cy, cz)``."""

    radius: float
    """Radius of the sphere (metres)."""

    @staticmethod
    def from_list(values: list[float]) -> BoundingVolumeSphere:
        """Construct from the 4-element array used in 3D Tiles JSON."""
        return BoundingVolumeSphere(
            center=np.array(values[:3], dtype=np.float64),
            radius=float(values[3]),
        )

    def to_dict(self) -> dict[str, list[float]]:
        """Serialize to the 3D Tiles JSON ``boundingVolume`` dict."""
        return {"sphere": self.center.tolist() + [float(self.radius)]}


BoundingVolume = BoundingVolumeBox | BoundingVolumeRegion | BoundingVolumeSphere


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

    bounding_volume: BoundingVolume | None = None
    """Parsed bounding volume from the tileset (box, region, or sphere)."""


def load_tileset(
    source: str | Path,
    *,
    max_tiles: int | None = None,
    leaves_only: bool = True,
) -> list[Tile3DContent]:
    """Load a 3D Tileset and return its tiles.

    Parameters
    ----------
    source:
        Local path or HTTP(S) URL to a ``tileset.json`` file.
    max_tiles:
        Optional cap on the number of tiles loaded (useful for large tilesets).
    leaves_only:
        When ``True`` (default), only tiles without children are loaded. When
        ``False``, every tile that has a ``content`` entry is loaded.

    Returns
    -------
    A list of :class:`Tile3DContent`.
    """
    base_url, tileset = _fetch_json(source)

    root = tileset.get("root")
    if root is None:
        raise ValueError("Tileset missing 'root' tile")

    results: list[Tile3DContent] = []
    identity = np.eye(4, dtype=np.float64)
    _traverse(
        root,
        base_url=base_url,
        parent_transform=identity,
        parent_refine="REPLACE",
        results=results,
        max_tiles=max_tiles,
        leaves_only=leaves_only,
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


def _parse_bounding_volume(raw: dict | None) -> BoundingVolume | None:
    """Convert a raw ``boundingVolume`` dict to a typed dataclass."""
    if raw is None:
        return None
    if "box" in raw:
        return BoundingVolumeBox.from_list(raw["box"])
    if "region" in raw:
        return BoundingVolumeRegion.from_list(raw["region"])
    if "sphere" in raw:
        return BoundingVolumeSphere.from_list(raw["sphere"])
    return None


def _traverse(
    tile: dict,
    *,
    base_url: str,
    parent_transform: np.ndarray,
    parent_refine: str,
    results: list[Tile3DContent],
    max_tiles: int | None,
    leaves_only: bool,
    _visited: frozenset[str] = frozenset(),
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
        geo_error = float(tile.get("geometricError", 0.0))
        bv = _parse_bounding_volume(tile.get("boundingVolume"))

        for entry in contents:
            if entry is None:
                continue
            if max_tiles is not None and len(results) >= max_tiles:
                return
            uri = entry.get("uri") or entry.get("url")
            if uri is None:
                continue

            resolved = _resolve_uri(base_url, uri)

            # External tileset (nested .json) — recurse into it.
            parsed_uri = urllib.parse.urlparse(resolved)
            if Path(parsed_uri.path).suffix.lower() == ".json":
                if resolved in _visited:
                    continue
                ext_base, ext_data = _fetch_json(resolved)
                ext_root = ext_data.get("root")
                if ext_root is not None:
                    _traverse(
                        ext_root,
                        base_url=ext_base,
                        parent_transform=transform,
                        parent_refine=refine,
                        results=results,
                        max_tiles=max_tiles,
                        leaves_only=leaves_only,
                        _visited=_visited | {resolved},
                    )
                continue

            cloud = _load_tile_content(resolved, load_gltf)
            results.append(
                Tile3DContent(
                    cloud=cloud,
                    transform=transform,
                    content_uri=resolved,
                    geometric_error=geo_error,
                    refine=refine,
                    bounding_volume=bv,
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
            _visited=_visited,
        )


def _fetch_json(source: str | Path) -> tuple[str, dict]:
    """Return (base_url, tileset_dict).

    ``base_url`` is the URL/path that relative content URIs should resolve
    against (the directory containing the tileset.json, with trailing slash).
    """
    src = str(source)
    parsed = urllib.parse.urlparse(src)
    if isinstance(source, Path) or parsed.scheme in ("", "file"):
        if isinstance(source, Path) or parsed.scheme == "":
            path = Path(source)
        else:
            path = Path(urllib.parse.unquote(parsed.path))
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        base = path.resolve().parent.as_uri() + "/"
        return base, data

    # with urllib.request.urlopen(src) as resp:  # noqa: S310 - intentional HTTP fetch
    #     data = json.loads(resp.read().decode("utf-8"))
    req = urllib.request.Request(
        src, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) 3dgs_io/1.0"}
    )
    with urllib.request.urlopen(req) as resp:  # noqa: S310 - intentional HTTP fetch
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

    quats_2d = np.atleast_2d(quats)
    combined = Rotation.from_matrix(rot) * Rotation.from_quat(quats_2d)
    return combined.as_quat()
