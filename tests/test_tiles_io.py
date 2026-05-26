"""Tests for the 3D Tiles reader.

Uses small publicly-hosted Gaussian-splatting tilesets from CesiumJS's test
fixtures (``CesiumGS/cesium`` repository).
"""

from __future__ import annotations

import importlib
import json
import socket
import struct
from pathlib import Path

import numpy as np
import pytest
import spz

_mod = importlib.import_module("3dgs_io")
load_tileset = _mod.load_tileset
merge_tileset = _mod.merge_tileset
Tile3DContent = _mod.Tile3DContent
LidarTile3DContent = _mod.LidarTile3DContent
LidarGaussianCloud = _mod.LidarGaussianCloud
BoundingVolumeBox = _mod.BoundingVolumeBox
BoundingVolumeRegion = _mod.BoundingVolumeRegion
BoundingVolumeSphere = _mod.BoundingVolumeSphere
save_gltf = _mod.save_gltf
save_lidar_gltf = _mod.save_lidar_gltf
GltfSaveOptions = _mod.GltfSaveOptions
GaussianCloud = spz.GaussianCloud

_CESIUM_RAW = (
    "https://raw.githubusercontent.com/CesiumGS/cesium/main/Specs/Data/Cesium3DTiles/GaussianSplats"
)
_SH_UNIT_CUBE_URL = f"{_CESIUM_RAW}/sh_unit_cube/tileset.json"
_TOWER_URL = f"{_CESIUM_RAW}/tower/tileset.json"


def _has_network(host: str = "raw.githubusercontent.com", port: int = 443) -> bool:
    try:
        with socket.create_connection((host, port), timeout=3):
            return True
    except OSError:
        return False


network = pytest.mark.skipif(
    not _has_network(), reason="needs network access to raw.githubusercontent.com"
)


# ── local-file tileset (no network) ─────────────────────────────────────────


def _build_local_tileset(tmp_path: Path, n: int = 20) -> Path:
    """Write a 3D Tiles 1.1 tileset with one SPZ-compressed GLB child."""
    rng = np.random.default_rng(7)
    sh_c0 = 0.2820947917738781

    gc = GaussianCloud()
    positions = rng.uniform(-5.0, 5.0, (n, 3)).astype(np.float32)
    rgb = rng.integers(0, 256, (n, 3), dtype=np.uint8)
    colors_sh = ((rgb.astype(np.float32) / 255.0) - 0.5) / sh_c0
    alpha_u8 = rng.integers(1, 255, (n,), dtype=np.uint8)
    alpha_01 = alpha_u8.astype(np.float64) / 255.0
    alphas = np.log(alpha_01 / (1 - alpha_01)).astype(np.float32)
    rots = rng.standard_normal((n, 4)).astype(np.float32)
    rots /= np.linalg.norm(rots, axis=1, keepdims=True)
    scales = rng.standard_normal((n, 3)).astype(np.float32)

    gc.positions = positions.reshape(-1)
    gc.colors = colors_sh.reshape(-1)
    gc.alphas = alphas
    gc.rotations = rots.reshape(-1)
    gc.scales = scales.reshape(-1)
    gc.sh = np.zeros(0, dtype=np.float32)

    tiles_dir = tmp_path / "0"
    tiles_dir.mkdir()
    glb_path = tiles_dir / "0.glb"
    save_gltf(gc, glb_path, GltfSaveOptions(spz_compression=True))

    tileset = {
        "asset": {"version": "1.1"},
        "extensionsUsed": ["3DTILES_content_gltf"],
        "extensions": {
            "3DTILES_content_gltf": {
                "extensionsRequired": [
                    "KHR_gaussian_splatting",
                    "KHR_gaussian_splatting_compression_spz_2",
                ],
                "extensionsUsed": [
                    "KHR_gaussian_splatting",
                    "KHR_gaussian_splatting_compression_spz_2",
                ],
            }
        },
        "geometricError": 10.0,
        "root": {
            "boundingVolume": {
                "box": [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0, 5],
            },
            "geometricError": 0.0,
            "refine": "REPLACE",
            "content": {"uri": "0/0.glb"},
        },
    }
    tileset_path = tmp_path / "tileset.json"
    tileset_path.write_text(json.dumps(tileset))
    return tileset_path


def test_load_local_tileset(tmp_path: Path) -> None:
    tileset_path = _build_local_tileset(tmp_path, n=30)

    tiles = load_tileset(tileset_path)
    assert len(tiles) == 1
    t = tiles[0]
    assert isinstance(t, Tile3DContent)
    assert t.cloud.num_points == 30
    assert t.refine == "REPLACE"
    assert t.geometric_error == 0.0
    # Identity transform when no 'transform' on tile
    np.testing.assert_array_almost_equal(t.transform.reshape(4, 4), np.eye(4))


def test_merge_local_tileset_identity(tmp_path: Path) -> None:
    """With identity transforms, merge should preserve positions exactly."""
    tileset_path = _build_local_tileset(tmp_path, n=25)
    tiles = load_tileset(tileset_path)
    merged = merge_tileset(tiles)

    assert merged.num_points == 25
    # Original tile cloud, since transform is identity
    orig = np.array(tiles[0].cloud.positions, dtype=np.float32).reshape(-1, 3)
    got = np.array(merged.positions, dtype=np.float32).reshape(-1, 3)
    np.testing.assert_allclose(got, orig, atol=1e-5)


def test_merge_applies_translation(tmp_path: Path) -> None:
    """Verify that tile ``transform`` translations are applied on merge."""
    tileset_path = _build_local_tileset(tmp_path, n=10)
    tileset = json.loads(tileset_path.read_text())
    # Column-major 4x4 with a translation of (100, 200, 300)
    tileset["root"]["transform"] = [
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
        100.0,
        200.0,
        300.0,
        1.0,
    ]
    tileset_path.write_text(json.dumps(tileset))

    tiles = load_tileset(tileset_path)
    assert len(tiles) == 1
    merged = merge_tileset(tiles)

    tile_pos = np.array(tiles[0].cloud.positions, dtype=np.float32).reshape(-1, 3)
    got = np.array(merged.positions, dtype=np.float32).reshape(-1, 3)
    np.testing.assert_allclose(got, tile_pos + np.array([100, 200, 300]), atol=1e-3)


# ── Cesium public tilesets (require network) ────────────────────────────────


@network
def test_load_cesium_sh_unit_cube() -> None:
    """Load the published ``sh_unit_cube`` GS tileset from CesiumGS/cesium."""
    tiles = load_tileset(_SH_UNIT_CUBE_URL)
    assert len(tiles) == 1

    t = tiles[0]
    assert t.cloud.num_points == 27  # 3x3x3 grid of gaussians
    # All positions lie within the declared bounding volume (±50 cube)
    pos = np.array(t.cloud.positions, dtype=np.float32).reshape(-1, 3)
    assert pos.min() >= -50.001 and pos.max() <= 50.001
    # Expects degree-3 spherical harmonics (15 coefficients x 3 channels per point)
    sh = np.array(t.cloud.sh, dtype=np.float32)
    assert sh.size == t.cloud.num_points * 15 * 3


@network
def test_load_cesium_tower_georeferenced() -> None:
    """Load the georeferenced ``tower`` GS tileset (~7 MB)."""
    tiles = load_tileset(_TOWER_URL)
    assert len(tiles) == 1

    t = tiles[0]
    assert t.cloud.num_points > 0
    # The tower tileset has a root transform -> non-identity
    assert not np.allclose(t.transform.reshape(4, 4), np.eye(4))

    # Merge translates positions into ECEF; magnitudes should jump into the
    # millions-of-metres range (Earth radius ~= 6.37e6 m).
    merged = merge_tileset(tiles)
    merged_pos = np.array(merged.positions, dtype=np.float32).reshape(-1, 3)
    assert np.linalg.norm(merged_pos, axis=1).mean() > 1.0e6


@network
def test_cesium_tileset_declares_khr_gaussian_splatting() -> None:
    """Sanity-check that the public tileset uses the expected glTF extensions."""
    import urllib.request

    with urllib.request.urlopen(_SH_UNIT_CUBE_URL) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    required = data["extensions"]["3DTILES_content_gltf"]["extensionsRequired"]
    assert "KHR_gaussian_splatting" in required
    assert "KHR_gaussian_splatting_compression_spz_2" in required


# ── structural GLB check on the published content ───────────────────────────


@network
def test_cesium_tile_content_is_valid_glb() -> None:
    """The tile content must be a GLB v2 with JSON + BIN chunks."""
    import urllib.request

    url = f"{_CESIUM_RAW}/sh_unit_cube/0/0.glb"
    with urllib.request.urlopen(url) as resp:
        buf = resp.read()

    assert buf[:4] == b"glTF"
    version, total = struct.unpack("<II", buf[4:12])
    assert version == 2
    assert total == len(buf)

    json_len, json_type = struct.unpack("<II", buf[12:20])
    assert json_type == 0x4E4F534A
    gltf = json.loads(buf[20 : 20 + json_len])

    assert "KHR_gaussian_splatting" in gltf["extensionsUsed"]


# ── multi-content tileset (camera + lidar) ─────────────────────────────────


def _make_lidar_cloud(rng: np.random.Generator, n: int) -> LidarGaussianCloud:
    """Create a small LidarGaussianCloud for testing."""
    positions = rng.uniform(-5.0, 5.0, (n, 3)).astype(np.float32)
    normals = rng.standard_normal((n, 3)).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    scales_2d = rng.uniform(0.01, 1.0, (n, 2)).astype(np.float32)
    rotations = rng.standard_normal((n, 4)).astype(np.float32)
    rotations /= np.linalg.norm(rotations, axis=1, keepdims=True)
    reflectance = rng.uniform(0.0, 1.0, (n,)).astype(np.float32)
    opacity = rng.uniform(0.0, 1.0, (n,)).astype(np.float32)

    return LidarGaussianCloud(
        num_points=n,
        positions=positions.reshape(-1),
        normals=normals.reshape(-1),
        scales_2d=scales_2d.reshape(-1),
        rotations=rotations.reshape(-1),
        reflectance=reflectance,
        opacity=opacity,
    )


def _build_multi_content_tileset(tmp_path: Path, n_camera: int = 15, n_lidar: int = 10) -> Path:
    """Write a 3D Tiles 1.1 tileset with both camera 3DGS and LiDAR 2DGS content."""
    rng = np.random.default_rng(42)
    sh_c0 = 0.2820947917738781

    # Camera 3DGS
    gc = GaussianCloud()
    positions = rng.uniform(-5.0, 5.0, (n_camera, 3)).astype(np.float32)
    rgb = rng.integers(0, 256, (n_camera, 3), dtype=np.uint8)
    colors_sh = ((rgb.astype(np.float32) / 255.0) - 0.5) / sh_c0
    alpha_u8 = rng.integers(1, 255, (n_camera,), dtype=np.uint8)
    alpha_01 = alpha_u8.astype(np.float64) / 255.0
    alphas = np.log(alpha_01 / (1 - alpha_01)).astype(np.float32)
    rots = rng.standard_normal((n_camera, 4)).astype(np.float32)
    rots /= np.linalg.norm(rots, axis=1, keepdims=True)
    scales = rng.standard_normal((n_camera, 3)).astype(np.float32)

    gc.positions = positions.reshape(-1)
    gc.colors = colors_sh.reshape(-1)
    gc.alphas = alphas
    gc.rotations = rots.reshape(-1)
    gc.scales = scales.reshape(-1)
    gc.sh = np.zeros(0, dtype=np.float32)

    # LiDAR 2DGS
    lidar = _make_lidar_cloud(rng, n_lidar)

    tiles_dir = tmp_path / "0"
    tiles_dir.mkdir()
    save_gltf(gc, tiles_dir / "camera.glb", GltfSaveOptions(spz_compression=True))
    save_lidar_gltf(lidar, tiles_dir / "lidar.glb")

    tileset = {
        "asset": {"version": "1.1"},
        "schema": {
            "classes": {
                "ContentType": {
                    "properties": {
                        "type": {"type": "STRING"},
                    }
                }
            }
        },
        "groups": [
            {"class": "ContentType", "properties": {"type": "camera_3dgs"}},
            {"class": "ContentType", "properties": {"type": "lidar_2dgs"}},
        ],
        "geometricError": 10.0,
        "root": {
            "boundingVolume": {
                "box": [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0, 5],
            },
            "geometricError": 0.0,
            "refine": "REPLACE",
            "contents": [
                {"uri": "0/camera.glb", "group": 0},
                {"uri": "0/lidar.glb", "group": 1},
            ],
        },
    }
    tileset_path = tmp_path / "tileset.json"
    tileset_path.write_text(json.dumps(tileset))
    return tileset_path


def test_load_multi_content_camera(tmp_path: Path) -> None:
    """Default layer returns only camera tiles from a multi-content tileset."""
    tileset_path = _build_multi_content_tileset(tmp_path, n_camera=15, n_lidar=10)

    tiles = load_tileset(tileset_path)
    assert len(tiles) == 1
    assert isinstance(tiles[0], Tile3DContent)
    assert tiles[0].cloud.num_points == 15


def test_load_multi_content_lidar(tmp_path: Path) -> None:
    """layer='lidar_2dgs' returns only LiDAR tiles."""
    tileset_path = _build_multi_content_tileset(tmp_path, n_camera=15, n_lidar=10)

    tiles = load_tileset(tileset_path, layer="lidar_2dgs")
    assert len(tiles) == 1
    assert isinstance(tiles[0], LidarTile3DContent)
    assert tiles[0].cloud.num_points == 10


def test_multi_content_tileset_structure(tmp_path: Path) -> None:
    """Verify the tileset.json structure is valid 3D Tiles 1.1."""
    tileset_path = _build_multi_content_tileset(tmp_path)
    tileset = json.loads(tileset_path.read_text())

    assert tileset["asset"]["version"] == "1.1"
    assert "schema" in tileset
    assert "groups" in tileset
    assert len(tileset["groups"]) == 2
    assert tileset["groups"][0]["properties"]["type"] == "camera_3dgs"
    assert tileset["groups"][1]["properties"]["type"] == "lidar_2dgs"

    root = tileset["root"]
    assert "contents" in root
    assert len(root["contents"]) == 2
    assert root["contents"][0]["group"] == 0
    assert root["contents"][1]["group"] == 1


def test_multi_content_lidar_data_integrity(tmp_path: Path) -> None:
    """Verify LiDAR data can be loaded correctly from a multi-content tileset."""
    tileset_path = _build_multi_content_tileset(tmp_path, n_lidar=20)

    tiles = load_tileset(tileset_path, layer="lidar_2dgs")
    assert len(tiles) == 1

    cloud = tiles[0].cloud
    assert cloud.num_points == 20
    assert len(cloud.positions) == 20 * 3
    assert len(cloud.normals) == 20 * 3
    assert len(cloud.scales_2d) == 20 * 2
    assert len(cloud.rotations) == 20 * 4
    assert len(cloud.reflectance) == 20
    assert len(cloud.opacity) == 20


def test_bounding_volume_box_populated(tmp_path: Path) -> None:
    """bounding_volume is populated as BoundingVolumeBox when boundingVolume.box exists."""
    tileset_path = _build_local_tileset(tmp_path, n=10)
    tiles = load_tileset(tileset_path)
    assert len(tiles) == 1
    t = tiles[0]
    assert t.bounding_volume is not None
    assert isinstance(t.bounding_volume, BoundingVolumeBox)
    np.testing.assert_array_equal(t.bounding_volume.center, [0, 0, 0])
    expected_half_axes = np.array([[5, 0, 0], [0, 5, 0], [0, 0, 5]], dtype=np.float64)
    np.testing.assert_array_equal(t.bounding_volume.half_axes, expected_half_axes)


def test_bounding_volume_region(tmp_path: Path) -> None:
    """bounding_volume is parsed as BoundingVolumeRegion."""
    tileset_path = _build_local_tileset(tmp_path, n=10)
    tileset = json.loads(tileset_path.read_text())
    tileset["root"]["boundingVolume"] = {
        "region": [-1.3197, 0.6988, -1.3196, 0.6989, 0.0, 100.0],
    }
    tileset_path.write_text(json.dumps(tileset))

    tiles = load_tileset(tileset_path)
    bv = tiles[0].bounding_volume
    assert isinstance(bv, BoundingVolumeRegion)
    assert bv.west == pytest.approx(-1.3197)
    assert bv.north == pytest.approx(0.6989)
    assert bv.min_height == pytest.approx(0.0)
    assert bv.max_height == pytest.approx(100.0)


def test_bounding_volume_sphere(tmp_path: Path) -> None:
    """bounding_volume is parsed as BoundingVolumeSphere."""
    tileset_path = _build_local_tileset(tmp_path, n=10)
    tileset = json.loads(tileset_path.read_text())
    tileset["root"]["boundingVolume"] = {
        "sphere": [1.0, 2.0, 3.0, 10.0],
    }
    tileset_path.write_text(json.dumps(tileset))

    tiles = load_tileset(tileset_path)
    bv = tiles[0].bounding_volume
    assert isinstance(bv, BoundingVolumeSphere)
    np.testing.assert_array_equal(bv.center, [1.0, 2.0, 3.0])
    assert bv.radius == pytest.approx(10.0)


def test_bounding_volume_none_when_absent(tmp_path: Path) -> None:
    """bounding_volume is None when the tile node has no boundingVolume."""
    tileset_path = _build_local_tileset(tmp_path, n=10)
    tileset = json.loads(tileset_path.read_text())
    del tileset["root"]["boundingVolume"]
    tileset_path.write_text(json.dumps(tileset))

    tiles = load_tileset(tileset_path)
    assert len(tiles) == 1
    assert tiles[0].bounding_volume is None


def test_bounding_volume_lidar(tmp_path: Path) -> None:
    """LidarTile3DContent also exposes bounding_volume."""
    tileset_path = _build_multi_content_tileset(tmp_path, n_lidar=10)
    tiles = load_tileset(tileset_path, layer="lidar_2dgs")
    assert len(tiles) == 1
    assert tiles[0].bounding_volume is not None
    assert isinstance(tiles[0].bounding_volume, BoundingVolumeBox)


def test_legacy_single_content_backward_compat(tmp_path: Path) -> None:
    """Legacy single-content tileset works with default layer."""
    tileset_path = _build_local_tileset(tmp_path, n=20)

    tiles = load_tileset(tileset_path)
    assert len(tiles) == 1
    assert tiles[0].cloud.num_points == 20

    # No LiDAR in a legacy tileset
    lidar = load_tileset(tileset_path, layer="lidar_2dgs")
    assert len(lidar) == 0
