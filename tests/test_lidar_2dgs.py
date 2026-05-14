"""Tests for LiDAR 2D Gaussian Splatting I/O."""

from __future__ import annotations

import importlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest

_mod = importlib.import_module("3dgs_io")
LidarGaussianCloud = _mod.LidarGaussianCloud
save_lidar_gltf = _mod.save_lidar_gltf
load_lidar_gltf = _mod.load_lidar_gltf
load_lidar_gltf_with_metadata = _mod.load_lidar_gltf_with_metadata


def _make_lidar_cloud(n: int = 20) -> LidarGaussianCloud:
    """Create a LidarGaussianCloud with valid data."""
    rng = np.random.default_rng(123)

    positions = rng.uniform(-10.0, 10.0, (n, 3)).astype(np.float32)

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


# ── round-trip ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("ext", [".glb", ".gltf"])
def test_roundtrip(ext: str, tmp_path: Path) -> None:
    cloud = _make_lidar_cloud()
    path = tmp_path / f"lidar{ext}"
    save_lidar_gltf(cloud, path)
    loaded = load_lidar_gltf(path)

    assert loaded.num_points == cloud.num_points

    np.testing.assert_array_equal(
        np.array(loaded.positions, dtype=np.float32),
        np.array(cloud.positions, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.array(loaded.normals, dtype=np.float32),
        np.array(cloud.normals, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.array(loaded.scales_2d, dtype=np.float32),
        np.array(cloud.scales_2d, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.array(loaded.rotations, dtype=np.float32),
        np.array(cloud.rotations, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.array(loaded.reflectance, dtype=np.float32),
        np.array(cloud.reflectance, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.array(loaded.opacity, dtype=np.float32),
        np.array(cloud.opacity, dtype=np.float32),
    )


# ── rt_properties round-trip ────────────────────────────────────────────────


@pytest.mark.parametrize("ext", [".glb", ".gltf"])
def test_rt_properties_roundtrip(ext: str, tmp_path: Path) -> None:
    rng = np.random.default_rng(456)
    cloud = _make_lidar_cloud(15)
    cloud.rt_properties = {
        "absorption": rng.uniform(0.0, 1.0, (15,)).astype(np.float32),
        "scatter": rng.uniform(0.0, 1.0, (15, 3)).astype(np.float32),
    }

    path = tmp_path / f"lidar_rt{ext}"
    save_lidar_gltf(cloud, path)
    loaded = load_lidar_gltf(path)

    assert "absorption" in loaded.rt_properties
    assert "scatter" in loaded.rt_properties

    np.testing.assert_array_equal(
        loaded.rt_properties["absorption"],
        cloud.rt_properties["absorption"],
    )
    np.testing.assert_array_equal(
        loaded.rt_properties["scatter"].reshape(15, 3),
        cloud.rt_properties["scatter"].reshape(15, 3),
    )


# ── glTF JSON structure ──────────────────────────────────────────────────────


def test_glb_json_structure(tmp_path: Path) -> None:
    cloud = _make_lidar_cloud(5)
    path = tmp_path / "lidar.glb"
    save_lidar_gltf(cloud, path)

    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"glTF"
        _ver, _total = struct.unpack("<II", f.read(8))
        json_len, json_type = struct.unpack("<II", f.read(8))
        json_bytes = f.read(json_len)
    gltf = json.loads(json_bytes)

    # No KHR_gaussian_splatting extension
    assert "extensionsUsed" not in gltf
    assert "extensionsRequired" not in gltf

    prim = gltf["meshes"][0]["primitives"][0]
    assert prim["mode"] == 0
    attrs = prim["attributes"]

    assert "POSITION" in attrs
    assert "NORMAL" in attrs
    assert "_SCALE_2D" in attrs
    assert "_ROTATION" in attrs
    assert "_REFLECTANCE" in attrs
    assert "_OPACITY" in attrs

    # No KHR_gaussian_splatting extension on the primitive
    assert "extensions" not in prim

    # Check accessor types
    pos_acc = gltf["accessors"][attrs["POSITION"]]
    assert pos_acc["type"] == "VEC3"
    assert pos_acc["componentType"] == 5126
    assert "min" in pos_acc
    assert "max" in pos_acc

    normal_acc = gltf["accessors"][attrs["NORMAL"]]
    assert normal_acc["type"] == "VEC3"

    scale_acc = gltf["accessors"][attrs["_SCALE_2D"]]
    assert scale_acc["type"] == "VEC2"

    rot_acc = gltf["accessors"][attrs["_ROTATION"]]
    assert rot_acc["type"] == "VEC4"

    ref_acc = gltf["accessors"][attrs["_REFLECTANCE"]]
    assert ref_acc["type"] == "SCALAR"

    opa_acc = gltf["accessors"][attrs["_OPACITY"]]
    assert opa_acc["type"] == "SCALAR"


# ── buffer alignment ──────────────────────────────────────────────────────


@pytest.mark.parametrize("n", [1, 3, 7, 50])
def test_buffer_alignment(n: int, tmp_path: Path) -> None:
    cloud = _make_lidar_cloud(n)
    path = tmp_path / "lidar.glb"
    save_lidar_gltf(cloud, path)

    with open(path, "rb") as f:
        f.read(4)
        _ver, _total = struct.unpack("<II", f.read(8))
        json_len, _ = struct.unpack("<II", f.read(8))
        json_bytes = f.read(json_len)
    gltf = json.loads(json_bytes)

    for i, bv in enumerate(gltf["bufferViews"]):
        offset = bv.get("byteOffset", 0)
        assert offset % 4 == 0, f"bufferView {i} offset {offset} not 4-byte aligned"


# ── metadata ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("ext", [".glb", ".gltf"])
def test_metadata_roundtrip(ext: str, tmp_path: Path) -> None:
    cloud = _make_lidar_cloud()
    meta = {"sensor_type": "lidar", "wavelength_nm": 905}
    path = tmp_path / f"lidar_meta{ext}"
    save_lidar_gltf(cloud, path, metadata=meta)
    _, loaded_meta = load_lidar_gltf_with_metadata(path)
    assert loaded_meta == meta


@pytest.mark.parametrize("ext", [".glb", ".gltf"])
def test_no_metadata_returns_none(ext: str, tmp_path: Path) -> None:
    cloud = _make_lidar_cloud()
    path = tmp_path / f"lidar_no_meta{ext}"
    save_lidar_gltf(cloud, path)
    _, loaded_meta = load_lidar_gltf_with_metadata(path)
    assert loaded_meta is None


# ── error handling ─────────────────────────────────────────────────────────


def test_save_empty_raises() -> None:
    cloud = LidarGaussianCloud()
    with pytest.raises(ValueError, match="empty"):
        save_lidar_gltf(cloud, "/tmp/empty.glb")


def test_load_non_lidar_glb_raises(tmp_path: Path) -> None:
    """Loading a camera 3DGS file as LiDAR should fail."""
    import spz

    _mod2 = importlib.import_module("3dgs_io")
    save_gltf = _mod2.save_gltf

    gc = spz.GaussianCloud()
    n = 5
    rng = np.random.default_rng(7)
    gc.positions = rng.standard_normal((n, 3)).astype(np.float32).reshape(-1)
    gc.colors = rng.standard_normal((n, 3)).astype(np.float32).reshape(-1)
    gc.alphas = rng.standard_normal((n,)).astype(np.float32)
    gc.rotations = rng.standard_normal((n, 4)).astype(np.float32).reshape(-1)
    gc.scales = rng.standard_normal((n, 3)).astype(np.float32).reshape(-1)
    gc.sh = np.zeros(0, dtype=np.float32)

    path = tmp_path / "camera.glb"
    save_gltf(gc, path)

    with pytest.raises(ValueError, match="No LiDAR"):
        load_lidar_gltf(path)
