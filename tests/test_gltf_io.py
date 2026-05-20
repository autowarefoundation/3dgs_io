from __future__ import annotations

import importlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest
import spz

_mod = importlib.import_module("3dgs_io")
GaussianCloud = spz.GaussianCloud
DatasetType = _mod.DatasetType
load_gltf = _mod.load_gltf
load_gltf_with_metadata = _mod.load_gltf_with_metadata
save_gltf = _mod.save_gltf
GltfSaveOptions = _mod.GltfSaveOptions

_SH_C0 = 0.2820947917738781
_EXT = "KHR_gaussian_splatting"
_SPZ_EXT = "KHR_gaussian_splatting_compression_spz_2"


def _make_gc(n: int = 10) -> GaussianCloud:
    """Create a GaussianCloud with valid pre-activation values."""
    rng = np.random.default_rng(42)
    gc = GaussianCloud()

    positions = rng.standard_normal((n, 3)).astype(np.float32)

    # SH DC coefficients (range roughly [-1.77, 1.77])
    rgb_u8 = rng.integers(0, 256, (n, 3), dtype=np.uint8)
    colors_sh = ((rgb_u8.astype(np.float32) / 255.0) - 0.5) / _SH_C0

    # Logit alphas (avoid extreme values for stable roundtrip)
    opacity_u8 = rng.integers(1, 255, (n,), dtype=np.uint8)
    opacity_01 = opacity_u8.astype(np.float64) / 255.0
    alphas_logit = np.log(opacity_01 / (1 - opacity_01)).astype(np.float32)

    rotations = rng.standard_normal((n, 4)).astype(np.float32)
    rotations /= np.linalg.norm(rotations, axis=1, keepdims=True)
    scales = rng.standard_normal((n, 3)).astype(np.float32)

    gc.positions = positions.reshape(-1)
    gc.colors = colors_sh.reshape(-1).astype(np.float32)
    gc.alphas = alphas_logit
    gc.rotations = rotations.reshape(-1)
    gc.scales = scales.reshape(-1)
    gc.sh = np.zeros(0, dtype=np.float32)
    return gc


# ── round-trip ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("ext", [".glb", ".gltf"])
def test_roundtrip(ext: str, tmp_path: Path) -> None:
    gc = _make_gc()
    path = tmp_path / f"test{ext}"
    save_gltf(gc, path)
    loaded = load_gltf(path)

    # positions, rotations, scales: float32 → float32, exact
    np.testing.assert_array_equal(
        np.array(loaded.positions, dtype=np.float32),
        np.array(gc.positions, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.array(loaded.rotations, dtype=np.float32),
        np.array(gc.rotations, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.array(loaded.scales, dtype=np.float32),
        np.array(gc.scales, dtype=np.float32),
    )
    # SH DC coefficients: float32 → float32, exact
    np.testing.assert_array_equal(
        np.array(loaded.colors, dtype=np.float32),
        np.array(gc.colors, dtype=np.float32),
    )
    # alphas go through sigmoid → uint8 → inverse_sigmoid (lossy)
    np.testing.assert_allclose(
        np.array(loaded.alphas, dtype=np.float32),
        np.array(gc.alphas, dtype=np.float32),
        atol=0.5,
    )


# ── SPZ compressed round-trip ─────────────────────────────────────────────


@pytest.mark.parametrize("ext", [".glb", ".gltf"])
def test_roundtrip_spz_compression(ext: str, tmp_path: Path) -> None:
    gc = _make_gc()
    path = tmp_path / f"test_spz{ext}"
    save_gltf(gc, path, GltfSaveOptions(spz_compression=True))
    loaded = load_gltf(path)

    n = gc.num_points
    assert loaded.num_points == n

    # SPZ compression is lossy (uint8 quantisation internally)
    np.testing.assert_allclose(
        np.array(loaded.positions, dtype=np.float32),
        np.array(gc.positions, dtype=np.float32),
        atol=0.1,
    )
    np.testing.assert_allclose(
        np.array(loaded.scales, dtype=np.float32),
        np.array(gc.scales, dtype=np.float32),
        atol=0.5,
    )
    np.testing.assert_allclose(
        np.array(loaded.alphas, dtype=np.float32),
        np.array(gc.alphas, dtype=np.float32),
        atol=0.5,
    )
    np.testing.assert_allclose(
        np.array(loaded.colors, dtype=np.float32),
        np.array(gc.colors, dtype=np.float32),
        atol=0.1,
    )


# ── glTF JSON structure ────────────────────────────────────────────────────


def test_glb_json_structure(tmp_path: Path) -> None:
    gc = _make_gc(5)
    path = tmp_path / "test.glb"
    save_gltf(gc, path)

    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"glTF"
        _ver, _total = struct.unpack("<II", f.read(8))
        json_len, json_type = struct.unpack("<II", f.read(8))
        json_bytes = f.read(json_len)
    gltf = json.loads(json_bytes)

    assert _EXT in gltf["extensionsUsed"]
    # extensionsRequired should NOT be present (graceful fallback)
    assert "extensionsRequired" not in gltf

    prim = gltf["meshes"][0]["primitives"][0]
    assert prim["mode"] == 0
    attrs = prim["attributes"]

    # Only standard attributes in the attributes dict
    assert "POSITION" in attrs
    assert "COLOR_0" in attrs
    assert len(attrs) == 2

    # No colon-prefixed keys in attributes (would break CesiumJS shaders)
    assert not any(":" in k for k in attrs)

    ext_props = prim["extensions"][_EXT]
    assert ext_props["kernel"] == "ellipse"
    assert ext_props["colorSpace"] == "srgb_rec709_display"

    # Gaussian data referenced from extension block as accessor indices
    assert isinstance(ext_props["rotation"], int)
    assert isinstance(ext_props["scale"], int)
    assert isinstance(ext_props["opacity"], int)
    assert isinstance(ext_props["sh"], list)
    assert len(ext_props["sh"]) >= 1  # at least DC

    assert len(gltf["accessors"]) == 6
    assert len(gltf["bufferViews"]) == 6

    pos_acc = gltf["accessors"][attrs["POSITION"]]
    assert "min" in pos_acc
    assert "max" in pos_acc

    opa_acc = gltf["accessors"][ext_props["opacity"]]
    assert opa_acc["type"] == "SCALAR"
    assert opa_acc["componentType"] == 5121
    assert opa_acc["normalized"] is True

    sh_acc = gltf["accessors"][ext_props["sh"][0]]
    assert sh_acc["type"] == "VEC3"
    assert sh_acc["componentType"] == 5126


def test_glb_spz_json_structure(tmp_path: Path) -> None:
    gc = _make_gc(5)
    path = tmp_path / "test_spz.glb"
    save_gltf(gc, path, GltfSaveOptions(spz_compression=True))

    with open(path, "rb") as f:
        f.read(4)
        _ver, _total = struct.unpack("<II", f.read(8))
        json_len, _ = struct.unpack("<II", f.read(8))
        json_bytes = f.read(json_len)
    gltf = json.loads(json_bytes)

    assert _EXT in gltf["extensionsUsed"]
    assert _SPZ_EXT in gltf["extensionsUsed"]
    assert _EXT in gltf["extensionsRequired"]
    assert _SPZ_EXT in gltf["extensionsRequired"]

    prim = gltf["meshes"][0]["primitives"][0]
    attrs = prim["attributes"]

    # Only standard attributes in the attributes dict
    assert "POSITION" in attrs
    assert "COLOR_0" in attrs
    assert len(attrs) == 2

    # No colon-prefixed keys in attributes
    assert not any(":" in k for k in attrs)

    # Gaussian data referenced from extension block
    ext_gs = prim["extensions"][_EXT]
    assert isinstance(ext_gs["scale"], int)
    assert isinstance(ext_gs["rotation"], int)
    spz_sub = ext_gs["extensions"][_SPZ_EXT]
    assert "bufferView" in spz_sub

    # Virtual accessors: POSITION, COLOR_0, SCALE, ROTATION (no SH for degree 0)
    assert len(gltf["accessors"]) == 4
    # Accessors must NOT have bufferView (CesiumJS requires SPZ as sole data source)
    for acc in gltf["accessors"]:
        assert "bufferView" not in acc, "SPZ accessors must be virtual (no bufferView)"
    # Single bufferView for SPZ blob
    assert len(gltf["bufferViews"]) == 1

    # SPZ bytes in the buffer must be gzip-wrapped (CesiumJS's @spz-loader/core
    # expects gzip-compressed SPZ, matching the .spz file format).
    with open(path, "rb") as f:
        f.seek(12)  # skip GLB header
        json_len, _ = struct.unpack("<II", f.read(8))
        f.seek(12 + 8 + json_len)  # skip JSON chunk
        _bin_len, _bin_type = struct.unpack("<II", f.read(8))
        bin_data = f.read(_bin_len)
    spz_bv = gltf["bufferViews"][spz_sub["bufferView"]]
    spz_offset = spz_bv.get("byteOffset", 0)
    spz_bytes = bin_data[spz_offset : spz_offset + spz_bv["byteLength"]]
    assert spz_bytes[:2] == b"\x1f\x8b", "SPZ data must be gzip-wrapped for CesiumJS"


# ── SH ↔ RGB conversion ───────────────────────────────────────────────────


def test_sh_rgb_roundtrip() -> None:
    """Verify SH DC ↔ RGB [0,1] conversion precision."""
    rgb_01 = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])
    sh = (rgb_01 - 0.5) / _SH_C0
    recovered = sh * _SH_C0 + 0.5
    np.testing.assert_allclose(recovered, rgb_01, atol=1e-6)


# ── buffer alignment ──────────────────────────────────────────────────────


@pytest.mark.parametrize("n", [1, 3, 7, 100])
def test_buffer_alignment(n: int, tmp_path: Path) -> None:
    gc = _make_gc(n)
    path = tmp_path / "test.glb"
    save_gltf(gc, path)

    with open(path, "rb") as f:
        f.read(4)
        _ver, _total = struct.unpack("<II", f.read(8))
        json_len, _ = struct.unpack("<II", f.read(8))
        json_bytes = f.read(json_len)
    gltf = json.loads(json_bytes)

    for i, bv in enumerate(gltf["bufferViews"]):
        acc = gltf["accessors"][i]
        if acc["componentType"] == 5126:
            offset = bv.get("byteOffset", 0)
            assert offset % 4 == 0, f"bufferView {i} offset {offset} not 4-byte aligned"


# ── backward compatibility with legacy format ─────────────────────────────


def test_load_legacy_format(tmp_path: Path) -> None:
    """Can load files with _ROTATION / _SCALE / COLOR_0-only (pre-spec format)."""
    n = 3
    rng = np.random.default_rng(99)
    positions = rng.standard_normal((n, 3)).astype(np.float32)
    colors_uint8 = rng.integers(0, 256, (n, 3), dtype=np.uint8)
    opacities_uint8 = rng.integers(1, 255, (n,), dtype=np.uint8)
    rotations = rng.standard_normal((n, 4)).astype(np.float32)
    scales = rng.standard_normal((n, 3)).astype(np.float32)

    # Build a legacy-format GLB manually
    pos_bytes = positions.tobytes()
    color_0 = np.empty((n, 4), dtype=np.uint8)
    color_0[:, :3] = colors_uint8
    color_0[:, 3] = opacities_uint8
    c0_bytes = color_0.tobytes()
    rot_bytes = rotations.tobytes()
    scl_bytes = scales.tobytes()

    offsets, lengths = [], []
    offset = 0
    for d in (pos_bytes, c0_bytes, rot_bytes, scl_bytes):
        offsets.append(offset)
        lengths.append(len(d))
        offset += len(d)
    buffer_data = pos_bytes + c0_bytes + rot_bytes + scl_bytes

    gltf_dict = {
        "asset": {"version": "2.0"},
        "extensionsUsed": [_EXT],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {
                        "mode": 0,
                        "attributes": {
                            "POSITION": 0,
                            "COLOR_0": 1,
                            "_ROTATION": 2,
                            "_SCALE": 3,
                        },
                        "extensions": {_EXT: {}},
                    }
                ]
            }
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": n, "type": "VEC3"},
            {
                "bufferView": 1,
                "componentType": 5121,
                "normalized": True,
                "count": n,
                "type": "VEC4",
            },
            {"bufferView": 2, "componentType": 5126, "count": n, "type": "VEC4"},
            {"bufferView": 3, "componentType": 5126, "count": n, "type": "VEC3"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": offsets[i], "byteLength": lengths[i]} for i in range(4)
        ],
        "buffers": [{"byteLength": len(buffer_data)}],
    }

    json_bytes = json.dumps(gltf_dict, separators=(",", ":")).encode("utf-8")
    json_pad = (4 - len(json_bytes) % 4) % 4
    json_bytes += b"\x20" * json_pad
    bin_pad = (4 - len(buffer_data) % 4) % 4
    bin_data = buffer_data + b"\x00" * bin_pad

    path = tmp_path / "legacy.glb"
    with open(path, "wb") as f:
        total = 12 + 8 + len(json_bytes) + 8 + len(bin_data)
        f.write(b"glTF")
        f.write(struct.pack("<II", 2, total))
        f.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        f.write(json_bytes)
        f.write(struct.pack("<II", len(bin_data), 0x004E4942))
        f.write(bin_data)

    loaded = load_gltf(path)
    loaded_n = loaded.num_points
    assert loaded_n == n

    np.testing.assert_array_equal(
        np.array(loaded.positions, dtype=np.float32).reshape(n, 3),
        positions,
    )
    np.testing.assert_array_equal(
        np.array(loaded.rotations, dtype=np.float32).reshape(n, 4),
        rotations,
    )
    np.testing.assert_array_equal(
        np.array(loaded.scales, dtype=np.float32).reshape(n, 3),
        scales,
    )

    # Colors: COLOR_0 uint8 → SH DC → COLOR_0 uint8 roundtrip
    loaded_sh = np.array(loaded.colors, dtype=np.float32).reshape(n, 3)
    loaded_rgb = np.clip((loaded_sh * _SH_C0 + 0.5) * 255 + 0.5, 0, 255).astype(np.uint8)
    np.testing.assert_array_equal(loaded_rgb, colors_uint8)

    # Opacity: COLOR_0 alpha uint8 → logit → sigmoid roundtrip
    loaded_opacity_01 = 1.0 / (1.0 + np.exp(-np.array(loaded.alphas, dtype=np.float64)))
    loaded_opacity_u8 = np.clip(loaded_opacity_01 * 255 + 0.5, 0, 255).astype(np.uint8)
    np.testing.assert_array_equal(loaded_opacity_u8, opacities_uint8)


# ── metadata ─────────────────────────────────────────────────────────────


_SAMPLE_METADATA = {
    "dataset_type": DatasetType.T4_DATASET,
    "dataset_id": "scene_001",
    "training_image_indices": [0, 1, 5, 7, 12, 45],
}


@pytest.mark.parametrize("ext", [".glb", ".gltf"])
def test_metadata_roundtrip(ext: str, tmp_path: Path) -> None:
    gc = _make_gc()
    path = tmp_path / f"meta{ext}"
    save_gltf(gc, path, GltfSaveOptions(metadata=_SAMPLE_METADATA))
    _, metadata = load_gltf_with_metadata(path)
    assert metadata == _SAMPLE_METADATA


@pytest.mark.parametrize("ext", [".glb", ".gltf"])
def test_metadata_roundtrip_spz(ext: str, tmp_path: Path) -> None:
    gc = _make_gc()
    path = tmp_path / f"meta_spz{ext}"
    save_gltf(gc, path, GltfSaveOptions(spz_compression=True, metadata=_SAMPLE_METADATA))
    _, metadata = load_gltf_with_metadata(path)
    assert metadata == _SAMPLE_METADATA


@pytest.mark.parametrize("ext", [".glb", ".gltf"])
def test_no_metadata_returns_none(ext: str, tmp_path: Path) -> None:
    gc = _make_gc()
    path = tmp_path / f"no_meta{ext}"
    save_gltf(gc, path)
    _, metadata = load_gltf_with_metadata(path)
    assert metadata is None


def test_metadata_in_glb_json(tmp_path: Path) -> None:
    gc = _make_gc(5)
    path = tmp_path / "meta.glb"
    save_gltf(gc, path, GltfSaveOptions(metadata=_SAMPLE_METADATA))

    with open(path, "rb") as f:
        f.read(4)
        _ver, _total = struct.unpack("<II", f.read(8))
        json_len, _ = struct.unpack("<II", f.read(8))
        json_bytes = f.read(json_len)
    gltf = json.loads(json_bytes)

    assert gltf["asset"]["extras"] == _SAMPLE_METADATA


def test_no_extras_key_when_metadata_is_none(tmp_path: Path) -> None:
    gc = _make_gc(5)
    path = tmp_path / "no_meta.glb"
    save_gltf(gc, path)

    with open(path, "rb") as f:
        f.read(4)
        _ver, _total = struct.unpack("<II", f.read(8))
        json_len, _ = struct.unpack("<II", f.read(8))
        json_bytes = f.read(json_len)
    gltf = json.loads(json_bytes)

    assert "extras" not in gltf["asset"]
