"""Tests for per-Gaussian extension attributes (EXT_gaussian_lidar)."""

from __future__ import annotations

import importlib
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest
import spz

_mod = importlib.import_module("3dgs_io")

GaussianCloud = spz.GaussianCloud
GltfSaveOptions = _mod.GltfSaveOptions
TilesetSaveOptions = _mod.TilesetSaveOptions
SceneUsdzOptions = _mod.SceneUsdzOptions
save_gltf = _mod.save_gltf
load_gltf_with_metadata = _mod.load_gltf_with_metadata
save_tileset = _mod.save_tileset
save_scene_usdz = _mod.save_scene_usdz
encode_lidar_sidecar = _mod.encode_lidar_sidecar
decode_lidar_sidecar = _mod.decode_lidar_sidecar
EXT_GAUSSIAN_LIDAR_NAME = _mod.EXT_GAUSSIAN_LIDAR_NAME

LIDAR_INTENSITY = "lidar_intensity_raw"
LIDAR_RAYDROP = "lidar_raydrop_logit"


def _make_cloud(rng: np.random.Generator, n: int, *, spread: float = 5.0) -> GaussianCloud:
    gc = GaussianCloud()
    gc.positions = rng.uniform(-spread, spread, (n, 3)).astype(np.float32).reshape(-1)
    gc.colors = rng.uniform(-1, 1, (n, 3)).astype(np.float32).reshape(-1)
    gc.alphas = rng.uniform(-3, 3, (n,)).astype(np.float32)
    rots = rng.standard_normal((n, 4)).astype(np.float32)
    rots /= np.linalg.norm(rots, axis=1, keepdims=True)
    gc.rotations = rots.reshape(-1)
    gc.scales = rng.standard_normal((n, 3)).astype(np.float32).reshape(-1)
    gc.sh = np.zeros(0, dtype=np.float32)
    return gc


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


# ── Sidecar encode/decode ────────────────────────────────────────────────────


def test_lidar_sidecar_round_trip() -> None:
    rng = np.random.default_rng(0)
    n = 1024
    intensity = rng.standard_normal(n).astype(np.float32) * 2.0  # wide range
    raydrop = rng.standard_normal(n).astype(np.float32) * 2.0
    payload = encode_lidar_sidecar({LIDAR_INTENSITY: intensity, LIDAR_RAYDROP: raydrop}, count=n)

    # 16-byte header + 2 bytes/point body
    assert len(payload) == 16 + n * 2

    out = decode_lidar_sidecar(payload)
    assert set(out.keys()) == {LIDAR_INTENSITY, LIDAR_RAYDROP}
    assert out[LIDAR_INTENSITY].shape == (n,)
    assert out[LIDAR_RAYDROP].shape == (n,)

    # u8 sigmoid round trip preserves sigmoid space to within 1/255 of the original.
    for key, src in [(LIDAR_INTENSITY, intensity), (LIDAR_RAYDROP, raydrop)]:
        src_sig = _sigmoid(src)
        got_sig = _sigmoid(out[key])
        assert np.max(np.abs(src_sig - got_sig)) < 2.0 / 255.0


def test_lidar_sidecar_bad_magic() -> None:
    with pytest.raises(ValueError, match="bad sidecar magic"):
        decode_lidar_sidecar(b"XXXX" + b"\x00" * 32)


def test_lidar_sidecar_too_short() -> None:
    with pytest.raises(ValueError, match="sidecar too short"):
        decode_lidar_sidecar(b"L1DR")


# ── GLB round trip ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("spz_compression", [False, True])
@pytest.mark.parametrize("ext", [".glb", ".gltf"])
def test_glb_ext_attributes_round_trip(spz_compression: bool, ext: str, tmp_path: Path) -> None:
    if spz_compression and ext == ".gltf":
        # .gltf needs external .bin; the SPZ writer is GLB-oriented in this codebase
        pytest.skip(".gltf + SPZ not exercised here")

    rng = np.random.default_rng(0)
    n = 50
    gc = _make_cloud(rng, n)
    intensity = rng.standard_normal(n).astype(np.float32)
    raydrop = rng.standard_normal(n).astype(np.float32)

    path = tmp_path / f"test{ext}"
    save_gltf(
        gc,
        path,
        GltfSaveOptions(spz_compression=spz_compression),
        ext_attributes={LIDAR_INTENSITY: intensity, LIDAR_RAYDROP: raydrop},
    )

    _gc2, _meta, ext_attrs = load_gltf_with_metadata(path)
    assert set(ext_attrs.keys()) == {LIDAR_INTENSITY, LIDAR_RAYDROP}
    assert ext_attrs[LIDAR_INTENSITY].shape == (n,)

    # Sigmoid-space round trip should be within 2/255 (one u8 step).
    src_sig = _sigmoid(intensity)
    got_sig = _sigmoid(ext_attrs[LIDAR_INTENSITY])
    assert np.max(np.abs(src_sig - got_sig)) < 2.0 / 255.0


def test_load_gltf_without_ext_returns_empty_dict(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    gc = _make_cloud(rng, 20)
    path = tmp_path / "no_ext.glb"
    save_gltf(gc, path)
    _gc, _meta, ext_attrs = load_gltf_with_metadata(path)
    assert ext_attrs == {}


# ── Tileset round trip ──────────────────────────────────────────────────────


def test_save_tileset_writes_ext_attributes_to_each_chunk(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 200
    gc = _make_cloud(rng, n, spread=5.0)
    intensity = rng.standard_normal(n).astype(np.float32)
    raydrop = rng.standard_normal(n).astype(np.float32)

    # Track positions and ext together so we can verify index alignment after re-chunking.
    positions = np.asarray(gc.positions, dtype=np.float32).reshape(n, 3)

    save_tileset(
        gc,
        tmp_path,
        TilesetSaveOptions(chunk_size=2.5, save_options=GltfSaveOptions(spz_compression=True)),
        ext_attributes={LIDAR_INTENSITY: intensity, LIDAR_RAYDROP: raydrop},
    )

    tileset = json.loads((tmp_path / "tileset.json").read_text())
    assert (
        EXT_GAUSSIAN_LIDAR_NAME in tileset["extensions"]["3DTILES_content_gltf"]["extensionsUsed"]
    )

    # SPZ slightly perturbs positions, so use nearest-neighbour matching to recover
    # the original source index for each chunk point and verify the (pos, ext) pair.
    glbs = sorted(tmp_path.glob("chunk_*.glb"))
    total = 0
    for glb in glbs:
        gc2, _, ext_attrs = load_gltf_with_metadata(glb)
        assert LIDAR_INTENSITY in ext_attrs
        chunk_pos = np.asarray(gc2.positions, dtype=np.float32).reshape(-1, 3)
        for j, p in enumerate(chunk_pos):
            d2 = np.sum((positions - p) ** 2, axis=1)
            i = int(np.argmin(d2))
            # Sigmoid-space comparison to compensate for u8 quantization.
            assert (
                abs(_sigmoid(intensity[i]) - _sigmoid(ext_attrs[LIDAR_INTENSITY][j])) < 2.0 / 255.0
            )
            assert abs(_sigmoid(raydrop[i]) - _sigmoid(ext_attrs[LIDAR_RAYDROP][j])) < 2.0 / 255.0
        total += gc2.num_points
    assert total == n


def test_save_tileset_without_ext_omits_extension(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    gc = _make_cloud(rng, 50)
    save_tileset(
        gc,
        tmp_path,
        TilesetSaveOptions(save_options=GltfSaveOptions(spz_compression=True)),
    )
    tileset = json.loads((tmp_path / "tileset.json").read_text())
    used = tileset["extensions"]["3DTILES_content_gltf"]["extensionsUsed"]
    assert EXT_GAUSSIAN_LIDAR_NAME not in used


# ── USDZ round trip ─────────────────────────────────────────────────────────


def test_usdz_round_trip_writes_sidecars(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 500
    gc = _make_cloud(rng, n, spread=5.0)
    intensity = rng.standard_normal(n).astype(np.float32)
    raydrop = rng.standard_normal(n).astype(np.float32)

    ts_dir = tmp_path / "tileset"
    save_tileset(
        gc,
        ts_dir,
        TilesetSaveOptions(chunk_size=2.5, save_options=GltfSaveOptions(spz_compression=True)),
        ext_attributes={LIDAR_INTENSITY: intensity, LIDAR_RAYDROP: raydrop},
    )

    usdz_path = tmp_path / "scene.usdz"
    # Force multiple final chunks too: smaller scene chunk_size.
    result = save_scene_usdz(
        ts_dir / "tileset.json",
        usdz_path,
        options=SceneUsdzOptions(chunk_size=3.0),
    )
    assert result.n_gaussians == n
    assert result.n_chunks >= 1

    with zipfile.ZipFile(usdz_path) as zf:
        names = zf.namelist()
        spz_files = sorted(n for n in names if n.endswith(".spz"))
        lidar_files = sorted(n for n in names if n.endswith(".lidar"))
        assert len(spz_files) == result.n_chunks
        assert len(lidar_files) == result.n_chunks

        tileset = json.loads(zf.read("tileset.json"))
        assert EXT_GAUSSIAN_LIDAR_NAME in tileset["extensionsUsed"]
        for child in tileset["root"]["children"]:
            assert EXT_GAUSSIAN_LIDAR_NAME in child["content"]["extensions"]

        scene = json.loads(zf.read("scene.json"))
        assert "ext_attributes" in scene["gaussians"]
        assert scene["gaussians"]["ext_attributes"]["extension"] == EXT_GAUSSIAN_LIDAR_NAME

        # Sidecar count matches SPZ point count for every chunk
        for spz_name, lidar_name in zip(spz_files, lidar_files, strict=True):
            ext_data = decode_lidar_sidecar(zf.read(lidar_name))
            assert LIDAR_INTENSITY in ext_data
            assert LIDAR_RAYDROP in ext_data
            # Decoded length should match the parsed SPZ point count.
            tmpdir = tmp_path / "_unzip"
            tmpdir.mkdir(exist_ok=True)
            target = tmpdir / os.path.basename(spz_name)
            target.write_bytes(zf.read(spz_name))
            loaded = spz.load_spz(str(target), spz.UnpackOptions())
            assert ext_data[LIDAR_INTENSITY].shape == (loaded.num_points,)


def test_usdz_index_alignment_preserved(tmp_path: Path) -> None:
    """attr[i] ↔ gaussian[i] holds after filter + chunking — verified by per-position lookup."""
    rng = np.random.default_rng(1)
    n = 300
    gc = _make_cloud(rng, n, spread=4.0)
    intensity = rng.standard_normal(n).astype(np.float32)
    raydrop = rng.standard_normal(n).astype(np.float32)

    src_positions = np.asarray(gc.positions, dtype=np.float32).reshape(n, 3)
    rub_to_enu = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    src_positions = src_positions @ rub_to_enu.T

    ts_dir = tmp_path / "tileset"
    save_tileset(
        gc,
        ts_dir,
        TilesetSaveOptions(chunk_size=2.0, save_options=GltfSaveOptions(spz_compression=True)),
        ext_attributes={LIDAR_INTENSITY: intensity, LIDAR_RAYDROP: raydrop},
    )

    usdz_path = tmp_path / "scene.usdz"
    save_scene_usdz(
        ts_dir / "tileset.json",
        usdz_path,
        options=SceneUsdzOptions(chunk_size=3.0),
    )

    matched = 0
    with zipfile.ZipFile(usdz_path) as zf:
        spz_files = sorted(n for n in zf.namelist() if n.endswith(".spz"))
        lidar_files = sorted(n for n in zf.namelist() if n.endswith(".lidar"))
        unzip_dir = tmp_path / "_unzip2"
        unzip_dir.mkdir(exist_ok=True)
        for spz_name, lidar_name in zip(spz_files, lidar_files, strict=True):
            sp = unzip_dir / os.path.basename(spz_name)
            sp.write_bytes(zf.read(spz_name))
            loaded = spz.load_spz(str(sp), spz.UnpackOptions())
            pos = np.asarray(loaded.positions, dtype=np.float32).reshape(-1, 3)
            ext = decode_lidar_sidecar(zf.read(lidar_name))
            for j in range(loaded.num_points):
                d2 = np.sum((src_positions - pos[j]) ** 2, axis=1)
                i = int(np.argmin(d2))
                src_sig = _sigmoid(intensity[i])
                got_sig = _sigmoid(ext[LIDAR_INTENSITY][j])
                assert abs(src_sig - got_sig) < 2.0 / 255.0
                matched += 1
    assert matched == n


def test_usdz_without_ext_attributes_unchanged(tmp_path: Path) -> None:
    """ext_attributes=None ⇒ no sidecars, no extension entries."""
    rng = np.random.default_rng(0)
    gc = _make_cloud(rng, 200, spread=3.0)

    ts_dir = tmp_path / "tileset"
    save_tileset(
        gc,
        ts_dir,
        TilesetSaveOptions(chunk_size=2.0, save_options=GltfSaveOptions(spz_compression=True)),
    )

    usdz_path = tmp_path / "scene.usdz"
    save_scene_usdz(ts_dir / "tileset.json", usdz_path)

    with zipfile.ZipFile(usdz_path) as zf:
        names = zf.namelist()
        assert not any(n.endswith(".lidar") for n in names)
        tileset = json.loads(zf.read("tileset.json"))
        assert EXT_GAUSSIAN_LIDAR_NAME not in tileset["extensionsUsed"]
        scene = json.loads(zf.read("scene.json"))
        assert "ext_attributes" not in scene["gaussians"]


def test_save_tileset_rejects_ext_for_list_source(tmp_path: Path) -> None:
    """ext_attributes is only valid for GaussianCloud sources."""
    rng = np.random.default_rng(0)
    Tile3DContent = _mod.Tile3DContent
    gc = _make_cloud(rng, 10)
    tile = Tile3DContent(
        cloud=gc,
        transform=np.eye(4, dtype=np.float64).reshape(-1),
        content_uri="dummy.glb",
    )
    with pytest.raises(ValueError, match="ext_attributes is only supported"):
        save_tileset(
            [tile],
            tmp_path,
            ext_attributes={LIDAR_INTENSITY: np.zeros(10, dtype=np.float32)},
        )


def test_save_tileset_ext_attribute_length_validation(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 50
    gc = _make_cloud(rng, n)
    with pytest.raises(ValueError, match="expected 50"):
        save_tileset(
            gc,
            tmp_path,
            ext_attributes={LIDAR_INTENSITY: np.zeros(n + 1, dtype=np.float32)},
        )
