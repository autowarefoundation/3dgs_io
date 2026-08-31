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
raydrop_sh_coefs = _mod.raydrop_sh_coefs
raydrop_sh_degree_from_coefs = _mod.raydrop_sh_degree_from_coefs
EXT_GAUSSIAN_LIDAR_NAME = _mod.EXT_GAUSSIAN_LIDAR_NAME

LIDAR_INTENSITY = "lidar_intensity_raw"
LIDAR_RAYDROP = "lidar_raydrop_logit"
LIDAR_MASK = "lidar_mask"
RAYDROP_SH = "raydrop_sh"


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


def test_lidar_sidecar_omitting_mask_is_byte_identical() -> None:
    """Encoding without a mask still writes exactly 2 channels (no behaviour change)."""
    rng = np.random.default_rng(1)
    n = 256
    intensity = rng.standard_normal(n).astype(np.float32)
    raydrop = rng.standard_normal(n).astype(np.float32)

    payload = encode_lidar_sidecar({LIDAR_INTENSITY: intensity, LIDAR_RAYDROP: raydrop}, count=n)

    # 16-byte header + 2 bytes/point body — identical to the pre-mask format.
    assert len(payload) == 16 + n * 2
    out = decode_lidar_sidecar(payload)
    assert set(out.keys()) == {LIDAR_INTENSITY, LIDAR_RAYDROP}
    assert LIDAR_MASK not in out


def test_lidar_sidecar_mask_round_trip() -> None:
    """A 3-channel sidecar round-trips intensity, raydrop AND the mask exactly."""
    rng = np.random.default_rng(2)
    n = 512
    intensity = rng.standard_normal(n).astype(np.float32)
    raydrop = rng.standard_normal(n).astype(np.float32)
    mask = rng.integers(0, 2, size=n).astype(np.float32)  # exact {0, 1}

    payload = encode_lidar_sidecar(
        {LIDAR_INTENSITY: intensity, LIDAR_RAYDROP: raydrop, LIDAR_MASK: mask},
        count=n,
    )

    # 16-byte header + 3 bytes/point body.
    assert len(payload) == 16 + n * 3

    out = decode_lidar_sidecar(payload)
    assert set(out.keys()) == {LIDAR_INTENSITY, LIDAR_RAYDROP, LIDAR_MASK}
    assert out[LIDAR_MASK].shape == (n,)
    # u8_linear over [0, 1] preserves {0, 1} exactly as {0.0, 1.0}.
    np.testing.assert_array_equal(out[LIDAR_MASK], mask)
    assert set(np.unique(out[LIDAR_MASK]).tolist()) <= {0.0, 1.0}


def test_lidar_sidecar_all_ones_mask_round_trip() -> None:
    """An explicit all-ones mask (all participate) round-trips to 1.0."""
    rng = np.random.default_rng(3)
    n = 64
    intensity = rng.standard_normal(n).astype(np.float32)
    raydrop = rng.standard_normal(n).astype(np.float32)
    mask = np.ones(n, dtype=np.float32)

    payload = encode_lidar_sidecar(
        {LIDAR_INTENSITY: intensity, LIDAR_RAYDROP: raydrop, LIDAR_MASK: mask},
        count=n,
    )
    out = decode_lidar_sidecar(payload)
    np.testing.assert_array_equal(out[LIDAR_MASK], np.ones(n, dtype=np.float32))


def test_old_reader_ignores_mask_channel() -> None:
    """A new 3-channel sidecar decoded against only the first 2 specs is forward compatible.

    Simulates an old reader (2 specs known) reading a new 3-channel sidecar: the
    body-size check passes and only intensity/raydrop are returned.
    """
    import importlib

    ext_mod = importlib.import_module("3dgs_io.ext_attributes")
    rng = np.random.default_rng(4)
    n = 100
    intensity = rng.standard_normal(n).astype(np.float32)
    raydrop = rng.standard_normal(n).astype(np.float32)
    mask = rng.integers(0, 2, size=n).astype(np.float32)

    payload = encode_lidar_sidecar(
        {LIDAR_INTENSITY: intensity, LIDAR_RAYDROP: raydrop, LIDAR_MASK: mask},
        count=n,
    )
    assert len(payload) == 16 + n * 3  # a genuine 3-channel sidecar

    # Temporarily restrict the known specs to the first two (old reader).
    original_specs = ext_mod.DEFAULT_LIDAR_SPECS
    try:
        ext_mod.DEFAULT_LIDAR_SPECS = original_specs[:2]
        out = ext_mod.decode_lidar_sidecar(payload)
    finally:
        ext_mod.DEFAULT_LIDAR_SPECS = original_specs

    assert set(out.keys()) == {LIDAR_INTENSITY, LIDAR_RAYDROP}


def test_lidar_mask_key_exported() -> None:
    assert _mod.LIDAR_MASK_KEY == LIDAR_MASK
    assert _mod.LIDAR_INTENSITY_KEY == LIDAR_INTENSITY
    assert _mod.LIDAR_RAYDROP_KEY == LIDAR_RAYDROP


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


# ── raydrop_sh (view-dependent SH raydrop) ───────────────────────────────────


def test_raydrop_sh_coefs_helpers() -> None:
    # (deg+1)^2 - 1 excludes the DC term.
    assert raydrop_sh_coefs(0) == 0
    assert raydrop_sh_coefs(1) == 3
    assert raydrop_sh_coefs(2) == 8
    assert raydrop_sh_coefs(3) == 15
    for deg in range(0, 6):
        assert raydrop_sh_degree_from_coefs(raydrop_sh_coefs(deg)) == deg
    with pytest.raises(ValueError, match="not \\(deg\\+1\\)"):
        raydrop_sh_degree_from_coefs(5)  # 5 + 1 = 6 is not a perfect square


def test_sidecar_scalar_only_is_version_1() -> None:
    """Without raydrop_sh the sidecar stays byte-identical version 1."""
    rng = np.random.default_rng(10)
    n = 128
    intensity = rng.standard_normal(n).astype(np.float32)
    raydrop = rng.standard_normal(n).astype(np.float32)
    payload = encode_lidar_sidecar({LIDAR_INTENSITY: intensity, LIDAR_RAYDROP: raydrop}, count=n)
    assert len(payload) == 16 + n * 2
    # version field (bytes 4..7) is 1.
    assert int.from_bytes(payload[4:8], "little") == 1
    out = decode_lidar_sidecar(payload)
    assert RAYDROP_SH not in out


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_sidecar_raydrop_sh_round_trip(degree: int) -> None:
    rng = np.random.default_rng(11)
    n = 256
    coefs = raydrop_sh_coefs(degree)
    intensity = rng.standard_normal(n).astype(np.float32)
    raydrop = rng.standard_normal(n).astype(np.float32)
    sh = rng.standard_normal((n, coefs)).astype(np.float32)

    payload = encode_lidar_sidecar(
        {LIDAR_INTENSITY: intensity, LIDAR_RAYDROP: raydrop, RAYDROP_SH: sh}, count=n
    )
    # version 2, and body = 16-byte header + 2 uint8 channels + 8-byte SH header + float16 block.
    assert int.from_bytes(payload[4:8], "little") == 2
    assert len(payload) == 16 + n * 2 + 8 + n * coefs * 2

    out = decode_lidar_sidecar(payload)
    assert set(out.keys()) == {LIDAR_INTENSITY, LIDAR_RAYDROP, RAYDROP_SH}
    assert out[RAYDROP_SH].shape == (n, coefs)
    assert raydrop_sh_degree_from_coefs(out[RAYDROP_SH].shape[1]) == degree
    # float16 half-precision round trip: exact once the source is cast to float16.
    np.testing.assert_array_equal(out[RAYDROP_SH], sh.astype(np.float16).astype(np.float32))
    # And close to the original float32 within half-precision resolution.
    assert np.max(np.abs(out[RAYDROP_SH] - sh)) < 1e-2


def test_sidecar_raydrop_sh_bad_coefs() -> None:
    rng = np.random.default_rng(14)
    n = 16
    sh = rng.standard_normal((n, 5)).astype(np.float32)  # 5 is not (deg+1)^2 - 1
    with pytest.raises(ValueError, match="not \\(deg\\+1\\)"):
        encode_lidar_sidecar(
            {
                LIDAR_INTENSITY: np.zeros(n, np.float32),
                LIDAR_RAYDROP: np.zeros(n, np.float32),
                RAYDROP_SH: sh,
            },
            count=n,
        )


def test_unsupported_sidecar_version_rejected() -> None:
    import struct

    # Craft a version-3 header (unknown) with a plausible body.
    header = struct.pack("<4sIII", b"L1DR", 3, 4, 2)
    with pytest.raises(ValueError, match="unsupported sidecar version 3"):
        decode_lidar_sidecar(header + b"\x00" * (4 * 2))


@pytest.mark.parametrize("spz_compression", [False, True])
def test_glb_raydrop_sh_round_trip(spz_compression: bool, tmp_path: Path) -> None:
    rng = np.random.default_rng(20)
    n = 60
    gc = _make_cloud(rng, n)
    intensity = rng.standard_normal(n).astype(np.float32)
    raydrop = rng.standard_normal(n).astype(np.float32)
    sh = rng.standard_normal((n, 8)).astype(np.float32)  # degree 2

    path = tmp_path / "sh.glb"
    save_gltf(
        gc,
        path,
        GltfSaveOptions(spz_compression=spz_compression),
        ext_attributes={LIDAR_INTENSITY: intensity, LIDAR_RAYDROP: raydrop, RAYDROP_SH: sh},
    )

    _gc2, _meta, ext_attrs = load_gltf_with_metadata(path)
    assert set(ext_attrs.keys()) == {LIDAR_INTENSITY, LIDAR_RAYDROP, RAYDROP_SH}
    assert ext_attrs[RAYDROP_SH].shape == (n, 8)
    # glTF carries raydrop_sh losslessly as float32.
    np.testing.assert_allclose(ext_attrs[RAYDROP_SH], sh, rtol=0, atol=0)


def test_usdz_round_trip_writes_raydrop_sh(tmp_path: Path) -> None:
    rng = np.random.default_rng(21)
    n = 500
    gc = _make_cloud(rng, n, spread=5.0)
    intensity = rng.standard_normal(n).astype(np.float32)
    raydrop = rng.standard_normal(n).astype(np.float32)
    sh = rng.standard_normal((n, 8)).astype(np.float32)  # degree 2

    ts_dir = tmp_path / "tileset"
    save_tileset(
        gc,
        ts_dir,
        TilesetSaveOptions(chunk_size=2.5, save_options=GltfSaveOptions(spz_compression=True)),
        ext_attributes={LIDAR_INTENSITY: intensity, LIDAR_RAYDROP: raydrop, RAYDROP_SH: sh},
    )

    usdz_path = tmp_path / "scene.usdz"
    result = save_scene_usdz(
        ts_dir / "tileset.json",
        usdz_path,
        options=SceneUsdzOptions(chunk_size=3.0),
    )

    with zipfile.ZipFile(usdz_path) as zf:
        scene = json.loads(zf.read("scene.json"))
        ext_block = scene["gaussians"]["ext_attributes"]
        assert RAYDROP_SH in ext_block["attributes"]
        assert ext_block["raydrop_sh_degree"] == 2

        tileset = json.loads(zf.read("tileset.json"))
        for child in tileset["root"]["children"]:
            lidar = child["content"]["extensions"][EXT_GAUSSIAN_LIDAR_NAME]
            assert lidar["raydrop_sh_degree"] == 2

        lidar_files = sorted(nm for nm in zf.namelist() if nm.endswith(".lidar"))
        total = 0
        for lidar_name in lidar_files:
            out = decode_lidar_sidecar(zf.read(lidar_name))
            assert out[RAYDROP_SH].shape[1] == 8
            total += out[RAYDROP_SH].shape[0]
        assert total == result.n_gaussians == n
