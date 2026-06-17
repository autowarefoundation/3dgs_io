"""Tests for the splatsim scene-bundle writer (alpasim USDZ → bundle).

Scope: this module only writes the 3D-GS portion of the bundle
(scene.json + tileset.json + chunks/*.spz + sky/). Non-gaussian sidecars
(map.osm / carla_world / tracks.parquet / trajectory.parquet) are expected
to be produced by upstream tooling — these tests assert that the writer
*does not* attempt to fabricate them, and that ``scene.json``'s ``extras``
block correctly surfaces sidecars that happen to be pre-placed in the
output directory.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

_mod = importlib.import_module("3dgs_io")
AlpasimGaussianCloud = _mod.AlpasimGaussianCloud
AlpasimSkyCubemap = _mod.AlpasimSkyCubemap
SceneBundleOptions = _mod.SceneBundleOptions
alpasim_to_spz = _mod.alpasim_to_spz
save_scene_bundle = _mod.save_scene_bundle
save_usdz = _mod.save_usdz


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_cloud(n: int = 256, *, with_sky: bool = True) -> AlpasimGaussianCloud:
    rng = np.random.default_rng(0)
    # Spread positions across several chunk_size=10 cells so chunking is exercised.
    positions = (rng.uniform(-20.0, 20.0, size=(n, 3))).astype(np.float16)
    rotations = rng.standard_normal((n, 4)).astype(np.float16)
    rotations[:, 3] = np.abs(rotations[:, 3])  # avoid all-zero quats
    scales = (rng.uniform(-3.0, 0.5, size=(n, 3))).astype(np.float16)  # log-space
    densities = rng.standard_normal((n, 1)).astype(np.float16)
    features_albedo = rng.standard_normal((n, 5, 3)).astype(np.float16)
    features_specular = rng.standard_normal((n, 45)).astype(np.float16)  # SH deg 3
    sky = None
    if with_sky:
        sky = AlpasimSkyCubemap(
            textures=rng.uniform(0.0, 1.0, size=(1, 6, 8, 8, 3)).astype(np.float16),
            texture_grads=None,
            n_grad_updates=10,
        )
    return AlpasimGaussianCloud(
        positions=positions,
        rotations=rotations,
        scales=scales,
        densities=densities,
        features_albedo=features_albedo,
        features_specular=features_specular,
        n_active_features=3,
        timestamps_us_min=0,
        timestamps_us_max=1_000_000,
        sky=sky,
        nre_offset=(-160.0, 20.0, -1.5),
    )


def _make_usdz(path: Path, cloud: AlpasimGaussianCloud | None = None) -> Path:
    save_usdz(cloud or _make_cloud(), path)
    return path


# ---------------------------------------------------------------------------
# alpasim_to_spz
# ---------------------------------------------------------------------------


def test_alpasim_to_spz_basic_shapes() -> None:
    cloud = _make_cloud(n=128)
    gc = alpasim_to_spz(cloud)
    n = gc.num_points
    assert n > 0
    assert gc.positions.shape == (n * 3,)
    assert gc.rotations.shape == (n * 4,)
    assert gc.scales.shape == (n * 3,)
    assert gc.colors.shape == (n * 3,)
    assert gc.alphas.shape == (n,)
    assert gc.sh_degree == 3
    assert gc.sh.shape == (n * 45,)
    quats = gc.rotations.reshape(n, 4)
    np.testing.assert_allclose(np.linalg.norm(quats, axis=1), 1.0, atol=1e-3)
    scales_lin = np.exp(gc.scales.reshape(n, 3))
    assert scales_lin.min() >= 0.05 - 1e-5


def test_alpasim_to_spz_filters_non_finite() -> None:
    cloud = _make_cloud(n=64)
    cloud.positions = np.asarray(cloud.positions).copy()
    cloud.positions[0] = np.nan
    cloud.positions[1] = np.inf
    gc = alpasim_to_spz(cloud)
    assert gc.num_points == 62


def test_alpasim_to_spz_opacity_threshold_drops_points() -> None:
    cloud = _make_cloud(n=256)
    full = alpasim_to_spz(cloud)
    gated = alpasim_to_spz(cloud, SceneBundleOptions(opacity_threshold=0.7))
    assert 0 < gated.num_points < full.num_points


def test_alpasim_to_spz_bbox_radius_filters_outliers() -> None:
    cloud = _make_cloud(n=64)
    pos = np.asarray(cloud.positions).copy()
    pos[0] = [1000.0, 1000.0, 1000.0]
    cloud.positions = pos
    gc = alpasim_to_spz(cloud, SceneBundleOptions(bbox_radius=100.0))
    assert gc.num_points == 63


# ---------------------------------------------------------------------------
# save_scene_bundle — 3D-GS outputs only
# ---------------------------------------------------------------------------


def test_save_scene_bundle_writes_3dgs_outputs(tmp_path: Path) -> None:
    usdz = _make_usdz(tmp_path / "in.usdz")
    out = tmp_path / "bundle"
    res = save_scene_bundle(usdz, out, SceneBundleOptions(chunk_size=8.0))

    assert (out / "scene.json").is_file()
    assert (out / "tileset.json").is_file()
    assert (out / "chunks").is_dir()
    assert any((out / "chunks").glob("chunk_*.spz")), "no spz chunks written"
    assert (out / "sky" / "manifest.json").is_file()
    for face in ("px", "nx", "py", "ny", "pz", "nz"):
        assert (out / "sky" / f"{face}.png").is_file()

    assert res.n_gaussians > 0
    assert res.sh_degree == 3
    assert res.chunks
    assert res.sky_dir == out / "sky"


def test_save_scene_bundle_does_not_create_sidecars(tmp_path: Path) -> None:
    """Non-gaussian sidecars must never be fabricated by this writer."""
    usdz = _make_usdz(tmp_path / "in.usdz")
    out = tmp_path / "bundle"
    save_scene_bundle(usdz, out, SceneBundleOptions(chunk_size=8.0))

    assert not (out / "map.osm").exists()
    assert not (out / "map.xodr").exists()
    assert not (out / "carla_world").exists()
    assert not (out / "tracks.parquet").exists()
    assert not (out / "trajectory.parquet").exists()


def test_scene_json_schema(tmp_path: Path) -> None:
    usdz = _make_usdz(tmp_path / "in.usdz")
    out = tmp_path / "bundle"
    save_scene_bundle(usdz, out, SceneBundleOptions(chunk_size=8.0))
    scene = json.loads((out / "scene.json").read_text())

    assert scene["schema"] == "splatsim.scene/v1"
    assert scene["producer"]["tool"] == "splatsim-import-usdz"
    assert scene["world"]["up_axis"] == "z"
    assert scene["world"]["units"] == "meters"
    assert scene["world"]["nre_offset"] == [-160.0, 20.0, -1.5]
    assert scene["gaussians"]["tileset"] == "tileset.json"
    assert scene["gaussians"]["tile_content_format"] == "spz/1"
    assert scene["gaussians"]["sh_degree"] == 3
    assert scene["gaussians"]["n_gaussians"] > 0
    assert scene["sky"]["manifest"] == "sky/manifest.json"
    # No pre-existing sidecars → all extras null.
    assert scene["extras"] == {
        "map_lanelet2": None,
        "map_opendrive": None,
        "carla_world": None,
        "tracks": None,
        "trajectory": None,
    }
    assert scene["render_defaults"]["exposure"] == 1.6


def test_scene_json_extras_passthrough_existing_sidecars(tmp_path: Path) -> None:
    """When sidecars are pre-placed in out_dir, scene.json must surface them."""
    usdz = _make_usdz(tmp_path / "in.usdz")
    out = tmp_path / "bundle"
    out.mkdir()
    # Pre-populate as if produced by upstream tooling.
    (out / "map.osm").write_text("<osm/>\n")
    (out / "map.xodr").write_text("<OpenDRIVE/>\n")
    (out / "tracks.parquet").write_bytes(b"PAR1")
    (out / "trajectory.parquet").write_bytes(b"PAR1")
    (out / "carla_world").mkdir()
    (out / "carla_world" / "manifest.json").write_text("{}")

    save_scene_bundle(usdz, out, SceneBundleOptions(chunk_size=8.0))
    scene = json.loads((out / "scene.json").read_text())
    assert scene["extras"] == {
        "map_lanelet2": "map.osm",
        "map_opendrive": "map.xodr",
        "carla_world": "carla_world/manifest.json",
        "tracks": "tracks.parquet",
        "trajectory": "trajectory.parquet",
    }


def test_tileset_json_uses_ext_3dgs_spz(tmp_path: Path) -> None:
    usdz = _make_usdz(tmp_path / "in.usdz")
    out = tmp_path / "bundle"
    save_scene_bundle(usdz, out, SceneBundleOptions(chunk_size=8.0))
    ts = json.loads((out / "tileset.json").read_text())

    assert ts["asset"]["version"] == "1.0"
    assert ts["asset"]["tilesetVersion"] == "splatsim-spz/1.0"
    assert "EXT_3dgs_spz" in ts["extensionsRequired"]
    assert "EXT_3dgs_spz" in ts["extensionsUsed"]
    assert ts["root"]["refine"] == "ADD"
    assert len(ts["root"]["children"]) > 0
    for child in ts["root"]["children"]:
        c = child["content"]
        assert c["uri"].startswith("chunks/chunk_")
        ext = c["extensions"]["EXT_3dgs_spz"]
        assert ext["format"] == "spz/1"
        assert ext["n_points"] > 0
        assert "box" in child["boundingVolume"]


def test_sky_manifest_schema(tmp_path: Path) -> None:
    usdz = _make_usdz(tmp_path / "in.usdz")
    out = tmp_path / "bundle"
    save_scene_bundle(usdz, out, SceneBundleOptions(chunk_size=8.0))
    m = json.loads((out / "sky" / "manifest.json").read_text())
    assert m["schema"] == "splatsim.sky_cubemap/v1"
    assert m["face_order"] == ["px", "nx", "py", "ny", "pz", "nz"]
    assert m["resolution"] == [8, 8]
    assert m["encoding"] == "sRGB_8"


def test_bundle_without_sky(tmp_path: Path) -> None:
    cloud = _make_cloud(with_sky=False)
    usdz = _make_usdz(tmp_path / "in.usdz", cloud)
    out = tmp_path / "bundle"
    res = save_scene_bundle(usdz, out)

    assert (out / "scene.json").is_file()
    assert (out / "tileset.json").is_file()
    assert any((out / "chunks").glob("chunk_*.spz"))
    # Placeholder 1x1 sky textures (written by save_usdz) are skipped.
    assert not (out / "sky").exists()
    assert res.sky_dir is None

    scene = json.loads((out / "scene.json").read_text())
    assert scene["sky"] is None


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def test_chunk_size_controls_number_of_tiles(tmp_path: Path) -> None:
    usdz = _make_usdz(tmp_path / "in.usdz")
    out_small = tmp_path / "small_chunks"
    out_big = tmp_path / "big_chunks"
    res_small = save_scene_bundle(usdz, out_small, SceneBundleOptions(chunk_size=5.0))
    res_big = save_scene_bundle(usdz, out_big, SceneBundleOptions(chunk_size=200.0))
    assert len(res_small.chunks) > len(res_big.chunks)


def test_max_points_per_chunk_splits_dense_cells(tmp_path: Path) -> None:
    """All points in one cell + low max_points → many sub-chunks."""
    cloud = _make_cloud(n=200, with_sky=False)
    cloud.positions = np.zeros_like(cloud.positions)
    usdz = _make_usdz(tmp_path / "in.usdz", cloud)
    out = tmp_path / "bundle"
    res = save_scene_bundle(
        usdz,
        out,
        SceneBundleOptions(chunk_size=1000.0, max_points_per_chunk=50),
    )
    assert len(res.chunks) >= 4


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_smoke(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from importlib import import_module

    cli = import_module("3dgs_io.scene_bundle_cli")
    usdz = _make_usdz(tmp_path / "in.usdz")
    out = tmp_path / "bundle"
    rc = cli.main([str(usdz), str(out), "--chunk-size", "8.0"])
    assert rc == 0
    cap = capsys.readouterr()
    summary = json.loads(cap.out)
    assert summary["scene_json"].endswith("scene.json")
    assert summary["n_gaussians"] > 0


def test_cli_quiet(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from importlib import import_module

    cli = import_module("3dgs_io.scene_bundle_cli")
    usdz = _make_usdz(tmp_path / "in.usdz")
    out = tmp_path / "bundle"
    rc = cli.main([str(usdz), str(out), "--chunk-size", "8.0", "--quiet"])
    assert rc == 0
    cap = capsys.readouterr()
    assert cap.out == ""


# ---------------------------------------------------------------------------
# spz chunk round-trip
# ---------------------------------------------------------------------------


def test_spz_chunks_are_loadable(tmp_path: Path) -> None:
    spz_io = importlib.import_module("3dgs_io.spz_io")
    usdz = _make_usdz(tmp_path / "in.usdz")
    out = tmp_path / "bundle"
    res = save_scene_bundle(usdz, out, SceneBundleOptions(chunk_size=8.0))

    total = 0
    for p in res.chunks:
        gc = spz_io.load_spz(p)
        assert gc.num_points > 0
        assert gc.sh_degree == 3
        total += gc.num_points
    assert total == res.n_gaussians
