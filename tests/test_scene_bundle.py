"""Tests for the splatsim scene-bundle writer (USDZ → bundle).

The writer only emits the 3D-GS portion of the bundle: scene.json,
tileset.json, chunks/*.spz. Non-gaussian sidecars (map.osm / carla_world /
parquet) must be produced by upstream tooling; the writer surfaces
pre-existing ones through scene.json's ``extras`` block.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import spz

_mod = importlib.import_module("3dgs_io")
SceneBundleOptions = _mod.SceneBundleOptions
save_scene_bundle = _mod.save_scene_bundle
save_usdz = _mod.save_usdz


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_cloud(n: int = 256, sh_degree: int = 3) -> spz.GaussianCloud:
    rng = np.random.default_rng(0)
    gc = spz.GaussianCloud()
    gc.antialiased = False
    # Spread positions across several chunk_size=10 cells so chunking is exercised.
    gc.positions = rng.uniform(-20.0, 20.0, size=n * 3).astype(np.float32)
    quats = rng.standard_normal((n, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    gc.rotations = quats.reshape(-1)
    gc.scales = rng.uniform(-3.0, 0.5, size=n * 3).astype(np.float32)
    gc.alphas = rng.standard_normal(n).astype(np.float32)
    gc.colors = rng.uniform(0.0, 1.0, size=n * 3).astype(np.float32)
    per_ch = (sh_degree + 1) ** 2 - 1
    if per_ch > 0:
        gc.sh_degree = sh_degree
        gc.sh = rng.standard_normal(n * per_ch * 3).astype(np.float32)
    else:
        gc.sh_degree = 0
        gc.sh = np.zeros(0, dtype=np.float32)
    return gc


def _make_usdz(path: Path, gc: spz.GaussianCloud | None = None) -> Path:
    save_usdz(gc if gc is not None else _make_cloud(), path)
    return path


# ---------------------------------------------------------------------------
# save_scene_bundle — 3D-GS outputs only
# ---------------------------------------------------------------------------


def test_writes_scene_tileset_and_chunks(tmp_path: Path) -> None:
    usdz = _make_usdz(tmp_path / "in.usdz")
    out = tmp_path / "bundle"
    res = save_scene_bundle(usdz, out, SceneBundleOptions(chunk_size=8.0))

    assert (out / "scene.json").is_file()
    assert (out / "tileset.json").is_file()
    assert any((out / "chunks").glob("chunk_*.spz")), "no spz chunks written"
    assert res.n_gaussians > 0
    assert res.sh_degree == 3
    assert res.chunks


def test_writer_does_not_create_sidecars(tmp_path: Path) -> None:
    """Non-gaussian sidecars must never be fabricated by this writer."""
    usdz = _make_usdz(tmp_path / "in.usdz")
    out = tmp_path / "bundle"
    save_scene_bundle(usdz, out, SceneBundleOptions(chunk_size=8.0))

    assert not (out / "map.osm").exists()
    assert not (out / "map.xodr").exists()
    assert not (out / "carla_world").exists()
    assert not (out / "tracks.parquet").exists()
    assert not (out / "trajectory.parquet").exists()
    assert not (out / "sky").exists()


def test_scene_json_schema(tmp_path: Path) -> None:
    usdz = _make_usdz(tmp_path / "in.usdz")
    out = tmp_path / "bundle"
    save_scene_bundle(usdz, out, SceneBundleOptions(chunk_size=8.0))
    scene = json.loads((out / "scene.json").read_text())

    assert scene["schema"] == "splatsim.scene/v1"
    assert scene["producer"]["tool"] == "splatsim-import-usdz"
    assert scene["world"]["up_axis"] == "z"
    assert scene["world"]["units"] == "meters"
    assert scene["gaussians"]["tileset"] == "tileset.json"
    assert scene["gaussians"]["tile_content_format"] == "spz/1"
    assert scene["gaussians"]["sh_degree"] == 3
    assert scene["gaussians"]["n_gaussians"] > 0
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


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def test_opacity_threshold_drops_points(tmp_path: Path) -> None:
    gc = _make_cloud(n=256)
    usdz = _make_usdz(tmp_path / "in.usdz", gc)
    out_a = tmp_path / "bundle_a"
    out_b = tmp_path / "bundle_b"
    full = save_scene_bundle(usdz, out_a, SceneBundleOptions(chunk_size=8.0))
    gated = save_scene_bundle(
        usdz, out_b, SceneBundleOptions(chunk_size=8.0, opacity_threshold=0.7)
    )
    assert 0 < gated.n_gaussians < full.n_gaussians


def test_bbox_radius_filters_outliers(tmp_path: Path) -> None:
    gc = _make_cloud(n=64)
    pos = np.array(gc.positions, dtype=np.float32).reshape(-1, 3)
    pos[0] = [1000.0, 1000.0, 1000.0]
    gc.positions = pos.reshape(-1)
    usdz = _make_usdz(tmp_path / "in.usdz", gc)
    out = tmp_path / "bundle"
    res = save_scene_bundle(usdz, out, SceneBundleOptions(bbox_radius=100.0))
    assert res.n_gaussians == 63


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def test_chunk_size_controls_number_of_tiles(tmp_path: Path) -> None:
    usdz = _make_usdz(tmp_path / "in.usdz")
    res_small = save_scene_bundle(usdz, tmp_path / "small", SceneBundleOptions(chunk_size=5.0))
    res_big = save_scene_bundle(usdz, tmp_path / "big", SceneBundleOptions(chunk_size=200.0))
    assert len(res_small.chunks) > len(res_big.chunks)


def test_max_points_per_chunk_splits_dense_cells(tmp_path: Path) -> None:
    """All points in one cell + low max_points → many sub-chunks."""
    gc = _make_cloud(n=200)
    gc.positions = np.zeros_like(gc.positions)
    usdz = _make_usdz(tmp_path / "in.usdz", gc)
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
    summary = json.loads(capsys.readouterr().out)
    assert summary["scene_json"].endswith("scene.json")
    assert summary["n_gaussians"] > 0


def test_cli_quiet(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from importlib import import_module

    cli = import_module("3dgs_io.scene_bundle_cli")
    usdz = _make_usdz(tmp_path / "in.usdz")
    out = tmp_path / "bundle"
    rc = cli.main([str(usdz), str(out), "--chunk-size", "8.0", "--quiet"])
    assert rc == 0
    assert capsys.readouterr().out == ""


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
