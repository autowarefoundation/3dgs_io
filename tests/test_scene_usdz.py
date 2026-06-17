"""Tests for the single-file USDZ scene-bundle writer.

`save_scene_usdz` packs an spz.GaussianCloud + optional sidecar files/dirs
into one self-contained `ZIP_STORED` USDZ archive. The archive carries
`default.usda` / `scene.json` / `tileset.json` / `chunks/*.spz` (always) and
whatever extras the caller supplies (verbatim).
"""

from __future__ import annotations

import importlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import spz

_mod = importlib.import_module("3dgs_io")
SceneUsdzOptions = _mod.SceneUsdzOptions
save_scene_usdz = _mod.save_scene_usdz
save_usdz = _mod.save_usdz


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_cloud(n: int = 256, sh_degree: int = 3) -> spz.GaussianCloud:
    rng = np.random.default_rng(0)
    gc = spz.GaussianCloud()
    gc.antialiased = False
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


def _names(usdz: Path) -> list[str]:
    with zipfile.ZipFile(usdz) as zf:
        return zf.namelist()


def _read(usdz: Path, name: str) -> bytes:
    with zipfile.ZipFile(usdz) as zf:
        return zf.read(name)


# ---------------------------------------------------------------------------
# Always-present entries
# ---------------------------------------------------------------------------


def test_writes_default_usda_scene_tileset_and_chunks(tmp_path: Path) -> None:
    out = tmp_path / "scene.usdz"
    res = save_scene_usdz(_make_cloud(), out, options=SceneUsdzOptions(chunk_size=8.0))

    names = _names(out)
    assert names[0] == "default.usda", "default.usda must come first per USDZ spec"
    assert "scene.json" in names
    assert "tileset.json" in names
    assert any(n.startswith("chunks/chunk_") and n.endswith(".spz") for n in names)
    assert res.n_chunks > 0
    assert res.n_gaussians > 0
    assert res.sh_degree == 3


def test_zip_entries_are_uncompressed(tmp_path: Path) -> None:
    out = tmp_path / "scene.usdz"
    save_scene_usdz(_make_cloud(), out)
    with zipfile.ZipFile(out) as zf:
        for info in zf.infolist():
            assert info.compress_type == zipfile.ZIP_STORED, f"{info.filename} is compressed"


# ---------------------------------------------------------------------------
# scene.json / tileset.json contents
# ---------------------------------------------------------------------------


def test_scene_json_schema(tmp_path: Path) -> None:
    out = tmp_path / "scene.usdz"
    save_scene_usdz(_make_cloud(), out, options=SceneUsdzOptions(chunk_size=8.0))
    scene = json.loads(_read(out, "scene.json"))
    assert scene["schema"] == "splatsim.scene/v1"
    assert scene["world"]["up_axis"] == "z"
    assert scene["gaussians"]["tileset"] == "tileset.json"
    assert scene["gaussians"]["sh_degree"] == 3
    assert scene["gaussians"]["n_gaussians"] > 0
    # Without extras, every key is None.
    assert scene["extras"] == {
        "map_lanelet2": None,
        "map_opendrive": None,
        "carla_world": None,
        "tracks": None,
        "trajectory": None,
    }


def test_tileset_uses_ext_3dgs_spz(tmp_path: Path) -> None:
    out = tmp_path / "scene.usdz"
    save_scene_usdz(_make_cloud(), out, options=SceneUsdzOptions(chunk_size=8.0))
    ts = json.loads(_read(out, "tileset.json"))
    assert ts["asset"]["tilesetVersion"] == "splatsim-spz/1.0"
    assert "EXT_3dgs_spz" in ts["extensionsRequired"]
    children = ts["root"]["children"]
    assert children
    for c in children:
        assert c["content"]["uri"].startswith("chunks/chunk_")
        ext = c["content"]["extensions"]["EXT_3dgs_spz"]
        assert ext["format"] == "spz/1"
        assert ext["n_points"] > 0


# ---------------------------------------------------------------------------
# Extras: file and directory sources, known + arbitrary keys
# ---------------------------------------------------------------------------


def test_extras_file_is_embedded_verbatim(tmp_path: Path) -> None:
    src = tmp_path / "tracks.parquet"
    src.write_bytes(b"PAR1\x00fake-parquet")

    out = tmp_path / "scene.usdz"
    save_scene_usdz(_make_cloud(), out, extras={"tracks.parquet": src})

    names = _names(out)
    assert "tracks.parquet" in names
    assert _read(out, "tracks.parquet") == b"PAR1\x00fake-parquet"

    scene = json.loads(_read(out, "scene.json"))
    assert scene["extras"]["tracks"] == "tracks.parquet"


def test_extras_known_archive_paths_populate_scene_extras(tmp_path: Path) -> None:
    src_dir = tmp_path / "carla_root"
    src_dir.mkdir()
    (src_dir / "manifest.json").write_text('{"schema": "splatsim.carla_world/v1"}')
    (src_dir / "extra.bin").write_bytes(b"\x00\x01")

    osm = tmp_path / "map.osm"
    osm.write_text("<osm/>")
    xodr = tmp_path / "map.xodr"
    xodr.write_text("<OpenDRIVE/>")
    traj = tmp_path / "trajectory.parquet"
    traj.write_bytes(b"PAR1")

    out = tmp_path / "scene.usdz"
    save_scene_usdz(
        _make_cloud(),
        out,
        extras={
            "carla_world": src_dir,
            "map.osm": osm,
            "map.xodr": xodr,
            "trajectory.parquet": traj,
        },
    )
    names = set(_names(out))
    assert "carla_world/manifest.json" in names
    assert "carla_world/extra.bin" in names
    assert "map.osm" in names
    assert "map.xodr" in names
    assert "trajectory.parquet" in names

    scene = json.loads(_read(out, "scene.json"))
    assert scene["extras"] == {
        "map_lanelet2": "map.osm",
        "map_opendrive": "map.xodr",
        "carla_world": "carla_world/manifest.json",
        "tracks": None,
        "trajectory": "trajectory.parquet",
    }


def test_arbitrary_extras_path_is_embedded_but_not_in_scene_meta(tmp_path: Path) -> None:
    f = tmp_path / "anything.bin"
    f.write_bytes(b"\xaa\xbb\xcc")
    out = tmp_path / "scene.usdz"
    save_scene_usdz(_make_cloud(), out, extras={"misc/data.bin": f})
    assert "misc/data.bin" in _names(out)
    scene = json.loads(_read(out, "scene.json"))
    # No known-path mapping, so the scene extras stay null.
    assert scene["extras"]["carla_world"] is None
    assert scene["extras"]["map_lanelet2"] is None


def test_extras_reserved_path_is_rejected(tmp_path: Path) -> None:
    f = tmp_path / "x"
    f.write_bytes(b"x")
    with pytest.raises(ValueError, match="reserved"):
        save_scene_usdz(_make_cloud(), tmp_path / "out.usdz", extras={"scene.json": f})
    with pytest.raises(ValueError, match="reserved"):
        save_scene_usdz(_make_cloud(), tmp_path / "out.usdz", extras={"chunks/foo.spz": f})


def test_extras_missing_source_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        save_scene_usdz(_make_cloud(), tmp_path / "out.usdz", extras={"x.bin": tmp_path / "nope"})


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def test_chunk_size_controls_number_of_tiles(tmp_path: Path) -> None:
    small = save_scene_usdz(
        _make_cloud(), tmp_path / "s.usdz", options=SceneUsdzOptions(chunk_size=5.0)
    )
    big = save_scene_usdz(
        _make_cloud(), tmp_path / "b.usdz", options=SceneUsdzOptions(chunk_size=200.0)
    )
    assert small.n_chunks > big.n_chunks


def test_max_points_per_chunk_splits_dense_cells(tmp_path: Path) -> None:
    gc = _make_cloud(n=200)
    gc.positions = np.zeros_like(gc.positions)
    res = save_scene_usdz(
        gc,
        tmp_path / "out.usdz",
        options=SceneUsdzOptions(chunk_size=1000.0, max_points_per_chunk=50),
    )
    assert res.n_chunks >= 4


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def test_bbox_radius_filters_outliers(tmp_path: Path) -> None:
    gc = _make_cloud(n=64)
    pos = np.array(gc.positions, dtype=np.float32).reshape(-1, 3)
    pos[0] = [1000.0, 1000.0, 1000.0]
    gc.positions = pos.reshape(-1)
    res = save_scene_usdz(gc, tmp_path / "out.usdz", options=SceneUsdzOptions(bbox_radius=100.0))
    assert res.n_gaussians == 63


def test_opacity_threshold_drops_points(tmp_path: Path) -> None:
    gc = _make_cloud(n=256)
    full = save_scene_usdz(gc, tmp_path / "full.usdz")
    gated = save_scene_usdz(
        gc, tmp_path / "gated.usdz", options=SceneUsdzOptions(opacity_threshold=0.7)
    )
    assert 0 < gated.n_gaussians < full.n_gaussians


# ---------------------------------------------------------------------------
# Spz tiles inside the USDZ are loadable
# ---------------------------------------------------------------------------


def test_chunks_are_loadable_spz(tmp_path: Path) -> None:
    spz_io = importlib.import_module("3dgs_io.spz_io")
    out = tmp_path / "scene.usdz"
    res = save_scene_usdz(_make_cloud(), out, options=SceneUsdzOptions(chunk_size=8.0))

    extracted_dir = tmp_path / "extracted"
    extracted_dir.mkdir()
    with zipfile.ZipFile(out) as zf:
        zf.extractall(extracted_dir)

    total = 0
    chunk_paths = sorted((extracted_dir / "chunks").glob("chunk_*.spz"))
    assert len(chunk_paths) == res.n_chunks
    for p in chunk_paths:
        gc = spz_io.load_spz(p)
        assert gc.num_points > 0
        assert gc.sh_degree == 3
        total += gc.num_points
    assert total == res.n_gaussians


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_with_usdz_input(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cli = importlib.import_module("3dgs_io.scene_usdz_cli")
    # build a plain USDZ input first
    src_usdz = tmp_path / "in.usdz"
    save_usdz(_make_cloud(), src_usdz)
    out = tmp_path / "out.usdz"
    rc = cli.main([str(src_usdz), str(out), "--chunk-size", "8.0"])
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["out_path"].endswith("out.usdz")
    assert summary["n_gaussians"] > 0
    assert "scene.json" in _names(out)


def test_cli_quiet_suppresses_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cli = importlib.import_module("3dgs_io.scene_usdz_cli")
    src_usdz = tmp_path / "in.usdz"
    save_usdz(_make_cloud(), src_usdz)
    rc = cli.main([str(src_usdz), str(tmp_path / "out.usdz"), "--quiet"])
    assert rc == 0
    assert capsys.readouterr().out == ""


def test_cli_extra_flag_embeds_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cli = importlib.import_module("3dgs_io.scene_usdz_cli")
    src_usdz = tmp_path / "in.usdz"
    save_usdz(_make_cloud(), src_usdz)
    extra = tmp_path / "tracks.parquet"
    extra.write_bytes(b"PAR1\x00x")

    out = tmp_path / "out.usdz"
    rc = cli.main(
        [
            str(src_usdz),
            str(out),
            "--extra",
            f"tracks.parquet={extra}",
            "--quiet",
        ]
    )
    assert rc == 0
    assert "tracks.parquet" in _names(out)
    scene = json.loads(_read(out, "scene.json"))
    assert scene["extras"]["tracks"] == "tracks.parquet"


def test_cli_unsupported_extension_raises(tmp_path: Path) -> None:
    cli = importlib.import_module("3dgs_io.scene_usdz_cli")
    bad = tmp_path / "weird.unknown"
    bad.write_bytes(b"")
    with pytest.raises(ValueError, match="Unsupported input extension"):
        cli.main([str(bad), str(tmp_path / "out.usdz")])
