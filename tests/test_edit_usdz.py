"""Tests for :mod:`3dgs_io.edit_usdz` and ``python -m 3dgs_io.edit_usdz_cli``."""

from __future__ import annotations

import importlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import spz

_mod = importlib.import_module("3dgs_io")
save_gltf = _mod.save_gltf
save_scene_usdz = _mod.save_scene_usdz

_edit = importlib.import_module("3dgs_io.edit_usdz")
_cli = importlib.import_module("3dgs_io.edit_usdz_cli")


_SAMPLE_OSM = b'<?xml version="1.0" encoding="UTF-8"?>\n<osm version="0.6"/>\n'


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------


def _make_cloud(n: int = 32) -> spz.GaussianCloud:
    rng = np.random.default_rng(0)
    gc = spz.GaussianCloud()
    gc.antialiased = False
    gc.positions = rng.uniform(-10.0, 10.0, size=n * 3).astype(np.float32)
    quats = rng.standard_normal((n, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    gc.rotations = quats.reshape(-1)
    gc.scales = rng.uniform(-3.0, 0.5, size=n * 3).astype(np.float32)
    gc.alphas = rng.standard_normal(n).astype(np.float32)
    gc.colors = rng.uniform(0.0, 1.0, size=n * 3).astype(np.float32)
    gc.sh_degree = 0
    gc.sh = np.zeros(0, dtype=np.float32)
    return gc


def _make_tileset(tmp_path: Path) -> Path:
    save_gltf(_make_cloud(), tmp_path / "model.glb")
    doc = {
        "asset": {"version": "1.1"},
        "geometricError": 100.0,
        "root": {
            "boundingVolume": {
                "box": [0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0]
            },
            "geometricError": 0,
            "refine": "ADD",
            "content": {"uri": "model.glb"},
        },
    }
    tp = tmp_path / "tileset.json"
    tp.write_text(json.dumps(doc))
    return tp


def _make_usdz(tmp_path: Path, *, extras: dict[str, Path] | None = None) -> Path:
    ts = _make_tileset(tmp_path)
    out = tmp_path / "scene.usdz"
    save_scene_usdz(ts, out, extras=extras)
    return out


def _make_osm(tmp_path: Path, *, name: str = "map.osm", data: bytes = _SAMPLE_OSM) -> Path:
    p = tmp_path / name
    p.write_bytes(data)
    return p


# ----------------------------------------------------------------------------
# Library-level behaviour
# ----------------------------------------------------------------------------


def test_add_lanelet2_inserts_map_osm_and_updates_scene_json(tmp_path: Path) -> None:
    src = _make_usdz(tmp_path)
    osm = _make_osm(tmp_path)
    out = tmp_path / "with_map.usdz"

    result = _edit.add_lanelet2_to_usdz(src, out, osm)
    assert result.added == ["map.osm"]
    assert result.replaced == []
    assert result.out_path == out

    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert names[0] == "default.usda", "default.usda must remain first per USDZ spec"
        assert "map.osm" in names
        assert zf.read("map.osm") == _SAMPLE_OSM
        scene = json.loads(zf.read("scene.json"))
    assert scene["extras"]["map_lanelet2"] == "map.osm"


def test_add_lanelet2_preserves_original_entry_order(tmp_path: Path) -> None:
    src = _make_usdz(tmp_path)
    osm = _make_osm(tmp_path)
    out = tmp_path / "with_map.usdz"

    _edit.add_lanelet2_to_usdz(src, out, osm)

    with zipfile.ZipFile(src) as zin:
        src_names = zin.namelist()
    with zipfile.ZipFile(out) as zout:
        out_names = zout.namelist()
    assert out_names[: len(src_names)] == src_names
    assert out_names[-1] == "map.osm"


def test_add_lanelet2_replaces_existing_map_osm(tmp_path: Path) -> None:
    old_osm = _make_osm(tmp_path, name="old.osm", data=b"<osm/><!-- old -->")
    src = _make_usdz(tmp_path, extras={"map.osm": old_osm})
    new_osm = _make_osm(tmp_path, name="new.osm", data=_SAMPLE_OSM)
    out = tmp_path / "replaced.usdz"

    result = _edit.add_lanelet2_to_usdz(src, out, new_osm, overwrite=True)
    assert result.replaced == ["map.osm"]
    assert result.added == []

    with zipfile.ZipFile(out) as zf:
        assert zf.read("map.osm") == _SAMPLE_OSM
        scene = json.loads(zf.read("scene.json"))
    assert scene["extras"]["map_lanelet2"] == "map.osm"


def test_add_lanelet2_no_overwrite_raises_when_present(tmp_path: Path) -> None:
    existing_osm = _make_osm(tmp_path, name="existing.osm", data=b"<osm/>")
    src = _make_usdz(tmp_path, extras={"map.osm": existing_osm})
    with pytest.raises(ValueError, match="already contains"):
        _edit.add_lanelet2_to_usdz(
            src,
            tmp_path / "out.usdz",
            _make_osm(tmp_path, name="fresh.osm"),
            overwrite=False,
        )


def test_add_lanelet2_zip_entries_are_uncompressed(tmp_path: Path) -> None:
    src = _make_usdz(tmp_path)
    out = tmp_path / "with_map.usdz"
    _edit.add_lanelet2_to_usdz(src, out, _make_osm(tmp_path))
    with zipfile.ZipFile(out) as zf:
        for info in zf.infolist():
            assert info.compress_type == zipfile.ZIP_STORED, info.filename


def test_add_lanelet2_output_can_equal_input(tmp_path: Path) -> None:
    src = _make_usdz(tmp_path)
    osm = _make_osm(tmp_path)
    result = _edit.add_lanelet2_to_usdz(src, src, osm)
    assert result.out_path == src
    with zipfile.ZipFile(src) as zf:
        names = zf.namelist()
        assert names[0] == "default.usda"
        assert "map.osm" in names
        scene = json.loads(zf.read("scene.json"))
    assert scene["extras"]["map_lanelet2"] == "map.osm"


def test_add_lanelet2_missing_scene_json_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.usdz"
    with zipfile.ZipFile(bad, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("default.usda", "")
    with pytest.raises(ValueError, match="scene.json"):
        _edit.add_lanelet2_to_usdz(bad, tmp_path / "out.usdz", _make_osm(tmp_path))


def test_add_lanelet2_missing_input_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _edit.add_lanelet2_to_usdz(
            tmp_path / "missing.usdz",
            tmp_path / "out.usdz",
            _make_osm(tmp_path),
        )


def test_add_lanelet2_missing_lanelet2_raises(tmp_path: Path) -> None:
    src = _make_usdz(tmp_path)
    with pytest.raises(FileNotFoundError):
        _edit.add_lanelet2_to_usdz(src, tmp_path / "out.usdz", tmp_path / "missing.osm")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def test_cli_writes_output_and_summary(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    src = _make_usdz(tmp_path)
    osm = _make_osm(tmp_path)
    out = tmp_path / "cli.usdz"

    rc = _cli.main(
        [
            "--input",
            str(src),
            "--output",
            str(out),
            "--lanelet2",
            str(osm),
        ]
    )
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["out_path"] == str(out)
    assert summary["added"] == ["map.osm"]
    assert summary["replaced"] == []

    with zipfile.ZipFile(out) as zf:
        assert "map.osm" in zf.namelist()
        scene = json.loads(zf.read("scene.json"))
    assert scene["extras"]["map_lanelet2"] == "map.osm"


def test_cli_quiet_suppresses_summary(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    src = _make_usdz(tmp_path)
    out = tmp_path / "cli.usdz"
    rc = _cli.main(
        [
            "--input",
            str(src),
            "--output",
            str(out),
            "--lanelet2",
            str(_make_osm(tmp_path)),
            "--quiet",
        ]
    )
    assert rc == 0
    assert capsys.readouterr().out == ""


def test_cli_no_overwrite_errors_when_map_osm_present(tmp_path: Path) -> None:
    existing_osm = _make_osm(tmp_path, name="existing.osm", data=b"<osm/>")
    src = _make_usdz(tmp_path, extras={"map.osm": existing_osm})
    with pytest.raises(ValueError, match="already contains"):
        _cli.main(
            [
                "--input",
                str(src),
                "--output",
                str(tmp_path / "out.usdz"),
                "--lanelet2",
                str(_make_osm(tmp_path, name="fresh.osm")),
                "--no-overwrite",
            ]
        )
