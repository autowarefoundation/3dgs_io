"""Tests for the PPISP dataclasses + ppisp.json (de)serialisation."""

from __future__ import annotations

import importlib
import json
import zipfile
from pathlib import Path

import pytest

_mod = importlib.import_module("3dgs_io")
Ppisp = _mod.Ppisp
PpispCamera = _mod.PpispCamera
PpispFrame = _mod.PpispFrame
PPISP_SCHEMA = _mod.PPISP_SCHEMA
add_ppisp_to_usdz = _mod.add_ppisp_to_usdz
parse_ppisp = _mod.parse_ppisp
save_scene_usdz = _mod.save_scene_usdz
serialize_ppisp = _mod.serialize_ppisp


def _read_ppisp_from_usdz(path: Path) -> dict:
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("ppisp.json").decode("utf-8-sig"))


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _sample_camera(seed: float = 0.0) -> PpispCamera:
    return PpispCamera(
        vignetting=(
            (0.5 + seed, 0.5 + seed, 0.1, -0.05, 0.02),
            (0.5, 0.5, 0.09, -0.04, 0.02),
            (0.5, 0.5, 0.08, -0.03, 0.01),
        ),
        crf=(
            (0.1, 0.9, 1.0, 0.5),
            (0.1, 0.9, 1.0, 0.5),
            (0.1, 0.9, 1.0, 0.5),
        ),
    )


def _sample_frame(timestamp_us: int, exposure: float = 0.031) -> PpispFrame:
    return PpispFrame(
        timestamp_us=timestamp_us,
        exposure=exposure,
        color=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
    )


def _sample_ppisp() -> Ppisp:
    return Ppisp(
        cameras={
            "CAM_FRONT": _sample_camera(),
            "CAM_BACK": _sample_camera(seed=0.01),
        },
        frames=[
            _sample_frame(27_567_868_848),
            _sample_frame(27_567_968_602, exposure=0.032),
        ],
        color_pinv_block_diag=[[1.0 if i == j else 0.0 for j in range(8)] for i in range(8)],
    )


# --------------------------------------------------------------------------
# Schema (de)serialisation
# --------------------------------------------------------------------------


def test_serialize_then_parse_roundtrip() -> None:
    ppisp = _sample_ppisp()
    doc = serialize_ppisp(ppisp)
    assert doc["schema"] == PPISP_SCHEMA
    assert doc["pipeline"] == ["exposure", "vignetting", "color", "crf"]
    assert set(doc["cameras"]) == {"CAM_FRONT", "CAM_BACK"}
    assert len(doc["frames"]) == 2
    assert doc["constants"]["color_pinv_block_diag"][0][0] == 1.0

    recovered = parse_ppisp(doc)
    assert set(recovered.cameras) == {"CAM_FRONT", "CAM_BACK"}
    assert recovered.cameras["CAM_FRONT"].vignetting[0] == (0.5, 0.5, 0.1, -0.05, 0.02)
    assert len(recovered.frames) == 2
    assert recovered.frames[0].timestamp_us == 27_567_868_848
    assert recovered.frames[0].exposure == pytest.approx(0.031)
    assert recovered.frames[0].color == (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
    assert recovered.color_pinv_block_diag is not None
    assert recovered.color_pinv_block_diag[0][0] == 1.0


def test_parse_rejects_wrong_schema() -> None:
    with pytest.raises(ValueError, match="unexpected ppisp schema"):
        parse_ppisp({"schema": "other/v1"})


def test_parse_rejects_bad_vignetting_shape() -> None:
    doc = serialize_ppisp(_sample_ppisp())
    doc["cameras"]["CAM_FRONT"]["vignetting"][0].append(999.0)
    with pytest.raises(ValueError, match=r"camera.vignetting\[0\] must have 5 entries"):
        parse_ppisp(doc)


def test_parse_rejects_bad_crf_shape() -> None:
    doc = serialize_ppisp(_sample_ppisp())
    doc["cameras"]["CAM_FRONT"]["crf"] = [[1.0, 2.0, 3.0]] * 3
    with pytest.raises(ValueError, match=r"camera.crf\[0\] must have 4 entries"):
        parse_ppisp(doc)


def test_parse_rejects_bad_color_dim() -> None:
    doc = serialize_ppisp(_sample_ppisp())
    doc["frames"][0]["color"] = [0.0, 0.1, 0.2]
    with pytest.raises(ValueError, match="frame.color must have 8 entries"):
        parse_ppisp(doc)


def test_serialize_rejects_duplicate_timestamps() -> None:
    dup = Ppisp(
        frames=[_sample_frame(1), _sample_frame(1)],
    )
    with pytest.raises(ValueError, match="duplicate frame timestamp_us"):
        serialize_ppisp(dup)


def test_parse_rejects_duplicate_timestamps() -> None:
    doc = {
        "schema": PPISP_SCHEMA,
        "cameras": {},
        "frames": [
            _sample_frame(42).to_dict(),
            _sample_frame(42, exposure=0.5).to_dict(),
        ],
    }
    with pytest.raises(ValueError, match="duplicate frame timestamp_us"):
        parse_ppisp(doc)


def test_ppisp_without_constants_roundtrips() -> None:
    ppisp = Ppisp(
        cameras={"CAM_FRONT": _sample_camera()},
        frames=[_sample_frame(1)],
        color_pinv_block_diag=None,
    )
    doc = serialize_ppisp(ppisp)
    assert "constants" not in doc
    recovered = parse_ppisp(doc)
    assert recovered.color_pinv_block_diag is None


# --------------------------------------------------------------------------
# Integration with save_scene_usdz
# --------------------------------------------------------------------------


def test_save_scene_usdz_embeds_ppisp(tmp_path: Path, make_minimal_tileset_with_glb) -> None:
    ts = make_minimal_tileset_with_glb(tmp_path)
    out = tmp_path / "scene.usdz"
    ppisp = _sample_ppisp()
    res = save_scene_usdz(ts, out, ppisp=ppisp)
    assert res.extras["ppisp"] == "ppisp.json"

    with zipfile.ZipFile(out) as zf:
        assert "ppisp.json" in zf.namelist()
        scene = json.loads(zf.read("scene.json"))
        assert scene["extras"]["ppisp"] == "ppisp.json"
    ppisp_doc = _read_ppisp_from_usdz(out)
    assert ppisp_doc["schema"] == PPISP_SCHEMA
    assert set(ppisp_doc["cameras"]) == {"CAM_FRONT", "CAM_BACK"}


def test_ppisp_round_trip_via_usdz(tmp_path: Path, make_minimal_tileset_with_glb) -> None:
    ts = make_minimal_tileset_with_glb(tmp_path)
    out = tmp_path / "scene.usdz"
    original = _sample_ppisp()
    save_scene_usdz(ts, out, ppisp=original)

    doc = _read_ppisp_from_usdz(out)
    recovered = parse_ppisp(doc)
    assert set(recovered.cameras) == set(original.cameras)
    assert len(recovered.frames) == len(original.frames)
    assert recovered.frames[0].timestamp_us == original.frames[0].timestamp_us


def test_ppisp_and_extras_collision_rejected(tmp_path: Path, make_minimal_tileset_with_glb) -> None:
    ts = make_minimal_tileset_with_glb(tmp_path)
    fake = tmp_path / "ppisp.json"
    fake.write_text("{}")
    with pytest.raises(ValueError, match="ppisp="):
        save_scene_usdz(
            ts,
            tmp_path / "out.usdz",
            ppisp=_sample_ppisp(),
            extras={"ppisp.json": fake},
        )


# --------------------------------------------------------------------------
# edit_usdz.add_ppisp_to_usdz
# --------------------------------------------------------------------------


def test_add_ppisp_to_usdz_adds_new_entry(tmp_path: Path, make_minimal_tileset_with_glb) -> None:
    ts = make_minimal_tileset_with_glb(tmp_path)
    scene = tmp_path / "scene.usdz"
    save_scene_usdz(ts, scene)  # no ppisp at build time
    with zipfile.ZipFile(scene) as zf:
        assert "ppisp.json" not in zf.namelist()

    edited = tmp_path / "scene_ppisp.usdz"
    result = add_ppisp_to_usdz(scene, edited, _sample_ppisp())
    assert "ppisp.json" in result.added
    with zipfile.ZipFile(edited) as zf:
        assert "ppisp.json" in zf.namelist()
        scene_doc = json.loads(zf.read("scene.json"))
    assert scene_doc["extras"]["ppisp"] == "ppisp.json"


def test_add_ppisp_to_usdz_accepts_json_path(tmp_path: Path, make_minimal_tileset_with_glb) -> None:
    ts = make_minimal_tileset_with_glb(tmp_path)
    scene = tmp_path / "scene.usdz"
    save_scene_usdz(ts, scene)

    ppisp_json = tmp_path / "ppisp.json"
    ppisp_json.write_text(json.dumps(serialize_ppisp(_sample_ppisp())))

    edited = tmp_path / "scene_ppisp.usdz"
    add_ppisp_to_usdz(scene, edited, ppisp_json)
    doc = _read_ppisp_from_usdz(edited)
    assert doc["schema"] == PPISP_SCHEMA


def test_add_ppisp_to_usdz_replaces_existing_by_default(
    tmp_path: Path, make_minimal_tileset_with_glb
) -> None:
    ts = make_minimal_tileset_with_glb(tmp_path)
    scene = tmp_path / "scene.usdz"
    save_scene_usdz(ts, scene, ppisp=_sample_ppisp())

    updated = Ppisp(
        cameras={"CAM_LEFT": _sample_camera(seed=0.02)},
        frames=[_sample_frame(99)],
    )
    edited = tmp_path / "scene_ppisp.usdz"
    result = add_ppisp_to_usdz(scene, edited, updated)
    assert "ppisp.json" in result.replaced

    doc = _read_ppisp_from_usdz(edited)
    assert set(doc["cameras"]) == {"CAM_LEFT"}
    assert doc["frames"][0]["timestamp_us"] == 99


def test_add_ppisp_to_usdz_no_overwrite_raises(
    tmp_path: Path, make_minimal_tileset_with_glb
) -> None:
    ts = make_minimal_tileset_with_glb(tmp_path)
    scene = tmp_path / "scene.usdz"
    save_scene_usdz(ts, scene, ppisp=_sample_ppisp())

    edited = tmp_path / "scene_ppisp.usdz"
    with pytest.raises(FileExistsError):
        add_ppisp_to_usdz(scene, edited, _sample_ppisp(), overwrite=False)


def test_add_ppisp_to_usdz_supports_in_place(tmp_path: Path, make_minimal_tileset_with_glb) -> None:
    ts = make_minimal_tileset_with_glb(tmp_path)
    scene = tmp_path / "scene.usdz"
    save_scene_usdz(ts, scene)

    add_ppisp_to_usdz(scene, scene, _sample_ppisp())
    doc = _read_ppisp_from_usdz(scene)
    assert doc["schema"] == PPISP_SCHEMA


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_scene_usdz_cli_accepts_ppisp_flag(tmp_path: Path, make_minimal_tileset_with_glb) -> None:
    cli = importlib.import_module("3dgs_io.scene_usdz_cli")
    ts = make_minimal_tileset_with_glb(tmp_path)
    ppisp_path = tmp_path / "ppisp.json"
    ppisp_path.write_text(json.dumps(serialize_ppisp(_sample_ppisp())))

    out = tmp_path / "scene.usdz"
    rc = cli.main([str(ts), str(out), "--ppisp", str(ppisp_path), "--quiet"])
    assert rc == 0
    doc = _read_ppisp_from_usdz(out)
    assert set(doc["cameras"]) == {"CAM_FRONT", "CAM_BACK"}


def test_edit_usdz_cli_ppisp_subcommand(tmp_path: Path, make_minimal_tileset_with_glb) -> None:
    cli = importlib.import_module("3dgs_io.edit_usdz_cli")
    ts = make_minimal_tileset_with_glb(tmp_path)
    scene = tmp_path / "scene.usdz"
    save_scene_usdz(ts, scene)

    ppisp_json = tmp_path / "ppisp.json"
    ppisp_json.write_text(json.dumps(serialize_ppisp(_sample_ppisp())))

    edited = tmp_path / "scene_ppisp.usdz"
    rc = cli.main(
        [
            "ppisp",
            "--input",
            str(scene),
            "--output",
            str(edited),
            "--ppisp",
            str(ppisp_json),
            "--quiet",
        ]
    )
    assert rc == 0
    doc = _read_ppisp_from_usdz(edited)
    assert doc["schema"] == PPISP_SCHEMA
