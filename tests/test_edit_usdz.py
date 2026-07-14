"""Tests for :mod:`3dgs_io.edit_usdz` and ``python -m 3dgs_io.edit_usdz_cli``."""

from __future__ import annotations

import importlib
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import spz

_mod = importlib.import_module("3dgs_io")
Camera = _mod.Camera
CameraExtrinsics = _mod.CameraExtrinsics
CameraModel = _mod.CameraModel
RigPose = _mod.RigPose
RigTrajectory = _mod.RigTrajectory
save_gltf = _mod.save_gltf
save_scene_usdz = _mod.save_scene_usdz
serialize_rig_trajectories = _mod.serialize_rig_trajectories

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


def _make_rig_trajectories_json(
    tmp_path: Path,
    *,
    name: str = "rig_trajectories.json",
    model: CameraModel | None = None,
    camera_name: str = "front",
    rig_id: str = "ego",
) -> Path:
    if model is None:
        model = CameraModel.pinhole(width=1920, height=1080, fx=500, fy=500, cx=960, cy=540)
    rig = RigTrajectory(
        rig_id=rig_id,
        poses=[RigPose(timestamp_us=0, translation=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0))],
        cameras=[
            Camera(
                name=camera_name,
                camera_model=model,
                extrinsics=CameraExtrinsics(
                    translation=(0.0, 0.0, 0.0),
                    rotation=(0.0, 0.0, 0.0, 1.0),
                ),
            )
        ],
    )
    p = tmp_path / name
    p.write_text(json.dumps(serialize_rig_trajectories([rig]), indent=2), encoding="utf-8")
    return p


def _make_usdz_with_rig(tmp_path: Path, **rig_kwargs: Any) -> Path:
    rig_path = _make_rig_trajectories_json(tmp_path, **rig_kwargs)
    return _make_usdz(tmp_path, extras={"rig_trajectories.json": rig_path})


def _read_rig_camera_params(usdz_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(usdz_path) as zf:
        doc = json.loads(zf.read("rig_trajectories.json").decode("utf-8-sig"))
    (params,) = (cam["camera_model"]["parameters"] for cam in doc["camera_calibrations"].values())
    return params


# ----------------------------------------------------------------------------
# add_lanelet2_to_usdz — library API
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
# update_camera_intrinsics_in_usdz — library API
# ----------------------------------------------------------------------------


def test_intrinsics_updates_pinhole_focal_and_resolution(tmp_path: Path) -> None:
    src = _make_usdz_with_rig(tmp_path)
    out = tmp_path / "edited.usdz"
    result = _edit.update_camera_intrinsics_in_usdz(
        src,
        out,
        camera_name="front",
        width=3840,
        height=2160,
        fx=1234.5,
        fy=1200.0,
    )
    assert result.out_path == out
    assert result.camera_name == "front"
    assert result.replaced == ["rig_trajectories.json"]
    assert result.updated_fields == ["fx", "fy", "height", "width"]

    params = _read_rig_camera_params(out)
    assert params["resolution"] == [3840, 2160]
    assert params["fx"] == pytest.approx(1234.5)
    assert params["fy"] == pytest.approx(1200.0)


def test_intrinsics_output_can_equal_input(tmp_path: Path) -> None:
    src = _make_usdz_with_rig(tmp_path)
    result = _edit.update_camera_intrinsics_in_usdz(src, src, camera_name="front", fx=999.0)
    assert result.out_path == src
    params = _read_rig_camera_params(src)
    assert params["fx"] == pytest.approx(999.0)


def test_intrinsics_preserves_original_entry_order(tmp_path: Path) -> None:
    src = _make_usdz_with_rig(tmp_path)
    out = tmp_path / "edited.usdz"
    _edit.update_camera_intrinsics_in_usdz(src, out, camera_name="front", fx=800.0)
    with zipfile.ZipFile(src) as zin:
        src_names = zin.namelist()
    with zipfile.ZipFile(out) as zout:
        out_names = zout.namelist()
    assert out_names == src_names
    assert out_names[0] == "default.usda"


def test_intrinsics_requires_at_least_one_update(tmp_path: Path) -> None:
    src = _make_usdz_with_rig(tmp_path)
    with pytest.raises(ValueError, match="at least one intrinsic update"):
        _edit.update_camera_intrinsics_in_usdz(src, tmp_path / "out.usdz", camera_name="front")


def test_intrinsics_missing_rig_trajectories_raises(tmp_path: Path) -> None:
    src = _make_usdz(tmp_path)  # no rig_trajectories embedded
    with pytest.raises(ValueError, match="rig_trajectories.json"):
        _edit.update_camera_intrinsics_in_usdz(
            src, tmp_path / "out.usdz", camera_name="front", fx=800.0
        )


def test_intrinsics_missing_input_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _edit.update_camera_intrinsics_in_usdz(
            tmp_path / "missing.usdz",
            tmp_path / "out.usdz",
            camera_name="front",
            fx=1.0,
        )


# ----------------------------------------------------------------------------
# CLI — lanelet2 subcommand
# ----------------------------------------------------------------------------


def test_cli_lanelet2_writes_output_and_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    src = _make_usdz(tmp_path)
    osm = _make_osm(tmp_path)
    out = tmp_path / "cli.usdz"

    rc = _cli.main(
        [
            "lanelet2",
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


def test_cli_lanelet2_quiet_suppresses_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    src = _make_usdz(tmp_path)
    out = tmp_path / "cli.usdz"
    rc = _cli.main(
        [
            "lanelet2",
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


def test_cli_lanelet2_no_overwrite_errors_when_map_osm_present(tmp_path: Path) -> None:
    existing_osm = _make_osm(tmp_path, name="existing.osm", data=b"<osm/>")
    src = _make_usdz(tmp_path, extras={"map.osm": existing_osm})
    with pytest.raises(ValueError, match="already contains"):
        _cli.main(
            [
                "lanelet2",
                "--input",
                str(src),
                "--output",
                str(tmp_path / "out.usdz"),
                "--lanelet2",
                str(_make_osm(tmp_path, name="fresh.osm")),
                "--no-overwrite",
            ]
        )


# ----------------------------------------------------------------------------
# CLI — intrinsics subcommand
# ----------------------------------------------------------------------------


def test_cli_intrinsics_updates_and_prints_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    src = _make_usdz_with_rig(tmp_path)
    out = tmp_path / "cli.usdz"
    rc = _cli.main(
        [
            "intrinsics",
            "--input",
            str(src),
            "--output",
            str(out),
            "--camera",
            "front",
            "--width",
            "3840",
            "--height",
            "2160",
            "--fx",
            "1000",
            "--fy",
            "1010",
        ]
    )
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["out_path"] == str(out)
    assert summary["camera_name"] == "front"
    assert summary["updated_fields"] == ["fx", "fy", "height", "width"]
    assert summary["replaced"] == ["rig_trajectories.json"]

    params = _read_rig_camera_params(out)
    assert params["resolution"] == [3840, 2160]
    assert params["fx"] == pytest.approx(1000.0)
    assert params["fy"] == pytest.approx(1010.0)


def test_cli_intrinsics_no_updates_returns_2(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    src = _make_usdz_with_rig(tmp_path)
    rc = _cli.main(
        [
            "intrinsics",
            "--input",
            str(src),
            "--output",
            str(tmp_path / "out.usdz"),
            "--camera",
            "front",
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "no intrinsic updates" in err


def test_cli_intrinsics_distortion_coeffs_on_opencv(tmp_path: Path) -> None:
    opencv_model = CameraModel.opencv(
        width=1920,
        height=1080,
        fx=500,
        fy=500,
        cx=960,
        cy=540,
        distortion_coeffs=[0.0, 0.0, 0.0, 0.0, 0.0],
    )
    src = _make_usdz_with_rig(tmp_path, model=opencv_model)
    out = tmp_path / "cli.usdz"
    rc = _cli.main(
        [
            "intrinsics",
            "--input",
            str(src),
            "--output",
            str(out),
            "--camera",
            "front",
            "--distortion-coeffs",
            "0.1,-0.05,0.001,0.002,0.0",
            "--quiet",
        ]
    )
    assert rc == 0
    params = _read_rig_camera_params(out)
    assert params["distortion_coeffs"] == pytest.approx([0.1, -0.05, 0.001, 0.002, 0.0])


# ----------------------------------------------------------------------------
# set_usdz_metadata — library API
# ----------------------------------------------------------------------------


def _read_metadata_yaml(usdz_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(usdz_path) as zf:
        return json.loads(zf.read("metadata.yaml").decode("utf-8-sig"))


def test_set_usdz_metadata_overwrites_default_manifest(tmp_path: Path) -> None:
    src = _make_usdz(tmp_path)
    out = tmp_path / "edited.usdz"

    result = _edit.set_usdz_metadata(
        src,
        out,
        uuid="odaibatest5",
        scene_id="odaibatest5",
        version_string="local-e2e",
    )
    assert result.out_path == out
    assert result.replaced == ["metadata.yaml"]
    assert result.added == []
    assert result.metadata == {
        "uuid": "odaibatest5",
        "scene_id": "odaibatest5",
        "version_string": "local-e2e",
    }

    doc = _read_metadata_yaml(out)
    assert doc["uuid"] == "odaibatest5"
    assert doc["scene_id"] == "odaibatest5"
    assert doc["version_string"] == "local-e2e"


def test_set_usdz_metadata_adds_manifest_when_missing(tmp_path: Path) -> None:
    src = _make_usdz(tmp_path)
    # Simulate a legacy archive that predates the metadata.yaml commitment.
    stripped = tmp_path / "legacy.usdz"
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(stripped, "w", zipfile.ZIP_STORED) as zout:
        for info in zin.infolist():
            if info.filename == "metadata.yaml":
                continue
            zout.writestr(info, zin.read(info.filename))
    with zipfile.ZipFile(stripped) as zf:
        assert "metadata.yaml" not in zf.namelist()

    out = tmp_path / "restored.usdz"
    result = _edit.set_usdz_metadata(stripped, out, uuid="u", scene_id="s", version_string="v")
    assert result.added == ["metadata.yaml"]
    assert result.replaced == []
    doc = _read_metadata_yaml(out)
    assert doc == {"uuid": "u", "scene_id": "s", "version_string": "v"}


def test_set_usdz_metadata_inherits_missing_fields_from_existing(tmp_path: Path) -> None:
    src = _make_usdz(tmp_path)
    existing = _read_metadata_yaml(src)

    out = tmp_path / "partial.usdz"
    result = _edit.set_usdz_metadata(src, out, uuid="fixed-uuid")
    assert result.metadata["uuid"] == "fixed-uuid"
    # scene_id / version_string carry through from the source manifest.
    assert result.metadata["scene_id"] == existing["scene_id"]
    assert result.metadata["version_string"] == existing["version_string"]


def test_set_usdz_metadata_output_can_equal_input(tmp_path: Path) -> None:
    src = _make_usdz(tmp_path)
    result = _edit.set_usdz_metadata(src, src, uuid="u2", scene_id="s2", version_string="v2")
    assert result.out_path == src
    doc = _read_metadata_yaml(src)
    assert doc == {"uuid": "u2", "scene_id": "s2", "version_string": "v2"}


def test_set_usdz_metadata_rejects_empty_required_field(tmp_path: Path) -> None:
    src = _make_usdz(tmp_path)
    with pytest.raises(ValueError, match="uuid"):
        _edit.set_usdz_metadata(src, tmp_path / "out.usdz", uuid="")


def test_set_usdz_metadata_extras_persist(tmp_path: Path) -> None:
    src = _make_usdz(tmp_path)
    out = tmp_path / "with_extras.usdz"
    _edit.set_usdz_metadata(
        src,
        out,
        uuid="u",
        scene_id="s",
        version_string="v",
        extras={"pipeline": "alpasim", "frame_count": 60},
    )
    doc = _read_metadata_yaml(out)
    assert doc["pipeline"] == "alpasim"
    assert doc["frame_count"] == 60


def test_set_usdz_metadata_missing_input_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _edit.set_usdz_metadata(
            tmp_path / "missing.usdz",
            tmp_path / "out.usdz",
            uuid="u",
            scene_id="s",
            version_string="v",
        )


# ----------------------------------------------------------------------------
# CLI — metadata subcommand
# ----------------------------------------------------------------------------


def test_cli_metadata_writes_manifest_and_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    src = _make_usdz(tmp_path)
    out = tmp_path / "cli.usdz"

    rc = _cli.main(
        [
            "metadata",
            "--input",
            str(src),
            "--output",
            str(out),
            "--uuid",
            "odaibatest5",
            "--scene-id",
            "odaibatest5",
            "--version-string",
            "local-e2e",
        ]
    )
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["out_path"] == str(out)
    assert summary["metadata"] == {
        "uuid": "odaibatest5",
        "scene_id": "odaibatest5",
        "version_string": "local-e2e",
    }
    assert summary["replaced"] == ["metadata.yaml"]

    doc = _read_metadata_yaml(out)
    assert doc["uuid"] == "odaibatest5"


def test_cli_metadata_extras_flag_is_json_parsed(tmp_path: Path) -> None:
    src = _make_usdz(tmp_path)
    out = tmp_path / "cli.usdz"
    rc = _cli.main(
        [
            "metadata",
            "--input",
            str(src),
            "--output",
            str(out),
            "--uuid",
            "u",
            "--scene-id",
            "s",
            "--version-string",
            "v",
            "--extra",
            "frame_count=60",
            "--extra",
            'tags=["mock","alpasim"]',
            "--extra",
            "pipeline=alpasim",
            "--quiet",
        ]
    )
    assert rc == 0
    doc = _read_metadata_yaml(out)
    assert doc["frame_count"] == 60
    assert doc["tags"] == ["mock", "alpasim"]
    assert doc["pipeline"] == "alpasim"


def test_cli_metadata_rejects_shadowing_extra(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    src = _make_usdz(tmp_path)
    rc = _cli.main(
        [
            "metadata",
            "--input",
            str(src),
            "--output",
            str(tmp_path / "out.usdz"),
            "--uuid",
            "u",
            "--scene-id",
            "s",
            "--version-string",
            "v",
            "--extra",
            "uuid=other",
        ]
    )
    assert rc == 2
    assert "shadow" in capsys.readouterr().err


# ----------------------------------------------------------------------------
# convert_rig_trajectories_to_alpasim_schema & bundle_usdz_for_alpasim
# ----------------------------------------------------------------------------


def _read_archive_json(usdz_path: Path, name: str) -> Any:
    with zipfile.ZipFile(usdz_path) as zf:
        return json.loads(zf.read(name).decode("utf-8-sig"))


def test_convert_rig_trajectories_to_alpasim_schema_rewrites_in_legacy_schema(
    tmp_path: Path,
) -> None:
    usdz = _make_usdz_with_rig(tmp_path)
    out = tmp_path / "scene.alpasim.usdz"

    result = _edit.convert_rig_trajectories_to_alpasim_schema(usdz, out)

    assert result.out_path == out
    assert "rig_trajectories.json" in result.replaced
    doc = _read_archive_json(out, "rig_trajectories.json")
    # Legacy alpasim keys must be present; splatsim v1 keys must NOT be.
    assert "rig_trajectories" in doc
    assert "camera_calibrations" in doc
    assert "world_to_nre" in doc
    assert "rigs" not in doc  # splatsim v1 top-level key
    assert "front" in doc["camera_calibrations"]
    assert doc["rig_trajectories"][0]["sequence_id"] == "ego"


def test_convert_rig_trajectories_requires_rig_trajectories_json(tmp_path: Path) -> None:
    usdz = _make_usdz(tmp_path)  # no rig_trajectories.json
    with pytest.raises(ValueError, match="rig_trajectories.json"):
        _edit.convert_rig_trajectories_to_alpasim_schema(usdz, tmp_path / "out.usdz")


def test_convert_rig_trajectories_in_place_overwrites_input(tmp_path: Path) -> None:
    usdz = _make_usdz_with_rig(tmp_path)
    _edit.convert_rig_trajectories_to_alpasim_schema(usdz, usdz)
    doc = _read_archive_json(usdz, "rig_trajectories.json")
    assert "camera_calibrations" in doc


def test_bundle_usdz_for_alpasim_converts_rig_and_writes_metadata(tmp_path: Path) -> None:
    usdz = _make_usdz_with_rig(tmp_path)
    out = tmp_path / "scene.bundle.usdz"

    result = _edit.bundle_usdz_for_alpasim(usdz, out)

    assert result.rig_schema_converted is True
    assert result.metadata_written is True
    assert result.lanelet2_embedded is False
    doc = _read_archive_json(out, "rig_trajectories.json")
    assert "camera_calibrations" in doc
    # metadata.yaml must be present
    with zipfile.ZipFile(out) as zf:
        assert _mod.USDZ_METADATA_ARCHIVE_PATH in zf.namelist()


def test_bundle_usdz_for_alpasim_embeds_lanelet2_when_provided(tmp_path: Path) -> None:
    usdz = _make_usdz_with_rig(tmp_path)
    osm = _make_osm(tmp_path)
    out = tmp_path / "scene.bundle.usdz"

    result = _edit.bundle_usdz_for_alpasim(usdz, out, lanelet2_path=osm)

    assert result.lanelet2_embedded is True
    scene = _read_archive_json(out, "scene.json")
    assert scene["extras"]["map_lanelet2"] == "map.osm"
    with zipfile.ZipFile(out) as zf:
        assert zf.read("map.osm") == _SAMPLE_OSM


def test_bundle_usdz_for_alpasim_applies_world_to_nre(tmp_path: Path) -> None:
    usdz = _make_usdz_with_rig(tmp_path)
    out = tmp_path / "scene.bundle.usdz"
    w2n = [
        [1.0, 0.0, 0.0, 10.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]

    _edit.bundle_usdz_for_alpasim(usdz, out, world_to_nre=w2n)

    doc = _read_archive_json(out, "rig_trajectories.json")
    np.testing.assert_allclose(doc["world_to_nre"]["matrix"], w2n)


def test_bundle_usdz_for_alpasim_missing_rig_trajectories_raises(tmp_path: Path) -> None:
    usdz = _make_usdz(tmp_path)
    with pytest.raises(ValueError, match="rig_trajectories.json"):
        _edit.bundle_usdz_for_alpasim(usdz, tmp_path / "out.usdz")


def test_bundle_usdz_for_alpasim_missing_lanelet2_raises(tmp_path: Path) -> None:
    usdz = _make_usdz_with_rig(tmp_path)
    with pytest.raises(FileNotFoundError):
        _edit.bundle_usdz_for_alpasim(
            usdz, tmp_path / "out.usdz", lanelet2_path=tmp_path / "does_not_exist.osm"
        )


def test_cli_alpasim_bundle_end_to_end(tmp_path: Path) -> None:
    usdz = _make_usdz_with_rig(tmp_path)
    osm = _make_osm(tmp_path)
    out = tmp_path / "scene.bundle.usdz"
    w2n_path = tmp_path / "w2n.json"
    w2n_path.write_text(
        json.dumps(
            [
                [1.0, 0.0, 0.0, 5.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    )

    rc = _cli.main(
        [
            "alpasim-bundle",
            "--input",
            str(usdz),
            "--output",
            str(out),
            "--lanelet2",
            str(osm),
            "--world-to-nre",
            str(w2n_path),
            "--quiet",
        ]
    )

    assert rc == 0
    doc = _read_archive_json(out, "rig_trajectories.json")
    assert "camera_calibrations" in doc
    assert doc["world_to_nre"]["matrix"][0][3] == 5.0
    with zipfile.ZipFile(out) as zf:
        assert "map.osm" in zf.namelist()
        assert _mod.USDZ_METADATA_ARCHIVE_PATH in zf.namelist()
