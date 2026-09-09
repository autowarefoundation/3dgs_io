from __future__ import annotations

import importlib
import json
import zipfile
from pathlib import Path

import pytest

_mod = importlib.import_module("3dgs_io")


def _pose(timestamp_us: int, x: float = 0.0):
    return _mod.RigPose(timestamp_us, (x, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))


def _camera(name: str = "front"):
    return _mod.Camera(
        name=name,
        camera_model=_mod.CameraModel.pinhole(
            width=1920, height=1080, fx=1000, fy=1000, cx=960, cy=540
        ),
        extrinsics=_mod.CameraExtrinsics(
            translation=(1.0, 0.0, 1.5), rotation=(0.0, 0.0, 0.0, 1.0)
        ),
    )


def _lidar(name: str = "top"):
    return _mod.LidarCalibration(
        name=name,
        extrinsics=_mod.CameraExtrinsics(
            translation=(0.0, 0.0, 2.0), rotation=(0.0, 0.0, 0.0, 1.0)
        ),
        lidar_model=_mod.LidarModel(
            type="spinning",
            parameters={
                "n_rows": 2,
                "n_columns": 4,
                "fps": 10,
                "min_range_m": 0.5,
                "max_range_m": 200.0,
                "elevation_deg": [-1.0, 1.0],
            },
        ),
    )


def _rig(rig_id: str = "ego"):
    return _mod.RigTrajectory(
        rig_id=rig_id,
        poses=[_pose(1_000_000, 0.0), _pose(1_100_000, 1.0)],
        cameras=[_camera()],
        lidars=[_lidar()],
    )


def test_v2_round_trip_is_alpasim_native() -> None:
    document = _mod.serialize_rig_trajectories([_rig()])
    assert document["schema"] == "splatsim.rig_trajectories/v2"
    assert document["frame"] == "world"
    assert document["frame_convention"] == _mod.FRAME_CONVENTION
    assert "world_to_nre" not in document
    assert "T_world_base" not in document
    assert "T_rig_worlds" not in json.dumps(document)
    camera = document["rigs"][0]["cameras"][0]
    lidar = document["rigs"][0]["lidars"][0]
    assert "sensor_in_rig" in camera and "T_sensor_rig" not in camera
    assert "sensor_in_rig" in lidar and "T_sensor_rig" not in lidar
    assert _mod.parse_rig_trajectories(document) == [_rig()]


def test_reader_rejects_any_non_v2_document() -> None:
    with pytest.raises(ValueError, match="unexpected rig_trajectories schema"):
        _mod.load_rig_trajectories_doc({"world_to_nre": {}, "rig_trajectories": []})
    for removed in (
        "parse_alpasim_rig_trajectories",
        "dump_alpasim_rig_trajectories",
        "parse_alpasim_sequence_tracks",
        "dump_alpasim_sequence_tracks",
        "convert_rig_trajectories_to_alpasim_schema",
        "bundle_usdz_for_alpasim",
    ):
        assert not hasattr(_mod, removed)


def test_pose_validation() -> None:
    rig = _rig()
    rig.poses[1].timestamp_us = rig.poses[0].timestamp_us
    with pytest.raises(ValueError, match="strictly increasing"):
        _mod.serialize_rig_trajectories([rig])

    rig = _rig()
    rig.poses[0].rotation = (0.0, 0.0, 0.0, 2.0)
    with pytest.raises(ValueError, match="unit-norm"):
        _mod.serialize_rig_trajectories([rig])


def test_sensor_validation_and_uniqueness() -> None:
    rig = _rig()
    rig.lidars[0].name = "front"
    with pytest.raises(ValueError, match="duplicate sensor name"):
        _mod.serialize_rig_trajectories([rig])

    rig = _rig()
    rig.cameras[0].extrinsics.rotation = (0.0, 0.0, 0.0, 0.5)
    with pytest.raises(ValueError, match="unit-norm"):
        _mod.serialize_rig_trajectories([rig])


def test_update_camera_intrinsics_round_trip() -> None:
    rigs = [_rig()]
    camera = _mod.update_camera_intrinsics(rigs, camera_name="front", fx=1234.0)
    assert camera.camera_model.parameters["fx"] == 1234.0
    recovered = _mod.parse_rig_trajectories(_mod.serialize_rig_trajectories(rigs))
    assert recovered[0].cameras[0].camera_model.parameters["fx"] == 1234.0


def test_scene_embeds_v2_rig_only(tmp_path: Path, make_minimal_tileset_with_glb) -> None:
    tileset = make_minimal_tileset_with_glb(tmp_path)
    output = tmp_path / "scene.usdz"
    _mod.save_scene_usdz(tileset, output, rig_trajectories=[_rig()])
    with zipfile.ZipFile(output) as archive:
        document = json.loads(archive.read("rig_trajectories.json"))
    assert document["schema"] == "splatsim.rig_trajectories/v2"
    assert document["frame"] == "world"
    assert not ({"world_to_nre", "T_world_base", "rig_trajectories"} & document.keys())


def test_cli_accepts_v2_rig(tmp_path: Path, make_minimal_tileset_with_glb) -> None:
    cli = importlib.import_module("3dgs_io.scene_usdz_cli")
    tileset = make_minimal_tileset_with_glb(tmp_path)
    rig_path = tmp_path / "rig.json"
    rig_path.write_text(json.dumps(_mod.serialize_rig_trajectories([_rig()])))
    output = tmp_path / "scene.usdz"
    args = [str(tileset), str(output), "--rig-trajectories", str(rig_path), "--quiet"]
    assert cli.main(args) == 0
