"""Tests for the sensor-rig trajectory dataclasses + rig_trajectories.json (de)serialisation."""

from __future__ import annotations

import importlib
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

_mod = importlib.import_module("3dgs_io")
Camera = _mod.Camera
CameraExtrinsics = _mod.CameraExtrinsics
CameraModel = _mod.CameraModel
FRAME_CONVENTION = _mod.FRAME_CONVENTION
LidarCalibration = _mod.LidarCalibration
LidarModel = _mod.LidarModel
RigPose = _mod.RigPose
RigTrajectory = _mod.RigTrajectory
dump_alpasim_rig_trajectories = _mod.dump_alpasim_rig_trajectories
load_rig_trajectories_doc = _mod.load_rig_trajectories_doc
parse_alpasim_rig_trajectories = _mod.parse_alpasim_rig_trajectories
parse_rig_trajectories = _mod.parse_rig_trajectories
save_scene_usdz = _mod.save_scene_usdz
serialize_rig_trajectories = _mod.serialize_rig_trajectories
update_camera_intrinsics = _mod.update_camera_intrinsics


def _read_rig_trajectories_from_usdz(path: Path) -> list[RigTrajectory]:
    """Test helper: pull rig_trajectories.json out of a USDZ and parse it."""
    with zipfile.ZipFile(path) as zf:
        doc = json.loads(zf.read("rig_trajectories.json").decode("utf-8-sig"))
    return load_rig_trajectories_doc(doc)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _pose(t_us: int, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> RigPose:
    return RigPose(
        timestamp_us=t_us,
        translation=(x, y, z),
        rotation=(0.0, 0.0, 0.0, 1.0),
    )


def _trajectory(rig_id: str = "ego", n_frames: int = 4) -> RigTrajectory:
    return RigTrajectory(
        rig_id=rig_id,
        poses=[_pose(27_000_000_000 + i * 100_000, x=float(i)) for i in range(n_frames)],
        metadata={"source": "test"},
    )


# ---------------------------------------------------------------------------
# Schema (de)serialisation
# ---------------------------------------------------------------------------


def test_serialize_then_parse_roundtrip() -> None:
    rigs = [_trajectory("ego"), _trajectory("aux", n_frames=2)]
    doc = serialize_rig_trajectories(rigs)
    assert doc["schema"] == "splatsim.rig_trajectories/v2"
    assert doc["frame"] == "world"
    assert len(doc["rigs"]) == 2

    recovered = parse_rig_trajectories(doc)
    assert {r.rig_id for r in recovered} == {"ego", "aux"}
    by_id = {r.rig_id: r for r in recovered}
    assert len(by_id["ego"].poses) == 4
    assert by_id["ego"].poses[0].timestamp_us == 27_000_000_000
    assert by_id["ego"].metadata == {"source": "test"}


def test_serialize_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="duplicate rig_id"):
        serialize_rig_trajectories([_trajectory("ego"), _trajectory("ego")])


def test_serialize_rejects_invalid_pose_contract() -> None:
    duplicate_time = _trajectory("ego", n_frames=2)
    duplicate_time.poses[1].timestamp_us = duplicate_time.poses[0].timestamp_us
    with pytest.raises(ValueError, match="strictly increasing"):
        serialize_rig_trajectories([duplicate_time])

    bad_rotation = _trajectory("ego", n_frames=1)
    bad_rotation.poses[0].rotation = (0.0, 0.0, 0.0, 2.0)
    with pytest.raises(ValueError, match="unit-norm"):
        serialize_rig_trajectories([bad_rotation])


def test_serialize_rejects_duplicate_camera_names_within_rig() -> None:
    Camera = _mod.Camera
    CameraExtrinsics = _mod.CameraExtrinsics
    CameraModel = _mod.CameraModel
    cam = Camera(
        name="front",
        camera_model=CameraModel.pinhole(width=640, height=480, fx=500, fy=500, cx=320, cy=240),
        extrinsics=CameraExtrinsics(translation=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0)),
    )
    rig = RigTrajectory(rig_id="ego", cameras=[cam, cam])
    with pytest.raises(ValueError, match="duplicate camera name 'front' in rig 'ego'"):
        serialize_rig_trajectories([rig])


def test_parse_rejects_duplicate_ids() -> None:
    """parse must also enforce uniqueness — symmetric with serialize."""
    rig = _trajectory("ego")
    bad = {
        "schema": "splatsim.rig_trajectories/v2",
        "frame": "world",
        "frame_convention": FRAME_CONVENTION,
        "rigs": [rig.to_dict(), rig.to_dict()],
    }
    with pytest.raises(ValueError, match="duplicate rig_id"):
        parse_rig_trajectories(bad)


def test_parse_rejects_wrong_schema() -> None:
    bad = {"schema": "other/v1", "rigs": []}
    with pytest.raises(ValueError, match="unexpected rig_trajectories schema"):
        parse_rig_trajectories(bad)


def test_parse_rejects_missing_rigs_list() -> None:
    bad = {
        "schema": "splatsim.rig_trajectories/v2",
        "frame": "world",
        "frame_convention": FRAME_CONVENTION,
    }
    with pytest.raises(ValueError, match="missing the 'rigs' list"):
        parse_rig_trajectories(bad)


def test_pose_validates_translation_length() -> None:
    bad_pose = {"timestamp_us": 1, "translation": [1, 2], "rotation": [0, 0, 0, 1]}
    bad_rig = {"rig_id": "ego", "poses": [bad_pose]}
    with pytest.raises(ValueError, match="translation must have 3"):
        parse_rig_trajectories(
            {
                "schema": "splatsim.rig_trajectories/v2",
                "frame": "world",
                "frame_convention": FRAME_CONVENTION,
                "rigs": [bad_rig],
            }
        )


# ---------------------------------------------------------------------------
# alpasim rig_trajectories.json ingestion
# ---------------------------------------------------------------------------


def _eye_4x4() -> list[list[float]]:
    return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def _translation_only_4x4(x: float, y: float, z: float) -> list[list[float]]:
    return [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]]


def test_alpasim_identity_world_to_nre_and_base_passthrough() -> None:
    """If both world_to_nre and T_world_base are identity, T_rig_worlds passes through."""
    doc = {
        "T_world_base": _eye_4x4(),
        "world_to_nre": {"matrix": _eye_4x4()},
        "rig_trajectories": [
            {
                "sequence_id": "ego",
                "T_rig_worlds": [
                    _eye_4x4(),
                    _translation_only_4x4(10.0, 0.0, 0.0),
                ],
                "T_rig_world_timestamps_us": [27_000_000_000, 27_000_100_000],
            }
        ],
    }
    rigs = parse_alpasim_rig_trajectories(doc)
    assert len(rigs) == 1
    rig = rigs[0]
    assert rig.rig_id == "ego"
    assert len(rig.poses) == 2
    assert rig.poses[0].translation == (0.0, 0.0, 0.0)
    assert rig.poses[1].translation == (10.0, 0.0, 0.0)
    np.testing.assert_allclose(rig.poses[0].rotation, (0.0, 0.0, 0.0, 1.0), atol=1e-6)


def test_alpasim_applies_world_to_nre_translation() -> None:
    """world_to_nre adds an offset to each leaf rig pose."""
    doc = {
        "T_world_base": _eye_4x4(),
        "world_to_nre": {"matrix": _translation_only_4x4(-100.0, 20.0, 0.0)},
        "rig_trajectories": [
            {
                "sequence_id": "ego",
                "T_rig_worlds": [_translation_only_4x4(50.0, 0.0, 0.0)],
                "T_rig_world_timestamps_us": [1],
            }
        ],
    }
    rigs = parse_alpasim_rig_trajectories(doc)
    # M_root_local = world_to_nre @ T_rig_world = (-100+50, 20+0, 0+0).
    # T_world_base is the ECEF anchor and is NOT composed into per-frame poses;
    # see ``test_alpasim_T_world_base_is_informational_not_composed`` below.
    assert rigs[0].poses[0].translation == (-50.0, 20.0, 0.0)


def test_alpasim_T_world_base_is_informational_not_composed() -> None:
    """T_world_base is the ECEF anchor only; it must NOT enter per-frame poses."""
    doc = {
        # Big "ECEF" anchor — must not leak into the resulting translations.
        "T_world_base": _translation_only_4x4(1_000_000.0, 0.0, 0.0),
        "world_to_nre": {"matrix": _eye_4x4()},
        "rig_trajectories": [
            {
                "sequence_id": "ego",
                "T_rig_worlds": [_translation_only_4x4(5.0, 0.0, 0.0)],
                "T_rig_world_timestamps_us": [1],
            }
        ],
    }
    rigs = parse_alpasim_rig_trajectories(doc)
    assert rigs[0].poses[0].translation == (5.0, 0.0, 0.0)
    # ECEF anchor is preserved in metadata for callers that need it.
    assert rigs[0].metadata["T_world_base"][0][3] == 1_000_000.0


def test_alpasim_world_to_nre_dict_without_matrix_key_raises_clearly() -> None:
    """A world_to_nre dict missing the 'matrix' key should raise a clear ValueError."""
    doc = {
        "T_world_base": _eye_4x4(),
        "world_to_nre": {"not_matrix": _eye_4x4()},
        "rig_trajectories": [
            {
                "sequence_id": "ego",
                "T_rig_worlds": [_eye_4x4()],
                "T_rig_world_timestamps_us": [1],
            }
        ],
    }
    with pytest.raises(ValueError, match="missing the 'matrix' key"):
        parse_alpasim_rig_trajectories(doc)


def test_alpasim_length_mismatch_raises() -> None:
    doc = {
        "T_world_base": _eye_4x4(),
        "world_to_nre": {"matrix": _eye_4x4()},
        "rig_trajectories": [
            {
                "sequence_id": "ego",
                "T_rig_worlds": [_eye_4x4(), _eye_4x4()],
                "T_rig_world_timestamps_us": [1],
            }
        ],
    }
    with pytest.raises(ValueError, match="T_rig_worlds length"):
        parse_alpasim_rig_trajectories(doc)


def test_alpasim_sequence_id_propagates_to_metadata() -> None:
    doc = {
        "T_world_base": _eye_4x4(),
        "world_to_nre": {"matrix": _eye_4x4()},
        "rig_trajectories": [
            {
                "sequence_id": "clipgt-abc",
                "T_rig_worlds": [_eye_4x4()],
                "T_rig_world_timestamps_us": [1],
            }
        ],
    }
    rigs = parse_alpasim_rig_trajectories(doc)
    assert rigs[0].rig_id == "clipgt-abc"
    assert rigs[0].metadata["sequence_id"] == "clipgt-abc"


_ALPASIM_SAMPLE = Path.home() / "Downloads" / "00040136-e651-4abd-991d-0655ccda9430.usdz"


@pytest.mark.skipif(
    not _ALPASIM_SAMPLE.exists() or os.environ.get("SKIP_USDZ_SAMPLE") == "1",
    reason="alpasim sample USDZ not available",
)
def test_alpasim_ingestion_against_real_sample() -> None:
    with zipfile.ZipFile(_ALPASIM_SAMPLE) as zf:
        doc = json.loads(zf.read("rig_trajectories.json"))
    rigs = parse_alpasim_rig_trajectories(doc)
    assert len(rigs) >= 1
    ego = rigs[0]
    assert len(ego.poses) > 100  # sample has 202 poses
    # Quaternions should be unit-norm
    for p in ego.poses:
        n = np.linalg.norm(p.rotation)
        np.testing.assert_allclose(n, 1.0, atol=1e-4)
    # Pose translations should be in NRE-local frame: bounded magnitudes
    # (not in the millions like raw ECEF).
    translations = np.array([p.translation for p in ego.poses])
    assert np.abs(translations).max() < 10_000.0, (
        "ego trajectory translations exceed reasonable NRE-local bounds — "
        "world_to_nre application may have failed"
    )
    # Cameras should be attached to the rig (sample has 6 ftheta cameras).
    assert len(ego.cameras) == 6
    cam = ego.cameras[0]
    assert cam.camera_model.type == "ftheta"
    res = cam.camera_model.parameters["resolution"]
    assert res == [1920, 1080]
    # rig-relative T_sensor_rig should have a sensible offset (sensor on the rig).
    t_sensor_rig = np.asarray(cam.extrinsics.to_t_sensor_rig())
    assert np.abs(t_sensor_rig[:3, 3]).max() < 10.0  # camera within a few metres of rig origin


# ---------------------------------------------------------------------------
# Integration with save_scene_usdz
# ---------------------------------------------------------------------------


def test_save_scene_usdz_embeds_rig_trajectories(
    tmp_path: Path, make_minimal_tileset_with_glb
) -> None:
    ts = make_minimal_tileset_with_glb(tmp_path)
    out = tmp_path / "scene.usdz"
    res = save_scene_usdz(ts, out, rig_trajectories=[_trajectory("ego")])
    assert res.extras["rig_trajectories"] == "rig_trajectories.json"

    with zipfile.ZipFile(out) as zf:
        assert "rig_trajectories.json" in zf.namelist()
        scene = json.loads(zf.read("scene.json"))
        rig_doc = json.loads(zf.read("rig_trajectories.json"))
    assert scene["extras"]["rig_trajectories"] == "rig_trajectories.json"
    assert rig_doc["schema"] == "splatsim.rig_trajectories/v2"
    assert rig_doc["frame"] == "world"
    assert rig_doc["frame_convention"] == FRAME_CONVENTION
    assert [r["rig_id"] for r in rig_doc["rigs"]] == ["ego"]


def test_save_scene_usdz_writes_only_v2_schema(
    tmp_path: Path, make_minimal_tileset_with_glb
) -> None:
    ts = make_minimal_tileset_with_glb(tmp_path)
    out = tmp_path / "scene.usdz"
    save_scene_usdz(ts, out, rig_trajectories=[_trajectory("ego")])

    with zipfile.ZipFile(out) as zf:
        rig_doc = json.loads(zf.read("rig_trajectories.json"))
    assert rig_doc["schema"] == "splatsim.rig_trajectories/v2"
    assert "world_to_nre" not in rig_doc
    assert "T_world_base" not in rig_doc
    assert [r["rig_id"] for r in rig_doc["rigs"]] == ["ego"]


def test_save_scene_usdz_rejects_legacy_transform_arguments(
    tmp_path: Path, make_minimal_tileset_with_glb
) -> None:
    ts = make_minimal_tileset_with_glb(tmp_path)
    out = tmp_path / "scene.usdz"
    with pytest.raises(TypeError, match="world_to_nre"):
        save_scene_usdz(ts, out, world_to_nre=np.eye(4))


def test_rig_trajectories_round_trip_via_usdz(
    tmp_path: Path, make_minimal_tileset_with_glb
) -> None:
    ts = make_minimal_tileset_with_glb(tmp_path)
    out = tmp_path / "scene.usdz"
    save_scene_usdz(ts, out, rig_trajectories=[_trajectory("ego", n_frames=10)])
    recovered = _read_rig_trajectories_from_usdz(out)
    assert len(recovered) == 1
    assert len(recovered[0].poses) == 10
    assert recovered[0].poses[5].translation == (5.0, 0.0, 0.0)


def test_rig_trajectories_and_extras_collision_rejected(
    tmp_path: Path, make_minimal_tileset_with_glb
) -> None:
    ts = make_minimal_tileset_with_glb(tmp_path)
    fake = tmp_path / "rig_trajectories.json"
    fake.write_text("{}")
    with pytest.raises(ValueError, match="rig_trajectories="):
        save_scene_usdz(
            ts,
            tmp_path / "out.usdz",
            rig_trajectories=[_trajectory()],
            extras={"rig_trajectories.json": fake},
        )


def test_cli_rig_trajectories_flag_writes_v2(tmp_path: Path, make_minimal_tileset_with_glb) -> None:
    cli = importlib.import_module("3dgs_io.scene_usdz_cli")
    ts = make_minimal_tileset_with_glb(tmp_path)
    rig_path = tmp_path / "rigs.json"
    rig_path.write_text(json.dumps(serialize_rig_trajectories([_trajectory("ego")])))

    out = tmp_path / "scene.usdz"
    rc = cli.main([str(ts), str(out), "--rig-trajectories", str(rig_path), "--quiet"])
    assert rc == 0
    with zipfile.ZipFile(out) as zf:
        rig_doc = json.loads(zf.read("rig_trajectories.json"))
    assert rig_doc["schema"] == "splatsim.rig_trajectories/v2"
    assert [r["rig_id"] for r in rig_doc["rigs"]] == ["ego"]
    recovered = _read_rig_trajectories_from_usdz(out)
    assert [r.rig_id for r in recovered] == ["ego"]


def test_cli_rig_trajectories_flag_rejects_alpasim_format(
    tmp_path: Path, make_minimal_tileset_with_glb
) -> None:
    cli = importlib.import_module("3dgs_io.scene_usdz_cli")
    ts = make_minimal_tileset_with_glb(tmp_path)
    alpasim_doc = {
        "T_world_base": _eye_4x4(),
        "world_to_nre": {"matrix": _eye_4x4()},
        "rig_trajectories": [
            {
                "sequence_id": "ego",
                "T_rig_worlds": [_eye_4x4(), _translation_only_4x4(1.0, 2.0, 3.0)],
                "T_rig_world_timestamps_us": [1, 2],
            }
        ],
    }
    alp_path = tmp_path / "alp.json"
    alp_path.write_text(json.dumps(alpasim_doc))

    out = tmp_path / "scene.usdz"
    with pytest.raises(ValueError, match="unexpected rig_trajectories schema"):
        cli.main([str(ts), str(out), "--rig-trajectories", str(alp_path), "--quiet"])


# ---------------------------------------------------------------------------
# update_camera_intrinsics
# ---------------------------------------------------------------------------


def _rig_with_camera(rig_id: str, camera_name: str, **model_kwargs: float) -> RigTrajectory:
    defaults = dict(width=1920, height=1080, fx=500.0, fy=500.0, cx=960.0, cy=540.0)
    defaults.update(model_kwargs)
    return RigTrajectory(
        rig_id=rig_id,
        poses=[_pose(0)],
        cameras=[
            Camera(
                name=camera_name,
                camera_model=CameraModel.pinhole(**defaults),
                extrinsics=CameraExtrinsics(
                    translation=(0.0, 0.0, 0.0),
                    rotation=(0.0, 0.0, 0.0, 1.0),
                ),
            )
        ],
    )


def test_update_camera_intrinsics_replaces_focal_length() -> None:
    rigs = [_rig_with_camera("ego", "front")]
    cam = update_camera_intrinsics(rigs, camera_name="front", fx=1234.5, fy=1234.5)
    assert cam.camera_model.parameters["fx"] == 1234.5
    assert cam.camera_model.parameters["fy"] == 1234.5
    # Round-trip through the schema preserves the edit.
    doc = serialize_rig_trajectories(rigs)
    again = parse_rig_trajectories(doc)
    assert again[0].cameras[0].camera_model.parameters["fx"] == 1234.5


def test_update_camera_intrinsics_resolves_rig_id_when_ambiguous() -> None:
    rigs = [_rig_with_camera("ego", "front"), _rig_with_camera("trailer", "front")]
    with pytest.raises(ValueError, match="multiple rigs"):
        update_camera_intrinsics(rigs, camera_name="front", fx=1000.0)
    cam = update_camera_intrinsics(rigs, camera_name="front", rig_id="trailer", fx=1000.0)
    assert cam.camera_model.parameters["fx"] == 1000.0
    # Only the trailer rig is touched.
    assert rigs[0].cameras[0].camera_model.parameters["fx"] == 500.0


def test_update_camera_intrinsics_unknown_camera_raises() -> None:
    rigs = [_rig_with_camera("ego", "front")]
    with pytest.raises(ValueError, match="camera 'rear' not found"):
        update_camera_intrinsics(rigs, camera_name="rear", fx=1.0)


def test_update_camera_intrinsics_unknown_rig_id_raises_distinct_error() -> None:
    rigs = [_rig_with_camera("ego", "front")]
    with pytest.raises(ValueError, match="rig 'trailer' not found"):
        update_camera_intrinsics(rigs, camera_name="front", rig_id="trailer", fx=1.0)


def test_update_camera_intrinsics_requires_at_least_one_update() -> None:
    rigs = [_rig_with_camera("ego", "front")]
    with pytest.raises(ValueError, match="at least one intrinsic update"):
        update_camera_intrinsics(rigs, camera_name="front")


def test_update_camera_intrinsics_preserves_extra_camera_fields() -> None:
    rigs = [_rig_with_camera("ego", "front")]
    rigs[0].cameras[0] = Camera(
        name="front",
        camera_model=rigs[0].cameras[0].camera_model,
        extrinsics=rigs[0].cameras[0].extrinsics,
        metadata={"sensor_id": "C1"},
    )
    cam = update_camera_intrinsics(rigs, camera_name="front", fx=1234.5)
    assert cam.metadata == {"sensor_id": "C1"}


# --- dump_alpasim_rig_trajectories --------------------------------------


def test_dump_alpasim_produces_required_top_level_keys() -> None:
    rigs = [_rig_with_camera("ego", "front")]
    doc = dump_alpasim_rig_trajectories(rigs)
    assert "world_to_nre" in doc and "matrix" in doc["world_to_nre"]
    assert "rig_trajectories" in doc and len(doc["rig_trajectories"]) == 1
    assert "camera_calibrations" in doc and "front" in doc["camera_calibrations"]
    rig0 = doc["rig_trajectories"][0]
    assert rig0["sequence_id"] == "ego"
    assert rig0["T_rig_world_timestamps_us"] == [0]
    assert len(rig0["T_rig_worlds"]) == 1
    # cameras_frame_timestamps_us must be [start, end] pairs, one per rig frame
    frame_ts = rig0["cameras_frame_timestamps_us"]["front"]
    assert len(frame_ts) == 1 and all(len(r) == 2 and r[1] > r[0] for r in frame_ts)


def test_dump_alpasim_camera_calibrations_use_logical_name_and_matrix() -> None:
    rigs = [_rig_with_camera("ego", "front")]
    doc = dump_alpasim_rig_trajectories(rigs)
    entry = doc["camera_calibrations"]["front"]
    assert entry["logical_sensor_name"] == "front"
    assert np.array(entry["T_sensor_rig"]).shape == (4, 4)
    assert "camera_model" in entry
    assert entry["camera_model"]["parameters"]["resolution"] == [1920, 1080]


def test_dump_alpasim_round_trip_preserves_poses_and_intrinsics() -> None:
    rigs_in = [_rig_with_camera("ego", "front")]
    doc = dump_alpasim_rig_trajectories(rigs_in)
    rigs_out = parse_alpasim_rig_trajectories(doc)
    assert len(rigs_out) == 1
    out = rigs_out[0]
    assert out.rig_id == "ego"
    assert [p.timestamp_us for p in out.poses] == [p.timestamp_us for p in rigs_in[0].poses]
    for pin, pout in zip(rigs_in[0].poses, out.poses, strict=False):
        np.testing.assert_allclose(pout.translation, pin.translation, atol=1e-9)
        np.testing.assert_allclose(pout.rotation, pin.rotation, atol=1e-9)
    # Intrinsics survive the alpasim serialisation
    assert out.cameras[0].camera_model.width == rigs_in[0].cameras[0].camera_model.width


def test_dump_alpasim_uses_metadata_t_world_base_when_arg_absent() -> None:
    rigs = [_rig_with_camera("ego", "front")]
    twb = _translation_only_4x4(1_000_000.0, 0.0, 0.0)
    rigs[0].metadata["T_world_base"] = twb
    doc = dump_alpasim_rig_trajectories(rigs)
    np.testing.assert_allclose(np.array(doc["T_world_base"]), twb, atol=1e-9)


def test_dump_alpasim_explicit_world_to_nre_inverts_into_base_frame() -> None:
    """When world_to_nre != I, T_rig_worlds must be inv(w2n) @ M_root_local."""
    rigs = [_rig_with_camera("ego", "front")]
    w2n = _translation_only_4x4(10.0, 0.0, 0.0)
    doc = dump_alpasim_rig_trajectories(rigs, world_to_nre=w2n)
    # Round-trip: parse should recover the exact same root-local poses.
    rigs_rt = parse_alpasim_rig_trajectories(doc)
    for pin, pout in zip(rigs[0].poses, rigs_rt[0].poses, strict=False):
        np.testing.assert_allclose(pout.translation, pin.translation, atol=1e-9)


def test_dump_alpasim_rejects_duplicate_camera_names_across_rigs() -> None:
    rigs = [
        _rig_with_camera("ego-a", "front"),
        _rig_with_camera("ego-b", "front"),
    ]
    with pytest.raises(ValueError, match="camera name 'front' appears in multiple rigs"):
        dump_alpasim_rig_trajectories(rigs)


def test_dump_alpasim_no_frame_timestamps_when_rig_has_no_cameras() -> None:
    rigs = [
        RigTrajectory(
            rig_id="ego",
            poses=[_pose(1_000_000, 0.0, 0.0, 0.0)],
            cameras=[],
        )
    ]
    doc = dump_alpasim_rig_trajectories(rigs)
    assert "cameras_frame_timestamps_us" not in doc["rig_trajectories"][0]


# ---------------------------------------------------------------------------
# LiDAR calibration support
# ---------------------------------------------------------------------------


def _lidar(name: str, tx: float = 0.0, ty: float = 0.0, tz: float = 0.0) -> LidarCalibration:
    return LidarCalibration(
        name=name,
        extrinsics=CameraExtrinsics(
            translation=(tx, ty, tz),
            rotation=(0.0, 0.0, 0.0, 1.0),
        ),
        logical_sensor_name=name,
    )


def test_dump_alpasim_emits_lidar_calibrations_when_present() -> None:
    rigs = [
        RigTrajectory(
            rig_id="ego",
            poses=[_pose(1_000_000)],
            lidars=[_lidar("lidar_top", tx=0.5, ty=0.0, tz=1.8)],
        )
    ]
    doc = dump_alpasim_rig_trajectories(rigs)
    assert "lidar_calibrations" in doc
    entry = doc["lidar_calibrations"]["lidar_top"]
    assert entry["logical_sensor_name"] == "lidar_top"
    mat = np.array(entry["T_sensor_rig"], dtype=np.float64)
    assert mat.shape == (4, 4)
    assert list(mat[:3, 3]) == [0.5, 0.0, 1.8]


def test_dump_alpasim_omits_lidar_calibrations_when_absent() -> None:
    # Preserve backwards compat: rigs without any LiDARs must produce the
    # exact same top-level key set as before this feature landed.
    rigs = [RigTrajectory(rig_id="ego", poses=[_pose(0)])]
    doc = dump_alpasim_rig_trajectories(rigs)
    assert "lidar_calibrations" not in doc


def test_dump_alpasim_rejects_duplicate_lidar_names_across_rigs() -> None:
    rigs = [
        RigTrajectory(rig_id="ego_a", poses=[_pose(0)], lidars=[_lidar("lidar_top")]),
        RigTrajectory(rig_id="ego_b", poses=[_pose(0)], lidars=[_lidar("lidar_top")]),
    ]
    with pytest.raises(ValueError, match="lidar name 'lidar_top' appears in multiple rigs"):
        dump_alpasim_rig_trajectories(rigs)


def test_parse_alpasim_attaches_lidars_via_frame_timestamps() -> None:
    doc = {
        "world_to_nre": {"matrix": np.eye(4).tolist()},
        "rig_trajectories": [
            {
                "sequence_id": "ego",
                "T_rig_worlds": [np.eye(4).tolist()],
                "T_rig_world_timestamps_us": [1_000_000],
                "lidars_frame_timestamps_us": {"lidar_top": [[1_000_000, 1_100_000]]},
            }
        ],
        "camera_calibrations": {},
        "lidar_calibrations": {
            "lidar_top": {
                "T_sensor_rig": np.eye(4).tolist(),
                "logical_sensor_name": "top",
            },
            "lidar_unused": {  # not referenced by any rig; should not be attached
                "T_sensor_rig": np.eye(4).tolist(),
                "logical_sensor_name": "unused",
            },
        },
    }
    rigs = parse_alpasim_rig_trajectories(doc)
    assert len(rigs) == 1
    assert [lidar.name for lidar in rigs[0].lidars] == ["lidar_top"]
    assert rigs[0].lidars[0].logical_sensor_name == "top"


def test_parse_alpasim_attaches_all_lidars_when_single_rig_has_no_membership() -> None:
    doc = {
        "world_to_nre": {"matrix": np.eye(4).tolist()},
        "rig_trajectories": [
            {
                "sequence_id": "ego",
                "T_rig_worlds": [np.eye(4).tolist()],
                "T_rig_world_timestamps_us": [1_000_000],
            }
        ],
        "camera_calibrations": {},
        "lidar_calibrations": {
            "lidar_top": {"T_sensor_rig": np.eye(4).tolist()},
            "lidar_front": {"T_sensor_rig": np.eye(4).tolist()},
        },
    }
    rigs = parse_alpasim_rig_trajectories(doc)
    assert sorted(lidar.name for lidar in rigs[0].lidars) == ["lidar_front", "lidar_top"]


def test_dump_alpasim_emits_lidars_frame_timestamps_for_membership() -> None:
    # Parse uses `lidars_frame_timestamps_us` as the rig-membership signal
    # in multi-rig documents. Without emitting it here, a multi-rig
    # round-trip would collapse or misattach LiDARs on re-parse.
    rigs = [
        RigTrajectory(
            rig_id="ego_a",
            poses=[_pose(1_000_000), _pose(1_100_000)],
            lidars=[_lidar("lidar_a")],
        ),
        RigTrajectory(
            rig_id="ego_b",
            poses=[_pose(1_000_000), _pose(1_100_000)],
            lidars=[_lidar("lidar_b")],
        ),
    ]
    doc = dump_alpasim_rig_trajectories(rigs)
    assert doc["rig_trajectories"][0]["lidars_frame_timestamps_us"] == {
        "lidar_a": [[1_000_000, 1_100_000], [1_100_000, 1_200_000]]
    }
    assert doc["rig_trajectories"][1]["lidars_frame_timestamps_us"] == {
        "lidar_b": [[1_000_000, 1_100_000], [1_100_000, 1_200_000]]
    }
    # Re-parsing must recover the same per-rig LiDAR attachment.
    reparsed = parse_alpasim_rig_trajectories(doc)
    by_id = {r.rig_id: r for r in reparsed}
    assert [lidar.name for lidar in by_id["ego_a"].lidars] == ["lidar_a"]
    assert [lidar.name for lidar in by_id["ego_b"].lidars] == ["lidar_b"]


def test_alpasim_lidar_round_trip_preserves_extrinsics_and_logical_name() -> None:
    original = {
        "world_to_nre": {"matrix": np.eye(4).tolist()},
        "rig_trajectories": [
            {
                "sequence_id": "ego",
                "T_rig_worlds": [np.eye(4).tolist()],
                "T_rig_world_timestamps_us": [1_000_000],
                "lidars_frame_timestamps_us": {"lidar_top": [[1_000_000, 1_100_000]]},
            }
        ],
        "camera_calibrations": {},
        "lidar_calibrations": {
            "lidar_top": {
                "T_sensor_rig": _translation_only_4x4(0.5, 0.0, 1.8),
                "logical_sensor_name": "top",
                "unique_sensor_idx": 7,
            }
        },
    }
    rigs = parse_alpasim_rig_trajectories(original)
    redumped = dump_alpasim_rig_trajectories(rigs)
    round_tripped = redumped["lidar_calibrations"]["lidar_top"]
    assert round_tripped["logical_sensor_name"] == "top"
    assert round_tripped["unique_sensor_idx"] == 7
    np.testing.assert_allclose(
        np.array(round_tripped["T_sensor_rig"]),
        np.array(_translation_only_4x4(0.5, 0.0, 1.8)),
    )


# ---------------------------------------------------------------------------
# LidarModel (type-tagged LiDAR intrinsics)
# ---------------------------------------------------------------------------


def _spinning_params(elevation_deg: list[float] | None = None) -> dict[str, object]:
    return {
        "n_rows": 128,
        "n_columns": 2048,
        "fps": 10.0,
        "min_range_m": 0.3,
        "max_range_m": 120.0,
        "elevation_deg": elevation_deg if elevation_deg is not None else [-25.0, 15.0],
    }


def test_lidar_model_spinning_accepts_elevation_table() -> None:
    m = LidarModel(type="spinning", parameters=_spinning_params())
    assert m.type == "spinning"
    assert m.parameters["n_rows"] == 128


def test_lidar_model_spinning_accepts_elevation_fov_deg() -> None:
    params = _spinning_params()
    del params["elevation_deg"]
    params["elevation_fov_deg"] = [-25.0, 15.0]
    LidarModel(type="spinning", parameters=params)  # must not raise


def test_lidar_model_spinning_rejects_missing_beam_layout() -> None:
    params = _spinning_params()
    del params["elevation_deg"]
    with pytest.raises(ValueError, match="beam layout"):
        LidarModel(type="spinning", parameters=params)


def test_lidar_model_spinning_rejects_missing_core_param() -> None:
    params = _spinning_params()
    del params["max_range_m"]
    with pytest.raises(ValueError, match="max_range_m"):
        LidarModel(type="spinning", parameters=params)


def test_lidar_model_unknown_type_requires_bare_shape_only() -> None:
    # Mirrors CameraModel's behaviour for unknown camera types: the bare
    # data shape is enough. Beam layout is spinning-specific.
    LidarModel(type="solid_state", parameters={"n_rows": 32, "n_columns": 1024})


def test_lidar_model_unknown_type_rejects_missing_bare_shape() -> None:
    with pytest.raises(ValueError, match=r"n_columns|n_rows"):
        LidarModel(type="mystery", parameters={"n_rows": 32})


def test_lidar_model_rejects_non_dict_parameters() -> None:
    with pytest.raises(ValueError, match="must be a dict"):
        LidarModel(type="spinning", parameters="not a dict")  # type: ignore[arg-type]


def test_lidar_model_to_from_dict_round_trip() -> None:
    original = LidarModel(type="spinning", parameters=_spinning_params())
    restored = LidarModel.from_dict(original.to_dict())
    assert restored.type == original.type
    assert restored.parameters == original.parameters


def test_lidar_model_from_dict_rejects_missing_top_level_keys() -> None:
    with pytest.raises(ValueError, match="missing required key"):
        LidarModel.from_dict({"type": "spinning"})


def test_lidar_calibration_lidar_model_optional_by_default() -> None:
    # Preserve existing behaviour: extrinsics-only calibrations still work,
    # and their emitted dict does not include a 'lidar_model' key.
    cal = LidarCalibration(
        name="lidar_top",
        extrinsics=CameraExtrinsics(translation=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0)),
    )
    assert cal.lidar_model is None
    assert "lidar_model" not in cal.to_dict()


def test_lidar_calibration_emits_lidar_model_when_set() -> None:
    cal = LidarCalibration(
        name="lidar_top",
        extrinsics=CameraExtrinsics(translation=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0)),
        lidar_model=LidarModel(type="spinning", parameters=_spinning_params()),
    )
    out = cal.to_dict()
    assert out["lidar_model"]["type"] == "spinning"
    assert out["lidar_model"]["parameters"]["n_rows"] == 128


def test_dump_alpasim_emits_lidar_model_in_calibrations() -> None:
    rigs = [
        RigTrajectory(
            rig_id="ego",
            poses=[_pose(1_000_000)],
            lidars=[
                LidarCalibration(
                    name="lidar_top",
                    extrinsics=CameraExtrinsics(
                        translation=(0.5, 0.0, 1.8), rotation=(0.0, 0.0, 0.0, 1.0)
                    ),
                    logical_sensor_name="top",
                    lidar_model=LidarModel(type="spinning", parameters=_spinning_params()),
                )
            ],
        )
    ]
    doc = dump_alpasim_rig_trajectories(rigs)
    entry = doc["lidar_calibrations"]["lidar_top"]
    assert entry["lidar_model"] == {
        "type": "spinning",
        "parameters": _spinning_params(),
    }


def test_alpasim_lidar_model_round_trip() -> None:
    original = {
        "world_to_nre": {"matrix": np.eye(4).tolist()},
        "rig_trajectories": [
            {
                "sequence_id": "ego",
                "T_rig_worlds": [np.eye(4).tolist()],
                "T_rig_world_timestamps_us": [1_000_000],
                "lidars_frame_timestamps_us": {"lidar_top": [[1_000_000, 1_100_000]]},
            }
        ],
        "camera_calibrations": {},
        "lidar_calibrations": {
            "lidar_top": {
                "T_sensor_rig": np.eye(4).tolist(),
                "logical_sensor_name": "top",
                "lidar_model": {
                    "type": "spinning",
                    "parameters": _spinning_params(),
                },
            }
        },
    }
    rigs = parse_alpasim_rig_trajectories(original)
    assert rigs[0].lidars[0].lidar_model is not None
    assert rigs[0].lidars[0].lidar_model.type == "spinning"
    # Round-trip via dump: the intrinsics must survive.
    redumped = dump_alpasim_rig_trajectories(rigs)
    assert redumped["lidar_calibrations"]["lidar_top"]["lidar_model"] == {
        "type": "spinning",
        "parameters": _spinning_params(),
    }
