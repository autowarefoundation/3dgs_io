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
RigPose = _mod.RigPose
RigTrajectory = _mod.RigTrajectory
load_rig_trajectories_from_usdz = _mod.load_rig_trajectories_from_usdz
parse_alpasim_rig_trajectories = _mod.parse_alpasim_rig_trajectories
parse_rig_trajectories = _mod.parse_rig_trajectories
save_scene_usdz = _mod.save_scene_usdz
serialize_rig_trajectories = _mod.serialize_rig_trajectories


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
    assert doc["schema"] == "splatsim.rig_trajectories/v1"
    assert doc["frame"] == "root_local"
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


def test_parse_rejects_duplicate_ids() -> None:
    """parse must also enforce uniqueness — symmetric with serialize."""
    rig = _trajectory("ego")
    bad = {
        "schema": "splatsim.rig_trajectories/v1",
        "rigs": [rig.to_dict(), rig.to_dict()],
    }
    with pytest.raises(ValueError, match="duplicate rig_id"):
        parse_rig_trajectories(bad)


def test_parse_rejects_wrong_schema() -> None:
    bad = {"schema": "other/v1", "rigs": []}
    with pytest.raises(ValueError, match="unexpected rig_trajectories schema"):
        parse_rig_trajectories(bad)


def test_parse_rejects_missing_rigs_list() -> None:
    bad = {"schema": "splatsim.rig_trajectories/v1"}
    with pytest.raises(ValueError, match="missing the 'rigs' list"):
        parse_rig_trajectories(bad)


def test_pose_validates_translation_length() -> None:
    bad_pose = {"timestamp_us": 1, "translation": [1, 2], "rotation": [0, 0, 0, 1]}
    bad_rig = {"rig_id": "ego", "poses": [bad_pose]}
    with pytest.raises(ValueError, match="translation must have 3"):
        parse_rig_trajectories({"schema": "splatsim.rig_trajectories/v1", "rigs": [bad_rig]})


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
    # M_root_local = world_to_nre @ T_world_base @ T_rig_world = (-100+50, 20+0, 0+0)
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


# ---------------------------------------------------------------------------
# Integration with save_scene_usdz / load_rig_trajectories_from_usdz
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
    assert rig_doc["schema"] == "splatsim.rig_trajectories/v1"
    assert [r["rig_id"] for r in rig_doc["rigs"]] == ["ego"]


def test_load_rig_trajectories_from_usdz_round_trip(
    tmp_path: Path, make_minimal_tileset_with_glb
) -> None:
    ts = make_minimal_tileset_with_glb(tmp_path)
    out = tmp_path / "scene.usdz"
    original = [_trajectory("ego", n_frames=10)]
    save_scene_usdz(ts, out, rig_trajectories=original)
    recovered = load_rig_trajectories_from_usdz(out)
    assert len(recovered) == 1
    assert len(recovered[0].poses) == 10
    assert recovered[0].poses[5].translation == (5.0, 0.0, 0.0)


def test_load_rig_trajectories_from_usdz_missing_raises(
    tmp_path: Path, make_minimal_tileset_with_glb
) -> None:
    ts = make_minimal_tileset_with_glb(tmp_path)
    out = tmp_path / "scene.usdz"
    save_scene_usdz(ts, out)
    with pytest.raises(FileNotFoundError, match="no rig_trajectories.json"):
        load_rig_trajectories_from_usdz(out)


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


def test_cli_rig_trajectories_flag_native_schema(
    tmp_path: Path, make_minimal_tileset_with_glb
) -> None:
    cli = importlib.import_module("3dgs_io.scene_usdz_cli")
    ts = make_minimal_tileset_with_glb(tmp_path)
    rig_path = tmp_path / "rigs.json"
    rig_path.write_text(json.dumps(serialize_rig_trajectories([_trajectory("ego")])))

    out = tmp_path / "scene.usdz"
    rc = cli.main([str(ts), str(out), "--rig-trajectories", str(rig_path), "--quiet"])
    assert rc == 0
    recovered = load_rig_trajectories_from_usdz(out)
    assert [r.rig_id for r in recovered] == ["ego"]


def test_cli_rig_trajectories_flag_accepts_alpasim_format(
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
    rc = cli.main([str(ts), str(out), "--rig-trajectories", str(alp_path), "--quiet"])
    assert rc == 0
    recovered = load_rig_trajectories_from_usdz(out)
    assert len(recovered) == 1
    assert recovered[0].rig_id == "ego"
    assert recovered[0].poses[1].translation == (1.0, 2.0, 3.0)
