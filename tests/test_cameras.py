"""Tests for the Camera dataclasses + cameras.json (de)serialisation."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

_mod = importlib.import_module("3dgs_io")
Camera = _mod.Camera
CameraExtrinsics = _mod.CameraExtrinsics
CameraIntrinsics = _mod.CameraIntrinsics
load_cameras_from_usdz = _mod.load_cameras_from_usdz
parse_cameras = _mod.parse_cameras
serialize_cameras = _mod.serialize_cameras


def _intrinsics(width: int = 1920, height: int = 1080) -> CameraIntrinsics:
    return CameraIntrinsics(
        width=width,
        height=height,
        fx=1234.5,
        fy=1234.5,
        cx=width / 2,
        cy=height / 2,
        distortion_model="opencv",
        distortion_coeffs=[0.01, -0.002, 0.0, 0.0, 0.0],
    )


def _extrinsics(tx: float = 1.5, ty: float = 0.0, tz: float = 1.8) -> CameraExtrinsics:
    return CameraExtrinsics(translation=(tx, ty, tz), rotation=(0.0, 0.0, 0.0, 1.0))


def _camera(name: str = "front_left") -> Camera:
    return Camera(
        name=name,
        intrinsics=_intrinsics(),
        extrinsics=_extrinsics(),
        timestamp_us=1_700_000_000_000,
        metadata={"rig": "rig_0"},
    )


# ---------------------------------------------------------------------------
# CameraExtrinsics matrix round-trip
# ---------------------------------------------------------------------------


def test_extrinsics_to_matrix_identity() -> None:
    e = CameraExtrinsics(translation=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0))
    np.testing.assert_allclose(e.to_matrix(), np.eye(4))


def test_extrinsics_to_matrix_translation_only() -> None:
    e = CameraExtrinsics(translation=(5.0, 6.0, 7.0), rotation=(0.0, 0.0, 0.0, 1.0))
    expected = np.eye(4)
    expected[:3, 3] = [5.0, 6.0, 7.0]
    np.testing.assert_allclose(e.to_matrix(), expected)


def test_extrinsics_to_matrix_z_rotation_90deg() -> None:
    # 90° about +Z, xyzw quaternion = (0, 0, sin(45°), cos(45°))
    e = CameraExtrinsics(
        translation=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)),
    )
    m = e.to_matrix()
    # rotZ90 maps (1, 0, 0) → (0, 1, 0)
    np.testing.assert_allclose(m @ [1, 0, 0, 1], [0, 1, 0, 1], atol=1e-10)


def test_extrinsics_from_matrix_roundtrip() -> None:
    # Rot Y 30° + translation (3, -2, 1)
    theta = np.pi / 6
    r = np.array(
        [
            [np.cos(theta), 0, np.sin(theta), 3.0],
            [0, 1, 0, -2.0],
            [-np.sin(theta), 0, np.cos(theta), 1.0],
            [0, 0, 0, 1],
        ]
    )
    e = CameraExtrinsics.from_matrix(r)
    np.testing.assert_allclose(e.to_matrix(), r, atol=1e-10)


def test_extrinsics_from_matrix_unit_quaternion() -> None:
    """from_matrix must always emit a unit-norm quaternion."""
    # Random rigid transform.
    rng = np.random.default_rng(0)
    rand = rng.standard_normal((3, 3))
    q, _ = np.linalg.qr(rand)
    if np.linalg.det(q) < 0:
        q[:, -1] *= -1
    m = np.eye(4)
    m[:3, :3] = q
    m[:3, 3] = rng.standard_normal(3)
    e = CameraExtrinsics.from_matrix(m)
    n = np.linalg.norm(e.rotation)
    np.testing.assert_allclose(n, 1.0, atol=1e-10)


def test_extrinsics_zero_norm_quaternion_raises() -> None:
    e = CameraExtrinsics(translation=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="near-zero norm"):
        e.to_matrix()


# ---------------------------------------------------------------------------
# Schema (de)serialisation
# ---------------------------------------------------------------------------


def test_serialize_then_parse_roundtrip() -> None:
    cams = [_camera("front_left"), _camera("front_right")]
    cams[1].extrinsics = _extrinsics(tx=-1.5)
    doc = serialize_cameras(cams)
    assert doc["schema"] == "splatsim.cameras/v1"
    assert doc["frame"] == "root_local"
    assert len(doc["cameras"]) == 2

    roundtripped = parse_cameras(doc)
    assert len(roundtripped) == 2
    assert roundtripped[0].name == "front_left"
    assert roundtripped[1].extrinsics.translation == (-1.5, 0.0, 1.8)
    assert roundtripped[0].metadata == {"rig": "rig_0"}
    assert roundtripped[0].timestamp_us == 1_700_000_000_000


def test_serialize_rejects_duplicate_names() -> None:
    cams = [_camera("c0"), _camera("c0")]
    with pytest.raises(ValueError, match="duplicate camera name"):
        serialize_cameras(cams)


def test_parse_rejects_wrong_schema() -> None:
    bad = {"schema": "other/v1", "cameras": []}
    with pytest.raises(ValueError, match="unexpected cameras schema"):
        parse_cameras(bad)


def test_parse_rejects_missing_cameras_list() -> None:
    bad = {"schema": "splatsim.cameras/v1"}
    with pytest.raises(ValueError, match="missing the 'cameras' list"):
        parse_cameras(bad)


def test_parse_validates_extrinsics_lengths() -> None:
    cam_dict = _camera().to_dict()
    cam_dict["extrinsics"]["translation"] = [1, 2]  # too short
    bad = {"schema": "splatsim.cameras/v1", "cameras": [cam_dict]}
    with pytest.raises(ValueError, match="translation must have 3"):
        parse_cameras(bad)


def test_intrinsics_defaults_pinhole_no_distortion() -> None:
    intr = CameraIntrinsics(width=640, height=480, fx=500, fy=500, cx=320, cy=240)
    d = intr.to_dict()
    assert d["distortion_model"] == "pinhole"
    assert d["distortion_coeffs"] == []


# ---------------------------------------------------------------------------
# Integration with save_scene_usdz / load_cameras_from_usdz
# ---------------------------------------------------------------------------


def _make_minimal_tileset_with_glb(tmp_path: Path) -> Path:
    """Reuse the scene-usdz test fixture pattern."""
    import spz

    save_gltf = _mod.save_gltf
    n = 32
    rng = np.random.default_rng(0)
    gc = spz.GaussianCloud()
    gc.antialiased = False
    gc.positions = rng.uniform(-10, 10, size=n * 3).astype(np.float32)
    quats = rng.standard_normal((n, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    gc.rotations = quats.reshape(-1)
    gc.scales = rng.uniform(-3, 0, size=n * 3).astype(np.float32)
    gc.alphas = rng.standard_normal(n).astype(np.float32)
    gc.colors = rng.uniform(0, 1, size=n * 3).astype(np.float32)
    gc.sh_degree = 0
    gc.sh = np.zeros(0, dtype=np.float32)
    save_gltf(gc, tmp_path / "model.glb")

    doc = {
        "asset": {"version": "1.1"},
        "geometricError": 100.0,
        "root": {
            "boundingVolume": {"box": [0, 0, 0, 100, 0, 0, 0, 100, 0, 0, 0, 100]},
            "geometricError": 0,
            "refine": "ADD",
            "content": {"uri": "model.glb"},
        },
    }
    tp = tmp_path / "tileset.json"
    tp.write_text(json.dumps(doc))
    return tp


def test_save_scene_usdz_embeds_cameras_json(tmp_path: Path) -> None:
    save_scene_usdz = _mod.save_scene_usdz
    ts = _make_minimal_tileset_with_glb(tmp_path)
    cams = [_camera("front_left"), _camera("front_right")]
    cams[1].extrinsics = _extrinsics(tx=-1.5)
    out = tmp_path / "scene.usdz"
    res = save_scene_usdz(ts, out, cameras=cams)
    assert res.extras["cameras"] == "cameras.json"

    # Archive contents
    import zipfile

    with zipfile.ZipFile(out) as zf:
        assert "cameras.json" in zf.namelist()
        scene = json.loads(zf.read("scene.json"))
        cameras_doc = json.loads(zf.read("cameras.json"))
    assert scene["extras"]["cameras"] == "cameras.json"
    assert cameras_doc["schema"] == "splatsim.cameras/v1"
    assert {c["name"] for c in cameras_doc["cameras"]} == {"front_left", "front_right"}


def test_load_cameras_from_usdz_round_trip(tmp_path: Path) -> None:
    save_scene_usdz = _mod.save_scene_usdz
    ts = _make_minimal_tileset_with_glb(tmp_path)
    original = [_camera("c0"), _camera("c1")]
    original[1].extrinsics = _extrinsics(tx=42.0)
    out = tmp_path / "scene.usdz"
    save_scene_usdz(ts, out, cameras=original)

    recovered = load_cameras_from_usdz(out)
    assert len(recovered) == 2
    assert {c.name for c in recovered} == {"c0", "c1"}
    by_name = {c.name: c for c in recovered}
    assert by_name["c1"].extrinsics.translation == (42.0, 0.0, 1.8)
    assert by_name["c0"].intrinsics.distortion_model == "opencv"


def test_load_cameras_from_usdz_missing_raises(tmp_path: Path) -> None:
    save_scene_usdz = _mod.save_scene_usdz
    ts = _make_minimal_tileset_with_glb(tmp_path)
    out = tmp_path / "scene.usdz"
    save_scene_usdz(ts, out)  # no cameras=
    with pytest.raises(FileNotFoundError, match="no cameras.json"):
        load_cameras_from_usdz(out)


def test_cameras_and_extras_collision_rejected(tmp_path: Path) -> None:
    save_scene_usdz = _mod.save_scene_usdz
    ts = _make_minimal_tileset_with_glb(tmp_path)
    fake = tmp_path / "cameras.json"
    fake.write_text("{}")
    out = tmp_path / "scene.usdz"
    with pytest.raises(ValueError, match="cameras="):
        save_scene_usdz(
            ts,
            out,
            cameras=[_camera()],
            extras={"cameras.json": fake},
        )


def test_cli_cameras_flag(tmp_path: Path) -> None:
    cli = importlib.import_module("3dgs_io.scene_usdz_cli")
    ts = _make_minimal_tileset_with_glb(tmp_path)
    cams = [_camera("front_left"), _camera("front_right")]
    cams_doc = serialize_cameras(cams)
    cams_path = tmp_path / "cameras.json"
    cams_path.write_text(json.dumps(cams_doc))

    out = tmp_path / "scene.usdz"
    rc = cli.main([str(ts), str(out), "--cameras", str(cams_path), "--quiet"])
    assert rc == 0
    recovered = load_cameras_from_usdz(out)
    assert {c.name for c in recovered} == {"front_left", "front_right"}
