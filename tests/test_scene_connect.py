"""Tests for scene-USDZ connection (`3dgs_io.scene_connect`).

Connecting bundles is pure coordinate algebra through the geodetic
``ecef_anchor`` values, so the core invariant tested here is that the ECEF
position of every gaussian and every rig pose is preserved by the merge —
i.e. the output anchor absorbs both the inter-scene transforms and the
recentring shift. This is exactly the invariant an earlier external pipeline
broke (it subtracted MGRS map coordinates from the anchor, placing the
connected scene ~95 km away geodetically).
"""

from __future__ import annotations

import importlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import spz
from scipy.spatial import cKDTree

_mod = importlib.import_module("3dgs_io")
_geodesy = importlib.import_module("3dgs_io._geodesy")
_connect_cli = importlib.import_module("3dgs_io.scene_connect_cli")

RigPose = _mod.RigPose
RigTrajectory = _mod.RigTrajectory
Track = _mod.Track
TrackFrame = _mod.TrackFrame
connect_scene_usdzs = _mod.connect_scene_usdzs
load_scene_bundle = _mod.load_scene_bundle
save_gltf = _mod.save_gltf
save_scene_usdz = _mod.save_scene_usdz
extract_lidar_extension = _mod.extract_lidar_extension

RUB_TO_ENU = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)

_LAT0, _LON0, _H0 = 35.6, 139.78, 40.0


def _enu_anchor(lat: float, lon: float, h: float) -> np.ndarray:
    """Row-major 4×4 world(ENU)→ECEF anchor at a geodetic point."""
    anchor = np.eye(4, dtype=np.float64)
    anchor[:3, :3] = _geodesy.enu_to_ecef_rotation(lat, lon)
    anchor[:3, 3] = _geodesy.geodetic_to_ecef(lat, lon, h).reshape(3)
    return anchor


def _rotz(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    m = np.eye(4, dtype=np.float64)
    m[:2, :2] = [[c, -s], [s, c]]
    return m


def _translate(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[:3, 3] = (x, y, z)
    return m


def _make_cloud(n: int, seed: int) -> spz.GaussianCloud:
    rng = np.random.default_rng(seed)
    gc = spz.GaussianCloud()
    gc.antialiased = False
    gc.positions = rng.uniform(-20.0, 20.0, size=n * 3).astype(np.float32)
    quats = rng.standard_normal((n, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    gc.rotations = quats.reshape(-1)
    gc.scales = rng.uniform(-3.0, 0.5, size=n * 3).astype(np.float32)
    gc.alphas = rng.standard_normal(n).astype(np.float32)
    gc.colors = rng.uniform(0.0, 1.0, size=n * 3).astype(np.float32)
    gc.sh_degree = 0
    gc.sh = np.zeros(0, dtype=np.float32)
    return gc


def _tiny_map_osm(lat: float, lon: float) -> str:
    nodes = "".join(
        f'<node id="{i}" lat="{lat + i * 1e-5:.8f}" lon="{lon + i * 1e-5:.8f}"/>'
        for i in range(1, 4)
    )
    return f"<osm>{nodes}</osm>"


def _make_scene_usdz(
    tmp_path: Path,
    name: str,
    anchor_enu: np.ndarray,
    *,
    n: int = 200,
    seed: int = 0,
    ext: dict[str, np.ndarray] | None = None,
    rig_timestamps: tuple[int, ...] = (),
    rig_id: str = "ego",
    tracks: list[Track] | None = None,
    map_osm: str | None = None,
) -> tuple[Path, np.ndarray]:
    """Build one scene USDZ; returns ``(path, expected_world_positions)``."""
    scene_dir = tmp_path / name
    scene_dir.mkdir()
    cloud = _make_cloud(n, seed)
    save_gltf(cloud, scene_dir / "model.glb", ext_attributes=ext)

    # save_scene_usdz reconciles RUB payloads into ENU and strips that
    # conversion from the recorded anchor: root_matrix = source @ inv(S).
    source_to_world = np.eye(4)
    source_to_world[:3, :3] = RUB_TO_ENU
    source_root = anchor_enu @ source_to_world  # row-major
    doc = {
        "asset": {"version": "1.1", "generator": "test"},
        "geometricError": 100.0,
        "root": {
            "boundingVolume": {
                "box": [0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0]
            },
            "geometricError": 0,
            "refine": "ADD",
            "content": {"uri": "model.glb"},
            "transform": source_root.T.ravel().tolist(),  # column-major
        },
    }
    (scene_dir / "tileset.json").write_text(json.dumps(doc))

    rigs = None
    if rig_timestamps:
        rigs = [
            RigTrajectory(
                rig_id=rig_id,
                poses=[
                    RigPose(
                        timestamp_us=ts,
                        translation=(float(i), 0.0, 0.0),
                        rotation=(0.0, 0.0, 0.0, 1.0),
                    )
                    for i, ts in enumerate(rig_timestamps)
                ],
            )
        ]

    extras = None
    if map_osm is not None:
        osm_path = scene_dir / "map.osm"
        osm_path.write_text(map_osm)
        extras = {"map.osm": osm_path}

    out = tmp_path / f"{name}.usdz"
    save_scene_usdz(
        scene_dir / "tileset.json",
        out,
        extras=extras,
        rig_trajectories=rigs,
        tracks=tracks,
    )
    positions = np.array(cloud.positions, dtype=np.float64).reshape(n, 3)
    expected_world = positions @ RUB_TO_ENU.T
    return out, expected_world


def _world_to_ecef(anchor: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ anchor[:3, :3].T + anchor[:3, 3]


def _load_out(out: Path) -> tuple[dict, np.ndarray]:
    """Read scene.json and the concatenated world positions of an output USDZ."""
    bundle = load_scene_bundle(out)
    n = bundle.cloud.num_points
    pos = np.array(bundle.cloud.positions, dtype=np.float64).reshape(n, 3)
    with zipfile.ZipFile(out) as zf:
        scene = json.loads(zf.read("scene.json"))
    return scene, pos


# ---------------------------------------------------------------------------
# Core geometry / anchor invariant
# ---------------------------------------------------------------------------


def test_connect_preserves_ecef_positions_and_anchor(tmp_path: Path) -> None:
    anchor_a = _enu_anchor(_LAT0, _LON0, _H0)
    # Scene B: 120 m away and yawed 25° relative to scene A's world frame.
    offset = _translate(120.0, -40.0, 1.5) @ _rotz(25.0)
    anchor_b = anchor_a @ offset

    usdz_a, world_a = _make_scene_usdz(
        tmp_path, "a", anchor_a, seed=1, map_osm=_tiny_map_osm(_LAT0, _LON0)
    )
    usdz_b, world_b = _make_scene_usdz(tmp_path, "b", anchor_b, seed=2)

    out = tmp_path / "connected.usdz"
    result = connect_scene_usdzs([usdz_a, usdz_b], out)
    assert result.n_gaussians == len(world_a) + len(world_b)

    scene, pos_out = _load_out(out)
    anchor_out = np.asarray(scene["world"]["ecef_anchor"], dtype=np.float64)

    # Every input gaussian keeps its ECEF position (nearest-match, since
    # chunking reorders points). SPZ quantises at ~0.25 mm.
    ecef_in = np.concatenate([_world_to_ecef(anchor_a, world_a), _world_to_ecef(anchor_b, world_b)])
    ecef_out = _world_to_ecef(anchor_out, pos_out)
    dist, _ = cKDTree(ecef_out).query(ecef_in)
    assert dist.max() < 5e-3

    # The merged cloud is recentred: its bbox centre sits at the origin.
    centre = (pos_out.min(axis=0) + pos_out.max(axis=0)) / 2.0
    assert np.abs(centre).max() < 1e-3

    # Output anchor decodes to (approximately) the input scenes' location,
    # not an MGRS grid corner 95 km away.
    lat, lon, _h = _geodesy.ecef_to_geodetic(anchor_out[:3, 3])
    assert abs(lat - _LAT0) < 0.01
    assert abs(lon - _LON0) < 0.01

    # Reference extras (map.osm) are carried into the output.
    with zipfile.ZipFile(out) as zf:
        assert "map.osm" in zf.namelist()

    assert scene["producer"]["source_scenes"][0]["path"] == "a.usdz"


def test_connect_merges_rig_poses_in_timestamp_order(tmp_path: Path) -> None:
    anchor_a = _enu_anchor(_LAT0, _LON0, _H0)
    offset = _translate(80.0, 0.0, 0.0)
    anchor_b = anchor_a @ offset

    usdz_a, _ = _make_scene_usdz(tmp_path, "a", anchor_a, seed=1, rig_timestamps=(100, 300))
    usdz_b, _ = _make_scene_usdz(tmp_path, "b", anchor_b, seed=2, rig_timestamps=(200, 400))

    out = tmp_path / "connected.usdz"
    connect_scene_usdzs([usdz_a, usdz_b], out)

    bundle = load_scene_bundle(out)
    assert len(bundle.rigs) == 1
    rig = bundle.rigs[0]
    assert [p.timestamp_us for p in rig.poses] == [100, 200, 300, 400]

    # Rig poses ride the same anchor math as the gaussians: their ECEF
    # positions are preserved.
    anchor_out = bundle.ecef_anchor
    pose_b0 = next(p for p in rig.poses if p.timestamp_us == 200)
    ecef_out = _world_to_ecef(anchor_out, np.asarray(pose_b0.translation))
    ecef_in = _world_to_ecef(anchor_b, np.array([0.0, 0.0, 0.0]))
    assert np.linalg.norm(ecef_out - ecef_in) < 1e-6


def test_connect_transforms_track_frames(tmp_path: Path) -> None:
    anchor_a = _enu_anchor(_LAT0, _LON0, _H0)
    anchor_b = anchor_a @ _translate(50.0, 0.0, 0.0)
    track = Track(
        track_id="car_1",
        class_name="vehicle.car",
        size=(4.0, 2.0, 1.5),
        frames=[TrackFrame(timestamp_us=10, translation=(1.0, 2.0, 0.0), rotation=(0, 0, 0, 1))],
    )

    usdz_a, _ = _make_scene_usdz(tmp_path, "a", anchor_a, seed=1, tracks=[track])
    usdz_b, _ = _make_scene_usdz(tmp_path, "b", anchor_b, seed=2, tracks=[track])

    out = tmp_path / "connected.usdz"
    connect_scene_usdzs([usdz_a, usdz_b], out)
    bundle = load_scene_bundle(out)
    assert len(bundle.tracks) == 2
    # Duplicate ids are disambiguated per scene.
    assert {t.track_id for t in bundle.tracks} == {"car_1", "scene1/car_1"}
    frame_a = next(t for t in bundle.tracks if t.track_id == "car_1").frames[0]
    frame_b = next(t for t in bundle.tracks if t.track_id != "car_1").frames[0]
    delta = np.asarray(frame_b.translation) - np.asarray(frame_a.translation)
    np.testing.assert_allclose(delta, [50.0, 0.0, 0.0], atol=1e-6)


# ---------------------------------------------------------------------------
# LiDAR ext attributes
# ---------------------------------------------------------------------------


def _lidar_ext(n: int, *, mask: bool) -> dict[str, np.ndarray]:
    ext = {
        "lidar_intensity_raw": np.linspace(-1.5, 1.5, n).astype(np.float32),
        "lidar_raydrop_logit": np.linspace(-1.0, 1.0, n).astype(np.float32),
    }
    if mask:
        ext["lidar_mask"] = (np.arange(n) % 2).astype(np.float32)
    return ext


def test_connect_fills_missing_lidar_mask_with_ones(tmp_path: Path) -> None:
    anchor_a = _enu_anchor(_LAT0, _LON0, _H0)
    anchor_b = anchor_a @ _translate(60.0, 0.0, 0.0)
    n = 64
    usdz_a, _ = _make_scene_usdz(tmp_path, "a", anchor_a, n=n, seed=1, ext=_lidar_ext(n, mask=True))
    usdz_b, _ = _make_scene_usdz(
        tmp_path, "b", anchor_b, n=n, seed=2, ext=_lidar_ext(n, mask=False)
    )

    out = tmp_path / "connected.usdz"
    connect_scene_usdzs([usdz_a, usdz_b], out)

    bundle = load_scene_bundle(out)
    mask = bundle.ext_attrs["lidar_mask"]
    assert mask.shape == (2 * n,)
    # Scene A contributed n/2 zero-mask gaussians; scene B (mask-less) must
    # come through as all-participating, NOT zero-filled.
    assert int((mask > 0.5).sum()) == n // 2 + n


def test_connect_rejects_mixed_lidar_and_non_lidar_scenes(tmp_path: Path) -> None:
    anchor_a = _enu_anchor(_LAT0, _LON0, _H0)
    anchor_b = anchor_a @ _translate(60.0, 0.0, 0.0)
    n = 32
    usdz_a, _ = _make_scene_usdz(
        tmp_path, "a", anchor_a, n=n, seed=1, ext=_lidar_ext(n, mask=False)
    )
    usdz_b, _ = _make_scene_usdz(tmp_path, "b", anchor_b, n=n, seed=2)

    with pytest.raises(ValueError, match="no LiDAR ext attributes"):
        connect_scene_usdzs([usdz_a, usdz_b], tmp_path / "connected.usdz")


def test_connected_chunks_carry_embedded_lidar_records(tmp_path: Path) -> None:
    anchor_a = _enu_anchor(_LAT0, _LON0, _H0)
    anchor_b = anchor_a @ _translate(60.0, 0.0, 0.0)
    n = 64
    usdz_a, _ = _make_scene_usdz(
        tmp_path, "a", anchor_a, n=n, seed=1, ext=_lidar_ext(n, mask=False)
    )
    usdz_b, _ = _make_scene_usdz(
        tmp_path, "b", anchor_b, n=n, seed=2, ext=_lidar_ext(n, mask=False)
    )
    out = tmp_path / "connected.usdz"
    connect_scene_usdzs([usdz_a, usdz_b], out)

    with zipfile.ZipFile(out) as zf:
        scene = json.loads(zf.read("scene.json"))
        assert scene["gaussians"]["ext_attributes"]["container"] == "spz_extension"
        for entry in scene["gaussians"]["chunks"]:
            ext = extract_lidar_extension(zf.read(entry["uri"]))
            assert ext is not None
            assert ext["lidar_intensity_raw"].shape == (entry["n_points"],)


# ---------------------------------------------------------------------------
# v2 (sidecar) inputs
# ---------------------------------------------------------------------------


def _make_v2_scene_usdz(
    tmp_path: Path,
    name: str,
    anchor_enu: np.ndarray,
    *,
    n: int = 64,
    seed: int = 0,
    opaque: bool = False,
) -> tuple[Path, np.ndarray]:
    """Hand-craft a legacy ``splatsim.scene/v2`` bundle (sidecar LiDAR files)."""
    spz_io = importlib.import_module("3dgs_io.spz_io")
    ext_mod = importlib.import_module("3dgs_io.ext_attributes")
    frame_mod = importlib.import_module("3dgs_io.frame_convention")

    cloud = _make_cloud(n, seed)
    if opaque:
        cloud.alphas = np.full(n, 50.0, dtype=np.float32)  # sigmoid → 1.0 → u8 255
    world = np.array(cloud.positions, dtype=np.float64).reshape(n, 3)  # stored verbatim

    chunk_path = tmp_path / f"{name}_chunk.spz"
    spz_io.save_spz_world(cloud, chunk_path)
    sidecar = ext_mod.encode_lidar_extension(_lidar_ext(n, mask=False), count=n)

    scene = {
        "schema": "splatsim.scene/v2",
        "producer": {"tool": "test", "tool_version": "0"},
        "world": {
            "frame_convention": frame_mod.FRAME_CONVENTION,
            "ecef_anchor": anchor_enu.tolist(),
        },
        "gaussians": {
            "tileset": "tileset.json",
            "n_gaussians": n,
            "sh_degree": 0,
            "ext_attributes": {
                "extension": "EXT_gaussian_lidar",
                "sidecar_suffix": ".lidar",
                "attributes": ["lidar_intensity_raw", "lidar_raydrop_logit"],
            },
            "frame": "world",
        },
        "extras": {},
        "render_defaults": {"exposure": 1.0, "near_plane": 0.1, "far_plane": 100.0},
    }
    out = tmp_path / f"{name}.usdz"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("scene.json", json.dumps(scene))
        zf.writestr("tileset.json", "{}")
        zf.write(chunk_path, "chunks/chunk_000000.spz")
        zf.writestr("chunks/chunk_000000.lidar", sidecar)
    return out, world


def test_connect_reads_v2_sidecar_bundles(tmp_path: Path) -> None:
    anchor_a = _enu_anchor(_LAT0, _LON0, _H0)
    anchor_b = anchor_a @ _translate(60.0, 0.0, 0.0)
    n = 64
    usdz_a, world_a = _make_v2_scene_usdz(tmp_path, "a", anchor_a, n=n, seed=1)
    usdz_b, world_b = _make_v2_scene_usdz(tmp_path, "b", anchor_b, n=n, seed=2)

    out = tmp_path / "connected.usdz"
    result = connect_scene_usdzs([usdz_a, usdz_b], out)
    assert result.n_gaussians == 2 * n

    scene, pos_out = _load_out(out)
    assert scene["schema"] == "splatsim.scene/v3"
    anchor_out = np.asarray(scene["world"]["ecef_anchor"], dtype=np.float64)
    ecef_in = np.concatenate([_world_to_ecef(anchor_a, world_a), _world_to_ecef(anchor_b, world_b)])
    dist, _ = cKDTree(_world_to_ecef(anchor_out, pos_out)).query(ecef_in)
    assert dist.max() < 5e-3

    bundle = load_scene_bundle(out)
    assert bundle.ext_attrs["lidar_intensity_raw"].shape == (2 * n,)
    # v2's Cesium tileset.json is not carried into the v3 output.
    with zipfile.ZipFile(out) as zf:
        assert "tileset.json" not in zf.namelist()


# ---------------------------------------------------------------------------
# Guard rails
# ---------------------------------------------------------------------------


def test_fully_opaque_gaussians_survive(tmp_path: Path) -> None:
    """SPZ decodes opacity 255 to a +inf logit; such points must not be dropped."""
    anchor_a = _enu_anchor(_LAT0, _LON0, _H0)
    anchor_b = anchor_a @ _translate(60.0, 0.0, 0.0)
    n = 64
    usdz_a, _ = _make_v2_scene_usdz(tmp_path, "a", anchor_a, n=n, seed=1, opaque=True)
    usdz_b, _ = _make_v2_scene_usdz(tmp_path, "b", anchor_b, n=n, seed=2, opaque=True)

    # The v2 inputs round-trip through SPZ's uint8 sigmoid quantiser, so the
    # loaded alphas contain +inf logits.
    bundle = load_scene_bundle(usdz_a)
    alphas = np.array(bundle.cloud.alphas, dtype=np.float32)
    assert np.isposinf(alphas).all()

    out = tmp_path / "connected.usdz"
    result = connect_scene_usdzs([usdz_a, usdz_b], out)
    assert result.n_gaussians == 2 * n


def test_connect_needs_two_inputs(tmp_path: Path) -> None:
    anchor = _enu_anchor(_LAT0, _LON0, _H0)
    usdz_a, _ = _make_scene_usdz(tmp_path, "a", anchor, seed=1)
    with pytest.raises(ValueError, match="at least two"):
        connect_scene_usdzs([usdz_a], tmp_path / "out.usdz")


def test_connect_rejects_scenes_beyond_spz_range(tmp_path: Path) -> None:
    anchor_a = _enu_anchor(_LAT0, _LON0, _H0)
    anchor_b = anchor_a @ _translate(9000.0, 0.0, 0.0)  # ±4.5 km after recentring
    usdz_a, _ = _make_scene_usdz(tmp_path, "a", anchor_a, seed=1)
    usdz_b, _ = _make_scene_usdz(tmp_path, "b", anchor_b, seed=2)
    with pytest.raises(ValueError, match="SPZ fixed-point range"):
        connect_scene_usdzs([usdz_a, usdz_b], tmp_path / "out.usdz")


def test_writer_rejects_anchor_inconsistent_with_map(tmp_path: Path) -> None:
    """Regression: the incident bundle's anchor decoded ~95 km from its map."""
    anchor = _enu_anchor(_LAT0, _LON0, _H0)
    far_map = _tiny_map_osm(_LAT0 + 0.9, _LON0 + 0.9)  # ~130 km away
    with pytest.raises(ValueError, match="inconsistent with the embedded Lanelet2 map"):
        _make_scene_usdz(tmp_path, "bad", anchor, seed=1, map_osm=far_map)


def test_writer_skips_geo_check_for_placeholder_map(tmp_path: Path) -> None:
    anchor = _enu_anchor(_LAT0, _LON0, _H0)
    usdz, _ = _make_scene_usdz(tmp_path, "a", anchor, seed=1, map_osm="<osm/>")
    assert usdz.exists()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_connects_two_bundles(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    anchor_a = _enu_anchor(_LAT0, _LON0, _H0)
    anchor_b = anchor_a @ _translate(70.0, 10.0, 0.0)
    usdz_a, _ = _make_scene_usdz(tmp_path, "a", anchor_a, seed=1)
    usdz_b, _ = _make_scene_usdz(tmp_path, "b", anchor_b, seed=2)
    out = tmp_path / "connected.usdz"

    rc = _connect_cli.main([str(usdz_a), str(usdz_b), "-o", str(out)])
    assert rc == 0
    assert out.exists()
    summary = json.loads(capsys.readouterr().out)
    assert summary["n_gaussians"] == 400
