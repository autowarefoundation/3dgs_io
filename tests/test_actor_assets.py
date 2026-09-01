"""Tests for the rigid dynamic-object ("actor") asset format.

``splatsim.actor_assets/v1`` adds the appearance half of a dynamic object to a
scene bundle: object-local Gaussian clouds plus the bindings that say which
track renders with which asset. The invariants worth defending are the ones a
consumer cannot recover on its own — that the object frame is declared and
checked, that an asset left in world coordinates is rejected at write time
rather than at render time, and that instancing an asset through its track's
pose reproduces the geometry it was cut from.
"""

from __future__ import annotations

import importlib
import json
import math
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import spz
from scipy.spatial.transform import Rotation

_mod = importlib.import_module("3dgs_io")
_actor = importlib.import_module("3dgs_io.actor_assets")
_edit_cli = importlib.import_module("3dgs_io.edit_usdz_cli")

ActorAsset = _mod.ActorAsset
ActorAssetBank = _mod.ActorAssetBank
ActorAssetSource = _mod.ActorAssetSource
ActorInstance = _mod.ActorInstance
Track = _mod.Track
TrackFrame = _mod.TrackFrame
add_actor_assets_to_usdz = _mod.add_actor_assets_to_usdz
build_actor_asset_bank = _mod.build_actor_asset_bank
decode_actor_asset = _mod.decode_actor_asset
encode_actor_asset = _mod.encode_actor_asset
extract_actor_asset = _mod.extract_actor_asset
load_actor_asset_dir = _mod.load_actor_asset_dir
parse_actor_assets = _mod.parse_actor_assets
rotate_sh_about_z = _mod.rotate_sh_about_z
save_actor_asset_dir = _mod.save_actor_asset_dir
save_scene_usdz = _mod.save_scene_usdz
serialize_actor_assets = _mod.serialize_actor_assets
validate_instances_against_tracks = _mod.validate_instances_against_tracks

CAR_SIZE = (4.5, 1.9, 1.5)

#: The ``make_minimal_tileset_with_glb`` conftest fixture, as a callable.
MakeTileset = Callable[[Path], Path]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _cloud(
    positions: np.ndarray,
    *,
    seed: int = 0,
    sh_degree: int = 0,
) -> spz.GaussianCloud:
    rng = np.random.default_rng(seed)
    n = len(positions)
    gc = spz.GaussianCloud()
    gc.antialiased = False
    gc.positions = np.ascontiguousarray(positions, dtype=np.float32).reshape(-1)
    quats = rng.standard_normal((n, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    gc.rotations = quats.astype(np.float32).reshape(-1)
    gc.scales = rng.uniform(-3.0, 0.0, size=n * 3).astype(np.float32)
    gc.alphas = rng.standard_normal(n).astype(np.float32)
    gc.colors = rng.uniform(0.0, 1.0, size=n * 3).astype(np.float32)
    per_ch = (sh_degree + 1) ** 2 - 1
    if per_ch:
        gc.sh_degree = sh_degree
        gc.sh = rng.standard_normal(n * per_ch * 3).astype(np.float32)
    else:
        gc.sh_degree = 0
        gc.sh = np.zeros(0, dtype=np.float32)
    return gc


def _object_local_cloud(n: int = 120, *, seed: int = 0, sh_degree: int = 0) -> spz.GaussianCloud:
    """A car-shaped cloud already in the object frame (centred on the origin)."""
    rng = np.random.default_rng(seed + 100)
    pts = (rng.random((n, 3)) - 0.5) * np.asarray(CAR_SIZE)
    return _cloud(pts, seed=seed, sh_degree=sh_degree)


def _source(
    asset_id: str = "sedan_0007",
    *,
    n: int = 120,
    seed: int = 0,
    with_lidar: bool = False,
    sh_degree: int = 0,
) -> ActorAssetSource:
    rng = np.random.default_rng(seed + 7)
    ext: dict[str, np.ndarray] = {}
    if with_lidar:
        ext = {
            "lidar_intensity_raw": rng.standard_normal(n).astype(np.float32),
            "lidar_raydrop_logit": rng.standard_normal(n).astype(np.float32),
        }
    return ActorAssetSource(
        asset_id=asset_id,
        cloud=_object_local_cloud(n, seed=seed, sh_degree=sh_degree),
        class_name="automobile",
        size=CAR_SIZE,
        ext_attrs=ext,
    )


def _track(track_id: str = "100", size: tuple[float, float, float] = CAR_SIZE) -> Track:
    return Track(
        track_id=track_id,
        class_name="automobile",
        size=size,
        frames=[
            TrackFrame(
                timestamp_us=1_000_000,
                translation=(1.0, 2.0, 0.75),
                rotation=(0.0, 0.0, 0.0, 1.0),
            ),
            TrackFrame(
                timestamp_us=1_100_000,
                translation=(3.0, 2.0, 0.75),
                rotation=(0.0, 0.0, 0.0, 1.0),
            ),
        ],
    )


def _asset(**overrides: object) -> ActorAsset:
    kwargs: dict = {
        "asset_id": "sedan_0007",
        "class_name": "automobile",
        "size": CAR_SIZE,
        "bbox_min": (-2.25, -0.95, -0.75),
        "bbox_max": (2.25, 0.95, 0.75),
        "n_points": 120,
    }
    kwargs.update(overrides)
    return ActorAsset(**kwargs)


# ---------------------------------------------------------------------------
# Document schema
# ---------------------------------------------------------------------------


def test_document_round_trips_through_serialize_and_parse() -> None:
    bank = ActorAssetBank(
        assets=[_asset(sh_degree=3, provenance={"clip": "odaibatest5"})],
        instances=[ActorInstance(track_id="100", asset_id="sedan_0007", fit_mode="uniform")],
    )
    doc = serialize_actor_assets(bank)
    assert doc["schema"] == "splatsim.actor_assets/v1"
    assert doc["frame"] == "object"
    assert doc["object_frame"] == _actor.OBJECT_FRAME_CONVENTION

    back = parse_actor_assets(json.loads(json.dumps(doc)))
    assert [a.to_dict() for a in back.assets] == [a.to_dict() for a in bank.assets]
    assert [i.to_dict() for i in back.instances] == [i.to_dict() for i in bank.instances]


def test_uri_defaults_to_the_canonical_archive_path() -> None:
    assert _asset().uri == "actor_assets/sedan_0007/asset.spz"


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        pytest.param(
            {"schema": "splatsim.actor_assets/v2"},
            "unexpected actor_assets schema",
            id="foreign-schema",
        ),
        pytest.param({"frame": "world"}, "frame must be 'object'", id="world-frame"),
        pytest.param(
            {"object_frame": {**_actor.OBJECT_FRAME_CONVENTION, "forward": "+y"}},
            "unsupported object_frame",
            id="rotated-object-frame",
        ),
        # A NuRec-style unit-scale bank must declare itself, not slip through.
        pytest.param(
            {"object_frame": {**_actor.OBJECT_FRAME_CONVENTION, "scale": "normalized"}},
            "unsupported object_frame",
            id="unit-normalised",
        ),
    ],
)
def test_parse_rejects_a_foreign_contract(mutate: dict, match: str) -> None:
    doc = serialize_actor_assets(ActorAssetBank(assets=[_asset()])) | mutate
    with pytest.raises(ValueError, match=match):
        parse_actor_assets(doc)


# ---------------------------------------------------------------------------
# Per-asset validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_id", ["", "../etc/passwd", "a/b", "sedan 7", ".hidden", "x" * 200])
def test_asset_id_must_be_a_safe_path_component(bad_id: str) -> None:
    with pytest.raises(ValueError, match="invalid asset_id"):
        _asset(asset_id=bad_id)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        pytest.param({"motion": "deformable"}, "unsupported motion model", id="motion"),
        pytest.param({"size": (4.5, 0.0, 1.5)}, "size must be positive", id="flat-size"),
        pytest.param(
            {"bbox_min": (1.0, -0.95, -0.75), "bbox_max": (-1.0, 0.95, 0.75)},
            "min .* above max",
            id="inverted-bbox",
        ),
        # The classic bug: the cloud was never brought into the object frame.
        pytest.param(
            {"bbox_min": (111.0, -59.0, 1.1), "bbox_max": (116.0, -57.0, 2.6)},
            "does not contain the object-local origin",
            id="world-coordinates",
        ),
        pytest.param(
            {"uri": "actor_assets/somewhere/else.spz"}, "uri must be", id="hand-edited-uri"
        ),
        pytest.param({"sh_degree": 4}, "sh_degree must be in 0..3", id="sh-degree"),
    ],
)
def test_asset_rejects_an_invalid_record(overrides: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _asset(**overrides)


def test_instance_rejects_an_unknown_fit_mode() -> None:
    with pytest.raises(ValueError, match="unknown fit_mode"):
        ActorInstance(track_id="100", asset_id="sedan_0007", fit_mode="squash")


def test_bank_rejects_duplicate_asset_ids() -> None:
    with pytest.raises(ValueError, match="duplicate asset_id"):
        serialize_actor_assets(ActorAssetBank(assets=[_asset(), _asset()]))


def test_bank_rejects_a_track_bound_to_two_assets() -> None:
    bank = ActorAssetBank(
        assets=[_asset(), _asset(asset_id="truck_0001")],
        instances=[
            ActorInstance(track_id="100", asset_id="sedan_0007"),
            ActorInstance(track_id="100", asset_id="truck_0001"),
        ],
    )
    with pytest.raises(ValueError, match="bound to more than one asset"):
        serialize_actor_assets(bank)


def test_bank_rejects_a_dangling_asset_reference() -> None:
    bank = ActorAssetBank(
        assets=[_asset()],
        instances=[ActorInstance(track_id="100", asset_id="not_packed")],
    )
    with pytest.raises(ValueError, match="unknown asset_id"):
        serialize_actor_assets(bank)


# ---------------------------------------------------------------------------
# Binding against tracks
# ---------------------------------------------------------------------------


def test_bindings_must_name_a_track_that_exists() -> None:
    bank = ActorAssetBank(
        assets=[_asset()], instances=[ActorInstance(track_id="999", asset_id="sedan_0007")]
    )
    with pytest.raises(ValueError, match="not among the bundle's sequence tracks"):
        validate_instances_against_tracks(bank, [_track("100")])


def test_rigid_binding_warns_when_the_asset_does_not_match_the_track_box(
    caplog: pytest.LogCaptureFixture,
) -> None:
    bank = ActorAssetBank(
        assets=[_asset()], instances=[ActorInstance(track_id="100", asset_id="sedan_0007")]
    )
    with caplog.at_level("WARNING"):
        validate_instances_against_tracks(bank, [_track("100", size=(12.0, 2.5, 3.8))])
    assert "rigid instancing keeps the asset's own size" in caplog.text


def test_one_asset_can_back_many_tracks() -> None:
    bank = ActorAssetBank(
        assets=[_asset()],
        instances=[
            ActorInstance(track_id="100", asset_id="sedan_0007"),
            ActorInstance(track_id="101", asset_id="sedan_0007"),
        ],
    )
    validate_instances_against_tracks(bank, [_track("100"), _track("101")])
    assert bank.asset_for_track("101").asset_id == "sedan_0007"
    assert bank.asset_for_track("777") is None


# ---------------------------------------------------------------------------
# Payload encode / decode
# ---------------------------------------------------------------------------


def test_encode_measures_the_bbox_and_point_count_from_the_cloud() -> None:
    asset, payload = encode_actor_asset(_source(n=64))
    assert asset.n_points == 64
    assert all(lo <= 0.0 <= hi for lo, hi in zip(asset.bbox_min, asset.bbox_max, strict=True))
    assert payload[:4] == b"NGSP"


def test_payload_round_trips_including_lidar_attributes() -> None:
    source = _source(with_lidar=True, n=80)
    asset, payload = encode_actor_asset(source)
    assert asset.ext_attributes is not None
    assert asset.ext_attributes["spz_extension_type"] == "0x54340001"

    cloud, ext = decode_actor_asset(asset, payload)
    assert cloud.num_points == 80
    assert sorted(ext) == ["lidar_intensity_raw", "lidar_raydrop_logit"]
    np.testing.assert_allclose(
        np.asarray(cloud.positions).reshape(80, 3),
        np.asarray(source.cloud.positions).reshape(80, 3),
        atol=1e-3,
    )


def test_encode_rejects_ext_attributes_of_the_wrong_length() -> None:
    source = _source(n=40)
    source.ext_attrs = {"lidar_intensity_raw": np.zeros(7, dtype=np.float32)}
    with pytest.raises(ValueError, match="has 7 entries for 40 gaussians"):
        encode_actor_asset(source)


def test_decode_rejects_a_payload_that_disagrees_with_the_index() -> None:
    asset, payload = encode_actor_asset(_source(n=40))
    asset.n_points = 41
    with pytest.raises(ValueError, match="index says 41"):
        decode_actor_asset(asset, payload)


def test_verify_rejects_a_payload_whose_sh_degree_disagrees() -> None:
    asset, payload = encode_actor_asset(_source(n=40, sh_degree=3))
    asset.sh_degree = 0
    with pytest.raises(ValueError, match="index says 0"):
        _actor.verify_actor_payload(asset, payload)


def test_verify_rejects_an_undeclared_extension_record() -> None:
    asset, payload = encode_actor_asset(_source(n=40, with_lidar=True))
    asset.ext_attributes = None
    with pytest.raises(ValueError, match="index does not declare"):
        _actor.verify_actor_payload(asset, payload)


def test_build_bank_rejects_duplicate_sources() -> None:
    with pytest.raises(ValueError, match="duplicate asset_id"):
        build_actor_asset_bank([_source(), _source()])


# ---------------------------------------------------------------------------
# Standalone bank directories
# ---------------------------------------------------------------------------


def test_resolve_is_a_no_op_without_actor_arguments() -> None:
    """Writers pass their kwargs straight through, including the absent case."""
    bank, payloads = _actor.resolve_actor_asset_bank(None, None)
    assert bank is None
    assert payloads == {}


def test_resolve_lets_explicit_instances_override_a_bank_directory(tmp_path: Path) -> None:
    bank, payloads = build_actor_asset_bank(
        [_source()], [ActorInstance(track_id="100", asset_id="sedan_0007")]
    )
    save_actor_asset_dir(tmp_path / "bank", bank, payloads)

    resolved, resolved_payloads = _actor.resolve_actor_asset_bank(
        tmp_path / "bank", [ActorInstance(track_id="200", asset_id="sedan_0007")]
    )
    assert resolved is not None
    assert [i.track_id for i in resolved.instances] == ["200"]
    assert resolved_payloads == payloads


def test_bank_directory_round_trip(tmp_path: Path) -> None:
    bank, payloads = build_actor_asset_bank(
        [_source(with_lidar=True)], [ActorInstance(track_id="100", asset_id="sedan_0007")]
    )
    save_actor_asset_dir(tmp_path / "bank", bank, payloads)
    assert (tmp_path / "bank" / "actor_assets.json").is_file()
    assert (tmp_path / "bank" / "actor_assets" / "sedan_0007" / "asset.spz").is_file()

    loaded_bank, loaded_payloads = load_actor_asset_dir(tmp_path / "bank")
    assert loaded_payloads == payloads
    assert loaded_bank.instances[0].track_id == "100"


def test_bank_directory_reports_a_missing_payload(tmp_path: Path) -> None:
    bank, payloads = build_actor_asset_bank([_source()])
    save_actor_asset_dir(tmp_path / "bank", bank, payloads)
    (tmp_path / "bank" / "actor_assets" / "sedan_0007" / "asset.spz").unlink()
    with pytest.raises(FileNotFoundError, match="missing from"):
        load_actor_asset_dir(tmp_path / "bank")


# ---------------------------------------------------------------------------
# Cutting an asset out of a world-frame cloud
# ---------------------------------------------------------------------------


def _world_car(
    yaw_deg: float, centre: tuple[float, float, float], n: int = 150, *, sh_degree: int = 0
) -> tuple[spz.GaussianCloud, np.ndarray, Rotation, np.ndarray]:
    """A car's gaussians placed in world coordinates, plus far-away background."""
    rng = np.random.default_rng(11)
    rot = Rotation.from_euler("z", math.radians(yaw_deg))
    t = np.asarray(centre, dtype=np.float64)
    local = (rng.random((n, 3)) - 0.5) * np.asarray(CAR_SIZE)
    background = rng.uniform(30.0, 60.0, size=(50, 3))
    pts = np.concatenate([rot.apply(local) + t, background])
    return _cloud(pts, seed=11, sh_degree=sh_degree), local, rot, t


def _extract(cloud: spz.GaussianCloud, rot: Rotation, t: np.ndarray, **kwargs: object):
    """`extract_actor_asset` with the boilerplate the tests never vary."""
    kwargs.setdefault("size", CAR_SIZE)
    return extract_actor_asset(
        cloud,
        asset_id="sedan_0007",
        class_name="automobile",
        translation=tuple(t),
        rotation=tuple(rot.as_quat()),
        **kwargs,  # ty: ignore[missing-argument]
    )


def test_extract_moves_the_box_centre_onto_the_object_origin() -> None:
    cloud, local, rot, t = _world_car(37.0, (113.6, -58.5, 1.92))
    source = _extract(cloud, rot, t)
    assert source.cloud.num_points == len(local)
    got = np.asarray(source.cloud.positions, dtype=np.float64).reshape(-1, 3)
    np.testing.assert_allclose(got, local, atol=1e-5)
    # The whole point of the object frame: the result is centred, so the
    # index-level "still in world coordinates" check passes.
    encode_actor_asset(source)


def test_extract_reproduces_the_world_cloud_when_instanced_back() -> None:
    cloud, _local, rot, t = _world_car(-104.0, (5.0, -7.0, 0.8))
    source = _extract(cloud, rot, t)
    n = source.cloud.num_points
    local = np.asarray(source.cloud.positions, dtype=np.float64).reshape(n, 3)
    world_again = rot.apply(local) + t

    original = np.asarray(cloud.positions, dtype=np.float64).reshape(-1, 3)[:n]
    np.testing.assert_allclose(world_again, original, atol=1e-4)

    # Orientation quaternions follow the same rigid transform.
    local_q = np.asarray(source.cloud.rotations, dtype=np.float64).reshape(n, 4)
    world_q = (rot * Rotation.from_quat(local_q)).as_quat()
    original_q = np.asarray(cloud.rotations, dtype=np.float64).reshape(-1, 4)[:n]
    dots = np.abs(np.sum(world_q * original_q, axis=1))  # q and -q are the same rotation
    np.testing.assert_allclose(dots, 1.0, atol=1e-5)


def test_extract_leaves_appearance_parameters_untouched() -> None:
    cloud, local, rot, t = _world_car(0.0, (0.0, 0.0, 0.0))
    source = _extract(cloud, rot, t)
    n = len(local)
    for name in ("colors", "scales", "alphas"):
        np.testing.assert_array_equal(
            np.asarray(getattr(source.cloud, name)).reshape(n, -1),
            np.asarray(getattr(cloud, name)).reshape(-1, 1 if name == "alphas" else 3)[:n],
        )


def test_extract_keeps_ext_attributes_aligned_with_their_gaussians() -> None:
    cloud, local, rot, t = _world_car(15.0, (2.0, 3.0, 0.7))
    total = cloud.num_points
    intensity = np.arange(total, dtype=np.float32)
    source = _extract(cloud, rot, t, ext_attrs={"lidar_intensity_raw": intensity})
    # The car's gaussians are the first block of the cloud, so the selected
    # attribute values are exactly their indices.
    np.testing.assert_array_equal(
        source.ext_attrs["lidar_intensity_raw"], np.arange(len(local), dtype=np.float32)
    )


def test_extract_margin_widens_the_selection() -> None:
    cloud, local, rot, t = _world_car(0.0, (0.0, 0.0, 0.0))
    tight = _extract(cloud, rot, t, size=(1.0, 1.0, 1.0))
    wide = _extract(cloud, rot, t, size=(1.0, 1.0, 1.0), margin=2.0)
    assert wide.cloud.num_points > tight.cloud.num_points
    assert wide.cloud.num_points <= len(local)


def test_extract_reports_an_empty_box() -> None:
    cloud, _local, rot, _t = _world_car(0.0, (0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="no gaussians inside the box"):
        _extract(cloud, rot, np.array([500.0, 500.0, 500.0]))


# ---------------------------------------------------------------------------
# View-dependent bands
# ---------------------------------------------------------------------------

_SH_C1 = 0.4886025119029199
_SH_C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
_SH_C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)


def _sh_basis(d: np.ndarray) -> np.ndarray:
    """The 3DGS real-SH basis for bands 1..3 (band 0 lives in ``colors``)."""
    x, y, z = d
    return np.array(
        [
            _SH_C1 * -y,
            _SH_C1 * z,
            _SH_C1 * -x,
            _SH_C2[0] * x * y,
            _SH_C2[1] * y * z,
            _SH_C2[2] * (2 * z * z - x * x - y * y),
            _SH_C2[3] * x * z,
            _SH_C2[4] * (x * x - y * y),
            _SH_C3[0] * y * (3 * x * x - y * y),
            _SH_C3[1] * x * y * z,
            _SH_C3[2] * y * (4 * z * z - x * x - y * y),
            _SH_C3[3] * z * (2 * z * z - 3 * x * x - 3 * y * y),
            _SH_C3[4] * x * (4 * z * z - x * x - y * y),
            _SH_C3[5] * z * (x * x - y * y),
            _SH_C3[6] * x * (x * x - 3 * y * y),
        ]
    )


@pytest.mark.parametrize("yaw_deg", [0.0, 30.0, -117.0, 180.0])
def test_sh_z_rotation_is_exact(yaw_deg: float) -> None:
    """``f_object(d) == f_world(R_z(yaw) @ d)`` for every direction."""
    rng = np.random.default_rng(5)
    sh = rng.standard_normal((2, 15, 3))
    gamma = math.radians(yaw_deg)
    rotated = rotate_sh_about_z(sh, gamma)
    rot = Rotation.from_euler("z", gamma)
    for _ in range(25):
        d = rng.standard_normal(3)
        d /= np.linalg.norm(d)
        np.testing.assert_allclose(
            _sh_basis(d) @ rotated[0], _sh_basis(rot.apply(d)) @ sh[0], atol=1e-12
        )


def test_extract_rotates_sh_into_the_object_frame() -> None:
    yaw_deg = 41.0
    cloud, local, rot, t = _world_car(yaw_deg, (4.0, 1.0, 0.75), sh_degree=3)
    source = _extract(cloud, rot, t)
    assert source.cloud.sh_degree == 3
    got = np.asarray(source.cloud.sh, dtype=np.float64).reshape(len(local), 15, 3)
    world = np.asarray(cloud.sh, dtype=np.float64).reshape(-1, 15, 3)[: len(local)]

    rng = np.random.default_rng(9)
    for _ in range(10):
        d = rng.standard_normal(3)
        d /= np.linalg.norm(d)
        np.testing.assert_allclose(
            _sh_basis(d) @ got[0], _sh_basis(rot.apply(d)) @ world[0], atol=1e-5
        )


def test_extract_can_drop_view_dependent_bands() -> None:
    cloud, _local, rot, t = _world_car(41.0, (4.0, 1.0, 0.75), sh_degree=3)
    source = _extract(cloud, rot, t, sh_policy="drop")
    assert source.cloud.sh_degree == 0
    assert np.asarray(source.cloud.sh).size == 0


def test_extract_can_keep_object_aligned_sh_verbatim() -> None:
    """``keep`` is for callers whose SH is already in the object frame."""
    cloud, local, rot, t = _world_car(41.0, (4.0, 1.0, 0.75), sh_degree=3)
    source = _extract(cloud, rot, t, sh_policy="keep")
    np.testing.assert_array_equal(
        np.asarray(source.cloud.sh, dtype=np.float32).reshape(len(local), 15, 3),
        np.asarray(cloud.sh, dtype=np.float32).reshape(-1, 15, 3)[: len(local)],
    )


def test_extract_refuses_to_fake_a_non_yaw_sh_rotation() -> None:
    cloud, _local, _rot, t = _world_car(0.0, (0.0, 0.0, 0.0), sh_degree=3)
    tilted = Rotation.from_euler("zyx", [0.3, 0.4, 0.0])
    with pytest.raises(ValueError, match="yaw-only SH rotation cannot express"):
        _extract(cloud, tilted, t, size=(20.0, 20.0, 20.0))


def test_extract_rejects_an_unknown_sh_policy() -> None:
    cloud, _local, rot, t = _world_car(0.0, (0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="unknown sh_policy"):
        _extract(cloud, rot, t, sh_policy="spin")


# ---------------------------------------------------------------------------
# Packing into a scene USDZ
# ---------------------------------------------------------------------------


def test_scene_usdz_carries_the_bank_and_its_payloads(
    tmp_path: Path, make_minimal_tileset_with_glb: MakeTileset
) -> None:
    ts = make_minimal_tileset_with_glb(tmp_path)
    out = tmp_path / "scene.usdz"
    result = save_scene_usdz(
        ts,
        out,
        tracks=[_track("100")],
        actor_assets=[_source(with_lidar=True)],
        actor_instances=[ActorInstance(track_id="100", asset_id="sedan_0007")],
    )
    assert result.n_actor_assets == 1
    assert result.extras["actor_assets"] == "actor_assets.json"

    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert "actor_assets.json" in names
        assert "actor_assets/sedan_0007/asset.spz" in names
        scene = json.loads(zf.read("scene.json"))
        assert scene["extras"]["actor_assets"] == "actor_assets.json"

        bank = parse_actor_assets(json.loads(zf.read("actor_assets.json")))
        asset = bank.asset_for_track("100")
        assert asset is not None and asset.class_name == "automobile"
        cloud, ext = decode_actor_asset(asset, zf.read(str(asset.uri)))
        assert cloud.num_points == asset.n_points
        assert sorted(ext) == ["lidar_intensity_raw", "lidar_raydrop_logit"]


def test_scene_usdz_rejects_a_binding_without_its_track(
    tmp_path: Path, make_minimal_tileset_with_glb: MakeTileset
) -> None:
    ts = make_minimal_tileset_with_glb(tmp_path)
    with pytest.raises(ValueError, match="not among the bundle's sequence tracks"):
        save_scene_usdz(
            ts,
            tmp_path / "scene.usdz",
            tracks=[_track("100")],
            actor_assets=[_source()],
            actor_instances=[ActorInstance(track_id="404", asset_id="sedan_0007")],
        )


def test_scene_usdz_reserves_the_actor_asset_paths(
    tmp_path: Path, make_minimal_tileset_with_glb: MakeTileset
) -> None:
    ts = make_minimal_tileset_with_glb(tmp_path)
    junk = tmp_path / "junk.bin"
    junk.write_bytes(b"\x00")
    for key in ("actor_assets.json", "actor_assets/sedan_0007/asset.spz"):
        with pytest.raises(ValueError, match="reserved scene-bundle path"):
            save_scene_usdz(ts, tmp_path / "scene.usdz", extras={key: junk})


def test_scene_usdz_accepts_a_bank_directory(
    tmp_path: Path, make_minimal_tileset_with_glb: MakeTileset
) -> None:
    bank, payloads = build_actor_asset_bank(
        [_source()], [ActorInstance(track_id="100", asset_id="sedan_0007")]
    )
    bank_dir = tmp_path / "bank"
    save_actor_asset_dir(bank_dir, bank, payloads)

    ts = make_minimal_tileset_with_glb(tmp_path)
    out = tmp_path / "scene.usdz"
    save_scene_usdz(ts, out, tracks=[_track("100")], actor_assets=bank_dir)
    with zipfile.ZipFile(out) as zf:
        # Payloads are packed byte-for-byte, never re-quantised.
        assert zf.read("actor_assets/sedan_0007/asset.spz") == payloads["sedan_0007"]


# ---------------------------------------------------------------------------
# Retrofitting an existing bundle
# ---------------------------------------------------------------------------


def _bundle_with_tracks(tmp_path: Path, make_minimal_tileset_with_glb: MakeTileset) -> Path:
    ts = make_minimal_tileset_with_glb(tmp_path)
    out = tmp_path / "scene.usdz"
    save_scene_usdz(ts, out, tracks=[_track("100")])
    return out


def test_add_actor_assets_to_an_existing_bundle(
    tmp_path: Path, make_minimal_tileset_with_glb: MakeTileset
) -> None:
    src = _bundle_with_tracks(tmp_path, make_minimal_tileset_with_glb)
    out = tmp_path / "scene_with_actors.usdz"
    result = add_actor_assets_to_usdz(
        src,
        out,
        [_source()],
        actor_instances=[ActorInstance(track_id="100", asset_id="sedan_0007")],
    )
    assert "actor_assets.json" in result.added
    with zipfile.ZipFile(out) as zf:
        assert json.loads(zf.read("scene.json"))["extras"]["actor_assets"] == "actor_assets.json"
        assert "actor_assets/sedan_0007/asset.spz" in zf.namelist()
        assert "chunks/chunk_000000.spz" in zf.namelist()


def test_replacing_a_bank_drops_the_payloads_it_no_longer_defines(
    tmp_path: Path, make_minimal_tileset_with_glb: MakeTileset
) -> None:
    src = _bundle_with_tracks(tmp_path, make_minimal_tileset_with_glb)
    first = tmp_path / "a.usdz"
    add_actor_assets_to_usdz(
        src,
        first,
        [_source("sedan_0007")],
        actor_instances=[ActorInstance(track_id="100", asset_id="sedan_0007")],
    )
    second = tmp_path / "b.usdz"
    result = add_actor_assets_to_usdz(
        first,
        second,
        [_source("truck_0001")],
        actor_instances=[ActorInstance(track_id="100", asset_id="truck_0001")],
    )
    assert result.removed == ["actor_assets/sedan_0007/asset.spz"]
    with zipfile.ZipFile(second) as zf:
        names = zf.namelist()
    assert "actor_assets/truck_0001/asset.spz" in names
    assert "actor_assets/sedan_0007/asset.spz" not in names


def test_add_actor_assets_can_refuse_to_overwrite(
    tmp_path: Path, make_minimal_tileset_with_glb: MakeTileset
) -> None:
    src = _bundle_with_tracks(tmp_path, make_minimal_tileset_with_glb)
    first = tmp_path / "a.usdz"
    add_actor_assets_to_usdz(
        src, first, [_source()], actor_instances=[ActorInstance("100", "sedan_0007")]
    )
    with pytest.raises(FileExistsError, match="already contains"):
        add_actor_assets_to_usdz(
            first,
            tmp_path / "b.usdz",
            [_source()],
            actor_instances=[ActorInstance("100", "sedan_0007")],
            overwrite=False,
        )


def test_edit_cli_actor_assets_subcommand(
    tmp_path: Path, capsys: pytest.CaptureFixture, make_minimal_tileset_with_glb: MakeTileset
) -> None:
    src = _bundle_with_tracks(tmp_path, make_minimal_tileset_with_glb)
    bank, payloads = build_actor_asset_bank(
        [_source()], [ActorInstance(track_id="100", asset_id="sedan_0007")]
    )
    bank_dir = tmp_path / "bank"
    save_actor_asset_dir(bank_dir, bank, payloads)

    out = tmp_path / "edited.usdz"
    code = _edit_cli.main(
        [
            "actor-assets",
            "--input",
            str(src),
            "--output",
            str(out),
            "--actor-assets",
            str(bank_dir),
        ]
    )
    assert code == 0
    assert "actor_assets.json" in json.loads(capsys.readouterr().out)["added"]
    with zipfile.ZipFile(out) as zf:
        assert "actor_assets/sedan_0007/asset.spz" in zf.namelist()
