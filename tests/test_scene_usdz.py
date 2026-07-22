"""Tests for the tileset-driven single-file USDZ scene-bundle writer.

`save_scene_usdz` reads a Cesium 3D Tiles tileset.json (which carries the
world-anchoring root.transform), loads the referenced glTF tile content(s),
and packs everything plus user-supplied sidecars into one self-contained
`ZIP_STORED` USDZ archive. The source tileset's root.transform is preserved
verbatim into the output's tileset.json.
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
UsdzMetadata = _mod.UsdzMetadata
save_gltf = _mod.save_gltf
save_scene_usdz = _mod.save_scene_usdz
FRAME_CONVENTION = _mod.FRAME_CONVENTION


def _expected_enu_root(transform: list[float]) -> list[float]:
    rub_to_enu = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    conversion = np.eye(4)
    conversion[:3, :3] = rub_to_enu
    source = np.asarray(transform, dtype=np.float64).reshape(4, 4).T
    return (source @ np.linalg.inv(conversion)).T.ravel().tolist()


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


# A non-identity ECEF-flavoured 4×4 (column-major, like the rainbow_bridge sample).
_NONIDENT_TRANSFORM = [
    -0.7878095269807162,
    -0.5554480619228042,
    -0.26614582413523163,
    0.0,
    0.2049411947563169,
    -0.6438890004038338,
    0.7371608113911139,
    0.0,
    -0.5808229126767246,
    0.5261980669530751,
    0.6210960782717704,
    0.0,
    -3961517.719569116,
    3352351.3421289744,
    3695591.763367203,
    1.0,
]


def _make_tileset(
    tmp_path: Path,
    *,
    cloud: spz.GaussianCloud | None = None,
    transform: list[float] | None = None,
    glb_name: str = "model.glb",
) -> Path:
    """Write ``model.glb`` + ``tileset.json`` under ``tmp_path``."""
    if cloud is None:
        cloud = _make_cloud()
    save_gltf(cloud, tmp_path / glb_name)
    doc = {
        "asset": {"version": "1.1", "generator": "test"},
        "geometricError": 100.0,
        "root": {
            "boundingVolume": {
                "box": [0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0]
            },
            "geometricError": 0,
            "refine": "ADD",
            "content": {"uri": glb_name},
        },
    }
    if transform is not None:
        doc["root"]["transform"] = transform
    tp = tmp_path / "tileset.json"
    tp.write_text(json.dumps(doc))
    return tp


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
    ts = _make_tileset(tmp_path)
    out = tmp_path / "scene.usdz"
    res = save_scene_usdz(ts, out, options=SceneUsdzOptions(chunk_size=8.0))

    names = _names(out)
    assert names[0] == "default.usda", "default.usda must come first per USDZ spec"
    assert "scene.json" in names
    assert "tileset.json" in names
    assert any(n.startswith("chunks/chunk_") and n.endswith(".spz") for n in names)
    assert res.n_chunks > 0
    assert res.n_gaussians > 0
    assert res.sh_degree == 3


def test_zip_entries_are_uncompressed(tmp_path: Path) -> None:
    ts = _make_tileset(tmp_path)
    out = tmp_path / "scene.usdz"
    save_scene_usdz(ts, out)
    with zipfile.ZipFile(out) as zf:
        for info in zf.infolist():
            assert info.compress_type == zipfile.ZIP_STORED, f"{info.filename} is compressed"


# ---------------------------------------------------------------------------
# Root transform propagation (the whole point of this redesign)
# ---------------------------------------------------------------------------


def test_root_transform_reconciled_for_enu_payload(tmp_path: Path) -> None:
    ts = _make_tileset(tmp_path, transform=_NONIDENT_TRANSFORM)
    out = tmp_path / "scene.usdz"
    res = save_scene_usdz(ts, out, options=SceneUsdzOptions(chunk_size=8.0))

    out_tileset = json.loads(_read(out, "tileset.json"))
    expected = _expected_enu_root(_NONIDENT_TRANSFORM)
    assert out_tileset["root"]["transform"] == expected
    assert res.root_transform == expected


def test_root_transform_recorded_in_scene_json(tmp_path: Path) -> None:
    ts = _make_tileset(tmp_path, transform=_NONIDENT_TRANSFORM)
    out = tmp_path / "scene.usdz"
    save_scene_usdz(ts, out, options=SceneUsdzOptions(chunk_size=8.0))
    scene = json.loads(_read(out, "scene.json"))
    expected = np.asarray(_expected_enu_root(_NONIDENT_TRANSFORM)).reshape(4, 4).T.tolist()
    assert scene["world"]["ecef_anchor"] == expected


def _make_nested_tileset(
    tmp_path: Path,
    *,
    child_transform: list[float],
    cloud: spz.GaussianCloud,
) -> Path:
    """Build a tileset whose ROOT has no transform but whose only CHILD does.

    The child's content is a glTF placed under the child tile; this exercises
    the sub-root transform path that the simple (root-only-transform) cases
    do not cover.
    """
    save_gltf(cloud, tmp_path / "model.glb")
    doc = {
        "asset": {"version": "1.1"},
        "geometricError": 100.0,
        "root": {
            "boundingVolume": {
                "box": [0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0]
            },
            "geometricError": 0,
            "refine": "ADD",
            "children": [
                {
                    "boundingVolume": {
                        "box": [
                            0.0,
                            0.0,
                            0.0,
                            100.0,
                            0.0,
                            0.0,
                            0.0,
                            100.0,
                            0.0,
                            0.0,
                            0.0,
                            100.0,
                        ]
                    },
                    "geometricError": 0,
                    "refine": "ADD",
                    "transform": child_transform,
                    "content": {"uri": "model.glb"},
                }
            ],
        },
    }
    tp = tmp_path / "tileset.json"
    tp.write_text(json.dumps(doc))
    return tp


def test_sub_root_translation_applied_to_positions(tmp_path: Path) -> None:
    """A child tile's translation must be baked into the leaf payload positions."""
    spz_io = importlib.import_module("3dgs_io.spz_io")
    n = 16
    gc = spz.GaussianCloud()
    gc.antialiased = False
    gc.positions = np.zeros(n * 3, dtype=np.float32)  # all gaussians at origin
    quats = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), n)
    gc.rotations = quats
    gc.scales = np.full(n * 3, -2.0, dtype=np.float32)
    gc.alphas = np.zeros(n, dtype=np.float32)
    gc.colors = np.zeros(n * 3, dtype=np.float32)
    gc.sh_degree = 0
    gc.sh = np.zeros(0, dtype=np.float32)

    # Column-major identity rotation + translation (10, 20, 30).
    child = [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        10.0,
        20.0,
        30.0,
        1.0,
    ]
    ts = _make_nested_tileset(tmp_path, child_transform=child, cloud=gc)
    out = tmp_path / "scene.usdz"
    save_scene_usdz(ts, out, options=SceneUsdzOptions(chunk_size=1000.0, min_scale=0.0))

    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with zipfile.ZipFile(out) as zf:
        zf.extractall(extracted)
    chunks = sorted((extracted / "chunks").glob("chunk_*.spz"))
    assert chunks
    pos = np.concatenate(
        [np.array(spz_io.load_spz(c).positions, dtype=np.float32).reshape(-1, 3) for c in chunks]
    )
    np.testing.assert_allclose(pos.mean(axis=0), [-30.0, -10.0, 20.0], atol=1e-4)


def test_nested_rotation_composition_order(tmp_path: Path) -> None:
    """Cumulative transform must be parent @ child (root first), not child @ parent.

    The leaf payload is a single point at the origin. The child tile applies a
    +Z rotation of 90° composed with a translation of (10, 0, 0). With the
    transform stored as `M_child` (column-major), the correct world-bound leaf
    position is ``M_child @ (0,0,0) = (10, 0, 0)``.  An accidental swap of
    operand order would surface a different value once a grandchild is added,
    but even at this single-level depth a mis-implementation that double-applies
    the translation (or rotates the translation) would produce a different
    point.
    """
    spz_io = importlib.import_module("3dgs_io.spz_io")
    n = 8
    gc = spz.GaussianCloud()
    gc.antialiased = False
    gc.positions = np.zeros(n * 3, dtype=np.float32)
    gc.rotations = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), n)
    gc.scales = np.full(n * 3, -2.0, dtype=np.float32)
    gc.alphas = np.zeros(n, dtype=np.float32)
    gc.colors = np.zeros(n * 3, dtype=np.float32)
    gc.sh_degree = 0
    gc.sh = np.zeros(0, dtype=np.float32)

    # column-major: rotation by 90° about Z, then translation (10, 0, 0).
    child = [
        0.0,
        1.0,
        0.0,
        0.0,  # column 0: rotZ90 maps (1,0,0)→(0,1,0)
        -1.0,
        0.0,
        0.0,
        0.0,  # column 1: (0,1,0)→(-1,0,0)
        0.0,
        0.0,
        1.0,
        0.0,
        10.0,
        0.0,
        0.0,
        1.0,
    ]
    ts = _make_nested_tileset(tmp_path, child_transform=child, cloud=gc)
    out = tmp_path / "scene.usdz"
    save_scene_usdz(ts, out, options=SceneUsdzOptions(chunk_size=1000.0, min_scale=0.0))

    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with zipfile.ZipFile(out) as zf:
        zf.extractall(extracted)
    chunks = sorted((extracted / "chunks").glob("chunk_*.spz"))
    pos = np.concatenate(
        [np.array(spz_io.load_spz(c).positions, dtype=np.float32).reshape(-1, 3) for c in chunks]
    )
    # Point at the origin moves to (10, 0, 0) under child.transform.
    np.testing.assert_allclose(pos.mean(axis=0), [0.0, -10.0, 0.0], atol=1e-4)


def test_tileset_with_utf8_bom_is_accepted(tmp_path: Path) -> None:
    ts = _make_tileset(tmp_path)
    raw = ts.read_bytes()
    ts.write_bytes(b"\xef\xbb\xbf" + raw)  # prepend UTF-8 BOM
    out = tmp_path / "scene.usdz"
    res = save_scene_usdz(ts, out)
    assert res.n_gaussians > 0


def test_extras_trailing_slash_still_rejected(tmp_path: Path) -> None:
    ts = _make_tileset(tmp_path)
    f = tmp_path / "x"
    f.write_bytes(b"x")
    with pytest.raises(ValueError, match="reserved"):
        save_scene_usdz(ts, tmp_path / "out.usdz", extras={"scene.json/": f})


def test_missing_root_transform_gets_enu_reconciliation(tmp_path: Path) -> None:
    ts = _make_tileset(tmp_path, transform=None)
    out = tmp_path / "scene.usdz"
    res = save_scene_usdz(ts, out)
    identity = np.eye(4).T.ravel().tolist()
    expected = _expected_enu_root(identity)
    assert res.root_transform == expected
    out_tileset = json.loads(_read(out, "tileset.json"))
    assert out_tileset["root"]["transform"] == expected


def test_positions_stay_in_root_local_frame(tmp_path: Path) -> None:
    """The big ECEF translation in root.transform must NOT be applied to positions.

    Positions inside the resulting chunks should match the source cloud's
    range (~ -20 to 20 from the test fixture), not the ECEF magnitude of ~6e6.
    """
    spz_io = importlib.import_module("3dgs_io.spz_io")
    cloud = _make_cloud(n=256)
    ts = _make_tileset(tmp_path, cloud=cloud, transform=_NONIDENT_TRANSFORM)
    out = tmp_path / "scene.usdz"
    save_scene_usdz(ts, out, options=SceneUsdzOptions(chunk_size=100.0))

    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with zipfile.ZipFile(out) as zf:
        zf.extractall(extracted)
    for chunk_path in sorted((extracted / "chunks").glob("chunk_*.spz")):
        gc = spz_io.load_spz(chunk_path)
        pos = np.array(gc.positions, dtype=np.float32).reshape(-1, 3)
        assert pos.size > 0
        # Far below the ECEF-magnitude (~6.4M) the root.transform would imply.
        assert np.abs(pos).max() < 1000.0, "positions accidentally transformed into ECEF"


# ---------------------------------------------------------------------------
# scene.json / tileset.json contents
# ---------------------------------------------------------------------------


def test_scene_json_schema(tmp_path: Path) -> None:
    ts = _make_tileset(tmp_path)
    out = tmp_path / "scene.usdz"
    save_scene_usdz(ts, out, options=SceneUsdzOptions(chunk_size=8.0))
    scene = json.loads(_read(out, "scene.json"))
    assert scene["schema"] == "splatsim.scene/v2"
    assert scene["producer"]["source_tileset"] == "tileset.json"
    assert scene["world"]["frame_convention"] == FRAME_CONVENTION
    assert "ecef_anchor" in scene["world"]
    assert scene["gaussians"]["frame"] == "world"
    assert scene["gaussians"]["tileset"] == "tileset.json"
    assert scene["gaussians"]["sh_degree"] == 3
    assert scene["gaussians"]["n_gaussians"] > 0
    assert scene["extras"] == {
        "map_lanelet2": None,
        "map_opendrive": None,
        "carla_world": None,
        "tracks": None,
        "trajectory": None,
        "sequence_tracks": None,
        "rig_trajectories": None,
        "ppisp": None,
    }


def test_tileset_uses_ext_3dgs_spz(tmp_path: Path) -> None:
    ts = _make_tileset(tmp_path)
    out = tmp_path / "scene.usdz"
    save_scene_usdz(ts, out, options=SceneUsdzOptions(chunk_size=8.0))
    out_ts = json.loads(_read(out, "tileset.json"))
    assert out_ts["asset"]["tilesetVersion"] == "splatsim-spz/1.0"
    assert "EXT_3dgs_spz" in out_ts["extensionsRequired"]
    children = out_ts["root"]["children"]
    assert children
    for c in children:
        assert c["content"]["uri"].startswith("chunks/chunk_")
        ext = c["content"]["extensions"]["EXT_3dgs_spz"]
        assert ext["format"] == "spz/1"
        assert ext["n_points"] > 0


# ---------------------------------------------------------------------------
# Tileset input validation
# ---------------------------------------------------------------------------


def test_tileset_missing_root_raises(tmp_path: Path) -> None:
    bad = tmp_path / "tileset.json"
    bad.write_text(json.dumps({"asset": {"version": "1.1"}}))
    with pytest.raises(ValueError, match="missing 'root'"):
        save_scene_usdz(bad, tmp_path / "out.usdz")


def test_tileset_no_content_raises(tmp_path: Path) -> None:
    bad = tmp_path / "tileset.json"
    bad.write_text(json.dumps({"asset": {"version": "1.1"}, "root": {"geometricError": 0}}))
    with pytest.raises(ValueError, match="no tile content"):
        save_scene_usdz(bad, tmp_path / "out.usdz")


def test_tileset_remote_uri_raises(tmp_path: Path) -> None:
    bad = tmp_path / "tileset.json"
    bad.write_text(
        json.dumps(
            {
                "asset": {"version": "1.1"},
                "root": {
                    "geometricError": 0,
                    "content": {"uri": "https://example.com/model.glb"},
                },
            }
        )
    )
    with pytest.raises(ValueError, match="Remote tile content"):
        save_scene_usdz(bad, tmp_path / "out.usdz")


def test_tileset_non_gltf_content_raises(tmp_path: Path) -> None:
    spz_path = tmp_path / "model.spz"
    spz_path.write_bytes(b"not really")
    bad = tmp_path / "tileset.json"
    bad.write_text(
        json.dumps(
            {
                "asset": {"version": "1.1"},
                "root": {"geometricError": 0, "content": {"uri": "model.spz"}},
            }
        )
    )
    with pytest.raises(ValueError, match="glTF tile content"):
        save_scene_usdz(bad, tmp_path / "out.usdz")


def test_tileset_bad_transform_length_raises(tmp_path: Path) -> None:
    bad = tmp_path / "tileset.json"
    bad.write_text(
        json.dumps(
            {
                "asset": {"version": "1.1"},
                "root": {
                    "geometricError": 0,
                    "transform": [1.0, 0.0, 0.0],  # only 3 elements
                    "content": {"uri": "model.glb"},
                },
            }
        )
    )
    with pytest.raises(ValueError, match="16 elements"):
        save_scene_usdz(bad, tmp_path / "out.usdz")


# ---------------------------------------------------------------------------
# Extras: file and directory sources, known + arbitrary keys
# ---------------------------------------------------------------------------


def test_extras_file_is_embedded_verbatim(tmp_path: Path) -> None:
    ts = _make_tileset(tmp_path)
    src = tmp_path / "tracks.parquet"
    src.write_bytes(b"PAR1\x00fake-parquet")

    out = tmp_path / "scene.usdz"
    save_scene_usdz(ts, out, extras={"tracks.parquet": src})

    assert "tracks.parquet" in _names(out)
    assert _read(out, "tracks.parquet") == b"PAR1\x00fake-parquet"
    scene = json.loads(_read(out, "scene.json"))
    assert scene["extras"]["tracks"] == "tracks.parquet"


def test_extras_known_archive_paths_populate_scene_extras(tmp_path: Path) -> None:
    ts = _make_tileset(tmp_path)
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
        ts,
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
        "sequence_tracks": None,
        "rig_trajectories": None,
        "ppisp": None,
    }


def test_arbitrary_extras_path_is_embedded_but_not_in_scene_meta(tmp_path: Path) -> None:
    ts = _make_tileset(tmp_path)
    f = tmp_path / "anything.bin"
    f.write_bytes(b"\xaa\xbb\xcc")
    out = tmp_path / "scene.usdz"
    save_scene_usdz(ts, out, extras={"misc/data.bin": f})
    assert "misc/data.bin" in _names(out)
    scene = json.loads(_read(out, "scene.json"))
    assert scene["extras"]["carla_world"] is None
    assert scene["extras"]["map_lanelet2"] is None


def test_extras_reserved_path_is_rejected(tmp_path: Path) -> None:
    ts = _make_tileset(tmp_path)
    f = tmp_path / "x"
    f.write_bytes(b"x")
    with pytest.raises(ValueError, match="reserved"):
        save_scene_usdz(ts, tmp_path / "out.usdz", extras={"scene.json": f})
    with pytest.raises(ValueError, match="reserved"):
        save_scene_usdz(ts, tmp_path / "out.usdz", extras={"chunks/foo.spz": f})


def test_extras_missing_source_raises(tmp_path: Path) -> None:
    ts = _make_tileset(tmp_path)
    with pytest.raises(FileNotFoundError):
        save_scene_usdz(ts, tmp_path / "out.usdz", extras={"x.bin": tmp_path / "nope"})


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def test_chunk_size_controls_number_of_tiles(tmp_path: Path) -> None:
    src_a = tmp_path / "a"
    src_b = tmp_path / "b"
    src_a.mkdir()
    src_b.mkdir()
    ts_a = _make_tileset(src_a)
    ts_b = _make_tileset(src_b)
    small = save_scene_usdz(ts_a, tmp_path / "s.usdz", options=SceneUsdzOptions(chunk_size=5.0))
    big = save_scene_usdz(ts_b, tmp_path / "big.usdz", options=SceneUsdzOptions(chunk_size=200.0))
    assert small.n_chunks > big.n_chunks


def test_max_points_per_chunk_splits_dense_cells(tmp_path: Path) -> None:
    gc = _make_cloud(n=200)
    gc.positions = np.zeros_like(gc.positions)
    ts = _make_tileset(tmp_path, cloud=gc)
    res = save_scene_usdz(
        ts,
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
    ts = _make_tileset(tmp_path, cloud=gc)
    res = save_scene_usdz(ts, tmp_path / "out.usdz", options=SceneUsdzOptions(bbox_radius=100.0))
    assert res.n_gaussians == 63


# ---------------------------------------------------------------------------
# Spz tiles inside the USDZ are loadable
# ---------------------------------------------------------------------------


def test_chunks_are_loadable_spz(tmp_path: Path) -> None:
    spz_io = importlib.import_module("3dgs_io.spz_io")
    ts = _make_tileset(tmp_path)
    out = tmp_path / "scene.usdz"
    res = save_scene_usdz(ts, out, options=SceneUsdzOptions(chunk_size=8.0))

    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with zipfile.ZipFile(out) as zf:
        zf.extractall(extracted)

    total = 0
    chunk_paths = sorted((extracted / "chunks").glob("chunk_*.spz"))
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


def test_cli_with_tileset_input(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cli = importlib.import_module("3dgs_io.scene_usdz_cli")
    ts = _make_tileset(tmp_path, transform=_NONIDENT_TRANSFORM)
    out = tmp_path / "out.usdz"
    rc = cli.main([str(ts), str(out), "--chunk-size", "8.0"])
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["out_path"].endswith("out.usdz")
    assert summary["n_gaussians"] > 0
    assert summary["root_transform"] == _expected_enu_root(_NONIDENT_TRANSFORM)


def test_cli_quiet_suppresses_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cli = importlib.import_module("3dgs_io.scene_usdz_cli")
    ts = _make_tileset(tmp_path)
    rc = cli.main([str(ts), str(tmp_path / "out.usdz"), "--quiet"])
    assert rc == 0
    assert capsys.readouterr().out == ""


def test_cli_extra_flag_embeds_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cli = importlib.import_module("3dgs_io.scene_usdz_cli")
    ts = _make_tileset(tmp_path)
    extra = tmp_path / "tracks.parquet"
    extra.write_bytes(b"PAR1\x00x")
    out = tmp_path / "out.usdz"
    rc = cli.main([str(ts), str(out), "--extra", f"tracks.parquet={extra}", "--quiet"])
    assert rc == 0
    assert "tracks.parquet" in _names(out)
    scene = json.loads(_read(out, "scene.json"))
    assert scene["extras"]["tracks"] == "tracks.parquet"


# ---------------------------------------------------------------------------
# metadata.yaml (identity card)
# ---------------------------------------------------------------------------


def _read_metadata(usdz_path: Path) -> dict:
    return json.loads(_read(usdz_path, "metadata.yaml"))


def test_save_scene_usdz_writes_default_metadata_yaml(tmp_path: Path) -> None:
    ts = _make_tileset(tmp_path)
    out = tmp_path / "odaibatest5.usdz"

    result = save_scene_usdz(ts, out)

    assert "metadata.yaml" in _names(out)
    doc = _read_metadata(out)
    for key in ("uuid", "scene_id", "version_string"):
        assert isinstance(doc[key], str) and doc[key], key
    assert doc["scene_id"] == "odaibatest5", "default scene_id follows the output filename stem"
    assert doc["version_string"].startswith("3dgs_io/")
    assert result.metadata == doc


def test_save_scene_usdz_honours_explicit_metadata(tmp_path: Path) -> None:
    ts = _make_tileset(tmp_path)
    out = tmp_path / "scene.usdz"

    metadata = UsdzMetadata(
        uuid="odaibatest5",
        scene_id="odaibatest5",
        version_string="local-e2e",
        extras={"pipeline": "unit-test"},
    )
    result = save_scene_usdz(ts, out, metadata=metadata)
    doc = _read_metadata(out)
    assert doc["uuid"] == "odaibatest5"
    assert doc["scene_id"] == "odaibatest5"
    assert doc["version_string"] == "local-e2e"
    assert doc["pipeline"] == "unit-test"
    assert result.metadata["pipeline"] == "unit-test"


def test_save_scene_usdz_metadata_yaml_is_yaml_parseable(tmp_path: Path) -> None:
    # JSON is a subset of YAML 1.2 so consumers that pull metadata.yaml through
    # yaml.safe_load must round-trip the required keys.
    yaml = pytest.importorskip("yaml")
    ts = _make_tileset(tmp_path)
    out = tmp_path / "scene.usdz"
    save_scene_usdz(ts, out)
    with zipfile.ZipFile(out) as zf, zf.open("metadata.yaml") as fh:
        doc = yaml.safe_load(fh)
    assert set(("uuid", "scene_id", "version_string")).issubset(doc)


def test_save_scene_usdz_metadata_yaml_is_uncompressed(tmp_path: Path) -> None:
    ts = _make_tileset(tmp_path)
    out = tmp_path / "scene.usdz"
    save_scene_usdz(ts, out)
    with zipfile.ZipFile(out) as zf:
        info = zf.getinfo("metadata.yaml")
    assert info.compress_type == zipfile.ZIP_STORED


def test_save_scene_usdz_metadata_yaml_reserved_from_extras(tmp_path: Path) -> None:
    ts = _make_tileset(tmp_path)
    (tmp_path / "conflict.yaml").write_text("uuid: hijacked\n")
    with pytest.raises(ValueError, match="reserved"):
        save_scene_usdz(
            ts,
            tmp_path / "out.usdz",
            extras={"metadata.yaml": tmp_path / "conflict.yaml"},
        )


def test_cli_metadata_flags_override_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli = importlib.import_module("3dgs_io.scene_usdz_cli")
    ts = _make_tileset(tmp_path)
    out = tmp_path / "out.usdz"
    rc = cli.main(
        [
            str(ts),
            str(out),
            "--uuid",
            "odaibatest5",
            "--scene-id",
            "odaibatest5",
            "--version-string",
            "local-e2e",
            "--quiet",
        ]
    )
    assert rc == 0
    doc = _read_metadata(out)
    assert doc["uuid"] == "odaibatest5"
    assert doc["scene_id"] == "odaibatest5"
    assert doc["version_string"] == "local-e2e"
