"""Tests for the dynamic-object Track dataclasses + sequence_tracks.json (de)serialisation."""

from __future__ import annotations

import importlib
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

_mod = importlib.import_module("3dgs_io")
Track = _mod.Track
TrackFrame = _mod.TrackFrame
FRAME_CONVENTION = _mod.FRAME_CONVENTION
dump_alpasim_sequence_tracks = _mod.dump_alpasim_sequence_tracks
parse_alpasim_sequence_tracks = _mod.parse_alpasim_sequence_tracks
parse_tracks = _mod.parse_tracks
save_scene_usdz = _mod.save_scene_usdz
serialize_tracks = _mod.serialize_tracks


def _read_tracks_from_usdz(path: Path) -> list[Track]:
    """Test helper: pull sequence_tracks.json out of a USDZ and parse it."""
    with zipfile.ZipFile(path) as zf:
        doc = json.loads(zf.read("sequence_tracks.json").decode("utf-8-sig"))
    return parse_tracks(doc)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _track(
    track_id: str = "veh_0",
    class_name: str = "automobile",
    base_xy: tuple[float, float] = (10.0, 5.0),
    n_frames: int = 4,
) -> Track:
    frames = [
        TrackFrame(
            timestamp_us=27_000_000_000 + i * 100_000,
            translation=(base_xy[0] + 0.1 * i, base_xy[1], 1.9),
            rotation=(0.0, 0.0, 0.0, 1.0),
        )
        for i in range(n_frames)
    ]
    return Track(
        track_id=track_id,
        class_name=class_name,
        size=(4.5, 1.8, 1.5),
        frames=frames,
        flag="NONE",
        metadata={"source": "test"},
    )


# Tileset+GLB fixture lives in conftest.py as ``make_minimal_tileset_with_glb``.


# ---------------------------------------------------------------------------
# Schema (de)serialisation
# ---------------------------------------------------------------------------


def test_serialize_then_parse_roundtrip() -> None:
    tracks = [_track("veh_0"), _track("veh_1", class_name="person")]
    doc = serialize_tracks(tracks)
    assert doc["schema"] == "splatsim.sequence_tracks/v2"
    assert doc["frame"] == "world"
    assert len(doc["tracks"]) == 2

    recovered = parse_tracks(doc)
    assert {t.track_id for t in recovered} == {"veh_0", "veh_1"}
    by_id = {t.track_id: t for t in recovered}
    assert by_id["veh_0"].class_name == "automobile"
    assert by_id["veh_1"].class_name == "person"
    assert by_id["veh_0"].size == (4.5, 1.8, 1.5)
    assert len(by_id["veh_0"].frames) == 4
    assert by_id["veh_0"].frames[0].timestamp_us == 27_000_000_000
    assert by_id["veh_0"].frames[0].translation == (10.0, 5.0, 1.9)
    assert by_id["veh_0"].metadata == {"source": "test"}


def test_serialize_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="duplicate track_id"):
        serialize_tracks([_track("a"), _track("a")])


def test_serialize_rejects_invalid_frame_contract() -> None:
    duplicate_time = _track("a")
    duplicate_time.frames[1].timestamp_us = duplicate_time.frames[0].timestamp_us
    with pytest.raises(ValueError, match="strictly increasing"):
        serialize_tracks([duplicate_time])

    bad_rotation = _track("b")
    bad_rotation.frames[0].rotation = (0.0, 0.0, 0.0, 0.5)
    with pytest.raises(ValueError, match="unit-norm"):
        serialize_tracks([bad_rotation])


def test_parse_rejects_wrong_schema() -> None:
    bad = {"schema": "other/v1", "tracks": []}
    with pytest.raises(ValueError, match="unexpected tracks schema"):
        parse_tracks(bad)


def test_parse_rejects_missing_tracks_list() -> None:
    bad = {
        "schema": "splatsim.sequence_tracks/v2",
        "frame": "world",
        "frame_convention": FRAME_CONVENTION,
    }
    with pytest.raises(ValueError, match="missing the 'tracks' list"):
        parse_tracks(bad)


def test_parse_validates_frame_lengths() -> None:
    bad_frame = {"timestamp_us": 1, "translation": [1, 2], "rotation": [0, 0, 0, 1]}
    bad_track = {
        "track_id": "x",
        "class_name": "c",
        "size": [1, 1, 1],
        "frames": [bad_frame],
    }
    with pytest.raises(ValueError, match="translation must have 3"):
        parse_tracks(
            {
                "schema": "splatsim.sequence_tracks/v2",
                "frame": "world",
                "frame_convention": FRAME_CONVENTION,
                "tracks": [bad_track],
            }
        )


def test_parse_validates_size_length() -> None:
    bad_track = {
        "track_id": "x",
        "class_name": "c",
        "size": [1, 1],
        "frames": [],
    }
    with pytest.raises(ValueError, match="size must have 3"):
        parse_tracks(
            {
                "schema": "splatsim.sequence_tracks/v2",
                "frame": "world",
                "frame_convention": FRAME_CONVENTION,
                "tracks": [bad_track],
            }
        )


# ---------------------------------------------------------------------------
# alpasim sequence_tracks.json ingestion
# ---------------------------------------------------------------------------


_ALPASIM_DOC = {
    "dummy_chunk_id": {
        "tracks_data": {
            "tracks_id": ["100", "104"],
            "tracks_poses": [
                [
                    [113.62, -58.55, 1.92, -0.0005, -0.0113, 0.6645, 0.7472],
                    [113.57, -58.59, 1.88, -0.0003, -0.0113, 0.6646, 0.7471],
                ],
                [
                    [50.0, 10.0, 1.5, 0.0, 0.0, 0.0, 1.0],
                ],
            ],
            "tracks_timestamps_us": [
                [27_567_868_848, 27_567_968_602],
                [27_567_868_848],
            ],
            "tracks_label_class": ["automobile", "person"],
            "tracks_flags": ["NONE", "NONE"],
        },
        "cuboidtracks_data": {
            "cuboids_dims": [[3.989, 1.803, 1.484], [0.6, 0.6, 1.7]],
        },
    }
}


def test_alpasim_ingestion_single_chunk() -> None:
    tracks = parse_alpasim_sequence_tracks(_ALPASIM_DOC)
    assert {t.track_id for t in tracks} == {"100", "104"}
    by_id = {t.track_id: t for t in tracks}

    assert by_id["100"].class_name == "automobile"
    assert by_id["100"].size == (3.989, 1.803, 1.484)
    assert len(by_id["100"].frames) == 2
    f0 = by_id["100"].frames[0]
    assert f0.timestamp_us == 27_567_868_848
    np.testing.assert_allclose(f0.translation, (113.62, -58.55, 1.92), atol=1e-6)
    np.testing.assert_allclose(f0.rotation, (-0.0005, -0.0113, 0.6645, 0.7472), atol=1e-6)

    assert by_id["104"].class_name == "person"
    assert len(by_id["104"].frames) == 1


def test_alpasim_ingestion_namespaces_track_ids_for_multi_chunk() -> None:
    doc = {
        "chunk_a": _ALPASIM_DOC["dummy_chunk_id"],
        "chunk_b": _ALPASIM_DOC["dummy_chunk_id"],
    }
    tracks = parse_alpasim_sequence_tracks(doc)
    ids = {t.track_id for t in tracks}
    assert ids == {"chunk_a/100", "chunk_a/104", "chunk_b/100", "chunk_b/104"}


def test_alpasim_ingestion_pose_timestamp_length_mismatch_raises() -> None:
    bad = {
        "c0": {
            "tracks_data": {
                "tracks_id": ["x"],
                "tracks_poses": [[[0, 0, 0, 0, 0, 0, 1]]],
                "tracks_timestamps_us": [[1, 2]],  # two timestamps but one pose
                "tracks_label_class": ["automobile"],
                "tracks_flags": ["NONE"],
            },
            "cuboidtracks_data": {"cuboids_dims": [[1, 1, 1]]},
        }
    }
    with pytest.raises(ValueError, match="pose count"):
        parse_alpasim_sequence_tracks(bad)


_ALPASIM_SAMPLE = Path.home() / "Downloads" / "00040136-e651-4abd-991d-0655ccda9430.usdz"


@pytest.mark.skipif(
    not _ALPASIM_SAMPLE.exists() or os.environ.get("SKIP_USDZ_SAMPLE") == "1",
    reason="alpasim sample USDZ not available",
)
def test_alpasim_ingestion_against_real_sample() -> None:
    with zipfile.ZipFile(_ALPASIM_SAMPLE) as zf:
        doc = json.loads(zf.read("sequence_tracks.json"))
    tracks = parse_alpasim_sequence_tracks(doc)
    assert len(tracks) > 0
    # The sample has automobile + person + rider + trailer
    classes = {t.class_name for t in tracks}
    assert "automobile" in classes
    # Every track has constant size and >= 1 frame
    for t in tracks:
        assert t.size != (0.0, 0.0, 0.0)
        assert len(t.frames) >= 1
        # Quaternions should be unit-norm (alpasim guarantees this)
        for f in t.frames:
            n = np.linalg.norm(f.rotation)
            np.testing.assert_allclose(n, 1.0, atol=1e-4)


# ---------------------------------------------------------------------------
# dump_alpasim_sequence_tracks
# ---------------------------------------------------------------------------


_EMPTY_ALPASIM_KEYS = {
    "tracks_id",
    "tracks_label_class",
    "tracks_flags",
    "tracks_timestamps_us",
    "tracks_poses",
}


def test_dump_alpasim_sequence_tracks_empty_still_emits_full_key_structure() -> None:
    # alpasim's TrafficObjects.load_from_json indexes these keys
    # unconditionally, so a bare {"tracks_data": {}, "cuboidtracks_data": {}}
    # would KeyError. Even a track-free scene must carry the full skeleton.
    doc = dump_alpasim_sequence_tracks([], chunk_id="ego")
    assert list(doc) == ["ego"]
    chunk = doc["ego"]
    assert set(chunk["tracks_data"].keys()) == _EMPTY_ALPASIM_KEYS
    for k in _EMPTY_ALPASIM_KEYS:
        assert chunk["tracks_data"][k] == []
    assert chunk["cuboidtracks_data"] == {"cuboids_dims": []}


def test_dump_alpasim_sequence_tracks_default_chunk_id_is_ego() -> None:
    # "ego" matches the minimal empty-doc example in the issue's follow-up
    # comment; alpasim treats the chunk key as the sequence identifier.
    doc = dump_alpasim_sequence_tracks([])
    assert list(doc) == ["ego"]


def test_dump_alpasim_sequence_tracks_round_trips_via_parse() -> None:
    tracks = [
        Track(
            track_id="100",
            class_name="automobile",
            size=(3.989, 1.803, 1.484),
            frames=[
                TrackFrame(
                    timestamp_us=27_567_868_848,
                    translation=(113.62, -58.55, 1.92),
                    rotation=(-0.0005, -0.0113, 0.6645, 0.7472),
                ),
                TrackFrame(
                    timestamp_us=27_567_968_602,
                    translation=(113.57, -58.59, 1.88),
                    rotation=(-0.0003, -0.0113, 0.6646, 0.7471),
                ),
            ],
            flag="NONE",
        ),
        Track(
            track_id="104",
            class_name="person",
            size=(0.6, 0.6, 1.7),
            frames=[
                TrackFrame(
                    timestamp_us=27_567_868_848,
                    translation=(50.0, 10.0, 1.5),
                    rotation=(0.0, 0.0, 0.0, 1.0),
                ),
            ],
            flag="NONE",
        ),
    ]
    doc = dump_alpasim_sequence_tracks(tracks, chunk_id="dummy_chunk_id")

    # Columnar shape matches what parse expects.
    td = doc["dummy_chunk_id"]["tracks_data"]
    assert td["tracks_id"] == ["100", "104"]
    assert td["tracks_label_class"] == ["automobile", "person"]
    assert td["tracks_flags"] == ["NONE", "NONE"]
    assert doc["dummy_chunk_id"]["cuboidtracks_data"]["cuboids_dims"] == [
        [3.989, 1.803, 1.484],
        [0.6, 0.6, 1.7],
    ]
    assert len(td["tracks_poses"][0]) == 2
    assert len(td["tracks_poses"][1]) == 1

    reparsed = parse_alpasim_sequence_tracks(doc)
    assert [t.track_id for t in reparsed] == ["100", "104"]
    assert [t.class_name for t in reparsed] == ["automobile", "person"]
    assert reparsed[0].size == (3.989, 1.803, 1.484)
    assert len(reparsed[0].frames) == 2
    np.testing.assert_allclose(reparsed[0].frames[0].translation, (113.62, -58.55, 1.92))
    np.testing.assert_allclose(reparsed[0].frames[0].rotation, (-0.0005, -0.0113, 0.6645, 0.7472))


def test_dump_alpasim_sequence_tracks_rejects_duplicate_track_ids() -> None:
    dup = [
        Track(track_id="a", class_name="x", size=(1.0, 1.0, 1.0), frames=[]),
        Track(track_id="a", class_name="y", size=(1.0, 1.0, 1.0), frames=[]),
    ]
    with pytest.raises(ValueError, match="duplicate track_id"):
        dump_alpasim_sequence_tracks(dup)


# ---------------------------------------------------------------------------
# Integration with save_scene_usdz
# ---------------------------------------------------------------------------


def test_save_scene_usdz_embeds_sequence_tracks(
    tmp_path: Path, make_minimal_tileset_with_glb
) -> None:
    ts = make_minimal_tileset_with_glb(tmp_path)
    out = tmp_path / "scene.usdz"
    tracks = [_track("a"), _track("b", class_name="person")]
    res = save_scene_usdz(ts, out, tracks=tracks)
    assert res.extras["sequence_tracks"] == "sequence_tracks.json"

    with zipfile.ZipFile(out) as zf:
        assert "sequence_tracks.json" in zf.namelist()
        scene = json.loads(zf.read("scene.json"))
        tracks_doc = json.loads(zf.read("sequence_tracks.json"))
    assert scene["extras"]["sequence_tracks"] == "sequence_tracks.json"
    assert tracks_doc["schema"] == "splatsim.sequence_tracks/v2"
    assert tracks_doc["frame"] == "world"
    assert tracks_doc["frame_convention"] == FRAME_CONVENTION
    assert {t["track_id"] for t in tracks_doc["tracks"]} == {"a", "b"}


def test_tracks_round_trip_via_usdz(tmp_path: Path, make_minimal_tileset_with_glb) -> None:
    ts = make_minimal_tileset_with_glb(tmp_path)
    out = tmp_path / "scene.usdz"
    original = [_track("a"), _track("b", base_xy=(42.0, -3.0))]
    save_scene_usdz(ts, out, tracks=original)

    recovered = _read_tracks_from_usdz(out)
    by_id = {t.track_id: t for t in recovered}
    assert by_id["b"].frames[0].translation == (42.0, -3.0, 1.9)
    assert by_id["a"].size == (4.5, 1.8, 1.5)


def test_tracks_and_extras_collision_rejected(
    tmp_path: Path, make_minimal_tileset_with_glb
) -> None:
    ts = make_minimal_tileset_with_glb(tmp_path)
    fake = tmp_path / "sequence_tracks.json"
    fake.write_text("{}")
    with pytest.raises(ValueError, match="tracks="):
        save_scene_usdz(
            ts,
            tmp_path / "out.usdz",
            tracks=[_track("a")],
            extras={"sequence_tracks.json": fake},
        )


def test_cli_tracks_flag_native_schema(tmp_path: Path, make_minimal_tileset_with_glb) -> None:
    cli = importlib.import_module("3dgs_io.scene_usdz_cli")
    ts = make_minimal_tileset_with_glb(tmp_path)
    tracks_path = tmp_path / "tracks.json"
    tracks_path.write_text(json.dumps(serialize_tracks([_track("c0")])))

    out = tmp_path / "scene.usdz"
    rc = cli.main([str(ts), str(out), "--tracks", str(tracks_path), "--quiet"])
    assert rc == 0
    recovered = _read_tracks_from_usdz(out)
    assert [t.track_id for t in recovered] == ["c0"]


def test_cli_tracks_flag_rejects_alpasim_format(
    tmp_path: Path, make_minimal_tileset_with_glb
) -> None:
    cli = importlib.import_module("3dgs_io.scene_usdz_cli")
    ts = make_minimal_tileset_with_glb(tmp_path)
    alpasim_path = tmp_path / "alpasim_tracks.json"
    alpasim_path.write_text(json.dumps(_ALPASIM_DOC))

    out = tmp_path / "scene.usdz"
    with pytest.raises(ValueError, match="unexpected tracks schema"):
        cli.main([str(ts), str(out), "--tracks", str(alpasim_path), "--quiet"])
