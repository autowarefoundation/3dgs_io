from __future__ import annotations

import importlib
import json
import zipfile
from pathlib import Path

import pytest

_mod = importlib.import_module("3dgs_io")


def _track(track_id: str = "vehicle"):
    return _mod.Track(
        track_id=track_id,
        class_name="automobile",
        size=(4.5, 1.8, 1.5),
        frames=[
            _mod.TrackFrame(1_000_000, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            _mod.TrackFrame(1_100_000, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        ],
    )


def test_v2_round_trip_is_world_frame() -> None:
    document = _mod.serialize_tracks([_track()])
    assert document["schema"] == "splatsim.sequence_tracks/v2"
    assert document["frame"] == "world"
    assert document["frame_convention"] == _mod.FRAME_CONVENTION
    assert _mod.parse_tracks(document) == [_track()]


def test_reader_rejects_legacy_shape() -> None:
    with pytest.raises(ValueError, match="unexpected tracks schema"):
        _mod.parse_tracks({"dummy_chunk": {"tracks_data": {}}})


def test_track_validation() -> None:
    track = _track()
    track.frames[1].timestamp_us = track.frames[0].timestamp_us
    with pytest.raises(ValueError, match="strictly increasing"):
        _mod.serialize_tracks([track])

    track = _track()
    track.frames[0].rotation = (0.0, 0.0, 0.0, 0.5)
    with pytest.raises(ValueError, match="unit-norm"):
        _mod.serialize_tracks([track])

    with pytest.raises(ValueError, match="duplicate track_id"):
        _mod.serialize_tracks([_track("a"), _track("a")])


def test_scene_embeds_v2_tracks(tmp_path: Path, make_minimal_tileset_with_glb) -> None:
    tileset = make_minimal_tileset_with_glb(tmp_path)
    output = tmp_path / "scene.usdz"
    _mod.save_scene_usdz(tileset, output, tracks=[_track()])
    with zipfile.ZipFile(output) as archive:
        document = json.loads(archive.read("sequence_tracks.json"))
    assert document["schema"] == "splatsim.sequence_tracks/v2"
    assert document["frame"] == "world"


def test_cli_accepts_v2_tracks(tmp_path: Path, make_minimal_tileset_with_glb) -> None:
    cli = importlib.import_module("3dgs_io.scene_usdz_cli")
    tileset = make_minimal_tileset_with_glb(tmp_path)
    tracks_path = tmp_path / "tracks.json"
    tracks_path.write_text(json.dumps(_mod.serialize_tracks([_track()])))
    output = tmp_path / "scene.usdz"
    assert cli.main([str(tileset), str(output), "--tracks", str(tracks_path), "--quiet"]) == 0
