"""Tests for :mod:`3dgs_io.usdz_metadata`."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

_mod = importlib.import_module("3dgs_io")
UsdzMetadata = _mod.UsdzMetadata
default_uuid = _mod.default_uuid
encode_usdz_metadata = _mod.encode_usdz_metadata
load_usdz_metadata = _mod.load_usdz_metadata
make_default_metadata = _mod.make_default_metadata


def test_required_fields_must_be_non_empty() -> None:
    with pytest.raises(ValueError, match="uuid"):
        UsdzMetadata(uuid="", scene_id="s", version_string="v")
    with pytest.raises(ValueError, match="scene_id"):
        UsdzMetadata(uuid="u", scene_id="", version_string="v")
    with pytest.raises(ValueError, match="version_string"):
        UsdzMetadata(uuid="u", scene_id="s", version_string="")


def test_extras_cannot_shadow_required_keys() -> None:
    with pytest.raises(ValueError, match="shadow"):
        UsdzMetadata(uuid="u", scene_id="s", version_string="v", extras={"uuid": "x"})


def test_to_dict_orders_required_first() -> None:
    m = UsdzMetadata(uuid="u", scene_id="s", version_string="v", extras={"pipeline": "alpasim"})
    assert list(m.to_dict().keys()) == ["uuid", "scene_id", "version_string", "pipeline"]


def test_from_dict_round_trips_through_extras() -> None:
    src = {
        "uuid": "u",
        "scene_id": "s",
        "version_string": "v",
        "pipeline": "alpasim",
        "frame_count": 60,
    }
    m = UsdzMetadata.from_dict(src)
    assert m.uuid == "u"
    assert m.extras == {"pipeline": "alpasim", "frame_count": 60}
    assert m.to_dict() == src


def test_from_dict_missing_required_key_raises() -> None:
    with pytest.raises(ValueError, match="version_string"):
        UsdzMetadata.from_dict({"uuid": "u", "scene_id": "s"})


def test_default_uuid_is_uuid4_shaped() -> None:
    val = default_uuid()
    # 8-4-4-4-12 hex form; 36 chars including dashes.
    assert len(val) == 36
    assert val.count("-") == 4


def test_make_default_metadata_uses_out_path_stem() -> None:
    m = make_default_metadata(out_path=Path("/tmp/odaibatest5.usdz"))
    assert m.scene_id == "odaibatest5"
    assert m.uuid  # non-empty
    assert m.version_string.startswith("3dgs_io/")


def test_encode_produces_valid_json_and_yaml() -> None:
    m = UsdzMetadata(uuid="u", scene_id="s", version_string="v", extras={"n": 3})
    raw = encode_usdz_metadata(m)
    assert raw.endswith(b"\n")
    assert json.loads(raw) == {"uuid": "u", "scene_id": "s", "version_string": "v", "n": 3}
    # Round-trip through our loader.
    assert load_usdz_metadata(raw).to_dict() == m.to_dict()


def test_load_usdz_metadata_accepts_utf8_bom() -> None:
    raw = b"\xef\xbb\xbf" + json.dumps(
        {"uuid": "u", "scene_id": "s", "version_string": "v"}
    ).encode("utf-8")
    m = load_usdz_metadata(raw)
    assert m.uuid == "u"
