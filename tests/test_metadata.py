from __future__ import annotations

import importlib
import json

import pytest

_mod = importlib.import_module("3dgs_io")
Checkpoint = _mod.Checkpoint
DatasetType = _mod.DatasetType
Export = _mod.Export
GlbMetadata = _mod.GlbMetadata
Model = _mod.Model
Placement = _mod.Placement
TrainingData = _mod.TrainingData
parse_metadata = _mod.parse_metadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metadata(
    *,
    task: str | None = "my_task",
    exp_name: str | None = "exp_001",
    max_sh_degree: int | None = None,
    object_keys: list[str] | None = None,
) -> GlbMetadata:
    return GlbMetadata(
        dataset_type=DatasetType.T4_DATASET,
        generator="test-generator",
        training_data=TrainingData(
            source_path="/data/t4/scene_001",
            data_type="T4",
            revision="abc123",
            scene_index=0,
            lidar_channel="LIDAR_CONCAT",
            selected_frames=[0, 60],
            cameras=["cam_front"],
            camera_channels=["cam_front_channel"],
            start_timestamp_us=1000000,
            end_timestamp_us=2000000,
            task=task,
            exp_name=exp_name,
        ),
        checkpoint=Checkpoint(path="/ckpt/iter_30000.pth", iteration=30000),
        export=Export(
            background_only=False,
            spz_compression=True,
            max_sh_degree=max_sh_degree,
            object_keys=object_keys,
        ),
        model=Model(total_gaussians=50000),
        placement=Placement(lat=35.6812, lon=139.7671, height=40.0),
    )


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------


def test_to_dict_structure() -> None:
    meta = _make_metadata()
    d = meta.to_dict()

    assert d["dataset_type"] == "t4_dataset"
    assert d["generator"] == "test-generator"
    assert d["training_data"]["source_path"] == "/data/t4/scene_001"
    assert d["checkpoint"]["iteration"] == 30000
    assert d["export"]["spz_compression"] is True
    assert d["model"]["total_gaussians"] == 50000
    assert d["placement"]["lat"] == pytest.approx(35.6812)


def test_to_dict_strips_none() -> None:
    meta = _make_metadata(task=None, exp_name=None)
    d = meta.to_dict()

    assert "task" not in d["training_data"]
    assert "exp_name" not in d["training_data"]
    assert "max_sh_degree" not in d["export"]
    assert "object_keys" not in d["export"]


def test_to_dict_keeps_optional_when_present() -> None:
    meta = _make_metadata(max_sh_degree=3, object_keys=["car", "person"])
    d = meta.to_dict()

    assert d["export"]["max_sh_degree"] == 3
    assert d["export"]["object_keys"] == ["car", "person"]


def test_to_dict_json_serializable() -> None:
    meta = _make_metadata(max_sh_degree=2, object_keys=["tree"])
    d = meta.to_dict()
    s = json.dumps(d)
    assert isinstance(s, str)
    roundtripped = json.loads(s)
    assert roundtripped == d


# ---------------------------------------------------------------------------
# from_dict
# ---------------------------------------------------------------------------


def test_from_dict_full() -> None:
    original = _make_metadata(max_sh_degree=2, object_keys=["building"])
    d = original.to_dict()
    restored = GlbMetadata.from_dict(d)

    assert restored == original
    assert isinstance(restored.dataset_type, DatasetType)
    assert restored.dataset_type is DatasetType.T4_DATASET


def test_from_dict_optional_absent() -> None:
    meta = _make_metadata(task=None, exp_name=None)
    d = meta.to_dict()
    # None keys are stripped, so they won't be in the dict
    assert "task" not in d["training_data"]

    restored = GlbMetadata.from_dict(d)
    assert restored.training_data.task is None
    assert restored.training_data.exp_name is None
    assert restored.export.max_sh_degree is None
    assert restored.export.object_keys is None


def test_from_dict_ignores_unknown_keys() -> None:
    meta = _make_metadata()
    d = meta.to_dict()
    d["training_data"]["unknown_future_field"] = "some_value"
    d["extra_section"] = {"foo": "bar"}

    # Should not raise — unknown keys in sections are silently dropped
    restored = GlbMetadata.from_dict(d)
    assert restored.training_data.source_path == "/data/t4/scene_001"


def test_from_dict_missing_required_raises() -> None:
    meta = _make_metadata()
    d = meta.to_dict()
    del d["training_data"]["source_path"]

    with pytest.raises(TypeError):
        GlbMetadata.from_dict(d)


# ---------------------------------------------------------------------------
# roundtrip
# ---------------------------------------------------------------------------


def test_roundtrip() -> None:
    original = _make_metadata(task="training_v2", exp_name="run_42", max_sh_degree=3)
    restored = GlbMetadata.from_dict(original.to_dict())
    assert restored == original


# ---------------------------------------------------------------------------
# parse_metadata
# ---------------------------------------------------------------------------


def test_parse_metadata_with_valid_schema() -> None:
    meta = _make_metadata()
    d = meta.to_dict()
    result = parse_metadata(d)
    assert isinstance(result, GlbMetadata)
    assert result == meta


def test_parse_metadata_with_legacy_dict() -> None:
    legacy = {"dataset_type": "t4_dataset", "dataset_id": "scene_001"}
    result = parse_metadata(legacy)
    assert isinstance(result, dict)
    assert result == legacy


def test_parse_metadata_with_none() -> None:
    assert parse_metadata(None) is None
