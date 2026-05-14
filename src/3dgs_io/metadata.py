from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Any


class DatasetType(str, Enum):
    """Supported dataset types for metadata."""

    T4_DATASET = "t4_dataset"


# ---------------------------------------------------------------------------
# Section dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TrainingData:
    """Training data provenance."""

    source_path: str
    """T4 dataset path or UUID."""

    data_type: str
    """Data type identifier (e.g. ``"T4"``)."""

    revision: str
    """Dataset revision."""

    scene_index: int
    """Scene index within the dataset."""

    lidar_channel: str
    """LiDAR channel used (e.g. ``"LIDAR_CONCAT"``)."""

    selected_frames: list[int]
    """Frame range used for training (e.g. ``[0, 60]``)."""

    cameras: list[str]
    """Camera names used."""

    camera_channels: list[str]
    """Camera channel names used."""

    start_timestamp_us: int
    """Start timestamp of the training frame range in microseconds."""

    end_timestamp_us: int
    """End timestamp of the training frame range in microseconds."""

    task: str | None = None
    """Training task name (optional)."""

    exp_name: str | None = None
    """Experiment name (optional)."""


@dataclass
class Checkpoint:
    """Model checkpoint information."""

    path: str
    """Checkpoint file path."""

    iteration: int
    """Training iteration number."""


@dataclass
class Export:
    """Export configuration."""

    background_only: bool
    """Whether only background was exported."""

    spz_compression: bool
    """Whether SPZ compression was used."""

    max_sh_degree: int | None = None
    """SH degree limit (present only when specified)."""

    object_keys: list[str] | None = None
    """Exported object keys (present only when applicable)."""


@dataclass
class Model:
    """Model statistics."""

    total_gaussians: int
    """Total number of Gaussians."""


@dataclass
class Placement:
    """Geospatial placement in geodetic coordinates."""

    lat: float
    """Latitude in degrees."""

    lon: float
    """Longitude in degrees."""

    height: float
    """Height in metres."""


@dataclass
class GlbMetadata:
    """Typed metadata for GLB files stored in ``asset.extras``.

    Use :meth:`to_dict` to produce a JSON-serializable dictionary and
    :meth:`from_dict` to reconstruct from one.
    """

    dataset_type: DatasetType
    generator: str
    training_data: TrainingData
    checkpoint: Checkpoint
    export: Export
    model: Model
    placement: Placement

    # -- serialization --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary.

        ``None`` values in optional fields are stripped from the output.
        """
        d = dataclasses.asdict(self)
        _strip_none(d)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GlbMetadata:
        """Reconstruct a :class:`GlbMetadata` from a plain dictionary.

        Unknown keys inside each section are silently ignored so that files
        written by a newer schema version can still be read.
        """
        return cls(
            dataset_type=DatasetType(data["dataset_type"]),
            generator=data["generator"],
            training_data=TrainingData(**_pick_fields(TrainingData, data["training_data"])),
            checkpoint=Checkpoint(**_pick_fields(Checkpoint, data["checkpoint"])),
            export=Export(**_pick_fields(Export, data["export"])),
            model=Model(**_pick_fields(Model, data["model"])),
            placement=Placement(**_pick_fields(Placement, data["placement"])),
        )


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def parse_metadata(raw: dict[str, Any] | None) -> GlbMetadata | dict[str, Any] | None:
    """Parse raw ``asset.extras`` metadata.

    Returns a :class:`GlbMetadata` when the dictionary matches the schema.
    Falls back to returning the raw dictionary unchanged for legacy files,
    or ``None`` when no metadata was present.
    """
    if raw is None:
        return None
    try:
        return GlbMetadata.from_dict(raw)
    except (KeyError, TypeError, ValueError):
        return raw


def serialize_metadata(metadata: GlbMetadata | dict[str, Any] | None) -> dict[str, Any] | None:
    """Serialize metadata for writing to ``asset.extras``.

    Accepts both :class:`GlbMetadata` (calls :meth:`~GlbMetadata.to_dict`)
    and plain dictionaries (passed through unchanged).
    """
    if metadata is None:
        return None
    if isinstance(metadata, GlbMetadata):
        return metadata.to_dict()
    return metadata


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _strip_none(d: dict[str, Any]) -> None:
    """Recursively remove keys whose values are ``None``."""
    keys_to_remove = [k for k, v in d.items() if v is None]
    for k in keys_to_remove:
        del d[k]
    for v in d.values():
        if isinstance(v, dict):
            _strip_none(v)


def _pick_fields(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    """Filter *data* to only the fields declared on dataclass *cls*."""
    field_names = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in data.items() if k in field_names}
