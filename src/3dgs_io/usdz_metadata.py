"""USDZ scene-bundle manifest (``metadata.yaml``).

Every scene USDZ produced by :func:`3dgs_io.save_scene_usdz` writes a
``metadata.yaml`` at the archive root as a stability commitment to
downstream consumers. It carries at least a ``uuid``, ``scene_id`` and
``version_string`` — the identity card downstream tools rely on to route or
tag an asset.

Schema
------

Required keys (all non-empty strings):

- ``uuid`` — globally unique identifier for this scene asset.
- ``scene_id`` — human-readable scene identifier (typically the dataset or
  run name).
- ``version_string`` — free-form identifier of the producing pipeline
  (e.g. the ``3dgs_io`` release, or the parent pipeline's own version).

Additional keys may be present; downstream consumers should ignore keys they
don't recognise.

Encoding
--------

The file is written as JSON, which is a subset of YAML 1.2, so consumers may
parse it with ``yaml.safe_load`` and get an ordinary ``dict``::

    with zipfile.ZipFile(usdz_file, "r") as zf, zf.open("metadata.yaml") as f:
        data = yaml.safe_load(f)
        uuid = data["uuid"]
        scene_id = data["scene_id"]
        version = data.get("version_string", "unknown")

Writing JSON avoids pulling PyYAML into ``3dgs_io``'s own dependency set
while remaining a valid ``metadata.yaml`` for downstream tools.
"""

from __future__ import annotations

import json
import uuid as _uuid
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any

__all__ = [
    "USDZ_METADATA_ALPASIM_OPTIONAL_KEYS",
    "USDZ_METADATA_ARCHIVE_PATH",
    "USDZ_METADATA_REQUIRED_KEYS",
    "UsdzMetadata",
    "default_uuid",
    "encode_usdz_metadata",
    "load_usdz_metadata",
    "make_default_metadata",
]

USDZ_METADATA_ARCHIVE_PATH = "metadata.yaml"
USDZ_METADATA_REQUIRED_KEYS: tuple[str, ...] = ("uuid", "scene_id", "version_string")

# alpasim's ``scene_metadata`` block carries these optional fields alongside
# the required trio above. They are first-class on :class:`UsdzMetadata` so
# producers don't have to route them through ``extras`` (which works but is
# untyped and easy to typo). Structure is not validated here — alpasim owns
# the schema — but each field, when set, is written at the top level of the
# emitted dict, in the same slot ``extras`` uses.
USDZ_METADATA_ALPASIM_OPTIONAL_KEYS: tuple[str, ...] = (
    "training_date",
    "dataset_hash",
    "is_resumable",
    "sensors",
    "logger",
    "time_range",
)


@dataclass
class UsdzMetadata:
    """Identity card written at the root of every USDZ scene bundle.

    Parameters
    ----------
    uuid:
        Globally unique identifier for this scene asset. Non-empty string.
    scene_id:
        Human-readable scene identifier (typically the dataset or run name).
        Non-empty string.
    version_string:
        Free-form identifier of the producing pipeline (e.g. the ``3dgs_io``
        release, or the parent pipeline's own version). Non-empty string.
    training_date, dataset_hash, is_resumable, sensors, logger, time_range:
        Optional alpasim ``scene_metadata`` fields. When set (not ``None``)
        they are emitted at the top level of the metadata dict, in the same
        slot ``extras`` uses. Structure is not validated — alpasim owns the
        schema. Producers that don't care about alpasim can leave these at
        ``None``; the emitted dict is unchanged in that case.
    extras:
        Free-form additional fields written alongside the required trio and
        the alpasim block. Values must be JSON-serialisable (``str`` /
        ``int`` / ``float`` / ``bool`` / ``list`` / ``dict`` / ``None``).
        Extras must not shadow the required keys or any alpasim key that is
        also set on the dataclass; use the typed field instead.
    """

    uuid: str
    scene_id: str
    version_string: str
    training_date: str | None = None
    dataset_hash: str | None = None
    is_resumable: bool | None = None
    sensors: Any = None
    logger: str | None = None
    time_range: Any = None
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key in USDZ_METADATA_REQUIRED_KEYS:
            val = getattr(self, key)
            if not isinstance(val, str) or not val:
                raise ValueError(f"UsdzMetadata.{key} must be a non-empty string, got {val!r}")
        overlap = set(USDZ_METADATA_REQUIRED_KEYS) & set(self.extras)
        if overlap:
            raise ValueError(
                f"UsdzMetadata.extras must not shadow required keys: {sorted(overlap)}"
            )
        # extras may not shadow an alpasim field that is also set on the
        # dataclass — that would silently drop one of the two on emit.
        typed_alpasim_set = {
            k for k in USDZ_METADATA_ALPASIM_OPTIONAL_KEYS if getattr(self, k) is not None
        }
        alpasim_overlap = typed_alpasim_set & set(self.extras)
        if alpasim_overlap:
            raise ValueError(
                "UsdzMetadata.extras must not shadow alpasim fields already set on the "
                f"dataclass: {sorted(alpasim_overlap)}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary of all fields."""
        out: dict[str, Any] = {
            "uuid": self.uuid,
            "scene_id": self.scene_id,
            "version_string": self.version_string,
        }
        for key in USDZ_METADATA_ALPASIM_OPTIONAL_KEYS:
            val = getattr(self, key)
            if val is not None:
                out[key] = val
        out.update(self.extras)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UsdzMetadata:
        """Reconstruct a :class:`UsdzMetadata` from a plain dictionary.

        Raises :class:`ValueError` if any required key is missing.
        """
        if not isinstance(data, dict):
            raise ValueError(f"metadata document must be a mapping, got {type(data).__name__}")
        missing = [k for k in USDZ_METADATA_REQUIRED_KEYS if k not in data]
        if missing:
            raise ValueError(f"metadata document missing required keys: {missing}")
        known = set(USDZ_METADATA_REQUIRED_KEYS) | set(USDZ_METADATA_ALPASIM_OPTIONAL_KEYS)
        extras = {k: v for k, v in data.items() if k not in known}
        return cls(
            uuid=data["uuid"],
            scene_id=data["scene_id"],
            version_string=data["version_string"],
            training_date=data.get("training_date"),
            dataset_hash=data.get("dataset_hash"),
            is_resumable=data.get("is_resumable"),
            sensors=data.get("sensors"),
            logger=data.get("logger"),
            time_range=data.get("time_range"),
            extras=extras,
        )


def default_uuid() -> str:
    """Return a fresh random UUID4 string suitable for :attr:`UsdzMetadata.uuid`."""
    return str(_uuid.uuid4())


def _get_package_version() -> str:
    try:
        return _pkg_version("3dgs-io")
    except PackageNotFoundError:
        return "unknown"


def make_default_metadata(
    *,
    out_path: str | Path,
    uuid: str | None = None,
    scene_id: str | None = None,
    version_string: str | None = None,
    training_date: str | None = None,
    dataset_hash: str | None = None,
    is_resumable: bool | None = None,
    sensors: Any = None,
    logger: str | None = None,
    time_range: Any = None,
    extras: dict[str, Any] | None = None,
) -> UsdzMetadata:
    """Build a :class:`UsdzMetadata` filling unset fields with sensible defaults.

    Defaults are:

    - ``uuid`` → a fresh random UUID4.
    - ``scene_id`` → the output USDZ filename stem (``out_path.stem``).
    - ``version_string`` → ``"3dgs_io/<installed-package-version>"``.

    The alpasim ``scene_metadata`` fields (``training_date``, ``dataset_hash``,
    ``is_resumable``, ``sensors``, ``logger``, ``time_range``) are passed
    through unchanged and are omitted from the emitted dict when left at
    their default ``None``.
    """
    out_path = Path(out_path)
    return UsdzMetadata(
        uuid=uuid or default_uuid(),
        scene_id=scene_id or out_path.stem,
        version_string=version_string or f"3dgs_io/{_get_package_version()}",
        training_date=training_date,
        dataset_hash=dataset_hash,
        is_resumable=is_resumable,
        sensors=sensors,
        logger=logger,
        time_range=time_range,
        extras=dict(extras) if extras else {},
    )


def encode_usdz_metadata(metadata: UsdzMetadata) -> bytes:
    """Serialise ``metadata`` as UTF-8 bytes for ``metadata.yaml``.

    Output is JSON, which is a subset of YAML 1.2 — ``yaml.safe_load`` on the
    written bytes yields the same document, so consumers relying on PyYAML
    are unaffected.
    """
    return (json.dumps(metadata.to_dict(), indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def load_usdz_metadata(raw: bytes | str) -> UsdzMetadata:
    """Parse ``metadata.yaml`` bytes back into a :class:`UsdzMetadata`.

    Only the JSON-flavoured output produced by :func:`encode_usdz_metadata`
    is understood; hand-written block-style YAML would need PyYAML, which is
    intentionally not a runtime dependency of this package.
    """
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8-sig")
    data = json.loads(raw)
    return UsdzMetadata.from_dict(data)
