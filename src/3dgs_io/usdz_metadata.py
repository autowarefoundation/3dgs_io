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
    extras:
        Free-form additional fields written alongside the required trio.
        Values must be JSON-serialisable (``str`` / ``int`` / ``float`` /
        ``bool`` / ``list`` / ``dict`` / ``None``). Extras must not shadow the
        required keys.
    """

    uuid: str
    scene_id: str
    version_string: str
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

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary of all fields."""
        out: dict[str, Any] = {
            "uuid": self.uuid,
            "scene_id": self.scene_id,
            "version_string": self.version_string,
        }
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
        extras = {k: v for k, v in data.items() if k not in USDZ_METADATA_REQUIRED_KEYS}
        return cls(
            uuid=data["uuid"],
            scene_id=data["scene_id"],
            version_string=data["version_string"],
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
    extras: dict[str, Any] | None = None,
) -> UsdzMetadata:
    """Build a :class:`UsdzMetadata` filling unset fields with sensible defaults.

    Defaults are:

    - ``uuid`` → a fresh random UUID4.
    - ``scene_id`` → the output USDZ filename stem (``out_path.stem``).
    - ``version_string`` → ``"3dgs_io/<installed-package-version>"``.
    """
    out_path = Path(out_path)
    return UsdzMetadata(
        uuid=uuid or default_uuid(),
        scene_id=scene_id or out_path.stem,
        version_string=version_string or f"3dgs_io/{_get_package_version()}",
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
