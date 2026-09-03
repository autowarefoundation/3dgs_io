"""In-place-like editors for existing single-file USDZ scene bundles.

Supports:

* :func:`add_lanelet2_to_usdz` — embed a lanelet2 ``map.osm`` file into an
  existing USDZ produced by :func:`3dgs_io.save_scene_usdz` and record it in
  ``scene.json.extras.map_lanelet2``.
* :func:`add_ppisp_to_usdz` — embed PPISP appearance-correction parameters
  as ``ppisp.json`` and record it in ``scene.json.extras.ppisp``.
* :func:`update_camera_intrinsics_in_usdz` — rewrite a camera's intrinsics
  inside the ``rig_trajectories.json`` embedded in the USDZ.
* :func:`set_usdz_metadata` — write (or overwrite) ``metadata.yaml`` at the
  archive root so downstream consumers can read a stable identity card
  (``uuid`` / ``scene_id`` / ``version_string``) without rebuilding the
  Gaussian chunks.

Every output archive preserves entry order (``default.usda`` stays first, per
the USDZ spec) and uses ``ZIP_STORED`` for every entry.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import zipfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .ppisp import Ppisp, parse_ppisp, serialize_ppisp
from .rig_trajectories import (
    load_rig_trajectories_doc,
    serialize_rig_trajectories,
    update_camera_intrinsics,
)
from .scene_usdz import (
    AUTOWARE_LANELET2_PATH,
    AUTOWARE_MAP_PREFIX,
    AUTOWARE_POINTCLOUD_DIR_PREFIX,
    AUTOWARE_POINTCLOUD_METADATA_PATH,
    AUTOWARE_POINTCLOUD_PATH,
    autoware_map_extras,
)
from .usdz_metadata import (
    USDZ_METADATA_ARCHIVE_PATH,
    UsdzMetadata,
    encode_usdz_metadata,
    load_usdz_metadata,
    make_default_metadata,
)

__all__ = [
    "EditUsdzResult",
    "IntrinsicsEditResult",
    "MetadataEditResult",
    "add_autoware_map_to_usdz",
    "add_clipgt_to_usdz",
    "add_lanelet2_to_usdz",
    "add_pointcloud_map_to_usdz",
    "add_ppisp_to_usdz",
    "set_usdz_metadata",
    "update_camera_intrinsics_in_usdz",
]

_log = logging.getLogger(__name__)

_LANELET2_ARCHIVE_PATH = AUTOWARE_LANELET2_PATH
_LANELET2_SCENE_KEY = "map_lanelet2"
_RIG_TRAJECTORIES_ARCHIVE_PATH = "rig_trajectories.json"
_CLIPGT_ARCHIVE_PREFIX = "clipgt/"
_PPISP_ARCHIVE_PATH = "ppisp.json"
_PPISP_SCENE_KEY = "ppisp"


@dataclass
class EditUsdzResult:
    """Summary of an :func:`add_lanelet2_to_usdz` invocation."""

    out_path: Path
    added: list[str] = field(default_factory=list)
    replaced: list[str] = field(default_factory=list)


@dataclass
class IntrinsicsEditResult:
    """Summary of an :func:`update_camera_intrinsics_in_usdz` invocation."""

    out_path: Path
    camera_name: str
    camera_model: dict[str, Any]
    updated_fields: list[str]
    replaced: list[str] = field(default_factory=list)


@dataclass
class MetadataEditResult:
    """Summary of a :func:`set_usdz_metadata` invocation."""

    out_path: Path
    metadata: dict[str, Any]
    added: list[str] = field(default_factory=list)
    replaced: list[str] = field(default_factory=list)


def add_lanelet2_to_usdz(
    input_usdz: str | Path,
    output_usdz: str | Path,
    lanelet2_path: str | Path,
    *,
    overwrite: bool = True,
) -> EditUsdzResult:
    """Add a lanelet2 ``.osm`` file to an existing USDZ scene bundle.

    The source file is embedded verbatim at archive path
    ``autoware_map/lanelet2_map.osm`` — mirroring Autoware's own map directory
    layout — and ``scene.json.extras.map_lanelet2`` is set to that path. All
    other archive entries are copied through in their original order.

    Parameters
    ----------
    input_usdz:
        Path to an existing ``.usdz`` produced by
        :func:`3dgs_io.save_scene_usdz` (must contain ``scene.json``).
    output_usdz:
        Destination ``.usdz`` path. May equal ``input_usdz`` — in that case
        the input is replaced atomically once the new archive is fully
        written.
    lanelet2_path:
        Source ``.osm`` file to embed at ``autoware_map/lanelet2_map.osm``.
    overwrite:
        When ``True`` (default), replace an existing
        ``autoware_map/lanelet2_map.osm`` entry. When ``False``, raise
        :class:`ValueError` if the archive already contains one.
    """
    input_usdz = Path(input_usdz).expanduser()
    output_usdz = Path(output_usdz).expanduser()
    lanelet2_path = Path(lanelet2_path).expanduser()

    if not input_usdz.is_file():
        raise FileNotFoundError(f"input USDZ not found: {input_usdz}")
    if not lanelet2_path.is_file():
        raise FileNotFoundError(f"lanelet2 file not found: {lanelet2_path}")

    lanelet2_bytes = lanelet2_path.read_bytes()

    with zipfile.ZipFile(input_usdz, "r") as zin:
        names = zin.namelist()
        if "scene.json" not in names:
            raise ValueError(f"{input_usdz} is not a splatsim scene USDZ (missing scene.json)")
        if _LANELET2_ARCHIVE_PATH in names and not overwrite:
            raise ValueError(
                f"{input_usdz} already contains {_LANELET2_ARCHIVE_PATH!r}; "
                "pass overwrite=True to replace"
            )
        scene_bytes = _rewrite_scene_extras(
            zin.read("scene.json"), _LANELET2_SCENE_KEY, _LANELET2_ARCHIVE_PATH
        )

    added, replaced = _repack_usdz(
        input_usdz,
        output_usdz,
        entries_to_write={
            "scene.json": scene_bytes,
            _LANELET2_ARCHIVE_PATH: lanelet2_bytes,
        },
    )
    # scene.json is always a replacement (validated to exist above), don't
    # surface it as a user-facing edit.
    replaced = [n for n in replaced if n != "scene.json"]
    return EditUsdzResult(out_path=output_usdz, added=added, replaced=replaced)


def add_clipgt_to_usdz(
    input_usdz: str | Path,
    output_usdz: str | Path,
    clipgt_dir: str | Path,
    *,
    overwrite: bool = True,
) -> EditUsdzResult:
    """Embed a clipgt vector-map directory into ``input_usdz`` under ``clipgt/``.

    Every regular file under ``clipgt_dir`` (recursively) is written into the
    output USDZ at ``clipgt/<relative_path>``. Runtimes that key on the
    ``clipgt/`` prefix (see alpasim ``artifact._extract_map_directories``) will
    pick up the map without any scene.json update.

    Parameters
    ----------
    input_usdz, output_usdz, clipgt_dir:
        Filesystem paths. ``output_usdz`` may equal ``input_usdz``.
    overwrite:
        If ``False`` and the input already contains ``clipgt/`` entries, raise
        ``FileExistsError``. Defaults to ``True``.
    """
    input_usdz = Path(input_usdz).expanduser()
    output_usdz = Path(output_usdz).expanduser()
    clipgt_dir = Path(clipgt_dir).expanduser()

    if not input_usdz.is_file():
        raise FileNotFoundError(f"Input USDZ not found: {input_usdz}")
    if not clipgt_dir.is_dir():
        raise FileNotFoundError(f"clipgt directory not found: {clipgt_dir}")

    entries_to_write: dict[str, bytes | Path] = dict(
        _dir_entries(clipgt_dir, _CLIPGT_ARCHIVE_PREFIX)
    )
    if not entries_to_write:
        raise ValueError(f"No files found under {clipgt_dir}")

    if not overwrite:
        with zipfile.ZipFile(input_usdz, "r") as zin:
            existing = [n for n in zin.namelist() if n.startswith(_CLIPGT_ARCHIVE_PREFIX)]
        if existing:
            raise FileExistsError(
                f"{input_usdz} already has clipgt/ entries; pass overwrite=True to replace"
            )

    added, replaced = _repack_usdz(
        input_usdz,
        output_usdz,
        entries_to_write=entries_to_write,
    )
    return EditUsdzResult(out_path=output_usdz, added=added, replaced=replaced)


def add_pointcloud_map_to_usdz(
    input_usdz: str | Path,
    output_usdz: str | Path,
    pointcloud: str | Path,
    *,
    metadata_yaml: str | Path | None = None,
    overwrite: bool = True,
) -> EditUsdzResult:
    """Add an Autoware point-cloud map to an existing USDZ scene bundle.

    ``pointcloud`` may be either:

    * a single ``.pcd`` file — embedded at
      ``autoware_map/pointcloud_map.pcd`` with
      ``scene.json.extras.map_pointcloud`` set to that path; or
    * a directory of split ``.pcd`` tiles — every regular file is embedded
      under ``autoware_map/pointcloud_map/<relative_path>`` and
      ``extras.map_pointcloud`` records the directory prefix
      ``"autoware_map/pointcloud_map/"``.

    When ``metadata_yaml`` is given (Autoware's ``pointcloud_map_metadata.yaml``
    that lists the split tiles), it is embedded at
    ``autoware_map/pointcloud_map_metadata.yaml`` and recorded under
    ``extras.map_pointcloud_metadata``.

    Parameters
    ----------
    input_usdz, output_usdz:
        Same semantics as :func:`add_lanelet2_to_usdz` — ``output_usdz`` may
        equal ``input_usdz`` for atomic in-place edits.
    pointcloud:
        A ``.pcd`` file or a directory of ``.pcd`` tiles.
    metadata_yaml:
        Optional ``pointcloud_map_metadata.yaml`` describing the split tiles.
    overwrite:
        When ``True`` (default), replace any existing point-cloud-map entries.
        When ``False``, raise :class:`FileExistsError` if the archive already
        carries a point-cloud map.
    """
    input_usdz = Path(input_usdz).expanduser()
    output_usdz = Path(output_usdz).expanduser()
    pointcloud = Path(pointcloud).expanduser()

    if not input_usdz.is_file():
        raise FileNotFoundError(f"input USDZ not found: {input_usdz}")
    if not pointcloud.exists():
        raise FileNotFoundError(f"point-cloud map not found: {pointcloud}")

    entries_to_write: dict[str, bytes | Path] = {}
    if pointcloud.is_dir():
        entries_to_write.update(_dir_entries(pointcloud, AUTOWARE_POINTCLOUD_DIR_PREFIX))
        if not entries_to_write:
            raise ValueError(f"No files found under {pointcloud}")
    else:
        entries_to_write[AUTOWARE_POINTCLOUD_PATH] = pointcloud

    if metadata_yaml is not None:
        metadata_yaml = Path(metadata_yaml).expanduser()
        if not metadata_yaml.is_file():
            raise FileNotFoundError(f"pointcloud metadata not found: {metadata_yaml}")
        entries_to_write[AUTOWARE_POINTCLOUD_METADATA_PATH] = metadata_yaml

    def _conflict(name: str) -> bool:
        return (
            name == AUTOWARE_POINTCLOUD_PATH
            or name.startswith(AUTOWARE_POINTCLOUD_DIR_PREFIX)
            or (metadata_yaml is not None and name == AUTOWARE_POINTCLOUD_METADATA_PATH)
        )

    # The writer's detection path (autoware_map_extras) owns the file→extras-key
    # rule, including the single-file vs split-directory choice; reuse it so the
    # editor and writer can never disagree.
    scene_updates = autoware_map_extras(set(entries_to_write))
    return _embed_map_with_extras(
        input_usdz,
        output_usdz,
        entries_to_write,
        scene_updates,
        conflict=_conflict,
        overwrite=overwrite,
    )


def add_autoware_map_to_usdz(
    input_usdz: str | Path,
    output_usdz: str | Path,
    map_dir: str | Path,
    *,
    overwrite: bool = True,
) -> EditUsdzResult:
    """Embed a whole Autoware map directory into a USDZ under ``autoware_map/``.

    Every regular file under ``map_dir`` (recursively) is copied into the
    output USDZ at ``autoware_map/<relative_path>``, reproducing Autoware's own
    map directory layout so a consumer can extract ``autoware_map/`` and hand
    it to ``autoware_map_loader`` unchanged. Well-known files
    (``lanelet2_map.osm``, ``pointcloud_map.pcd`` or a ``pointcloud_map/``
    directory, ``pointcloud_map_metadata.yaml``, ``map_projector_info.yaml``)
    are auto-recorded in the corresponding ``scene.json.extras`` keys.

    Parameters
    ----------
    input_usdz, output_usdz:
        Same semantics as :func:`add_lanelet2_to_usdz` — ``output_usdz`` may
        equal ``input_usdz`` for atomic in-place edits.
    map_dir:
        An Autoware map directory (containing e.g. ``lanelet2_map.osm``).
    overwrite:
        When ``True`` (default), replace any existing ``autoware_map/``
        entries. When ``False``, raise :class:`FileExistsError` if the archive
        already carries an ``autoware_map/`` tree.
    """
    input_usdz = Path(input_usdz).expanduser()
    output_usdz = Path(output_usdz).expanduser()
    map_dir = Path(map_dir).expanduser()

    if not input_usdz.is_file():
        raise FileNotFoundError(f"input USDZ not found: {input_usdz}")
    if not map_dir.is_dir():
        raise FileNotFoundError(f"autoware map directory not found: {map_dir}")

    entries_to_write: dict[str, bytes | Path] = dict(_dir_entries(map_dir, AUTOWARE_MAP_PREFIX))
    if not entries_to_write:
        raise ValueError(f"No files found under {map_dir}")

    scene_updates = autoware_map_extras(set(entries_to_write))
    return _embed_map_with_extras(
        input_usdz,
        output_usdz,
        entries_to_write,
        scene_updates,
        conflict=lambda name: name.startswith(AUTOWARE_MAP_PREFIX),
        overwrite=overwrite,
    )


def add_ppisp_to_usdz(
    input_usdz: str | Path,
    output_usdz: str | Path,
    ppisp: Ppisp | str | Path,
    *,
    overwrite: bool = True,
) -> EditUsdzResult:
    """Add PPISP appearance-correction parameters to an existing USDZ scene bundle.

    The PPISP payload is serialised into ``ppisp.json`` at the archive root
    (schema ``splatsim.ppisp/v1``) and ``scene.json.extras.ppisp`` is set to
    ``"ppisp.json"``. All other archive entries are copied through in their
    original order.

    Parameters
    ----------
    input_usdz:
        Path to an existing ``.usdz`` produced by
        :func:`3dgs_io.save_scene_usdz` (must contain ``scene.json``).
    output_usdz:
        Destination ``.usdz`` path. May equal ``input_usdz`` for atomic
        in-place edits.
    ppisp:
        Either an in-memory :class:`Ppisp` object or a filesystem path to a
        JSON document following the ``splatsim.ppisp/v1`` schema. Path
        inputs are parsed via :func:`3dgs_io.parse_ppisp`.
    overwrite:
        When ``True`` (default), replace an existing ``ppisp.json`` entry.
        When ``False``, raise :class:`FileExistsError` if the archive
        already contains one.
    """
    input_usdz = Path(input_usdz).expanduser()
    output_usdz = Path(output_usdz).expanduser()

    if not input_usdz.is_file():
        raise FileNotFoundError(f"input USDZ not found: {input_usdz}")

    if isinstance(ppisp, Ppisp):
        ppisp_obj = ppisp
    else:
        ppisp_path = Path(ppisp).expanduser()
        if not ppisp_path.is_file():
            raise FileNotFoundError(f"ppisp JSON file not found: {ppisp_path}")
        ppisp_obj = parse_ppisp(json.loads(ppisp_path.read_text(encoding="utf-8-sig")))

    ppisp_bytes = (json.dumps(serialize_ppisp(ppisp_obj), indent=2) + "\n").encode("utf-8")

    with zipfile.ZipFile(input_usdz, "r") as zin:
        names = zin.namelist()
        if "scene.json" not in names:
            raise ValueError(f"{input_usdz} is not a splatsim scene USDZ (missing scene.json)")
        if _PPISP_ARCHIVE_PATH in names and not overwrite:
            raise FileExistsError(
                f"{input_usdz} already contains {_PPISP_ARCHIVE_PATH!r}; "
                "pass overwrite=True to replace"
            )
        scene_bytes = _rewrite_scene_extras(
            zin.read("scene.json"), _PPISP_SCENE_KEY, _PPISP_ARCHIVE_PATH
        )

    added, replaced = _repack_usdz(
        input_usdz,
        output_usdz,
        entries_to_write={
            "scene.json": scene_bytes,
            _PPISP_ARCHIVE_PATH: ppisp_bytes,
        },
    )
    # scene.json is always a pre-existing replacement, not a user-facing edit.
    replaced = [n for n in replaced if n != "scene.json"]
    return EditUsdzResult(out_path=output_usdz, added=added, replaced=replaced)


def update_camera_intrinsics_in_usdz(
    input_usdz: str | Path,
    output_usdz: str | Path,
    *,
    camera_name: str,
    rig_id: str | None = None,
    **intrinsic_updates: Any,
) -> IntrinsicsEditResult:
    """Rewrite a camera's intrinsics inside a USDZ's ``rig_trajectories.json``.

    The USDZ must contain a ``rig_trajectories.json`` embedded at the archive
    root (produced by :func:`3dgs_io.save_scene_usdz` with a
    ``rig_trajectories`` extra). The file is loaded, the camera addressed by
    ``camera_name`` (and optionally ``rig_id``) is updated via
    :func:`3dgs_io.rig_trajectories.update_camera_intrinsics`, and the result
    is written back in the same frame-explicit v2 layout.

    Parameters
    ----------
    input_usdz, output_usdz:
        Same semantics as :func:`add_lanelet2_to_usdz`.
    camera_name:
        Value of :attr:`Camera.name` to locate.
    rig_id:
        Restrict the search to this rig id (required when multiple rigs share
        the camera name).
    **intrinsic_updates:
        Keyword arguments forwarded to
        :func:`3dgs_io.rig_trajectories.update_camera_intrinsics`
        (``width``, ``height``, ``fx``, ``fy``, ``cx``, ``cy``,
        ``distortion_coeffs``, ``principal_point``, ``shutter_type``,
        ``reference_poly``). At least one is required.
    """
    if not intrinsic_updates:
        raise ValueError("at least one intrinsic update must be provided")

    input_usdz = Path(input_usdz).expanduser()
    output_usdz = Path(output_usdz).expanduser()

    if not input_usdz.is_file():
        raise FileNotFoundError(f"input USDZ not found: {input_usdz}")

    with zipfile.ZipFile(input_usdz, "r") as zin:
        names = zin.namelist()
        if "scene.json" not in names:
            raise ValueError(f"{input_usdz} is not a splatsim scene USDZ (missing scene.json)")
        if _RIG_TRAJECTORIES_ARCHIVE_PATH not in names:
            raise ValueError(f"{input_usdz} does not contain {_RIG_TRAJECTORIES_ARCHIVE_PATH!r}")
        rig_doc = json.loads(zin.read(_RIG_TRAJECTORIES_ARCHIVE_PATH).decode("utf-8-sig"))

    rigs = load_rig_trajectories_doc(rig_doc)
    cam = update_camera_intrinsics(
        rigs, camera_name=camera_name, rig_id=rig_id, **intrinsic_updates
    )
    new_rig_doc = serialize_rig_trajectories(rigs)
    new_rig_bytes = (json.dumps(new_rig_doc, indent=2) + "\n").encode("utf-8")

    _added, replaced = _repack_usdz(
        input_usdz,
        output_usdz,
        entries_to_write={_RIG_TRAJECTORIES_ARCHIVE_PATH: new_rig_bytes},
    )

    return IntrinsicsEditResult(
        out_path=output_usdz,
        camera_name=cam.name,
        camera_model=cam.camera_model.to_dict(),
        updated_fields=sorted(intrinsic_updates),
        replaced=replaced,
    )


def set_usdz_metadata(
    input_usdz: str | Path,
    output_usdz: str | Path,
    *,
    uuid: str | None = None,
    scene_id: str | None = None,
    version_string: str | None = None,
    extras: dict[str, Any] | None = None,
) -> MetadataEditResult:
    """Write (or overwrite) ``metadata.yaml`` at the root of an existing USDZ.

    Any field not explicitly passed is inherited from the input's existing
    ``metadata.yaml`` (if one is present and parseable). Fields that remain
    unset after the merge fall back to the same defaults as
    :func:`3dgs_io.save_scene_usdz`:

    - ``uuid`` → a fresh random UUID4.
    - ``scene_id`` → the *output* USDZ filename stem.
    - ``version_string`` → ``"3dgs_io/<installed-package-version>"``.

    Gaussian chunks and every other archive entry are copied through
    unchanged; only ``metadata.yaml`` is added / replaced.

    Parameters
    ----------
    input_usdz, output_usdz:
        Same semantics as :func:`add_lanelet2_to_usdz` — ``output_usdz`` may
        equal ``input_usdz`` for atomic in-place edits.
    uuid, scene_id, version_string:
        Overrides for the three required manifest keys. Non-empty strings.
    extras:
        Free-form additional keys to merge into the manifest. Values must be
        JSON-serialisable and must not shadow the required keys.
    """
    input_usdz = Path(input_usdz).expanduser()
    output_usdz = Path(output_usdz).expanduser()

    for name, value in (("uuid", uuid), ("scene_id", scene_id), ("version_string", version_string)):
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"{name} override must be a non-empty string, got {value!r}")

    if not input_usdz.is_file():
        raise FileNotFoundError(f"input USDZ not found: {input_usdz}")

    existing: UsdzMetadata | None = None
    with zipfile.ZipFile(input_usdz, "r") as zin:
        if USDZ_METADATA_ARCHIVE_PATH in zin.namelist():
            try:
                existing = load_usdz_metadata(zin.read(USDZ_METADATA_ARCHIVE_PATH))
            except (ValueError, json.JSONDecodeError) as exc:
                _log.warning(
                    "%s: existing %s could not be parsed (%s); regenerating from scratch.",
                    input_usdz,
                    USDZ_METADATA_ARCHIVE_PATH,
                    exc,
                )

    merged_extras: dict[str, Any] = dict(existing.extras) if existing is not None else {}
    if extras:
        merged_extras.update(extras)

    new_metadata = make_default_metadata(
        out_path=output_usdz,
        uuid=uuid or (existing.uuid if existing is not None else None),
        scene_id=scene_id or (existing.scene_id if existing is not None else None),
        version_string=version_string
        or (existing.version_string if existing is not None else None),
        extras=merged_extras,
    )
    payload = encode_usdz_metadata(new_metadata)

    added, replaced = _repack_usdz(
        input_usdz,
        output_usdz,
        entries_to_write={USDZ_METADATA_ARCHIVE_PATH: payload},
    )

    return MetadataEditResult(
        out_path=output_usdz,
        metadata=new_metadata.to_dict(),
        added=added,
        replaced=replaced,
    )


def _rewrite_scene_extras(scene_json_bytes: bytes, key: str, value: str) -> bytes:
    """Load ``scene.json``, set ``extras[key] = value``, return re-encoded bytes."""
    return _rewrite_scene_extras_multi(scene_json_bytes, {key: value})


def _rewrite_scene_extras_multi(scene_json_bytes: bytes, updates: dict[str, str]) -> bytes:
    """Load ``scene.json``, merge ``updates`` into ``extras``, re-encode."""
    scene_doc = json.loads(scene_json_bytes.decode("utf-8-sig"))
    if not isinstance(scene_doc, dict):
        raise ValueError("scene.json is not a JSON object")
    extras = scene_doc.get("extras")
    if extras is None:
        extras = {}
        scene_doc["extras"] = extras
    elif not isinstance(extras, dict):
        raise ValueError("scene.json.extras is not an object")
    extras.update(updates)
    return json.dumps(scene_doc, indent=2).encode("utf-8")


def _dir_entries(root: Path, prefix: str) -> dict[str, Path]:
    """Map every regular file under ``root`` to ``{prefix}<rel>`` → source path.

    Sources are returned as :class:`~pathlib.Path` values so :func:`_repack_usdz`
    streams them from disk instead of slurping (potentially multi-GB) map files
    into memory.
    """
    out: dict[str, Path] = {}
    for file_path in sorted(root.rglob("*")):
        if file_path.is_file():
            rel = file_path.relative_to(root).as_posix()
            out[f"{prefix}{rel}"] = file_path
    return out


def _embed_map_with_extras(
    input_usdz: Path,
    output_usdz: Path,
    entries_to_write: dict[str, bytes | Path],
    scene_updates: dict[str, str],
    *,
    conflict: Callable[[str], bool],
    overwrite: bool,
) -> EditUsdzResult:
    """Embed ``entries_to_write`` and merge ``scene_updates`` into ``scene.json``.

    Shared tail for the map editors: validate ``scene.json`` is present, reject
    clashing entries (per the ``conflict`` predicate) unless ``overwrite``,
    rewrite the ``extras`` block, and repack. ``scene.json`` is always a
    pre-existing replacement, so it is stripped from the reported ``replaced``.
    """
    with zipfile.ZipFile(input_usdz, "r") as zin:
        names = zin.namelist()
        if "scene.json" not in names:
            raise ValueError(f"{input_usdz} is not a splatsim scene USDZ (missing scene.json)")
        if not overwrite:
            clashing = sorted(n for n in names if conflict(n))
            if clashing:
                raise FileExistsError(
                    f"{input_usdz} already contains conflicting entries "
                    f"({clashing}); pass overwrite=True to replace"
                )
        scene_bytes = _rewrite_scene_extras_multi(zin.read("scene.json"), scene_updates)

    entries_to_write["scene.json"] = scene_bytes
    added, replaced = _repack_usdz(input_usdz, output_usdz, entries_to_write=entries_to_write)
    replaced = [n for n in replaced if n != "scene.json"]
    return EditUsdzResult(out_path=output_usdz, added=added, replaced=replaced)


def _repack_usdz(
    input_usdz: Path,
    output_usdz: Path,
    *,
    entries_to_write: dict[str, bytes | Path],
) -> tuple[list[str], list[str]]:
    """Copy ``input_usdz`` to ``output_usdz`` replacing/appending given entries.

    Original entry order is preserved, entries in ``entries_to_write`` that
    already exist are replaced in place, and new entries are appended after
    the copied ones. Values may be ``bytes`` (written in memory) or a
    :class:`~pathlib.Path` (streamed from disk). Every entry is written
    ``ZIP_STORED``. Safe when ``output_usdz == input_usdz`` (writes to a
    sibling temp file, then :func:`os.replace`).
    """
    output_usdz.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(input_usdz, "r") as zin:
        names = zin.namelist()
        replaced = sorted(n for n in entries_to_write if n in names)
        added = sorted(n for n in entries_to_write if n not in names)

        with tempfile.NamedTemporaryFile(
            "wb",
            delete=False,
            dir=output_usdz.parent,
            prefix=output_usdz.stem + ".",
            suffix=".usdz.tmp",
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_STORED, allowZip64=True) as zout:
                seen: set[str] = set()
                for name in names:
                    if name in seen:
                        continue
                    seen.add(name)
                    if name in entries_to_write:
                        _write_entry(zout, name, entries_to_write[name])
                    else:
                        _write_stored(zout, name, zin.read(name))
                for name in added:
                    _write_entry(zout, name, entries_to_write[name])
            os.replace(tmp_path, output_usdz)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    return added, replaced


def _write_entry(zf: zipfile.ZipFile, name: str, source: bytes | Path) -> None:
    """Write ``source`` at ``name`` as ``ZIP_STORED`` — streamed if it's a Path."""
    if isinstance(source, Path):
        zf.write(source, name, compress_type=zipfile.ZIP_STORED)
    else:
        _write_stored(zf, name, source)


def _write_stored(zf: zipfile.ZipFile, name: str, content: bytes) -> None:
    zi = zipfile.ZipInfo(name)
    zi.compress_type = zipfile.ZIP_STORED
    zf.writestr(zi, content)


def _result_summary(
    result: EditUsdzResult | IntrinsicsEditResult | MetadataEditResult,
) -> dict[str, Any]:
    """Stringified summary used by the CLI."""
    d = asdict(result)
    d["out_path"] = str(result.out_path)
    return d
