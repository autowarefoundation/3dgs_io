"""In-place-like editors for existing single-file USDZ scene bundles.

Supports:

* :func:`add_lanelet2_to_usdz` — embed a lanelet2 ``map.osm`` file into an
  existing USDZ produced by :func:`3dgs_io.save_scene_usdz` and record it in
  ``scene.json.extras.map_lanelet2``.
* :func:`update_camera_intrinsics_in_usdz` — rewrite a camera's intrinsics
  inside the ``rig_trajectories.json`` embedded in the USDZ.

Every output archive preserves entry order (``default.usda`` stays first, per
the USDZ spec) and uses ``ZIP_STORED`` for every entry.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .rig_trajectories import (
    RIG_TRAJECTORIES_SCHEMA,
    load_rig_trajectories_doc,
    serialize_rig_trajectories,
    update_camera_intrinsics,
)

__all__ = [
    "EditUsdzResult",
    "IntrinsicsEditResult",
    "add_lanelet2_to_usdz",
    "update_camera_intrinsics_in_usdz",
]

_log = logging.getLogger(__name__)

_LANELET2_ARCHIVE_PATH = "map.osm"
_LANELET2_SCENE_KEY = "map_lanelet2"
_RIG_TRAJECTORIES_ARCHIVE_PATH = "rig_trajectories.json"


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


def add_lanelet2_to_usdz(
    input_usdz: str | Path,
    output_usdz: str | Path,
    lanelet2_path: str | Path,
    *,
    overwrite: bool = True,
) -> EditUsdzResult:
    """Add a lanelet2 ``map.osm`` file to an existing USDZ scene bundle.

    The source file is embedded verbatim at archive path ``map.osm`` and
    ``scene.json.extras.map_lanelet2`` is set to ``"map.osm"``. All other
    archive entries are copied through in their original order.

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
        Source ``.osm`` file to embed at ``map.osm``.
    overwrite:
        When ``True`` (default), replace an existing ``map.osm`` entry.
        When ``False``, raise :class:`ValueError` if the archive already
        contains one.
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
    is written back — always in the splatsim schema, even if the original
    file was in the alpasim shape.

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

    if isinstance(rig_doc, dict) and rig_doc.get("schema") != RIG_TRAJECTORIES_SCHEMA:
        _log.warning(
            "%s: embedded %s is not %s; treating as alpasim and rewriting in our schema "
            "(alpasim-only fields not retained by the splatsim schema will be lost).",
            input_usdz,
            _RIG_TRAJECTORIES_ARCHIVE_PATH,
            RIG_TRAJECTORIES_SCHEMA,
        )

    rigs = load_rig_trajectories_doc(rig_doc)
    cam = update_camera_intrinsics(
        rigs, camera_name=camera_name, rig_id=rig_id, **intrinsic_updates
    )
    new_rig_bytes = (json.dumps(serialize_rig_trajectories(rigs), indent=2) + "\n").encode("utf-8")

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


def _rewrite_scene_extras(scene_json_bytes: bytes, key: str, value: str) -> bytes:
    """Load ``scene.json``, set ``extras[key] = value``, return re-encoded bytes."""
    scene_doc = json.loads(scene_json_bytes.decode("utf-8-sig"))
    if not isinstance(scene_doc, dict):
        raise ValueError("scene.json is not a JSON object")
    extras = scene_doc.get("extras")
    if extras is None:
        extras = {}
        scene_doc["extras"] = extras
    elif not isinstance(extras, dict):
        raise ValueError("scene.json.extras is not an object")
    extras[key] = value
    return json.dumps(scene_doc, indent=2).encode("utf-8")


def _repack_usdz(
    input_usdz: Path,
    output_usdz: Path,
    *,
    entries_to_write: dict[str, bytes],
) -> tuple[list[str], list[str]]:
    """Copy ``input_usdz`` to ``output_usdz`` replacing/appending given entries.

    Original entry order is preserved, entries in ``entries_to_write`` that
    already exist are replaced in place, and new entries are appended after
    the copied ones. Every entry is written ``ZIP_STORED``. Safe when
    ``output_usdz == input_usdz`` (writes to a sibling temp file, then
    :func:`os.replace`).
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
                        _write_stored(zout, name, entries_to_write[name])
                    else:
                        _write_stored(zout, name, zin.read(name))
                for name in added:
                    _write_stored(zout, name, entries_to_write[name])
            os.replace(tmp_path, output_usdz)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    return added, replaced


def _write_stored(zf: zipfile.ZipFile, name: str, content: bytes) -> None:
    zi = zipfile.ZipInfo(name)
    zi.compress_type = zipfile.ZIP_STORED
    zf.writestr(zi, content)


def _result_summary(result: EditUsdzResult | IntrinsicsEditResult) -> dict[str, Any]:
    """Stringified summary used by the CLI."""
    d = asdict(result)
    d["out_path"] = str(result.out_path)
    return d
