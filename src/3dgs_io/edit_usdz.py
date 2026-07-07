"""In-place-like editors for existing single-file USDZ scene bundles.

Currently supports:

* :func:`add_lanelet2_to_usdz` — embed a lanelet2 ``map.osm`` file into an
  existing USDZ produced by :func:`3dgs_io.save_scene_usdz` and record it in
  ``scene.json.extras.map_lanelet2``.

The output archive preserves entry order (``default.usda`` stays first, per the
USDZ spec) and uses ``ZIP_STORED`` for every entry.
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

__all__ = [
    "EditUsdzResult",
    "add_lanelet2_to_usdz",
]

_log = logging.getLogger(__name__)

_LANELET2_ARCHIVE_PATH = "map.osm"
_LANELET2_SCENE_KEY = "map_lanelet2"


@dataclass
class EditUsdzResult:
    """Summary of an :func:`add_lanelet2_to_usdz` invocation."""

    out_path: Path
    added: list[str] = field(default_factory=list)
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
    output_usdz.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(input_usdz, "r") as zin:
        names = zin.namelist()
        if "scene.json" not in names:
            raise ValueError(f"{input_usdz} is not a splatsim scene USDZ (missing scene.json)")
        has_map_osm = _LANELET2_ARCHIVE_PATH in names
        if has_map_osm and not overwrite:
            raise ValueError(
                f"{input_usdz} already contains {_LANELET2_ARCHIVE_PATH!r}; "
                "pass overwrite=True to replace"
            )

        scene_doc = json.loads(zin.read("scene.json").decode("utf-8-sig"))
        if not isinstance(scene_doc, dict):
            raise ValueError(f"{input_usdz}: scene.json is not a JSON object")
        extras = scene_doc.get("extras")
        if extras is None:
            extras = {}
            scene_doc["extras"] = extras
        elif not isinstance(extras, dict):
            raise ValueError(f"{input_usdz}: scene.json.extras is not an object")
        extras[_LANELET2_SCENE_KEY] = _LANELET2_ARCHIVE_PATH
        scene_bytes = json.dumps(scene_doc, indent=2).encode("utf-8")

        # Write to a sibling temp file so out == in is safe.
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
                    if name == "scene.json":
                        _write_stored(zout, name, scene_bytes)
                    elif name == _LANELET2_ARCHIVE_PATH:
                        _write_stored(zout, name, lanelet2_bytes)
                    else:
                        _write_stored(zout, name, zin.read(name))
                if not has_map_osm:
                    _write_stored(zout, _LANELET2_ARCHIVE_PATH, lanelet2_bytes)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    os.replace(tmp_path, output_usdz)

    return EditUsdzResult(
        out_path=output_usdz,
        added=[] if has_map_osm else [_LANELET2_ARCHIVE_PATH],
        replaced=[_LANELET2_ARCHIVE_PATH] if has_map_osm else [],
    )


def _write_stored(zf: zipfile.ZipFile, name: str, content: bytes) -> None:
    zi = zipfile.ZipInfo(name)
    zi.compress_type = zipfile.ZIP_STORED
    zf.writestr(zi, content)


def _result_summary(result: EditUsdzResult) -> dict[str, Any]:
    """Stringified summary used by the CLI."""
    d = asdict(result)
    d["out_path"] = str(result.out_path)
    return d
