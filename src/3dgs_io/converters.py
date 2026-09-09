"""External-tool runners invoked via ``uvx`` (uv tool run).

Some converters we want to chain into the scene-bundle pipeline pin to an
older Python version or carry heavy native deps. ``uvx`` solves both: it
creates an ephemeral, version-pinned venv on first use and re-uses the
cache on subsequent runs. We shell out to it as a normal subprocess.

The headline helper is :func:`lanelet2_to_clipgt`, a thin wrapper around
`hakuturu583/autoware_lanelet2_to_clipgt`_ that auto-derives the Lanelet2
UTM projection origin (``map.mgrs_grid`` + ``map.offset.{x,y,z}``) from a
Cesium tileset's ``root.transform`` so the resulting ClipGT parquet bundle
is aligned with the SPZ scene origin.

.. _hakuturu583/autoware_lanelet2_to_clipgt:
   https://github.com/hakuturu583/autoware_lanelet2_to_clipgt
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_LANELET2_CONVERTER_PACKAGE",
    "lanelet2_to_clipgt",
    "mgrs_overrides_from_root_transform",
    "run_uvx_tool",
]


DEFAULT_LANELET2_CONVERTER_PACKAGE = (
    "git+https://github.com/hakuturu583/autoware_lanelet2_to_clipgt"
)


# ---------------------------------------------------------------------------
# Generic uvx runner
# ---------------------------------------------------------------------------


def run_uvx_tool(
    *,
    package: str,
    command: Sequence[str],
    python: str | None = None,
    extra_deps: Iterable[str] | None = None,
    cwd: str | Path | None = None,
    capture: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run an external tool via ``uvx`` in an isolated, version-pinned venv.

    Parameters
    ----------
    package:
        ``--from`` value for ``uvx``. PyPI name (``"foo"``), version spec
        (``"foo>=1.2"``), or git URL (``"git+https://..."``).
    command:
        The CLI tokens to execute inside the prepared venv, e.g.
        ``["python", "-m", "foo", "--input", "x"]``.
    python:
        Python version to fetch/use, e.g. ``"3.10"``. ``None`` lets uv
        pick whatever satisfies the package's ``requires-python``.
    extra_deps:
        Additional packages to install alongside ``package`` (passed as
        repeated ``--with`` flags).
    cwd:
        Working directory for the subprocess.
    capture:
        Capture stdout/stderr and return them on the
        :class:`subprocess.CompletedProcess` (also disables live tty
        streaming). Defaults to ``False`` so the tool's output reaches the
        caller's terminal.
    check:
        Raise :class:`subprocess.CalledProcessError` on non-zero exit
        (default ``True``).

    Returns
    -------
    The :class:`subprocess.CompletedProcess` produced by :func:`subprocess.run`.
    """
    uvx = shutil.which("uvx")
    if uvx is None:
        raise RuntimeError("uvx not found on PATH; install uv from https://docs.astral.sh/uv/")
    argv: list[str] = [uvx, "--from", str(package)]
    if python is not None:
        argv += ["--python", str(python)]
    for dep in extra_deps or ():
        argv += ["--with", str(dep)]
    argv += [str(c) for c in command]
    return subprocess.run(
        argv,
        cwd=str(cwd) if cwd else None,
        check=check,
        capture_output=capture,
        text=True,
    )


# ---------------------------------------------------------------------------
# ECEF → MGRS + sub-grid offset (Lanelet2 UtmProjector input)
# ---------------------------------------------------------------------------


def _origin_from_root_transform(transform: Sequence[float]) -> tuple[float, float, float]:
    """Extract the ECEF translation (last column) from a 16-float 4×4."""
    if len(transform) != 16:
        raise ValueError(f"root.transform must have 16 elements, got {len(transform)}")
    # Cesium 3D Tiles JSON convention: column-major. Translation is the
    # 4th column → indices 12, 13, 14.
    return float(transform[12]), float(transform[13]), float(transform[14])


def _origin_from_tileset_path(tileset_path: str | Path) -> tuple[float, float, float]:
    """Read ``root.transform`` ECEF translation from a ``tileset.json``."""
    doc = json.loads(Path(tileset_path).read_text(encoding="utf-8-sig"))
    root = doc.get("root") or {}
    transform = root.get("transform")
    if transform is None:
        raise ValueError(f"{tileset_path}: tileset.json has no root.transform")
    return _origin_from_root_transform(transform)


def mgrs_overrides_from_root_transform(
    *,
    transform: Sequence[float] | None = None,
    tileset_path: str | Path | None = None,
) -> list[str]:
    """Compute Lanelet2-converter Hydra overrides from a scene origin.

    Convert a Cesium ``root.transform`` (4×4 ECEF anchor) into:

    * ``map.mgrs_grid=<5-char identifier>``
    * ``map.offset.x=<easting metres in grid>``
    * ``map.offset.y=<northing metres in grid>``
    * ``map.offset.z=<altitude metres>``

    Pass exactly one of ``transform`` or ``tileset_path``.
    """
    if (transform is None) == (tileset_path is None):
        raise ValueError("pass exactly one of `transform` or `tileset_path`")

    if tileset_path is not None:
        ecef = _origin_from_tileset_path(tileset_path)
    else:
        assert transform is not None
        ecef = _origin_from_root_transform(transform)

    try:
        import mgrs as _mgrs  # noqa: PLC0415
        import pyproj  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover - guarded at runtime
        raise ImportError(
            "Origin auto-derivation requires the `converters` extras. "
            "Install with: pip install 3dgs-io[converters]"
        ) from e

    # ECEF (EPSG:4978) → geodetic WGS84 (EPSG:4326).
    to_geodetic = pyproj.Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)
    lon, lat, alt = to_geodetic.transform(*ecef)

    # Geodetic → MGRS at 1-metre precision so we can read the sub-grid
    # easting/northing back from the string.
    mgrs_str = _mgrs.MGRS().toMGRS(lat, lon, MGRSPrecision=5)
    # MGRS 1-m form is GZD(3) + 100km(2) + easting(5) + northing(5).
    grid = mgrs_str[:5]
    easting = int(mgrs_str[5:10])
    northing = int(mgrs_str[10:15])

    return [
        f"map.mgrs_grid={grid}",
        f"map.offset.x={float(easting)}",
        f"map.offset.y={float(northing)}",
        f"map.offset.z={float(alt)}",
    ]


# ---------------------------------------------------------------------------
# Lanelet2 → ClipGT (high-level)
# ---------------------------------------------------------------------------


def lanelet2_to_clipgt(
    input_osm: str | Path,
    output_dir: str | Path,
    *,
    tileset_path: str | Path | None = None,
    root_transform: Sequence[float] | None = None,
    hydra_overrides: Sequence[str] | None = None,
    package: str = DEFAULT_LANELET2_CONVERTER_PACKAGE,
    ref: str | None = None,
    python: str = "3.10",
    capture: bool = False,
    check: bool = True,
    _cwd: str | Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Convert a Lanelet2 ``.osm`` file into a ClipGT parquet bundle via ``uvx``.

    Wraps `hakuturu583/autoware_lanelet2_to_clipgt`_. The converter pins to
    Python ``>=3.10,<3.11``; uvx fetches that interpreter on demand and
    isolates the converter's deps from the host environment.

    Origin alignment
    ----------------
    The converter's UTM projection origin is normally a bundled config
    (default: ``map=odaiba``). To align the ClipGT parquet's ``(0, 0, 0)``
    with the SPZ scene origin, pass either ``tileset_path`` (a Cesium
    ``tileset.json``) or ``root_transform`` (a 16-float 4×4 column-major
    ECEF anchor). The ECEF translation is converted to MGRS grid +
    sub-grid offset and supplied as Hydra overrides
    (``map.mgrs_grid=...`` / ``map.offset.{x,y,z}=...``). Explicit
    ``hydra_overrides`` are appended after the auto-derived ones, so they
    win on conflict.

    Parameters
    ----------
    input_osm:
        Path to the input Lanelet2 ``.osm`` file.
    output_dir:
        Directory the parquet bundle is written to (created if missing).
    tileset_path:
        Cesium ``tileset.json`` whose ``root.transform`` provides the
        ECEF origin. Mutually exclusive with ``root_transform``.
    root_transform:
        16-float column-major ECEF 4×4 directly. Mutually exclusive with
        ``tileset_path``.
    hydra_overrides:
        Additional Hydra ``KEY=VALUE`` strings, e.g. ``["clip_id=foo"]``
        or manual ``map.mgrs_grid=...`` to override auto-derivation.
    package:
        ``--from`` package for ``uvx``. Defaults to the upstream git URL.
    ref:
        Git ref (tag/commit) to pin the converter version. Ignored if
        ``package`` is not a git URL.
    python:
        Python interpreter version for uvx. The converter requires
        ``>=3.10,<3.11``.
    capture / check:
        Forwarded to :func:`run_uvx_tool`.

    .. _hakuturu583/autoware_lanelet2_to_clipgt:
       https://github.com/hakuturu583/autoware_lanelet2_to_clipgt
    """
    if tileset_path is not None and root_transform is not None:
        raise ValueError("pass at most one of `tileset_path` or `root_transform`")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overrides: list[str] = []
    if tileset_path is not None:
        overrides += mgrs_overrides_from_root_transform(tileset_path=tileset_path)
    elif root_transform is not None:
        overrides += mgrs_overrides_from_root_transform(transform=root_transform)
    if hydra_overrides:
        overrides += list(hydra_overrides)

    pkg = str(package)
    if ref is not None and pkg.startswith("git+"):
        pkg = f"{pkg}@{ref}"

    command: list[Any] = [
        "python",
        "-m",
        "autoware_lanelet2_to_clipgt",
        f"input_map_path={Path(input_osm).resolve()}",
        f"output_dir={out_dir.resolve()}",
        *overrides,
    ]
    return run_uvx_tool(
        package=pkg,
        command=command,
        python=python,
        cwd=_cwd,
        capture=capture,
        check=check,
    )
