"""Tests for ``3dgs_io.converters`` (uvx-driven external tool runners).

Unit-level tests cover the ECEF→MGRS origin derivation (no subprocess).
The end-to-end test invokes ``uvx`` to actually run
``hakuturu583/autoware_lanelet2_to_clipgt`` against a vendored
4-node/2-way/1-lanelet ``.osm`` fixture; it runs in CI because the test
workflow installs ``uv`` via ``astral-sh/setup-uv@v6``.
"""

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

_mod = importlib.import_module("3dgs_io")
DEFAULT_LANELET2_CONVERTER_PACKAGE = _mod.DEFAULT_LANELET2_CONVERTER_PACKAGE
lanelet2_to_clipgt = _mod.lanelet2_to_clipgt
mgrs_overrides_from_root_transform = _mod.mgrs_overrides_from_root_transform
run_uvx_tool = _mod.run_uvx_tool


TINY_OSM = Path(__file__).parent / "data" / "tiny_lanelet2.osm"


# Cesium identity-rotation 4×4 with ECEF translation chosen to match the
# rainbow_bridge sample. Column-major, last column = translation.
_RAINBOW_BRIDGE_TRANSFORM: list[float] = [
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    -3961517.719569116,
    3352351.3421289744,
    3695591.763367203,
    1.0,
]


# ---------------------------------------------------------------------------
# Origin derivation (pure math, no subprocess)
# ---------------------------------------------------------------------------


def test_mgrs_overrides_from_root_transform_matches_known_rainbow_bridge_values() -> None:
    overrides = mgrs_overrides_from_root_transform(transform=_RAINBOW_BRIDGE_TRANSFORM)
    # rainbow_bridge is in Tokyo's Odaiba — MGRS grid 54SUE.
    assert any(o == "map.mgrs_grid=54SUE" for o in overrides), overrides

    by_key = dict(o.split("=", 1) for o in overrides)
    assert "map.offset.x" in by_key
    assert "map.offset.y" in by_key
    assert "map.offset.z" in by_key
    # Sub-grid offset must land near (87831, 44428) m and altitude ~54 m
    # (within the MGRS 100km grid for Odaiba).
    assert abs(float(by_key["map.offset.x"]) - 87831.0) < 5.0
    assert abs(float(by_key["map.offset.y"]) - 44428.0) < 5.0
    assert abs(float(by_key["map.offset.z"]) - 54.27) < 2.0


def test_mgrs_overrides_from_tileset_path(tmp_path: Path) -> None:
    ts = tmp_path / "tileset.json"
    ts.write_text(
        json.dumps(
            {
                "asset": {"version": "1.1"},
                "geometricError": 100.0,
                "root": {
                    "boundingVolume": {
                        "box": [0, 0, 0, 100, 0, 0, 0, 100, 0, 0, 0, 100],
                    },
                    "geometricError": 0,
                    "refine": "ADD",
                    "transform": _RAINBOW_BRIDGE_TRANSFORM,
                },
            }
        )
    )
    overrides = mgrs_overrides_from_root_transform(tileset_path=ts)
    assert any(o == "map.mgrs_grid=54SUE" for o in overrides)


def test_mgrs_overrides_requires_exactly_one_input() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        mgrs_overrides_from_root_transform()  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="exactly one"):
        mgrs_overrides_from_root_transform(
            transform=_RAINBOW_BRIDGE_TRANSFORM,
            tileset_path="ignored.json",
        )


def test_mgrs_overrides_rejects_bad_transform_length() -> None:
    with pytest.raises(ValueError, match="16 elements"):
        mgrs_overrides_from_root_transform(transform=[1.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# run_uvx_tool guards
# ---------------------------------------------------------------------------


def test_run_uvx_tool_missing_uvx_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="uvx not found"):
        run_uvx_tool(package="anything", command=["python", "-V"])


def test_lanelet2_to_clipgt_rejects_both_tileset_and_root_transform(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at most one"):
        lanelet2_to_clipgt(
            input_osm=TINY_OSM,
            output_dir=tmp_path,
            tileset_path="x.json",
            root_transform=_RAINBOW_BRIDGE_TRANSFORM,
        )


# ---------------------------------------------------------------------------
# End-to-end conversion via uvx
# ---------------------------------------------------------------------------


def _have_uvx() -> bool:
    return shutil.which("uvx") is not None


@pytest.mark.skipif(not _have_uvx(), reason="uvx not on PATH")
def test_lanelet2_to_clipgt_end_to_end(tmp_path: Path) -> None:
    """Run the actual converter on a 1-lanelet .osm fixture.

    First invocation triggers uv to fetch Python 3.10 + the converter's
    deps (lanelet2-python-api, mgrs, hydra, pandas, …); subsequent runs
    use the uv cache. Anything succeeding means the uvx plumbing and the
    auto-derived hydra overrides actually work end-to-end.
    """
    out_dir = tmp_path / "clipgt"
    # Hydra drops an ``outputs/<date>/<time>/`` directory in CWD on every
    # run; chdir into tmp_path so it gets cleaned up with the fixture.
    hydra_cwd = tmp_path / "hydra_cwd"
    hydra_cwd.mkdir()
    try:
        result = lanelet2_to_clipgt(
            input_osm=TINY_OSM,
            output_dir=out_dir,
            root_transform=_RAINBOW_BRIDGE_TRANSFORM,
            capture=True,
            check=False,
            _cwd=hydra_cwd,
        )
    except subprocess.SubprocessError as e:  # pragma: no cover - rare CI failure
        pytest.fail(f"uvx subprocess failed: {e}")

    if result.returncode != 0:
        pytest.fail(
            "lanelet2_to_clipgt exited non-zero "
            f"(returncode={result.returncode}).\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    # The converter writes a parquet bundle into output_dir. We don't pin the
    # exact filenames (they're the tool's contract, not ours), but the dir
    # must contain at least one .parquet file.
    parquet_files = list(out_dir.rglob("*.parquet"))
    assert parquet_files, (
        f"expected at least one .parquet under {out_dir}; "
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
