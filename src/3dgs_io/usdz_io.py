"""USDZ I/O for :class:`spz.GaussianCloud`.

A gaussian cloud is wrapped in a `ZIP_STORED` USDZ archive containing:

* ``default.usda`` — a minimal USD root stage that holds an asset reference
  to ``model.spz``. The reference makes the archive a structurally valid
  USDZ payload (openable in USDView etc.) even though USD has no native
  support for SPZ gaussians.
* ``model.spz`` — the Niantic SPZ binary (the entire cloud).
"""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import spz

from .spz_io import load_spz, save_spz

__all__ = ["load_usdz", "save_usdz"]


_MODEL_FILENAME = "model.spz"

_DEFAULT_USDA = """#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{
    custom asset gaussianCloud = @./model.spz@
}
"""


def save_usdz(gc: spz.GaussianCloud, path: str | Path) -> None:
    """Write a :class:`spz.GaussianCloud` to a USDZ archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        spz_path = Path(td) / _MODEL_FILENAME
        save_spz(gc, spz_path)
        spz_bytes = spz_path.read_bytes()

    entries = [
        ("default.usda", _DEFAULT_USDA.encode("utf-8")),
        (_MODEL_FILENAME, spz_bytes),
    ]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        for name, data in entries:
            zi = zipfile.ZipInfo(name)
            zi.compress_type = zipfile.ZIP_STORED
            zf.writestr(zi, data)


def load_usdz(path: str | Path) -> spz.GaussianCloud:
    """Read a USDZ archive produced by :func:`save_usdz` and return the cloud."""
    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        if _MODEL_FILENAME not in zf.namelist():
            raise ValueError(f"{path}: missing {_MODEL_FILENAME} (not a 3dgs_io USDZ)")
        with tempfile.TemporaryDirectory() as td:
            spz_path = Path(td) / _MODEL_FILENAME
            spz_path.write_bytes(zf.read(_MODEL_FILENAME))
            return load_spz(spz_path)
