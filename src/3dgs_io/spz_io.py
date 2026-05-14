from __future__ import annotations

from pathlib import Path

import spz


def load_spz(path: str | Path) -> spz.GaussianCloud:
    """Load an SPZ file and return a GaussianCloud.

    Coordinates are converted to glTF (RUB) coordinate system.
    """
    opts = spz.UnpackOptions()
    opts.to_coord = spz.RUB
    return spz.load_spz(str(path), opts)


def save_spz(gc: spz.GaussianCloud, path: str | Path) -> None:
    """Save a GaussianCloud to an SPZ file.

    Converts from internal glTF (RUB) coordinate system.
    """
    opts = spz.PackOptions()
    opts.from_coord = spz.RUB
    spz.save_spz(gc, opts, str(path))


def load_ply(path: str | Path) -> spz.GaussianCloud:
    """Load a PLY file (3DGS training output) and return a GaussianCloud.

    Coordinates are converted from COLMAP (RDF) to glTF (RUB) coordinate system.
    """
    opts = spz.UnpackOptions()
    opts.to_coord = spz.RUB
    return spz.load_splat_from_ply(str(path), opts)


def save_ply(gc: spz.GaussianCloud, path: str | Path) -> None:
    """Save a GaussianCloud to a PLY file.

    Converts from internal glTF (RUB) to COLMAP (RDF) coordinate system.
    """
    opts = spz.PackOptions()
    opts.from_coord = spz.RUB
    spz.save_splat_to_ply(gc, opts, str(path))
