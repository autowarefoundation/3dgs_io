"""CLI for exporting a scene USDZ to Cesium 3D Tiles."""

from __future__ import annotations

import argparse

from .tiles_export import TilesetSaveOptions
from .usdz_tiles_export import export_usdz_tileset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m 3dgs_io export-tiles")
    parser.add_argument("source_usdz")
    parser.add_argument("output_dir")
    parser.add_argument("--chunk-size", type=float, default=10.0)
    parser.add_argument("--geometric-error", type=float, default=100.0)
    args = parser.parse_args(argv)
    path = export_usdz_tileset(
        args.source_usdz,
        args.output_dir,
        TilesetSaveOptions(
            chunk_size=args.chunk_size,
            geometric_error=args.geometric_error,
        ),
    )
    print(path)
    return 0
