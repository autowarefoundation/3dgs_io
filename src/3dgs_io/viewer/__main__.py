"""CLI entry point: ``python -m 3dgs_io.viewer``."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m 3dgs_io.viewer",
        description="Launch a Cesium-based viewer for 3D Gaussian Splatting tilesets.",
    )
    parser.add_argument(
        "tiles_directory",
        help="Path to the directory containing tileset.json (or direct path to the JSON file).",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Bind address (default: localhost).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="HTTP port (default: 8080).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open a browser automatically.",
    )
    args = parser.parse_args(argv)

    from . import launch_viewer

    launch_viewer(
        args.tiles_directory,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
