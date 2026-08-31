"""CLI for :func:`3dgs_io.scene_connect.connect_scene_usdzs`.

Invoke with ``python -m 3dgs_io.scene_connect_cli``::

    python -m 3dgs_io.scene_connect_cli scene1.usdz scene2.usdz ... -o connected.usdz

The first input is the reference scene: its world frame orientation and
extras (``map.osm``, CARLA world, ...) are carried into the output, and the
merged geometry is re-anchored so the connected bundle stays geodetically
aligned with its inputs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .scene_connect import connect_scene_usdzs
from .scene_usdz import _result_summary
from .usdz_metadata import make_default_metadata


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m 3dgs_io.scene_connect_cli",
        description=(
            "Merge multiple scene USDZ bundles into one connected bundle. "
            "The first input is the reference scene; every other scene is "
            "baked into its frame through the geodetic ecef_anchor values."
        ),
    )
    p.add_argument("inputs", type=Path, nargs="+", help="Input scene USDZ bundles (2+)")
    p.add_argument("-o", "--out", type=Path, required=True, help="Output USDZ path")

    p.add_argument("--chunk-size", type=float, default=50.0)
    p.add_argument("--max-points-per-chunk", type=int, default=200_000)

    p.add_argument(
        "--uuid",
        default=None,
        help="metadata.yaml uuid for the output USDZ (default: fresh random UUID4)",
    )
    p.add_argument(
        "--scene-id",
        dest="scene_id",
        default=None,
        help="metadata.yaml scene_id for the output USDZ (default: output filename stem)",
    )
    p.add_argument(
        "--version-string",
        dest="version_string",
        default=None,
        help="metadata.yaml version_string (default: '3dgs_io/<installed-version>')",
    )

    p.add_argument("-v", "--verbose", action="count", default=0)
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the JSON result summary on stdout",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    level = logging.WARNING - 10 * args.verbose
    logging.basicConfig(
        level=max(level, logging.DEBUG),
        format="%(levelname)s %(name)s: %(message)s",
    )
    if len(args.inputs) < 2:
        _build_parser().error("need at least two input bundles")

    metadata = make_default_metadata(
        out_path=args.out,
        uuid=args.uuid,
        scene_id=args.scene_id,
        version_string=args.version_string,
    )
    result = connect_scene_usdzs(
        list(args.inputs),
        args.out,
        chunk_size=args.chunk_size,
        max_points_per_chunk=args.max_points_per_chunk,
        metadata=metadata,
    )

    if not args.quiet:
        json.dump(_result_summary(result), sys.stdout, indent=2)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
