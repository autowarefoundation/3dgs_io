"""CLI for :func:`3dgs_io.edit_usdz.add_lanelet2_to_usdz`.

Invoke with ``python -m 3dgs_io.edit_usdz_cli``::

    python -m 3dgs_io.edit_usdz_cli                  \\
        --input    path/to/scene.usdz                \\
        --output   path/to/scene_with_map.usdz       \\
        --lanelet2 path/to/map.osm

The input USDZ must contain a ``scene.json`` (splatsim.scene/v1). The output
carries the same entries plus ``map.osm`` at the archive root, with
``scene.json.extras.map_lanelet2`` set to ``"map.osm"``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .edit_usdz import _result_summary, add_lanelet2_to_usdz


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m 3dgs_io.edit_usdz_cli",
        description=(
            "Edit an existing splatsim USDZ scene bundle. Currently supports "
            "adding a lanelet2 map.osm file, recorded under "
            "scene.json.extras.map_lanelet2."
        ),
    )
    p.add_argument("--input", "-i", type=Path, required=True, help="Input .usdz")
    p.add_argument("--output", "-o", type=Path, required=True, help="Output .usdz")
    p.add_argument(
        "--lanelet2",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to a lanelet2 .osm file to embed as map.osm",
    )
    p.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Fail if the input archive already contains map.osm",
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

    result = add_lanelet2_to_usdz(
        args.input,
        args.output,
        args.lanelet2,
        overwrite=not args.no_overwrite,
    )

    if not args.quiet:
        json.dump(_result_summary(result), sys.stdout, indent=2)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
