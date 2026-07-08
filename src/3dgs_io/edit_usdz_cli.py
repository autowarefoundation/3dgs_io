"""CLI for editing existing splatsim USDZ scene bundles.

Invoke with ``python -m 3dgs_io.edit_usdz_cli <subcommand> ...``.

Subcommands
-----------

``lanelet2``
    Embed a lanelet2 ``.osm`` at archive path ``map.osm`` and record it under
    ``scene.json.extras.map_lanelet2``::

        python -m 3dgs_io.edit_usdz_cli lanelet2                       \\
            --input    path/to/scene.usdz                              \\
            --output   path/to/scene_with_map.usdz                     \\
            --lanelet2 path/to/map.osm

``intrinsics``
    Rewrite a camera's intrinsics inside the ``rig_trajectories.json``
    embedded in the USDZ (splatsim schema on output)::

        python -m 3dgs_io.edit_usdz_cli intrinsics                     \\
            --input  path/to/scene.usdz                                \\
            --output path/to/scene.edited.usdz                         \\
            --camera CAM_NAME                                          \\
            [--rig-id RIG_ID]                                          \\
            [--width W] [--height H]                                   \\
            [--fx FX] [--fy FY] [--cx CX] [--cy CY]                    \\
            [--distortion-coeffs c0,c1,...]                            \\
            [--principal-point px,py]                                  \\
            [--shutter-type STR] [--reference-poly STR]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from .edit_usdz import (
    _result_summary,
    add_lanelet2_to_usdz,
    update_camera_intrinsics_in_usdz,
)

_INTRINSIC_KEYS = (
    "width",
    "height",
    "fx",
    "fy",
    "cx",
    "cy",
    "distortion_coeffs",
    "principal_point",
    "shutter_type",
    "reference_poly",
)


def _parse_float_list(spec: str) -> list[float]:
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected comma-separated floats, got empty value")
    try:
        return [float(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"could not parse {spec!r} as floats: {exc}") from exc


def _parse_xy(spec: str) -> tuple[float, float]:
    vals = _parse_float_list(spec)
    if len(vals) != 2:
        raise argparse.ArgumentTypeError(f"expected exactly 2 comma-separated floats, got {spec!r}")
    return vals[0], vals[1]


def _add_common_io_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input", "-i", type=Path, required=True, help="Input .usdz")
    p.add_argument("--output", "-o", type=Path, required=True, help="Output .usdz")
    p.add_argument("-v", "--verbose", action="count", default=0)
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the JSON result summary on stdout",
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m 3dgs_io.edit_usdz_cli",
        description="Edit an existing splatsim USDZ scene bundle.",
    )
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")

    lanelet2 = sub.add_parser(
        "lanelet2",
        help="Embed a lanelet2 map.osm into the USDZ",
        description=(
            "Embed a lanelet2 .osm at archive path map.osm and record it in "
            "scene.json.extras.map_lanelet2."
        ),
    )
    _add_common_io_args(lanelet2)
    lanelet2.add_argument(
        "--lanelet2",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to a lanelet2 .osm file to embed as map.osm",
    )
    lanelet2.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Fail if the input archive already contains map.osm",
    )

    intr = sub.add_parser(
        "intrinsics",
        help="Rewrite a camera's intrinsics inside the USDZ's rig_trajectories.json",
        description=(
            "Load the rig_trajectories.json embedded in the USDZ, update the "
            "addressed camera's intrinsics, and write the result back."
        ),
    )
    _add_common_io_args(intr)
    intr.add_argument(
        "--camera",
        required=True,
        help="Name of the camera whose intrinsics to edit (matches Camera.name).",
    )
    intr.add_argument(
        "--rig-id",
        default=None,
        help=(
            "Restrict the search to this rig_id "
            "(required when multiple rigs share the camera name)."
        ),
    )
    intr.add_argument("--width", type=int, default=None, help="New image width in pixels")
    intr.add_argument("--height", type=int, default=None, help="New image height in pixels")
    intr.add_argument("--fx", type=float, default=None)
    intr.add_argument("--fy", type=float, default=None)
    intr.add_argument("--cx", type=float, default=None)
    intr.add_argument("--cy", type=float, default=None)
    intr.add_argument(
        "--distortion-coeffs",
        type=_parse_float_list,
        default=None,
        metavar="C0,C1,...",
        help="OpenCV distortion coefficients as a comma-separated float list",
    )
    intr.add_argument(
        "--principal-point",
        type=_parse_xy,
        default=None,
        metavar="PX,PY",
        help="ftheta principal point as PX,PY",
    )
    intr.add_argument("--shutter-type", default=None, help="ftheta shutter_type")
    intr.add_argument("--reference-poly", default=None, help="ftheta reference_poly")

    return p


def _collect_intrinsic_updates(args: argparse.Namespace) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    for key in _INTRINSIC_KEYS:
        val = getattr(args, key, None)
        if val is not None:
            updates[key] = val
    return updates


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    level = logging.WARNING - 10 * args.verbose
    logging.basicConfig(
        level=max(level, logging.DEBUG),
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.command == "lanelet2":
        result: Any = add_lanelet2_to_usdz(
            args.input,
            args.output,
            args.lanelet2,
            overwrite=not args.no_overwrite,
        )
    elif args.command == "intrinsics":
        updates = _collect_intrinsic_updates(args)
        if not updates:
            print(
                "error: no intrinsic updates specified; pass at least one of "
                "--width/--height/--fx/--fy/--cx/--cy/--distortion-coeffs/--principal-point/"
                "--shutter-type/--reference-poly",
                file=sys.stderr,
            )
            return 2
        result = update_camera_intrinsics_in_usdz(
            args.input,
            args.output,
            camera_name=args.camera,
            rig_id=args.rig_id,
            **updates,
        )
    else:  # pragma: no cover — argparse enforces choices
        parser.error(f"unknown command: {args.command}")

    if not args.quiet:
        json.dump(_result_summary(result), sys.stdout, indent=2)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
