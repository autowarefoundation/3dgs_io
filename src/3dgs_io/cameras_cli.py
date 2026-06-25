"""CLI for editing camera intrinsics inside a ``rig_trajectories.json``.

Invoke with ``python -m 3dgs_io.cameras_cli``::

    python -m 3dgs_io.cameras_cli                              \\
        --input  path/to/rig_trajectories.json                  \\
        --output path/to/rig_trajectories.edited.json           \\
        --camera CAM_NAME                                       \\
        [--rig-id RIG_ID]                                       \\
        [--width W]  [--height H]                               \\
        [--fx FX] [--fy FY] [--cx CX] [--cy CY]                 \\
        [--distortion-coeffs c0,c1,...]                         \\
        [--principal-point px,py]

The input may be either our ``splatsim.rig_trajectories/v1`` schema or an
alpasim ``rig_trajectories.json`` (auto-detected). The output is always
written in our schema.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from .rig_trajectories import (
    RIG_TRAJECTORIES_SCHEMA,
    RigTrajectory,
    load_rig_trajectories_doc,
    serialize_rig_trajectories,
    update_camera_intrinsics,
)

_log = logging.getLogger(__name__)


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


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m 3dgs_io.cameras_cli",
        description=(
            "Edit camera intrinsics inside a rig_trajectories.json. "
            "Reads the input (our schema or alpasim), updates the addressed "
            "camera's intrinsics, and writes the result in our schema."
        ),
    )
    p.add_argument("--input", "-i", type=Path, required=True, help="Input rig_trajectories.json")
    p.add_argument("--output", "-o", type=Path, required=True, help="Output rig_trajectories.json")
    p.add_argument(
        "--camera",
        required=True,
        help="Name of the camera whose intrinsics to edit (matches Camera.name).",
    )
    p.add_argument(
        "--rig-id",
        default=None,
        help=(
            "Restrict the search to this rig_id "
            "(required when multiple rigs share the camera name)."
        ),
    )

    p.add_argument("--width", type=int, default=None, help="New image width in pixels")
    p.add_argument("--height", type=int, default=None, help="New image height in pixels")
    p.add_argument("--fx", type=float, default=None)
    p.add_argument("--fy", type=float, default=None)
    p.add_argument("--cx", type=float, default=None)
    p.add_argument("--cy", type=float, default=None)
    p.add_argument(
        "--distortion-coeffs",
        type=_parse_float_list,
        default=None,
        metavar="C0,C1,...",
        help="OpenCV distortion coefficients as a comma-separated float list",
    )
    p.add_argument(
        "--principal-point",
        type=_parse_xy,
        default=None,
        metavar="PX,PY",
        help="ftheta principal point as PX,PY",
    )
    p.add_argument("--shutter-type", default=None, help="ftheta shutter_type")
    p.add_argument("--reference-poly", default=None, help="ftheta reference_poly")

    p.add_argument("-v", "--verbose", action="count", default=0)
    p.add_argument("--quiet", action="store_true", help="Suppress the JSON summary on stdout")
    return p


def _collect_updates(args: argparse.Namespace) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    for key in (
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
    ):
        val = getattr(args, key, None)
        if val is not None:
            updates[key] = val
    return updates


def _load_rigs(path: Path) -> list[RigTrajectory]:
    doc = json.loads(path.read_text(encoding="utf-8-sig"))
    schema = doc.get("schema") if isinstance(doc, dict) else None
    if schema != RIG_TRAJECTORIES_SCHEMA:
        _log.warning(
            "%s is not %s; treating as alpasim and rewriting output in our schema "
            "(alpasim-only fields not retained by the splatsim schema will be lost).",
            path,
            RIG_TRAJECTORIES_SCHEMA,
        )
    return load_rig_trajectories_doc(doc)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    level = logging.WARNING - 10 * args.verbose
    logging.basicConfig(
        level=max(level, logging.DEBUG),
        format="%(levelname)s %(name)s: %(message)s",
    )

    updates = _collect_updates(args)
    if not updates:
        print(
            "error: no intrinsic updates specified; pass at least one of "
            "--width/--height/--fx/--fy/--cx/--cy/--distortion-coeffs/--principal-point/"
            "--shutter-type/--reference-poly",
            file=sys.stderr,
        )
        return 2

    rigs = _load_rigs(Path(args.input).expanduser())
    cam = update_camera_intrinsics(
        rigs,
        camera_name=args.camera,
        rig_id=args.rig_id,
        **updates,
    )

    out_doc = serialize_rig_trajectories(rigs)
    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_doc, indent=2) + "\n", encoding="utf-8")

    if not args.quiet:
        summary = {
            "output": str(out_path),
            "camera": cam.name,
            "camera_model": cam.camera_model.to_dict(),
            "updated_fields": sorted(updates),
        }
        json.dump(summary, sys.stdout, indent=2)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
