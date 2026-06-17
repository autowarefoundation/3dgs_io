"""CLI for :func:`3dgs_io.scene_bundle.save_scene_bundle`.

Invoke with ``python -m 3dgs_io`` (or the equivalent
``python -m 3dgs_io.scene_bundle_cli``).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

from .scene_bundle import (
    SceneBundleOptions,
    _result_summary,
    save_scene_bundle,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m 3dgs_io",
        description=(
            "Convert an alpasim USDZ into the 3D-GS portion of a splatsim scene "
            "bundle (scene.json + tileset.json + chunks/*.spz + sky/). "
            "Non-gaussian sidecars are expected to be produced by other tools."
        ),
    )
    p.add_argument("usdz", type=Path, help="Input alpasim USDZ archive")
    p.add_argument("out_dir", type=Path, help="Output directory for the bundle")

    # Match the table ⑩ defaults from the format spec.
    p.add_argument("--chunk-size", type=float, default=50.0)
    p.add_argument("--max-points-per-chunk", type=int, default=200_000)
    p.add_argument("--min-scale", type=float, default=0.05)
    p.add_argument("--max-aspect-ratio", type=float, default=5.0)
    p.add_argument("--opacity-threshold", type=float, default=0.0)
    p.add_argument(
        "--bbox-radius",
        type=float,
        default=math.inf,
        help="Drop gaussians whose distance to median exceeds this radius (default: inf)",
    )

    p.add_argument("--sky-format", choices=("png", "exr"), default="png")

    p.add_argument("--sky-intensity", type=float, default=0.4)
    p.add_argument("--exposure", type=float, default=1.6)
    p.add_argument("--near-plane", type=float, default=0.5)
    p.add_argument("--far-plane", type=float, default=300.0)
    p.add_argument("--geometric-error", type=float, default=100.0)

    p.add_argument("-v", "--verbose", action="count", default=0)
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the JSON result summary on stdout",
    )
    return p


def _options_from_args(args: argparse.Namespace) -> SceneBundleOptions:
    return SceneBundleOptions(
        chunk_size=args.chunk_size,
        max_points_per_chunk=args.max_points_per_chunk,
        min_scale=args.min_scale,
        max_aspect_ratio=args.max_aspect_ratio,
        opacity_threshold=args.opacity_threshold,
        bbox_radius=args.bbox_radius,
        sky_format=args.sky_format,
        sky_intensity=args.sky_intensity,
        exposure=args.exposure,
        near_plane=args.near_plane,
        far_plane=args.far_plane,
        geometric_error=args.geometric_error,
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    level = logging.WARNING - 10 * args.verbose
    logging.basicConfig(
        level=max(level, logging.DEBUG),
        format="%(levelname)s %(name)s: %(message)s",
    )

    options = _options_from_args(args)
    result = save_scene_bundle(args.usdz, args.out_dir, options)

    if not args.quiet:
        json.dump(_result_summary(result), sys.stdout, indent=2)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
