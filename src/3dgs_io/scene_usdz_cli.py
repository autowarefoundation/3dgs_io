"""CLI for :func:`3dgs_io.scene_usdz.save_scene_usdz`.

Invoke with ``python -m 3dgs_io`` (or ``python -m 3dgs_io.scene_usdz_cli``)::

    python -m 3dgs_io input.{usdz,glb,ply,spz}  output.usdz  \\
        [--extra ARCHIVE_PATH=SOURCE_PATH ...]                \\
        [--chunk-size N]  [--min-scale F]  ...

The input is loaded with the appropriate ``load_*`` helper based on its
extension. Extras are user-supplied files or directories that get embedded
verbatim into the output archive at the requested path.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import sys
from pathlib import Path

import spz

from .scene_usdz import (
    SceneUsdzOptions,
    _result_summary,
    save_scene_usdz,
)


def _load_cloud(path: Path) -> spz.GaussianCloud:
    """Dispatch on file extension to the matching ``load_*`` helper."""
    ext = path.suffix.lower()
    mod = importlib.import_module("3dgs_io")
    if ext == ".usdz":
        return mod.load_usdz(path)
    if ext == ".glb":
        return mod.load_gltf(path)
    if ext == ".ply":
        return mod.load_ply(path)
    if ext == ".spz":
        return mod.load_spz(path)
    raise ValueError(
        f"Unsupported input extension {ext!r} for {path}; "
        "expected one of .usdz / .glb / .ply / .spz"
    )


def _parse_extra(spec: str) -> tuple[str, Path]:
    """``ARCHIVE_PATH=SOURCE_PATH`` → ``(archive_path, Path(source))``."""
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--extra value {spec!r} must be ARCHIVE_PATH=SOURCE_PATH")
    arc, src = spec.split("=", 1)
    arc = arc.strip()
    src_path = Path(src.strip()).expanduser()
    if not arc:
        raise argparse.ArgumentTypeError(f"--extra {spec!r}: archive path is empty")
    return arc, src_path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m 3dgs_io",
        description=(
            "Pack a gaussian-splat asset + optional sidecar files into a "
            "single self-contained USDZ scene bundle "
            "(default.usda + scene.json + tileset.json + chunks/*.spz + <extras>)."
        ),
    )
    p.add_argument(
        "input",
        type=Path,
        help="Input gaussian cloud: .usdz / .glb / .ply / .spz",
    )
    p.add_argument("out_usdz", type=Path, help="Output single-file USDZ path")

    p.add_argument(
        "--extra",
        action="append",
        dest="extras",
        default=[],
        metavar="ARC=SRC",
        type=_parse_extra,
        help=(
            "Embed SRC (file or directory) at archive path ARC. Repeatable. "
            "Known archive paths get recorded in scene.json's extras block: "
            "map.osm, map.xodr, carla_world/manifest.json, tracks.parquet, "
            "trajectory.parquet."
        ),
    )

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


def _options_from_args(args: argparse.Namespace) -> SceneUsdzOptions:
    return SceneUsdzOptions(
        chunk_size=args.chunk_size,
        max_points_per_chunk=args.max_points_per_chunk,
        min_scale=args.min_scale,
        max_aspect_ratio=args.max_aspect_ratio,
        opacity_threshold=args.opacity_threshold,
        bbox_radius=args.bbox_radius,
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

    cloud = _load_cloud(args.input)
    extras = dict(args.extras) if args.extras else None
    result = save_scene_usdz(
        cloud,
        args.out_usdz,
        extras=extras,
        options=_options_from_args(args),
    )

    if not args.quiet:
        json.dump(_result_summary(result), sys.stdout, indent=2)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
