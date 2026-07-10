"""CLI for :func:`3dgs_io.scene_usdz.save_scene_usdz`.

Invoke with ``python -m 3dgs_io`` (or ``python -m 3dgs_io.scene_usdz_cli``)::

    python -m 3dgs_io  path/to/tileset.json  output.usdz  \\
        [--extra ARCHIVE_PATH=SOURCE_PATH ...]              \\
        [--chunk-size N]  [--min-scale F]  ...

The input must be a Cesium 3D Tiles ``tileset.json``. Its ``root.transform``
(the world anchor — typically an ECEF placement) is preserved verbatim into
the output archive. Extras are user-supplied files or directories that get
embedded verbatim into the output archive at the requested path.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

from .rig_trajectories import load_rig_trajectories_doc
from .scene_usdz import (
    SceneUsdzOptions,
    _result_summary,
    save_scene_usdz,
)
from .tracks import parse_alpasim_sequence_tracks, parse_tracks
from .usdz_metadata import make_default_metadata


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
            "Pack a Cesium 3D Tiles tileset.json (+ optional sidecar files) "
            "into a single self-contained USDZ scene bundle. The source "
            "tileset's root.transform (world anchor) is preserved into the "
            "output."
        ),
    )
    p.add_argument(
        "tileset",
        type=Path,
        help="Input Cesium 3D Tiles tileset.json",
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
            "trajectory.parquet, sequence_tracks.json, rig_trajectories.json."
        ),
    )

    p.add_argument(
        "--tracks",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Path to a splatsim.sequence_tracks/v1 JSON file (or an alpasim "
            "sequence_tracks.json, auto-detected) describing dynamic-object "
            "trajectories. Embedded as sequence_tracks.json in the output USDZ."
        ),
    )
    p.add_argument(
        "--rig-trajectories",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Path to a splatsim.rig_trajectories/v1 JSON file (or an alpasim "
            "rig_trajectories.json, auto-detected) describing ego / sensor-rig "
            "pose time-series. Embedded as rig_trajectories.json in the output USDZ."
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
        help=(
            "metadata.yaml version_string for the output USDZ "
            "(default: '3dgs_io/<installed-version>')"
        ),
    )

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

    extras = dict(args.extras) if args.extras else None
    tracks = None
    if args.tracks is not None:
        tracks_path = Path(args.tracks).expanduser()
        tracks_doc = json.loads(tracks_path.read_text(encoding="utf-8-sig"))
        if not isinstance(tracks_doc, dict):
            raise ValueError(
                f"--tracks: {tracks_path} top-level value must be a JSON object, "
                f"got {type(tracks_doc).__name__}"
            )
        # Auto-detect: our schema has top-level "schema" key; alpasim's doesn't.
        if tracks_doc.get("schema") == "splatsim.sequence_tracks/v1":
            tracks = parse_tracks(tracks_doc)
        else:
            tracks = parse_alpasim_sequence_tracks(tracks_doc)
    rig_trajectories = None
    if args.rig_trajectories is not None:
        rig_path = Path(args.rig_trajectories).expanduser()
        rig_doc = json.loads(rig_path.read_text(encoding="utf-8-sig"))
        rig_trajectories = load_rig_trajectories_doc(rig_doc)
    metadata = make_default_metadata(
        out_path=args.out_usdz,
        uuid=args.uuid,
        scene_id=args.scene_id,
        version_string=args.version_string,
    )
    result = save_scene_usdz(
        args.tileset,
        args.out_usdz,
        extras=extras,
        tracks=tracks,
        rig_trajectories=rig_trajectories,
        metadata=metadata,
        options=_options_from_args(args),
    )

    if not args.quiet:
        json.dump(_result_summary(result), sys.stdout, indent=2)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
