"""Rigid dynamic-object ("actor") 3DGS assets — object-local Gaussian bank.

``splatsim.sequence_tracks/v2`` already says *where* every dynamic object is at
every moment (``track_id`` / ``class_name`` / box ``size`` / per-frame pose).
What it cannot say is what the object *looks like*: a scene bundle's Gaussians
are one static world-frame cloud, so a moving car is either smeared into the
background or cut out of it and gone.

This module supplies the missing half — an **asset bank**: Gaussian clouds
authored in a canonical *object-local* frame, plus the bindings that say which
track is rendered with which asset. A consumer renders frame ``t`` by taking
the track's pose ``(R, p)`` at ``t`` and instancing the asset's Gaussians
through it: means rotated and translated, orientation quaternions
pre-multiplied by ``R``, scales / opacity / colour untouched.

Scope of ``v1``: **rigid** objects only — vehicles, trailers, traffic cones,
anything whose shape does not change over time. One cloud, one pose per frame.
Articulated (pedestrians, cyclists) and deformable objects are deliberately out
of scope; the per-asset ``motion`` tag exists so they can arrive later without
a schema break, and a ``v1`` reader rejects anything but ``"rigid"`` loudly
rather than rendering a walking pedestrian as a frozen statue.

Gaussian parameters are **identical to the static background**: the payload is
an NGSP v4 SPZ stream with the same fields (positions, rotations, scales,
colours, alphas, SH) and the same optional per-Gaussian LiDAR extension record
(:data:`~3dgs_io.SPZ_EXT_TYPE_TIER4_GAUSSIAN_LIDAR`), so an actor's Gaussians
travel through exactly the loader, LOD and rasteriser paths a background chunk
does. Nothing about an actor is a special kind of Gaussian.

Object-local frame
------------------

The most common way to get dynamic objects wrong is to leave the object frame
implicit and discover downstream that every asset faces the wrong way (NVIDIA's
NuRec asset-insertion workflow, for instance, needs a hand-applied 90° rotation
to reconcile Asset-Harvester output with its renderer). The convention is
therefore written into every document as ``object_frame`` and compared exactly
on read:

* right-handed, **+X forward** (the direction the object drives), **+Y left**,
  **+Z up** — the same FLU triad the bundle already uses for sensor rigs;
* origin at the **centre of the object's oriented bounding box**, in all three
  axes — the exact point ``Track.frames[].translation`` refers to, so
  instancing is a pure rigid transform with no hidden offset;
* metres, right-handed rotations, xyzw quaternions — as everywhere else in a
  scene bundle.

An asset authored around a ground-level origin (common for CARLA / Autoware
vehicle models) must be shifted by ``+size_z / 2`` along ``+Z`` before it is
packed. An asset in the glTF/SPZ RUB triad must be rotated by
:data:`~3dgs_io.frame_convention.RUB_TO_ENU`.

Spherical harmonics live in the object frame too: a renderer evaluates them
with the view direction rotated *into* the object frame, so the coefficients
never need re-rotating per frame.

Metric scale, not unit scale
----------------------------

Assets are stored **at true metric scale**. NuRec's external assets are
unit-normalised and stretched to each track's cuboid at render time; that suits
a generative asset library but it distorts geometry non-uniformly, which is
exactly what a LiDAR simulator must not do — a range return off a car stretched
into a truck's box is wrong by construction. The default binding therefore
instances the asset rigidly and ignores the track's box dimensions. Consumers
that want the NuRec behaviour opt in per instance:

``fit_mode``
    * ``"rigid"`` (default) — use the asset as authored; the track's ``size``
      is metadata only. Metrically faithful.
    * ``"uniform"`` — one isotropic scale factor fitting the asset's declared
      ``size`` into the track's ``size``. Shape preserved, scale approximate.
    * ``"stretch"`` — per-axis scale (NuRec-equivalent). Interoperable, but
      **not** metrically faithful; never use it for LiDAR ground truth.

Assets and tracks are decoupled
-------------------------------

``asset_id`` is not ``track_id``. One asset can back many tracks (fifty
instances of the same sedan), and a track is re-skinned by editing one binding
— the same decoupling NuRec's ``DynamicObjectTrack.asset_id`` uses. Bindings
live in this document rather than inside ``sequence_tracks.json`` so the track
schema stays untouched and a bundle's tracks remain readable by consumers that
know nothing about actor assets.

On-disk layout
--------------

Inside a scene USDZ — and, identically, in a standalone asset-bank directory::

    actor_assets.json                        # this schema
    actor_assets/<asset_id>/asset.spz        # NGSP v4 SPZ, object-local frame

``actor_assets.json`` is recorded in ``scene.json.extras.actor_assets``.

Document schema — ``splatsim.actor_assets/v1``::

    {
      "schema": "splatsim.actor_assets/v1",
      "frame": "object",
      "object_frame": {...},                 # the convention above, verbatim
      "assets": [
        {
          "asset_id":   "sedan_0007",
          "uri":        "actor_assets/sedan_0007/asset.spz",
          "class_name": "automobile",
          "motion":     "rigid",
          "size":       [4.62, 1.90, 1.48],  # box (dx, dy, dz), metres
          "bbox_min":   [-2.41, -0.98, -0.79],
          "bbox_max":   [ 2.40,  0.97,  0.81],
          "n_points":   48213,
          "sh_degree":  3,
          "ext_attributes": {                # present iff the SPZ carries one
            "extension": "EXT_gaussian_lidar",
            "container": "spz_extension",
            "spz_extension_type": "0x54340001",
            "attributes": ["lidar_intensity_raw", "lidar_raydrop_logit"]
          },
          "provenance": {},
          "metadata":   {}
        }
      ],
      "instances": [
        {"track_id": "100", "asset_id": "sedan_0007", "fit_mode": "rigid"}
      ]
    }

``size`` is the *declared* box the track pose refers to; ``bbox_min`` /
``bbox_max`` is the *measured* AABB of the Gaussian means, which legitimately
overhangs the box (splats have radius, reconstructions have outliers). Both are
kept: placement and fitting use ``size``, culling uses the AABB. The AABB must
contain the origin — an asset whose means sit at ``(113, -58, 1.9)`` is still
in world coordinates, and that check catches it at write time rather than at
render time.
"""

from __future__ import annotations

import json
import logging
import math
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import spz
from scipy.spatial.transform import Rotation

from .ext_attributes import (
    EXT_GAUSSIAN_LIDAR_NAME,
    RAYDROP_SH_KEY,
    SPZ_EXT_TYPE_TIER4_GAUSSIAN_LIDAR_HEX,
    embed_lidar_extension,
    extract_lidar_extension,
    raydrop_sh_degree_from_coefs,
)
from .spz_io import load_spz_world, save_spz_world
from .tracks import Track

__all__ = [
    "ACTOR_ASSETS_ARCHIVE_PATH",
    "ACTOR_ASSETS_PREFIX",
    "ACTOR_ASSETS_SCHEMA",
    "FIT_MODES",
    "MOTION_MODELS",
    "OBJECT_FRAME_CONVENTION",
    "SH_POLICIES",
    "ActorAsset",
    "ActorAssetBank",
    "ActorAssetSource",
    "ActorInstance",
    "asset_archive_uri",
    "build_actor_asset_bank",
    "decode_actor_asset",
    "encode_actor_asset",
    "extract_actor_asset",
    "load_actor_asset_dir",
    "parse_actor_assets",
    "rotate_sh_about_z",
    "save_actor_asset_dir",
    "serialize_actor_assets",
    "validate_instances_against_tracks",
    "validate_object_frame",
]

_log = logging.getLogger(__name__)

ACTOR_ASSETS_SCHEMA = "splatsim.actor_assets/v1"

#: Archive-root path of the bank document inside a scene USDZ.
ACTOR_ASSETS_ARCHIVE_PATH = "actor_assets.json"

#: Archive prefix owned by the asset payloads.
ACTOR_ASSETS_PREFIX = "actor_assets/"

#: Assets are authored in the object frame, never the scene's world frame.
_FRAME = "object"

#: The canonical object-local frame. Written into every document and compared
#: exactly on read, exactly as ``FRAME_CONVENTION`` is for world-frame data.
OBJECT_FRAME_CONVENTION: dict[str, Any] = {
    "handedness": "right",
    "forward": "+x",
    "left": "+y",
    "up": "+z",
    "origin": "bbox_center",
    "quaternion_order": "xyzw",
    "length_units": "meters",
    "scale": "metric",
}

#: Motion models a document may declare. ``v1`` implements ``rigid`` only.
MOTION_MODELS: tuple[str, ...] = ("rigid",)

#: How an instance reconciles the asset's own size with its track's box size.
FIT_MODES: tuple[str, ...] = ("rigid", "uniform", "stretch")

#: How :func:`extract_actor_asset` treats view-dependent SH bands.
SH_POLICIES: tuple[str, ...] = ("rotate", "drop", "keep")

# ``asset_id`` becomes an archive path component, so keep it boring: no
# separators, no traversal, no surprises for consumers that write the bank out
# to a filesystem.
_ASSET_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

# Warn (don't fail) when a rigidly-instanced asset is this far off its track's
# declared box on any axis — usually the wrong asset bound to the track.
_SIZE_MISMATCH_RATIO = 0.25

# The object origin is the box centre, so a correctly authored asset's AABB
# straddles it. Allow slack for assets whose reconstruction is one-sided (a car
# only ever observed from the left).
_ORIGIN_SLACK_M = 0.5

_MAX_SH_DEGREE = 3

# Beyond this much pitch/roll an exact yaw-only SH rotation is no longer a
# faithful re-expression of the coefficients (see :func:`extract_actor_asset`).
_MAX_NON_YAW_RAD = math.radians(2.0)


def validate_object_frame(value: Any) -> None:
    """Reject documents whose object-frame contract differs from ours."""
    if value != OBJECT_FRAME_CONVENTION:
        raise ValueError(
            "unsupported object_frame; expected the canonical +x-forward / "
            "+y-left / +z-up, bbox-centred, metric-scale actor frame"
        )


def asset_archive_uri(asset_id: str) -> str:
    """Archive path of ``asset_id``'s SPZ payload."""
    return f"{ACTOR_ASSETS_PREFIX}{asset_id}/asset.spz"


def _validate_asset_id(asset_id: str) -> str:
    if not _ASSET_ID_RE.match(asset_id):
        raise ValueError(
            f"invalid asset_id {asset_id!r}: must match {_ASSET_ID_RE.pattern} "
            "(it is used verbatim as an archive path component)"
        )
    return asset_id


def _as_vec3(value: Any, *, where: str) -> tuple[float, float, float]:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{where} must be 3 finite floats, got {value!r}")
    return (float(arr[0]), float(arr[1]), float(arr[2]))


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ActorAsset:
    """One rigid object's Gaussian cloud, described in the object-local frame.

    The payload itself lives at :attr:`uri` inside the archive; this record is
    the index entry a consumer reads before deciding whether to load it.
    """

    asset_id: str
    class_name: str
    size: tuple[float, float, float]
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    n_points: int
    sh_degree: int = 0
    motion: str = "rigid"
    uri: str | None = None
    """Archive path of the SPZ payload. Defaults to :func:`asset_archive_uri`."""
    ext_attributes: dict[str, Any] | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_asset_id(self.asset_id)
        if not self.class_name:
            raise ValueError(f"asset {self.asset_id!r}: class_name must not be empty")
        if self.motion not in MOTION_MODELS:
            raise ValueError(
                f"asset {self.asset_id!r}: unsupported motion model {self.motion!r}; "
                f"{ACTOR_ASSETS_SCHEMA} supports {list(MOTION_MODELS)}"
            )
        self.size = _as_vec3(self.size, where=f"asset {self.asset_id!r} size")
        if any(v <= 0.0 for v in self.size):
            raise ValueError(f"asset {self.asset_id!r}: size must be positive, got {self.size}")
        self.bbox_min = _as_vec3(self.bbox_min, where=f"asset {self.asset_id!r} bbox_min")
        self.bbox_max = _as_vec3(self.bbox_max, where=f"asset {self.asset_id!r} bbox_max")
        for axis, (lo, hi) in enumerate(zip(self.bbox_min, self.bbox_max, strict=True)):
            if hi < lo:
                raise ValueError(
                    f"asset {self.asset_id!r}: bbox axis {axis} has min {lo} above max {hi}"
                )
            # The origin is the box centre, so a correctly authored asset's
            # AABB straddles it. One that does not was never brought into the
            # object frame.
            if lo > _ORIGIN_SLACK_M or hi < -_ORIGIN_SLACK_M:
                raise ValueError(
                    f"asset {self.asset_id!r}: bbox axis {axis} spans [{lo:.3f}, {hi:.3f}], "
                    "which does not contain the object-local origin — the gaussians are "
                    "probably still in world coordinates (see the object_frame contract)"
                )
        if self.n_points <= 0:
            raise ValueError(f"asset {self.asset_id!r}: n_points must be positive")
        if not 0 <= self.sh_degree <= _MAX_SH_DEGREE:
            raise ValueError(
                f"asset {self.asset_id!r}: sh_degree must be in 0..{_MAX_SH_DEGREE}, "
                f"got {self.sh_degree}"
            )
        expected_uri = asset_archive_uri(self.asset_id)
        if self.uri is None:
            self.uri = expected_uri
        elif self.uri != expected_uri:
            raise ValueError(
                f"asset {self.asset_id!r}: uri must be {expected_uri!r}, got {self.uri!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "asset_id": self.asset_id,
            "uri": self.uri,
            "class_name": self.class_name,
            "motion": self.motion,
            "size": [float(v) for v in self.size],
            "bbox_min": [float(v) for v in self.bbox_min],
            "bbox_max": [float(v) for v in self.bbox_max],
            "n_points": int(self.n_points),
            "sh_degree": int(self.sh_degree),
        }
        if self.ext_attributes is not None:
            value["ext_attributes"] = dict(self.ext_attributes)
        value["provenance"] = dict(self.provenance)
        value["metadata"] = dict(self.metadata)
        return value

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ActorAsset:
        ext = d.get("ext_attributes")
        return cls(
            asset_id=str(d["asset_id"]),
            class_name=str(d["class_name"]),
            size=_as_vec3(d["size"], where="asset size"),
            bbox_min=_as_vec3(d["bbox_min"], where="asset bbox_min"),
            bbox_max=_as_vec3(d["bbox_max"], where="asset bbox_max"),
            n_points=int(d["n_points"]),
            sh_degree=int(d.get("sh_degree", 0)),
            motion=str(d.get("motion", "rigid")),
            uri=str(d["uri"]) if d.get("uri") is not None else None,
            ext_attributes=dict(ext) if isinstance(ext, dict) else None,
            provenance=dict(d.get("provenance") or {}),
            metadata=dict(d.get("metadata") or {}),
        )


@dataclass
class ActorInstance:
    """Binds one track to the asset it is rendered with."""

    track_id: str
    asset_id: str
    fit_mode: str = "rigid"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.track_id:
            raise ValueError("instance.track_id must not be empty")
        _validate_asset_id(self.asset_id)
        if self.fit_mode not in FIT_MODES:
            raise ValueError(
                f"instance {self.track_id!r}: unknown fit_mode {self.fit_mode!r}; "
                f"expected one of {list(FIT_MODES)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "asset_id": self.asset_id,
            "fit_mode": self.fit_mode,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ActorInstance:
        return cls(
            track_id=str(d["track_id"]),
            asset_id=str(d["asset_id"]),
            fit_mode=str(d.get("fit_mode", "rigid")),
            metadata=dict(d.get("metadata") or {}),
        )


@dataclass
class ActorAssetBank:
    """The asset index plus the track bindings — one ``actor_assets.json``."""

    assets: list[ActorAsset] = field(default_factory=list)
    instances: list[ActorInstance] = field(default_factory=list)

    def asset_by_id(self, asset_id: str) -> ActorAsset:
        for asset in self.assets:
            if asset.asset_id == asset_id:
                return asset
        raise KeyError(f"no asset {asset_id!r} in bank")

    def asset_for_track(self, track_id: str) -> ActorAsset | None:
        """The asset bound to ``track_id``, or ``None`` when it is unbound."""
        for instance in self.instances:
            if instance.track_id == track_id:
                return self.asset_by_id(instance.asset_id)
        return None


@dataclass
class ActorAssetSource:
    """An in-memory asset on its way into a bundle.

    ``cloud`` holds object-local values *numerically* — it is written with no
    implicit axis conversion, exactly like the world-frame background chunks.
    ``ext_attrs`` carries the same optional per-Gaussian LiDAR arrays the
    background uses, threaded so ``attr[i] ↔ gaussian[i]`` keeps holding.
    """

    asset_id: str
    cloud: spz.GaussianCloud
    class_name: str
    size: tuple[float, float, float]
    ext_attrs: dict[str, np.ndarray] = field(default_factory=dict)
    motion: str = "rigid"
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Document (de)serialisation
# ---------------------------------------------------------------------------


def _validate_bank(bank: ActorAssetBank) -> None:
    seen: set[str] = set()
    for asset in bank.assets:
        if asset.asset_id in seen:
            raise ValueError(f"duplicate asset_id: {asset.asset_id!r}")
        seen.add(asset.asset_id)
    bound: set[str] = set()
    for instance in bank.instances:
        if instance.track_id in bound:
            raise ValueError(
                f"track {instance.track_id!r} is bound to more than one asset; "
                "a track renders exactly one asset"
            )
        bound.add(instance.track_id)
        if instance.asset_id not in seen:
            raise ValueError(
                f"instance for track {instance.track_id!r} references unknown "
                f"asset_id {instance.asset_id!r}"
            )


def serialize_actor_assets(bank: ActorAssetBank) -> dict[str, Any]:
    """Build a frame-explicit object-local actor-asset document."""
    _validate_bank(bank)
    return {
        "schema": ACTOR_ASSETS_SCHEMA,
        "frame": _FRAME,
        "object_frame": OBJECT_FRAME_CONVENTION,
        "assets": [asset.to_dict() for asset in bank.assets],
        "instances": [instance.to_dict() for instance in bank.instances],
    }


def parse_actor_assets(doc: dict[str, Any]) -> ActorAssetBank:
    """Inverse of :func:`serialize_actor_assets`."""
    schema = doc.get("schema")
    if schema != ACTOR_ASSETS_SCHEMA:
        raise ValueError(
            f"unexpected actor_assets schema {schema!r}; expected {ACTOR_ASSETS_SCHEMA!r}"
        )
    if doc.get("frame") != _FRAME:
        raise ValueError(f"actor_assets frame must be {_FRAME!r}")
    validate_object_frame(doc.get("object_frame"))
    raw_assets = doc.get("assets")
    if not isinstance(raw_assets, list):
        raise ValueError("actor_assets document is missing the 'assets' list")
    raw_instances = doc.get("instances")
    if raw_instances is not None and not isinstance(raw_instances, list):
        raise ValueError("actor_assets 'instances' must be a list when present")
    bank = ActorAssetBank(
        assets=[ActorAsset.from_dict(entry) for entry in raw_assets],
        instances=[ActorInstance.from_dict(entry) for entry in raw_instances or []],
    )
    _validate_bank(bank)
    return bank


def validate_instances_against_tracks(bank: ActorAssetBank, tracks: list[Track]) -> None:
    """Cross-check bindings against the bundle's tracks.

    Every bound ``track_id`` must exist. A rigidly-instanced asset whose own
    box disagrees badly with its track's box is warned about — it renders, it
    is just very likely the wrong asset for that track.
    """
    by_id = {track.track_id: track for track in tracks}
    for instance in bank.instances:
        track = by_id.get(instance.track_id)
        if track is None:
            raise ValueError(
                f"actor instance references track {instance.track_id!r}, which is not "
                "among the bundle's sequence tracks"
            )
        if instance.fit_mode != "rigid":
            continue
        asset = bank.asset_by_id(instance.asset_id)
        for axis, (asset_dim, track_dim) in enumerate(zip(asset.size, track.size, strict=True)):
            if track_dim <= 0.0:
                continue
            if abs(asset_dim - track_dim) / track_dim > _SIZE_MISMATCH_RATIO:
                _log.warning(
                    "actor asset %r is %.2f m on axis %d but track %r declares %.2f m; "
                    "rigid instancing keeps the asset's own size",
                    asset.asset_id,
                    asset_dim,
                    axis,
                    track.track_id,
                    track_dim,
                )
                break


# ---------------------------------------------------------------------------
# Payload encoding / decoding
# ---------------------------------------------------------------------------


def _positions(cloud: spz.GaussianCloud) -> np.ndarray:
    return np.asarray(cloud.positions, dtype=np.float64).reshape(cloud.num_points, 3)


def _ext_attributes_block(ext_attrs: dict[str, np.ndarray]) -> dict[str, Any] | None:
    """The ``ext_attributes`` index block, shaped exactly like the background's."""
    if not ext_attrs:
        return None
    block: dict[str, Any] = {
        "extension": EXT_GAUSSIAN_LIDAR_NAME,
        "container": "spz_extension",
        "spz_extension_type": SPZ_EXT_TYPE_TIER4_GAUSSIAN_LIDAR_HEX,
        "attributes": sorted(ext_attrs.keys()),
    }
    sh = ext_attrs.get(RAYDROP_SH_KEY)
    if sh is not None:
        degree = raydrop_sh_degree_from_coefs(int(np.asarray(sh).shape[1]))
        if degree > 0:
            block["raydrop_sh_degree"] = degree
    return block


def encode_actor_asset(source: ActorAssetSource) -> tuple[ActorAsset, bytes]:
    """Encode one source into its index record and its SPZ payload bytes.

    The cloud is written with no implicit axis conversion (``save_spz_world``
    semantics), so the numbers on disk are the object-local numbers in memory.
    """
    cloud = source.cloud
    n = int(cloud.num_points)
    if n <= 0:
        raise ValueError(f"asset {source.asset_id!r}: cloud is empty")
    for name, arr in source.ext_attrs.items():
        if len(np.asarray(arr)) != n:
            raise ValueError(
                f"asset {source.asset_id!r}: ext attribute {name!r} has "
                f"{len(np.asarray(arr))} entries for {n} gaussians"
            )
    positions = _positions(cloud)
    asset = ActorAsset(
        asset_id=source.asset_id,
        class_name=source.class_name,
        size=_as_vec3(source.size, where=f"asset {source.asset_id!r} size"),
        bbox_min=_as_vec3(positions.min(axis=0), where="bbox_min"),
        bbox_max=_as_vec3(positions.max(axis=0), where="bbox_max"),
        n_points=n,
        sh_degree=int(cloud.sh_degree),
        motion=source.motion,
        ext_attributes=_ext_attributes_block(source.ext_attrs),
        provenance=dict(source.provenance),
        metadata=dict(source.metadata),
    )
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "asset.spz"
        save_spz_world(cloud, path)
        payload = path.read_bytes()
    if source.ext_attrs:
        payload = embed_lidar_extension(payload, source.ext_attrs, count=n)
    return asset, payload


def build_actor_asset_bank(
    sources: list[ActorAssetSource],
    instances: list[ActorInstance] | None = None,
) -> tuple[ActorAssetBank, dict[str, bytes]]:
    """Encode ``sources`` into a bank document plus ``{asset_id: spz_bytes}``."""
    assets: list[ActorAsset] = []
    payloads: dict[str, bytes] = {}
    for source in sources:
        asset, payload = encode_actor_asset(source)
        if asset.asset_id in payloads:
            raise ValueError(f"duplicate asset_id: {asset.asset_id!r}")
        assets.append(asset)
        payloads[asset.asset_id] = payload
    bank = ActorAssetBank(assets=assets, instances=list(instances or []))
    _validate_bank(bank)
    return bank, payloads


def decode_actor_asset(
    asset: ActorAsset,
    payload: bytes,
) -> tuple[spz.GaussianCloud, dict[str, np.ndarray]]:
    """Decode one asset payload into its cloud and per-Gaussian ext attributes."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "asset.spz"
        path.write_bytes(payload)
        cloud = load_spz_world(path)
    if cloud.num_points != asset.n_points:
        raise ValueError(
            f"asset {asset.asset_id!r}: payload has {cloud.num_points} points, "
            f"index says {asset.n_points}"
        )
    ext: dict[str, np.ndarray] = {}
    if asset.ext_attributes is not None:
        decoded = extract_lidar_extension(payload)
        if decoded is None:
            raise ValueError(
                f"asset {asset.asset_id!r}: index declares ext_attributes but the "
                "SPZ payload carries no extension record"
            )
        ext = decoded
    return cloud, ext


# ---------------------------------------------------------------------------
# Standalone asset-bank directories
# ---------------------------------------------------------------------------


def load_actor_asset_dir(directory: str | Path) -> tuple[ActorAssetBank, dict[str, bytes]]:
    """Read a standalone asset-bank directory.

    The directory mirrors the in-archive layout exactly — an
    ``actor_assets.json`` at its root with ``actor_assets/<asset_id>/asset.spz``
    payloads beside it — so a bank can be authored once and packed into many
    bundles.
    """
    directory = Path(directory).expanduser()
    doc_path = directory / ACTOR_ASSETS_ARCHIVE_PATH
    if not doc_path.is_file():
        raise FileNotFoundError(f"no {ACTOR_ASSETS_ARCHIVE_PATH} in {directory}")
    bank = parse_actor_assets(json.loads(doc_path.read_text(encoding="utf-8-sig")))
    payloads: dict[str, bytes] = {}
    for asset in bank.assets:
        payload_path = directory / str(asset.uri)
        if not payload_path.is_file():
            raise FileNotFoundError(
                f"asset {asset.asset_id!r}: payload {asset.uri} missing from {directory}"
            )
        payload = payload_path.read_bytes()
        # Cheap integrity gate: the index must describe the payload it names.
        decode_actor_asset(asset, payload)
        payloads[asset.asset_id] = payload
    return bank, payloads


def save_actor_asset_dir(
    directory: str | Path,
    bank: ActorAssetBank,
    payloads: dict[str, bytes],
) -> Path:
    """Write a bank out as a standalone directory (inverse of the loader)."""
    directory = Path(directory).expanduser()
    missing = [a.asset_id for a in bank.assets if a.asset_id not in payloads]
    if missing:
        raise ValueError(f"no payload supplied for asset(s) {missing}")
    directory.mkdir(parents=True, exist_ok=True)
    doc_path = directory / ACTOR_ASSETS_ARCHIVE_PATH
    doc_path.write_text(json.dumps(serialize_actor_assets(bank), indent=2) + "\n", encoding="utf-8")
    for asset in bank.assets:
        payload_path = directory / str(asset.uri)
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload_path.write_bytes(payloads[asset.asset_id])
    return doc_path


# ---------------------------------------------------------------------------
# Cutting an asset out of a world-frame reconstruction
# ---------------------------------------------------------------------------


def rotate_sh_about_z(sh: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate real-SH coefficients about the ``+Z`` axis by ``angle_rad``.

    ``sh`` is ``(n, coefs, 3)`` in the 3DGS / SPZ layout: the DC band is stored
    separately in ``colors``, so ``coefs`` covers bands ``1..degree`` with
    ``m`` ascending from ``-l`` to ``+l`` inside each band.

    Rotation about the SH polar axis mixes only the ``(m, -m)`` pair, so this
    is exact and closed-form (no Wigner-D machinery). ``angle_rad`` is the
    yaw of the *source* frame relative to the target frame: the returned
    coefficients evaluate at ``d`` to what the input evaluated at
    ``R_z(angle_rad) @ d``.
    """
    src = np.asarray(sh, dtype=np.float64)
    coefs = int(src.shape[1])
    out = src.copy()
    for degree in range(1, _MAX_SH_DEGREE + 1):
        base = degree * degree - 1  # coefficients of bands 1..degree-1
        if base + 2 * degree >= coefs:
            break
        for m in range(1, degree + 1):
            cos_m = math.cos(m * angle_rad)
            sin_m = math.sin(m * angle_rad)
            i_pos = base + degree + m
            i_neg = base + degree - m
            c_pos = src[:, i_pos, :]
            c_neg = src[:, i_neg, :]
            out[:, i_pos, :] = c_pos * cos_m + c_neg * sin_m
            out[:, i_neg, :] = -c_pos * sin_m + c_neg * cos_m
    return out


def extract_actor_asset(
    cloud: spz.GaussianCloud,
    *,
    asset_id: str,
    class_name: str,
    size: tuple[float, float, float],
    translation: tuple[float, float, float],
    rotation: tuple[float, float, float, float],
    ext_attrs: dict[str, np.ndarray] | None = None,
    margin: float = 0.0,
    sh_policy: str = "rotate",
    provenance: dict[str, Any] | None = None,
) -> ActorAssetSource:
    """Cut one object out of a world-frame cloud into an object-local asset.

    This is the canonical way to turn a track's box — its ``size`` plus one of
    its :class:`~3dgs_io.TrackFrame` poses, in the bundle's ENU world frame —
    into a rigid asset: Gaussians inside the box are selected, then transformed
    by the inverse pose so the box centre lands on the object-local origin and
    ``+X`` points the way the object drives.

    ``margin`` grows the selection box on every side (metres). Tracked boxes
    are rarely tight, and a splat whose centre sits just inside the box can
    still have its tail outside it.

    ``sh_policy`` decides what happens to view-dependent bands, whose
    coefficients are expressed in the source (world) frame:

    * ``"rotate"`` (default) — re-express them in the object frame. Exact for
      the yaw-only poses road vehicles have; a pose with more than ~2° of
      pitch/roll raises, because that case needs full SH rotation.
    * ``"drop"`` — emit a view-independent (``sh_degree = 0``) asset. Always
      safe, loses specular highlights.
    * ``"keep"`` — copy the coefficients verbatim, for callers whose SH is
      already object-aligned.

    Colours, opacities and scales are copied untouched; only the means, the
    orientation quaternions and (under ``"rotate"``) the SH bands move, exactly
    as a rigid transform should.
    """
    if sh_policy not in SH_POLICIES:
        raise ValueError(f"unknown sh_policy {sh_policy!r}; expected one of {list(SH_POLICIES)}")

    n = int(cloud.num_points)
    rot = Rotation.from_quat(np.asarray(rotation, dtype=np.float64))
    centre = np.asarray(translation, dtype=np.float64)
    half = np.asarray(size, dtype=np.float64) / 2.0 + float(margin)
    if np.any(half <= 0.0):
        raise ValueError(f"asset {asset_id!r}: size must be positive, got {size}")

    local = rot.inv().apply(_positions(cloud) - centre)
    inside = np.all(np.abs(local) <= half, axis=1)
    count = int(inside.sum())
    if count == 0:
        raise ValueError(
            f"asset {asset_id!r}: no gaussians inside the box at {tuple(centre)} "
            f"(size {size}, margin {margin})"
        )

    quats = np.asarray(cloud.rotations, dtype=np.float64).reshape(n, 4)[inside]

    out = spz.GaussianCloud()
    out.antialiased = cloud.antialiased
    out.positions = np.ascontiguousarray(local[inside], dtype=np.float32).reshape(-1)
    out.rotations = np.ascontiguousarray(
        (rot.inv() * Rotation.from_quat(quats)).as_quat(), dtype=np.float32
    ).reshape(-1)
    out.scales = np.ascontiguousarray(
        np.asarray(cloud.scales, dtype=np.float32).reshape(n, 3)[inside]
    ).reshape(-1)
    out.colors = np.ascontiguousarray(
        np.asarray(cloud.colors, dtype=np.float32).reshape(n, 3)[inside]
    ).reshape(-1)
    out.alphas = np.ascontiguousarray(
        np.asarray(cloud.alphas, dtype=np.float32).reshape(n)[inside]
    ).reshape(-1)

    sh_flat = np.asarray(cloud.sh, dtype=np.float32)
    degree = int(cloud.sh_degree)
    if degree > 0 and sh_flat.size and sh_policy != "drop":
        sh = sh_flat.reshape(n, -1, 3)[inside]
        if sh_policy == "rotate":
            yaw, pitch, roll = rot.as_euler("zyx")
            if max(abs(pitch), abs(roll)) > _MAX_NON_YAW_RAD:
                raise ValueError(
                    f"asset {asset_id!r}: pose has {math.degrees(pitch):.1f}° pitch / "
                    f"{math.degrees(roll):.1f}° roll, which yaw-only SH rotation cannot "
                    'express; pass sh_policy="drop" or supply a yaw-only pose'
                )
            sh = rotate_sh_about_z(sh, yaw)
        out.sh_degree = degree
        out.sh = np.ascontiguousarray(sh, dtype=np.float32).reshape(-1)
    else:
        out.sh_degree = 0
        out.sh = np.zeros(0, dtype=np.float32)

    return ActorAssetSource(
        asset_id=asset_id,
        cloud=out,
        class_name=class_name,
        size=_as_vec3(size, where=f"asset {asset_id!r} size"),
        ext_attrs={name: np.asarray(arr)[inside] for name, arr in (ext_attrs or {}).items()},
        provenance=dict(provenance or {}),
    )
