"""USDZ I/O for the alpasim (NVIDIA NRE / NuRec) reconstruction format.

Only the background gaussian layer is supported here; dynamic objects, the
ground mesh and other auxiliary assets are written as empty placeholders when
the file is required by the format but its content is not available.

================================================================================
USDZ archive layout (alpasim format)
================================================================================

A USDZ file is a ``ZIP_STORED`` archive (per the Pixar USDZ spec). The alpasim
sample archive does not align asset payloads to 64 bytes, so we rely on plain
``zipfile`` access.

Required for a background-only file are the first three rows; the rest are
preserved when present on load but written as empty placeholders on save.

| File                              | Req? | Description                                |
| --------------------------------- | ---- | ------------------------------------------ |
| ``default.usda``                  | yes  | Root USD stage; references ``volume.usda`` |
| ``volume.usda``                   | yes  | NuRec Volume prim → ``volume.nurec``       |
| ``volume.nurec``                  | yes  | gzip+MessagePack encoded ``nre_data``      |
| ``mesh_ground.usd`` / ``.ply``    | no   | Ground mesh                                |
| ``rig_trajectories.usda``/``json``| no   | Sensor rig poses                           |
| ``sequence_tracks.usda``/``json`` | no   | Dynamic object track boxes                 |
| ``checkpoint.ckpt``               | no   | Training checkpoint                        |
| ``metadata.yaml``                 | no   | Scene metadata (sensors, time_range, …)    |
| ``data_info.json``                | no   | Shard info                                 |
| ``pose_record.json``              | no   | Per-frame poses                            |
| ``parsed_config.yaml``            | no   | Full training config                       |
| ``map.xodr``                      | no   | OpenDRIVE map                              |
| ``datasource_summary.json``       | no   | Datasource summary                         |
| ``clipgt/*.parquet``              | no   | Ground-truth annotations                   |

================================================================================
``volume.nurec`` structure (gunzip → msgpack → nested dict)
================================================================================

Top-level::

    {
      "nre_data": {
        "version": str,         # e.g. "26.4.96"
        "model":   str,         # always "nre" in the alpasim samples
        "config":  dict,        # full training/render config (see _minimal_config)
        "state_dict": dict,     # flattened PyTorch state dict (dotted keys)
      }
    }

Each tensor entry in ``state_dict`` is stored as raw little-endian bytes; an
accompanying ``<key>.shape`` entry (``list[int]``) gives the tensor shape.
The state-dict keys for the background gaussian layer (``N`` = #gaussians,
key prefix omitted: ``BG = .gaussians_nodes.background``):

| key                            | dtype | shape       | meaning                            |
| ------------------------------ | ----- | ----------- | ---------------------------------- |
| ``BG.positions``               | fp16  | (N, 3)      | xyz pos (NRE-local frame)          |
| ``BG.rotations``               | fp16  | (N, 4)      | quat (activation: normalize)       |
| ``BG.scales``                  | fp16  | (N, 3)      | log-scale (activation: exp)        |
| ``BG.densities``               | fp16  | (N, 1)      | logit opacity (act: sigmoid)       |
| ``BG.features_albedo``         | fp16  | (N, F, 3)   | Fourier-feature albedo, F = dim    |
| ``BG.features_specular``       | fp16  | (N, K)      | SH specular, K=((d+1)^2 - O0) * 3  |
| ``BG.camera_extra_signal``     | fp16  | (N, C)      | camera extra signal (semantic etc) |
| ``BG.lidar_extra_signal``      | fp16  | (N, L)      | lidar extra signal                 |
| ``BG.extra_signal``            | fp16  | (N, E)      | shared extra signal                |
| ``BG.n_active_features``       | int64 | ()          | active feature count (progressive) |
| ``BG.time_embed._extra_state`` | dict  | -           | timestamps_us_{min,max}            |
| ``.background.textures``       | fp16  | (1,6,H,W,3) | sky cubemap RGB                    |
| ``.background.texture_grads``  | fp16  | (6,H/2,W/2) | sky cubemap grad mask              |
| ``.background._extra_state``   | dict  | -           | ``{n_grad_updates}``               |
| ``.gaussians_strategy.invisible_steps.background`` | int32 | (N,) | invis ctr (MCMC) |

The top-level ``._extra_state.obj_track_ids`` dict lists track IDs per layer;
for a background-only file it is an empty list.
"""

from __future__ import annotations

import gzip
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import msgpack
except ImportError as _e:  # pragma: no cover - guarded at import time
    raise ImportError(
        "USDZ I/O requires the `msgpack` package; install with `pip install msgpack`"
    ) from _e


__all__ = [
    "AlpasimGaussianCloud",
    "AlpasimSkyCubemap",
    "load_usdz",
    "save_usdz",
]


_DEFAULT_USDA = """#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{
    over "volume" (
        prepend references = @volume.usda@
    )
    {
    }
}
"""

_VOLUME_USDA_TEMPLATE = """#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{{
    def Volume "volume"
    {{
        float3[] extent = [(-inf, -inf, -inf), (inf, inf, inf)]
        custom rel field:density = </World/volume/density_field>
        custom rel field:emissiveColor = </World/volume/emissive_color_field>
        custom float3 omni:nurec:crop:maxBounds = (inf, inf, inf)
        custom float3 omni:nurec:crop:minBounds = (-inf, -inf, -inf)
        custom bool omni:nurec:isNuRecVolume = 1
        custom float3 omni:nurec:offset = ({ox}, {oy}, {oz})
        custom bool omni:nurec:useProxyTransform = 0

        def OmniNuRecFieldAsset "density_field"
        {{
            custom token fieldDataType = "float"
            custom token fieldName = "density"
            custom token fieldRole = "density"
            custom asset filePath = @./volume.nurec@
        }}

        def OmniNuRecFieldAsset "emissive_color_field"
        {{
            custom token fieldDataType = "float3"
            custom token fieldName = "emissiveColor"
            custom token fieldRole = "emissiveColor"
            custom asset filePath = @./volume.nurec@
            custom float4 omni:nurec:ccmB = (0, 0, 1, 0)
            custom float4 omni:nurec:ccmG = (0, 1, 0, 0)
            custom float4 omni:nurec:ccmR = (1, 0, 0, 0)
        }}
    }}
}}
"""


@dataclass
class AlpasimSkyCubemap:
    """Sky environment cubemap stored alongside the background gaussians."""

    textures: np.ndarray  # (1, 6, H, W, 3) fp16
    texture_grads: np.ndarray | None = None  # (6, H/2, W/2) fp16
    n_grad_updates: int = 0


@dataclass
class AlpasimGaussianCloud:
    """Background gaussian layer extracted from (or to be written to) ``volume.nurec``.

    Tensors are kept in their native fp16 representation so that the bytes
    round-trip exactly. ``positions`` are in the NuRec-local frame; combine
    with ``nre_offset`` (taken from ``volume.usda``) to recover world coords.
    """

    positions: np.ndarray  # (N, 3) fp16
    rotations: np.ndarray  # (N, 4) fp16
    scales: np.ndarray  # (N, 3) fp16 (log-space; activation = exp)
    densities: np.ndarray  # (N, 1) fp16 (logit; activation = sigmoid)
    features_albedo: np.ndarray  # (N, F, 3) fp16
    features_specular: np.ndarray  # (N, K) fp16
    camera_extra_signal: np.ndarray | None = None  # (N, C) fp16
    lidar_extra_signal: np.ndarray | None = None  # (N, L) fp16
    extra_signal: np.ndarray | None = None  # (N, E) fp16
    n_active_features: int = 3
    timestamps_us_min: int = 0
    timestamps_us_max: int = 0
    invisible_steps: np.ndarray | None = None  # (N,) int32
    sky: AlpasimSkyCubemap | None = None
    nre_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    version: str = "26.4.96"
    config: dict[str, Any] | None = None  # full nre_data.config; written verbatim if set

    @property
    def num_points(self) -> int:
        return int(self.positions.shape[0])


# ---------------------------------------------------------------------------
# msgpack helpers
# ---------------------------------------------------------------------------


def _put_tensor(sd: dict, key: str, arr: np.ndarray) -> None:
    sd[key] = np.ascontiguousarray(arr).tobytes()
    sd[key + ".shape"] = list(arr.shape)


def _get_tensor(sd: dict, key: str, dtype: np.dtype) -> np.ndarray:
    raw = sd[key]
    shape = sd[key + ".shape"]
    arr = np.frombuffer(raw, dtype=dtype)
    if not shape:
        return arr.reshape(()).copy()
    return arr.reshape(shape).copy()


# ---------------------------------------------------------------------------
# Default ``nre_data.config`` for a background-only save. Sections required
# by NuRec but unused for background-only playback are filled with minimal
# but structurally valid placeholders.
# ---------------------------------------------------------------------------


def _minimal_bg_layer_config(cloud: AlpasimGaussianCloud) -> dict:
    fourier_dim = int(cloud.features_albedo.shape[1])
    spec_per_channel = int(cloud.features_specular.shape[1]) // 3
    # SH degree d satisfies (d+1)^2 == spec_per_channel + (1 if O0 else 0).
    sph_O0 = True
    n_per_ch = spec_per_channel + (1 if sph_O0 else 0)
    sph_deg = int(round(math.sqrt(n_per_ch))) - 1

    def _dim(arr: np.ndarray | None) -> int:
        return int(arr.shape[1]) if (arr is not None and arr.ndim == 2) else 0

    return {
        "name": "sh-gaussians",
        "nearest_neighbor_track_for_lidar": False,
        "initialization": {"name": "empty"},
        "ignore_classes_from_layers": [],
        "use_slang": False,
        "density_activation": "sigmoid",
        "scale_activation": "exp",
        "rotation_activation": "normalize",
        "progressive_training": {
            "init_n_features": 0,
            "max_n_features": max(0, sph_deg),
            "increase_frequency": 1,
            "increase_step": 1,
        },
        "particle": {
            "density_kernel_planar": False,
            "density_kernel_degree": 2,
            "density_kernel_min_response": 0.0113,
            "ray_spread_filter_enabled": False,
            "radiance_sph_degree": sph_deg,
            "radiance_sph_O0": sph_O0,
            "extra_signal_dim": _dim(cloud.extra_signal),
            "camera_extra_signal_dim": _dim(cloud.camera_extra_signal),
            "lidar_extra_signal_dim": _dim(cloud.lidar_extra_signal),
            "lidar_extra_signal_sph_degree": 0,
        },
        "fourier_features_dim": fourier_dim,
        "time_embed": {
            "name": "holistic-remap-time-input-embedding",
            "remap_min": 0,
            "remap_max": 1,
        },
        "precision": 32,
        "transmittance_threshold": 0.0001,
        "scale_pos_lr_by_scene_extent": True,
        "debug_viz": False,
        "extra_signal": {},
        "tracks": {"is_standalone": True},
    }


def _minimal_config(cloud: AlpasimGaussianCloud) -> dict:
    if cloud.sky is not None:
        h = int(cloud.sky.textures.shape[2])
        w = int(cloud.sky.textures.shape[3])
    else:
        h = w = 1
    return {
        "name": "gaussians-composite",
        "debug_viz": False,
        "log_to_prog_bar": True,
        "saturate_radiance": True,
        "train_num_rays": 0,
        "layers": {"background": _minimal_bg_layer_config(cloud)},
        "calib": {
            "name": "free-pose-calib",
            "enabled": False,
            "start_global_step": 0,
            "skip_first_pose_delta": True,
            "enable_torch_compile": False,
            "lidar": {"enabled": False},
            "camera": {"enabled": False},
            "optimizer": {"name": "fused_adam", "args": {}},
            "scheduler": {
                "name": "ExponentialLR",
                "interval": "step",
                "milestones": [],
                "schedulers": [],
            },
        },
        "background": {
            "name": "sky-env-map",
            "composite_in_linear_space": False,
            "width": w,
            "height": h,
            "envmap_type": "cubemap",
            "should_inpaint": False,
            "inpaint_threshold": 0.05,
            "inpaint_kernel_size": 10,
            "min_grad_updates": 0,
            "saturate_radiance": True,
        },
        "renderer": {"name": "3dgut-nrend"},
        "strategy": {
            "name": "mcmc",
            "print_stats": False,
            "exclude_layer_ids": [],
            "binom_n_max": 51,
            "opacity_threshold": 0.005,
            "relocate": {
                "start_iteration": 0,
                "end_iteration": 0,
                "frequency": 0,
                "max_invisible_steps": 0,
            },
            "perturb": {
                "start_iteration": 0,
                "end_iteration": 0,
                "frequency": 0,
                "noise_lr": {},
                "move_outside_of_cuboid": False,
            },
            "add": {
                "start_iteration": 0,
                "end_iteration": 0,
                "frequency": 0,
                "max_n_gaussians": 0,
            },
        },
        "post_processing": {},
        "appearance_embedding": {
            "name": "skip-appearance",
            "embedding_dim": 0,
            "device": "cuda",
        },
    }


# ---------------------------------------------------------------------------
# volume.nurec ↔ AlpasimGaussianCloud
# ---------------------------------------------------------------------------


def _parse_volume_nurec(raw: bytes, nre_offset: tuple[float, float, float]) -> AlpasimGaussianCloud:
    decompressed = gzip.decompress(raw)
    obj = msgpack.unpackb(
        decompressed,
        raw=False,
        strict_map_key=False,
    )
    nre = obj["nre_data"]
    sd = nre["state_dict"]
    cfg = nre.get("config", {})

    def _opt(key: str, dtype: np.dtype) -> np.ndarray | None:
        if key in sd and sd[key]:
            return _get_tensor(sd, key, dtype)
        return None

    positions = _get_tensor(sd, ".gaussians_nodes.background.positions", np.float16)
    rotations = _get_tensor(sd, ".gaussians_nodes.background.rotations", np.float16)
    scales = _get_tensor(sd, ".gaussians_nodes.background.scales", np.float16)
    densities = _get_tensor(sd, ".gaussians_nodes.background.densities", np.float16)
    features_albedo = _get_tensor(sd, ".gaussians_nodes.background.features_albedo", np.float16)
    features_specular = _get_tensor(sd, ".gaussians_nodes.background.features_specular", np.float16)

    cam_extra = _opt(".gaussians_nodes.background.camera_extra_signal", np.float16)
    lid_extra = _opt(".gaussians_nodes.background.lidar_extra_signal", np.float16)
    extra = _opt(".gaussians_nodes.background.extra_signal", np.float16)

    n_active = 3
    if ".gaussians_nodes.background.n_active_features" in sd:
        n_active = int(
            np.frombuffer(sd[".gaussians_nodes.background.n_active_features"], dtype=np.int64)[0]
        )

    time_es = sd.get(".gaussians_nodes.background.time_embed._extra_state", {}) or {}

    invisible = None
    if ".gaussians_strategy.invisible_steps.background" in sd:
        invisible = _get_tensor(sd, ".gaussians_strategy.invisible_steps.background", np.int32)

    sky: AlpasimSkyCubemap | None = None
    if ".background.textures" in sd:
        tex = _get_tensor(sd, ".background.textures", np.float16)
        grads = (
            _get_tensor(sd, ".background.texture_grads", np.float16)
            if ".background.texture_grads" in sd
            else None
        )
        bg_es = sd.get(".background._extra_state", {}) or {}
        sky = AlpasimSkyCubemap(
            textures=tex,
            texture_grads=grads,
            n_grad_updates=int(bg_es.get("n_grad_updates", 0)),
        )

    return AlpasimGaussianCloud(
        positions=positions,
        rotations=rotations,
        scales=scales,
        densities=densities,
        features_albedo=features_albedo,
        features_specular=features_specular,
        camera_extra_signal=cam_extra,
        lidar_extra_signal=lid_extra,
        extra_signal=extra,
        n_active_features=n_active,
        timestamps_us_min=int(time_es.get("timestamps_us_min", 0)),
        timestamps_us_max=int(time_es.get("timestamps_us_max", 0)),
        invisible_steps=invisible,
        sky=sky,
        nre_offset=nre_offset,
        version=str(nre.get("version", "0.0.0")),
        config=cfg,
    )


def _serialize_volume_nurec(cloud: AlpasimGaussianCloud) -> bytes:
    n = cloud.num_points
    sd: dict[str, Any] = {"._extra_state": {"obj_track_ids": {"background": []}}}

    _put_tensor(sd, ".gaussians_nodes.background.positions", cloud.positions)
    _put_tensor(sd, ".gaussians_nodes.background.rotations", cloud.rotations)
    _put_tensor(sd, ".gaussians_nodes.background.scales", cloud.scales)
    _put_tensor(sd, ".gaussians_nodes.background.densities", cloud.densities)
    _put_tensor(sd, ".gaussians_nodes.background.features_albedo", cloud.features_albedo)
    _put_tensor(sd, ".gaussians_nodes.background.features_specular", cloud.features_specular)

    def _opt_arr(arr: np.ndarray | None, last_dim: int = 0) -> np.ndarray:
        if arr is not None:
            return arr
        return np.zeros((n, last_dim), dtype=np.float16)

    _put_tensor(sd, ".gaussians_nodes.background.extra_signal", _opt_arr(cloud.extra_signal))
    _put_tensor(
        sd,
        ".gaussians_nodes.background.camera_extra_signal",
        _opt_arr(cloud.camera_extra_signal),
    )
    _put_tensor(
        sd,
        ".gaussians_nodes.background.lidar_extra_signal",
        _opt_arr(cloud.lidar_extra_signal),
    )

    n_active = np.array(int(cloud.n_active_features), dtype=np.int64)
    _put_tensor(
        sd,
        ".gaussians_nodes.background.n_active_features",
        n_active.reshape(()),
    )
    sd[".gaussians_nodes.background.time_embed._extra_state"] = {
        "timestamps_us_min": int(cloud.timestamps_us_min),
        "timestamps_us_max": int(cloud.timestamps_us_max),
    }

    if cloud.sky is not None:
        _put_tensor(sd, ".background.textures", cloud.sky.textures)
        if cloud.sky.texture_grads is not None:
            _put_tensor(sd, ".background.texture_grads", cloud.sky.texture_grads)
        sd[".background._extra_state"] = {"n_grad_updates": int(cloud.sky.n_grad_updates)}
    else:
        # Required by the model but unused for background-only payloads.
        _put_tensor(
            sd,
            ".background.textures",
            np.zeros((1, 6, 1, 1, 3), dtype=np.float16),
        )
        sd[".background._extra_state"] = {"n_grad_updates": 0}

    if cloud.invisible_steps is not None:
        _put_tensor(
            sd,
            ".gaussians_strategy.invisible_steps.background",
            cloud.invisible_steps.astype(np.int32, copy=False),
        )

    nre = {
        "version": cloud.version,
        "model": "nre",
        "config": cloud.config if cloud.config is not None else _minimal_config(cloud),
        "state_dict": sd,
    }
    raw = msgpack.packb({"nre_data": nre}, use_bin_type=True)
    return gzip.compress(raw, compresslevel=1)


# ---------------------------------------------------------------------------
# USDZ archive helpers
# ---------------------------------------------------------------------------

_USDA_NRE_OFFSET_RE = "custom float3 omni:nurec:offset"


def _read_nre_offset(volume_usda: str) -> tuple[float, float, float]:
    for line in volume_usda.splitlines():
        if _USDA_NRE_OFFSET_RE in line:
            lhs, _, rhs = line.partition("=")
            rhs = rhs.strip().strip("()")
            parts = [p.strip() for p in rhs.split(",")]
            if len(parts) == 3:
                try:
                    return (float(parts[0]), float(parts[1]), float(parts[2]))
                except ValueError:
                    pass
    return (0.0, 0.0, 0.0)


def load_usdz(path: str | Path) -> AlpasimGaussianCloud:
    """Load an alpasim USDZ archive and return the background gaussian layer."""
    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        try:
            volume_usda = zf.read("volume.usda").decode("utf-8", errors="replace")
        except KeyError:
            volume_usda = ""
        try:
            raw = zf.read("volume.nurec")
        except KeyError as e:
            raise ValueError(f"{path}: not an alpasim USDZ (missing volume.nurec)") from e
    offset = _read_nre_offset(volume_usda)
    return _parse_volume_nurec(raw, offset)


def save_usdz(cloud: AlpasimGaussianCloud, path: str | Path) -> None:
    """Save a background gaussian layer as an alpasim-compatible USDZ archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    volume_usda = _VOLUME_USDA_TEMPLATE.format(
        ox=cloud.nre_offset[0],
        oy=cloud.nre_offset[1],
        oz=cloud.nre_offset[2],
    )
    nurec_bytes = _serialize_volume_nurec(cloud)

    entries = [
        ("default.usda", _DEFAULT_USDA.encode("utf-8")),
        ("volume.usda", volume_usda.encode("utf-8")),
        ("volume.nurec", nurec_bytes),
    ]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        for name, data in entries:
            zi = zipfile.ZipInfo(name)
            zi.compress_type = zipfile.ZIP_STORED
            zf.writestr(zi, data)
