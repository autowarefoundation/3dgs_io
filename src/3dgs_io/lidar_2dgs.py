"""LiDAR 2D Gaussian Splatting (2DGS) I/O for glTF/GLB files.

Stores flat surfel-like 2D Gaussians with LiDAR-specific properties
(reflectance, ray-tracing parameters) using standard glTF POINTS
primitives with custom ``_``-prefixed attributes.

This module intentionally does **not** use the ``KHR_gaussian_splatting``
extension — LiDAR 2DGS data is semantically distinct from camera 3DGS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ._gltf_common import (
    FLOAT as _FLOAT,
)
from ._gltf_common import (
    pack_buffer as _pack_buffer,
)
from ._gltf_common import (
    read_accessor as _read_accessor,
)
from ._gltf_common import (
    read_file as _read_file,
)
from ._gltf_common import (
    write_file as _write_file,
)
from .metadata import GlbMetadata, parse_metadata, serialize_metadata


@dataclass
class LidarGaussianCloud:
    """A cloud of 2D Gaussian surfels for LiDAR simulation.

    All arrays use ``float32`` and are stored **flat** (1-D) unless noted.
    Reshape with ``(num_points, K)`` where *K* is the per-point component count.
    """

    num_points: int = 0
    """Number of surfels."""

    positions: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    """Flat (N*3,) float32 — surfel centre positions."""

    normals: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    """Flat (N*3,) float32 — surface normal directions (unit vectors)."""

    scales_2d: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    """Flat (N*2,) float32 — 2-D scale along tangent / bi-tangent."""

    rotations: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    """Flat (N*4,) float32 — quaternion (x, y, z, w) orientation."""

    reflectance: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    """Flat (N,) float32 — LiDAR reflectance / intensity in [0, 1]."""

    opacity: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    """Flat (N,) float32 — opacity in [0, 1]."""

    rt_properties: dict[str, np.ndarray] = field(default_factory=dict)
    """Extensible dict of per-point ray-tracing properties.

    Keys become custom glTF attributes prefixed with ``_RT_`` (upper-cased).
    Values must be float32 arrays of shape ``(N,)`` or ``(N, K)``."""


def save_lidar_gltf(
    cloud: LidarGaussianCloud,
    path: str | Path,
    *,
    metadata: GlbMetadata | dict[str, Any] | None = None,
) -> None:
    """Save a :class:`LidarGaussianCloud` to a glTF/GLB file."""
    path = Path(path)
    n = cloud.num_points
    if n == 0:
        raise ValueError("Cannot save an empty LidarGaussianCloud")

    positions = np.asarray(cloud.positions, dtype=np.float32).reshape(n, 3)
    normals = np.asarray(cloud.normals, dtype=np.float32).reshape(n, 3)
    scales_2d = np.asarray(cloud.scales_2d, dtype=np.float32).reshape(n, 2)
    rotations = np.asarray(cloud.rotations, dtype=np.float32).reshape(n, 4)
    reflectance = np.asarray(cloud.reflectance, dtype=np.float32).reshape(n)
    opacity = np.asarray(cloud.opacity, dtype=np.float32).reshape(n)

    pos_min = positions.min(axis=0).tolist()
    pos_max = positions.max(axis=0).tolist()

    # (name, bytes, componentType, accessorType, extra_fields)
    attr_list: list[tuple[str, bytes, int, str, dict]] = [
        ("POSITION", positions.tobytes(), _FLOAT, "VEC3", {"min": pos_min, "max": pos_max}),
        ("NORMAL", normals.tobytes(), _FLOAT, "VEC3", {}),
        ("_SCALE_2D", scales_2d.tobytes(), _FLOAT, "VEC2", {}),
        ("_ROTATION", rotations.tobytes(), _FLOAT, "VEC4", {}),
        ("_REFLECTANCE", reflectance.tobytes(), _FLOAT, "SCALAR", {}),
        ("_OPACITY", opacity.tobytes(), _FLOAT, "SCALAR", {}),
    ]

    for key, arr in sorted(cloud.rt_properties.items()):
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            attr_list.append((f"_RT_{key.upper()}", arr.tobytes(), _FLOAT, "SCALAR", {}))
        else:
            components = arr.shape[1]
            type_map = {2: "VEC2", 3: "VEC3", 4: "VEC4"}
            acc_type = type_map.get(components)
            if acc_type is None:
                raise ValueError(
                    f"RT property '{key}' has unsupported component count {components}"
                )
            attr_list.append((f"_RT_{key.upper()}", arr.tobytes(), _FLOAT, acc_type, {}))

    buffer_data, offsets, lengths = _pack_buffer([d for _, d, _, _, _ in attr_list])

    attrs_dict = {name: i for i, (name, _, _, _, _) in enumerate(attr_list)}
    num_attrs = len(attr_list)

    accessors = []
    for _, _, comp_type, acc_type, extras in attr_list:
        acc: dict = {
            "bufferView": len(accessors),
            "componentType": comp_type,
            "count": n,
            "type": acc_type,
        }
        acc.update(extras)
        accessors.append(acc)

    asset: dict[str, Any] = {"version": "2.0", "generator": "3dgs-io"}
    extras = serialize_metadata(metadata)
    if extras is not None:
        asset["extras"] = extras

    gltf_dict: dict = {
        "asset": asset,
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {
                        "mode": 0,
                        "attributes": attrs_dict,
                    }
                ]
            }
        ],
        "accessors": accessors,
        "bufferViews": [
            {"buffer": 0, "byteOffset": offsets[i], "byteLength": lengths[i]}
            for i in range(num_attrs)
        ],
        "buffers": [{"byteLength": len(buffer_data)}],
    }

    _write_file(path, gltf_dict, buffer_data)


def load_lidar_gltf(path: str | Path) -> LidarGaussianCloud:
    """Load a :class:`LidarGaussianCloud` from a glTF/GLB file."""
    cloud, _ = load_lidar_gltf_with_metadata(path)
    return cloud


def load_lidar_gltf_with_metadata(
    path: str | Path,
) -> tuple[LidarGaussianCloud, GlbMetadata | dict[str, Any] | None]:
    """Load a :class:`LidarGaussianCloud` and ``asset.extras`` metadata."""
    path = Path(path)
    gltf_dict, buffer_data = _read_file(path)
    cloud = _parse_lidar_cloud(gltf_dict, buffer_data)
    raw_metadata = gltf_dict.get("asset", {}).get("extras")
    return cloud, parse_metadata(raw_metadata)


def _find_lidar_primitive(gltf_dict: dict) -> dict | None:
    """Find the first primitive that has ``_SCALE_2D`` and ``_REFLECTANCE``."""
    for mesh in gltf_dict.get("meshes", []):
        for prim in mesh.get("primitives", []):
            a = prim.get("attributes", {})
            if "_SCALE_2D" in a and "_REFLECTANCE" in a:
                return prim
    return None


def _parse_lidar_cloud(gltf_dict: dict, buffer_data: bytes) -> LidarGaussianCloud:
    primitive = _find_lidar_primitive(gltf_dict)
    if primitive is None:
        raise ValueError("No LiDAR 2DGS primitive found (missing _SCALE_2D / _REFLECTANCE)")

    attrs = primitive["attributes"]
    accessors = gltf_dict["accessors"]
    bvs = gltf_dict["bufferViews"]

    positions = _read_accessor(accessors[attrs["POSITION"]], bvs, buffer_data)
    n = positions.shape[0] if positions.ndim > 1 else len(positions) // 3

    normals = _read_accessor(accessors[attrs["NORMAL"]], bvs, buffer_data)
    scales_2d = _read_accessor(accessors[attrs["_SCALE_2D"]], bvs, buffer_data)
    rotations = _read_accessor(accessors[attrs["_ROTATION"]], bvs, buffer_data)
    reflectance = _read_accessor(accessors[attrs["_REFLECTANCE"]], bvs, buffer_data)

    opacity: np.ndarray
    if "_OPACITY" in attrs:
        opacity = _read_accessor(accessors[attrs["_OPACITY"]], bvs, buffer_data)
    else:
        opacity = np.ones(n, dtype=np.float32)

    rt_properties: dict[str, np.ndarray] = {}
    for key, idx in attrs.items():
        if key.startswith("_RT_"):
            prop_name = key[4:].lower()
            rt_properties[prop_name] = _read_accessor(accessors[idx], bvs, buffer_data).astype(
                np.float32, copy=False
            )

    return LidarGaussianCloud(
        num_points=n,
        positions=positions.reshape(-1),
        normals=normals.reshape(-1),
        scales_2d=scales_2d.reshape(-1),
        rotations=rotations.reshape(-1),
        reflectance=reflectance.reshape(-1),
        opacity=opacity.reshape(-1),
        rt_properties=rt_properties,
    )
