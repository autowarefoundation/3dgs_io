from __future__ import annotations

import gzip
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import spz

from ._gltf_common import (
    FLOAT as _FLOAT,
)
from ._gltf_common import (
    UNSIGNED_BYTE as _UNSIGNED_BYTE,
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
from .ext_attributes import (
    DEFAULT_LIDAR_SPECS,
    EXT_GAUSSIAN_LIDAR_NAME,
    ExtAttributeSpec,
)
from .metadata import DatasetType as DatasetType  # noqa: F401 – re-export for backward compat
from .metadata import GlbMetadata, parse_metadata, serialize_metadata

_EXTENSION_NAME = "KHR_gaussian_splatting"
_SPZ_EXTENSION_NAME = "KHR_gaussian_splatting_compression_spz_2"

# Spherical Harmonics degree-0 constant: 1 / (2 * sqrt(pi))
_SH_C0 = 0.2820947917738781


@dataclass
class GltfSaveOptions:
    """Options for saving to glTF/GLB format."""

    spz_compression: bool = False
    """Use SPZ compression (KHR_gaussian_splatting_compression_spz_2)."""

    metadata: GlbMetadata | dict[str, Any] | None = None
    """Metadata written to ``asset.extras`` in the glTF JSON.

    Accepts a :class:`GlbMetadata` instance (recommended) or a plain dict."""


# ---------------------------------------------------------------------------
# Activation helpers (GaussianCloud uses pre-activation values)
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64))).astype(np.float32)


def _inverse_sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float64), 1e-7, 1 - 1e-7)
    return np.log(x / (1 - x)).astype(np.float32)


# ---------------------------------------------------------------------------
# EXT_gaussian_lidar quantization helpers
# ---------------------------------------------------------------------------

_EXT_SPEC_BY_NAME: dict[str, ExtAttributeSpec] = {s.name: s for s in DEFAULT_LIDAR_SPECS}


def _spec_for(name: str) -> ExtAttributeSpec:
    return _EXT_SPEC_BY_NAME.get(name, ExtAttributeSpec(name=name, quantization="u8_sigmoid"))


def _quantize_ext(arr: np.ndarray, spec: ExtAttributeSpec) -> bytes:
    """Quantize a (N,) float array per ``spec`` for storage in a glTF accessor."""
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if spec.quantization == "u8_sigmoid":
        sig = 1.0 / (1.0 + np.exp(-arr.astype(np.float64)))
        q = np.clip(np.round(sig * 255.0), 0.0, 255.0).astype(np.uint8)
        return q.tobytes()
    if spec.quantization == "u8_linear":
        scaled = (arr.astype(np.float64) - spec.vmin) / max(spec.vmax - spec.vmin, 1e-12)
        q = np.clip(np.round(scaled * 255.0), 0.0, 255.0).astype(np.uint8)
        return q.tobytes()
    raise ValueError(f"unsupported ext quantization {spec.quantization!r}")


def _dequantize_ext(raw: np.ndarray, spec: ExtAttributeSpec) -> np.ndarray:
    """Inverse of :func:`_quantize_ext` — returns float32 in the original (logit/raw) space."""
    if spec.quantization == "u8_sigmoid":
        q = raw.astype(np.float64) / 255.0
        q = np.clip(q, 1e-9, 1.0 - 1e-9)
        return np.log(q / (1.0 - q)).astype(np.float32)
    if spec.quantization == "u8_linear":
        q = raw.astype(np.float64) / 255.0
        return (q * (spec.vmax - spec.vmin) + spec.vmin).astype(np.float32)
    raise ValueError(f"unsupported ext quantization {spec.quantization!r}")


def _validate_ext_attributes(
    ext_attributes: dict[str, np.ndarray] | None,
    n: int,
) -> dict[str, np.ndarray]:
    """Normalize ``ext_attributes`` to ``{name: (N,) float32}`` and validate shapes."""
    if ext_attributes is None:
        return {}
    out: dict[str, np.ndarray] = {}
    for name, arr in ext_attributes.items():
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        if arr.shape[0] != n:
            raise ValueError(f"ext attribute {name!r} has {arr.shape[0]} entries, expected {n}")
        out[name] = arr
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_gltf(
    gc: spz.GaussianCloud,
    path: str | Path,
    options: GltfSaveOptions | None = None,
    *,
    ext_attributes: dict[str, np.ndarray] | None = None,
) -> None:
    """Save a GaussianCloud to a KHR_gaussian_splatting compliant glTF/GLB file.

    ``ext_attributes`` is an optional ``{name: (N,) float32}`` mapping of
    per-Gaussian scalars (e.g. ``lidar_intensity_raw``, ``lidar_raydrop_logit``).
    Each is stored as a quantized accessor under the ``EXT_gaussian_lidar``
    extension on the primitive, kept out of the ``attributes`` dict to avoid
    GLSL-attribute-name issues in generic viewers.
    """
    path = Path(path)
    if options is None:
        options = GltfSaveOptions()

    ext_attrs = _validate_ext_attributes(ext_attributes, gc.num_points)

    if options.spz_compression:
        _save_gltf_spz(gc, path, options, ext_attrs)
    else:
        _save_gltf_standard(gc, path, options, ext_attrs)


def load_gltf(path: str | Path) -> spz.GaussianCloud:
    """Load a GaussianCloud from a KHR_gaussian_splatting compliant glTF/GLB file."""
    gc, _, _ = load_gltf_with_metadata(path)
    return gc


def load_gltf_with_metadata(
    path: str | Path,
) -> tuple[spz.GaussianCloud, GlbMetadata | dict[str, Any] | None, dict[str, np.ndarray]]:
    """Load a GaussianCloud, ``asset.extras`` metadata, and ``EXT_gaussian_lidar`` arrays.

    Returns:
        A 3-tuple ``(GaussianCloud, metadata, ext_attributes)``.

        * *metadata* is a :class:`GlbMetadata` when the extras match the
          schema, the raw dict for legacy files, or ``None`` if none.
        * *ext_attributes* is a ``{name: (N,) float32}`` dict — empty if
          the file carries no ``EXT_gaussian_lidar`` block.
    """
    path = Path(path)

    gltf_dict, buffer_data = _read_file(path)

    gc, ext_attrs = _parse_gaussian_cloud(gltf_dict, buffer_data)
    raw_metadata = gltf_dict.get("asset", {}).get("extras")
    return gc, parse_metadata(raw_metadata), ext_attrs


# ---------------------------------------------------------------------------
# Standard (per-attribute) save
# ---------------------------------------------------------------------------


def _save_gltf_standard(
    gc: spz.GaussianCloud,
    path: Path,
    options: GltfSaveOptions,
    ext_attributes: dict[str, np.ndarray],
) -> None:
    n = gc.num_points

    positions = np.array(gc.positions, dtype=np.float32).reshape(n, 3)
    rotations = np.array(gc.rotations, dtype=np.float32).reshape(n, 4)
    scales = np.array(gc.scales, dtype=np.float32).reshape(n, 3)
    # gc.colors = SH DC coefficients, gc.alphas = logit values
    sh_dc = np.array(gc.colors, dtype=np.float32).reshape(n, 3)
    alphas_logit = np.array(gc.alphas, dtype=np.float32)
    sh_raw = np.array(gc.sh, dtype=np.float32)

    # Post-activation values for glTF storage
    opacity_01 = _sigmoid(alphas_logit)
    opacity_u8 = np.clip(opacity_01 * 255 + 0.5, 0, 255).astype(np.uint8)

    rgb_01 = np.clip(sh_dc * _SH_C0 + 0.5, 0, 1)
    rgb_u8 = np.clip(rgb_01 * 255 + 0.5, 0, 255).astype(np.uint8)

    # COLOR_0 fallback (VEC4 uint8 normalized)
    color_0 = np.empty((n, 4), dtype=np.uint8)
    color_0[:, :3] = rgb_u8
    color_0[:, 3] = opacity_u8

    pos_min = positions.min(axis=0).tolist()
    pos_max = positions.max(axis=0).tolist()

    # All data arrays for the binary buffer, in accessor order.
    # Indices 0-1: standard glTF attributes (safe for any viewer).
    # Indices 2+: Gaussian data referenced only from the extension block,
    # keeping them out of the attributes dict to avoid GLSL variable name
    # issues in viewers that generate shader code from attribute names.
    data_list: list[tuple[bytes, int, str, dict]] = [
        (positions.tobytes(), _FLOAT, "VEC3", {"min": pos_min, "max": pos_max}),
        (color_0.tobytes(), _UNSIGNED_BYTE, "VEC4", {"normalized": True}),
        (rotations.tobytes(), _FLOAT, "VEC4", {}),
        (scales.tobytes(), _FLOAT, "VEC3", {}),
        (opacity_u8.tobytes(), _UNSIGNED_BYTE, "SCALAR", {"normalized": True}),
        (sh_dc.tobytes(), _FLOAT, "VEC3", {}),
    ]

    # Higher SH degrees
    sh_degree = _sh_degree_from_array(n, sh_raw)
    if sh_degree >= 1:
        sh_reshaped = sh_raw.reshape(n, -1, 3)
        coef_idx = 0
        for degree in range(1, sh_degree + 1):
            num_coefs = 2 * degree + 1
            for _j in range(num_coefs):
                data = (
                    np.ascontiguousarray(sh_reshaped[:, coef_idx, :]).astype(np.float32).tobytes()
                )
                data_list.append((data, _FLOAT, "VEC3", {}))
                coef_idx += 1

    # EXT_gaussian_lidar accessors (one SCALAR uint8 normalized accessor per attribute)
    ext_name_to_accessor: dict[str, int] = {}
    for name, arr in ext_attributes.items():
        spec = _spec_for(name)
        data = _quantize_ext(arr, spec)
        ext_name_to_accessor[name] = len(data_list)
        data_list.append((data, _UNSIGNED_BYTE, "SCALAR", {"normalized": True}))

    # Pack buffer
    buffer_data, offsets, lengths = _pack_buffer([d for d, _, _, _ in data_list])
    num_data = len(data_list)

    # Build accessors
    accessors = []
    for i, (_, comp_type, acc_type, extras) in enumerate(data_list):
        acc: dict = {
            "bufferView": i,
            "componentType": comp_type,
            "count": n,
            "type": acc_type,
        }
        acc.update(extras)
        accessors.append(acc)

    asset: dict[str, Any] = {"version": "2.0", "generator": "3dgs-io"}
    extras = serialize_metadata(options.metadata)
    if extras is not None:
        asset["extras"] = extras

    extensions_used = [_EXTENSION_NAME]
    primitive_extensions: dict[str, Any] = {
        _EXTENSION_NAME: {
            "kernel": "ellipse",
            "colorSpace": "srgb_rec709_display",
            "rotation": 2,
            "scale": 3,
            "opacity": 4,
            # 5..(5+sh_count-1) are SH; ext attrs come after.
            "sh": list(range(5, 5 + max(0, num_data - 5 - len(ext_name_to_accessor)))),
        }
    }
    if ext_name_to_accessor:
        primitive_extensions[EXT_GAUSSIAN_LIDAR_NAME] = dict(ext_name_to_accessor)
        extensions_used.append(EXT_GAUSSIAN_LIDAR_NAME)

    gltf_dict: dict = {
        "asset": asset,
        "extensionsUsed": extensions_used,
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {
                        "mode": 0,
                        "attributes": {
                            "POSITION": 0,
                            "COLOR_0": 1,
                        },
                        "extensions": primitive_extensions,
                    }
                ]
            }
        ],
        "accessors": accessors,
        "bufferViews": [
            {"buffer": 0, "byteOffset": offsets[i], "byteLength": lengths[i]}
            for i in range(num_data)
        ],
        "buffers": [{"byteLength": len(buffer_data)}],
    }

    _write_file(path, gltf_dict, buffer_data)


# ---------------------------------------------------------------------------
# SPZ-compressed save
# ---------------------------------------------------------------------------


def _save_gltf_spz(
    gc: spz.GaussianCloud,
    path: Path,
    options: GltfSaveOptions,
    ext_attributes: dict[str, np.ndarray],
) -> None:
    n = gc.num_points

    positions = np.array(gc.positions, dtype=np.float32).reshape(n, 3)
    pos_min = positions.min(axis=0).tolist()
    pos_max = positions.max(axis=0).tolist()

    sh_raw = np.array(gc.sh, dtype=np.float32)
    sh_degree = _sh_degree_from_array(n, sh_raw)

    spz_bytes = _compress_to_spz_bytes(gc)

    # The buffer is the SPZ blob optionally followed by EXT_gaussian_lidar
    # accessor payloads (4-byte aligned per bufferView).
    ext_chunks: list[tuple[str, bytes]] = []
    for name, arr in ext_attributes.items():
        ext_chunks.append((name, _quantize_ext(arr, _spec_for(name))))

    ext_bytes_packed, ext_offsets, ext_lengths = _pack_buffer([b for _, b in ext_chunks])

    # The SPZ blob lives at offset 0; ext payloads come after (also at
    # 4-byte alignment, since len(spz_bytes) may not be a multiple of 4).
    spz_padding = (4 - len(spz_bytes) % 4) % 4
    buffer_data = spz_bytes + b"\x00" * spz_padding + ext_bytes_packed
    ext_buffer_offset = len(spz_bytes) + spz_padding

    buffer_views: list[dict] = [
        {"buffer": 0, "byteLength": len(spz_bytes)},
    ]
    ext_name_to_accessor: dict[str, int] = {}

    # Virtual accessors (no bufferView – data comes from SPZ decompression)
    accessors: list[dict] = [
        {  # 0: POSITION
            "componentType": _FLOAT,
            "count": n,
            "type": "VEC3",
            "min": pos_min,
            "max": pos_max,
        },
        {  # 1: COLOR_0
            "componentType": _UNSIGNED_BYTE,
            "normalized": True,
            "count": n,
            "type": "VEC4",
        },
        {  # 2: SCALE
            "componentType": _FLOAT,
            "count": n,
            "type": "VEC3",
        },
        {  # 3: ROTATION
            "componentType": _FLOAT,
            "count": n,
            "type": "VEC4",
        },
    ]

    # SH coefficient virtual accessors
    sh_accessor_indices: list[int] = []
    acc_idx = 4
    for degree in range(1, sh_degree + 1):
        num_coefs = 2 * degree + 1
        for _j in range(num_coefs):
            accessors.append(
                {
                    "componentType": _FLOAT,
                    "count": n,
                    "type": "VEC3",
                }
            )
            sh_accessor_indices.append(acc_idx)
            acc_idx += 1

    # EXT_gaussian_lidar accessors: each backed by a real bufferView pointing
    # into the packed ext block. These are concrete (non-virtual) because
    # SPZ decompression does not produce them.
    for idx, (name, _) in enumerate(ext_chunks):
        bv_idx = len(buffer_views)
        buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": ext_buffer_offset + ext_offsets[idx],
                "byteLength": ext_lengths[idx],
            }
        )
        accessors.append(
            {
                "bufferView": bv_idx,
                "componentType": _UNSIGNED_BYTE,
                "count": n,
                "type": "SCALAR",
                "normalized": True,
            }
        )
        ext_name_to_accessor[name] = acc_idx
        acc_idx += 1

    # CesiumJS's loadPrimitive() only iterates gltfPrimitive.attributes to
    # create vertex buffer loaders.  For SPZ GLBs every Gaussian attribute
    # must appear in the attributes dict with underscore-prefixed semantics
    # so that CesiumJS's processSpz() can map decoded SPZ data to each one.
    # (Non-SPZ GLBs keep these out of the attributes dict to avoid GLSL
    # variable-name issues in viewers that derive shader code from attribute
    # names; SPZ GLBs don't hit that path because the data comes from the
    # SPZ decoder, not from per-attribute buffer views.)
    # Use colon-prefixed attribute names matching the KHR_gaussian_splatting
    # spec.  CesiumJS's processSpz() accepts both "_SCALE" and
    # "KHR_gaussian_splatting:SCALE"; the colon form is canonical.
    _EXT_PFX = f"{_EXTENSION_NAME}:"
    attributes: dict[str, int] = {
        "POSITION": 0,
        "COLOR_0": 1,
        f"{_EXT_PFX}SCALE": 2,
        f"{_EXT_PFX}ROTATION": 3,
    }
    coef_acc_idx = 4
    for degree in range(1, sh_degree + 1):
        num_coefs = 2 * degree + 1
        for j in range(num_coefs):
            attributes[f"{_EXT_PFX}SH_DEGREE_{degree}_COEF_{j}"] = coef_acc_idx
            coef_acc_idx += 1

    # SPZ extension block: only the SPZ sub-extension reference.
    # Do NOT include "scale", "rotation", or "sh" accessor indices here —
    # CesiumJS ignores them for SPZ and extra keys may confuse the parser.
    gs_ext: dict[str, Any] = {
        "kernel": "ellipse",
        "projection": "perspective",
        "extensions": {
            _SPZ_EXTENSION_NAME: {
                "bufferView": 0,
            }
        },
    }

    asset: dict[str, Any] = {"version": "2.0", "generator": "3dgs-io"}
    extras = serialize_metadata(options.metadata)
    if extras is not None:
        asset["extras"] = extras

    extensions_used = [_EXTENSION_NAME, _SPZ_EXTENSION_NAME]
    extensions_required = [_EXTENSION_NAME, _SPZ_EXTENSION_NAME]
    primitive_extensions: dict[str, Any] = {_EXTENSION_NAME: gs_ext}
    if ext_name_to_accessor:
        primitive_extensions[EXT_GAUSSIAN_LIDAR_NAME] = dict(ext_name_to_accessor)
        extensions_used.append(EXT_GAUSSIAN_LIDAR_NAME)

    gltf_dict: dict = {
        "asset": asset,
        "extensionsUsed": extensions_used,
        "extensionsRequired": extensions_required,
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {
                        "mode": 0,
                        "attributes": attributes,
                        "extensions": primitive_extensions,
                    }
                ]
            }
        ],
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(buffer_data)}],
    }

    _write_file(path, gltf_dict, buffer_data)


# ---------------------------------------------------------------------------
# SPZ compression helpers
# ---------------------------------------------------------------------------


def _compress_to_spz_bytes(gc: spz.GaussianCloud) -> bytes:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "temp.spz"
        opts = spz.PackOptions()
        # Do NOT set from_coord — CesiumJS's @spz-loader/core WASM
        # decoder outputs raw positions without any coordinate conversion,
        # so positions must be stored as-is in the input coordinate space.
        spz.save_spz(gc, opts, str(path))
        # Keep gzip wrapper — CesiumJS's @spz-loader/core expects
        # gzip-compressed SPZ data, not raw NGSP bytes.
        return path.read_bytes()


def _decompress_from_spz_bytes(data: bytes) -> spz.GaussianCloud:
    # spz.load_spz() expects gzip-wrapped data (.spz file format).
    # The KHR_gaussian_splatting_compression_spz_2 spec stores raw SPZ bytes,
    # but legacy files may contain gzip-wrapped SPZ bytes.  Detect and handle both.
    _GZIP_MAGIC = b"\x1f\x8b"
    if not data[:2] == _GZIP_MAGIC:
        data = gzip.compress(data)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "temp.spz"
        path.write_bytes(data)
        opts = spz.UnpackOptions()
        # Do NOT set to_coord — must match _compress_to_spz_bytes() which
        # stores positions without coordinate conversion.
        return spz.load_spz(str(path), opts)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _sh_degree_from_array(num_points: int, sh: np.ndarray) -> int:
    if len(sh) == 0:
        return 0
    coefficients = len(sh) // (num_points * 3)
    if coefficients >= 15:
        return 3
    if coefficients >= 8:
        return 2
    if coefficients >= 3:
        return 1
    return 0


def _parse_ext_attributes(
    primitive: dict,
    gltf_dict: dict,
    buffer_data: bytes,
) -> dict[str, np.ndarray]:
    """Parse the ``EXT_gaussian_lidar`` block, returning ``{name: float32 (N,)}``."""
    ext_block = primitive.get("extensions", {}).get(EXT_GAUSSIAN_LIDAR_NAME)
    if not ext_block:
        return {}

    accessors = gltf_dict["accessors"]
    buffer_views = gltf_dict["bufferViews"]
    out: dict[str, np.ndarray] = {}
    for name, acc_idx in ext_block.items():
        raw = _read_accessor(accessors[acc_idx], buffer_views, buffer_data)
        out[name] = _dequantize_ext(raw, _spec_for(name))
    return out


def _parse_gaussian_cloud(
    gltf_dict: dict, buffer_data: bytes
) -> tuple[spz.GaussianCloud, dict[str, np.ndarray]]:
    primitive = _find_gaussian_primitive(gltf_dict)
    if primitive is None:
        raise ValueError("No KHR_gaussian_splatting primitive found in glTF")

    ext_attrs = _parse_ext_attributes(primitive, gltf_dict, buffer_data)

    # Check for SPZ compression sub-extension
    ext = primitive.get("extensions", {}).get(_EXTENSION_NAME, {})
    spz_ext = ext.get("extensions", {}).get(_SPZ_EXTENSION_NAME)

    if spz_ext is not None:
        bv_idx = spz_ext["bufferView"]
        bv = gltf_dict["bufferViews"][bv_idx]
        offset = bv.get("byteOffset", 0)
        length = bv["byteLength"]
        spz_data = buffer_data[offset : offset + length]
        return _decompress_from_spz_bytes(spz_data), ext_attrs

    return _parse_standard(primitive, gltf_dict, buffer_data), ext_attrs


def _parse_standard(
    primitive: dict,
    gltf_dict: dict,
    buffer_data: bytes,
) -> spz.GaussianCloud:
    attrs = primitive["attributes"]
    accessors = gltf_dict["accessors"]
    buffer_views = gltf_dict["bufferViews"]

    # Extension block (new format stores accessor indices here)
    gs_ext = primitive.get("extensions", {}).get(_EXTENSION_NAME, {})

    # POSITION
    positions = _read_accessor(accessors[attrs["POSITION"]], buffer_views, buffer_data)

    # ROTATION
    rot_idx = gs_ext.get("rotation")
    if rot_idx is None:
        raise ValueError("No rotation attribute found in extension block")
    rotations = _read_accessor(accessors[rot_idx], buffer_views, buffer_data)

    # SCALE
    scl_idx = gs_ext.get("scale")
    if scl_idx is None:
        raise ValueError("No scale attribute found in extension block")
    scales = _read_accessor(accessors[scl_idx], buffer_views, buffer_data)

    # OPACITY → logit
    opa_idx = gs_ext.get("opacity")
    if opa_idx is not None:
        raw = _read_accessor(accessors[opa_idx], buffer_views, buffer_data)
        if raw.dtype == np.float32:
            alphas = _inverse_sigmoid(raw)
        else:
            alphas = _inverse_sigmoid(raw.astype(np.float32) / 255.0)
    elif "COLOR_0" in attrs:
        color_0 = _read_accessor(accessors[attrs["COLOR_0"]], buffer_views, buffer_data)
        if color_0.dtype == np.float32:
            alphas = _inverse_sigmoid(color_0[:, 3])
        else:
            alphas = _inverse_sigmoid(color_0[:, 3].astype(np.float32) / 255.0)
    else:
        raise ValueError("No opacity data found")

    # COLORS (SH DC) and higher SH coefficients
    sh_indices = gs_ext.get("sh")
    if sh_indices is not None and len(sh_indices) > 0:
        # sh[0] is DC, sh[1:] are higher degree coefficients
        colors_sh = _read_accessor(accessors[sh_indices[0]], buffer_views, buffer_data).astype(
            np.float32
        )
        sh_coefficients: list[np.ndarray] = []
        for idx in sh_indices[1:]:
            coef = _read_accessor(accessors[idx], buffer_views, buffer_data)
            sh_coefficients.append(coef.astype(np.float32))
    elif "COLOR_0" in attrs:
        raw = _read_accessor(accessors[attrs["COLOR_0"]], buffer_views, buffer_data)
        if raw.dtype == np.float32:
            rgb_01 = raw[:, :3]
        else:
            rgb_01 = raw[:, :3].astype(np.float32) / 255.0
        colors_sh = (rgb_01 - 0.5) / _SH_C0
        sh_coefficients = []
    else:
        raise ValueError("No color data found")

    # Build GaussianCloud
    gc = spz.GaussianCloud()
    gc.positions = positions.astype(np.float32).reshape(-1)
    gc.colors = colors_sh.reshape(-1).astype(np.float32)
    gc.alphas = alphas.astype(np.float32)
    gc.rotations = rotations.astype(np.float32).reshape(-1)
    gc.scales = scales.astype(np.float32).reshape(-1)

    if sh_coefficients:
        # Infer SH degree from coefficient count and set BEFORE assigning sh
        n_coefs = len(sh_coefficients)
        if n_coefs >= 15:
            gc.sh_degree = 3
        elif n_coefs >= 8:
            gc.sh_degree = 2
        elif n_coefs >= 3:
            gc.sh_degree = 1
        sh_stacked = np.stack(sh_coefficients, axis=1)  # (N, num_coef, 3)
        gc.sh = sh_stacked.reshape(-1).astype(np.float32)
    else:
        gc.sh = np.zeros(0, dtype=np.float32)

    return gc


def _find_gaussian_primitive(gltf_dict: dict) -> dict | None:
    for mesh in gltf_dict.get("meshes", []):
        for prim in mesh.get("primitives", []):
            exts = prim.get("extensions", {})
            if _EXTENSION_NAME in exts:
                return prim
    return None
