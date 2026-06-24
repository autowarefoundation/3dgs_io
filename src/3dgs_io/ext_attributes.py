"""Per-Gaussian extension attributes (``EXT_gaussian_lidar``).

Issue #26 needs to carry optional per-Gaussian scalars — currently
``lidar_intensity_raw`` and ``lidar_raydrop_logit`` for LiDAR simulation —
alongside each splat without touching the fixed-schema ``spz.GaussianCloud``
or the SPZ on-disk format. The data rides as parallel ``(N,)`` arrays
threaded through the same masks and reorderings as the gaussians, so
``attr[i] ↔ gaussian[i]`` always holds within a tile/chunk.

This module defines:

* :data:`EXT_GAUSSIAN_LIDAR_NAME` — the glTF / tileset extension key
  (``"EXT_gaussian_lidar"``).
* :class:`ExtAttributeSpec` — per-attribute quantization metadata.
* :func:`encode_lidar_sidecar` / :func:`decode_lidar_sidecar` — the binary
  sidecar format written next to each ``chunks/chunk_NNNNNN.spz`` in the
  final USDZ.

Sidecar binary layout
---------------------

``chunks/chunk_NNNNNN.lidar`` is a small, self-describing binary file::

    bytes  0..3   magic       ``"L1DR"``
    bytes  4..7   version     uint32 little-endian, currently ``1``
    bytes  8..11  count       uint32 little-endian, ``N`` points
    bytes 12..15  channels    uint32 little-endian, channel count ``C``
    bytes 16..N   body        ``count * channels`` bytes, interleaved per point

The current encoder writes ``C = 2`` channels in fixed order:

* channel 0: ``sigmoid(lidar_intensity_raw) * 255``  — ``uint8``
* channel 1: ``sigmoid(lidar_raydrop_logit) * 255``  — ``uint8``

Both attributes survive the pipeline as their original float32 values
inside ``EXT_gaussian_lidar`` glTF accessors; quantization is only applied
on the final write-out to the per-chunk sidecar.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

EXT_GAUSSIAN_LIDAR_NAME = "EXT_gaussian_lidar"

LIDAR_INTENSITY_KEY = "lidar_intensity_raw"
LIDAR_RAYDROP_KEY = "lidar_raydrop_logit"

_LIDAR_SIDECAR_MAGIC = b"L1DR"
_LIDAR_SIDECAR_VERSION = 1
_LIDAR_SIDECAR_HEADER_FMT = "<4sIII"  # magic, version, count, channels
_LIDAR_SIDECAR_HEADER_SIZE = struct.calcsize(_LIDAR_SIDECAR_HEADER_FMT)
LIDAR_SIDECAR_SUFFIX = ".lidar"


@dataclass(frozen=True)
class ExtAttributeSpec:
    """Metadata for a single per-Gaussian extension attribute.

    Parameters
    ----------
    name:
        Attribute key, e.g. ``"lidar_intensity_raw"``.
    quantization:
        Quantization mode applied when writing the per-chunk sidecar:

        * ``"u8_sigmoid"`` — apply ``sigmoid`` then scale to ``uint8 [0..255]``.
          Inverse is ``logit(x/255)`` after divide.
        * ``"u8_linear"`` — clamp to ``[vmin, vmax]``, scale to
          ``uint8 [0..255]``.
        * ``"f32"`` — no quantization (debug; not used in default sidecar).
    vmin, vmax:
        Range for ``"u8_linear"``; ignored otherwise.
    """

    name: str
    quantization: str = "u8_sigmoid"
    vmin: float = 0.0
    vmax: float = 1.0


DEFAULT_LIDAR_SPECS: tuple[ExtAttributeSpec, ...] = (
    ExtAttributeSpec(name=LIDAR_INTENSITY_KEY, quantization="u8_sigmoid"),
    ExtAttributeSpec(name=LIDAR_RAYDROP_KEY, quantization="u8_sigmoid"),
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


def _logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float64), 1e-9, 1.0 - 1e-9)
    return np.log(x / (1.0 - x))


def encode_lidar_sidecar(
    ext_attributes: dict[str, np.ndarray],
    *,
    count: int,
) -> bytes:
    """Encode the default 2-channel LiDAR sidecar for ``count`` points.

    Channels are written in fixed order: ``lidar_intensity_raw`` then
    ``lidar_raydrop_logit``. Missing channels are treated as zeros
    (``sigmoid(0) = 0.5`` ⇒ ``128`` after quantization), so that the
    sidecar can still be written if only one of the two attributes is
    available — downstream readers can detect this via the all-128
    pattern, but for v1 we require both.
    """
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")

    body = np.zeros((count, 2), dtype=np.uint8)
    for ch_idx, spec in enumerate(DEFAULT_LIDAR_SPECS):
        arr = ext_attributes.get(spec.name)
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        if arr.shape[0] != count:
            raise ValueError(
                f"ext attribute {spec.name!r} has {arr.shape[0]} entries, expected {count}"
            )
        if spec.quantization == "u8_sigmoid":
            q = np.clip(np.round(_sigmoid(arr) * 255.0), 0.0, 255.0).astype(np.uint8)
        elif spec.quantization == "u8_linear":
            scaled = (arr.astype(np.float64) - spec.vmin) / max(spec.vmax - spec.vmin, 1e-12)
            q = np.clip(np.round(scaled * 255.0), 0.0, 255.0).astype(np.uint8)
        else:
            raise ValueError(f"unsupported quantization {spec.quantization!r}")
        body[:, ch_idx] = q

    header = struct.pack(
        _LIDAR_SIDECAR_HEADER_FMT,
        _LIDAR_SIDECAR_MAGIC,
        _LIDAR_SIDECAR_VERSION,
        count,
        len(DEFAULT_LIDAR_SPECS),
    )
    return header + body.tobytes()


def decode_lidar_sidecar(data: bytes) -> dict[str, np.ndarray]:
    """Decode a LiDAR sidecar back to a dict of ``{name: float32 (N,)}``.

    Quantization is undone by the inverse of the encoder: ``sigmoid``-quantized
    values are returned as their *pre-sigmoid* logits (so the round trip
    preserves the original semantic field, modulo quantization error).
    """
    if len(data) < _LIDAR_SIDECAR_HEADER_SIZE:
        raise ValueError(f"sidecar too short ({len(data)} bytes)")

    magic, version, count, channels = struct.unpack(
        _LIDAR_SIDECAR_HEADER_FMT, data[:_LIDAR_SIDECAR_HEADER_SIZE]
    )
    if magic != _LIDAR_SIDECAR_MAGIC:
        raise ValueError(f"bad sidecar magic {magic!r}")
    if version != _LIDAR_SIDECAR_VERSION:
        raise ValueError(f"unsupported sidecar version {version}")
    expected_body = count * channels
    body = data[_LIDAR_SIDECAR_HEADER_SIZE:]
    if len(body) != expected_body:
        raise ValueError(
            f"sidecar body size mismatch: header says {expected_body} bytes, got {len(body)}"
        )

    arr = np.frombuffer(body, dtype=np.uint8).reshape(count, channels)
    out: dict[str, np.ndarray] = {}
    for ch_idx in range(min(channels, len(DEFAULT_LIDAR_SPECS))):
        spec = DEFAULT_LIDAR_SPECS[ch_idx]
        q = arr[:, ch_idx].astype(np.float64) / 255.0
        if spec.quantization == "u8_sigmoid":
            out[spec.name] = _logit(q).astype(np.float32)
        elif spec.quantization == "u8_linear":
            out[spec.name] = (q * (spec.vmax - spec.vmin) + spec.vmin).astype(np.float32)
        else:
            raise ValueError(f"unsupported quantization {spec.quantization!r}")
    return out
