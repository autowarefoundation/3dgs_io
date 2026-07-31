"""Per-Gaussian extension attributes (``EXT_gaussian_lidar``).

Issue #26 needs to carry optional per-Gaussian scalars — currently
``lidar_intensity_raw``, ``lidar_raydrop_logit`` and ``lidar_mask`` for LiDAR
simulation — alongside each splat without touching the fixed-schema ``spz.GaussianCloud``
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
    bytes  4..7   version     uint32 little-endian, ``1`` (scalar) or ``2`` (+SH)
    bytes  8..11  count       uint32 little-endian, ``N`` points
    bytes 12..15  channels    uint32 little-endian, channel count ``C``
    bytes 16..M   body        ``count * channels`` bytes, interleaved per point
    bytes  M..    sh_block    version-2 only (see below)

The encoder writes channels in a fixed, append-only order:

* channel 0: ``sigmoid(lidar_intensity_raw) * 255``  — ``uint8``
* channel 1: ``sigmoid(lidar_raydrop_logit) * 255``  — ``uint8``
* channel 2: ``lidar_mask``                          — ``uint8`` (optional)

``lidar_mask`` is an **optional, append-only** third channel:

* ``lidar_mask[i] == 1`` → the Gaussian participates in LiDAR simulation
  (near-field, geometrically faithful).
* ``lidar_mask[i] == 0`` → the Gaussian is appearance-only (far-field, tuned
  for RGB); consumers should hard-exclude it from the LiDAR geometry pass.

It is quantized with ``u8_linear`` over ``[0, 1]``, which round-trips the
values ``{0, 1}`` exactly (``0 → 0``, ``1 → 255`` on encode; ``0 → 0.0``,
``255 → 1.0`` on decode); consumers threshold decoded values at ``0.5``.

The channel is fully backward/forward compatible and needs **no** header
``version`` bump: because :func:`decode_lidar_sidecar` reads ``channels`` from
the header and clamps its decode loop to ``min(channels, len(DEFAULT_LIDAR_SPECS))``,

* an old 2-channel reader reading a new 3-channel sidecar silently ignores the
  mask channel, and
* a new 3-channel reader reading an old 2-channel sidecar simply omits the
  ``lidar_mask`` key (consumers treat its absence as "all participate").

When a caller does not supply a mask, the encoder writes only the two required
channels, so its output is byte-identical to the previous 2-channel format.

Both scalar attributes survive the pipeline as their original float32 values
inside ``EXT_gaussian_lidar`` glTF accessors; quantization is only applied
on the final write-out to the per-chunk sidecar.

View-dependent raydrop (SH) trailing block — version 2
------------------------------------------------------

LiDAR raydrop is view/ray-dependent, so a single per-Gaussian scalar cannot
express a Gaussian that must *drop* for one sensor but *return* for another.
Version 2 appends per-Gaussian spherical-harmonics raydrop coefficients after
the fixed uint8 body — the DC (band-0) term stays in the scalar
``lidar_raydrop_logit`` channel, and the trailing block carries only the
*higher-order* bands ``raydrop_sh`` so the renderer can evaluate drop
probability at the sensor ray direction (exactly like colour SH)::

    sh_header  8 bytes   sh_degree uint32 LE, sh_coefs uint32 LE (= (deg+1)^2 - 1)
    sh_body    count * sh_coefs * 2 bytes, ``float16`` little-endian, row-major

The block is written **only** when a caller supplies ``raydrop_sh`` with a
positive degree; otherwise the encoder emits a version-1 sidecar that is
byte-identical to the previous format. A version-1 reader that encounters a
version-2 sidecar raises ``unsupported sidecar version`` — consumers unable to
evaluate view-dependent raydrop cannot use the data anyway, so the version gate
fails loudly rather than silently dropping the SH bands.

``float16`` (half precision) is used because the higher-order bands are small
logit-space deltas whose view-dependence is the whole point of the feature;
``uint8`` quantisation would lose too much of it.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numpy as np

EXT_GAUSSIAN_LIDAR_NAME = "EXT_gaussian_lidar"

LIDAR_INTENSITY_KEY = "lidar_intensity_raw"
LIDAR_RAYDROP_KEY = "lidar_raydrop_logit"
LIDAR_MASK_KEY = "lidar_mask"
# View-dependent (spherical-harmonics) raydrop: the *higher-order* SH bands
# ``(N, (deg+1)**2 - 1)``. The DC (band-0) term is the existing scalar
# ``lidar_raydrop_logit``; ``raydrop_sh`` carries bands ``1..deg`` only.
RAYDROP_SH_KEY = "raydrop_sh"

_LIDAR_SIDECAR_MAGIC = b"L1DR"
# Version 1: scalar uint8 channels only. Version 2: identical uint8 channels
# followed by a self-describing float16 ``raydrop_sh`` trailing block.
_LIDAR_SIDECAR_VERSION = 1
_LIDAR_SIDECAR_VERSION_SH = 2
_LIDAR_SIDECAR_HEADER_FMT = "<4sIII"  # magic, version, count, channels
_LIDAR_SIDECAR_HEADER_SIZE = struct.calcsize(_LIDAR_SIDECAR_HEADER_FMT)
_LIDAR_SIDECAR_SH_HEADER_FMT = "<II"  # raydrop_sh_degree, raydrop_sh_coefs (=(deg+1)^2-1)
_LIDAR_SIDECAR_SH_HEADER_SIZE = struct.calcsize(_LIDAR_SIDECAR_SH_HEADER_FMT)
LIDAR_SIDECAR_SUFFIX = ".lidar"


def raydrop_sh_coefs(degree: int) -> int:
    """Number of *higher-order* SH raydrop coefficients for ``degree``.

    Excludes the DC (band-0) term, which is carried by the scalar
    ``lidar_raydrop_logit`` channel: ``(degree + 1)**2 - 1``. ``degree == 0``
    ⇒ ``0`` (scalar-only, no ``raydrop_sh`` block).
    """
    if degree < 0:
        raise ValueError(f"raydrop_sh degree must be non-negative, got {degree}")
    return (degree + 1) ** 2 - 1


def raydrop_sh_degree_from_coefs(coefs: int) -> int:
    """Inverse of :func:`raydrop_sh_coefs`.

    Raises ``ValueError`` unless ``coefs == (deg + 1)**2 - 1`` for some
    non-negative integer ``deg``.
    """
    if coefs < 0:
        raise ValueError(f"raydrop_sh coefs must be non-negative, got {coefs}")
    total = coefs + 1
    root = math.isqrt(total)
    if root * root != total:
        raise ValueError(f"raydrop_sh coefs {coefs} is not (deg+1)^2 - 1 for any integer degree")
    return root - 1


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
    # Append-only optional channel. ``u8_linear`` over [0, 1] round-trips the
    # boolean values {0, 1} exactly. Omitted by the encoder when no mask is
    # supplied, keeping the 2-channel output byte-identical to before.
    ExtAttributeSpec(name=LIDAR_MASK_KEY, quantization="u8_linear", vmin=0.0, vmax=1.0),
)

# Channels that must always be present in a sidecar. Any spec in
# ``DEFAULT_LIDAR_SPECS`` not listed here is optional (append-only) and is
# written only when the caller supplies it.
_REQUIRED_LIDAR_KEYS: frozenset[str] = frozenset({LIDAR_INTENSITY_KEY, LIDAR_RAYDROP_KEY})


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
    """Encode the LiDAR sidecar for ``count`` points.

    Channels are written in the fixed, append-only order defined by
    :data:`DEFAULT_LIDAR_SPECS`: ``lidar_intensity_raw``, ``lidar_raydrop_logit``
    then the optional ``lidar_mask``.

    The two scalar channels (``lidar_intensity_raw`` / ``lidar_raydrop_logit``)
    are required — passing a dict that is missing one of them raises
    ``KeyError``. ``lidar_mask`` is **optional**: when it is omitted the encoder
    writes only the two required channels, producing a byte-identical result to
    the previous 2-channel format. When supplied, a third ``u8_linear`` channel
    is appended (``1`` = participates in LiDAR simulation, ``0`` = appearance
    only); its absence is interpreted downstream as "all participate", so a
    caller wanting that behaviour explicitly may pass an all-ones array.

    View-dependent raydrop is carried by the :data:`RAYDROP_SH_KEY`
    (``raydrop_sh``) entry of ``ext_attributes``: a ``(count, coefs)`` array of
    the *higher-order* SH bands where ``coefs == (deg+1)**2 - 1`` (the DC term
    stays in the ``lidar_raydrop_logit`` channel). When present with a positive
    degree the encoder emits a version-2 sidecar with a trailing ``float16`` SH
    block; otherwise the output is a byte-identical version-1 sidecar.
    """
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")

    # ``raydrop_sh`` is a 2-D array, not a scalar channel, so it rides in the
    # same dict but never participates in the uint8 body below.
    scalars = dict(ext_attributes)
    raydrop_sh = scalars.pop(RAYDROP_SH_KEY, None)

    # Select the channels to write, in spec order. Required channels must be
    # present; optional (append-only) channels are written only when supplied.
    active_specs: list[ExtAttributeSpec] = []
    for spec in DEFAULT_LIDAR_SPECS:
        if spec.name in scalars:
            active_specs.append(spec)
        elif spec.name in _REQUIRED_LIDAR_KEYS:
            raise KeyError(f"ext attribute {spec.name!r} is required for the LiDAR sidecar")
        # else: optional channel not supplied -> drop the trailing channel.

    channels = len(active_specs)
    body = np.zeros((count, channels), dtype=np.uint8)
    for ch_idx, spec in enumerate(active_specs):
        arr = np.asarray(scalars[spec.name], dtype=np.float32).reshape(-1)
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

    version = _LIDAR_SIDECAR_VERSION
    sh_block = b""
    if raydrop_sh is not None:
        sh = np.asarray(raydrop_sh)
        if sh.ndim != 2 or sh.shape[0] != count:
            raise ValueError(f"raydrop_sh must have shape (count={count}, coefs), got {sh.shape}")
        coefs = int(sh.shape[1])
        degree = raydrop_sh_degree_from_coefs(coefs)
        if degree <= 0:
            raise ValueError(
                "raydrop_sh must carry at least the degree-1 bands (coefs >= 3); "
                f"got coefs={coefs}. Use the scalar lidar_raydrop_logit channel for degree 0."
            )
        version = _LIDAR_SIDECAR_VERSION_SH
        sh_header = struct.pack(_LIDAR_SIDECAR_SH_HEADER_FMT, degree, coefs)
        # A single float16 copy; ``tobytes`` always emits C-order regardless of
        # the source layout, so no separate contiguity pass is needed.
        sh_block = sh_header + np.asarray(sh, dtype=np.float16).tobytes()

    header = struct.pack(
        _LIDAR_SIDECAR_HEADER_FMT,
        _LIDAR_SIDECAR_MAGIC,
        version,
        count,
        channels,
    )
    return header + body.tobytes() + sh_block


def decode_lidar_sidecar(data: bytes) -> dict[str, np.ndarray]:
    """Decode a LiDAR sidecar back to a dict of ``{name: float32 (N,)}``.

    Quantization is undone by the inverse of the encoder: ``sigmoid``-quantized
    values are returned as their *pre-sigmoid* logits (so the round trip
    preserves the original semantic field, modulo quantization error).

    The number of returned keys matches the sidecar's channel count: a
    2-channel sidecar yields ``{lidar_intensity_raw, lidar_raydrop_logit}`` and
    omits ``lidar_mask`` (consumers treat its absence as "all participate"),
    while a 3-channel sidecar additionally returns ``lidar_mask`` as ``{0.0,
    1.0}`` floats (threshold at ``0.5``).

    A version-2 sidecar additionally returns ``raydrop_sh`` as a
    ``(count, coefs)`` float32 array of the higher-order SH raydrop bands (the
    degree is recoverable via :func:`raydrop_sh_degree_from_coefs`); the DC term
    remains in ``lidar_raydrop_logit``.
    """
    if len(data) < _LIDAR_SIDECAR_HEADER_SIZE:
        raise ValueError(f"sidecar too short ({len(data)} bytes)")

    magic, version, count, channels = struct.unpack(
        _LIDAR_SIDECAR_HEADER_FMT, data[:_LIDAR_SIDECAR_HEADER_SIZE]
    )
    if magic != _LIDAR_SIDECAR_MAGIC:
        raise ValueError(f"bad sidecar magic {magic!r}")
    if version not in (_LIDAR_SIDECAR_VERSION, _LIDAR_SIDECAR_VERSION_SH):
        raise ValueError(f"unsupported sidecar version {version}")

    expected_body = count * channels
    offset = _LIDAR_SIDECAR_HEADER_SIZE
    body = data[offset : offset + expected_body]
    if len(body) != expected_body:
        raise ValueError(
            f"sidecar body size mismatch: header says {expected_body} bytes, got {len(body)}"
        )
    offset += expected_body

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

    if version == _LIDAR_SIDECAR_VERSION_SH:
        if len(data) - offset < _LIDAR_SIDECAR_SH_HEADER_SIZE:
            raise ValueError("sidecar version 2 is missing its raydrop_sh header")
        degree, coefs = struct.unpack(
            _LIDAR_SIDECAR_SH_HEADER_FMT, data[offset : offset + _LIDAR_SIDECAR_SH_HEADER_SIZE]
        )
        offset += _LIDAR_SIDECAR_SH_HEADER_SIZE
        if raydrop_sh_coefs(degree) != coefs:
            raise ValueError(
                f"raydrop_sh header inconsistent: degree={degree} implies "
                f"{raydrop_sh_coefs(degree)} coefs, header says {coefs}"
            )
        expected_sh = count * coefs * 2  # float16
        sh_body = data[offset:]
        if len(sh_body) != expected_sh:
            raise ValueError(
                f"raydrop_sh body size mismatch: expected {expected_sh} bytes, got {len(sh_body)}"
            )
        out[RAYDROP_SH_KEY] = (
            np.frombuffer(sh_body, dtype=np.float16).reshape(count, coefs).astype(np.float32)
        )
    elif offset != len(data):
        raise ValueError(
            f"sidecar body size mismatch: header says {expected_body} bytes, "
            f"got {len(data) - _LIDAR_SIDECAR_HEADER_SIZE}"
        )

    return out
