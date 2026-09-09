from __future__ import annotations

import struct
import tempfile
from pathlib import Path

import spz

# NGSP v4 (SPZ) fixed 32-byte file header:
#   u32 magic, u32 version, u32 numPoints, u8 shDegree, u8 fractionalBits,
#   u8 flags, u8 numStreams, u32 tocByteOffset, 12 reserved bytes.
# Extension records live in the plaintext zone [32, tocByteOffset), each as
# [u32 type][u32 byteLength][payload...]. Readers without extension support
# skip the zone (the TOC is located via tocByteOffset), so embedding records
# never breaks core Gaussian loading.
_NGSP_HEADER_FMT = "<IIIBBBBI12x"
_NGSP_HEADER_SIZE = struct.calcsize(_NGSP_HEADER_FMT)
NGSP_MAGIC = b"NGSP"
_NGSP_MAGIC_U32 = int.from_bytes(NGSP_MAGIC, "little")
_NGSP_MIN_EXTENSION_VERSION = 4  # plaintext extension zone exists from v4 on
_NGSP_FLAG_HAS_EXTENSIONS = 0x2

# Unpacked header fields, in _NGSP_HEADER_FMT order.
_NgspHeader = tuple[int, int, int, int, int, int, int, int]


def is_ngsp_stream(data: bytes) -> bool:
    """True when ``data`` starts with the NGSP (SPZ v4+, zstd) magic."""
    return data[:4] == NGSP_MAGIC


def _parse_ngsp_header(data: bytes) -> _NgspHeader:
    """Validate an NGSP v4+ header and return its unpacked fields.

    Field order matches ``_NGSP_HEADER_FMT``: magic, version, num_points,
    sh_degree, fractional_bits, flags, num_streams, toc_byte_offset.
    """
    if len(data) < _NGSP_HEADER_SIZE:
        raise ValueError(f"SPZ data too short for NGSP header ({len(data)} bytes)")
    fields: _NgspHeader = struct.unpack_from(_NGSP_HEADER_FMT, data)
    magic, version = fields[0], fields[1]
    toc = fields[7]
    if magic != _NGSP_MAGIC_U32:
        raise ValueError(f"not an NGSP SPZ stream (magic 0x{magic:08X})")
    if version < _NGSP_MIN_EXTENSION_VERSION:
        raise ValueError(
            f"SPZ version {version} predates the extension zone (needs >= "
            f"{_NGSP_MIN_EXTENSION_VERSION})"
        )
    if not _NGSP_HEADER_SIZE <= toc <= len(data):
        raise ValueError(f"corrupt SPZ: tocByteOffset {toc} outside file of {len(data)} bytes")
    return fields


def append_spz_extension(data: bytes, ext_type: int, payload: bytes) -> bytes:
    """Return ``data`` with one extension record appended to its extension zone.

    ``ext_type`` is the 32-bit SPZ extension type (vendor ID in the high 16
    bits, extension ID in the low 16). The record is inserted at the end of
    the plaintext extension zone and the header's ``tocByteOffset`` /
    has-extensions flag are updated accordingly.
    """
    if not 0 < ext_type <= 0xFFFFFFFF:
        raise ValueError(f"ext_type must be a non-zero u32, got {ext_type:#x}")
    magic, version, npts, shd, fb, flags, nstreams, toc = _parse_ngsp_header(data)
    record = struct.pack("<II", ext_type, len(payload)) + payload
    header = struct.pack(
        _NGSP_HEADER_FMT,
        magic,
        version,
        npts,
        shd,
        fb,
        flags | _NGSP_FLAG_HAS_EXTENSIONS,
        nstreams,
        toc + len(record),
    )
    view = memoryview(data)
    return b"".join((header, view[_NGSP_HEADER_SIZE:toc], record, view[toc:]))


def read_spz_extensions(data: bytes) -> dict[int, bytes]:
    """Parse the extension zone of an NGSP v4+ stream into ``{type: payload}``.

    Returns an empty dict when the file carries no extensions. Duplicate
    types keep the last record, mirroring spz's own last-wins parsing.
    """
    *_, flags, _nstreams, toc = _parse_ngsp_header(data)
    out: dict[int, bytes] = {}
    if not flags & _NGSP_FLAG_HAS_EXTENSIONS:
        return out
    off = _NGSP_HEADER_SIZE
    while off + 8 <= toc:
        ext_type, length = struct.unpack_from("<II", data, off)
        off += 8
        if off + length > toc:
            raise ValueError(f"corrupt SPZ extension record at offset {off - 8}")
        out[ext_type] = data[off : off + length]
        off += length
    if off != toc:
        raise ValueError(f"trailing garbage in SPZ extension zone ({toc - off} bytes)")
    return out


def load_spz(path: str | Path) -> spz.GaussianCloud:
    """Load an SPZ file and return a GaussianCloud.

    Coordinates are converted to glTF (RUB) coordinate system.
    """
    opts = spz.UnpackOptions()
    opts.to_coord = spz.RUB
    return spz.load_spz(str(path), opts)


def save_spz(gc: spz.GaussianCloud, path: str | Path) -> None:
    """Save a GaussianCloud to an SPZ file.

    Converts from internal glTF (RUB) coordinate system.
    """
    opts = spz.PackOptions()
    opts.from_coord = spz.RUB
    spz.save_spz(gc, opts, str(path))


def load_spz_world(path: str | Path) -> spz.GaussianCloud:
    """Load an SPZ whose numeric axes already use the scene's ENU world frame."""
    opts = spz.UnpackOptions()
    opts.to_coord = spz.UNSPECIFIED
    return spz.load_spz(str(path), opts)


def save_spz_world(gc: spz.GaussianCloud, path: str | Path) -> None:
    """Save ENU world-frame values without applying an implicit axis conversion."""
    opts = spz.PackOptions()
    opts.from_coord = spz.UNSPECIFIED
    spz.save_spz(gc, opts, str(path))


# ``spz`` only exposes path-based (de)serialisation, so the in-memory forms
# below round-trip through a temporary file. Kept here, beside the coord-option
# decisions they share, so callers that hold SPZ *bytes* (archive entries,
# actor-asset payloads) do not each re-invent the temp-file dance.


def load_spz_world_bytes(data: bytes) -> spz.GaussianCloud:
    """Decode an in-memory SPZ stream whose axes are already the world frame."""
    with tempfile.NamedTemporaryFile(suffix=".spz") as tmp:
        tmp.write(data)
        tmp.flush()
        return load_spz_world(tmp.name)


def save_spz_world_bytes(gc: spz.GaussianCloud) -> bytes:
    """Encode ``gc`` to an SPZ stream without an implicit axis conversion."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "cloud.spz"
        save_spz_world(gc, path)
        return path.read_bytes()


def load_ply(path: str | Path) -> spz.GaussianCloud:
    """Load a PLY file (3DGS training output) and return a GaussianCloud.

    Coordinates are converted from COLMAP (RDF) to glTF (RUB) coordinate system.
    """
    opts = spz.UnpackOptions()
    opts.to_coord = spz.RUB
    return spz.load_splat_from_ply(str(path), opts)


def save_ply(gc: spz.GaussianCloud, path: str | Path) -> None:
    """Save a GaussianCloud to a PLY file.

    Converts from internal glTF (RUB) to COLMAP (RDF) coordinate system.
    """
    opts = spz.PackOptions()
    opts.from_coord = spz.RUB
    spz.save_splat_to_ply(gc, opts, str(path))
