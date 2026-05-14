"""Shared glTF/GLB read/write infrastructure.

Used by both ``gltf_io`` (camera 3DGS) and ``lidar_2dgs`` (LiDAR 2DGS).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

# glTF component types
FLOAT = 5126
UNSIGNED_BYTE = 5121

_GLB_MAGIC = b"glTF"
_GLB_VERSION = 2
_GLB_JSON_TYPE = 0x4E4F534A  # 'JSON'
_GLB_BIN_TYPE = 0x004E4942  # 'BIN\0'

COMPONENT_DTYPE = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}

TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


def read_accessor(
    accessor: dict,
    buffer_views: list[dict],
    buffer_data: bytes,
) -> np.ndarray:
    """Read a glTF accessor into a numpy array."""
    bv = buffer_views[accessor["bufferView"]]
    byte_offset = bv.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    count = accessor["count"]
    dtype = COMPONENT_DTYPE[accessor["componentType"]]
    components = TYPE_COMPONENTS[accessor["type"]]
    arr = np.frombuffer(buffer_data, dtype=dtype, count=count * components, offset=byte_offset)
    if components > 1:
        arr = arr.reshape(count, components)
    return arr.copy()


def pack_buffer(data_list: list[bytes]) -> tuple[bytes, list[int], list[int]]:
    """Pack byte arrays into a single buffer with 4-byte alignment."""
    buffer_parts: list[bytes] = []
    offsets: list[int] = []
    lengths: list[int] = []
    current_offset = 0

    for data in data_list:
        padding = (4 - current_offset % 4) % 4
        if padding:
            buffer_parts.append(b"\x00" * padding)
            current_offset += padding
        offsets.append(current_offset)
        lengths.append(len(data))
        buffer_parts.append(data)
        current_offset += len(data)

    return b"".join(buffer_parts), offsets, lengths


def write_glb(path: Path, gltf_dict: dict, buffer_data: bytes) -> None:
    """Write a GLB (binary glTF) file."""
    json_bytes = json.dumps(gltf_dict, separators=(",", ":")).encode("utf-8")
    json_pad = (4 - len(json_bytes) % 4) % 4
    json_bytes += b"\x20" * json_pad
    json_length = len(json_bytes)

    bin_pad = (4 - len(buffer_data) % 4) % 4
    bin_data = buffer_data + b"\x00" * bin_pad
    bin_length = len(bin_data)

    total_length = 12 + 8 + json_length + 8 + bin_length

    with open(path, "wb") as f:
        f.write(_GLB_MAGIC)
        f.write(struct.pack("<II", _GLB_VERSION, total_length))
        f.write(struct.pack("<II", json_length, _GLB_JSON_TYPE))
        f.write(json_bytes)
        f.write(struct.pack("<II", bin_length, _GLB_BIN_TYPE))
        f.write(bin_data)


def read_glb(path: Path) -> tuple[dict, bytes]:
    """Read a GLB file and return (gltf_dict, buffer_data)."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != _GLB_MAGIC:
            raise ValueError(f"Invalid GLB magic: {path}")
        version = struct.unpack("<I", f.read(4))[0]
        if version != 2:
            raise ValueError(f"Unsupported GLB version: {version}")
        total_length = struct.unpack("<I", f.read(4))[0]

        json_data: dict | None = None
        bin_data: bytes | None = None

        while f.tell() < total_length:
            chunk_length, chunk_type = struct.unpack("<II", f.read(8))
            chunk_bytes = f.read(chunk_length)
            if chunk_type == _GLB_JSON_TYPE:
                json_data = json.loads(chunk_bytes)
            elif chunk_type == _GLB_BIN_TYPE:
                bin_data = chunk_bytes

    if json_data is None:
        raise ValueError("GLB file missing JSON chunk")
    if bin_data is None:
        raise ValueError("GLB file missing BIN chunk")
    return json_data, bin_data


def write_gltf(path: Path, gltf_dict: dict, buffer_data: bytes) -> None:
    """Write separate .gltf (JSON) and .bin (binary) files."""
    bin_path = path.with_suffix(".bin")
    gltf_dict["buffers"][0]["uri"] = bin_path.name

    with open(path, "w", encoding="utf-8") as f:
        json.dump(gltf_dict, f, indent=2)
    with open(bin_path, "wb") as f:
        f.write(buffer_data)


def read_gltf(path: Path) -> tuple[dict, bytes]:
    """Read a .gltf file and its companion .bin, return (gltf_dict, buffer_data)."""
    with open(path, encoding="utf-8") as f:
        gltf_dict = json.load(f)

    uri = gltf_dict["buffers"][0].get("uri")
    if uri is None:
        raise ValueError("glTF buffer has no URI")

    bin_path = path.parent / uri
    with open(bin_path, "rb") as f:
        buffer_data = f.read()
    return gltf_dict, buffer_data


def write_file(path: Path, gltf_dict: dict, buffer_data: bytes) -> None:
    """Write glTF data to either .glb or .gltf based on file extension."""
    suffix = path.suffix.lower()
    if suffix == ".glb":
        write_glb(path, gltf_dict, buffer_data)
    elif suffix == ".gltf":
        write_gltf(path, gltf_dict, buffer_data)
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix} (use .glb or .gltf)")


def read_file(path: Path) -> tuple[dict, bytes]:
    """Read glTF data from either .glb or .gltf based on file extension."""
    suffix = path.suffix.lower()
    if suffix == ".glb":
        return read_glb(path)
    if suffix == ".gltf":
        return read_gltf(path)
    raise ValueError(f"Unsupported file extension: {path.suffix} (use .glb or .gltf)")
