"""Export a scene USDZ as a standalone Cesium 3D Tiles 1.1 tileset."""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from .frame_convention import validate_frame_convention, validate_rigid_transform
from .scene_usdz import _apply_transform_to_cloud, _concat_clouds, _walk_leaves
from .spz_io import load_spz
from .tiles_export import TilesetSaveOptions, save_tileset


def export_usdz_tileset(
    source_usdz: str | Path,
    output_dir: str | Path,
    options: TilesetSaveOptions | None = None,
) -> Path:
    """Convert embedded ``EXT_3dgs_spz`` chunks to 3D Tiles 1.1 GLBs.

    Gaussian data stays in the declared Z-up ENU world frame and the Cesium
    root transform is preserved exactly, so geolocation is unchanged.
    """
    source_usdz = Path(source_usdz)
    with zipfile.ZipFile(source_usdz) as archive:
        scene = json.loads(archive.read("scene.json"))
        if scene.get("schema") != "splatsim.scene/v2":
            raise ValueError("scene USDZ must use splatsim.scene/v2")
        validate_frame_convention(scene.get("world", {}).get("frame_convention"))

        tileset = json.loads(archive.read(scene["gaussians"]["tileset"]))
        root = tileset.get("root")
        if not isinstance(root, dict):
            raise ValueError("embedded tileset is missing its root tile")
        root_transform = root.get("transform")
        if root_transform is None:
            root_transform = np.eye(4, dtype=np.float64).T.ravel().tolist()
        if len(root_transform) != 16:
            raise ValueError("embedded tileset root.transform must contain 16 values")
        root_matrix = np.asarray(root_transform, dtype=np.float64).reshape(4, 4).T
        validate_rigid_transform(root_matrix, where="embedded tileset root.transform")

        local_root = {key: value for key, value in root.items() if key != "transform"}
        leaves = list(_walk_leaves(local_root, np.eye(4, dtype=np.float64)))
        if not leaves:
            raise ValueError("embedded tileset contains no SPZ content")

        clouds = []
        with tempfile.TemporaryDirectory() as temp_dir:
            for index, (uri, transform) in enumerate(leaves):
                if Path(uri).suffix.lower() != ".spz":
                    raise ValueError(f"embedded tile content must be SPZ, got {uri!r}")
                temp_path = Path(temp_dir) / f"chunk_{index:06d}.spz"
                temp_path.write_bytes(archive.read(uri))
                cloud = load_spz(temp_path)
                if not np.allclose(transform, np.eye(4)):
                    cloud = _apply_transform_to_cloud(cloud, transform)
                clouds.append(cloud)

    cloud = clouds[0] if len(clouds) == 1 else _concat_clouds(clouds)
    return save_tileset(
        cloud,
        output_dir,
        options,
        root_transform=np.asarray(root_transform, dtype=np.float64),
    )
