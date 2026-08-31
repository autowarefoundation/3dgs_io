"""Export a scene USDZ as a standalone Cesium 3D Tiles 1.1 tileset."""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from .frame_convention import RUB_TO_ENU, validate_frame_convention, validate_rigid_transform
from .scene_usdz import _apply_transform_to_cloud, _concat_clouds, _walk_leaves
from .spz_io import load_spz_world
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
        if "transform" in root:
            raise ValueError("scene USDZ tileset must not contain a Cesium root.transform")
        root_matrix = validate_rigid_transform(
            scene["world"]["ecef_anchor"], where="scene world.ecef_anchor"
        )

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
                cloud = load_spz_world(temp_path)
                if not np.allclose(transform, np.eye(4)):
                    cloud = _apply_transform_to_cloud(cloud, transform)
                clouds.append(cloud)

    cloud = clouds[0] if len(clouds) == 1 else _concat_clouds(clouds)
    world_to_gltf = np.eye(4, dtype=np.float64)
    world_to_gltf[:3, :3] = RUB_TO_ENU.T
    cloud = _apply_transform_to_cloud(cloud, world_to_gltf)
    gltf_to_world = np.linalg.inv(world_to_gltf)
    cesium_root = root_matrix @ gltf_to_world
    return save_tileset(
        cloud,
        output_dir,
        options,
        root_transform=cesium_root.T,
    )
