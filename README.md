# 3dgs_io

A Python library for reading and writing 3D Gaussian Splatting data in [glTF](https://www.khronos.org/gltf/) (`KHR_gaussian_splatting`), [3D Tiles](https://www.ogc.org/standard/3dtiles/) (OGC), [SPZ](https://github.com/nianticlabs/spz), and PLY formats.

## Features

- **glTF/GLB** save & load with the [`KHR_gaussian_splatting`](https://github.com/KhronosGroup/glTF/pull/2446) extension
- **SPZ compression** via `KHR_gaussian_splatting_compression_spz_2`
- **3D Tiles** reader with multi-layer support (camera 3DGS / LiDAR 2DGS)
- **LiDAR 2DGS** dedicated I/O for surfel-based Gaussian representations
- **SPZ / PLY** import & export with automatic coordinate system conversion
- **Typed metadata** (`GlbMetadata`) stored in `asset.extras` with backward-compatible parsing

## Installation

Requires Python >= 3.10.

```bash
pip install 3dgs-io
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add 3dgs-io
```

## Quick start

### Save & load a glTF/GLB file

```python
import importlib

gs = importlib.import_module("3dgs_io")

# Load from SPZ
cloud = gs.load_spz("input.spz")

# Save as glTF (standard)
gs.save_gltf(cloud, "output.glb")

# Save with SPZ compression
gs.save_gltf(cloud, "output_spz.glb", gs.GltfSaveOptions(spz_compression=True))

# Load back
cloud = gs.load_gltf("output.glb")
```

### Load a 3D Tiles tileset

```python
import importlib

gs = importlib.import_module("3dgs_io")

# Load tiles from a local path or URL
tiles = gs.load_tileset("path/to/tileset.json")

# Merge all tiles into a single cloud
merged = gs.merge_tileset(tiles)
gs.save_gltf(merged, "merged.glb")

# Load LiDAR 2DGS layer
lidar_tiles = gs.load_tileset(
    "path/to/tileset.json",
    layer=gs.LayerType.LIDAR_2DGS,
)
```

### Save & load with metadata

```python
import importlib

gs = importlib.import_module("3dgs_io")

metadata = gs.GlbMetadata(
    dataset_type=gs.DatasetType.T4_DATASET,
    generator="my-pipeline",
    training_data=gs.TrainingData(
        source_path="/data/scene_001",
        data_type="T4",
        revision="abc123",
        scene_index=0,
        lidar_channel="LIDAR_CONCAT",
        selected_frames=[0, 60],
        cameras=["cam_front"],
        camera_channels=["cam_front_channel"],
        start_timestamp_us=1000000,
        end_timestamp_us=2000000,
    ),
    checkpoint=gs.Checkpoint(path="/ckpt/iter_30000.pth", iteration=30000),
    export=gs.Export(background_only=False, spz_compression=True),
    model=gs.Model(total_gaussians=50000),
    placement=gs.Placement(lat=35.6812, lon=139.7671, height=40.0),
)

cloud = gs.load_spz("input.spz")
gs.save_gltf(cloud, "output.glb", gs.GltfSaveOptions(metadata=metadata))

# Load with metadata
cloud, meta = gs.load_gltf_with_metadata("output.glb")
```

### SPZ & PLY conversion

```python
import importlib

gs = importlib.import_module("3dgs_io")

# PLY (COLMAP output) -> SPZ
cloud = gs.load_ply("point_cloud.ply")
gs.save_spz(cloud, "output.spz")

# SPZ -> PLY
cloud = gs.load_spz("input.spz")
gs.save_ply(cloud, "output.ply")
```

### LiDAR 2DGS

```python
import importlib

gs = importlib.import_module("3dgs_io")

# Load LiDAR surfel cloud
cloud = gs.load_lidar_gltf("lidar.glb")

# Save with metadata
gs.save_lidar_gltf(cloud, "lidar_out.glb", metadata=metadata)

# Load with metadata
cloud, meta = gs.load_lidar_gltf_with_metadata("lidar.glb")
```

## API reference

| Module | Description |
|---|---|
| `gltf_io` | glTF/GLB save & load with `KHR_gaussian_splatting` |
| `spz_io` | SPZ and PLY import/export with coordinate conversion |
| `tiles_io` | 3D Tiles (OGC) reader, tile merging |
| `lidar_2dgs` | LiDAR 2DGS surfel I/O |
| `metadata` | Typed `GlbMetadata` schema for `asset.extras` |

## Development

```bash
uv sync
uv run pre-commit install
uv run pytest
```

## License

See [LICENSE](LICENSE) for details.
