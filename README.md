# 3dgs_io

A Python library for reading and writing 3D Gaussian Splatting data in [glTF](https://www.khronos.org/gltf/) (`KHR_gaussian_splatting`), [3D Tiles](https://www.ogc.org/standard/3dtiles/) (OGC), [SPZ](https://github.com/nianticlabs/spz), and PLY formats.

Documentation: https://tier4.github.io/3dgs_io/

## Frame-explicit USDZ scenes

Scene bundles use the breaking `splatsim.scene/v3`,
`splatsim.rig_trajectories/v2`, and `splatsim.sequence_tracks/v2` schemas.
Since `splatsim.scene/v3`:

- Per-Gaussian LiDAR attributes (`EXT_gaussian_lidar`) are embedded inside
  each `chunks/chunk_NNNNNN.spz` as an SPZ extension record (type
  `0x54340001`) instead of `.lidar` sidecar files; chunks are NGSP v4 SPZ
  (requires spz >= 3.0.0 to read).
- Bundles contain no Cesium 3D Tiles structures. The chunk list lives in
  `scene.json.gaussians.chunks` (`uri` / `n_points` / world-frame
  `bbox_min`/`bbox_max`), and chunk SPZ values are stored directly in the
  alpasim ENU world frame. The Cesium `tileset.json` format is accepted only
  as *input* to `save_scene_usdz`; the Cesium tileset export
  (`export_usdz_tileset`) and the bundled viewer were removed.

Gaussians, rigs, and tracks share one right-handed Z-up ENU world frame;
rigs use X-forward/Y-left/Z-up, quaternions are xyzw, and timestamps are
strictly increasing u64 microseconds. Writers reject reflections, invalid
rotations, and mismatched frame declarations.

## Autoware map bundle (`autoware_map/`)

HD maps are embedded under an `autoware_map/` prefix that mirrors Autoware's
own [map directory layout](https://autowarefoundation.github.io/autoware-documentation/main/how-to-guides/integrating-autoware/launch-autoware/map/),
so a consumer can extract `autoware_map/` and hand it to `autoware_map_loader`
verbatim:

```
autoware_map/
├── lanelet2_map.osm              # vector map      -> extras.map_lanelet2
├── pointcloud_map.pcd            # single PCD, OR  -> extras.map_pointcloud
├── pointcloud_map/               #   split tiles   -> extras.map_pointcloud ("autoware_map/pointcloud_map/")
│   ├── A.pcd
│   └── B.pcd
├── pointcloud_map_metadata.yaml  # split-tile index -> extras.map_pointcloud_metadata
└── map_projector_info.yaml       # projection info  -> extras.map_projector_info
```

The lanelet2 map lives here now (`autoware_map/lanelet2_map.osm`); the earlier
archive-root `map.osm` location was removed. When the point-cloud map is split
into multiple `.pcd` files, `extras.map_pointcloud` records the directory
prefix (trailing slash) instead of a single file. The writer still
cross-checks a bundle's `ecef_anchor` against the embedded lanelet2 map and
refuses to produce one whose world origin decodes far from the map.

Embed maps into an existing bundle with the editor CLI — a single file, a
whole map directory, or a point-cloud map (single `.pcd` or a tile directory):

```bash
# whole Autoware map directory
python -m 3dgs_io.edit_usdz_cli autoware-map \
    --input scene.usdz --output scene.usdz --map-dir path/to/autoware_map

# lanelet2 only
python -m 3dgs_io.edit_usdz_cli lanelet2 \
    --input scene.usdz --output scene.usdz --lanelet2 path/to/lanelet2_map.osm

# point cloud (file or directory, with optional metadata)
python -m 3dgs_io.edit_usdz_cli pointcloud \
    --input scene.usdz --output scene.usdz \
    --pcd path/to/pointcloud_map --metadata path/to/pointcloud_map_metadata.yaml
```

## USDZ scene-bundle manifest (`metadata.yaml`)

Every USDZ produced by `3dgs_io.save_scene_usdz` writes a `metadata.yaml` at
the archive root as a stability commitment to downstream consumers. Fields:

| Key | Type | Required | Description |
| --- | --- | --- | --- |
| `uuid` | non-empty string | yes | Globally unique identifier for the scene asset. |
| `scene_id` | non-empty string | yes | Human-readable scene identifier (typically the dataset or run name). |
| `version_string` | non-empty string | yes | Free-form identifier of the producing pipeline (e.g. the `3dgs_io` release, or the parent pipeline's own version). |
| _extras_ | any JSON-serialisable | no | Additional keys downstream tools may ignore. |

The file is encoded as JSON — a subset of YAML 1.2 — so consumers can parse
it with `yaml.safe_load`:

```python
import yaml, zipfile

with zipfile.ZipFile(usdz_file, "r") as zf, zf.open("metadata.yaml") as fh:
    data = yaml.safe_load(fh)
    uuid = data["uuid"]
    scene_id = data["scene_id"]
    version = data.get("version_string", "unknown")
```

Retrofit an older USDZ that predates the commitment with the `metadata`
sub-command of the editor CLI (no Gaussian chunks are touched):

```bash
python -m 3dgs_io.edit_usdz_cli metadata \
    --input  path/to/scene.usdz \
    --output path/to/scene.usdz \
    --uuid odaibatest5 \
    --scene-id odaibatest5 \
    --version-string local-e2e
```
