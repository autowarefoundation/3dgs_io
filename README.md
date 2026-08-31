# 3dgs_io

A Python library for reading and writing 3D Gaussian Splatting data in [glTF](https://www.khronos.org/gltf/) (`KHR_gaussian_splatting`), [3D Tiles](https://www.ogc.org/standard/3dtiles/) (OGC), [SPZ](https://github.com/nianticlabs/spz), and PLY formats.

Documentation: https://tier4.github.io/3dgs_io/

## Frame-explicit USDZ scenes

Scene bundles use the breaking `splatsim.scene/v2`,
`splatsim.rig_trajectories/v2`, and `splatsim.sequence_tracks/v2` schemas.
Gaussians, rigs, and tracks share one right-handed Z-up ENU world frame;
rigs use X-forward/Y-left/Z-up, quaternions are xyzw, and timestamps are
strictly increasing u64 microseconds. Writers reject reflections, invalid
rotations, and mismatched frame declarations.

Export an embedded SPZ scene to a standalone Cesium 3D Tiles 1.1 tileset:

```bash
python -m 3dgs_io export-tiles scene.usdz output_tiles/
```

The API equivalent is `3dgs_io.export_usdz_tileset(...)`. Both paths retain
the reconciled ECEF root anchor.

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
