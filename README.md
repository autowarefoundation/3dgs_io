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

## Dynamic objects: rigid actor assets (`splatsim.actor_assets/v1`)

`sequence_tracks.json` says *where* every dynamic object is at every moment; it
cannot say what it looks like, because a bundle's Gaussians are one static
world-frame cloud. `actor_assets.json` supplies the other half — an **asset
bank** of object-local Gaussian clouds plus the bindings that say which track
renders with which asset:

```
actor_assets.json                     # index + track bindings
actor_assets/<asset_id>/asset.spz     # NGSP v4 SPZ, object-local frame
```

recorded in `scene.json.extras.actor_assets`. A renderer draws frame `t` by
taking the track's pose `(R, p)` and instancing the asset's Gaussians through
it — means rotated and translated, orientation quaternions pre-multiplied by
`R`, everything else untouched.

**Scope is rigid objects** — cars, trucks, trailers, cones: anything whose
shape does not change over time. Each asset declares `motion: "rigid"`, and a
v1 reader rejects any other value rather than rendering a walking pedestrian as
a frozen statue.

**Gaussian parameters are identical to the static background.** The payload is
the same NGSP v4 SPZ container the chunks use, carrying the same optional
per-Gaussian LiDAR extension record (`0x54340001`: `lidar_intensity_raw`,
`lidar_raydrop_logit`, `lidar_mask`, `raydrop_sh`). Nothing about an actor is a
special kind of Gaussian.

### Object-local frame

Every document carries the convention and it is compared exactly on read:

| | |
| --- | --- |
| axes | right-handed, **+X forward** (drive direction), **+Y left**, **+Z up** — the same FLU triad as the sensor rig |
| origin | **centre of the oriented bounding box**, all three axes — exactly what `Track.frames[].translation` refers to |
| units | metres, xyzw quaternions |
| scale | **metric**, 1:1 — not unit-normalised |

Leaving this implicit is the classic failure: NVIDIA's NuRec asset-insertion
workflow needs a hand-applied 90° rotation to reconcile Asset-Harvester output
with its renderer. Convert before packing instead — shift a ground-origin model
by `+size_z / 2` along `+Z`, rotate a glTF/SPZ RUB model by
`3dgs_io.frame_convention.RUB_TO_ENU`. An asset whose AABB does not contain the
origin is rejected at write time, which catches the "still in world
coordinates" bug before it reaches a renderer.

### Metric scale and `fit_mode`

NuRec normalises assets to unit scale and stretches them to each track's cuboid
at render time. That distorts geometry non-uniformly, which a LiDAR simulator
must not do — a range return off a car stretched into a truck's box is wrong by
construction. So assets are stored at true metric scale, and each binding picks
how it reconciles the asset's box with the track's:

| `fit_mode` | behaviour |
| --- | --- |
| `rigid` (default) | use the asset as authored; the track's `size` is metadata only. Metrically faithful. |
| `uniform` | one isotropic scale factor. Shape preserved, scale approximate. |
| `stretch` | per-axis scale (NuRec-equivalent). Interoperable, **not** metrically faithful — never for LiDAR ground truth. |

`asset_id` is decoupled from `track_id` (as NuRec's
`DynamicObjectTrack.asset_id` is), so one asset can back fifty tracks and a
track is re-skinned by editing one binding. `sequence_tracks/v2` is unchanged.

### Usage

Cut an actor out of a world-frame reconstruction and pack it:

```python
import 3dgs_io as io

asset = io.extract_actor_asset(
    world_cloud,                      # the scene's ENU world-frame gaussians
    asset_id="sedan_0007",
    class_name="automobile",
    size=track.size,                  # the tracked box (dx, dy, dz), metres
    translation=frame.translation,    # one TrackFrame pose of that box
    rotation=frame.rotation,          # xyzw
    margin=0.2,                       # splat tails spill past a tight box
)

io.save_scene_usdz(
    "tileset.json", "scene.usdz",
    tracks=[track],
    actor_assets=[asset],
    actor_instances=[io.ActorInstance(track_id=track.track_id, asset_id="sedan_0007")],
)
```

View-dependent SH lives in the object frame too. `extract_actor_asset` rotates
it there exactly for the yaw-only poses road vehicles have; pass
`sh_policy="drop"` for a view-independent asset, and note that a pose with more
than ~2° of pitch/roll is refused rather than silently approximated.

Retrofit a bank onto an existing bundle (the usual path — bundles are built
before their actors are harvested):

```bash
python -m 3dgs_io.edit_usdz_cli actor-assets \
    --input        path/to/scene.usdz \
    --output       path/to/scene_with_actors.usdz \
    --actor-assets path/to/actor_asset_bank_dir
```

A bank directory mirrors the in-archive layout exactly, so one bank can be
authored once (`save_actor_asset_dir`) and packed into many bundles
(`--actor-assets` on `scene_usdz_cli`, or `actor_assets=<dir>` on
`save_scene_usdz`) with its payloads carried byte-for-byte.

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
