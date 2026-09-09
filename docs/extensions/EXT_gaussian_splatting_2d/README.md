# EXT_gaussian_splatting_2d

## Contributors

- 3dgs_io maintainers (TIER IV / Autoware Foundation)

## Status

Draft

## Dependencies

Written against the glTF 2.0 specification.

Requires [`KHR_gaussian_splatting`](https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_gaussian_splatting). This extension is a **kernel child extension**: it defines an alternative Gaussian kernel and MUST be nested inside a `KHR_gaussian_splatting` extension object.

## Overview

`KHR_gaussian_splatting` defines a single kernel, `"ellipse"`, which stores each splat as an anisotropic **3D ellipsoid** (position + quaternion + `VEC3` scale) that is rendered by projecting its 3D covariance to screen space (EWA splatting).

This extension defines a new kernel, **`"ellipse2d"`**, which stores each splat as an oriented **2D elliptical disk** (a *surfel*), following the 2D Gaussian Splatting formulation of Huang et al., *"2D Gaussian Splatting for Geometrically Accurate Radiance Fields"* (SIGGRAPH 2024). A 2D splat lives in a local tangent plane rather than as a volumetric ellipsoid, which yields multi-view-consistent geometry and well-defined surface normals — properties that matter for sensor-simulation workflows (LiDAR, depth/normal supervision) where both camera- and LiDAR-derived Gaussians are represented as thin surface elements.

The two kernels differ in **geometry storage** and **rendering model**, not in their shared appearance data. `POSITION`, rotation, opacity, and spherical-harmonics color are inherited unchanged from `KHR_gaussian_splatting`; only the scale becomes 2-dimensional and the evaluation switches from 3D-covariance projection to ray–disk intersection.

The `kernel` value is the selector: a loader reads `KHR_gaussian_splatting.kernel` and dispatches to the 3D ellipsoid path when it is `"ellipse"` (or absent, per KHR default) and to the 2D disk path when it is `"ellipse2d"`.

## Extending Gaussian Splatting Primitives

A mesh primitive that already declares `KHR_gaussian_splatting` opts into the 2D kernel by:

1. Setting `KHR_gaussian_splatting.kernel` to `"ellipse2d"`.
2. Adding `EXT_gaussian_splatting_2d` to the `extensions` property **of the `KHR_gaussian_splatting` extension object** (nested), per the KHR nesting rule.
3. Replacing the `KHR_gaussian_splatting:SCALE` attribute with the 2-component `EXT_gaussian_splatting_2d:SCALE` attribute.

`EXT_gaussian_splatting_2d` MUST also be listed in the top-level `extensionsUsed`.

```json
{
  "meshes": [
    {
      "primitives": [
        {
          "mode": 0,
          "attributes": {
            "POSITION": 0,
            "KHR_gaussian_splatting:ROTATION": 1,
            "EXT_gaussian_splatting_2d:SCALE": 2,
            "KHR_gaussian_splatting:OPACITY": 3,
            "KHR_gaussian_splatting:SH_DEGREE_0_COEF_0": 4
          },
          "extensions": {
            "KHR_gaussian_splatting": {
              "kernel": "ellipse2d",
              "colorSpace": "srgb_rec709_display",
              "extensions": {
                "EXT_gaussian_splatting_2d": {
                  "doubleSided": true
                }
              }
            }
          }
        }
      ]
    }
  ],
  "extensionsUsed": [
    "KHR_gaussian_splatting",
    "EXT_gaussian_splatting_2d"
  ]
}
```

## Attribute Semantics

Attributes are inherited from `KHR_gaussian_splatting` except for `SCALE`, which this extension redefines.

| Semantic | Accessor Type(s) | Component Type(s) | Description |
|---|---|---|---|
| `POSITION` | `VEC3` | `float` | Splat center (disk center), in the primitive's local space. Inherited from glTF / KHR. |
| `KHR_gaussian_splatting:ROTATION` | `VEC4` | `float`, `byte` normalized, `short` normalized | Unit quaternion `(x, y, z, w)` mapping the splat's local frame to primitive space. Inherited from KHR. Defines the disk's tangent frame (see below). |
| `EXT_gaussian_splatting_2d:SCALE` | **`VEC2`** | `float`, `unsigned byte` normalized, `unsigned short` normalized | Linear standard deviations `(s_u, s_v)` of the Gaussian along the local tangent axes **u** and **v**. Values MUST NOT be negative. **This attribute replaces `KHR_gaussian_splatting:SCALE`; a primitive using the `"ellipse2d"` kernel MUST NOT also define `KHR_gaussian_splatting:SCALE`.** |
| `KHR_gaussian_splatting:OPACITY` | `SCALAR` | `float`, `unsigned byte` normalized, `unsigned short` normalized | Normalized linear opacity in `[0.0, 1.0]`. Inherited from KHR. |
| `KHR_gaussian_splatting:SH_DEGREE_ℓ_COEF_n` | `VEC3` | per KHR | Spherical-harmonics color coefficients (degree 0 required, degrees 1–3 optional). Inherited from KHR unchanged. |

As permitted by `KHR_gaussian_splatting`, two attributes MAY share a single accessor by referencing the same accessor index.

## Geometry Definition

Each 2D splat is a planar elliptical Gaussian disk. Its local frame is defined by the unit quaternion `ROTATION`, applied to the canonical local basis:

- **Tangent axis u** = `ROTATION · (1, 0, 0)`
- **Tangent axis v** = `ROTATION · (0, 1, 0)`
- **Normal n** = `ROTATION · (0, 0, 1)` = **u** × **v**

The disk lies in the plane spanned by **u** and **v** through `POSITION`. A point on the plane with local tangent coordinates `(u, v)` maps to world space as:

```
P(u, v) = POSITION + u · s_u · û + v · s_v · v̂
```

The Gaussian weight at tangent coordinates `(u, v)` is:

```
G(u, v) = exp( -0.5 · ( u² / s_u²  +  v² / s_v² ) )
```

The normal **n** is fully determined by `ROTATION`; it is **not** stored separately. Because the third local axis carries no scale, this kernel represents an infinitely thin surface element rather than a flattened ellipsoid.

## Rendering Model

Conforming renderers MUST evaluate `"ellipse2d"` splats by **ray–disk intersection in the local tangent plane**, not by projecting a 3D covariance to screen space:

1. Intersect the view ray with the splat's tangent plane to obtain local coordinates `(u, v)`.
2. Evaluate `G(u, v)` (above) and multiply by `OPACITY` to obtain the fragment's alpha.
3. Evaluate SH color per the KHR rules and composite front-to-back.

This is the low-pass-filtered 2D Gaussian evaluation described by Huang et al. (2024). Renderers MAY apply the same object-space / screen-space low-pass filter described there to bound the minimum footprint of distant disks.

A `"ellipse2d"` splat MUST NOT be rendered by naively reusing a 3D ellipsoid rasterizer with a zero third scale component; doing so produces line/needle artifacts under EWA projection and is explicitly out of conformance.

## Sidedness

The nested `EXT_gaussian_splatting_2d` object MAY carry rendering hints:

- **`doubleSided`** (`boolean`, default `true`): whether the disk is visible from both sides of its tangent plane. When `false`, the splat contributes only where the view ray meets the front face (the hemisphere the normal **n** points into). Consumers using these splats for oriented-surface reconstruction (e.g. LiDAR normal supervision) SHOULD set this to `false`.

## Quantization

Quantization of the reused attributes (`ROTATION`, `OPACITY`, SH) follows `KHR_gaussian_splatting`.

`EXT_gaussian_splatting_2d:SCALE` MAY be stored as `float`, or quantized as `unsigned byte` / `unsigned short` with `accessor.normalized = true`. Because scales are unbounded above, quantized storage SHOULD encode `(s_u, s_v)` relative to a per-primitive maximum recorded outside this attribute (e.g. via a future compression companion extension); this draft does not yet define such a companion.

## Interaction with SPZ Compression

`KHR_gaussian_splatting_compression_spz*` currently defines a 3D-ellipsoid payload only and is therefore incompatible with `kernel: "ellipse2d"`. A 2D splat cloud MUST be stored uncompressed (per-attribute accessors) until a 2D-aware compression companion is specified. This is an open item (see below).

## 3dgs_io Implementation Notes (non-normative)

This section records how the reference library (`3dgs_io`) dispatches between the two kernels. It is informative, not part of the glTF extension contract.

- **Kernel is currently write-only.** `gltf_io.py` hard-codes `"kernel": "ellipse"` on write (`_save_gltf_standard` ~`gltf_io.py:333`, `_save_gltf_spz` ~`gltf_io.py:535`) and never reads `kernel` on load. Adding 2D support means reading `kernel` in `_parse_gaussian_cloud` (`gltf_io.py:662-683`) and branching.
- **Dispatch point.** In `_parse_gaussian_cloud`, after locating the primitive via `_find_gaussian_primitive` (`gltf_io.py:777-783`), read `kernel = ext.get("kernel", "ellipse")`. Route `"ellipse"` (and the SPZ sub-extension case) to the existing 3D path (`_parse_standard`, `gltf_io.py:686-774`); route `"ellipse2d"` to a new `_parse_standard_2d` handler that reads a `VEC2` scale accessor.
- **Name constant.** Add a module-level `EXT_GAUSSIAN_SPLATTING_2D_NAME = "EXT_gaussian_splatting_2d"` alongside the existing extension-name constants (`gltf_io.py:40-41`), mirroring `EXT_GAUSSIAN_LIDAR_NAME` (`ext_attributes.py:98`), and register it in `extensionsUsed` in both writers when the 2D kernel is selected.
- **Accessor reading.** Reuse `read_accessor` / `TYPE_COMPONENTS` (`_gltf_common.py:29-54`); `VEC2` is already supported by `TYPE_COMPONENTS`.
- **Docs.** The stale stub `docs/api/lidar_2dgs.rst` (references a nonexistent `3dgs_io.lidar_2dgs` module) should be reconciled with the module that ultimately implements this kernel.

## Known Limitations & Open Questions

1. **Compression.** No SPZ/quantized companion for the 2D kernel yet (see above).
2. **Scale quantization reference.** How to store the per-primitive scale maximum for normalized quantization is unspecified.
3. **Naming.** `"ellipse2d"` is proposed to pair with KHR's `"ellipse"`; alternatives considered were `"surfel"` and `"disk"`.
4. **Vendor prefix.** Registered here as an `EXT_` (multi-vendor intent) extension; may need a vendor prefix until adopted by additional implementations.
5. **Explicit normals.** This draft derives the normal from `ROTATION`. If a use case needs a stored normal decoupled from the tangent frame, a future optional `EXT_gaussian_splatting_2d:NORMAL` attribute could be added.

## Appendix: Relation to 2D Gaussian Splatting (Huang et al., 2024)

| 2D GS paper concept | This extension |
|---|---|
| Splat center **p_k** | `POSITION` |
| Tangent vectors **t_u**, **t_v** | Columns of `ROTATION` (local +X, +Y) |
| Normal **t_w = t_u × t_v** | `ROTATION · (0,0,1)`, derived |
| Scales `(s_u, s_v)` | `EXT_gaussian_splatting_2d:SCALE` (`VEC2`) |
| Opacity `α` | `KHR_gaussian_splatting:OPACITY` |
| View-dependent color (SH) | `KHR_gaussian_splatting:SH_DEGREE_*` |
| Ray–splat intersection render | Rendering Model section |
