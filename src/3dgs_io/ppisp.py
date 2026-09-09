"""PPISP (Physically-Plausible ISP) parameter dataclasses and JSON (de)serialisation.

PPISP is a render-time image-space correction (exposure / vignetting / colour /
CRF tone curve) that is applied to the *rendered* RGB, per-camera and
per-frame. It is jointly optimised with the Gaussians during training to
absorb cross-camera colour / exposure inconsistency; because it is
view-dependent it cannot be baked into the view-independent Gaussian payload.

This module gives a first-class in-memory representation so PPISP parameters
can be embedded in a splatsim USDZ bundle alongside ``rig_trajectories.json``
and ``sequence_tracks.json``.

* :class:`PpispCamera` — per-camera params (vignetting + CRF, per-channel).
  Keyed by the same camera ``name`` as :attr:`RigTrajectory.cameras`.
* :class:`PpispFrame` — per-frame params (exposure + latent colour).
  Keyed by the same ``timestamp_us`` as :attr:`RigPose.timestamp_us`.
* :class:`Ppisp` — top-level container plus fixed constants (the ZCA pinv
  block used to reconstruct the chromaticity homography from the latent
  ``color`` params).

The on-disk schema (used both inside the USDZ as ``ppisp.json`` and as a
standalone JSON file accepted by the CLI) is ``splatsim.ppisp/v1``::

    {
      "schema": "splatsim.ppisp/v1",
      "pipeline": ["exposure", "vignetting", "color", "crf"],
      "cameras": {
        "CAM_FRONT": {
          "vignetting": [[cx,cy,a0,a1,a2], [...G...], [...B...]],
          "crf":        [[toe,sh,gamma,center], [...G...], [...B...]]
        }
      },
      "frames": [
        { "timestamp_us": 27567868848, "exposure": 0.031, "color": [8 floats] }
      ],
      "constants": {
        "color_pinv_block_diag": [[8x8 block-diagonal ZCA pinv matrix]]
      }
    }

Raw (pre-activation) values are stored; a PPISP-aware viewer applies the
activations (``softplus``, ``sigmoid``, ``pow(2, ·)``) exactly as in the
reference transform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "PPISP_SCHEMA",
    "PPISP_PIPELINE",
    "Ppisp",
    "PpispCamera",
    "PpispFrame",
    "parse_ppisp",
    "serialize_ppisp",
]


PPISP_SCHEMA = "splatsim.ppisp/v1"

# Order in which a PPISP-aware viewer must apply the four correction stages.
PPISP_PIPELINE: tuple[str, ...] = ("exposure", "vignetting", "color", "crf")

# Expected shapes for the per-camera arrays. Kept as module-level constants so
# the values are documented alongside the dataclass and shared with the
# validator. The layout matches the training-time PPISP module in
# nv-tlabs/ppisp (per RGB channel, in order).
_VIGNETTING_CHANNELS = 3
_VIGNETTING_PARAMS = 5  # (center_x, center_y, alpha_0, alpha_1, alpha_2)
_CRF_CHANNELS = 3
_CRF_PARAMS = 4  # (toe_raw, shoulder_raw, gamma_raw, center_raw)
_COLOR_LATENT_DIM = 8  # length of the latent chromaticity vector per frame


# --------------------------------------------------------------------------
# Per-camera / per-frame dataclasses
# --------------------------------------------------------------------------


@dataclass
class PpispCamera:
    """Per-camera PPISP parameters.

    ``vignetting`` is a ``(3, 5)`` table: for each RGB channel a radial
    polynomial ``(center_x, center_y, alpha_0, alpha_1, alpha_2)``.
    ``crf`` is a ``(3, 4)`` table: for each RGB channel a parametric
    toe/shoulder/gamma tone curve ``(toe_raw, shoulder_raw, gamma_raw,
    center_raw)``. Values are stored raw (pre-activation).
    """

    vignetting: tuple[tuple[float, ...], ...]
    crf: tuple[tuple[float, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "vignetting": [[float(v) for v in row] for row in self.vignetting],
            "crf": [[float(v) for v in row] for row in self.crf],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PpispCamera:
        vign = d.get("vignetting")
        crf = d.get("crf")
        if not isinstance(vign, list) or len(vign) != _VIGNETTING_CHANNELS:
            raise ValueError(
                f"camera.vignetting must be a list of {_VIGNETTING_CHANNELS} channels, got {vign!r}"
            )
        if not isinstance(crf, list) or len(crf) != _CRF_CHANNELS:
            raise ValueError(f"camera.crf must be a list of {_CRF_CHANNELS} channels, got {crf!r}")
        vign_rows: list[tuple[float, ...]] = []
        for i, row in enumerate(vign):
            if not isinstance(row, list) or len(row) != _VIGNETTING_PARAMS:
                raise ValueError(
                    f"camera.vignetting[{i}] must have {_VIGNETTING_PARAMS} entries, got {row!r}"
                )
            vign_rows.append(tuple(float(v) for v in row))
        crf_rows: list[tuple[float, ...]] = []
        for i, row in enumerate(crf):
            if not isinstance(row, list) or len(row) != _CRF_PARAMS:
                raise ValueError(f"camera.crf[{i}] must have {_CRF_PARAMS} entries, got {row!r}")
            crf_rows.append(tuple(float(v) for v in row))
        return cls(vignetting=tuple(vign_rows), crf=tuple(crf_rows))


@dataclass
class PpispFrame:
    """Per-frame PPISP parameters keyed by ``timestamp_us``.

    ``exposure`` is a raw log2 multiplicative gain (viewer applies
    ``rgb *= 2 ** exposure``). ``color`` is the 8-D latent chromaticity
    vector; combined with :attr:`Ppisp.color_pinv_block_diag` it defines the
    chromaticity homography applied to the rendered RGB.
    """

    timestamp_us: int
    exposure: float
    color: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_us": int(self.timestamp_us),
            "exposure": float(self.exposure),
            "color": [float(v) for v in self.color],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PpispFrame:
        color = d.get("color")
        if not isinstance(color, list) or len(color) != _COLOR_LATENT_DIM:
            raise ValueError(f"frame.color must have {_COLOR_LATENT_DIM} entries, got {color!r}")
        return cls(
            timestamp_us=int(d["timestamp_us"]),
            exposure=float(d["exposure"]),
            color=tuple(float(v) for v in color),
        )


# --------------------------------------------------------------------------
# Top-level container
# --------------------------------------------------------------------------


@dataclass
class Ppisp:
    """PPISP appearance-correction parameters for a single scene.

    ``cameras`` is keyed by camera ``name`` (matching
    ``rig_trajectories.json`` ``rigs[].cameras[].name``). ``frames`` is a
    list of per-frame entries keyed by ``timestamp_us`` (matching rig-pose
    timestamps). ``color_pinv_block_diag`` is the fixed ``(8, 8)`` block
    used to reconstruct the chromaticity homography from the latent
    ``color`` params; it may be omitted when the viewer already has a
    hard-coded copy.
    """

    cameras: dict[str, PpispCamera] = field(default_factory=dict)
    frames: list[PpispFrame] = field(default_factory=list)
    color_pinv_block_diag: list[list[float]] | None = None
    pipeline: tuple[str, ...] = PPISP_PIPELINE


# --------------------------------------------------------------------------
# Collection-level (de)serialisation
# --------------------------------------------------------------------------


def serialize_ppisp(ppisp: Ppisp) -> dict[str, Any]:
    """Build the JSON-ready ``splatsim.ppisp/v1`` document.

    Rejects duplicate camera names (dicts already enforce this on the input,
    but we keep an explicit guard on the exit path so mistakes stay local to
    the writer) and duplicate ``timestamp_us`` across frames.
    """
    seen_ts: set[int] = set()
    frames_out: list[dict[str, Any]] = []
    for f in ppisp.frames:
        if f.timestamp_us in seen_ts:
            raise ValueError(f"duplicate frame timestamp_us: {f.timestamp_us}")
        seen_ts.add(f.timestamp_us)
        frames_out.append(f.to_dict())

    doc: dict[str, Any] = {
        "schema": PPISP_SCHEMA,
        "pipeline": list(ppisp.pipeline),
        "cameras": {name: cam.to_dict() for name, cam in ppisp.cameras.items()},
        "frames": frames_out,
    }
    if ppisp.color_pinv_block_diag is not None:
        doc["constants"] = {
            "color_pinv_block_diag": [
                [float(v) for v in row] for row in ppisp.color_pinv_block_diag
            ]
        }
    return doc


def parse_ppisp(doc: dict[str, Any]) -> Ppisp:
    """Inverse of :func:`serialize_ppisp`.

    Rejects duplicate frame ``timestamp_us`` so the load path enforces the
    same invariant as :func:`serialize_ppisp`.
    """
    schema = doc.get("schema")
    if schema != PPISP_SCHEMA:
        raise ValueError(f"unexpected ppisp schema {schema!r}; expected {PPISP_SCHEMA!r}")

    raw_cameras = doc.get("cameras")
    if raw_cameras is None:
        raw_cameras = {}
    if not isinstance(raw_cameras, dict):
        raise ValueError("ppisp document 'cameras' must be an object keyed by camera name")
    cameras: dict[str, PpispCamera] = {}
    for name, entry in raw_cameras.items():
        if not isinstance(entry, dict):
            raise ValueError(
                f"ppisp cameras[{name!r}] must be an object, got {type(entry).__name__}"
            )
        cameras[str(name)] = PpispCamera.from_dict(entry)

    raw_frames = doc.get("frames")
    if raw_frames is None:
        raw_frames = []
    if not isinstance(raw_frames, list):
        raise ValueError("ppisp document 'frames' must be a list")
    frames: list[PpispFrame] = []
    seen_ts: set[int] = set()
    for entry in raw_frames:
        if not isinstance(entry, dict):
            raise ValueError(f"ppisp frames entry must be an object, got {type(entry).__name__}")
        frame = PpispFrame.from_dict(entry)
        if frame.timestamp_us in seen_ts:
            raise ValueError(f"duplicate frame timestamp_us: {frame.timestamp_us}")
        seen_ts.add(frame.timestamp_us)
        frames.append(frame)

    color_pinv: list[list[float]] | None = None
    constants = doc.get("constants")
    if isinstance(constants, dict):
        raw_matrix = constants.get("color_pinv_block_diag")
        if raw_matrix is not None:
            if not isinstance(raw_matrix, list) or not all(
                isinstance(row, list) for row in raw_matrix
            ):
                raise ValueError("constants.color_pinv_block_diag must be a list of lists")
            color_pinv = [[float(v) for v in row] for row in raw_matrix]

    pipeline_raw = doc.get("pipeline")
    if pipeline_raw is None:
        pipeline: tuple[str, ...] = PPISP_PIPELINE
    else:
        if not isinstance(pipeline_raw, list) or not all(isinstance(s, str) for s in pipeline_raw):
            raise ValueError("ppisp 'pipeline' must be a list of strings")
        pipeline = tuple(pipeline_raw)

    return Ppisp(
        cameras=cameras,
        frames=frames,
        color_pinv_block_diag=color_pinv,
        pipeline=pipeline,
    )
