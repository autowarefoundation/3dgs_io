"""Camera intrinsic + extrinsic dataclasses and JSON (de)serialisation.

A :class:`Camera` pairs:

* :class:`CameraIntrinsics` — pinhole / OpenCV-style intrinsic parameters
  (focal length, principal point, image size, optional distortion).
* :class:`CameraExtrinsics` — the camera-to-scene rigid pose, expressed as a
  translation plus an xyzw quaternion in the **root-local frame** (the same
  coordinate system as the SPZ chunks embedded by :func:`save_scene_usdz`).
  To lift a camera into world space apply the corresponding tileset's
  ``root.transform`` after the cam-to-world pose.

The on-disk schema (used both inside the USDZ as ``cameras.json`` and as a
standalone JSON file accepted by the CLI) is ``splatsim.cameras/v1``::

    {
      "schema": "splatsim.cameras/v1",
      "frame": "root_local",
      "cameras": [
        {
          "name": "front_left",
          "intrinsics": {
            "width": 1920, "height": 1080,
            "fx": 1234.5, "fy": 1234.5,
            "cx": 960.0,  "cy": 540.0,
            "distortion_model": "opencv",
            "distortion_coeffs": [0.01, -0.002, 0.0, 0.0, 0.0]
          },
          "extrinsics": {
            "translation": [1.5, 0.0, 1.8],
            "rotation":    [0.0, 0.0, 0.0, 1.0]
          },
          "timestamp_us": 1700000000000,
          "metadata": {}
        },
        ...
      ]
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._quat import quat_from_rotation_matrix

__all__ = [
    "CAMERAS_SCHEMA",
    "Camera",
    "CameraExtrinsics",
    "CameraIntrinsics",
    "parse_cameras",
    "serialize_cameras",
]


CAMERAS_SCHEMA = "splatsim.cameras/v1"

# Frame label written into the JSON. Cameras are in the same frame as the
# embedded SPZ chunks (root-local); apply tileset.json root.transform to
# place the camera into world space.
_FRAME = "root_local"


@dataclass
class CameraIntrinsics:
    """Pinhole / OpenCV-style intrinsics.

    ``fx`` / ``fy`` are focal lengths in pixels, ``cx`` / ``cy`` the principal
    point. ``distortion_model`` documents how ``distortion_coeffs`` should be
    interpreted; for ``"pinhole"`` the coefficient list is ignored.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str = "pinhole"  # "pinhole" | "opencv" | "opencv_fisheye"
    distortion_coeffs: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": int(self.width),
            "height": int(self.height),
            "fx": float(self.fx),
            "fy": float(self.fy),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "distortion_model": str(self.distortion_model),
            "distortion_coeffs": [float(c) for c in self.distortion_coeffs],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CameraIntrinsics:
        return cls(
            width=int(d["width"]),
            height=int(d["height"]),
            fx=float(d["fx"]),
            fy=float(d["fy"]),
            cx=float(d["cx"]),
            cy=float(d["cy"]),
            distortion_model=str(d.get("distortion_model", "pinhole")),
            distortion_coeffs=[float(c) for c in d.get("distortion_coeffs", [])],
        )


@dataclass
class CameraExtrinsics:
    """Camera-to-scene rigid pose in root-local frame.

    Stored as a translation triple plus an xyzw quaternion. Use
    :meth:`from_matrix` / :meth:`to_matrix` to convert to and from a 4×4
    row-major homogeneous transform.
    """

    translation: tuple[float, float, float]
    rotation: tuple[float, float, float, float]  # xyzw, unit-norm

    def to_dict(self) -> dict[str, Any]:
        return {
            "translation": [float(v) for v in self.translation],
            "rotation": [float(v) for v in self.rotation],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CameraExtrinsics:
        tr = d["translation"]
        ro = d["rotation"]
        if len(tr) != 3:
            raise ValueError(f"extrinsics.translation must have 3 elements, got {len(tr)}")
        if len(ro) != 4:
            raise ValueError(f"extrinsics.rotation must have 4 elements (xyzw), got {len(ro)}")
        return cls(
            translation=(float(tr[0]), float(tr[1]), float(tr[2])),
            rotation=(float(ro[0]), float(ro[1]), float(ro[2]), float(ro[3])),
        )

    def to_matrix(self) -> np.ndarray:
        """Return a 4×4 row-major cam-to-scene transform."""
        x, y, z, w = self.rotation
        n = float(x * x + y * y + z * z + w * w)
        if n < 1e-12:
            raise ValueError("extrinsics.rotation has near-zero norm")
        s = 2.0 / n
        xx, yy, zz = x * x * s, y * y * s, z * z * s
        xy, xz, yz = x * y * s, x * z * s, y * z * s
        wx, wy, wz = w * x * s, w * y * s, w * z * s
        m = np.eye(4, dtype=np.float64)
        m[0, 0] = 1.0 - (yy + zz)
        m[0, 1] = xy - wz
        m[0, 2] = xz + wy
        m[1, 0] = xy + wz
        m[1, 1] = 1.0 - (xx + zz)
        m[1, 2] = yz - wx
        m[2, 0] = xz - wy
        m[2, 1] = yz + wx
        m[2, 2] = 1.0 - (xx + yy)
        m[:3, 3] = self.translation
        return m

    @classmethod
    def from_matrix(cls, m: np.ndarray) -> CameraExtrinsics:
        """Build an extrinsics from a 4×4 row-major cam-to-scene transform.

        The rotation component must be approximately orthonormal; the
        translation is taken from the last column.
        """
        a = np.asarray(m, dtype=np.float64).reshape(4, 4)
        t = a[:3, 3]
        x, y, z, w = quat_from_rotation_matrix(a[:3, :3])
        return cls(
            translation=(float(t[0]), float(t[1]), float(t[2])),
            rotation=(x, y, z, w),
        )


@dataclass
class Camera:
    """A single camera observation: intrinsics + extrinsics + optional metadata."""

    name: str
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
    timestamp_us: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": str(self.name),
            "intrinsics": self.intrinsics.to_dict(),
            "extrinsics": self.extrinsics.to_dict(),
            "metadata": dict(self.metadata),
        }
        if self.timestamp_us is not None:
            d["timestamp_us"] = int(self.timestamp_us)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Camera:
        return cls(
            name=str(d["name"]),
            intrinsics=CameraIntrinsics.from_dict(d["intrinsics"]),
            extrinsics=CameraExtrinsics.from_dict(d["extrinsics"]),
            timestamp_us=int(d["timestamp_us"]) if d.get("timestamp_us") is not None else None,
            metadata=dict(d.get("metadata") or {}),
        )


# ---------------------------------------------------------------------------
# Collection-level (de)serialisation
# ---------------------------------------------------------------------------


def serialize_cameras(cameras: list[Camera]) -> dict[str, Any]:
    """Build the JSON-ready ``splatsim.cameras/v1`` document for ``cameras``."""
    seen: set[str] = set()
    out_list: list[dict[str, Any]] = []
    for cam in cameras:
        if cam.name in seen:
            raise ValueError(f"duplicate camera name: {cam.name!r}")
        seen.add(cam.name)
        out_list.append(cam.to_dict())
    return {
        "schema": CAMERAS_SCHEMA,
        "frame": _FRAME,
        "cameras": out_list,
    }


def parse_cameras(doc: dict[str, Any]) -> list[Camera]:
    """Inverse of :func:`serialize_cameras`. Accepts a parsed JSON object.

    Rejects duplicate camera names so the load path enforces the same
    invariant as :func:`serialize_cameras` (downstream tooling keys by
    name and silently dropping a duplicate would be a surprise).
    """
    schema = doc.get("schema")
    if schema != CAMERAS_SCHEMA:
        raise ValueError(f"unexpected cameras schema {schema!r}; expected {CAMERAS_SCHEMA!r}")
    raw = doc.get("cameras")
    if not isinstance(raw, list):
        raise ValueError("cameras document is missing the 'cameras' list")
    out: list[Camera] = []
    seen: set[str] = set()
    for entry in raw:
        cam = Camera.from_dict(entry)
        if cam.name in seen:
            raise ValueError(f"duplicate camera name: {cam.name!r}")
        seen.add(cam.name)
        out.append(cam)
    return out
