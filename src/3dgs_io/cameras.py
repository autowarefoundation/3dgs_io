"""Camera intrinsics + extrinsics dataclasses, nested inside a :class:`RigTrajectory`.

A :class:`Camera` is mounted on a sensor rig and lives under
:attr:`RigTrajectory.cameras`. Its layout matches alpasim's
``rig_trajectories.json.camera_calibrations`` schema closely:

* :class:`CameraModel` — type-tagged intrinsics. Supported ``type`` values:

  - ``"pinhole"`` — ``parameters = {resolution, fx, fy, cx, cy}``
  - ``"opencv"``  — ``parameters = {resolution, fx, fy, cx, cy, distortion_coeffs}``
  - ``"ftheta"``  — NVIDIA polynomial fisheye (alpasim default). ``parameters``
    keys: ``resolution``, ``principal_point``,
    ``pixeldist_to_angle_poly``, ``angle_to_pixeldist_poly``, ``shutter_type``,
    ``reference_poly``, ``external_distortion_parameters``.

* :class:`CameraExtrinsics` — **sensor-to-rig** 4×4 rigid transform
  (``T_sensor_rig``). Stored in Python as translation + xyzw quaternion,
  serialised as the alpasim-style flat 4×4 list of lists.

To compute a camera's pose in scene coordinates, compose the camera
extrinsics with the rig's pose at the relevant timestamp::

    T_sensor_scene = T_rig_scene @ T_sensor_rig
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._quat import quat_from_rotation_matrix

__all__ = [
    "Camera",
    "CameraExtrinsics",
    "CameraModel",
]


# ---------------------------------------------------------------------------
# Intrinsics
# ---------------------------------------------------------------------------


@dataclass
class CameraModel:
    """Type-tagged camera intrinsics."""

    type: str
    parameters: dict[str, Any] = field(default_factory=dict)

    # ------------ resolution accessors ------------

    @property
    def resolution(self) -> tuple[int, int]:
        res = self.parameters.get("resolution")
        if res is None or len(res) != 2:
            raise ValueError(f"camera_model.parameters.resolution missing or malformed: {res!r}")
        return int(res[0]), int(res[1])

    @property
    def width(self) -> int:
        return self.resolution[0]

    @property
    def height(self) -> int:
        return self.resolution[1]

    # ------------ constructors ------------

    @classmethod
    def pinhole(
        cls,
        *,
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> CameraModel:
        return cls(
            type="pinhole",
            parameters={
                "resolution": [int(width), int(height)],
                "fx": float(fx),
                "fy": float(fy),
                "cx": float(cx),
                "cy": float(cy),
            },
        )

    @classmethod
    def opencv(
        cls,
        *,
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        distortion_coeffs: list[float],
    ) -> CameraModel:
        return cls(
            type="opencv",
            parameters={
                "resolution": [int(width), int(height)],
                "fx": float(fx),
                "fy": float(fy),
                "cx": float(cx),
                "cy": float(cy),
                "distortion_coeffs": [float(c) for c in distortion_coeffs],
            },
        )

    @classmethod
    def ftheta(
        cls,
        *,
        width: int,
        height: int,
        principal_point: tuple[float, float],
        pixeldist_to_angle_poly: list[float],
        angle_to_pixeldist_poly: list[float],
        shutter_type: str = "ROLLING_TOP_TO_BOTTOM",
        reference_poly: str = "PIXELDIST_TO_ANGLE",
        external_distortion_parameters: Any = None,
    ) -> CameraModel:
        return cls(
            type="ftheta",
            parameters={
                "resolution": [int(width), int(height)],
                "principal_point": [float(principal_point[0]), float(principal_point[1])],
                "pixeldist_to_angle_poly": [float(c) for c in pixeldist_to_angle_poly],
                "angle_to_pixeldist_poly": [float(c) for c in angle_to_pixeldist_poly],
                "shutter_type": str(shutter_type),
                "reference_poly": str(reference_poly),
                "external_distortion_parameters": external_distortion_parameters,
            },
        )

    # ------------ (de)serialisation ------------

    def to_dict(self) -> dict[str, Any]:
        return {"type": str(self.type), "parameters": dict(self.parameters)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CameraModel:
        return cls(type=str(d["type"]), parameters=dict(d.get("parameters") or {}))


# ---------------------------------------------------------------------------
# Extrinsics (sensor-to-rig)
# ---------------------------------------------------------------------------


@dataclass
class CameraExtrinsics:
    """Sensor-to-rig rigid pose.

    Stored as a translation triple + xyzw unit quaternion (rig-relative).
    Serialises as ``{"T_sensor_rig": <4x4 row-major nested list>}`` to match
    alpasim's ``rig_trajectories.json.camera_calibrations[*].T_sensor_rig``
    layout.
    """

    translation: tuple[float, float, float]
    rotation: tuple[float, float, float, float]  # xyzw, unit-norm

    # ------------ matrix conversions ------------

    def to_matrix(self) -> np.ndarray:
        """Return the 4×4 row-major sensor-to-rig transform."""
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
        """Build an extrinsics from a 4×4 row-major sensor-to-rig transform."""
        a = np.asarray(m, dtype=np.float64).reshape(4, 4)
        t = a[:3, 3]
        x, y, z, w = quat_from_rotation_matrix(a[:3, :3])
        return cls(translation=(float(t[0]), float(t[1]), float(t[2])), rotation=(x, y, z, w))

    # ------------ (de)serialisation ------------

    def to_t_sensor_rig(self) -> list[list[float]]:
        """Return the 4×4 as a nested list of Python floats (JSON-safe)."""
        m = self.to_matrix()
        return [[float(m[i, j]) for j in range(4)] for i in range(4)]

    @classmethod
    def from_t_sensor_rig(cls, mat: Any) -> CameraExtrinsics:
        return cls.from_matrix(np.asarray(mat, dtype=np.float64))


# ---------------------------------------------------------------------------
# Camera (mounted on a rig)
# ---------------------------------------------------------------------------


@dataclass
class Camera:
    """A camera mounted on a sensor rig (peer to :class:`RigPose` under a rig)."""

    name: str
    camera_model: CameraModel
    extrinsics: CameraExtrinsics
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "T_sensor_rig": self.extrinsics.to_t_sensor_rig(),
            "camera_model": self.camera_model.to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Camera:
        return cls(
            name=str(d["name"]),
            camera_model=CameraModel.from_dict(d["camera_model"]),
            extrinsics=CameraExtrinsics.from_t_sensor_rig(d["T_sensor_rig"]),
            metadata=dict(d.get("metadata") or {}),
        )
