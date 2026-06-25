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

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

__all__ = [
    "Camera",
    "CameraExtrinsics",
    "CameraModel",
]


# ---------------------------------------------------------------------------
# Intrinsics
# ---------------------------------------------------------------------------


# Required intrinsic-parameter keys per camera-model ``type``. Construction of
# a :class:`CameraModel` raises ``ValueError`` if any of these keys are missing
# so that intrinsics can never be silently dropped on the way in/out of disk.
# Unknown ``type`` strings still require ``resolution`` — the bare minimum any
# downstream consumer needs to interpret the image data.
_REQUIRED_INTRINSIC_KEYS: dict[str, tuple[str, ...]] = {
    "pinhole": ("resolution", "fx", "fy", "cx", "cy"),
    "opencv": ("resolution", "fx", "fy", "cx", "cy", "distortion_coeffs"),
    "ftheta": (
        "resolution",
        "principal_point",
        "pixeldist_to_angle_poly",
        "angle_to_pixeldist_poly",
    ),
}

# Optional (non-required) intrinsic-parameter keys per camera-model ``type``.
# Together with :data:`_REQUIRED_INTRINSIC_KEYS` these define the full set of
# keys :meth:`CameraModel.with_intrinsics` will accept.
_OPTIONAL_INTRINSIC_KEYS: dict[str, tuple[str, ...]] = {
    "ftheta": ("shutter_type", "reference_poly", "external_distortion_parameters"),
}


def _coerce_xy_pair(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"principal_point must be a 2-element list/tuple; got {value!r}")
    return [float(value[0]), float(value[1])]


# Per-key coercion for :meth:`CameraModel.with_intrinsics`. Keys not listed
# here are stored verbatim (e.g. ``external_distortion_parameters``).
_INTRINSIC_COERCERS: dict[str, Any] = {
    "fx": float,
    "fy": float,
    "cx": float,
    "cy": float,
    "distortion_coeffs": lambda v: [float(c) for c in v],
    "pixeldist_to_angle_poly": lambda v: [float(c) for c in v],
    "angle_to_pixeldist_poly": lambda v: [float(c) for c in v],
    "principal_point": _coerce_xy_pair,
    "shutter_type": str,
    "reference_poly": str,
}


@dataclass
class CameraModel:
    """Type-tagged camera intrinsics.

    ``parameters`` is required and must contain the intrinsic keys listed in
    the module docstring for the given ``type``; missing keys raise
    :class:`ValueError` at construction time.
    """

    type: str
    parameters: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, dict):
            raise ValueError(
                f"camera_model.parameters must be a dict; got {type(self.parameters).__name__}"
            )
        required = _REQUIRED_INTRINSIC_KEYS.get(self.type, ("resolution",))
        missing = [k for k in required if k not in self.parameters]
        if missing:
            raise ValueError(
                f"camera_model(type={self.type!r}) is missing required intrinsic "
                f"key(s) {missing}; got keys {sorted(self.parameters)}"
            )
        # Eagerly validate the resolution shape so a malformed value can't
        # slip past construction and surface later as a confusing error.
        _ = self.resolution

    # ------------ resolution accessors ------------

    @property
    def resolution(self) -> tuple[int, int]:
        res = self.parameters.get("resolution")
        # Reject scalars / strings explicitly — len() on a string would be
        # accepted by a naive `len(res) != 2` check.
        if not isinstance(res, (list, tuple)) or len(res) != 2:
            raise ValueError(
                f"camera_model.parameters.resolution must be a 2-element list/tuple; got {res!r}"
            )
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

    # ------------ intrinsic edits ------------

    def with_intrinsics(self, **updates: Any) -> CameraModel:
        """Return a copy of this model with the given intrinsic fields replaced.

        Accepts ``width`` / ``height`` (mapped jointly to ``resolution``) plus
        any parameter key valid for the model's ``type``. Coercion mirrors the
        per-type constructors (``pinhole`` / ``opencv`` / ``ftheta``) so a value
        coming in as ``int`` lands as ``float`` for ``fx``/``fy`` etc. Unknown
        keys for the model's ``type`` raise ``ValueError`` rather than silently
        attaching to ``parameters``. The returned model fully owns its
        ``parameters`` (nested lists/dicts are deep-copied) so mutating it
        cannot affect the original.
        """
        new_params: dict[str, Any] = copy.deepcopy(self.parameters)
        width = updates.pop("width", None)
        height = updates.pop("height", None)
        if width is not None or height is not None:
            cur_w, cur_h = self.resolution
            new_params["resolution"] = [
                int(width if width is not None else cur_w),
                int(height if height is not None else cur_h),
            ]

        allowed = set(_REQUIRED_INTRINSIC_KEYS.get(self.type, ("resolution",)))
        allowed |= set(_OPTIONAL_INTRINSIC_KEYS.get(self.type, ()))
        unknown = sorted(set(updates) - allowed)
        if unknown:
            raise ValueError(
                f"camera_model(type={self.type!r}) does not accept intrinsic key(s) "
                f"{unknown}; allowed keys (excluding width/height): {sorted(allowed)}"
            )

        for key, value in updates.items():
            coercer = _INTRINSIC_COERCERS.get(key)
            new_params[key] = coercer(value) if coercer is not None else value

        return CameraModel(type=self.type, parameters=new_params)

    # ------------ (de)serialisation ------------

    def to_dict(self) -> dict[str, Any]:
        return {"type": str(self.type), "parameters": dict(self.parameters)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CameraModel:
        missing = [k for k in ("type", "parameters") if k not in d]
        if missing:
            raise ValueError(
                f"camera_model dict is missing required key(s) {missing}; got keys {sorted(d)}"
            )
        raw_params = d["parameters"]
        # Reject non-dict (incl. JSON ``null``) here so the error surfaces as
        # a ValueError matching __post_init__, not the bare TypeError from
        # ``dict(None)``.
        if not isinstance(raw_params, dict):
            raise ValueError(
                f"camera_model.parameters must be a dict; got {type(raw_params).__name__}"
            )
        return cls(type=str(d["type"]), parameters=dict(raw_params))


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
        x, y, z, w = Rotation.from_matrix(a[:3, :3]).as_quat()
        return cls(
            translation=(float(t[0]), float(t[1]), float(t[2])),
            rotation=(float(x), float(y), float(z), float(w)),
        )

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
        missing = {k for k in ("name", "T_sensor_rig", "camera_model") if k not in d}
        if missing:
            raise ValueError(
                f"camera entry is missing required key(s) {sorted(missing)}; got keys {sorted(d)}"
            )
        return cls(
            name=str(d["name"]),
            camera_model=CameraModel.from_dict(d["camera_model"]),
            extrinsics=CameraExtrinsics.from_t_sensor_rig(d["T_sensor_rig"]),
            metadata=dict(d.get("metadata") or {}),
        )
