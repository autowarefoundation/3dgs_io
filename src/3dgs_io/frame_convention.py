"""The single coordinate convention used by scene bundles."""

from __future__ import annotations

from typing import Any

import numpy as np

FRAME_CONVENTION: dict[str, Any] = {
    "world": {"handedness": "right", "up_axis": "z", "type": "ENU"},
    "rig": {
        "forward": "+x",
        "left": "+y",
        "up": "+z",
        "origin": "ground_under_rear_axle",
    },
    "camera": {"model_axes": "opencv_+z_forward"},
    "lidar": {"axes": "+x_forward_+y_left_+z_up"},
    "quaternion_order": "xyzw",
    "matrix_layout": "row_major",
    "vector_convention": "column_vector_child_to_parent",
    "length_units": "meters",
    "time_units": "microseconds",
}

# Internal glTF/SPZ RUB (right, up, back) -> alpasim ENU/FLU
# (forward, left, up). This is a proper rotation: det(R) == +1.
RUB_TO_ENU = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def validate_frame_convention(value: Any) -> None:
    """Reject documents whose coordinate contract differs from ours."""
    if value != FRAME_CONVENTION:
        raise ValueError("unsupported frame_convention; expected the alpasim-native ENU contract")


def validate_rotation(rotation: Any, *, where: str) -> None:
    q = np.asarray(rotation, dtype=np.float64)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError(f"{where} must be a finite xyzw quaternion")
    norm = float(np.linalg.norm(q))
    if not np.isclose(norm, 1.0, atol=1e-5):
        raise ValueError(f"{where} must be unit-norm; got {norm:.8g}")


def validate_timestamps(values: list[int], *, where: str) -> None:
    limit = 2**64
    for timestamp in values:
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, np.integer))
            or not 0 <= timestamp < limit
        ):
            raise ValueError(f"{where} timestamps must be u64 microseconds")
    if any(b <= a for a, b in zip(values, values[1:], strict=False)):
        raise ValueError(f"{where} timestamps must be strictly increasing")


def validate_rigid_transform(matrix: Any, *, where: str) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (4, 4) or not np.all(np.isfinite(m)):
        raise ValueError(f"{where} must be a finite 4x4 matrix")
    if not np.allclose(m[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{where} must be an affine transform")
    r = m[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-5):
        raise ValueError(f"{where} rotation must be orthonormal")
    det = float(np.linalg.det(r))
    if not np.isclose(det, 1.0, atol=1e-5):
        raise ValueError(f"{where} rotation must have det(R)=+1; got {det:.8g}")
    return m
