"""Tests for the Camera / CameraModel / CameraExtrinsics dataclasses.

Cameras live as children of a :class:`RigTrajectory` and are exercised via
that path; see ``test_rig_trajectories.py`` for the ``save_scene_usdz``
integration tests. This file covers only the standalone dataclass behaviour.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest

_mod = importlib.import_module("3dgs_io")
Camera = _mod.Camera
CameraExtrinsics = _mod.CameraExtrinsics
CameraModel = _mod.CameraModel


# ---------------------------------------------------------------------------
# CameraModel
# ---------------------------------------------------------------------------


def test_pinhole_constructor() -> None:
    m = CameraModel.pinhole(width=1920, height=1080, fx=500, fy=500, cx=960, cy=540)
    assert m.type == "pinhole"
    assert m.resolution == (1920, 1080)
    assert m.width == 1920
    assert m.height == 1080
    assert m.parameters["fx"] == 500.0
    assert m.parameters["fy"] == 500.0


def test_opencv_constructor_carries_distortion_coeffs() -> None:
    m = CameraModel.opencv(
        width=1920,
        height=1080,
        fx=1234.5,
        fy=1234.5,
        cx=960.0,
        cy=540.0,
        distortion_coeffs=[0.01, -0.002, 0.0, 0.0, 0.0],
    )
    assert m.type == "opencv"
    assert m.parameters["distortion_coeffs"] == [0.01, -0.002, 0.0, 0.0, 0.0]


def test_ftheta_constructor() -> None:
    m = CameraModel.ftheta(
        width=1920,
        height=1080,
        principal_point=(961.3, 744.8),
        pixeldist_to_angle_poly=[0.0, 1e-3, 4e-9],
        angle_to_pixeldist_poly=[0.0, 938.5, -1.3],
    )
    assert m.type == "ftheta"
    p = m.parameters
    assert p["resolution"] == [1920, 1080]
    assert p["principal_point"] == [961.3, 744.8]
    assert p["pixeldist_to_angle_poly"] == [0.0, 1e-3, 4e-9]
    assert p["shutter_type"] == "ROLLING_TOP_TO_BOTTOM"
    assert p["reference_poly"] == "PIXELDIST_TO_ANGLE"


def test_model_dict_roundtrip() -> None:
    m = CameraModel.pinhole(width=640, height=480, fx=500, fy=500, cx=320, cy=240)
    again = CameraModel.from_dict(m.to_dict())
    assert again.type == "pinhole"
    assert again.parameters == m.parameters


def test_construction_missing_intrinsics_raises() -> None:
    with pytest.raises(ValueError, match="missing required intrinsic key"):
        CameraModel(type="pinhole", parameters={})


def test_pinhole_partial_intrinsics_raises() -> None:
    with pytest.raises(ValueError, match=r"missing required intrinsic key.*'fy'"):
        CameraModel(
            type="pinhole",
            parameters={"resolution": [1920, 1080], "fx": 1000, "cx": 960, "cy": 540},
        )


def test_opencv_missing_distortion_raises() -> None:
    with pytest.raises(ValueError, match="distortion_coeffs"):
        CameraModel(
            type="opencv",
            parameters={
                "resolution": [1920, 1080],
                "fx": 1000,
                "fy": 1000,
                "cx": 960,
                "cy": 540,
            },
        )


def test_unknown_type_still_requires_resolution() -> None:
    with pytest.raises(ValueError, match="resolution"):
        CameraModel(type="custom_fisheye", parameters={"foo": 1})


def test_resolution_rejects_string() -> None:
    """A string value would pass a naive ``len() != 2`` check; reject it explicitly."""
    with pytest.raises(ValueError, match="2-element list/tuple"):
        CameraModel(
            type="pinhole",
            parameters={
                "resolution": "1920x1080",
                "fx": 1000,
                "fy": 1000,
                "cx": 960,
                "cy": 540,
            },
        )


def test_from_dict_missing_parameters_raises() -> None:
    with pytest.raises(ValueError, match=r"missing required key.*'parameters'"):
        CameraModel.from_dict({"type": "pinhole"})


def test_from_dict_null_parameters_raises_valueerror() -> None:
    # JSON ``null`` decodes to None; this used to crash with TypeError from
    # ``dict(None)`` instead of the documented ValueError.
    with pytest.raises(ValueError, match="parameters must be a dict"):
        CameraModel.from_dict({"type": "pinhole", "parameters": None})


# ---------------------------------------------------------------------------
# CameraExtrinsics — T_sensor_rig form
# ---------------------------------------------------------------------------


def test_extrinsics_identity_round_trip() -> None:
    e = CameraExtrinsics(translation=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0))
    np.testing.assert_allclose(e.to_matrix(), np.eye(4))
    t_sensor_rig = e.to_t_sensor_rig()
    assert t_sensor_rig == [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def test_extrinsics_translation_only() -> None:
    e = CameraExtrinsics(translation=(5.0, 6.0, 7.0), rotation=(0.0, 0.0, 0.0, 1.0))
    t = e.to_t_sensor_rig()
    assert t[0][3] == 5.0
    assert t[1][3] == 6.0
    assert t[2][3] == 7.0


def test_extrinsics_z_rotation_90deg_round_trip() -> None:
    # 90° about +Z; xyzw quaternion = (0, 0, sin45, cos45)
    qz, qw = float(np.sin(np.pi / 4)), float(np.cos(np.pi / 4))
    e = CameraExtrinsics(translation=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, qz, qw))
    m = e.to_matrix()
    np.testing.assert_allclose(m @ [1, 0, 0, 1], [0, 1, 0, 1], atol=1e-10)
    back = CameraExtrinsics.from_matrix(m)
    np.testing.assert_allclose(back.to_matrix(), m, atol=1e-10)


def test_extrinsics_from_t_sensor_rig_helper() -> None:
    mat = [
        [1.0, 0.0, 0.0, 10.0],
        [0.0, 1.0, 0.0, 20.0],
        [0.0, 0.0, 1.0, 30.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    e = CameraExtrinsics.from_t_sensor_rig(mat)
    assert e.translation == (10.0, 20.0, 30.0)
    np.testing.assert_allclose(e.rotation, (0.0, 0.0, 0.0, 1.0), atol=1e-10)


def test_extrinsics_zero_norm_quaternion_raises() -> None:
    e = CameraExtrinsics(translation=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="near-zero norm"):
        e.to_matrix()


# ---------------------------------------------------------------------------
# Camera (full entry)
# ---------------------------------------------------------------------------


def _camera_dict() -> dict[str, Any]:
    return {
        "name": "front_left",
        "sensor_in_rig": {
            "translation": [1.5, 0.0, 1.8],
            "rotation": [0.0, 0.0, 0.0, 1.0],
        },
        "camera_model": {
            "type": "pinhole",
            "parameters": {
                "resolution": [1920, 1080],
                "fx": 1000,
                "fy": 1000,
                "cx": 960,
                "cy": 540,
            },
        },
        "metadata": {"sequence_id": "test_seq"},
    }


def test_camera_round_trip() -> None:
    d = _camera_dict()
    cam = Camera.from_dict(d)
    again = cam.to_dict()
    assert again["name"] == "front_left"
    assert again["sensor_in_rig"] == d["sensor_in_rig"]
    assert again["calibration_resolution"] == [1920, 1080]
    assert again["camera_model"]["type"] == "pinhole"
    assert again["camera_model"]["parameters"]["fx"] == 1000.0
    assert again["metadata"] == {"sequence_id": "test_seq"}


def test_camera_from_dict_default_metadata() -> None:
    d = _camera_dict()
    del d["metadata"]
    cam = Camera.from_dict(d)
    assert cam.metadata == {}


def test_camera_from_dict_missing_keys_raises_clearly() -> None:
    d = _camera_dict()
    del d["sensor_in_rig"]
    with pytest.raises(ValueError, match=r"missing required key.*sensor_in_rig"):
        Camera.from_dict(d)


# ---------------------------------------------------------------------------
# CameraModel.with_intrinsics
# ---------------------------------------------------------------------------


def test_with_intrinsics_pinhole_updates_focal_and_principal() -> None:
    m = CameraModel.pinhole(width=1920, height=1080, fx=500, fy=500, cx=960, cy=540)
    out = m.with_intrinsics(fx=1234.5, cx=961.0)
    # Original is untouched.
    assert m.parameters["fx"] == 500.0
    assert m.parameters["cx"] == 960.0
    # New instance carries the updates with float coercion.
    assert out.type == "pinhole"
    assert out.parameters["fx"] == 1234.5
    assert out.parameters["fy"] == 500.0
    assert out.parameters["cx"] == 961.0
    assert out.parameters["cy"] == 540.0
    assert out.resolution == (1920, 1080)


def test_with_intrinsics_width_height_remap_resolution() -> None:
    m = CameraModel.pinhole(width=1920, height=1080, fx=500, fy=500, cx=960, cy=540)
    out_w = m.with_intrinsics(width=3840)
    assert out_w.resolution == (3840, 1080)
    out_both = m.with_intrinsics(width=640, height=480)
    assert out_both.resolution == (640, 480)


def test_with_intrinsics_opencv_replaces_distortion() -> None:
    m = CameraModel.opencv(
        width=1920,
        height=1080,
        fx=1234.5,
        fy=1234.5,
        cx=960,
        cy=540,
        distortion_coeffs=[0.01, -0.002, 0.0, 0.0, 0.0],
    )
    out = m.with_intrinsics(distortion_coeffs=[0.05, -0.01, 0.0, 0.0])
    assert out.parameters["distortion_coeffs"] == [0.05, -0.01, 0.0, 0.0]


def test_with_intrinsics_rejects_keys_outside_model_type() -> None:
    m = CameraModel.pinhole(width=1920, height=1080, fx=500, fy=500, cx=960, cy=540)
    with pytest.raises(ValueError, match="does not accept intrinsic key"):
        m.with_intrinsics(distortion_coeffs=[0.0])
    with pytest.raises(ValueError, match="does not accept intrinsic key"):
        m.with_intrinsics(principal_point=(1.0, 2.0))


def test_with_intrinsics_ftheta_updates_principal_point_and_polys() -> None:
    m = CameraModel.ftheta(
        width=1920,
        height=1080,
        principal_point=(961.3, 744.8),
        pixeldist_to_angle_poly=[0.0, 1e-3, 4e-9],
        angle_to_pixeldist_poly=[0.0, 938.5, -1.3],
    )
    out = m.with_intrinsics(
        principal_point=(960.0, 740.0),
        pixeldist_to_angle_poly=[0.0, 1.5e-3],
        shutter_type="GLOBAL",
    )
    assert out.parameters["principal_point"] == [960.0, 740.0]
    assert out.parameters["pixeldist_to_angle_poly"] == [0.0, 1.5e-3]
    assert out.parameters["shutter_type"] == "GLOBAL"


def test_with_intrinsics_principal_point_must_be_length_2() -> None:
    m = CameraModel.ftheta(
        width=1920,
        height=1080,
        principal_point=(961.3, 744.8),
        pixeldist_to_angle_poly=[0.0, 1e-3],
        angle_to_pixeldist_poly=[0.0, 938.5],
    )
    with pytest.raises(ValueError, match="principal_point must be a 2-element"):
        m.with_intrinsics(principal_point=(1.0, 2.0, 3.0))


def test_with_intrinsics_principal_point_rejects_non_sequence() -> None:
    m = CameraModel.ftheta(
        width=1920,
        height=1080,
        principal_point=(961.3, 744.8),
        pixeldist_to_angle_poly=[0.0, 1e-3],
        angle_to_pixeldist_poly=[0.0, 938.5],
    )
    with pytest.raises(ValueError, match="principal_point must be a 2-element"):
        m.with_intrinsics(principal_point=42)


def test_with_intrinsics_returns_deep_copy_of_distortion_coeffs() -> None:
    m = CameraModel.opencv(
        width=1920,
        height=1080,
        fx=1234.5,
        fy=1234.5,
        cx=960,
        cy=540,
        distortion_coeffs=[0.01, -0.002, 0.0, 0.0, 0.0],
    )
    # Update only fx; distortion_coeffs are unchanged but must NOT alias the
    # original list (mutating one mustn't mutate the other).
    out = m.with_intrinsics(fx=1500.0)
    out.parameters["distortion_coeffs"].append(0.999)
    assert m.parameters["distortion_coeffs"] == [0.01, -0.002, 0.0, 0.0, 0.0]
