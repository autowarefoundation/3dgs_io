"""Tests for compute_bounding_volume."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

_mod = importlib.import_module("3dgs_io")
compute_bounding_volume = _mod.compute_bounding_volume
BoundingVolumeBox = _mod.BoundingVolumeBox


class TestComputeBoundingVolume:
    def test_returns_bounding_volume_box(self) -> None:
        cameras = np.array([[0.0, 0.0, 0.0], [10.0, 20.0, 30.0]])
        result = compute_bounding_volume(cameras)
        assert isinstance(result, BoundingVolumeBox)

    def test_basic_aabb(self) -> None:
        cameras = np.array([[0.0, 0.0, 0.0], [10.0, 20.0, 30.0]])
        bv = compute_bounding_volume(cameras)

        # center = (5, 10, 15)
        np.testing.assert_allclose(bv.center, [5.0, 10.0, 15.0])
        # half_axes = diag(5, 10, 15)
        np.testing.assert_allclose(bv.half_axes, np.diag([5.0, 10.0, 15.0]))

    def test_single_camera(self) -> None:
        cameras = np.array([[5.0, 10.0, 15.0]])
        bv = compute_bounding_volume(cameras)

        np.testing.assert_allclose(bv.center, [5.0, 10.0, 15.0])
        np.testing.assert_allclose(bv.half_axes, np.diag([0.0, 0.0, 0.0]))

    def test_many_cameras(self) -> None:
        cameras = np.array(
            [
                [-1.0, -2.0, -3.0],
                [1.0, 2.0, 3.0],
                [0.5, 0.5, 0.5],
            ]
        )
        bv = compute_bounding_volume(cameras)

        # AABB: [-1,1] x [-2,2] x [-3,3]
        np.testing.assert_allclose(bv.center, [0.0, 0.0, 0.0])
        np.testing.assert_allclose(bv.half_axes, np.diag([1.0, 2.0, 3.0]))

    def test_to_dict_roundtrip(self) -> None:
        cameras = np.array([[0.0, 0.0, 0.0], [10.0, 20.0, 30.0]])
        bv = compute_bounding_volume(cameras)
        d = bv.to_dict()

        assert "box" in d
        box = d["box"]
        assert len(box) == 12
        assert box[0] == pytest.approx(5.0)
        assert box[1] == pytest.approx(10.0)
        assert box[2] == pytest.approx(15.0)

    def test_empty_raises(self) -> None:
        cameras = np.zeros((0, 3), dtype=np.float64)
        with pytest.raises(ValueError, match="empty"):
            compute_bounding_volume(cameras)

    def test_invalid_shape_1d(self) -> None:
        with pytest.raises(ValueError, match="camera_positions must be"):
            compute_bounding_volume(np.array([1.0, 2.0, 3.0]))

    def test_invalid_shape_wrong_columns(self) -> None:
        with pytest.raises(ValueError, match="camera_positions must be"):
            compute_bounding_volume(np.array([[1.0, 2.0]]))
