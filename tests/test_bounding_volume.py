"""Tests for compute_bounding_volume."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

_mod = importlib.import_module("3dgs_io")
compute_bounding_volume = _mod.compute_bounding_volume


class TestComputeBoundingVolume:
    def test_basic_aabb(self) -> None:
        cameras = np.array([[0.0, 0.0, 0.0], [10.0, 20.0, 30.0]])
        result = compute_bounding_volume(cameras)

        assert "box" in result
        box = result["box"]
        assert len(box) == 12
        # center = (5, 10, 15), half-extents = (5, 10, 15)
        assert box[0] == pytest.approx(5.0)
        assert box[1] == pytest.approx(10.0)
        assert box[2] == pytest.approx(15.0)
        assert box[3] == pytest.approx(5.0)
        assert box[7] == pytest.approx(10.0)
        assert box[11] == pytest.approx(15.0)
        # off-diagonal zeros
        assert box[4] == 0.0
        assert box[5] == 0.0
        assert box[6] == 0.0
        assert box[8] == 0.0
        assert box[9] == 0.0
        assert box[10] == 0.0

    def test_single_camera(self) -> None:
        cameras = np.array([[5.0, 10.0, 15.0]])
        result = compute_bounding_volume(cameras)

        box = result["box"]
        # center = (5, 10, 15), half-extents = (0, 0, 0)
        assert box[0] == pytest.approx(5.0)
        assert box[1] == pytest.approx(10.0)
        assert box[2] == pytest.approx(15.0)
        assert box[3] == pytest.approx(0.0)
        assert box[7] == pytest.approx(0.0)
        assert box[11] == pytest.approx(0.0)

    def test_many_cameras(self) -> None:
        cameras = np.array(
            [
                [-1.0, -2.0, -3.0],
                [1.0, 2.0, 3.0],
                [0.5, 0.5, 0.5],
            ]
        )
        result = compute_bounding_volume(cameras)

        box = result["box"]
        # AABB: [-1,1] x [-2,2] x [-3,3] → center=(0,0,0), half=(1,2,3)
        assert box[0] == pytest.approx(0.0)
        assert box[1] == pytest.approx(0.0)
        assert box[2] == pytest.approx(0.0)
        assert box[3] == pytest.approx(1.0)
        assert box[7] == pytest.approx(2.0)
        assert box[11] == pytest.approx(3.0)

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
