"""Tests for compute_bounding_volume."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import spz

_mod = importlib.import_module("3dgs_io")
compute_bounding_volume = _mod.compute_bounding_volume


def _make_cloud(positions: np.ndarray) -> spz.GaussianCloud:
    """Create a minimal GaussianCloud with the given (N, 3) positions."""
    n = positions.shape[0]
    gc = spz.GaussianCloud()
    gc.positions = positions.astype(np.float32).reshape(-1)
    gc.rotations = np.tile([0.0, 0.0, 0.0, 1.0], n).astype(np.float32)
    gc.scales = np.zeros(n * 3, dtype=np.float32)
    gc.colors = np.zeros(n * 3, dtype=np.float32)
    gc.alphas = np.zeros(n, dtype=np.float32)
    gc.sh = np.zeros(0, dtype=np.float32)
    return gc


class TestComputeBoundingVolume:
    def test_basic_aabb(self) -> None:
        positions = np.array([[0.0, 0.0, 0.0], [10.0, 20.0, 30.0]])
        gc = _make_cloud(positions)
        result = compute_bounding_volume(gc)

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

    def test_with_camera_positions_expands_aabb(self) -> None:
        positions = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        gc = _make_cloud(positions)
        cameras = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
        result = compute_bounding_volume(gc, camera_positions=cameras)

        box = result["box"]
        # AABB should expand to [0,5] in all axes → center=2.5, half=2.5
        assert box[0] == pytest.approx(2.5)
        assert box[1] == pytest.approx(2.5)
        assert box[2] == pytest.approx(2.5)
        assert box[3] == pytest.approx(2.5)
        assert box[7] == pytest.approx(2.5)
        assert box[11] == pytest.approx(2.5)

    def test_camera_positions_inside_aabb_no_change(self) -> None:
        positions = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
        gc = _make_cloud(positions)
        cameras = np.array([[3.0, 3.0, 3.0], [7.0, 7.0, 7.0]])
        result_with = compute_bounding_volume(gc, camera_positions=cameras)
        result_without = compute_bounding_volume(gc)

        assert result_with["box"] == pytest.approx(result_without["box"])

    def test_single_point(self) -> None:
        positions = np.array([[5.0, 10.0, 15.0]])
        gc = _make_cloud(positions)
        result = compute_bounding_volume(gc)

        box = result["box"]
        # center = (5, 10, 15), half-extents = (0, 0, 0)
        assert box[0] == pytest.approx(5.0)
        assert box[1] == pytest.approx(10.0)
        assert box[2] == pytest.approx(15.0)
        assert box[3] == pytest.approx(0.0)
        assert box[7] == pytest.approx(0.0)
        assert box[11] == pytest.approx(0.0)

    def test_empty_cloud_raises(self) -> None:
        gc = spz.GaussianCloud()
        with pytest.raises(ValueError, match="empty"):
            compute_bounding_volume(gc)

    def test_empty_camera_positions_ignored(self) -> None:
        positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        gc = _make_cloud(positions)
        cameras = np.zeros((0, 3), dtype=np.float64)
        result = compute_bounding_volume(gc, camera_positions=cameras)
        result_no_cam = compute_bounding_volume(gc)

        assert result["box"] == pytest.approx(result_no_cam["box"])

    def test_invalid_camera_positions_shape(self) -> None:
        positions = np.array([[1.0, 2.0, 3.0]])
        gc = _make_cloud(positions)
        with pytest.raises(ValueError, match="camera_positions must be"):
            compute_bounding_volume(gc, camera_positions=np.array([1.0, 2.0, 3.0]))
