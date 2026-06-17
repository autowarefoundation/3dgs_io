"""Tests for USDZ I/O (spz.GaussianCloud ↔ USDZ archive)."""

from __future__ import annotations

import importlib
import zipfile
from pathlib import Path

import numpy as np
import pytest
import spz

_mod = importlib.import_module("3dgs_io")
load_usdz = _mod.load_usdz
save_usdz = _mod.save_usdz


def _make_cloud(n: int = 64, sh_degree: int = 3) -> spz.GaussianCloud:
    rng = np.random.default_rng(0)
    gc = spz.GaussianCloud()
    gc.antialiased = False
    gc.positions = rng.standard_normal(n * 3).astype(np.float32)
    quats = rng.standard_normal((n, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    gc.rotations = quats.reshape(-1)
    gc.scales = rng.uniform(-3.0, 0.5, size=n * 3).astype(np.float32)
    gc.alphas = rng.standard_normal(n).astype(np.float32)
    gc.colors = rng.uniform(0.0, 1.0, size=n * 3).astype(np.float32)
    per_ch = (sh_degree + 1) ** 2 - 1
    if per_ch > 0:
        gc.sh_degree = sh_degree
        gc.sh = rng.standard_normal(n * per_ch * 3).astype(np.float32)
    else:
        gc.sh_degree = 0
        gc.sh = np.zeros(0, dtype=np.float32)
    return gc


def test_roundtrip_preserves_num_points_and_shapes(tmp_path: Path) -> None:
    gc = _make_cloud(n=128)
    out = tmp_path / "scene.usdz"
    save_usdz(gc, out)
    loaded = load_usdz(out)

    assert loaded.num_points == gc.num_points
    assert loaded.sh_degree == gc.sh_degree
    assert loaded.positions.shape == gc.positions.shape
    assert loaded.rotations.shape == gc.rotations.shape
    assert loaded.scales.shape == gc.scales.shape
    assert loaded.colors.shape == gc.colors.shape
    assert loaded.alphas.shape == gc.alphas.shape
    assert loaded.sh.shape == gc.sh.shape


def test_roundtrip_values_close(tmp_path: Path) -> None:
    """spz applies lossy quantisation; check values survive within the codec's tolerance."""
    gc = _make_cloud(n=64)
    out = tmp_path / "scene.usdz"
    save_usdz(gc, out)
    loaded = load_usdz(out)

    np.testing.assert_allclose(loaded.positions, gc.positions, atol=1e-3)
    # quaternions: spz packs to fixed point; allow loose tolerance and check unit-norm.
    qs = np.asarray(loaded.rotations).reshape(-1, 4)
    np.testing.assert_allclose(np.linalg.norm(qs, axis=1), 1.0, atol=1e-2)


def test_archive_contents(tmp_path: Path) -> None:
    gc = _make_cloud(n=16)
    out = tmp_path / "scene.usdz"
    save_usdz(gc, out)
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
    assert names == {"default.usda", "model.spz"}


def test_load_missing_model_raises(tmp_path: Path) -> None:
    out = tmp_path / "bad.usdz"
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr("default.usda", "#usda 1.0\n")
    with pytest.raises(ValueError, match="model.spz"):
        load_usdz(out)


def test_save_creates_parent_dir(tmp_path: Path) -> None:
    gc = _make_cloud(n=8)
    out = tmp_path / "nested" / "subdir" / "scene.usdz"
    save_usdz(gc, out)
    assert out.is_file()
    loaded = load_usdz(out)
    assert loaded.num_points == 8
