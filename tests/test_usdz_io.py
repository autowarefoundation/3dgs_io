"""Tests for alpasim USDZ I/O."""

from __future__ import annotations

import importlib
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

_mod = importlib.import_module("3dgs_io")
AlpasimGaussianCloud = _mod.AlpasimGaussianCloud
AlpasimSkyCubemap = _mod.AlpasimSkyCubemap
load_usdz = _mod.load_usdz
save_usdz = _mod.save_usdz


def _make_cloud(n: int = 32) -> AlpasimGaussianCloud:
    rng = np.random.default_rng(0)
    positions = rng.standard_normal((n, 3)).astype(np.float16)
    rotations = rng.standard_normal((n, 4)).astype(np.float16)
    scales = rng.standard_normal((n, 3)).astype(np.float16)
    densities = rng.standard_normal((n, 1)).astype(np.float16)
    features_albedo = rng.standard_normal((n, 5, 3)).astype(np.float16)
    features_specular = rng.standard_normal((n, 45)).astype(np.float16)
    camera_extra = rng.standard_normal((n, 20)).astype(np.float16)
    invisible = rng.integers(0, 1000, size=(n,), dtype=np.int32)
    sky = AlpasimSkyCubemap(
        textures=rng.standard_normal((1, 6, 4, 4, 3)).astype(np.float16),
        texture_grads=rng.standard_normal((6, 2, 2)).astype(np.float16),
        n_grad_updates=42,
    )
    return AlpasimGaussianCloud(
        positions=positions,
        rotations=rotations,
        scales=scales,
        densities=densities,
        features_albedo=features_albedo,
        features_specular=features_specular,
        camera_extra_signal=camera_extra,
        n_active_features=3,
        timestamps_us_min=27563309000,
        timestamps_us_max=27583309000,
        invisible_steps=invisible,
        sky=sky,
        nre_offset=(-1.0, 2.0, 3.5),
    )


def test_roundtrip_bytewise_equal(tmp_path: Path) -> None:
    cloud = _make_cloud()
    out = tmp_path / "scene.usdz"
    save_usdz(cloud, out)
    loaded = load_usdz(out)

    assert loaded.num_points == cloud.num_points
    np.testing.assert_array_equal(loaded.positions, cloud.positions)
    np.testing.assert_array_equal(loaded.rotations, cloud.rotations)
    np.testing.assert_array_equal(loaded.scales, cloud.scales)
    np.testing.assert_array_equal(loaded.densities, cloud.densities)
    np.testing.assert_array_equal(loaded.features_albedo, cloud.features_albedo)
    np.testing.assert_array_equal(loaded.features_specular, cloud.features_specular)
    np.testing.assert_array_equal(loaded.camera_extra_signal, cloud.camera_extra_signal)
    np.testing.assert_array_equal(loaded.invisible_steps, cloud.invisible_steps)
    assert loaded.sky is not None
    np.testing.assert_array_equal(loaded.sky.textures, cloud.sky.textures)
    np.testing.assert_array_equal(loaded.sky.texture_grads, cloud.sky.texture_grads)
    assert loaded.sky.n_grad_updates == cloud.sky.n_grad_updates

    assert loaded.timestamps_us_min == cloud.timestamps_us_min
    assert loaded.timestamps_us_max == cloud.timestamps_us_max
    assert loaded.n_active_features == cloud.n_active_features
    assert loaded.nre_offset == pytest.approx(cloud.nre_offset)


def test_saved_archive_contains_required_files(tmp_path: Path) -> None:
    cloud = _make_cloud()
    out = tmp_path / "scene.usdz"
    save_usdz(cloud, out)
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
    assert {"default.usda", "volume.usda", "volume.nurec"} <= names


def test_save_without_sky_writes_placeholder(tmp_path: Path) -> None:
    cloud = _make_cloud()
    cloud.sky = None
    out = tmp_path / "no_sky.usdz"
    save_usdz(cloud, out)
    loaded = load_usdz(out)
    assert loaded.sky is not None  # placeholder is written
    assert loaded.sky.textures.shape == (1, 6, 1, 1, 3)


def test_load_missing_volume_nurec_raises(tmp_path: Path) -> None:
    out = tmp_path / "bad.usdz"
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr("default.usda", "#usda 1.0\n")
    with pytest.raises(ValueError, match="volume.nurec"):
        load_usdz(out)


_SAMPLE = Path.home() / "Downloads" / "00040136-e651-4abd-991d-0655ccda9430.usdz"


@pytest.mark.skipif(
    not _SAMPLE.exists() or os.environ.get("SKIP_USDZ_SAMPLE") == "1",
    reason="alpasim sample USDZ not available",
)
def test_load_sample_usdz() -> None:
    cloud = load_usdz(_SAMPLE)
    assert cloud.num_points > 0
    assert cloud.positions.shape == (cloud.num_points, 3)
    assert cloud.features_albedo.shape[-1] == 3
    assert cloud.features_specular.shape[1] % 3 == 0
    assert cloud.sky is not None
    assert cloud.version
