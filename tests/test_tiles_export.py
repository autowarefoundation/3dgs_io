"""Tests for the 3D Tiles exporter with spatial chunk splitting."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import spz

_mod = importlib.import_module("3dgs_io")
save_tileset = _mod.save_tileset
load_tileset = _mod.load_tileset
load_gltf = _mod.load_gltf
TilesetSaveOptions = _mod.TilesetSaveOptions
GltfSaveOptions = _mod.GltfSaveOptions
GaussianCloud = spz.GaussianCloud

_SH_C0 = 0.2820947917738781


def _make_cloud(n: int = 200, seed: int = 42, spread: float = 10.0) -> GaussianCloud:
    """Create a random GaussianCloud for testing."""
    rng = np.random.default_rng(seed)
    gc = GaussianCloud()
    gc.positions = rng.uniform(-spread, spread, (n, 3)).astype(np.float32).reshape(-1)
    rgb = rng.integers(0, 256, (n, 3), dtype=np.uint8)
    gc.colors = (((rgb.astype(np.float32) / 255.0) - 0.5) / _SH_C0).reshape(-1)
    alpha_01 = np.clip(rng.uniform(0.01, 0.99, n), 1e-7, 1 - 1e-7)
    gc.alphas = np.log(alpha_01 / (1 - alpha_01)).astype(np.float32)
    rots = rng.standard_normal((n, 4)).astype(np.float32)
    rots /= np.linalg.norm(rots, axis=1, keepdims=True)
    gc.rotations = rots.reshape(-1)
    gc.scales = rng.standard_normal((n, 3)).astype(np.float32).reshape(-1)
    gc.sh = np.zeros(0, dtype=np.float32)
    return gc


class TestSaveTileset:
    def test_produces_tileset_json(self, tmp_path: Path) -> None:
        gc = _make_cloud(100)
        result = save_tileset(gc, tmp_path / "tiles")
        assert result.name == "tileset.json"
        assert result.exists()
        tileset = json.loads(result.read_text())
        assert tileset["asset"]["version"] == "1.1"
        assert "root" in tileset
        assert len(tileset["root"]["children"]) >= 1

    def test_small_chunk_size_produces_many_tiles(self, tmp_path: Path) -> None:
        """spread=10 → bbox ~20m per axis. chunk_size=5 → up to 4^3=64 cells."""
        gc = _make_cloud(200, spread=10.0)
        out = tmp_path / "tiles"
        save_tileset(gc, out, TilesetSaveOptions(chunk_size=5.0))
        glb_files = list(out.glob("chunk_*.glb"))
        assert len(glb_files) >= 2

    def test_all_points_preserved(self, tmp_path: Path) -> None:
        n = 500
        gc = _make_cloud(n, spread=10.0)
        out = tmp_path / "tiles"
        save_tileset(gc, out, TilesetSaveOptions(chunk_size=5.0))

        total = 0
        for glb in sorted(out.glob("chunk_*.glb")):
            chunk_gc = load_gltf(glb)
            total += chunk_gc.num_points
        assert total == n

    def test_large_chunk_size_produces_single_tile(self, tmp_path: Path) -> None:
        """chunk_size larger than the entire bbox → 1 tile."""
        gc = _make_cloud(50, spread=5.0)
        out = tmp_path / "tiles"
        save_tileset(gc, out, TilesetSaveOptions(chunk_size=1000.0))
        glb_files = list(out.glob("chunk_*.glb"))
        assert len(glb_files) == 1

    def test_bounding_volumes_valid(self, tmp_path: Path) -> None:
        gc = _make_cloud(200)
        out = tmp_path / "tiles"
        save_tileset(gc, out, TilesetSaveOptions(chunk_size=5.0))
        tileset = json.loads((out / "tileset.json").read_text())
        root_box = tileset["root"]["boundingVolume"]["box"]
        assert len(root_box) == 12
        assert root_box[3] >= 0  # hx
        assert root_box[7] >= 0  # hy
        assert root_box[11] >= 0  # hz
        for child in tileset["root"]["children"]:
            child_box = child["boundingVolume"]["box"]
            assert len(child_box) == 12

    def test_empty_cloud_raises(self, tmp_path: Path) -> None:
        gc = GaussianCloud()
        with pytest.raises(ValueError, match="empty"):
            save_tileset(gc, tmp_path / "tiles")

    def test_invalid_chunk_size_raises(self, tmp_path: Path) -> None:
        gc = _make_cloud(50)
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            save_tileset(gc, tmp_path / "tiles", TilesetSaveOptions(chunk_size=0.0))

    def test_spz_compression_forwarded(self, tmp_path: Path) -> None:
        gc = _make_cloud(100)
        out = tmp_path / "tiles"
        opts = TilesetSaveOptions(
            save_options=GltfSaveOptions(spz_compression=True),
        )
        save_tileset(gc, out, opts)
        glb = next(out.glob("chunk_*.glb"))
        chunk_gc = load_gltf(glb)
        assert chunk_gc.num_points > 0

    def test_roundtrip_via_load_tileset(self, tmp_path: Path) -> None:
        n = 300
        gc = _make_cloud(n, spread=10.0)
        out = tmp_path / "tiles"
        tileset_path = save_tileset(gc, out, TilesetSaveOptions(chunk_size=8.0))

        tiles = load_tileset(tileset_path)
        total = sum(t.cloud.num_points for t in tiles)
        assert total == n

    def test_creates_output_directory(self, tmp_path: Path) -> None:
        gc = _make_cloud(50)
        out = tmp_path / "nested" / "deep" / "tiles"
        save_tileset(gc, out)
        assert (out / "tileset.json").exists()
