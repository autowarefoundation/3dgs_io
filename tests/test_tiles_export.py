"""Tests for the 3D Tiles exporter."""

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
save_gltf = _mod.save_gltf
TilesetSaveOptions = _mod.TilesetSaveOptions
GltfSaveOptions = _mod.GltfSaveOptions
Tile3DContent = _mod.Tile3DContent
BoundingVolumeBox = _mod.BoundingVolumeBox
BoundingVolumeSphere = _mod.BoundingVolumeSphere
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


class TestSaveFromCloud:
    """Tests for save_tileset with a GaussianCloud source."""

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

    def test_sh_degree_1_roundtrip(self, tmp_path: Path) -> None:
        """SH degree 1 data must survive chunked tileset save/load."""
        n = 200
        gc = _make_cloud(n, spread=10.0)
        rng = np.random.default_rng(99)
        gc.sh_degree = 1
        # SH degree 1: 3 coefficient groups × 3 channels = 9 values per point
        gc.sh = rng.standard_normal(n * 9).astype(np.float32)

        out = tmp_path / "tiles"
        save_tileset(gc, out, TilesetSaveOptions(chunk_size=5.0))

        total = 0
        for glb in sorted(out.glob("chunk_*.glb")):
            chunk_gc = load_gltf(glb)
            total += chunk_gc.num_points
            assert chunk_gc.sh_degree == 1
            sh = np.array(chunk_gc.sh, dtype=np.float32)
            assert sh.size == chunk_gc.num_points * 9
        assert total == n

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


class TestSaveFromTileset:
    """Tests for save_tileset with a tileset path source (streaming re-chunk)."""

    def _save_source(self, tmp_path: Path, n: int = 200, chunk_size: float = 100.0) -> Path:
        """Create a source tileset via save_tileset for re-chunking."""
        gc = _make_cloud(n, spread=10.0)
        src = tmp_path / "source"
        save_tileset(gc, src, TilesetSaveOptions(chunk_size=chunk_size))
        return src / "tileset.json"

    def test_produces_tileset_json(self, tmp_path: Path) -> None:
        source = self._save_source(tmp_path)
        out = tmp_path / "rechunked"
        result = save_tileset(source, out)
        assert result.name == "tileset.json"
        assert result.exists()
        tileset = json.loads(result.read_text())
        assert tileset["asset"]["version"] == "1.1"
        assert "root" in tileset
        assert len(tileset["root"]["children"]) >= 1

    def test_all_points_preserved(self, tmp_path: Path) -> None:
        n = 300
        source = self._save_source(tmp_path, n=n, chunk_size=100.0)
        out = tmp_path / "rechunked"
        save_tileset(source, out, TilesetSaveOptions(chunk_size=5.0))

        total = 0
        for glb in sorted(out.glob("chunk_*.glb")):
            chunk_gc = load_gltf(glb)
            total += chunk_gc.num_points
        assert total == n

    def test_rechunk_splits_further(self, tmp_path: Path) -> None:
        """Re-chunking with a smaller grid should produce more tiles."""
        source = self._save_source(tmp_path, n=200, chunk_size=100.0)

        out = tmp_path / "rechunked"
        save_tileset(source, out, TilesetSaveOptions(chunk_size=5.0))
        glb_files = list(out.glob("chunk_*.glb"))
        assert len(glb_files) >= 2

    def test_large_chunk_size_merges(self, tmp_path: Path) -> None:
        """Re-chunking with a huge grid should merge everything into one tile."""
        source = self._save_source(tmp_path, n=200, chunk_size=5.0)

        out = tmp_path / "rechunked"
        save_tileset(source, out, TilesetSaveOptions(chunk_size=1000.0))
        glb_files = list(out.glob("chunk_*.glb"))
        assert len(glb_files) == 1

    def test_roundtrip_via_load_tileset(self, tmp_path: Path) -> None:
        n = 200
        source = self._save_source(tmp_path, n=n, chunk_size=100.0)
        out = tmp_path / "rechunked"
        tileset_path = save_tileset(source, out, TilesetSaveOptions(chunk_size=8.0))

        tiles = load_tileset(tileset_path)
        total = sum(t.cloud.num_points for t in tiles)
        assert total == n

    def test_bounding_volumes_valid(self, tmp_path: Path) -> None:
        source = self._save_source(tmp_path, n=200)
        out = tmp_path / "rechunked"
        save_tileset(source, out, TilesetSaveOptions(chunk_size=5.0))
        tileset = json.loads((out / "tileset.json").read_text())
        root_box = tileset["root"]["boundingVolume"]["box"]
        assert len(root_box) == 12
        assert root_box[3] >= 0
        assert root_box[7] >= 0
        assert root_box[11] >= 0
        for child in tileset["root"]["children"]:
            child_box = child["boundingVolume"]["box"]
            assert len(child_box) == 12

    def test_with_transform(self, tmp_path: Path) -> None:
        """Source tileset with a root transform should apply it to positions."""
        gc = _make_cloud(50, spread=5.0)
        src = tmp_path / "source"
        src.mkdir()
        save_gltf(gc, src / "0.glb")

        tileset = {
            "asset": {"version": "1.1"},
            "geometricError": 10.0,
            "root": {
                "boundingVolume": {"box": [50, 50, 50, 55, 0, 0, 0, 55, 0, 0, 0, 55]},
                "geometricError": 0.0,
                "refine": "ADD",
                "transform": [
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    100,
                    100,
                    100,
                    1,
                ],
                "content": {"uri": "0.glb"},
            },
        }
        tileset_path = src / "tileset.json"
        tileset_path.write_text(json.dumps(tileset))

        out = tmp_path / "rechunked"
        save_tileset(tileset_path, out, TilesetSaveOptions(chunk_size=1000.0))

        # All points should be shifted by (100, 100, 100)
        rechunked_gc = load_gltf(next(out.glob("chunk_*.glb")))
        pos = np.array(rechunked_gc.positions, dtype=np.float32).reshape(-1, 3)
        # Original positions are in [-5, 5], shifted to [95, 105]
        assert pos.min() > 90.0
        assert pos.max() < 110.0

    def test_creates_output_directory(self, tmp_path: Path) -> None:
        source = self._save_source(tmp_path, n=50)
        out = tmp_path / "nested" / "deep" / "tiles"
        save_tileset(source, out)
        assert (out / "tileset.json").exists()

    def test_spz_compression_forwarded(self, tmp_path: Path) -> None:
        source = self._save_source(tmp_path, n=100)
        out = tmp_path / "rechunked"
        opts = TilesetSaveOptions(
            save_options=GltfSaveOptions(spz_compression=True),
        )
        save_tileset(source, out, opts)
        glb = next(out.glob("chunk_*.glb"))
        chunk_gc = load_gltf(glb)
        assert chunk_gc.num_points > 0


class TestSaveFromTiles:
    """Tests for save_tileset with a list[Tile3DContent] source."""

    @staticmethod
    def _make_tile(n: int = 50, seed: int = 42, spread: float = 5.0) -> Tile3DContent:
        gc = _make_cloud(n, seed=seed, spread=spread)
        positions = np.array(gc.positions, dtype=np.float32).reshape(n, 3)
        bmin = positions.min(axis=0).astype(np.float64)
        bmax = positions.max(axis=0).astype(np.float64)
        center = (bmin + bmax) / 2
        half = (bmax - bmin) / 2
        half_axes = np.diag(half)
        return Tile3DContent(
            cloud=gc,
            transform=np.eye(4, dtype=np.float64).ravel(),
            content_uri="dummy.glb",
            bounding_volume=BoundingVolumeBox(center=center, half_axes=half_axes),
        )

    def test_produces_valid_tileset_json(self, tmp_path: Path) -> None:
        tiles = [self._make_tile(seed=i) for i in range(3)]
        out = tmp_path / "tiles"
        result = save_tileset(tiles, out)
        assert result.name == "tileset.json"
        assert result.exists()
        tileset = json.loads(result.read_text())
        assert tileset["asset"]["version"] == "1.1"
        assert len(tileset["root"]["children"]) == 3

    def test_all_points_preserved(self, tmp_path: Path) -> None:
        n1, n2 = 60, 80
        tiles = [self._make_tile(n=n1, seed=1), self._make_tile(n=n2, seed=2)]
        out = tmp_path / "tiles"
        save_tileset(tiles, out)

        total = 0
        for glb in sorted(out.glob("tile_*.glb")):
            total += load_gltf(glb).num_points
        assert total == n1 + n2

    def test_bounding_volumes_preserved(self, tmp_path: Path) -> None:
        tile = self._make_tile()
        out = tmp_path / "tiles"
        save_tileset([tile], out)
        tileset = json.loads((out / "tileset.json").read_text())
        child_box = tileset["root"]["children"][0]["boundingVolume"]["box"]
        assert len(child_box) == 12
        expected = np.concatenate(
            [tile.bounding_volume.center, tile.bounding_volume.half_axes.ravel()]
        )
        np.testing.assert_allclose(child_box, expected.tolist(), atol=1e-10)

    def test_root_bounding_volume_is_union(self, tmp_path: Path) -> None:
        t1 = self._make_tile(n=30, seed=1, spread=5.0)
        t2 = self._make_tile(n=30, seed=2, spread=5.0)
        out = tmp_path / "tiles"
        save_tileset([t1, t2], out)
        tileset = json.loads((out / "tileset.json").read_text())
        root_box = tileset["root"]["boundingVolume"]["box"]
        assert len(root_box) == 12
        # Root must enclose both children
        assert root_box[3] >= 0  # hx
        assert root_box[7] >= 0  # hy
        assert root_box[11] >= 0  # hz

    def test_root_transform_written(self, tmp_path: Path) -> None:
        tile = self._make_tile()
        # column-major translation (100, 200, 300)
        transform = np.eye(4, dtype=np.float64)
        transform[3, 0] = 100.0
        transform[3, 1] = 200.0
        transform[3, 2] = 300.0
        out = tmp_path / "tiles"
        save_tileset([tile], out, root_transform=transform)
        tileset = json.loads((out / "tileset.json").read_text())
        got = tileset["root"]["transform"]
        assert len(got) == 16
        assert got[12] == 100.0
        assert got[13] == 200.0
        assert got[14] == 300.0

    def test_tile_transform_written(self, tmp_path: Path) -> None:
        gc = _make_cloud(30, seed=10)
        # column-major translation (10, 20, 30)
        transform = np.eye(4, dtype=np.float64)
        transform[3, 0] = 10.0
        transform[3, 1] = 20.0
        transform[3, 2] = 30.0
        tile = Tile3DContent(
            cloud=gc,
            transform=transform.ravel(),
            content_uri="dummy.glb",
            geometric_error=5.0,
            bounding_volume=BoundingVolumeBox(center=np.zeros(3), half_axes=np.eye(3) * 5.0),
        )
        out = tmp_path / "tiles"
        save_tileset([tile], out)
        tileset = json.loads((out / "tileset.json").read_text())
        child = tileset["root"]["children"][0]
        assert child["geometricError"] == 5.0
        got = child["transform"]
        assert len(got) == 16
        assert got[12] == 10.0
        assert got[13] == 20.0
        assert got[14] == 30.0

    def test_identity_transform_omitted(self, tmp_path: Path) -> None:
        tile = self._make_tile()
        out = tmp_path / "tiles"
        save_tileset([tile], out)
        tileset = json.loads((out / "tileset.json").read_text())
        assert "transform" not in tileset["root"]["children"][0]

    def test_no_root_transform_by_default(self, tmp_path: Path) -> None:
        tile = self._make_tile()
        out = tmp_path / "tiles"
        save_tileset([tile], out)
        tileset = json.loads((out / "tileset.json").read_text())
        assert "transform" not in tileset["root"]

    def test_empty_list_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="empty"):
            save_tileset([], tmp_path / "tiles")

    def test_roundtrip_load_save_load(self, tmp_path: Path) -> None:
        """load_tileset -> save_tileset -> load_tileset produces equivalent data."""
        # Step 1: create a source tileset from a cloud
        gc = _make_cloud(200, spread=10.0)
        src = tmp_path / "source"
        save_tileset(gc, src, TilesetSaveOptions(chunk_size=8.0))

        # Step 2: load it
        tiles = load_tileset(src / "tileset.json")
        n_original = sum(t.cloud.num_points for t in tiles)

        # Step 3: save from loaded tiles
        out = tmp_path / "roundtrip"
        save_tileset(tiles, out)

        # Step 4: load again
        tiles2 = load_tileset(out / "tileset.json")
        n_roundtrip = sum(t.cloud.num_points for t in tiles2)
        assert n_roundtrip == n_original

    def test_spz_compression_forwarded(self, tmp_path: Path) -> None:
        tile = self._make_tile()
        out = tmp_path / "tiles"
        opts = TilesetSaveOptions(
            save_options=GltfSaveOptions(spz_compression=True),
        )
        save_tileset([tile], out, opts)
        tileset = json.loads((out / "tileset.json").read_text())
        exts = tileset["extensions"]["3DTILES_content_gltf"]["extensionsUsed"]
        assert "KHR_gaussian_splatting_compression_spz_2" in exts
        glb = next(out.glob("tile_*.glb"))
        assert load_gltf(glb).num_points > 0

    def test_sphere_bounding_volume(self, tmp_path: Path) -> None:
        gc = _make_cloud(30, seed=99)
        tile = Tile3DContent(
            cloud=gc,
            transform=np.eye(4, dtype=np.float64).ravel(),
            content_uri="dummy.glb",
            bounding_volume=BoundingVolumeSphere(center=np.array([1.0, 2.0, 3.0]), radius=10.0),
        )
        out = tmp_path / "tiles"
        save_tileset([tile], out)
        tileset = json.loads((out / "tileset.json").read_text())
        child_bv = tileset["root"]["children"][0]["boundingVolume"]
        assert "sphere" in child_bv
        assert child_bv["sphere"] == [1.0, 2.0, 3.0, 10.0]

    def test_no_bounding_volume_fallback(self, tmp_path: Path) -> None:
        """Tiles without bounding_volume get an AABB computed from positions."""
        gc = _make_cloud(30, seed=55)
        tile = Tile3DContent(
            cloud=gc,
            transform=np.eye(4, dtype=np.float64).ravel(),
            content_uri="dummy.glb",
            bounding_volume=None,
        )
        out = tmp_path / "tiles"
        save_tileset([tile], out)
        tileset = json.loads((out / "tileset.json").read_text())
        child_box = tileset["root"]["children"][0]["boundingVolume"]["box"]
        assert len(child_box) == 12
