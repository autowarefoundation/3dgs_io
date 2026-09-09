"""Shared pytest fixtures."""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import spz


@pytest.fixture
def make_minimal_tileset_with_glb() -> Callable[[Path], Path]:
    """Return a factory that materialises ``model.glb`` + ``tileset.json`` in a directory.

    The fixture is used by both ``test_cameras.py`` and ``test_tracks.py`` (and
    is available to any other test that needs a trivially-valid tileset.json
    pointing at an in-memory GLB).
    """

    def _build(tmp_path: Path) -> Path:
        mod = importlib.import_module("3dgs_io")
        save_gltf = mod.save_gltf

        n = 32
        rng = np.random.default_rng(0)
        gc = spz.GaussianCloud()
        gc.antialiased = False
        gc.positions = rng.uniform(-10, 10, size=n * 3).astype(np.float32)
        quats = rng.standard_normal((n, 4)).astype(np.float32)
        quats /= np.linalg.norm(quats, axis=1, keepdims=True)
        gc.rotations = quats.reshape(-1)
        gc.scales = rng.uniform(-3, 0, size=n * 3).astype(np.float32)
        gc.alphas = rng.standard_normal(n).astype(np.float32)
        gc.colors = rng.uniform(0, 1, size=n * 3).astype(np.float32)
        gc.sh_degree = 0
        gc.sh = np.zeros(0, dtype=np.float32)
        save_gltf(gc, tmp_path / "model.glb")

        doc = {
            "asset": {"version": "1.1"},
            "geometricError": 100.0,
            "root": {
                "boundingVolume": {"box": [0, 0, 0, 100, 0, 0, 0, 100, 0, 0, 0, 100]},
                "geometricError": 0,
                "refine": "ADD",
                "content": {"uri": "model.glb"},
            },
        }
        tp = tmp_path / "tileset.json"
        tp.write_text(json.dumps(doc))
        return tp

    return _build
