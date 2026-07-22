from __future__ import annotations

import importlib
import json
import zipfile
from pathlib import Path

import numpy as np

_mod = importlib.import_module("3dgs_io")


def test_export_usdz_tileset_preserves_anchor_and_gaussians(
    tmp_path: Path, make_minimal_tileset_with_glb
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_tileset = make_minimal_tileset_with_glb(source_dir)
    usdz = tmp_path / "scene.usdz"
    result = _mod.save_scene_usdz(source_tileset, usdz)

    output = tmp_path / "cesium"
    tileset_path = _mod.export_usdz_tileset(usdz, output)
    document = json.loads(tileset_path.read_text())

    assert document["asset"]["version"] == "1.1"
    assert document["root"]["transform"] == result.root_transform
    assert (
        "KHR_gaussian_splatting"
        in document["extensions"]["3DTILES_content_gltf"]["extensionsRequired"]
    )
    assert document["root"]["children"]
    assert all(child["content"]["uri"].endswith(".glb") for child in document["root"]["children"])

    total = 0
    for child in document["root"]["children"]:
        total += _mod.load_gltf(output / child["content"]["uri"]).num_points
    assert total == result.n_gaussians


def test_export_usdz_tileset_rejects_wrong_frame_contract(
    tmp_path: Path, make_minimal_tileset_with_glb
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_tileset = make_minimal_tileset_with_glb(source_dir)
    usdz = tmp_path / "scene.usdz"
    _mod.save_scene_usdz(source_tileset, usdz)

    broken = tmp_path / "broken.usdz"
    with zipfile.ZipFile(usdz) as source, zipfile.ZipFile(broken, "w") as target:
        for info in source.infolist():
            data = source.read(info.filename)
            if info.filename == "scene.json":
                scene = json.loads(data)
                scene["world"]["frame_convention"]["world"]["up_axis"] = "y"
                data = json.dumps(scene).encode()
            target.writestr(info, data)

    try:
        _mod.export_usdz_tileset(broken, tmp_path / "out")
    except ValueError as error:
        assert "frame_convention" in str(error)
    else:
        raise AssertionError("invalid frame convention was accepted")


def test_save_scene_rejects_reflective_root_transform(
    tmp_path: Path, make_minimal_tileset_with_glb
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    tileset_path = make_minimal_tileset_with_glb(source_dir)
    document = json.loads(tileset_path.read_text())
    reflection = np.eye(4)
    reflection[0, 0] = -1.0
    document["root"]["transform"] = reflection.T.ravel().tolist()
    tileset_path.write_text(json.dumps(document))

    try:
        _mod.save_scene_usdz(tileset_path, tmp_path / "scene.usdz")
    except ValueError as error:
        assert "det(R)=+1" in str(error)
    else:
        raise AssertionError("reflective root transform was accepted")
