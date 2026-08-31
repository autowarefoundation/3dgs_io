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
    rub_to_enu = np.eye(4)
    rub_to_enu[:3, :3] = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    cesium_root = np.asarray(result.ecef_anchor) @ rub_to_enu
    assert document["root"]["transform"] == cesium_root.T.ravel().tolist()
    assert (
        "KHR_gaussian_splatting"
        in document["extensions"]["3DTILES_content_gltf"]["extensionsRequired"]
    )
    assert document["root"]["children"]
    assert all(child["content"]["uri"].endswith(".glb") for child in document["root"]["children"])

    total = 0
    gltf_positions = []
    for child in document["root"]["children"]:
        cloud = _mod.load_gltf(output / child["content"]["uri"])
        total += cloud.num_points
        gltf_positions.append(np.asarray(cloud.positions).reshape(-1, 3))
    assert total == result.n_gaussians

    world_positions = []
    spz_io = importlib.import_module("3dgs_io.spz_io")
    with zipfile.ZipFile(usdz) as archive:
        for index, name in enumerate(name for name in archive.namelist() if name.endswith(".spz")):
            chunk = tmp_path / f"world_{index}.spz"
            chunk.write_bytes(archive.read(name))
            world_positions.append(
                np.asarray(spz_io.load_spz_world(chunk).positions).reshape(-1, 3)
            )
    world_centroid = np.concatenate(world_positions).mean(axis=0)
    gltf_centroid = np.concatenate(gltf_positions).mean(axis=0)
    world_ecef = np.asarray(result.ecef_anchor) @ np.r_[world_centroid, 1.0]
    gltf_ecef = cesium_root @ np.r_[gltf_centroid, 1.0]
    np.testing.assert_allclose(gltf_ecef, world_ecef, atol=1e-3)


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
