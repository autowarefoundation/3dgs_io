"""Tests for ``python -m 3dgs_io.cameras_cli``."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest

_mod = importlib.import_module("3dgs_io")
Camera = _mod.Camera
CameraExtrinsics = _mod.CameraExtrinsics
CameraModel = _mod.CameraModel
RigPose = _mod.RigPose
RigTrajectory = _mod.RigTrajectory
serialize_rig_trajectories = _mod.serialize_rig_trajectories

_cli = importlib.import_module("3dgs_io.cameras_cli")


def _rig_doc(tmp_path: Path, *, model: CameraModel | None = None) -> Path:
    """Write a minimal splatsim.rig_trajectories/v1 doc and return its path."""
    if model is None:
        model = CameraModel.pinhole(width=1920, height=1080, fx=500, fy=500, cx=960, cy=540)
    rig = RigTrajectory(
        rig_id="ego",
        poses=[RigPose(timestamp_us=0, translation=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0))],
        cameras=[
            Camera(
                name="front",
                camera_model=model,
                extrinsics=CameraExtrinsics(
                    translation=(0.0, 0.0, 0.0),
                    rotation=(0.0, 0.0, 0.0, 1.0),
                ),
            )
        ],
    )
    doc = serialize_rig_trajectories([rig])
    p = tmp_path / "rig_trajectories.json"
    p.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return p


def _read_camera(path: Path) -> dict[str, Any]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    return doc["rigs"][0]["cameras"][0]["camera_model"]["parameters"]


def test_cli_updates_focal_and_resolution(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    src = _rig_doc(tmp_path)
    out = tmp_path / "out.json"
    rc = _cli.main(
        [
            "--input",
            str(src),
            "--output",
            str(out),
            "--camera",
            "front",
            "--fx",
            "1234.5",
            "--width",
            "3840",
            "--height",
            "2160",
            "--quiet",
        ]
    )
    assert rc == 0
    params = _read_camera(out)
    assert params["fx"] == 1234.5
    assert params["fy"] == 500.0
    assert params["resolution"] == [3840, 2160]


def test_cli_emits_summary_to_stdout(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    src = _rig_doc(tmp_path)
    out = tmp_path / "out.json"
    rc = _cli.main(["--input", str(src), "--output", str(out), "--camera", "front", "--cx", "961"])
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["camera"] == "front"
    assert summary["updated_fields"] == ["cx"]
    assert summary["camera_model"]["parameters"]["cx"] == 961.0


def test_cli_requires_at_least_one_update(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    src = _rig_doc(tmp_path)
    out = tmp_path / "out.json"
    rc = _cli.main(["--input", str(src), "--output", str(out), "--camera", "front"])
    assert rc == 2
    assert "no intrinsic updates" in capsys.readouterr().err


def test_cli_opencv_distortion_coeffs(tmp_path: Path) -> None:
    src = _rig_doc(
        tmp_path,
        model=CameraModel.opencv(
            width=1920,
            height=1080,
            fx=1000,
            fy=1000,
            cx=960,
            cy=540,
            distortion_coeffs=[0.0, 0.0, 0.0, 0.0, 0.0],
        ),
    )
    out = tmp_path / "out.json"
    rc = _cli.main(
        [
            "--input",
            str(src),
            "--output",
            str(out),
            "--camera",
            "front",
            "--distortion-coeffs",
            "0.05,-0.01,0.0,0.0",
            "--quiet",
        ]
    )
    assert rc == 0
    params = _read_camera(out)
    assert params["distortion_coeffs"] == [0.05, -0.01, 0.0, 0.0]


def test_cli_unknown_camera_raises(tmp_path: Path) -> None:
    src = _rig_doc(tmp_path)
    out = tmp_path / "out.json"
    with pytest.raises(ValueError, match="camera 'rear' not found"):
        _cli.main(
            [
                "--input",
                str(src),
                "--output",
                str(out),
                "--camera",
                "rear",
                "--fx",
                "1.0",
                "--quiet",
            ]
        )


def test_cli_accepts_alpasim_input_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The CLI should auto-detect alpasim, log a warning, and apply the edit."""
    alpasim_doc: dict[str, Any] = {
        "T_world_base": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "world_to_nre": {
            "matrix": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        },
        "rig_trajectories": [
            {
                "sequence_id": "seq0",
                "T_rig_worlds": [
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ],
                "T_rig_world_timestamps_us": [0],
            }
        ],
        "camera_calibrations": {
            "front": {
                "camera_model": CameraModel.pinhole(
                    width=1920, height=1080, fx=500, fy=500, cx=960, cy=540
                ).to_dict(),
                "T_sensor_rig": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        },
    }
    src = tmp_path / "alpasim.json"
    src.write_text(json.dumps(alpasim_doc), encoding="utf-8")
    out = tmp_path / "out.json"

    with caplog.at_level("WARNING"):
        rc = _cli.main(
            [
                "--input",
                str(src),
                "--output",
                str(out),
                "--camera",
                "front",
                "--fx",
                "999.0",
                "--quiet",
            ]
        )
    assert rc == 0
    assert any("alpasim" in r.message for r in caplog.records)
    # Output is written in our schema and carries the edit.
    out_doc = json.loads(out.read_text(encoding="utf-8"))
    assert out_doc["schema"] == "splatsim.rig_trajectories/v1"
    assert out_doc["rigs"][0]["cameras"][0]["camera_model"]["parameters"]["fx"] == 999.0
