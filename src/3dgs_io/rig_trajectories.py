"""Alpasim-native, frame-explicit sensor-rig trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from .cameras import Camera, CameraExtrinsics
from .frame_convention import (
    FRAME_CONVENTION,
    validate_frame_convention,
    validate_rigid_transform,
    validate_rotation,
    validate_timestamps,
)

RIG_TRAJECTORIES_SCHEMA = "splatsim.rig_trajectories/v2"
_FRAME = "world"

__all__ = [
    "RIG_TRAJECTORIES_SCHEMA",
    "LidarCalibration",
    "LidarModel",
    "RigPose",
    "RigTrajectory",
    "load_rig_trajectories_doc",
    "parse_rig_trajectories",
    "serialize_rig_trajectories",
    "update_camera_intrinsics",
]


@dataclass
class RigPose:
    """A rig pose expressed directly in the Z-up ENU world frame."""

    timestamp_us: int
    translation: tuple[float, float, float]
    rotation: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_us": int(self.timestamp_us),
            "translation": [float(v) for v in self.translation],
            "rotation": [float(v) for v in self.rotation],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RigPose:
        translation = value["translation"]
        rotation = value["rotation"]
        if len(translation) != 3:
            raise ValueError("pose.translation must have 3 elements")
        if len(rotation) != 4:
            raise ValueError("pose.rotation must have 4 elements (xyzw)")
        validate_timestamps([value["timestamp_us"]], where="rig pose")
        validate_rotation(rotation, where="rig pose rotation")
        return cls(
            timestamp_us=int(value["timestamp_us"]),
            translation=tuple(float(v) for v in translation),
            rotation=tuple(float(v) for v in rotation),
        )


_REQUIRED_LIDAR_INTRINSIC_KEYS: dict[str, tuple[str, ...]] = {
    "spinning": ("n_rows", "n_columns", "fps", "min_range_m", "max_range_m"),
}


@dataclass
class LidarModel:
    """Type-tagged LiDAR intrinsics."""

    type: str
    parameters: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, dict):
            raise ValueError("lidar_model.parameters must be a dict")
        required = _REQUIRED_LIDAR_INTRINSIC_KEYS.get(self.type, ("n_rows", "n_columns"))
        missing = [key for key in required if key not in self.parameters]
        if missing:
            raise ValueError(
                f"lidar_model(type={self.type!r}) is missing required intrinsic key(s) {missing}"
            )
        if self.type == "spinning" and not (
            "elevation_deg" in self.parameters or "elevation_fov_deg" in self.parameters
        ):
            raise ValueError(
                "lidar_model(type='spinning') requires elevation_deg or elevation_fov_deg"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "parameters": dict(self.parameters)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> LidarModel:
        return cls(type=str(value["type"]), parameters=dict(value["parameters"]))


@dataclass
class LidarCalibration:
    """A LiDAR pose expressed directly as sensor-in-rig."""

    name: str
    extrinsics: CameraExtrinsics
    logical_sensor_name: str | None = None
    lidar_model: LidarModel | None = None
    unique_sensor_idx: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "name": self.name,
            "sensor_in_rig": {
                "translation": [float(v) for v in self.extrinsics.translation],
                "rotation": [float(v) for v in self.extrinsics.rotation],
            },
        }
        if self.logical_sensor_name is not None:
            value["logical_sensor_name"] = self.logical_sensor_name
        if self.lidar_model is not None:
            value["lidar_model"] = self.lidar_model.to_dict()
        if self.unique_sensor_idx is not None:
            value["unique_sensor_idx"] = int(self.unique_sensor_idx)
        if self.metadata:
            value["metadata"] = dict(self.metadata)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> LidarCalibration:
        pose = value["sensor_in_rig"]
        return cls(
            name=str(value["name"]),
            extrinsics=CameraExtrinsics(
                translation=tuple(float(v) for v in pose["translation"]),
                rotation=tuple(float(v) for v in pose["rotation"]),
            ),
            logical_sensor_name=value.get("logical_sensor_name"),
            lidar_model=(
                LidarModel.from_dict(value["lidar_model"])
                if value.get("lidar_model") is not None
                else None
            ),
            unique_sensor_idx=value.get("unique_sensor_idx"),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass
class RigTrajectory:
    """One rig and its alpasim-native world-frame trajectory."""

    rig_id: str
    poses: list[RigPose] = field(default_factory=list)
    cameras: list[Camera] = field(default_factory=list)
    lidars: list[LidarCalibration] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rig_id": self.rig_id,
            "poses": [pose.to_dict() for pose in self.poses],
            "cameras": [camera.to_dict() for camera in self.cameras],
            "lidars": [lidar.to_dict() for lidar in self.lidars],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RigTrajectory:
        return cls(
            rig_id=str(value["rig_id"]),
            poses=[RigPose.from_dict(pose) for pose in value.get("poses") or []],
            cameras=[Camera.from_dict(camera) for camera in value.get("cameras") or []],
            lidars=[LidarCalibration.from_dict(lidar) for lidar in value.get("lidars") or []],
            metadata=dict(value.get("metadata") or {}),
        )


def _validate_rig(rig: RigTrajectory) -> None:
    validate_timestamps([pose.timestamp_us for pose in rig.poses], where=f"rig {rig.rig_id!r}")
    for index, pose in enumerate(rig.poses):
        validate_rotation(pose.rotation, where=f"rig {rig.rig_id!r} pose {index} rotation")
    names: set[str] = set()
    for sensor_type, sensors in (("camera", rig.cameras), ("lidar", rig.lidars)):
        for sensor in sensors:
            if sensor.name in names:
                raise ValueError(f"duplicate sensor name {sensor.name!r} in rig {rig.rig_id!r}")
            names.add(sensor.name)
            validate_rotation(
                sensor.extrinsics.rotation,
                where=f"{sensor_type} {sensor.name!r} sensor_in_rig rotation",
            )
            validate_rigid_transform(
                sensor.extrinsics.to_matrix(),
                where=f"{sensor_type} {sensor.name!r} sensor_in_rig",
            )


def serialize_rig_trajectories(rigs: list[RigTrajectory]) -> dict[str, Any]:
    seen: set[str] = set()
    for rig in rigs:
        if rig.rig_id in seen:
            raise ValueError(f"duplicate rig_id: {rig.rig_id!r}")
        seen.add(rig.rig_id)
        _validate_rig(rig)
    return {
        "schema": RIG_TRAJECTORIES_SCHEMA,
        "frame": _FRAME,
        "frame_convention": FRAME_CONVENTION,
        "rigs": [rig.to_dict() for rig in rigs],
    }


def parse_rig_trajectories(document: dict[str, Any]) -> list[RigTrajectory]:
    if document.get("schema") != RIG_TRAJECTORIES_SCHEMA:
        raise ValueError(
            f"unexpected rig_trajectories schema {document.get('schema')!r}; "
            f"expected {RIG_TRAJECTORIES_SCHEMA!r}"
        )
    if document.get("frame") != _FRAME:
        raise ValueError("rig_trajectories frame must be 'world'")
    validate_frame_convention(document.get("frame_convention"))
    raw = document.get("rigs")
    if not isinstance(raw, list):
        raise ValueError("rig_trajectories document is missing the 'rigs' list")
    rigs = [RigTrajectory.from_dict(value) for value in raw]
    seen: set[str] = set()
    for rig in rigs:
        if rig.rig_id in seen:
            raise ValueError(f"duplicate rig_id: {rig.rig_id!r}")
        seen.add(rig.rig_id)
        _validate_rig(rig)
    return rigs


def load_rig_trajectories_doc(document: dict[str, Any]) -> list[RigTrajectory]:
    if not isinstance(document, dict):
        raise ValueError("rig_trajectories document must be a JSON object")
    return parse_rig_trajectories(document)


def update_camera_intrinsics(
    rigs: list[RigTrajectory],
    *,
    camera_name: str,
    rig_id: str | None = None,
    **updates: Any,
) -> Camera:
    if not updates:
        raise ValueError("update_camera_intrinsics requires at least one intrinsic update")
    if rig_id is not None and not any(rig.rig_id == rig_id for rig in rigs):
        raise ValueError(f"rig {rig_id!r} not found")
    matches = [
        (rig, index)
        for rig in rigs
        if rig_id is None or rig.rig_id == rig_id
        for index, camera in enumerate(rig.cameras)
        if camera.name == camera_name
    ]
    if not matches:
        raise ValueError(f"camera {camera_name!r} not found")
    if len(matches) != 1:
        raise ValueError(f"camera {camera_name!r} is ambiguous; pass rig_id=")
    rig, index = matches[0]
    rig.cameras[index] = replace(
        rig.cameras[index],
        camera_model=rig.cameras[index].camera_model.with_intrinsics(**updates),
    )
    return rig.cameras[index]
