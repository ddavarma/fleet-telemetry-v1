"""Domain types for robot telemetry, mirroring the v2 wire format."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Pose:
    """Robot pose in the fleet map frame.

    Position is in metres. Orientation is a unit quaternion (Hamilton
    convention, scalar last) giving the body-to-map rotation, as served by
    v2. ``qw`` may be negative: ``q`` and ``-q`` are the same rotation.

    ``roll``/``pitch``/``yaw`` remain as derived Euler angles in degrees
    (intrinsic Z-Y-X) for source compatibility. When the quaternion is not
    given (legacy Euler construction), it is derived from the Euler angles.
    """

    x: float
    y: float
    z: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    qx: Optional[float] = None
    qy: Optional[float] = None
    qz: Optional[float] = None
    qw: Optional[float] = None

    def __post_init__(self) -> None:
        # Deferred import: orientation imports Pose at module level.
        from fleet_telemetry.orientation import normalize_quaternion, quaternion_from_euler

        if self.qx is None or self.qy is None or self.qz is None or self.qw is None:
            qx, qy, qz, qw = quaternion_from_euler(self.roll, self.pitch, self.yaw)
        else:
            qx, qy, qz, qw = normalize_quaternion(self.qx, self.qy, self.qz, self.qw)
        object.__setattr__(self, "qx", qx)
        object.__setattr__(self, "qy", qy)
        object.__setattr__(self, "qz", qz)
        object.__setattr__(self, "qw", qw)


def _check_covariance_shape(covariance: tuple[tuple[float, ...], ...]) -> None:
    if len(covariance) != 6 or any(len(row) != 6 for row in covariance):
        raise ValueError(
            "covariance must be a 6x6 nested array, "
            f"got {[len(row) for row in covariance] if isinstance(covariance, tuple) else type(covariance)}"
        )


@dataclass(frozen=True)
class TelemetryFrame:
    robot_id: str
    timestamp: float  # unix seconds
    pose: Pose
    noise: float  # scalar 1-sigma position noise, metres; derived as sqrt(trace(P_pos)/3)
    battery: float  # state of charge, 0..1
    covariance: tuple[tuple[float, ...], ...] = ((0.0,) * 6,) * 6  # 6x6 pose covariance, row-major
    health: float = 1.0  # composite localisation/drive health, 0..1

    def __post_init__(self) -> None:
        _check_covariance_shape(self.covariance)
