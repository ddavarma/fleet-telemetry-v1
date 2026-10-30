"""Domain types for robot telemetry, mirroring the v2 wire format."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple


def euler_to_quaternion(roll_deg: float, pitch_deg: float, yaw_deg: float) -> Tuple[float, float, float, float]:
    """Euler (degrees, intrinsic Z-Y-X) to unit quaternion (Hamilton, scalar last).

    Encodes the same body-to-map rotation R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    """
    r, p, y = math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg)
    cr, sr = math.cos(r / 2.0), math.sin(r / 2.0)
    cp, sp = math.cos(p / 2.0), math.sin(p / 2.0)
    cy, sy = math.cos(y / 2.0), math.sin(y / 2.0)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw


def quaternion_to_euler(qx: float, qy: float, qz: float, qw: float) -> Tuple[float, float, float]:
    """Unit quaternion to Euler angles (degrees, intrinsic Z-Y-X).

    Returns (roll, pitch, yaw) with yaw normalised to [0, 360).
    Sign-invariant: q and -q give the same angles.
    """
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n == 0.0:
        raise ValueError("zero quaternion has no orientation")
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r10 = 2.0 * (qx * qy + qz * qw)
    r20 = 2.0 * (qx * qz - qy * qw)
    r21 = 2.0 * (qy * qz + qx * qw)
    r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.degrees(math.atan2(r21, r22))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -r20))))
    yaw = math.degrees(math.atan2(r10, r00)) % 360.0
    return roll, pitch, yaw


@dataclass(frozen=True)
class Pose:
    """Robot pose in the fleet map frame.

    Position is in metres. Orientation is a unit quaternion (Hamilton,
    scalar last: qx, qy, qz, qw) describing the body-to-map rotation, as
    served by v2. Euler angles in degrees (intrinsic Z-Y-X) are kept as
    derived values for source compatibility.
    """

    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    qx: Optional[float] = None
    qy: Optional[float] = None
    qz: Optional[float] = None
    qw: Optional[float] = None

    def __post_init__(self) -> None:
        if self.qx is None or self.qy is None or self.qz is None or self.qw is None:
            qx, qy, qz, qw = euler_to_quaternion(self.roll, self.pitch, self.yaw)
            object.__setattr__(self, "qx", qx)
            object.__setattr__(self, "qy", qy)
            object.__setattr__(self, "qz", qz)
            object.__setattr__(self, "qw", qw)

    @classmethod
    def from_quaternion(
        cls, x: float, y: float, z: float, qx: float, qy: float, qz: float, qw: float
    ) -> "Pose":
        """Build a pose from a (possibly unnormalised) quaternion, deriving Euler."""
        roll, pitch, yaw = quaternion_to_euler(qx, qy, qz, qw)
        return cls(x, y, z, roll, pitch, yaw, float(qx), float(qy), float(qz), float(qw))


Covariance = Tuple[
    Tuple[float, float, float, float, float, float],
    Tuple[float, float, float, float, float, float],
    Tuple[float, float, float, float, float, float],
    Tuple[float, float, float, float, float, float],
    Tuple[float, float, float, float, float, float],
    Tuple[float, float, float, float, float, float],
]


@dataclass(frozen=True)
class TelemetryFrame:
    robot_id: str
    timestamp: float  # unix seconds
    pose: Pose
    noise: float  # scalar 1-sigma position noise, metres (derived from covariance)
    battery: float  # state of charge, 0..1
    covariance: Covariance  # required 6x6 pose covariance, row-major
    health: float  # required composite localisation/drive health, 0..1

    def __post_init__(self) -> None:
        rows = self.covariance
        if rows is None or len(rows) != 6 or any(len(r) != 6 for r in rows):
            raise ValueError("covariance must be 6x6")
