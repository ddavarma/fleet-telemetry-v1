"""Domain types for robot telemetry, mirroring the v2 wire format."""
from __future__ import annotations

import math
from dataclasses import dataclass


def euler_to_quaternion(roll_deg: float, pitch_deg: float, yaw_deg: float) -> tuple[float, float, float, float]:
    """Intrinsic Z-Y-X Euler degrees -> unit quaternion (Hamilton, scalar last).

    The rotation is R = Rz(yaw) @ Ry(pitch) @ Rx(roll), the same body-to-map
    rotation v1 encoded directly.
    """
    r, p, y = math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg)
    cr, sr = math.cos(r / 2.0), math.sin(r / 2.0)
    cp, sp = math.cos(p / 2.0), math.sin(p / 2.0)
    cy, sy = math.cos(y / 2.0), math.sin(y / 2.0)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def quaternion_to_euler(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float, float]:
    """Unit quaternion (Hamilton, scalar last) -> intrinsic Z-Y-X Euler degrees.

    Sign-invariant: ``q`` and ``-q`` give the same angles.
    """
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n <= 0.0:
        raise ValueError("quaternion has zero norm")
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    # Body-to-map rotation entries needed for the Z-Y-X extraction.
    r20 = 2.0 * (qx * qz - qy * qw)
    r21 = 2.0 * (qy * qz + qx * qw)
    r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
    r10 = 2.0 * (qx * qy + qz * qw)
    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -r20))))
    roll = math.degrees(math.atan2(r21, r22))
    yaw = math.degrees(math.atan2(r10, r00))
    return roll, pitch, yaw


@dataclass(frozen=True)
class Pose:
    """Robot pose in the fleet map frame.

    Position is in metres. Orientation is a unit quaternion (Hamilton
    convention, scalar last) describing the body-to-map rotation
    ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``, as served by v2. The server does
    not canonicalise sign, so ``qw`` may be negative.

    ``roll``/``pitch``/``yaw`` (intrinsic Z-Y-X degrees) remain available as
    derived values computed from the quaternion, keeping the v1 Python API
    source-compatible: ``Pose`` still accepts ``roll``/``pitch``/``yaw`` (in
    that case the quaternion is derived from them).
    """

    x: float
    y: float
    z: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0

    def __post_init__(self) -> None:
        n = math.sqrt(self.qx**2 + self.qy**2 + self.qz**2 + self.qw**2)
        if n <= 0.0:
            raise ValueError("quaternion has zero norm")
        object.__setattr__(self, "qx", self.qx / n)
        object.__setattr__(self, "qy", self.qy / n)
        object.__setattr__(self, "qz", self.qz / n)
        object.__setattr__(self, "qw", self.qw / n)
        quat_is_default = (self.qx, self.qy, self.qz, self.qw) == (0.0, 0.0, 0.0, 1.0)
        euler_is_default = (self.roll, self.pitch, self.yaw) == (0.0, 0.0, 0.0)
        if not quat_is_default or euler_is_default:
            roll, pitch, yaw = quaternion_to_euler(self.qx, self.qy, self.qz, self.qw)
            object.__setattr__(self, "roll", roll)
            object.__setattr__(self, "pitch", pitch)
            object.__setattr__(self, "yaw", yaw)
        else:
            qx, qy, qz, qw = euler_to_quaternion(self.roll, self.pitch, self.yaw)
            object.__setattr__(self, "qx", qx)
            object.__setattr__(self, "qy", qy)
            object.__setattr__(self, "qz", qz)
            object.__setattr__(self, "qw", qw)


@dataclass(frozen=True)
class TelemetryFrame:
    robot_id: str
    timestamp: float  # unix seconds
    pose: Pose
    noise: float  # scalar 1-sigma position noise, metres; sqrt(trace(P_pos) / 3)
    battery: float  # state of charge, 0..1
    covariance: tuple[tuple[float, ...], ...] | None = None  # 6x6 pose covariance, row-major
    health: float | None = None  # composite localisation/drive health, 0..1
