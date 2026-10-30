"""Domain types for robot telemetry, mirroring the v2 wire format."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


def normalize_quaternion(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float, float, float]:
    """Scale a quaternion to unit norm (defensive: ``q`` and ``-q`` agree)."""
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm == 0.0:
        raise ValueError("zero-norm quaternion")
    return qx / norm, qy / norm, qz / norm, qw / norm


def euler_to_quaternion(roll_deg: float, pitch_deg: float, yaw_deg: float) -> tuple[float, float, float, float]:
    """Intrinsic Z-Y-X Euler degrees -> unit quaternion (qx, qy, qz, qw).

    Hamilton convention, scalar last, for the body-to-map rotation
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    """
    r, p, y = math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg)
    cr, sr = math.cos(r / 2.0), math.sin(r / 2.0)
    cp, sp = math.cos(p / 2.0), math.sin(p / 2.0)
    cy, sy = math.cos(y / 2.0), math.sin(y / 2.0)
    return (
        cy * cp * sr - sy * sp * cr,
        sy * cp * sr + cy * sp * cr,
        sy * cp * cr - cy * sp * sr,
        cy * cp * cr + sy * sp * sr,
    )


def quaternion_to_euler(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float, float]:
    """Unit quaternion (qx, qy, qz, qw) -> intrinsic Z-Y-X Euler degrees (roll, pitch, yaw).

    Sign-invariant: ``q`` and ``-q`` describe the same rotation and yield the
    same angles. Handles the pitch ~= +/-90 deg gimbal-lock case without NaNs.
    """
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm == 0.0:
        raise ValueError("zero-norm quaternion")
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    r20 = 2.0 * (qx * qz - qy * qw)
    r20 = max(-1.0, min(1.0, r20))
    pitch = math.asin(-r20)
    if abs(abs(r20) - 1.0) < 1e-12:
        # Gimbal lock: yaw and roll share one degree of freedom; fix roll at 0.
        roll = 0.0
        r01 = 2.0 * (qx * qy - qz * qw)
        r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
        yaw = math.atan2(-r01, r11)
    else:
        roll = math.atan2(2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy))
        yaw = math.atan2(2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qy * qy + qz * qz))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


@dataclass(frozen=True)
class Pose:
    """Robot pose in the fleet map frame.

    Position is in metres. The primary orientation is the unit quaternion
    ``(qx, qy, qz, qw)`` (Hamilton convention, scalar last) served by v2 for
    the body-to-map rotation ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``. The
    server does not canonicalise the sign: ``q`` and ``-q`` are the same
    rotation, so every consumer must be sign-invariant.

    ``roll``/``pitch``/``yaw`` remain as a source-compatible derived view
    (Euler degrees, intrinsic Z-Y-X). When the quaternion is given it is
    normalised defensively; when omitted it is derived from the Euler angles.
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
        else:
            qx, qy, qz, qw = normalize_quaternion(self.qx, self.qy, self.qz, self.qw)
            object.__setattr__(self, "qx", qx)
            object.__setattr__(self, "qy", qy)
            object.__setattr__(self, "qz", qz)
            object.__setattr__(self, "qw", qw)


@dataclass(frozen=True)
class TelemetryFrame:
    robot_id: str
    timestamp: float  # unix seconds
    pose: Pose
    noise: float  # scalar 1-sigma position noise (m), derived as sqrt(trace(P_pos) / 3)
    battery: float  # state of charge, 0..1
    covariance: tuple[tuple[float, ...], ...]  # 6x6 pose covariance, row-major; x,y,z (m) then rot X,Y,Z (rad)
    health: float  # composite localisation/drive health, 0..1
