"""Domain types for robot telemetry, mirroring the v2 wire format."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


def euler_to_quaternion(roll_deg: float, pitch_deg: float, yaw_deg: float) -> tuple[float, float, float, float]:
    """Intrinsic Z-Y-X Euler angles (degrees) to unit quaternion ``(qx, qy, qz, qw)``.

    The rotation is ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``, Hamilton convention,
    scalar last.
    """
    r, p, y = (math.radians(a) / 2.0 for a in (roll_deg, pitch_deg, yaw_deg))
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def quaternion_to_euler_deg(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float, float]:
    """Unit quaternion (Hamilton, scalar last) to intrinsic Z-Y-X Euler degrees.

    Returns ``(roll, pitch, yaw)`` describing the same body-to-map rotation
    ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`` that v1 encoded. Sign-invariant:
    ``q`` and ``-q`` give the same angles (modulo the gimbal-lock degeneracy
    at pitch = +/-90 degrees, where roll/yaw share one degree of freedom).
    """
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if not math.isfinite(n) or n <= 0.0:
        raise ValueError(f"invalid quaternion {(qx, qy, qz, qw)}")
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r10 = 2.0 * (qx * qy + qz * qw)
    r20 = 2.0 * (qx * qz - qy * qw)
    r21 = 2.0 * (qy * qz + qx * qw)
    r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -r20))))
    roll = math.degrees(math.atan2(r21, r22))
    yaw = math.degrees(math.atan2(r10, r00))
    return roll, pitch, yaw


def validate_covariance(covariance: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    """Validate a 6x6 pose covariance and normalise it to tuples of floats."""
    try:
        rows = [tuple(float(v) for v in row) for row in covariance]
    except TypeError as exc:
        raise ValueError(f"covariance must be a 6x6 nested array, got {covariance!r}") from exc
    except (ValueError, TypeError) as exc:
        raise ValueError(f"covariance entries must be numbers: {exc}") from exc
    if len(rows) != 6 or any(len(row) != 6 for row in rows):
        raise ValueError(f"covariance must be 6x6, got {[len(row) for row in rows]}")
    if any(not math.isfinite(v) for row in rows for v in row):
        raise ValueError("covariance entries must be finite")
    return tuple(rows)


@dataclass(frozen=True)
class Pose:
    """Robot pose in the fleet map frame.

    Position is in metres. Orientation is a unit quaternion (Hamilton
    convention, scalar last) giving the body-to-map rotation, as served by v2.
    The Euler roll/pitch/yaw angles (degrees, intrinsic Z-Y-X order) are kept
    as derived values so code written against v1 keeps working.
    """

    x: float
    y: float
    z: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    qx: float | None = None
    qy: float | None = None
    qz: float | None = None
    qw: float | None = None

    def __post_init__(self) -> None:
        given = (self.qx, self.qy, self.qz, self.qw)
        if any(v is None for v in given):
            if not all(v is None for v in given):
                raise ValueError("quaternion must be given as all of qx, qy, qz, qw or none")
            qx, qy, qz, qw = euler_to_quaternion(self.roll, self.pitch, self.yaw)
            object.__setattr__(self, "qx", qx)
            object.__setattr__(self, "qy", qy)
            object.__setattr__(self, "qz", qz)
            object.__setattr__(self, "qw", qw)
        else:
            n = math.sqrt(self.qx**2 + self.qy**2 + self.qz**2 + self.qw**2)
            if not math.isfinite(n) or n <= 0.0:
                raise ValueError(f"invalid quaternion {(self.qx, self.qy, self.qz, self.qw)}")
            object.__setattr__(self, "qx", self.qx / n)
            object.__setattr__(self, "qy", self.qy / n)
            object.__setattr__(self, "qz", self.qz / n)
            object.__setattr__(self, "qw", self.qw / n)
            roll, pitch, yaw = quaternion_to_euler_deg(self.qx, self.qy, self.qz, self.qw)
            object.__setattr__(self, "roll", roll)
            object.__setattr__(self, "pitch", pitch)
            object.__setattr__(self, "yaw", yaw)


@dataclass(frozen=True)
class TelemetryFrame:
    robot_id: str
    timestamp: float  # unix seconds
    pose: Pose
    covariance: tuple[tuple[float, ...], ...]  # 6x6 pose covariance, x,y,z (m) then rot X,Y,Z (rad)
    health: float  # composite localisation/drive health, 0..1
    battery: float  # state of charge, 0..1
    noise: float | None = None  # scalar 1-sigma position noise, metres; derived from covariance

    def __post_init__(self) -> None:
        cov = validate_covariance(self.covariance)
        object.__setattr__(self, "covariance", cov)
        if not math.isfinite(self.health) or not 0.0 <= self.health <= 1.0:
            raise ValueError(f"health must be in [0, 1], got {self.health}")
        if self.noise is None:
            sigma = math.sqrt((cov[0][0] + cov[1][1] + cov[2][2]) / 3.0)
            if not math.isfinite(sigma) or sigma <= 0.0:
                raise ValueError(f"covariance implies non-positive position sigma: {sigma}")
            object.__setattr__(self, "noise", sigma)
        elif not math.isfinite(self.noise) or self.noise <= 0.0:
            raise ValueError(f"noise must be positive, got {self.noise}")
