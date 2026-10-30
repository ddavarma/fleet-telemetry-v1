"""Domain types for robot telemetry, mirroring the v2 wire format."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


def quaternion_from_euler_deg(roll_deg: float, pitch_deg: float, yaw_deg: float) -> tuple[float, float, float, float]:
    """Unit quaternion ``(qx, qy, qz, qw)`` (Hamilton, scalar last) for ``Rz(yaw)·Ry(pitch)·Rx(roll)``.

    Angles are intrinsic Z-Y-X Euler degrees, the convention v1 served.
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


def euler_from_quaternion_deg(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float, float]:
    """Intrinsic Z-Y-X Euler degrees ``(roll, pitch, yaw)`` from a quaternion.

    Sign-invariant: ``q`` and ``-q`` give the same angles. The quaternion is
    normalised defensively; a zero quaternion raises ``ValueError``.
    """
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 0.0:
        raise ValueError("orientation quaternion must be non-zero")
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    yaw = math.degrees(math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx)))))
    roll = math.degrees(math.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy)))
    return roll, pitch, yaw


def position_sigma_from_covariance(covariance: Sequence[Sequence[float]]) -> float:
    """Scalar 1-sigma position uncertainty (m): ``sqrt(trace(P_pos) / 3)``.

    ``P_pos`` is the position block ``covariance[0:3][0:3]`` (m²). The covariance
    must be 6x6; anything else raises ``ValueError``.
    """
    try:
        rows = list(covariance)
        shape_ok = len(rows) == 6 and all(len(row) == 6 for row in rows)
    except TypeError:
        rows = []
        shape_ok = False
    if not shape_ok:
        try:
            desc = f"{len(rows)}x{[len(r) for r in rows]}"
        except TypeError:
            desc = type(covariance).__name__
        raise ValueError(f"covariance must be 6x6, got {desc}")
    trace = float(rows[0][0]) + float(rows[1][1]) + float(rows[2][2])
    if trace <= 0.0:
        raise ValueError(f"covariance position block must have positive trace, got {trace}")
    return math.sqrt(trace / 3.0)


@dataclass(frozen=True)
class Pose:
    """Robot pose in the fleet map frame.

    Position is in metres. Orientation is a unit quaternion ``(qx, qy, qz, qw)``
    (Hamilton convention, scalar last) giving the body-to-map rotation, as served
    by v2. The server does not canonicalise the sign, so ``qw`` may be negative.
    ``roll``/``pitch``/``yaw`` are intrinsic Z-Y-X Euler degrees kept as derived
    values for source compatibility.
    """

    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    qx: float | None = None
    qy: float | None = None
    qz: float | None = None
    qw: float | None = None

    def __post_init__(self) -> None:
        if self.qx is None or self.qy is None or self.qz is None or self.qw is None:
            qx, qy, qz, qw = quaternion_from_euler_deg(self.roll, self.pitch, self.yaw)
            object.__setattr__(self, "qx", qx)
            object.__setattr__(self, "qy", qy)
            object.__setattr__(self, "qz", qz)
            object.__setattr__(self, "qw", qw)
        else:
            norm = math.sqrt(self.qx**2 + self.qy**2 + self.qz**2 + self.qw**2)
            if norm <= 0.0:
                raise ValueError("orientation quaternion must be non-zero")
            object.__setattr__(self, "qx", self.qx / norm)
            object.__setattr__(self, "qy", self.qy / norm)
            object.__setattr__(self, "qz", self.qz / norm)
            object.__setattr__(self, "qw", self.qw / norm)
            roll, pitch, yaw = euler_from_quaternion_deg(self.qx, self.qy, self.qz, self.qw)
            object.__setattr__(self, "roll", roll)
            object.__setattr__(self, "pitch", pitch)
            object.__setattr__(self, "yaw", yaw)

    @classmethod
    def from_quaternion(
        cls, x: float, y: float, z: float, qx: float, qy: float, qz: float, qw: float
    ) -> "Pose":
        """Build a pose from a v2 quaternion, deriving the Euler compat angles."""
        roll, pitch, yaw = euler_from_quaternion_deg(qx, qy, qz, qw)
        return cls(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, qx=qx, qy=qy, qz=qz, qw=qw)


@dataclass(frozen=True)
class TelemetryFrame:
    """One robot's telemetry: pose, 6x6 covariance, health and battery.

    ``covariance`` is the 6x6 symmetric pose covariance (row-major, x,y,z then
    rotation about X,Y,Z; position block in m²). ``health`` is the 0..1 composite
    health score. ``noise`` is the derived scalar 1-sigma
    ``sqrt(trace(P_pos) / 3)`` kept for source compatibility.
    """

    robot_id: str
    timestamp: float
    pose: Pose
    covariance: tuple[tuple[float, ...], ...]
    health: float
    battery: float
    noise: float | None = None

    def __post_init__(self) -> None:
        cov = tuple(tuple(float(v) for v in row) for row in self.covariance)
        if len(cov) != 6 or any(len(row) != 6 for row in cov):
            raise ValueError(f"covariance must be 6x6, got {len(cov)}x{[len(r) for r in cov]}")
        object.__setattr__(self, "covariance", cov)
        if not 0.0 <= float(self.health) <= 1.0:
            raise ValueError(f"health {self.health!r} outside 0..1")
        if self.noise is None:
            object.__setattr__(self, "noise", position_sigma_from_covariance(cov))
        elif self.noise <= 0:
            raise ValueError(f"noise must be positive, got {self.noise}")
