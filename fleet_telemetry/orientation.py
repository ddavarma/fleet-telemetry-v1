"""Orientation math on v2 quaternion poses (Hamilton convention, scalar last)."""
from __future__ import annotations

import math

import numpy as np

from fleet_telemetry.models import Pose


def normalize_quaternion(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float, float, float]:
    """Scale (qx, qy, qz, qw) to unit length.

    The server guarantees |q| == 1 within 1e-6, but callers must not assume
    it; every consumer below normalises defensively. Sign is preserved as-is
    (``qw`` may be negative): ``q`` and ``-q`` describe the same rotation and
    all formulas used here are sign-invariant.
    """
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError(f"degenerate quaternion ({qx}, {qy}, {qz}, {qw})")
    return (qx / norm, qy / norm, qz / norm, qw / norm)


def _hamilton_product(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """Hamilton product a (x) b: rotation b applied first, then a."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quaternion_from_euler(roll_deg: float, pitch_deg: float, yaw_deg: float) -> tuple[float, float, float, float]:
    """Unit quaternion for R = Rz(yaw) @ Ry(pitch) @ Rx(roll), angles in degrees."""
    r, p, y = math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg)
    qx = (math.sin(r / 2.0), 0.0, 0.0, math.cos(r / 2.0))
    qy = (0.0, math.sin(p / 2.0), 0.0, math.cos(p / 2.0))
    qz = (0.0, 0.0, math.sin(y / 2.0), math.cos(y / 2.0))
    return _hamilton_product(qz, _hamilton_product(qy, qx))


def euler_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float, float]:
    """Derived (roll, pitch, yaw) in degrees, intrinsic Z-Y-X, from a quaternion.

    Kept for source compatibility only; orientation math never round-trips
    through Euler (tilt comes straight from the rotation matrix).
    """
    qx, qy, qz, qw = normalize_quaternion(qx, qy, qz, qw)
    roll = math.degrees(math.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy)))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx)))))
    yaw = math.degrees(math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))
    return (roll, pitch, yaw)


def rotation_matrix(pose: Pose) -> np.ndarray:
    """Body-to-map rotation built directly from the unit quaternion.

    Quadratic in the quaternion components, hence identical for ``q`` and
    ``-q``; the quaternion is normalised defensively first.
    """
    qx, qy, qz, qw = normalize_quaternion(pose.qx, pose.qy, pose.qz, pose.qw)
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ]
    )


def heading_deg(pose: Pose) -> float:
    """Yaw normalised to [0, 360), via the sign-invariant atan2 formula."""
    qx, qy, qz, qw = normalize_quaternion(pose.qx, pose.qy, pose.qz, pose.qw)
    yaw = math.degrees(math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))
    return float(yaw % 360.0)


def heading_vector(pose: Pose) -> np.ndarray:
    """Unit vector of the robot's forward axis projected onto the map plane."""
    fwd = rotation_matrix(pose) @ np.array([1.0, 0.0, 0.0])
    xy = fwd[:2]
    return xy / np.linalg.norm(xy)


def tilt_deg(pose: Pose) -> float:
    """Angle between the body z-axis and map z-axis, in degrees.

    Read straight from the quaternion rotation matrix; never round-trips
    through Euler angles.
    """
    cos_tilt = np.clip(rotation_matrix(pose)[2, 2], -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_tilt)))
