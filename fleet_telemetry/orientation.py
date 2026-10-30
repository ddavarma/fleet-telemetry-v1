"""Orientation math on v2 quaternion poses (Hamilton, scalar last)."""
from __future__ import annotations

import numpy as np

from fleet_telemetry.models import Pose


def _unit_quaternion(pose: Pose) -> tuple[float, float, float, float]:
    """Quaternion normalised defensively; q and -q describe the same rotation."""
    q = np.array([pose.qx, pose.qy, pose.qz, pose.qw], dtype=float)
    norm = float(np.linalg.norm(q))
    if norm <= 0.0:
        raise ValueError("orientation quaternion must be non-zero")
    qx, qy, qz, qw = q / norm
    return float(qx), float(qy), float(qz), float(qw)


def rotation_matrix(pose: Pose) -> np.ndarray:
    """Body-to-map rotation built directly from the unit quaternion.

    Sign-invariant: negating the quaternion leaves the matrix unchanged.
    """
    qx, qy, qz, qw = _unit_quaternion(pose)
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ]
    )


def heading_deg(pose: Pose) -> float:
    """Compass heading in degrees, normalised to [0, 360).

    Computed with the sign-invariant ``atan2(2(qw·qz + qx·qy), 1 − 2(qy² + qz²))``;
    never assumes ``qw >= 0``.
    """
    qx, qy, qz, qw = _unit_quaternion(pose)
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return float(np.degrees(yaw) % 360.0)


def heading_vector(pose: Pose) -> np.ndarray:
    """Unit vector of the robot's forward axis projected onto the map plane."""
    fwd = rotation_matrix(pose) @ np.array([1.0, 0.0, 0.0])
    xy = fwd[:2]
    return xy / np.linalg.norm(xy)


def tilt_deg(pose: Pose) -> float:
    """Angle between the body z-axis and map z-axis, in degrees."""
    cos_tilt = np.clip(rotation_matrix(pose)[2, 2], -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_tilt)))
