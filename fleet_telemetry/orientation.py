"""Orientation math on v2 quaternion poses (Hamilton, scalar last)."""
from __future__ import annotations

import math

import numpy as np

from fleet_telemetry.models import Pose


def _unit_quat(pose: Pose) -> tuple[float, float, float, float]:
    """Quaternion normalised defensively; never assumes ``qw >= 0``."""
    n = math.sqrt(pose.qx**2 + pose.qy**2 + pose.qz**2 + pose.qw**2)
    if n <= 0.0:
        raise ValueError("quaternion has zero norm")
    return pose.qx / n, pose.qy / n, pose.qz / n, pose.qw / n


def rotation_matrix(pose: Pose) -> np.ndarray:
    """Body-to-map rotation R = Rz(yaw) @ Ry(pitch) @ Rx(roll), from the quaternion.

    Built directly from ``q`` with a sign-invariant formula, so ``q`` and
    ``-q`` give the same rotation.
    """
    qx, qy, qz, qw = _unit_quat(pose)
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ]
    )


def heading_deg(pose: Pose) -> float:
    """Yaw normalised to [0, 360), via ``atan2`` on the quaternion (sign-invariant)."""
    qx, qy, qz, qw = _unit_quat(pose)
    return float(math.degrees(math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))) % 360.0)


def heading_vector(pose: Pose) -> np.ndarray:
    """Unit vector of the robot's forward axis projected onto the map plane."""
    fwd = rotation_matrix(pose) @ np.array([1.0, 0.0, 0.0])
    xy = fwd[:2]
    return xy / np.linalg.norm(xy)


def tilt_deg(pose: Pose) -> float:
    """Angle between the body z-axis and map z-axis, in degrees, from the rotation matrix."""
    cos_tilt = np.clip(rotation_matrix(pose)[2, 2], -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_tilt)))
