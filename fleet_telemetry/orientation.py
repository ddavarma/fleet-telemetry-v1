"""Orientation math on v2 quaternion poses (Hamilton convention, scalar last)."""
from __future__ import annotations

import math

import numpy as np

from fleet_telemetry.models import Pose, normalize_quaternion


def _normalised(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float, float, float]:
    """Return the quaternion scaled to unit norm (defensive: q and -q agree)."""
    return normalize_quaternion(qx, qy, qz, qw)


def rotation_matrix(pose: Pose) -> np.ndarray:
    """Body-to-map rotation R = Rz(yaw) @ Ry(pitch) @ Rx(roll), from the quaternion.

    Built directly from ``(qx, qy, qz, qw)`` with sign-invariant (quadratic)
    terms, so ``q`` and ``-q`` give bit-identical rotations; never assumes
    ``qw >= 0`` and never round-trips through Euler angles.
    """
    qx, qy, qz, qw = _normalised(pose.qx, pose.qy, pose.qz, pose.qw)
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ]
    )


def heading_deg(pose: Pose) -> float:
    """Compass heading (yaw) in degrees, normalised to [0, 360).

    Uses the sign-invariant ``atan2(2(qw*qz + qx*qy), 1 - 2(qy^2 + qz^2))`` form.
    """
    qx, qy, qz, qw = _normalised(pose.qx, pose.qy, pose.qz, pose.qw)
    return float(math.degrees(math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))) % 360.0)


def heading_vector(pose: Pose) -> np.ndarray:
    """Unit vector of the robot's forward axis projected onto the map plane."""
    fwd = rotation_matrix(pose) @ np.array([1.0, 0.0, 0.0])
    xy = fwd[:2]
    return xy / np.linalg.norm(xy)


def tilt_deg(pose: Pose) -> float:
    """Angle between the body z-axis and map z-axis, in degrees.

    Read straight off the quaternion-built rotation matrix (``R[2, 2]``);
    never round-trips through Euler angles.
    """
    cos_tilt = np.clip(rotation_matrix(pose)[2, 2], -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_tilt)))
