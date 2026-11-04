"""Orientation math on v2 quaternion poses (Hamilton convention, scalar last)."""
from __future__ import annotations

import math

import numpy as np

from fleet_telemetry.models import Pose


def _normalised(pose: Pose) -> tuple[float, float, float, float]:
    """Defensively normalised quaternion; q and -q describe the same rotation."""
    n = math.sqrt(pose.qx**2 + pose.qy**2 + pose.qz**2 + pose.qw**2)
    if not math.isfinite(n) or n <= 0.0:
        raise ValueError(f"invalid quaternion {(pose.qx, pose.qy, pose.qz, pose.qw)}")
    return pose.qx / n, pose.qy / n, pose.qz / n, pose.qw / n


def rotation_matrix(pose: Pose) -> np.ndarray:
    """Body-to-map rotation built directly from the unit quaternion.

    Quadratic in q, so q and -q give the identical matrix even though the
    server does not canonicalise the sign.
    """
    qx, qy, qz, qw = _normalised(pose)
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ]
    )


def heading_deg(pose: Pose) -> float:
    """Yaw in [0, 360) via atan2 on the quaternion (sign-invariant)."""
    qx, qy, qz, qw = _normalised(pose)
    yaw = math.degrees(math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))
    return float(yaw % 360.0)


def heading_vector(pose: Pose) -> np.ndarray:
    """Unit vector of the robot's forward axis projected onto the map plane."""
    fwd = rotation_matrix(pose) @ np.array([1.0, 0.0, 0.0])
    xy = fwd[:2]
    return xy / np.linalg.norm(xy)


def tilt_deg(pose: Pose) -> float:
    """Angle between the body z-axis and map z-axis, in degrees."""
    cos_tilt = np.clip(rotation_matrix(pose)[2, 2], -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_tilt)))
