"""Orientation math on v2 quaternion poses (Hamilton, scalar last)."""
from __future__ import annotations

import math

import numpy as np

from fleet_telemetry.models import Pose


def _normalised(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float, float, float]:
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n == 0.0:
        raise ValueError("zero quaternion has no orientation")
    return qx / n, qy / n, qz / n, qw / n


def _rotation_from_quat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    qx, qy, qz, qw = _normalised(qx, qy, qz, qw)
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ]
    )


def _rotation_from_euler(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    r, p, y = np.radians([roll_deg, pitch_deg, yaw_deg])
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def rotation_matrix(pose: Pose) -> np.ndarray:
    """Body-to-map rotation built directly from the unit quaternion.

    Quadratic in q, so q and -q give the same rotation. Falls back to Euler
    only when a pose carries no quaternion.
    """
    if pose.qx is not None and pose.qy is not None and pose.qz is not None and pose.qw is not None:
        return _rotation_from_quat(pose.qx, pose.qy, pose.qz, pose.qw)
    return _rotation_from_euler(pose.roll, pose.pitch, pose.yaw)


def heading_deg(pose: Pose) -> float:
    """Yaw normalised to [0, 360), from the quaternion via atan2."""
    if pose.qx is not None and pose.qy is not None and pose.qz is not None and pose.qw is not None:
        qx, qy, qz, qw = _normalised(pose.qx, pose.qy, pose.qz, pose.qw)
        yaw = math.degrees(math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))
        return float(yaw % 360.0)
    return float(pose.yaw % 360.0)


def heading_vector(pose: Pose) -> np.ndarray:
    """Unit vector of the robot's forward axis projected onto the map plane."""
    fwd = rotation_matrix(pose) @ np.array([1.0, 0.0, 0.0])
    xy = fwd[:2]
    return xy / np.linalg.norm(xy)


def tilt_deg(pose: Pose) -> float:
    """Angle between the body z-axis and map z-axis, in degrees."""
    cos_tilt = np.clip(rotation_matrix(pose)[2, 2], -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_tilt)))
