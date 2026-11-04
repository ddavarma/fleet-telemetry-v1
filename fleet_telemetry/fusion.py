"""Combine observations from several frames using the position covariance."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from fleet_telemetry.models import TelemetryFrame


def position_variance(frame: TelemetryFrame) -> float:
    """Scalar position variance (m^2): ``trace(P_pos) / 3``.

    ``P_pos`` is the 3x3 position block ``covariance[0:3][0:3]`` in m^2, so for
    isotropic fixtures this equals the square of the old scalar noise sigma.
    """
    cov = frame.covariance
    return float((cov[0][0] + cov[1][1] + cov[2][2]) / 3.0)


def fuse_positions(frames: Sequence[TelemetryFrame]) -> tuple[np.ndarray, float]:
    """Inverse-variance weighted position and the fused 1-sigma, e.g. for a
    landmark seen by several robots."""
    if not frames:
        raise ValueError("need at least one frame")
    weights = np.array([1.0 / position_variance(f) for f in frames])
    points = np.array([[f.pose.x, f.pose.y, f.pose.z] for f in frames])
    fused = (weights[:, None] * points).sum(axis=0) / weights.sum()
    sigma = float(np.sqrt(1.0 / weights.sum()))
    return fused, sigma


def fuse_heading(frames: Sequence[TelemetryFrame]) -> float:
    """Weighted circular mean of yaw in degrees, [0, 360)."""
    if not frames:
        raise ValueError("need at least one frame")
    weights = np.array([1.0 / position_variance(f) for f in frames])
    yaws = np.radians([f.pose.yaw for f in frames])
    s = (weights * np.sin(yaws)).sum()
    c = (weights * np.cos(yaws)).sum()
    return float(np.degrees(np.arctan2(s, c)) % 360.0)
