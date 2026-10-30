"""Combine observations from several frames using the per-frame covariance."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from fleet_telemetry.models import TelemetryFrame


def position_covariance(frame: TelemetryFrame) -> np.ndarray:
    """3x3 position covariance block (m^2), i.e. covariance[0:3][0:3]."""
    cov = np.asarray(frame.covariance, dtype=float)
    return cov[0:3, 0:3]


def position_variance(frame: TelemetryFrame) -> float:
    """Scalar position variance (m^2): trace of the position block over 3.

    For isotropic covariances this equals the old scalar ``noise`` squared,
    so fusion weights and noise alerts are unchanged on such fixtures.
    """
    p = position_covariance(frame)
    return float((p[0, 0] + p[1, 1] + p[2, 2]) / 3.0)


def position_sigma(frame: TelemetryFrame) -> float:
    """Scalar 1-sigma position uncertainty (m): sqrt(trace(P_pos) / 3)."""
    return math.sqrt(position_variance(frame))


def fuse_positions(frames: Sequence[TelemetryFrame]) -> tuple[np.ndarray, float]:
    """Inverse-covariance weighted position and the fused 1-sigma, e.g. for a
    landmark seen by several robots."""
    if not frames:
        raise ValueError("need at least one frame")
    precisions = [np.linalg.inv(position_covariance(f)) for f in frames]
    points = np.array([[f.pose.x, f.pose.y, f.pose.z] for f in frames])
    total = sum(precisions)
    fused = np.linalg.solve(total, sum(p @ x for p, x in zip(precisions, points)))
    fused_cov = np.linalg.inv(total)
    sigma = float(math.sqrt(np.trace(fused_cov) / 3.0))
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
