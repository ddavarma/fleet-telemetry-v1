"""Combine observations from several frames using the position covariance."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from fleet_telemetry.models import TelemetryFrame
from fleet_telemetry.orientation import heading_deg


def position_sigma(frame: TelemetryFrame) -> float:
    """Scalar position 1-sigma (m) as sqrt(trace(P_pos) / 3)."""
    if frame.covariance is not None:
        p = np.asarray(frame.covariance, dtype=float)
        if p.shape != (6, 6):
            raise ValueError("covariance must be 6x6")
        return float(math.sqrt((p[0, 0] + p[1, 1] + p[2, 2]) / 3.0))
    return float(frame.noise)


def position_variance(frame: TelemetryFrame) -> float:
    """Isotropic position variance (m^2) implied by the position covariance."""
    return position_sigma(frame) ** 2


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
    """Weighted circular mean of heading in degrees, [0, 360)."""
    if not frames:
        raise ValueError("need at least one frame")
    weights = np.array([1.0 / position_variance(f) for f in frames])
    yaws = np.radians([heading_deg(f.pose) for f in frames])
    s = (weights * np.sin(yaws)).sum()
    c = (weights * np.cos(yaws)).sum()
    return float(np.degrees(np.arctan2(s, c)) % 360.0)
