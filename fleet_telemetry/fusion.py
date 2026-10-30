"""Combine observations from several frames using the per-frame covariance estimate."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from fleet_telemetry.models import TelemetryFrame
from fleet_telemetry.orientation import heading_deg


def position_variance(frame: TelemetryFrame) -> float:
    """Isotropic position variance (m^2): mean of the position-block diagonal.

    That is ``trace(P_pos) / 3`` with ``P_pos = covariance[0:3][0:3]``, which
    equals ``noise**2`` for the scalar ``noise`` carried on the frame. Falls
    back to ``noise**2`` when no covariance is attached.
    """
    covariance = getattr(frame, "covariance", None)
    if covariance is not None:
        block = np.asarray(covariance, dtype=float)[0:3, 0:3]
        return float((block[0, 0] + block[1, 1] + block[2, 2]) / 3.0)
    return frame.noise**2


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
