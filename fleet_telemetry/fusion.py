"""Combine observations from several frames using the per-frame covariance estimate."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from fleet_telemetry.models import TelemetryFrame
from fleet_telemetry.orientation import heading_deg


def position_covariance(frame: TelemetryFrame) -> np.ndarray:
    """3x3 position covariance (m^2): the ``covariance[0:3][0:3]`` block."""
    return np.asarray(frame.covariance, dtype=float)[:3, :3].copy()


def position_variance(frame: TelemetryFrame) -> float:
    """Scalar position variance (m^2) equivalent to the old squared noise sigma.

    Defined as ``trace(P_pos) / 3`` where ``P_pos = covariance[0:3][0:3]``, so
    isotropic covariances keep exactly the old fusion weights and noise alerts.
    """
    return float(np.trace(position_covariance(frame)) / 3.0)


def fuse_positions(frames: Sequence[TelemetryFrame]) -> tuple[np.ndarray, float]:
    """Inverse-covariance weighted position and the fused 1-sigma, e.g. for a
    landmark seen by several robots."""
    if not frames:
        raise ValueError("need at least one frame")
    infos = [np.linalg.inv(position_covariance(f)) for f in frames]
    info_sum = sum(infos)
    fused_cov = np.linalg.inv(info_sum)
    points = np.array([[f.pose.x, f.pose.y, f.pose.z] for f in frames])
    fused = fused_cov @ sum(info @ p for info, p in zip(infos, points))
    sigma = float(np.sqrt(np.trace(fused_cov) / 3.0))
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
