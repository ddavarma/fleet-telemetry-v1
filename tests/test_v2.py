"""v2-only behaviour: health alerts, sign-invariant quaternions, covariance."""

import dataclasses
import math

import numpy as np
import pytest

from fleet_telemetry.alerts import MIN_HEALTH, check_frame
from fleet_telemetry.client import TelemetryParseError, parse_frame
from fleet_telemetry.fusion import position_variance
from fleet_telemetry.models import Pose
from fleet_telemetry.orientation import heading_deg, rotation_matrix, tilt_deg


def kinds(frame):
    return {a.kind for a in check_frame(frame)}


def test_min_health_threshold_is_half():
    assert MIN_HEALTH == 0.5


def test_health_alert_only_below_threshold(frames):
    assert "health" in kinds(frames["amr-03"])  # health 0.41
    assert "health" not in kinds(frames["amr-01"])  # health 0.97
    assert "health" not in kinds(frames["amr-02"])  # health 0.58
    assert "health" not in kinds(frames["amr-04"])  # health 0.72


def test_health_boundary_and_override(frames):
    assert "health" not in kinds(dataclasses.replace(frames["amr-01"], health=0.5))
    assert "health" in kinds(dataclasses.replace(frames["amr-01"], health=0.49))
    strict = check_frame(frames["amr-01"], min_health=0.99)
    assert "health" in {a.kind for a in strict}


def test_negated_quaternion_gives_same_rotation(frames):
    pose = frames["amr-03"].pose  # qw is negative in the fixture
    assert pose.qw < 0.0
    neg = Pose(pose.x, pose.y, pose.z, qx=-pose.qx, qy=-pose.qy, qz=-pose.qz, qw=-pose.qw)
    assert np.allclose(rotation_matrix(pose), rotation_matrix(neg))
    assert heading_deg(neg) == pytest.approx(heading_deg(pose))
    assert tilt_deg(neg) == pytest.approx(tilt_deg(pose))


def test_negative_qw_heading_is_359(frames):
    assert heading_deg(frames["amr-03"].pose) == pytest.approx(359.0)


def test_position_sigma_is_trace_over_three(frames):
    for frame in frames.values():
        p = frame.covariance
        expected = math.sqrt((p[0][0] + p[1][1] + p[2][2]) / 3.0)
        assert frame.noise == pytest.approx(expected)
        assert position_variance(frame) == pytest.approx(expected**2)


def test_covariance_shape_is_validated(fleet_payload):
    raw = dict(fleet_payload["frames"][0])
    raw["covariance"] = [[0.01] * 6] * 2
    with pytest.raises(TelemetryParseError, match="covariance"):
        parse_frame(raw)
    raw["covariance"] = [[0.01] * 5] * 6
    with pytest.raises(TelemetryParseError, match="covariance"):
        parse_frame(raw)
