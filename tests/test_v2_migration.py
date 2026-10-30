"""v2 migration tests: health alerts and quaternion sign invariance."""
import numpy as np
import pytest

from fleet_telemetry.alerts import MIN_HEALTH, check_frame
from fleet_telemetry.client import TelemetryParseError, parse_frame
from fleet_telemetry.fusion import position_sigma
from fleet_telemetry.models import Pose
from fleet_telemetry.orientation import heading_deg, rotation_matrix, tilt_deg


def test_health_below_threshold_raises_health_alert(frames):
    assert MIN_HEALTH == 0.5
    kinds_low = {a.kind for a in check_frame(frames["amr-03"])}
    assert "health" in kinds_low
    kinds_ok = {a.kind for a in check_frame(frames["amr-01"])}
    assert "health" not in kinds_ok
    # Overridable like the other thresholds.
    assert "health" not in {a.kind for a in check_frame(frames["amr-03"], min_health=0.0)}
    assert "health" in {a.kind for a in check_frame(frames["amr-01"], min_health=0.99)}


def test_negative_qw_robot_heading_is_359(frames):
    pose = frames["amr-03"].pose
    assert pose.qw < 0
    assert heading_deg(pose) == pytest.approx(359.0, abs=0.05)


def test_quaternion_and_negation_give_same_rotation():
    qx, qy, qz, qw = 0.006958, -0.002679, 0.008708, -0.999934
    pos = Pose.from_quaternion(15.2, 7.7, 0.1, qx, qy, qz, qw)
    neg = Pose.from_quaternion(15.2, 7.7, 0.1, -qx, -qy, -qz, -qw)
    assert np.allclose(rotation_matrix(pos), rotation_matrix(neg), atol=1e-12)
    assert heading_deg(pos) == pytest.approx(heading_deg(neg))
    assert tilt_deg(pos) == pytest.approx(tilt_deg(neg))


def test_position_sigma_uses_covariance_trace(frames):
    assert position_sigma(frames["amr-01"]) == pytest.approx(0.04)
    assert position_sigma(frames["amr-03"]) == pytest.approx(0.31)


def test_bad_covariance_shape_raises_parse_error(fleet_payload):
    raw = dict(fleet_payload["frames"][0])
    raw["covariance"] = [[0.0016] * 6] * 5
    with pytest.raises(TelemetryParseError, match="covariance"):
        parse_frame(raw)
