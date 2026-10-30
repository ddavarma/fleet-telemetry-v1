from dataclasses import replace

import pytest

from fleet_telemetry.alerts import MIN_HEALTH, check_frame


def kinds(frame):
    return {a.kind for a in check_frame(frame)}


def test_low_health_raises_health_alert(frames):
    assert MIN_HEALTH == 0.5
    assert kinds(frames["amr-03"]) == {"noisy", "health"}
    assert "health" not in kinds(frames["amr-01"])
    assert "health" not in kinds(frames["amr-02"])
    assert "health" not in kinds(frames["amr-04"])


def test_health_threshold_is_overridable_like_other_thresholds(frames):
    frame = frames["amr-03"]
    assert check_frame(frame, min_health=0.99)[-1].kind == "health"
    assert "health" not in {a.kind for a in check_frame(frame, min_health=0.1)}


def test_health_carried_on_parsed_frame(fleet_payload):
    from fleet_telemetry.client import parse_frame

    by_id = {f["robot_id"]: f for f in fleet_payload["frames"]}
    assert parse_frame(by_id["amr-01"]).health == 0.97
    assert parse_frame(by_id["amr-03"]).health == 0.41


def test_health_at_threshold_does_not_alert(frames):
    frame = replace(frames["amr-01"], health=0.5)
    assert "health" not in kinds(frame)
    assert "health" in kinds(replace(frame, health=0.499))
