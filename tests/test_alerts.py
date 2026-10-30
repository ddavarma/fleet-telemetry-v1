from fleet_telemetry.alerts import MIN_HEALTH, check_frame
from fleet_telemetry.client import parse_frame
from fleet_telemetry.report import format_summary, summarize


def kinds(frame):
    return {a.kind for a in check_frame(frame)}


def test_tilted_robot_raises_tilt_alert(frames):
    assert kinds(frames["amr-02"]) == {"tilt"}


def test_noisy_localisation_raises_noisy_alert(frames):
    assert kinds(frames["amr-03"]) == {"noisy", "health"}


def test_low_battery_and_healthy_robot(frames):
    assert kinds(frames["amr-04"]) == {"battery"}
    assert kinds(frames["amr-01"]) == set()


def test_fleet_summary(frames):
    summary = summarize(list(frames.values()))
    assert summary.n_robots == 4
    assert summary.healthy_ids == ["amr-01"]
    assert len(summary.alerts) == 4
    text = format_summary(summary)
    assert "ALERT [tilt] amr-02" in text
    assert "amr-04  heading  315.0 deg" in text


def test_low_health_raises_health_alert(fleet_payload):
    assert MIN_HEALTH == 0.5
    raw = dict(fleet_payload["frames"][0])
    raw["health"] = 0.2
    assert kinds(parse_frame(raw)) == {"health"}


def test_health_at_threshold_is_ok(fleet_payload):
    raw = dict(fleet_payload["frames"][0])
    raw["health"] = 0.5
    assert kinds(parse_frame(raw)) == set()


def test_health_threshold_is_overridable(frames):
    assert kinds(frames["amr-01"]) == set()
    assert "health" in {a.kind for a in check_frame(frames["amr-01"], min_health=0.99)}
