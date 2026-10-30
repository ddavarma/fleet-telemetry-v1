"""Parse fleet telemetry API v2 payloads into domain objects."""
from __future__ import annotations

import math
from typing import Any, Callable

from fleet_telemetry.models import Pose, TelemetryFrame

API_VERSION = "v2"
BASE_PATH = "/v2"

Transport = Callable[[str], dict[str, Any]]


class TelemetryParseError(ValueError):
    """Raised when a payload does not match the v2 schema."""


def _parse_covariance(raw: Any) -> tuple[tuple[float, ...], ...]:
    try:
        rows = [[float(v) for v in row] for row in raw]
    except (TypeError, ValueError) as exc:
        raise TelemetryParseError(f"invalid covariance {raw!r}") from exc
    if len(rows) != 6 or any(len(r) != 6 for r in rows):
        raise TelemetryParseError(f"covariance must be 6x6, got shape {[len(r) for r in rows]!r}")
    for r in rows:
        for v in r:
            if not math.isfinite(v):
                raise TelemetryParseError(f"invalid covariance entry {v!r}")
    return tuple(tuple(r) for r in rows)


def parse_frame(payload: dict[str, Any]) -> TelemetryFrame:
    try:
        pos = payload["position"]
        ori = payload["orientation"]
        x, y, z = float(pos["x"]), float(pos["y"]), float(pos["z"])
        qx = float(ori["qx"])
        qy = float(ori["qy"])
        qz = float(ori["qz"])
        qw = float(ori["qw"])
        covariance = _parse_covariance(payload["covariance"])
        health = float(payload["health"])
        robot_id = str(payload["robot_id"])
        timestamp = float(payload["timestamp"])
        battery = float(payload["battery"])
    except KeyError as exc:
        raise TelemetryParseError(f"missing field {exc.args[0]!r}") from exc
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm == 0.0 or not math.isfinite(norm):
        raise TelemetryParseError(f"invalid orientation quaternion {(qx, qy, qz, qw)!r}")
    if not (0.0 <= health <= 1.0):
        raise TelemetryParseError(f"health must be within 0..1, got {health}")
    sigma = math.sqrt((covariance[0][0] + covariance[1][1] + covariance[2][2]) / 3.0)
    if not math.isfinite(sigma) or sigma <= 0:
        raise TelemetryParseError(f"position sigma must be positive, got {sigma}")
    pose = Pose.from_quaternion(x, y, z, qx, qy, qz, qw)
    return TelemetryFrame(
        robot_id=robot_id,
        timestamp=timestamp,
        pose=pose,
        noise=sigma,
        battery=battery,
        covariance=covariance,  # type: ignore[arg-type]
        health=health,
    )


def parse_batch(payload: dict[str, Any]) -> list[TelemetryFrame]:
    version = payload.get("api_version")
    if version != API_VERSION:
        raise TelemetryParseError(f"unsupported api_version {version!r}, expected {API_VERSION!r}")
    return [parse_frame(f) for f in payload["frames"]]


class FleetClient:
    """Thin client over an injected transport (HTTP in prod, fixtures in tests)."""

    def __init__(self, transport: Transport, base_path: str = BASE_PATH):
        self._transport = transport
        self._base = base_path.rstrip("/")

    def latest(self, robot_id: str) -> TelemetryFrame:
        return parse_frame(self._transport(f"{self._base}/robots/{robot_id}/telemetry/latest"))

    def fleet_snapshot(self) -> list[TelemetryFrame]:
        return parse_batch(self._transport(f"{self._base}/fleet/telemetry"))
