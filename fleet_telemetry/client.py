"""Parse fleet telemetry API v2 payloads into domain objects."""
from __future__ import annotations

import math
from typing import Any, Callable

from fleet_telemetry.models import Pose, TelemetryFrame
from fleet_telemetry.orientation import euler_from_quaternion, normalize_quaternion

API_VERSION = "v2"
BASE_PATH = "/v2"

Transport = Callable[[str], dict[str, Any]]


class TelemetryParseError(ValueError):
    """Raised when a payload does not match the v2 schema."""


def _parse_covariance(raw: Any) -> tuple[tuple[float, ...], ...]:
    try:
        rows = tuple(tuple(float(v) for v in row) for row in raw)
    except TypeError as exc:
        raise TelemetryParseError(f"covariance must be a 6x6 nested array: {exc}") from exc
    if len(rows) != 6 or any(len(row) != 6 for row in rows):
        raise TelemetryParseError(
            f"covariance must be a 6x6 nested array, got {[len(r) for r in rows]}"
        )
    return rows


def parse_frame(payload: dict[str, Any]) -> TelemetryFrame:
    try:
        pos = payload["position"]
        ori = payload["orientation"]
        qx, qy, qz, qw = normalize_quaternion(
            float(ori["qx"]),
            float(ori["qy"]),
            float(ori["qz"]),
            float(ori["qw"]),
        )
        covariance = _parse_covariance(payload["covariance"])
        health = float(payload["health"])
        trace_pos = covariance[0][0] + covariance[1][1] + covariance[2][2]
        noise = math.sqrt(trace_pos / 3.0)
        if noise <= 0:
            raise TelemetryParseError(f"noise must be positive, got {noise}")
        roll, pitch, yaw = euler_from_quaternion(qx, qy, qz, qw)
        pose = Pose(
            x=float(pos["x"]),
            y=float(pos["y"]),
            z=float(pos["z"]),
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            qx=qx,
            qy=qy,
            qz=qz,
            qw=qw,
        )
        return TelemetryFrame(
            robot_id=str(payload["robot_id"]),
            timestamp=float(payload["timestamp"]),
            pose=pose,
            noise=noise,
            battery=float(payload["battery"]),
            covariance=covariance,
            health=health,
        )
    except KeyError as exc:
        raise TelemetryParseError(f"missing field {exc.args[0]!r}") from exc


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
