"""Parse fleet telemetry API v2 payloads into domain objects."""
from __future__ import annotations

import math
from typing import Any, Callable

from fleet_telemetry.models import Pose, TelemetryFrame, position_sigma_from_covariance

API_VERSION = "v2"
BASE_PATH = "/v2"

Transport = Callable[[str], dict[str, Any]]


class TelemetryParseError(ValueError):
    """Raised when a payload does not match the v2 schema."""


def _parse_covariance(raw: Any) -> tuple[tuple[float, ...], ...]:
    if (
        not isinstance(raw, (list, tuple))
        or len(raw) != 6
        or any(not isinstance(row, (list, tuple)) or len(row) != 6 for row in raw)
    ):
        try:
            desc = f"{len(raw)}x{[len(row) for row in raw]}"
        except TypeError:
            desc = type(raw).__name__
        raise TelemetryParseError(f"covariance must be 6x6, got {desc}")
    try:
        return tuple(tuple(float(v) for v in row) for row in raw)
    except (TypeError, ValueError) as exc:
        raise TelemetryParseError(f"covariance must hold numbers: {exc}") from exc


def parse_frame(payload: dict[str, Any]) -> TelemetryFrame:
    try:
        pos = payload["position"]
        ori = payload["orientation"]
        qx = float(ori["qx"])
        qy = float(ori["qy"])
        qz = float(ori["qz"])
        qw = float(ori["qw"])
        if math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) <= 0.0:
            raise TelemetryParseError("orientation quaternion must be non-zero")
        pose = Pose.from_quaternion(
            x=float(pos["x"]),
            y=float(pos["y"]),
            z=float(pos["z"]),
            qx=qx,
            qy=qy,
            qz=qz,
            qw=qw,
        )
        covariance = _parse_covariance(payload["covariance"])
        try:
            noise = position_sigma_from_covariance(covariance)
        except ValueError as exc:
            raise TelemetryParseError(str(exc)) from exc
        health = float(payload["health"])
        if not 0.0 <= health <= 1.0:
            raise TelemetryParseError(f"health {health!r} outside 0..1")
        return TelemetryFrame(
            robot_id=str(payload["robot_id"]),
            timestamp=float(payload["timestamp"]),
            pose=pose,
            covariance=covariance,
            health=health,
            battery=float(payload["battery"]),
            noise=noise,
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
