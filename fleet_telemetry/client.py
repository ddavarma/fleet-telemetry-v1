"""Parse fleet telemetry API v2 payloads into domain objects."""
from __future__ import annotations

import math
from typing import Any, Callable

from fleet_telemetry.models import Pose, TelemetryFrame, normalize_quaternion, quaternion_to_euler

API_VERSION = "v2"
BASE_PATH = "/v2"

Transport = Callable[[str], dict[str, Any]]


class TelemetryParseError(ValueError):
    """Raised when a payload does not match the v2 schema."""


def _parse_covariance(value: Any) -> tuple[tuple[float, ...], ...]:
    """Validate the required 6x6 pose covariance and return it as nested tuples."""
    try:
        rows = [[float(v) for v in row] for row in value]
    except (TypeError, ValueError) as exc:
        raise TelemetryParseError(f"covariance must be a 6x6 array of numbers: {exc}") from exc
    if len(rows) != 6 or any(len(row) != 6 for row in rows):
        raise TelemetryParseError(f"covariance must be 6x6, got {len(rows)}x{[len(r) for r in rows]}")
    if any(not math.isfinite(v) for row in rows for v in row):
        raise TelemetryParseError("covariance must hold finite values")
    return tuple(tuple(row) for row in rows)


def parse_frame(payload: dict[str, Any]) -> TelemetryFrame:
    try:
        pos = payload["position"]
        ori = payload["orientation"]
        qx = float(ori["qx"])
        qy = float(ori["qy"])
        qz = float(ori["qz"])
        qw = float(ori["qw"])
        covariance = _parse_covariance(payload["covariance"])
        health = float(payload["health"])
        try:
            nqx, nqy, nqz, nqw = normalize_quaternion(qx, qy, qz, qw)
        except ValueError as exc:
            raise TelemetryParseError(f"orientation quaternion must have nonzero norm: {exc}") from exc
        # Source-compatible Euler view derived from the quaternion. The wire
        # carries ~6-decimal quaternions, so round to millidegrees (well
        # within v1's 0.1-degree precision); orientation math always uses the
        # quaternion itself, never these angles.
        eroll, epitch, eyaw = quaternion_to_euler(nqx, nqy, nqz, nqw)
        pose = Pose(
            x=float(pos["x"]),
            y=float(pos["y"]),
            z=float(pos["z"]),
            roll=round(eroll, 3),
            pitch=round(epitch, 3),
            yaw=round(eyaw, 3),
            qx=nqx,
            qy=nqy,
            qz=nqz,
            qw=nqw,
        )
        sigma = math.sqrt((covariance[0][0] + covariance[1][1] + covariance[2][2]) / 3.0)
        if not math.isfinite(sigma) or sigma <= 0:
            raise TelemetryParseError(f"covariance position block must imply positive sigma, got {sigma}")
        return TelemetryFrame(
            robot_id=str(payload["robot_id"]),
            timestamp=float(payload["timestamp"]),
            pose=pose,
            noise=sigma,
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
