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
    """Validate the required 6x6 pose covariance and return it as nested tuples."""
    if not isinstance(raw, (list, tuple)) or len(raw) != 6:
        raise TelemetryParseError(f"covariance must be 6x6, got {raw!r}")
    rows: list[tuple[float, ...]] = []
    for row in raw:
        if not isinstance(row, (list, tuple)) or len(row) != 6:
            raise TelemetryParseError(f"covariance must be 6x6, got {raw!r}")
        try:
            rows.append(tuple(float(v) for v in row))
        except (TypeError, ValueError) as exc:
            raise TelemetryParseError(f"covariance must hold numbers: {exc}") from exc
    return tuple(rows)


def position_sigma(covariance: tuple[tuple[float, ...], ...]) -> float:
    """Scalar 1-sigma position noise (m) equivalent to the old v1 ``noise``.

    ``sqrt(trace(P_pos) / 3)`` where ``P_pos = covariance[0:3][0:3]``.
    """
    return math.sqrt((covariance[0][0] + covariance[1][1] + covariance[2][2]) / 3.0)


def parse_frame(payload: dict[str, Any]) -> TelemetryFrame:
    try:
        pos = payload["position"]
        ori = payload["orientation"]
        pose = Pose(
            x=float(pos["x"]),
            y=float(pos["y"]),
            z=float(pos["z"]),
            qx=float(ori["qx"]),
            qy=float(ori["qy"]),
            qz=float(ori["qz"]),
            qw=float(ori["qw"]),
        )
        covariance = _parse_covariance(payload["covariance"])
        noise = position_sigma(covariance)
        if noise <= 0:
            raise TelemetryParseError(f"noise must be positive, got {noise}")
        health = float(payload["health"])
        if not 0.0 <= health <= 1.0:
            raise TelemetryParseError(f"health must be within 0..1, got {health}")
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
