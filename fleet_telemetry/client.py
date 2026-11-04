"""Parse fleet telemetry API v2 payloads into domain objects."""
from __future__ import annotations

from typing import Any, Callable

from fleet_telemetry.models import Pose, TelemetryFrame

API_VERSION = "v2"
BASE_PATH = "/v2"

Transport = Callable[[str], dict[str, Any]]


class TelemetryParseError(ValueError):
    """Raised when a payload does not match the v2 schema."""


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
        return TelemetryFrame(
            robot_id=str(payload["robot_id"]),
            timestamp=float(payload["timestamp"]),
            pose=pose,
            covariance=payload["covariance"],
            health=float(payload["health"]),
            battery=float(payload["battery"]),
        )
    except KeyError as exc:
        raise TelemetryParseError(f"missing field {exc.args[0]!r}") from exc
    except (ValueError, TypeError) as exc:
        raise TelemetryParseError(str(exc)) from exc


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
