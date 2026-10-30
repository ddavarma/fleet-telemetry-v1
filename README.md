# fleet-telemetry (v2)

**This is a demo target for [DriftZero](https://github.com/ddavarma/driftzero)**, an autonomous API-drift repair
agent built for the Meta Global AI Developer Hackathon (Oct–Nov 2026). It is not a maintained library.

A small Python client for a fictional robot-fleet telemetry API, written against **API v1**: Euler roll/pitch/yaw in
degrees and a scalar position `noise`. It parses frames, does orientation math (rotation matrix, heading, tilt),
fuses positions by inverse variance, raises tilt / noise / battery alerts, and schedules charging docks.

The upstream API has shipped **v2**: unit quaternions (sign not canonicalised), a 6×6 pose covariance, a required
health score, and `/v2` paths. See [`CHANGELOG_v2.md`](CHANGELOG_v2.md), [`openapi_v1.json`](openapi_v1.json) and
[`openapi_v2.json`](openapi_v2.json). DriftZero opens pull requests here that migrate it.

```bash
pip install -e '.[test]'
pytest -q
```
