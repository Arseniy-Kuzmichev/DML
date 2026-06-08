from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ServiceConfig:
    host: str = "localhost"
    port: int = 8000
    base_url: str = "http://localhost:8000"
    start_command: str = ""
    startup_wait_seconds: int = 5


@dataclass(frozen=True)
class EndpointsConfig:
    health: str = "/health"
    predict: str = "/predict"
    predict_file_field: str = "file"
    expected_prediction_keys: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MonitoringConfig:
    check_interval_seconds: int = 30
    samples_per_check: int = 3
    request_timeout_seconds: int = 10
    test_images_dir: str = "test_images"


@dataclass(frozen=True)
class MetricThreshold:
    warning: float
    critical: float


@dataclass(frozen=True)
class ThresholdsConfig:
    response_time_ms: MetricThreshold
    p95_latency_ms: MetricThreshold
    error_rate_percent: MetricThreshold
    consecutive_failures: MetricThreshold


@dataclass(frozen=True)
class AlertsConfig:
    enabled: bool = True
    cooldown_minutes: int = 5


@dataclass(frozen=True)
class LoggingConfig:
    console_colors: bool = True
    log_file: str = "logs/monitoring.log"
    metrics_file: str = "logs/metrics.jsonl"


@dataclass(frozen=True)
class AppConfig:
    service: ServiceConfig
    endpoints: EndpointsConfig
    monitoring: MonitoringConfig
    thresholds: ThresholdsConfig
    alerts: AlertsConfig
    logging: LoggingConfig
    project_root: Path


def _merge(defaults: dict[str, Any], incoming: dict[str, Any] | None) -> dict[str, Any]:
    result = dict(defaults)
    if not incoming:
        return result
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _threshold(
    data: dict[str, Any], name: str, warning: float, critical: float
) -> MetricThreshold:
    raw = data.get(name, {})
    return MetricThreshold(
        warning=float(raw.get("warning", warning)),
        critical=float(raw.get("critical", critical)),
    )


def load_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}

    defaults: dict[str, Any] = {
        "service": {
            "host": "localhost",
            "port": 8000,
            "base_url": "http://localhost:8000",
            "start_command": "",
            "startup_wait_seconds": 5,
        },
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_file_field": "file",
            "expected_prediction_keys": [],
        },
        "monitoring": {
            "check_interval_seconds": 30,
            "samples_per_check": 3,
            "request_timeout_seconds": 10,
            "test_images_dir": "test_images",
        },
        "thresholds": {
            "response_time_ms": {"warning": 2000, "critical": 5000},
            "p95_latency_ms": {"warning": 3000, "critical": 6000},
            "error_rate_percent": {"warning": 10, "critical": 25},
            "consecutive_failures": {"warning": 3, "critical": 5},
        },
        "alerts": {"enabled": True, "cooldown_minutes": 5},
        "logging": {
            "console_colors": True,
            "log_file": "logs/monitoring.log",
            "metrics_file": "logs/metrics.jsonl",
        },
    }
    data = _merge(defaults, loaded)

    thresholds = data["thresholds"]
    project_root = path.parent.parent.resolve()

    return AppConfig(
        service=ServiceConfig(**data["service"]),
        endpoints=EndpointsConfig(**data["endpoints"]),
        monitoring=MonitoringConfig(**data["monitoring"]),
        thresholds=ThresholdsConfig(
            response_time_ms=_threshold(thresholds, "response_time_ms", 2000, 5000),
            p95_latency_ms=_threshold(thresholds, "p95_latency_ms", 3000, 6000),
            error_rate_percent=_threshold(thresholds, "error_rate_percent", 10, 25),
            consecutive_failures=_threshold(thresholds, "consecutive_failures", 3, 5),
        ),
        alerts=AlertsConfig(**data["alerts"]),
        logging=LoggingConfig(**data["logging"]),
        project_root=project_root,
    )
