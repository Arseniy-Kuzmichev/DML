from __future__ import annotations

import asyncio
import base64
import mimetypes
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from .config import AppConfig, MetricThreshold
from .logger import write_jsonl

DUMMY_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class RequestResult:
    endpoint: str
    method: str
    success: bool
    status_code: int | None
    response_time_ms: float
    error: str | None = None
    response_json: Any | None = None


@dataclass
class MetricsSnapshot:
    timestamp: str
    severity: str
    health_status: bool
    response_time_ms: float
    p95_latency_ms: float
    error_rate_percent: float
    consecutive_failures: int
    total_requests: int
    failed_requests: int
    details: list[dict[str, Any]]


class FastAPIMonitor:
    def __init__(self, config: AppConfig, logger) -> None:
        self.config = config
        self.logger = logger
        self.base_url = config.service.base_url.rstrip("/")
        self.consecutive_failures = 0
        self.last_alert_at: dict[str, float] = {}
        self.last_severity: str | None = None
        self.service_process: asyncio.subprocess.Process | None = None

        self.metrics_file = self._resolve_path(config.logging.metrics_file)
        self.test_images_dir = self._resolve_path(config.monitoring.test_images_dir)

    def _resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return self.config.project_root / path

    async def maybe_start_service(self) -> None:
        command = self.config.service.start_command.strip()
        if not command:
            self.logger.info("FastAPI service is expected to be already running")
            return

        self.logger.info(
            "Starting FastAPI service programmatically", extra={"command": command}
        )
        self.service_process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.config.project_root.parent),
        )
        await asyncio.sleep(self.config.service.startup_wait_seconds)

    async def stop_service(self) -> None:
        if not self.service_process:
            return
        if self.service_process.returncode is not None:
            return

        self.logger.info("Stopping FastAPI service started by monitor")
        self.service_process.terminate()
        try:
            await asyncio.wait_for(self.service_process.wait(), timeout=5)
        except asyncio.TimeoutError:
            self.service_process.kill()
            await self.service_process.wait()

    async def run_forever(self) -> None:
        await self.maybe_start_service()
        try:
            while True:
                await self.run_once()
                await asyncio.sleep(self.config.monitoring.check_interval_seconds)
        finally:
            await self.stop_service()

    async def run_once(self) -> MetricsSnapshot:
        timeout = httpx.Timeout(self.config.monitoring.request_timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            results: list[RequestResult] = []

            health = await self.check_health(client)
            results.append(health)

            for _ in range(self.config.monitoring.samples_per_check):
                prediction = await self.check_predict(client)
                results.append(prediction)

        metrics = self.calculate_metrics(results)
        self.log_metrics(metrics)
        self.check_alerts(metrics)
        return metrics

    async def check_health(self, client: httpx.AsyncClient) -> RequestResult:
        url = f"{self.base_url}{self.config.endpoints.health}"
        start = time.perf_counter()
        try:
            response = await client.get(url)
            elapsed_ms = (time.perf_counter() - start) * 1000
            response_json = self._safe_json(response)
            success = response.status_code == 200
            self._update_consecutive_failures(success)
            return RequestResult(
                endpoint=self.config.endpoints.health,
                method="GET",
                success=success,
                status_code=response.status_code,
                response_time_ms=round(elapsed_ms, 2),
                response_json=response_json,
                error=(
                    None
                    if success
                    else f"Unexpected status code: {response.status_code}"
                ),
            )
        except (
            Exception
        ) as exc:  # noqa: BLE001 - мониторинг должен ловить все сетевые ошибки
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._update_consecutive_failures(False)
            return RequestResult(
                endpoint=self.config.endpoints.health,
                method="GET",
                success=False,
                status_code=None,
                response_time_ms=round(elapsed_ms, 2),
                error=str(exc),
            )

    async def check_predict(self, client: httpx.AsyncClient) -> RequestResult:
        image_path, image_bytes, content_type = self._get_test_image()
        url = f"{self.base_url}{self.config.endpoints.predict}"
        start = time.perf_counter()
        try:
            files = {
                self.config.endpoints.predict_file_field: (
                    image_path.name,
                    image_bytes,
                    content_type,
                )
            }
            response = await client.post(url, files=files)
            elapsed_ms = (time.perf_counter() - start) * 1000
            response_json = self._safe_json(response)
            success, validation_error = self._validate_prediction_response(
                response, response_json
            )
            self._update_consecutive_failures(success)

            result = RequestResult(
                endpoint=self.config.endpoints.predict,
                method="POST",
                success=success,
                status_code=response.status_code,
                response_time_ms=round(elapsed_ms, 2),
                response_json=response_json,
                error=validation_error,
            )

            if success:
                self.logger.info(
                    "Prediction request completed",
                    extra={
                        "endpoint": self.config.endpoints.predict,
                        "image": image_path.name,
                        "response_time_ms": result.response_time_ms,
                        "prediction": response_json,
                    },
                )
            return result
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._update_consecutive_failures(False)
            return RequestResult(
                endpoint=self.config.endpoints.predict,
                method="POST",
                success=False,
                status_code=None,
                response_time_ms=round(elapsed_ms, 2),
                error=str(exc),
            )

    def _update_consecutive_failures(self, success: bool) -> None:
        if success:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1

    def _safe_json(self, response: httpx.Response) -> Any | None:
        try:
            return response.json()
        except Exception:  # noqa: BLE001
            return None

    def _validate_prediction_response(
        self, response: httpx.Response, response_json: Any | None
    ) -> tuple[bool, str | None]:
        if response.status_code != 200:
            return False, f"Unexpected status code: {response.status_code}"
        if response_json is None:
            return False, "Response is not valid JSON"

        expected_keys = self.config.endpoints.expected_prediction_keys
        if expected_keys:
            if not isinstance(response_json, dict):
                return (
                    False,
                    "Response JSON must be an object when expected_prediction_keys is configured",
                )
            missing = [key for key in expected_keys if key not in response_json]
            if missing:
                return False, f"Missing expected keys: {', '.join(missing)}"

        return True, None

    def _get_test_image(self) -> tuple[Path, bytes, str]:
        self.test_images_dir.mkdir(parents=True, exist_ok=True)
        candidates = sorted(
            path
            for path in self.test_images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if candidates:
            path = candidates[int(time.time()) % len(candidates)]
            content_type = (
                mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            )
            return path, path.read_bytes(), content_type

        # Фолбэк нужен, чтобы мониторинг запускался даже без тестовой картинки.
        # Для реальной проверки модели лучше положить jpg/png в папку test_images.
        return Path("dummy.png"), DUMMY_PNG_BYTES, "image/png"

    def calculate_metrics(self, results: list[RequestResult]) -> MetricsSnapshot:
        latencies = [item.response_time_ms for item in results]
        failed_requests = sum(1 for item in results if not item.success)
        total_requests = len(results)

        response_time_ms = statistics.mean(latencies) if latencies else 0.0
        p95_latency_ms = self._percentile(latencies, 95)
        error_rate_percent = (
            (failed_requests / total_requests * 100) if total_requests else 0.0
        )
        health_status = next(
            (
                item.success
                for item in results
                if item.endpoint == self.config.endpoints.health
            ),
            False,
        )

        severity = self.evaluate_severity(
            health_status=health_status,
            response_time_ms=response_time_ms,
            p95_latency_ms=p95_latency_ms,
            error_rate_percent=error_rate_percent,
            consecutive_failures=self.consecutive_failures,
        )

        return MetricsSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            severity=severity,
            health_status=health_status,
            response_time_ms=round(response_time_ms, 2),
            p95_latency_ms=round(p95_latency_ms, 2),
            error_rate_percent=round(error_rate_percent, 2),
            consecutive_failures=self.consecutive_failures,
            total_requests=total_requests,
            failed_requests=failed_requests,
            details=[asdict(item) for item in results],
        )

    def _percentile(self, values: list[float], percentile: int) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        if len(sorted_values) == 1:
            return sorted_values[0]
        index = (len(sorted_values) - 1) * percentile / 100
        lower = int(index)
        upper = min(lower + 1, len(sorted_values) - 1)
        weight = index - lower
        return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight

    def _threshold_severity(self, value: float, threshold: MetricThreshold) -> str:
        if value >= threshold.critical:
            return "critical"
        if value >= threshold.warning:
            return "warning"
        return "normal"

    def evaluate_severity(
        self,
        health_status: bool,
        response_time_ms: float,
        p95_latency_ms: float,
        error_rate_percent: float,
        consecutive_failures: int,
    ) -> str:
        severities = [
            self._threshold_severity(
                response_time_ms, self.config.thresholds.response_time_ms
            ),
            self._threshold_severity(
                p95_latency_ms, self.config.thresholds.p95_latency_ms
            ),
            self._threshold_severity(
                error_rate_percent, self.config.thresholds.error_rate_percent
            ),
            self._threshold_severity(
                consecutive_failures, self.config.thresholds.consecutive_failures
            ),
        ]
        if not health_status:
            severities.append("critical")
        if "critical" in severities:
            return "critical"
        if "warning" in severities:
            return "warning"
        return "normal"

    def log_metrics(self, metrics: MetricsSnapshot) -> None:
        payload = asdict(metrics)
        write_jsonl(self.metrics_file, payload)

        message = (
            f"health={metrics.health_status} | "
            f"avg={metrics.response_time_ms}ms | "
            f"p95={metrics.p95_latency_ms}ms | "
            f"errors={metrics.error_rate_percent}% | "
            f"consecutive_failures={metrics.consecutive_failures}"
        )

        if metrics.severity == "critical":
            self.logger.error(
                message, extra={"severity": metrics.severity, "metrics": payload}
            )
        elif metrics.severity == "warning":
            self.logger.warning(
                message, extra={"severity": metrics.severity, "metrics": payload}
            )
        else:
            self.logger.info(
                message, extra={"severity": metrics.severity, "metrics": payload}
            )

    def check_alerts(self, metrics: MetricsSnapshot) -> None:
        if not self.config.alerts.enabled:
            return

        if metrics.severity == "normal":
            self.last_severity = "normal"
            return

        now = time.time()
        cooldown_seconds = self.config.alerts.cooldown_minutes * 60
        last_alert = self.last_alert_at.get(metrics.severity, 0)
        should_alert = (
            metrics.severity != self.last_severity
            or now - last_alert >= cooldown_seconds
        )

        if not should_alert:
            return

        self.last_alert_at[metrics.severity] = now
        self.last_severity = metrics.severity

        message = self._build_alert_message(metrics)
        if metrics.severity == "critical":
            self.logger.critical(
                message, extra={"severity": "critical", "metrics": asdict(metrics)}
            )
        else:
            self.logger.warning(
                message, extra={"severity": "warning", "metrics": asdict(metrics)}
            )

    def _build_alert_message(self, metrics: MetricsSnapshot) -> str:
        reasons: list[str] = []
        if not metrics.health_status:
            reasons.append("/health is not OK")
        if metrics.response_time_ms >= self.config.thresholds.response_time_ms.warning:
            reasons.append(f"avg response time {metrics.response_time_ms}ms")
        if metrics.p95_latency_ms >= self.config.thresholds.p95_latency_ms.warning:
            reasons.append(f"p95 latency {metrics.p95_latency_ms}ms")
        if (
            metrics.error_rate_percent
            >= self.config.thresholds.error_rate_percent.warning
        ):
            reasons.append(f"error rate {metrics.error_rate_percent}%")
        if (
            metrics.consecutive_failures
            >= self.config.thresholds.consecutive_failures.warning
        ):
            reasons.append(f"consecutive failures {metrics.consecutive_failures}")
        return (
            f"ALERT: service state is {metrics.severity.upper()} ({'; '.join(reasons)})"
        )
