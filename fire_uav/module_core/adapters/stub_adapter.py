from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from fire_uav.core.telemetry import normalize_battery_value
from fire_uav.module_core.adapters.interfaces import IUavAdapter, IUavTelemetryConsumer
from fire_uav.module_core.schema import Route, TelemetrySample
from fire_uav.utils.time import utc_now


class StubUavAdapter(IUavAdapter):
    """
    In-process stub that emits dummy telemetry so the module can run without real hardware.
    """
    def __init__(
        self,
        *,
        default_lat: float = 56.031609,
        default_lon: float = 92.948315,
        default_alt: float = 30.0,
        default_yaw: float = 0.0,
        interval_sec: float = 1.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self.default_lat = default_lat
        self.default_lon = default_lon
        self.default_alt = default_alt
        self.default_yaw = default_yaw
        self.interval_sec = interval_sec
        self.log = logger or logging.getLogger(self.__class__.__name__)
        self._telemetry_callback: IUavTelemetryConsumer | None = None
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self, telemetry_callback: IUavTelemetryConsumer) -> None:
        self._telemetry_callback = telemetry_callback
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._telemetry_loop())
        self.log.info("StubUavAdapter started (emitting fake telemetry every %.1fs)", self.interval_sec)

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self.log.info("StubUavAdapter stopped")

    async def push_route(self, route: Route) -> None:
        self.log.debug("Stub route received (waypoints=%d)", len(route.waypoints))

    async def send_simple_command(self, command: str, payload: dict | None = None) -> None:
        self.log.info("Stub command: %s payload=%s", command, payload)

    async def _telemetry_loop(self) -> None:
        """
        Periodically emit a synthetic telemetry sample to keep pipelines alive.
        """
        while self._running:
            if self._telemetry_callback:
                battery_fraction, battery_percent = normalize_battery_value(1.0)
                sample = TelemetrySample(
                    lat=self.default_lat,
                    lon=self.default_lon,
                    alt=self.default_alt,
                    yaw=self.default_yaw,
                    pitch=0.0,
                    roll=0.0,
                    battery=battery_fraction,
                    battery_percent=battery_percent,
                    timestamp=utc_now(),
                    source="stub",
                )
                try:
                    await self._telemetry_callback.on_telemetry(sample)
                except Exception:  # noqa: BLE001
                    self.log.exception("Stub telemetry callback failed")
            await asyncio.sleep(self.interval_sec)


__all__ = ["StubUavAdapter"]
