from __future__ import annotations

import asyncio
import logging
from typing import Any
from uuid import uuid4

import httpx

from fire_uav.core.telemetry import normalize_battery_value
from fire_uav.module_core.adapters.interfaces import IUavAdapter, IUavTelemetryConsumer
from fire_uav.module_core.contract.v1 import RouteV1
from fire_uav.module_core.schema import Route, TelemetrySample
from fire_uav.services.bus import bus
from fire_uav.utils.time import utc_now


class ModuleAdapter(IUavAdapter):
    def __init__(
        self,
        *,
        telemetry_base_url: str = "http://127.0.0.1:8000",
        telemetry_poll_interval_sec: float = 1.0,
        request_timeout_sec: float = 2.0,
        uav_id: str = "uav",
        logger: logging.Logger | None = None,
    ) -> None:
        self.telemetry_base_url = telemetry_base_url.rstrip("/")
        self.telemetry_poll_interval_sec = telemetry_poll_interval_sec
        self.request_timeout_sec = request_timeout_sec
        self.uav_id = uav_id

        self.log = logger or logging.getLogger(self.__class__.__name__)

        self._telemetry_callback: IUavTelemetryConsumer | None = None
        self._task: asyncio.Task | None = None
        self._client: httpx.AsyncClient | None = None
        self._running = False
        self._loop: asyncio.AbstractEventLoop | None = None

    async def start(self, telemetry_callback: IUavTelemetryConsumer) -> None:
        if self._running:
            return

        self._running = True
        self._loop = asyncio.get_running_loop()
        self._telemetry_callback = telemetry_callback

        self._client = httpx.AsyncClient(
            base_url=self.telemetry_base_url,
            timeout=self.request_timeout_sec,
            trust_env=False,
        )

        bus.subscribe("route_sent", self._on_route_sent)

        self._task = asyncio.create_task(self._telemetry_loop())

        self.log.info("ModuleAdapter started")

    async def stop(self) -> None:
        if not self._running:
            return

        self._running = False

        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        if self._client is not None:
            await self._client.aclose()
            self._client = None

        self.log.info("ModuleAdapter stopped")

    async def push_route(self, route: Route) -> None:
        await self.send_route(route)

    async def send_route(self, route: Route | RouteV1) -> None:
        if isinstance(route, RouteV1):
            await self.send_route_v1(route)
            return

        route_payload = route.model_dump(mode="json")
        route_v1 = RouteV1(
            protocol_version=1,
            uav_id=self.uav_id,
            route_id=f"route-{uuid4()}",
            mode="mission",
            created_at=utc_now(),
            waypoints=[
                {
                    "lat": wp["lat"],
                    "lon": wp["lon"],
                    "alt": wp["alt"],
                    "speed_mps": None,
                    "loiter_radius_m": None,
                    "action": None,
                }
                for wp in route_payload.get("waypoints", [])
            ],
        )
        await self.send_route_v1(route_v1)

    async def send_route_v1(self, route: RouteV1) -> None:
        await self._post_json("/mission", route.model_dump(mode="json"))

    async def send_simple_command(
        self,
        command: str,
        payload: dict | None = None,
    ) -> None:
        command_payload = {
            "protocol_version": 1,
            "uav_id": self.uav_id,
            "timestamp": utc_now().isoformat(),
            "command_id": str(uuid4()),
            "type": command,
            "params": payload or {},
        }

        await self._post_json("/command", command_payload)

    async def _telemetry_loop(self) -> None:
        while self._running:
            try:
                data = await self._get_json("/telemetry")

                battery_fraction = None
                battery_percent = None

                if data.get("battery_percent") is not None:
                    battery_fraction, battery_percent = normalize_battery_value(
                        data["battery_percent"]
                    )

                sample = TelemetrySample(
                    lat=data["lat"],
                    lon=data["lon"],
                    alt=data["alt"],
                    yaw=data.get("yaw", 0.0),
                    pitch=data.get("pitch", 0.0),
                    roll=data.get("roll", 0.0),
                    battery=battery_fraction if battery_fraction is not None else 1.0,
                    battery_percent=battery_percent,
                    timestamp=utc_now(),
                    source="telemetry_api",
                )

                if self._telemetry_callback is not None:
                    await self._telemetry_callback.on_telemetry(sample)

            except asyncio.CancelledError:
                raise

            except Exception:
                self.log.exception("Telemetry polling failed")

            await asyncio.sleep(self.telemetry_poll_interval_sec)

    def _on_route_sent(self, payload: dict[str, Any]) -> None:
        if self._loop is None:
            return

        try:
            route = RouteV1(**payload)
        except Exception:
            self.log.exception("Invalid RouteV1 payload received from route_sent event")
            return

        future = asyncio.run_coroutine_threadsafe(
            self.send_route_v1(route),
            self._loop,
        )

        future.add_done_callback(self._log_future_exception)

    async def _get_json(self, path: str) -> dict[str, Any]:
        client = self._require_client()

        response = await client.get(path)
        response.raise_for_status()

        return response.json()

    async def _post_json(
        self,
        path: str,
        payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        client = self._require_client()

        response = await client.post(path, json=payload)
        response.raise_for_status()

        if not response.content:
            return None

        return response.json()

    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("ModuleAdapter is not started")

        return self._client

    def _log_future_exception(self, future: Any) -> None:
        exc = future.exception()
        if exc is not None:
            self.log.error("Background adapter operation failed", exc_info=exc)


__all__ = ["ModuleAdapter"]