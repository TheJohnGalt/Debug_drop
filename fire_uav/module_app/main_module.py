"""Headless onboard runtime entry point."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import datetime

import uvicorn

import fire_uav.infrastructure.providers as deps
from fire_uav.bootstrap import init_module_core
from fire_uav.config.logging_config import setup_logging
from fire_uav.module_app.config import load_module_settings
from fire_uav.module_app.health_api import app as health_app, configure_health, health_state
from fire_uav.core.telemetry import coerce_battery_percent
from fire_uav.module_core.adapters import IUavAdapter, IUavTelemetryConsumer
from fire_uav.module_core.drivers.registry import create_driver
from fire_uav.module_core.detections import DetectionAggregator, DetectionPipeline, DetectionBatchPayload, RawDetectionPayload
from fire_uav.module_core.factories import get_energy_model, get_geo_projector
from fire_uav.module_core.interfaces.energy import IEnergyModel
from fire_uav.module_core.route.base_location import resolve_base_location
from fire_uav.module_core.route.python_planner import PythonRoutePlanner
from fire_uav.module_core.schema import Route, TelemetrySample, Waypoint
from fire_uav.services.bus import Event, bus
from fire_uav.services.visualizer_adapter import VisualizerAdapter
from fire_uav.services.telemetry.transmitter import Transmitter
from fire_uav.services.telemetry_ingest import TelemetryIngestContext, ingest_telemetry
from fire_uav.utils.time import utc_now

log = logging.getLogger(__name__)
_health_server: uvicorn.Server | None = None


class _ModuleTelemetryConsumer(IUavTelemetryConsumer):
    """Feeds telemetry into downstream pipelines and keeps latest sample in memory."""

    def __init__(
        self,
        *,
        pipeline: DetectionPipeline,
        planner: PythonRoutePlanner,
        energy_model: IEnergyModel,
        visualizer: VisualizerAdapter | None,
        adapter: IUavAdapter,
        settings,
    ) -> None:
        self.pipeline = pipeline
        self.planner = planner
        self.energy_model = energy_model
        self.latest: TelemetrySample | None = None
        self.visualizer = visualizer
        self.adapter = adapter
        self.settings = settings
        self._emergency_active = False
        self._base_warned = False
        self._ingest_context = TelemetryIngestContext(
            planner=planner,
            health_state_updater=health_state.update_telemetry,
            visualizer=visualizer,
        )

    def _resolve_base_location(self, sample: TelemetrySample) -> tuple[float, float]:
        route = Route(version=1, waypoints=[], active_index=None)
        resolved = resolve_base_location(self.settings, route, sample)
        if resolved is not None:
            return resolved.lat, resolved.lon
        if not self._base_warned:
            log.warning("Base location unavailable; falling back to current telemetry position.")
            self._base_warned = True
        return sample.lat, sample.lon

    async def on_telemetry(self, sample: TelemetrySample) -> None:
        self.latest = sample
        await ingest_telemetry(sample, context=self._ingest_context)
        log.debug(
            "Telemetry update: lat=%.6f lon=%.6f alt=%.1f batt=%.2f",
            sample.lat,
            sample.lon,
            sample.alt,
            sample.battery,
        )

        battery_percent = coerce_battery_percent(sample.battery, sample.battery_percent)
        critical_threshold = getattr(self.settings, "critical_battery_percent", 10.0)

        if battery_percent is None or battery_percent > critical_threshold:
            return
        if self._emergency_active:
            return

        base_lat, base_lon = self._resolve_base_location(sample)
        return_route = Route(
            version=1,
            waypoints=[
                Waypoint(lat=sample.lat, lon=sample.lon, alt=sample.alt),
                Waypoint(lat=base_lat, lon=base_lon, alt=sample.alt),
            ],
            active_index=0,
        )
        self._emergency_active = True
        log.warning(
            "Critical battery detected (%.1f%% <= %.1f%%). Sending return-to-base route.",
            battery_percent,
            critical_threshold,
        )
        bus.emit(Event.BATTERY_CRITICAL, {"battery_percent": battery_percent})

        await self.adapter.push_route(return_route)


def _make_transmitter() -> Transmitter | None:
    cfg = load_module_settings()
    if not cfg.ground_station_enabled:
        return None
    try:
        return Transmitter(
            host=cfg.ground_station_host,
            port=cfg.ground_station_port,
            udp=cfg.ground_station_udp,
        )
    except Exception:  # noqa: BLE001
        log.exception("Failed to connect transmitter to ground station")
        return None


def _build_adapter(cfg) -> IUavAdapter:
    return create_driver(cfg, logger=log)


async def _run_health_server(cfg) -> None:
    """Run a lightweight uvicorn server for the health endpoint (option 1)."""
    global _health_server
    config = uvicorn.Config(
        health_app,
        host=cfg.health_host,
        port=cfg.health_port,
        log_level="warning",
        loop="asyncio",
    )
    server = uvicorn.Server(config)
    _health_server = server
    await server.serve()


async def _detection_loop(
    pipeline: DetectionPipeline,
    planner: PythonRoutePlanner,
) -> None:
    while True:
        # queue.Queue.get() блокирующий
        batch = await asyncio.to_thread(deps.dets_queue.get)

        telemetry = planner.latest_telemetry
        if telemetry is None:
            continue

        now = utc_now()

        raw_detections = [
            RawDetectionPayload(
                class_id=det.class_id,
                confidence=det.confidence,
                bbox=det.bbox,
                frame_id=batch.frame.camera_id,
                timestamp=now,
                track_id=None,
            )
            for det in batch.detections
        ]

        payload = DetectionBatchPayload(
            frame_id=batch.frame.camera_id,
            frame_width=batch.frame.width,
            frame_height=batch.frame.height,
            captured_at=now,
            telemetry=telemetry,
            detections=raw_detections,
        )

        try:
            confirmed = pipeline.process_batch(payload)

            if confirmed:
                log.info("Confirmed detections: %d", len(confirmed))

        except Exception:
            log.exception("Detection pipeline failed")

async def _watchdog_loop(cfg) -> None:
    """Periodic watchdog that checks telemetry and detections recency."""
    telemetry_flagged = False
    detection_flagged = False
    while True:
        await asyncio.sleep(cfg.watchdog_interval_sec)
        now = utc_now()
        last_tel = health_state.last_telemetry
        if last_tel is None or (now - last_tel).total_seconds() > cfg.no_telemetry_timeout_sec:
            if not telemetry_flagged:
                log.error(
                    "Watchdog: no telemetry for > %.1fs", cfg.no_telemetry_timeout_sec
                )
                health_state.mark_unhealthy("telemetry_stale")
                bus.emit(Event.MODULE_UNHEALTHY, {"reason": "telemetry"})
                telemetry_flagged = True
        else:
            telemetry_flagged = False
            health_state.clear_reason("telemetry_stale")

        if getattr(cfg, "watchdog_expect_detections", False):
            last_det = health_state.last_detection
            if last_det is None or (now - last_det).total_seconds() > cfg.no_detection_timeout_sec:
                if not detection_flagged:
                    log.warning(
                        "Watchdog: no detections for > %.1fs", cfg.no_detection_timeout_sec
                    )
                    detection_flagged = True
            else:
                detection_flagged = False
                health_state.clear_reason("detections_stale")


async def _run() -> None:
    """Configure shared services and start headless processing loop."""
    cfg = load_module_settings()
    setup_logging(cfg)
    loop = asyncio.get_running_loop()
    health_server: asyncio.Task | None = None
    watchdog: asyncio.Task | None = None

    init_module_core()  # queues + lifecycle; camera/detector threads if camera present

    energy_model = get_energy_model(cfg)
    planner = PythonRoutePlanner(energy_model=energy_model, settings=cfg)
    projector = get_geo_projector(cfg)
    visualizer = VisualizerAdapter(cfg)
    transmitter = _make_transmitter()
    aggregator = DetectionAggregator(
        window=cfg.agg_window,
        votes_required=cfg.agg_votes_required,
        min_confidence=cfg.agg_min_confidence,
        max_distance_m=cfg.agg_max_distance_m,
        ttl_seconds=cfg.agg_ttl_seconds,
    )
    pipeline = DetectionPipeline(
        aggregator=aggregator,
        projector=projector,
        transmitter=transmitter,
        visualizer_adapter=visualizer if getattr(cfg, "visualizer_enabled", False) else None,
        loop=loop,
        detection_callback=health_state.update_detection,
    )

    adapter = _build_adapter(cfg)
    telemetry_consumer = _ModuleTelemetryConsumer(
        pipeline=pipeline,
        planner=planner,
        energy_model=energy_model,
        visualizer=visualizer if getattr(cfg, "visualizer_enabled", False) else None,
        adapter=adapter,
        settings=cfg,
    )

    log.info(
        "Module runtime initialised | backend=%s planner=%s energy=%s projector=%s pipeline=%s",
        getattr(cfg, "uav_backend", "unknown"),
        planner.__class__.__name__,
        energy_model.__class__.__name__,
        projector.__class__.__name__,
        pipeline.__class__.__name__,
    )

    
    # Start capture/detect threads if available.
    bus.emit(Event.APP_START)
    log.info("Started core lifecycle threads")
    health_state.mark_start()
    configure_health(
        telemetry_timeout_sec=cfg.no_telemetry_timeout_sec,
        detection_timeout_sec=cfg.no_detection_timeout_sec,
        expect_detections=getattr(cfg, "watchdog_expect_detections", False),
    )

    health_server = asyncio.create_task(_run_health_server(cfg))
    watchdog = asyncio.create_task(_watchdog_loop(cfg))

    detection_task = asyncio.create_task(
        _detection_loop(
            pipeline=pipeline,
            planner=planner,
        )
    )

    await adapter.start(telemetry_consumer)
    
    log.info("UAV adapter started (%s)", adapter.__class__.__name__)

    try:
        while True:
            await asyncio.sleep(1.0)
    except (KeyboardInterrupt, asyncio.CancelledError):
        log.info("Stopping module runtime...")
    finally:
        bus.emit(Event.APP_STOP)
        if watchdog:
            watchdog.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watchdog

        if detection_task:
            detection_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await detection_task
        try:
            await adapter.stop()
        except Exception:  # noqa: BLE001
            log.exception("Failed to stop UAV adapter cleanly")
        if transmitter:
            try:
                transmitter.close()
            except Exception:  # noqa: BLE001
                log.exception("Failed to close transmitter socket")
        if getattr(cfg, "visualizer_enabled", False) and visualizer:
            try:
                await visualizer.aclose()
            except Exception:  # noqa: BLE001
                log.exception("Failed to close visualizer adapter")
        if _health_server:
            _health_server.should_exit = True
        if health_server:
            with contextlib.suppress(asyncio.CancelledError):
                await health_server


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":  # pragma: no cover
    main()
