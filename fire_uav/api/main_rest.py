# mypy: ignore-errors
"""
FastAPI-сервис: REST-доступ к запуску потоков, плану облёта и детекциям.
"""

from __future__ import annotations

import logging
from typing import Any, List

from fastapi import Depends, FastAPI, HTTPException, Response, status
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse
from prometheus_client import REGISTRY, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

import fire_uav.infrastructure.providers as deps
from fire_uav.bootstrap import init_core
from fire_uav.config import settings
from fire_uav.api.security import require_api_key
from fire_uav.module_core.schema import GeoDetection
from fire_uav.services.bus import Event, bus
from fire_uav.services.detections import DetectionBatchPayload, DetectionPipeline
from fire_uav.services.telemetry.transmitter import Transmitter

# Инициализируем ядро (очереди, LifecycleManager, шина)
init_core()

log = logging.getLogger("api")
app = FastAPI(
    title="fire-uav API",
    version="0.1.0",
    default_response_class=ORJSONResponse,
    dependencies=[Depends(require_api_key)],
)
app.add_middleware(GZipMiddleware, minimum_size=512)

# Expose HTTP metrics separately to avoid clashing with the legacy /metrics endpoint.
Instrumentator().instrument(app).expose(app, endpoint="/metrics/http")

_transmitter: Transmitter | None = None
if settings.ground_station_enabled:
    try:
        _transmitter = Transmitter(
            host=settings.ground_station_host,
            port=settings.ground_station_port,
            udp=settings.ground_station_udp,
        )
    except Exception:  # noqa: BLE001
        log.exception("Failed to connect transmitter to ground station")
        _transmitter = None

detection_pipeline = DetectionPipeline()


class Waypoint(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    alt_m: float = Field(..., gt=0, description="Altitude in metres")


class Plan(BaseModel):
    waypoints: List[Waypoint]


@app.on_event("startup")
async def on_startup() -> None:
    """Повторная инициализация ядра при hot-reload."""
    init_core()
    log.info(
        "Core initialised: lifecycle=%s",
        bool(deps.lifecycle_manager),
    )


@app.on_event("shutdown")
async def on_shutdown() -> None:
    """Чисто закрываем соединение с наземной станцией."""
    try:
        if _transmitter:
            _transmitter.close()
    except Exception:  # noqa: BLE001
        log.exception("Failed to close transmitter")


def ensure_running() -> None:
    """Гарантирует, что core services запущены."""
    if deps.lifecycle_manager is None:
        init_core()
    if deps.lifecycle_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Core not initialised",
        )


@app.post("/api/camera/start", status_code=status.HTTP_202_ACCEPTED)
def camera_start() -> dict[str, str]:
    """Start camera and detector threads."""
    ensure_running()
    bus.emit(Event.APP_START)
    log.info("APP_START via REST")
    return {"status": "starting"}


@app.post("/api/camera/stop", status_code=status.HTTP_202_ACCEPTED)
def camera_stop() -> dict[str, str]:
    """Stop all core components."""
    ensure_running()
    bus.emit(Event.APP_STOP)
    log.info("APP_STOP via REST")
    return {"status": "stopping"}


@app.get("/api/plan", response_model=Plan)
def get_plan() -> Plan:
    """Возвращает текущий план облёта."""
    if deps.plan_data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No plan uploaded",
        )
    return deps.plan_data


@app.post("/api/plan", status_code=status.HTTP_201_CREATED)
def set_plan(plan: Plan) -> dict[str, str]:
    """Загружает новый план облёта."""
    deps.plan_data = plan
    log.info("Plan uploaded: %d waypoints", len(plan.waypoints))
    return {"status": "ok"}


@app.get("/api/detect")
def get_last_detection() -> Any:
    """Возвращает последний результат детекции."""
    if deps.last_detection is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No detections yet",
        )
    return deps.last_detection


@app.post("/api/detections", response_model=List[GeoDetection])
def process_detections(batch: DetectionBatchPayload) -> List[GeoDetection]:
    """
    Принять сырые детекции модели и телеметрию, выполнить голосование K из N и
    вернуть подтверждённые объекты с координатами.
    """
    ensure_running()
    result = detection_pipeline.process_batch(batch)
    log.info("Detections ingested: raw=%d confirmed=%d", len(batch.detections), len(result))
    return result


@app.get("/metrics", summary="Prometheus metrics")
def metrics() -> Response:
    """Экспонирует метрики Prometheus."""
    data = generate_latest(REGISTRY)
    return Response(
        content=data,
        media_type="text/plain; version=0.0.4",
    )
