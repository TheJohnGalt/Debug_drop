# mypy: ignore-errors
from __future__ import annotations

import json
import logging
import math
import queue
import tempfile
import time
import uuid
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
# костыль, так как jetson не работает с python 3.11
try:
    from enum import StrEnum
except ImportError:
    from enum import Enum
    class StrEnum(str, Enum):
        pass
from typing import Callable, Final

import cv2
import httpx
import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import Property, QObject, QUrl, Signal, Slot, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuick import QQuickImageProvider

import fire_uav.infrastructure.providers as deps
from fire_uav.config import settings
from fire_uav.core.protocol import MapBounds
from fire_uav.gui.map_providers import (
    FoliumMapProvider,
    MapProvider,
    OpenLayersMapProvider,
    StaticImageMapProvider,
)
from fire_uav.gui.viewmodels.detector_vm import DetectorVM
from fire_uav.gui.viewmodels.planner_vm import PlannerVM
from fire_uav.module_core.factories import build_camera_params, get_energy_model
from fire_uav.module_core.geometry import haversine_m
from fire_uav.module_core.route.maneuvers import build_rejoin
from fire_uav.module_core.route.base_location import resolve_base_location
from fire_uav.module_core.drivers.registry import resolve_driver_type
from fire_uav.module_core.detections.pipeline import (
    DetectionBatchPayload,
    DetectionPipeline,
    RawDetectionPayload,
)
from fire_uav.module_core.detections.aggregator import DetectionAggregator
from fire_uav.module_core.schema import Route, TelemetrySample, Waypoint, WorldCoord
from fire_uav.domain.video.camera import CameraParams
from fire_uav.gui.services.unreal_link_service import UnrealLinkService
from fire_uav.services.components.camera import CameraThread
from fire_uav.services.bus import Event, bus
from fire_uav.services.detection_exports import (
    make_confirmed_detection_log_entry,
    make_raw_detection_log_entry,
)
from fire_uav.services.debug_sim import DebugSimulationService
from fire_uav.services.flight_recorder import FlightRecorder, FlightSummary
from fire_uav.services.mission import CameraMonitor, CameraStatus, LinkMonitor, LinkStatus, MissionState, MissionStateMachine
from fire_uav.services.mission.action_policy import MissionActionPolicy
from fire_uav.services.mission.active_path import ActivePathController, ActivePathMode
from fire_uav.services.mission.route_edit import dedupe_path, split_route_for_edit
from fire_uav.services.notifications import ToastDeduplicator
from fire_uav.services.objects_store import ConfirmedObject, ConfirmedObjectsStore
from fire_uav.services.targets.target_tracker import TargetObservation, TargetTracker
from fire_uav.services.ui_throttle import TelemetryStore
from fire_uav.utils.time import utc_iso_z, utc_now

_log: Final = logging.getLogger(__name__)


class OrbitFlowState(StrEnum):
    NORMAL_FLIGHT = "normal_flight"
    TARGET_DETECTED_REACTION_WINDOW = "target_detected_reaction_window"
    ORBIT_ACTIVE = "orbit_active"
    ROUTE_RESUME = "route_resume"


@dataclass(slots=True)
class RecoverableMissionSnapshot:
    path: list[tuple[float, float]]
    home: dict[str, float] | None
    confirmed_objects: list[dict[str, object]]
    selected_object_id: str | None


# --------------------------------------------------------------------------- #
#                            Video helpers
# --------------------------------------------------------------------------- #
class VideoFrameProvider(QQuickImageProvider):
    """Simple QML image provider that keeps last rendered frame."""

    def __init__(self) -> None:
        super().__init__(QQuickImageProvider.Image)
        self._image = QImage()

    def requestImage(self, _id, size, requestedSize):  # type: ignore[override]
        # PySide6 expects ONLY QImage as return value.
        img = self._image

        if img is None or img.isNull():
            # placeholder
            w = requestedSize.width() if requestedSize and requestedSize.width() > 0 else 1280
            h = requestedSize.height() if requestedSize and requestedSize.height() > 0 else 720
            placeholder = QImage(w, h, QImage.Format.Format_ARGB32)
            placeholder.fill(QColor("#0a111b"))
            if size is not None:
                size.setWidth(placeholder.width())
                size.setHeight(placeholder.height())
            return placeholder

        if size is not None:
            size.setWidth(img.width())
            size.setHeight(img.height())
        return img


    def set_image(self, image: QImage) -> None:
        self._image = image


class VideoBridge(QObject):
    """Stores last frame + bboxes and notifies QML to refresh the Image."""

    frameReady = Signal(str)

    def __init__(self, provider: VideoFrameProvider) -> None:
        super().__init__()
        self._provider = provider
        self._last_frame: NDArray[np.uint8] | None = None
        self._last_qimage: QImage | None = None
        self._bboxes: list[tuple[float, float, float, float]] = []
        self._counter = 0

    @Slot(object)
    def set_bboxes(self, boxes: list[tuple[int, int, int, int]] | None) -> None:
        parsed: list[tuple[float, float, float, float]] = []
        for box in boxes or []:
            if not isinstance(box, (tuple, list)) or len(box) != 4:
                continue
            try:
                parsed.append((float(box[0]), float(box[1]), float(box[2]), float(box[3])))
            except Exception:
                continue
        self._bboxes = parsed
        self._render()

    def update_frame(self, frame: NDArray[np.uint8]) -> None:
        self._last_frame = frame
        self._last_qimage = None
        self._render()

    def update_qimage(self, image: QImage) -> None:
        if image.isNull():
            return
        self._last_qimage = image.copy()
        self._last_frame = None
        self._render()

    def clear_overlays(self) -> None:
        self._bboxes = []
        self._render()

    # ---------- internal ---------- #
    def _render(self) -> None:
        if self._last_frame is not None:
            h, w, _ = self._last_frame.shape
            base = QImage(self._last_frame.data, w, h, 3 * w, QImage.Format.Format_BGR888).copy()
        elif self._last_qimage is not None:
            base = self._last_qimage.copy()
            w = base.width()
            h = base.height()
        else:
            return

        if self._bboxes:
            painter = QPainter(base)
            pen = QPen(QColor("#70e0ff"), 2)
            painter.setPen(pen)
            for x1, y1, x2, y2 in self._bboxes:
                normalized = all(abs(v) <= 1.5 for v in (x1, y1, x2, y2))
                scale_x = float(w) if normalized else 1.0
                scale_y = float(h) if normalized else 1.0
                px1 = x1 * scale_x
                py1 = y1 * scale_y
                px2 = x2 * scale_x
                py2 = y2 * scale_y
                # Fallback for xywh-formatted detections.
                if px2 <= px1 or py2 <= py1:
                    px2 = px1 + (x2 * scale_x)
                    py2 = py1 + (y2 * scale_y)
                cx1 = max(0, min(int(round(px1)), w - 1))
                cy1 = max(0, min(int(round(py1)), h - 1))
                cx2 = max(0, min(int(round(px2)), w - 1))
                cy2 = max(0, min(int(round(py2)), h - 1))
                if cx2 <= cx1 or cy2 <= cy1:
                    continue
                painter.drawRect(cx1, cy1, cx2 - cx1, cy2 - cy1)
            painter.end()

        self._provider.set_image(base)
        self._counter += 1
        self.frameReady.emit(f"image://video/live?{self._counter}")


# --------------------------------------------------------------------------- #
#                             UAV state
# --------------------------------------------------------------------------- #
class UavState(QObject):
    """Persistent QObject per UAV to avoid rebuilding UI models."""

    positionChanged = Signal()
    headingChanged = Signal()
    batteryChanged = Signal()

    def __init__(self, uav_id: str) -> None:
        super().__init__()
        self._uav_id = uav_id
        self._lat = 0.0
        self._lon = 0.0
        self._heading = 0.0
        self._battery_percent = -1.0

    @Property(str, constant=True)
    def uavId(self) -> str:
        return self._uav_id

    @Property(float, notify=positionChanged)
    def lat(self) -> float:
        return self._lat

    @Property(float, notify=positionChanged)
    def lon(self) -> float:
        return self._lon

    @Property(float, notify=headingChanged)
    def heading(self) -> float:
        return self._heading

    @Property(float, notify=batteryChanged)
    def batteryPercent(self) -> float:
        return self._battery_percent

    def update_from_sample(self, sample: TelemetrySample) -> None:
        lat = float(sample.lat)
        lon = float(sample.lon)
        heading = float(getattr(sample, "yaw", 0.0) or 0.0)
        battery = getattr(sample, "battery_percent", None)
        battery_percent = float(battery) if battery is not None else -1.0
        if lat != self._lat or lon != self._lon:
            self._lat = lat
            self._lon = lon
            self.positionChanged.emit()
        if heading != self._heading:
            self._heading = heading
            self.headingChanged.emit()
        if battery_percent != self._battery_percent:
            self._battery_percent = battery_percent
            self.batteryChanged.emit()


# --------------------------------------------------------------------------- #
#                             Logging bridge
# --------------------------------------------------------------------------- #
class QmlLogHandler(logging.Handler, QObject):
    message = Signal(str)

    def __init__(self) -> None:
        logging.Handler.__init__(self)
        QObject.__init__(self)
        self.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        try:
            self.message.emit(self.format(record))
        except Exception:  # noqa: BLE001
            pass


class MapBridge(QObject):
    """Generates Folium map with draw controls and bridges JS to Python via console logs."""

    urlChanged = Signal(QUrl)
    toastRequested = Signal(str)
    planChanged = Signal(list)
    objectSelected = Signal(str, float, float)

    def __init__(self, vm: PlannerVM, provider: MapProvider | None = None) -> None:
        super().__init__()
        self._vm = vm
        self._path_provider: Callable[[], list[tuple[float, float]]] | None = None
        cache_dir = Path(__file__).resolve().parents[2] / "data" / "cache" / "tiles"
        self._default_cache_dir = cache_dir
        self._provider: MapProvider = provider or self._select_provider()
        self._map_path: Path = Path(tempfile.gettempdir()) / "plan_map.html"
        self._token = 0
        self._last_render_progress: float | None = None
        self._last_telemetry_pos: tuple[float, float] | None = None
        self._last_render_ts: float | None = None
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(33)
        self._render_timer.timeout.connect(self._render_map_now)
        self._render_map_now()

    # ----- properties exposed to QML ----- #
    @Property(QUrl, notify=urlChanged)
    def url(self) -> QUrl:
        url = QUrl.fromLocalFile(str(self._map_path))
        url.setQuery(f"v={self._token}")
        return url

    @Property(str, constant=True)
    def bridgeScript(self) -> str:
        return getattr(self._provider, "bridge_script", "")

    # ----- slots for QML ----- #
    @Slot()
    def render_map(self) -> None:
        if self._render_timer.isActive():
            return
        self._render_timer.start()

    def set_path_provider(self, provider: Callable[[], list[tuple[float, float]]] | None) -> None:
        self._path_provider = provider

    def _render_map_now(self) -> None:
        path = self._path_provider() if self._path_provider is not None else self._vm.get_active_path()
        self._map_path = self._provider.render_map(path, self._token)
        self._token += 1
        self._last_render_ts = time.monotonic()
        try:
            self._last_render_progress = float(getattr(deps, "debug_flight_progress", 0.0))
        except Exception:
            self._last_render_progress = 0.0
        try:
            tel = getattr(deps, "latest_telemetry", None)
            if tel is not None and hasattr(tel, "lat") and hasattr(tel, "lon"):
                self._last_telemetry_pos = (float(tel.lat), float(tel.lon))
        except Exception:
            self._last_telemetry_pos = None
        self.urlChanged.emit(self.url)

    @Slot(str)
    def handle_console(self, message: str) -> None:
        if message.startswith("PY_TOAST "):
            try:
                payload = json.loads(message.split(" ", 1)[1])
            except Exception:
                return
            msg = str(payload.get("message", "") or "")
            if msg:
                self.toastRequested.emit(msg)
            return
        if message.startswith("PY_PATH "):
            gj = json.loads(message.split(" ", 1)[1])
            pts = [(lat, lon) for lon, lat in gj.get("coordinates", [])]
            if pts:
                self._vm.save_plan(pts)
            else:
                self._vm.clear_plan()
            deps.debug_target = None
            deps.debug_orbit_path = None
            self._after_path_changed()
            self.toastRequested.emit("Path updated" if pts else "Path cleared")
            return

        if message.startswith("PY_TARGET "):
            gj = json.loads(message.split(" ", 1)[1])
            coords = gj.get("coordinates") or []
            if len(coords) == 2:
                try:
                    lon, lat = coords
                    self._vm.set_debug_target(lat, lon)
                    self._vm.compute_orbit_preview()
                    self.render_map()
                    self.toastRequested.emit("Debug target set, orbit preview updated")
                except Exception as exc:  # noqa: BLE001
                    self.toastRequested.emit(f"Orbit preview error: {exc}")
            else:
                self._vm.clear_debug_target()
                self.render_map()
                self.toastRequested.emit("Debug target cleared")
            return

        if message.startswith("PY_OBJECT "):
            try:
                payload = json.loads(message.split(" ", 1)[1])
            except Exception:
                return
            object_id = str(payload.get("object_id", ""))
            lat = payload.get("lat")
            lon = payload.get("lon")
            if object_id and lat is not None and lon is not None:
                self.objectSelected.emit(object_id, float(lat), float(lon))
            return

    @Slot()
    def generate_path(self) -> None:
        try:
            fn = self._vm.generate_path()
            rel = fn
            try:
                rel = fn.relative_to(fn.parents[1])
            except Exception:
                pass
            self.toastRequested.emit(f"Path saved -> {rel}")
        except Exception as exc:  # noqa: BLE001
            self.toastRequested.emit(str(exc))

    @Slot()
    def save_plan(self) -> None:
        try:
            fn = self._vm.export_qgc_plan(alt_m=120.0)
            self.toastRequested.emit(f"Mission saved -> {fn.relative_to(fn.parents[1])}")
        except Exception as exc:  # noqa: BLE001
            self.toastRequested.emit(str(exc))

    @Slot(str)
    def import_geojson(self, fn: str) -> None:
        if not fn:
            return
        try:
            self._vm.import_geojson(Path(fn))
            self._after_path_changed()
            self.toastRequested.emit("Polyline imported")
        except Exception as exc:  # noqa: BLE001
            self.toastRequested.emit(f"Error: {exc}")

    @Slot(str)
    def import_kml(self, fn: str) -> None:
        if not fn:
            return
        try:
            pts = self._parse_kml(Path(fn))
            self._vm.save_plan(pts)
            self._after_path_changed()
            self.toastRequested.emit("KML imported")
        except Exception as exc:  # noqa: BLE001
            self.toastRequested.emit(f"Error: {exc}")

    @Slot(str, bool, str)
    def set_provider(self, provider: str, offline: bool, cache_dir: str = "") -> None:
        provider = (provider or "osm").lower()
        cd = Path(cache_dir) if cache_dir else self._default_cache_dir
        if provider == "static_image":
            static_provider = self._build_static_provider()
            if static_provider:
                self._provider = static_provider
                self.render_map()
            else:
                _log.warning("Static image provider unavailable; keeping current map provider")
            return
        if provider.startswith("openlayers") or provider.startswith("ol_"):
            if isinstance(self._provider, OpenLayersMapProvider):
                self._provider.set_provider(provider, offline=offline, cache_dir=cd)
            else:
                self._provider = OpenLayersMapProvider(provider=provider)
            self.render_map()
            return
        if isinstance(self._provider, StaticImageMapProvider):
            self._provider = FoliumMapProvider(provider=provider, offline=offline, cache_dir=cd)
            self.render_map()
            return
        if isinstance(self._provider, OpenLayersMapProvider):
            self._provider = FoliumMapProvider(provider=provider, offline=offline, cache_dir=cd)
            self.render_map()
            return
        self._provider.set_provider(provider, offline=offline, cache_dir=cd)
        self.render_map()

    @Slot()
    def recomputeOrbitPreview(self) -> None:
        """Rebuild orbit preview around current debug target and refresh the map."""
        if deps.debug_target is None:
            self.toastRequested.emit("Set a debug target marker first")
            return
        if not self._vm.get_path():
            self.toastRequested.emit("Draw a main route first")
            return
        try:
            self._vm.compute_orbit_preview()
            self.render_map()
            if deps.debug_orbit_path:
                self.toastRequested.emit("Orbit preview recomputed")
            else:
                self.toastRequested.emit("Orbit preview unavailable (energy/route?)")
        except Exception as exc:  # noqa: BLE001
            self.toastRequested.emit(f"Orbit preview error: {exc}")

    @Slot()
    def clearDebugTarget(self) -> None:
        """Clear debug target marker and orbit overlay."""
        self._vm.clear_debug_target()
        self.render_map()
        self.toastRequested.emit("Debug target cleared")

    @Slot()
    def refresh_drone(self) -> None:
        """Rerender map if drone progress changed enough to matter."""
        if getattr(deps, "debug_map_manual_refresh", False):
            return
        if getattr(deps, "debug_flight_enabled", False):
            now = time.monotonic()
            if self._last_render_ts is not None and (now - self._last_render_ts) < 2.5:
                return
        tel = getattr(deps, "latest_telemetry", None)
        if tel is not None and hasattr(tel, "lat") and hasattr(tel, "lon"):
            try:
                pos = (float(tel.lat), float(tel.lon))
            except Exception:
                pos = None
            if pos is not None:
                if self._last_telemetry_pos is None or haversine_m(pos, self._last_telemetry_pos) >= 5.0:
                    self.render_map()
                return
        try:
            prog = float(getattr(deps, "debug_flight_progress", 0.0))
        except Exception:
            prog = 0.0
        if self._last_render_progress is None or abs(prog - self._last_render_progress) >= 0.02:
            self.render_map()

    # ----- helpers ----- #
    def _parse_kml(self, fn: Path) -> list[tuple[float, float]]:
        text = fn.read_text(encoding="utf-8", errors="ignore")
        if "<coordinates" not in text:
            raise RuntimeError("Coordinates not found in KML")
        coords_block = (
            text.split("<coordinates", 1)[1].split(">", 1)[1].split("</coordinates>", 1)[0]
        )
        pts: list[tuple[float, float]] = []
        for raw in coords_block.strip().replace("\n", " ").split():
            parts = raw.split(",")
            if len(parts) < 2:
                continue
            lon, lat = float(parts[0]), float(parts[1])
            pts.append((lat, lon))
        if not pts:
            raise RuntimeError("No valid points in KML")
        return pts

    def _after_path_changed(self) -> None:
        """Reset sim position and rebuild orbit preview if needed after path edits."""
        deps.debug_flight_progress = 0.0
        if deps.debug_target:
            try:
                self._vm.compute_orbit_preview()
            except Exception:
                pass
        self.render_map()
        self.planChanged.emit(self._vm.get_path())

    def _select_provider(self) -> MapProvider:
        provider = str(getattr(settings, "map_provider", "osm") or "osm").lower()
        if provider == "static_image":
            static_provider = self._build_static_provider()
            if static_provider:
                return static_provider
            _log.warning("Static image provider requested but unavailable; falling back to OSM")
        if provider.startswith("openlayers") or provider.startswith("ol_"):
            return OpenLayersMapProvider(provider=provider)
        return FoliumMapProvider(cache_dir=self._default_cache_dir)

    def _fetch_snapshot_info(self) -> tuple[str, MapBounds] | None:
        base_url = str(getattr(settings, "visualizer_url", "http://127.0.0.1:8000")).rstrip("/")
        uav_id = getattr(settings, "uav_id", None) or "uav"
        url = f"{base_url}/api/v1/map_snapshot/{uav_id}"
        try:
            resp = httpx.get(url, timeout=1.0)
            if resp.status_code == 404:
                _log.warning("Map snapshot not found for UAV %s at %s", uav_id, url)
                return None
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            _log.warning("Failed to fetch map snapshot from %s: %s", url, exc)
            return None
        image_path = payload.get("image_path")
        bounds_raw = payload.get("bounds")
        if not image_path or not bounds_raw:
            _log.warning("Map snapshot payload missing image_path or bounds")
            return None
        try:
            bounds = bounds_raw if isinstance(bounds_raw, MapBounds) else MapBounds(**bounds_raw)
        except Exception:  # noqa: BLE001
            _log.warning("Map snapshot bounds invalid")
            return None
        return str(image_path), bounds

    def _build_static_provider(self) -> MapProvider | None:
        image_path = getattr(settings, "static_map_image_path", None)
        bounds = getattr(settings, "static_map_bounds", None)
        if not image_path or not bounds:
            fetched = self._fetch_snapshot_info()
            if fetched:
                image_path, bounds = fetched
            else:
                _log.warning("Static map config missing image_path or bounds")
                return None
        if isinstance(bounds, dict):
            try:
                bounds = MapBounds(**bounds)
            except Exception:
                _log.warning("Static map bounds invalid")
                return None
        if not isinstance(bounds, MapBounds):
            _log.warning("Static map bounds invalid")
            return None
        if bounds.lat_min >= bounds.lat_max or bounds.lon_min >= bounds.lon_max:
            _log.warning("Static map bounds invalid: min must be < max")
            return None
        image_path = Path(str(image_path)).expanduser()
        if not image_path.exists():
            _log.warning("Static map image not found at %s", image_path)
            return None
        _log.info(
            "Static map ready: image=%s bounds=(%.6f, %.6f)-(%.6f, %.6f)",
            image_path,
            bounds.lat_min,
            bounds.lon_min,
            bounds.lat_max,
            bounds.lon_max,
        )
        return OpenLayersMapProvider(
            provider="openlayers_de",
            static_image_path=str(image_path),
            static_bounds=bounds,
        )


# --------------------------------------------------------------------------- #
#                             App controller (QML-facing)
# --------------------------------------------------------------------------- #
class AppController(QObject):
    toastRequested = Signal(str)
    objectNotificationReceived = Signal(str, int, float, str, object)
    frameReady = Signal(str)
    mapUrlChanged = Signal(QUrl)
    logsChanged = Signal()
    confidenceChanged = Signal()
    detectorRunningChanged = Signal()
    cameraAvailableChanged = Signal()
    cameraStatusDetailChanged = Signal()
    statsChanged = Signal()
    debugModeChanged = Signal()
    debugDetectorOrbitGuardChanged = Signal()
    simCameraEnabledChanged = Signal()
    bridgeModeChanged = Signal()
    bridgeBatteryProfileChanged = Signal()
    unsafeStartChanged = Signal()
    routeBatteryChanged = Signal()
    routeBatteryAdvisoryChanged = Signal()
    confirmedObjectsChanged = Signal()
    missionStateChanged = Signal()
    planConfirmedChanged = Signal()
    linkStatusChanged = Signal()
    cameraStatusChanged = Signal()
    flightControlsChanged = Signal()
    flightSummaryChanged = Signal()
    uavStatesChanged = Signal()
    homePickModeChanged = Signal()
    objectSpawnModeChanged = Signal()
    mapRefreshNeededChanged = Signal()
    backendChanged = Signal()
    unrealVideoModeChanged = Signal()
    autoOrbitEnabledChanged = Signal()
    orbitRadiusMChanged = Signal()
    recoverableMissionChanged = Signal()
    orbitBatteryAdvisoryChanged = Signal()

    def __init__(
        self,
        det_vm: DetectorVM,
        map_bridge: MapBridge,
        video_bridge: VideoBridge,
        camera_available: bool,
        camera_switcher: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__()
        self.det_vm = det_vm
        self.map_bridge = map_bridge
        self.video_bridge = video_bridge
        self._camera_switcher = camera_switcher

        self._logs: list[str] = []
        self._confidence = getattr(settings, "yolo_conf", 0.25)
        self._detector_running = False
        self._camera_available = camera_available
        self._camera_status_detail = "Camera not found"
        self._fps = 0.0
        self._latency_ms = 0.0
        self._bus_alive = False
        self._detection_conf = 0.0
        self._last_frame_ts: float | None = None
        self._last_detection_ts: float | None = None
        self._last_detection_log_ts: float | None = None
        self._last_detection_sig: tuple[int, tuple[int, ...], tuple[int, int, int, int] | None] | None = None
        self._ground_station_enabled = bool(getattr(settings, "ground_station_enabled", False))
        self._det_json_path = Path(getattr(settings, "output_root", Path("data/outputs"))) / (
            "detections.jsonl"
        )
        self._confirmed_det_json_path = Path(
            getattr(settings, "output_root", Path("data/outputs"))
        ) / "confirmed_detections.jsonl"
        self._log_history_path = (
            Path(__file__).resolve().parents[3] / "data" / "artifacts" / "logs" / "fire_uav_debug.log"
        )
        self._plan_vm = map_bridge._vm  # internal access for debug helpers
        self._tile_cache_dir = Path(__file__).resolve().parents[2] / "data" / "cache" / "tiles"
        self._link_monitor = LinkMonitor()
        self._camera_monitor = CameraMonitor()
        self._mission = MissionStateMachine(
            link_monitor=self._link_monitor,
            camera_monitor=self._camera_monitor,
        )
        self._link_status = LinkStatus.DISCONNECTED
        self._camera_status = CameraStatus.NOT_READY
        self._mission_state = self._mission.current_state
        deps.mission_state = self._mission_state.value
        self._allow_unsafe_start = False
        self._pending_unreal_autostart = False
        self._mission.set_allow_unsafe_start(self._allow_unsafe_start)
        self._commands_enabled = True
        self._capabilities = {
            "supports_waypoints": True,
            "supports_rtl": True,
            "supports_orbit": True,
            "supports_set_speed": True,
            "supports_camera": True,
        }
        self._route_edit_mode = False
        self._hold_sent_for_edit = False
        self._staged_plan: list[tuple[float, float]] | None = None
        self._route_edit_anchor: tuple[float, float] | None = None
        self._route_edit_original_plan: list[tuple[float, float]] | None = None
        self._route_edit_locked_path: list[tuple[float, float]] | None = None
        self._latest_telemetry: TelemetrySample | None = None
        self._uav_states: dict[str, UavState] = {}
        self._telemetry_store = TelemetryStore()
        self._active_path = ActivePathController(
            confirmed_path_provider=self.get_confirmed_path,
            draft_path_provider=self._plan_vm.get_path,
            mission_state_provider=lambda: self._mission_state,
            plan_confirmed_provider=lambda: self._mission.plan_confirmed,
            on_change=self._on_active_path_changed,
        )
        self._energy_model = get_energy_model(settings)
        self._toast_dedupe = ToastDeduplicator()
        self._flight_recorder = FlightRecorder()
        self._objects_store = ConfirmedObjectsStore(on_change=self._on_objects_changed)
        self._target_tracker = TargetTracker(
            match_radius_m=float(getattr(settings, "match_radius_m", 30.0) or 30.0),
            suppression_radius_m=float(getattr(settings, "suppression_radius_m", 60.0) or 60.0),
            suppression_ttl_s=float(getattr(settings, "suppression_ttl_s", 180.0) or 180.0),
            stable_frames_n=int(getattr(settings, "stable_frames_n", 1) or 1),
        )
        self._known_confirmed_ids: set[str] = set()
        self._auto_orbit_enabled = bool(getattr(settings, "auto_orbit_enabled", False))
        self._orbit_radius_m = float(getattr(settings, "orbit_radius_m", 50.0) or 50.0)
        self._orbit_flow_state = OrbitFlowState.NORMAL_FLIGHT
        self._reaction_target_id: str | None = None
        self._reaction_started_monotonic = 0.0
        self._reaction_window_s = float(getattr(settings, "target_reaction_window_s", 4.0) or 4.0)
        self._reaction_slow_speed_mps = float(getattr(settings, "reaction_slow_speed_mps", 1.0) or 1.0)
        self._reaction_speed_override_active = False
        self._camera_info_ready = True
        self._pending_orbit_queue: list[tuple[str, str]] = []
        self._pending_orbit_ids: set[str] = set()
        self._last_orbit_availability: tuple[bool, bool] | None = None
        setattr(deps, "route_edit_anchor", None)
        setattr(deps, "route_edit_preview_path", None)
        setattr(deps, "route_edit_locked_path", None)
        setattr(deps, "debug_orbit_preview_paths", [])
        self._orbit_active = False
        self._orbit_rejoin_wp: Waypoint | None = None
        self._orbit_rejoin_close_hits = 0
        self._orbit_rejoin_threshold_m = float(getattr(settings, "orbit_rejoin_threshold_m", 15.0) or 15.0)
        self._orbit_target_track_ids: set[int] = set()
        self._orbit_target_centers: list[tuple[float, float]] = []
        self._orbit_route_end_wp: Waypoint | None = None
        self._orbit_resume_route: Route | None = None
        self._orbit_resume_min_index: int | None = None
        self._orbit_started_monotonic = 0.0
        self._orbit_min_complete_time_s = 0.0
        self._route_complete_announced = False
        self._debug_sim: DebugSimulationService | None = None
        self._rtl_forced = False
        self._rtl_route_sent = False
        self._flight_summary: FlightSummary | None = None
        self._bridge_mode_enabled = False
        self._bridge_profiles = self._load_bridge_profiles()
        self._bridge_profile_id = next(iter(self._bridge_profiles.keys()), "default")
        self._bridge_battery_wh = float(
            self._bridge_profiles.get(self._bridge_profile_id, {}).get("battery_wh", 4500.0)
        )
        self._bridge_speed_mps = float(
            self._bridge_profiles.get(self._bridge_profile_id, {}).get("speed_mps", 8.0)
        )
        self._backend = resolve_driver_type(settings)
        self._camera_info_ready = self._backend != "unreal"
        self._unreal_link: UnrealLinkService | None = None
        self._unreal_uav_id = str(getattr(settings, "uav_id", None) or "sim")
        self._unreal_detection_source = str(
            getattr(settings, "unreal_detection_source", "local_yolo") or "local_yolo"
        ).lower()
        if self._unreal_detection_source not in ("backend", "local_yolo", "both"):
            self._unreal_detection_source = "local_yolo"
        self._unreal_local_yolo_enabled = (
            self._backend == "unreal" and self._unreal_detection_source in ("local_yolo", "both")
        )
        self._unreal_local_detect_hz = max(
            0.1,
            float(getattr(settings, "unreal_local_detect_hz", 5.0) or 5.0),
        )
        self._unreal_local_detect_interval_s = 1.0 / self._unreal_local_detect_hz
        self._unreal_local_last_push_ts = 0.0
        self._unreal_local_feed_window_ts = 0.0
        self._unreal_local_feed_pushed = 0
        self._unreal_local_feed_dropped = 0
        self._unreal_local_feed_skipped = 0
        self._unreal_local_feed_log_window_s = 5.0
        self._unreal_local_pipeline: DetectionPipeline | None = None
        self._projector_camera_params = build_camera_params(settings)
        self._unreal_local_detector_autostart_done = False
        self._unreal_local_first_confirmed_logged = False
        self._unreal_local_telemetry_by_frame_id: dict[str, TelemetrySample] = {}
        self._unreal_local_frame_order: deque[str] = deque()
        self._debug_disable_detector_during_orbit = bool(
            getattr(settings, "debug_disable_detector_during_orbit", False)
        )
        self._detector_forced_paused_by_orbit = False
        self._recoverable_mission: RecoverableMissionSnapshot | None = None
        self._unreal_status = "disconnected"
        self._last_unreal_route_sent_monotonic = 0.0
        self._unreal_waiting_route_grace_s = float(
            getattr(settings, "unreal_waiting_route_grace_s", 2.5) or 2.5
        )
        self._unreal_video_mode = str(
            getattr(settings, "unreal_video_mode", "h264_stream") or "h264_stream"
        ).lower()
        self._video_visible = False
        self._route_battery_text = "Route: --"
        self._route_battery_remaining_text = "Remaining: --"
        self._route_battery_warning = False
        self._route_battery_advisory_visible = False
        self._route_battery_advisory_text = ""
        self._route_battery_rtl_available = False
        self._pending_route_battery_action = ""
        self._pending_route_battery_path: list[tuple[float, float]] = []
        self._orbit_battery_advisory_visible = False
        self._orbit_battery_advisory_text = ""
        self._orbit_battery_rtl_available = False
        self._pending_orbit_battery_targets: list[ConfirmedObject] = []
        self._pending_orbit_battery_source = "manual"
        self._pending_orbit_advisory_route: Route | None = None
        self._pending_orbit_advisory_base_route: Route | None = None
        self._pending_orbit_advisory_target: ConfirmedObject | None = None
        self._home_persist_path = Path(__file__).resolve().parents[3] / "data" / "artifacts" / "home.json"
        self._home_pick_mode = False
        self._object_spawn_mode = False
        self._map_refresh_needed = False
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(500)
        self._status_timer.timeout.connect(self._on_status_tick)
        self._status_timer.start()

        # wire signals
        self.det_vm.detection.connect(self._on_detections)
        self.det_vm.bboxes.connect(self.video_bridge.set_bboxes)
        self.video_bridge.frameReady.connect(self.frameReady)
        self.map_bridge.set_path_provider(self.get_active_path_for_sim)
        self.map_bridge.urlChanged.connect(self.mapUrlChanged)
        self.map_bridge.toastRequested.connect(self.toastRequested)
        self.map_bridge.planChanged.connect(self._on_plan_changed)
        self.map_bridge.objectSelected.connect(self._on_object_selected)

        # logging to QML
        self._log_handler = QmlLogHandler()
        self._log_handler.setLevel(logging.DEBUG)
        self._log_handler.message.connect(self._append_log)
        root = logging.getLogger()
        root.addHandler(self._log_handler)
        self._load_log_history()
        bus.subscribe(Event.OBJECT_CONFIRMED_UI, self._on_object_confirmed)
        bus.subscribe(Event.MISSION_STATE_CHANGED, self._on_mission_state)
        bus.subscribe(Event.UAV_LINK_STATUS_CHANGED, self._on_link_status)
        bus.subscribe(Event.CAMERA_STATUS_CHANGED, self._on_camera_status)
        bus.subscribe(Event.CAPABILITIES_UPDATED, self._on_capabilities_updated)
        bus.subscribe(Event.WARNING_TOAST, self._on_warning_toast)
        bus.subscribe(Event.FLIGHT_SESSION_ENDED, self._on_flight_session_ended)
        if self._unreal_local_yolo_enabled:
            votes_required = max(
                1,
                int(getattr(settings, "unreal_local_votes_required", 1) or 1),
            )
            window = max(
                1,
                int(getattr(settings, "unreal_local_agg_window", 3) or 3),
            )
            aggregator = DetectionAggregator(
                window=window,
                votes_required=votes_required,
                min_confidence=float(getattr(settings, "agg_min_confidence", 0.4) or 0.4),
                max_distance_m=float(getattr(settings, "agg_max_distance_m", 25.0) or 25.0),
                ttl_seconds=float(getattr(settings, "agg_ttl_seconds", 3.0) or 3.0),
            )
            self._unreal_local_pipeline = DetectionPipeline(
                aggregator=aggregator,
                camera_params=self._projector_camera_params,
                target_tracker=self._target_tracker,
            )
            bus.subscribe(Event.DETECTION, self._on_local_detection_batch)
            _log.info(
                "Unreal local YOLO pipeline enabled (source=%s, feed_hz=%.1f, votes=%d, window=%d)",
                self._unreal_detection_source,
                self._unreal_local_detect_hz,
                votes_required,
                window,
            )
        setattr(deps, "route_edit_anchor", None)
        setattr(deps, "route_edit_preview_path", None)
        setattr(deps, "route_edit_locked_path", None)
        self._load_persisted_home_location()
        self._update_route_estimate(self._active_path.get_active_path())
        self.map_bridge.render_map()

    # ---------- properties ---------- #
    @Property("QStringList", notify=logsChanged)
    def logs(self) -> list[str]:
        return self._logs

    @Property(float, notify=confidenceChanged)
    def confidence(self) -> float:
        return self._confidence

    @Property(bool, notify=detectorRunningChanged)
    def detectorRunning(self) -> bool:
        return self._detector_running

    @Property(bool, notify=cameraAvailableChanged)
    def cameraAvailable(self) -> bool:
        return self._camera_available

    @Property(str, notify=cameraStatusDetailChanged)
    def cameraStatusDetail(self) -> str:
        return self._camera_status_detail

    @Property(str, notify=missionStateChanged)
    def missionState(self) -> str:
        return self._mission_state.value

    @Property(bool, notify=planConfirmedChanged)
    def planConfirmed(self) -> bool:
        return self._mission.plan_confirmed

    @Property(str, notify=linkStatusChanged)
    def linkStatus(self) -> str:
        return self._link_status.value

    @Property(str, notify=cameraStatusChanged)
    def cameraStatus(self) -> str:
        return self._camera_status.value

    @Property(bool, notify=flightControlsChanged)
    def linkOk(self) -> bool:
        return self._link_monitor.is_link_ok()

    @Property(bool, notify=flightControlsChanged)
    def cameraOk(self) -> bool:
        return self._camera_monitor.is_camera_ok()

    @Property(bool, notify=flightControlsChanged)
    def startFlightEnabled(self) -> bool:
        return self._current_action_policy().can_start_flight

    @Property(bool, notify=flightControlsChanged)
    def canConfirmPlan(self) -> bool:
        return self._current_action_policy().can_confirm_plan

    @Property(bool, notify=flightControlsChanged)
    def canStartFlight(self) -> bool:
        return self._current_action_policy().can_start_flight

    @Property(bool, notify=flightControlsChanged)
    def canEditRoute(self) -> bool:
        return self._current_action_policy().can_edit_route

    @Property(bool, notify=flightControlsChanged)
    def canApplyRouteEdits(self) -> bool:
        return self._current_action_policy().can_apply_route_edits

    @Property(bool, notify=flightControlsChanged)
    def canCancelRouteEdits(self) -> bool:
        return self._route_edit_mode

    @Property(bool, notify=flightControlsChanged)
    def canOpenOrbit(self) -> bool:
        return self._current_action_policy().can_open_orbit

    @Property(bool, notify=flightControlsChanged)
    def canOrbit(self) -> bool:
        return self._current_action_policy().can_orbit

    @Property(bool, notify=flightControlsChanged)
    def canRtl(self) -> bool:
        return self._current_action_policy().can_rtl

    @Property(bool, notify=flightControlsChanged)
    def canSendRtlRoute(self) -> bool:
        return self._current_action_policy().can_send_rtl_route

    @Property(bool, notify=flightControlsChanged)
    def canCompleteLanding(self) -> bool:
        return self._current_action_policy().can_complete_landing

    @Property(bool, notify=flightControlsChanged)
    def canAbortToPreflight(self) -> bool:
        return self._current_action_policy().can_abort_to_preflight

    @Property(bool, notify=flightControlsChanged)
    def flightCommandsEnabled(self) -> bool:
        return self._commands_enabled

    @Property(bool, notify=flightControlsChanged)
    def routeEditMode(self) -> bool:
        return self._route_edit_mode

    @Property(bool, notify=autoOrbitEnabledChanged)
    def autoOrbitEnabled(self) -> bool:
        return self._auto_orbit_enabled

    @Property(float, notify=orbitRadiusMChanged)
    def orbitRadiusM(self) -> float:
        return self._orbit_radius_m

    @Property(int, notify=flightControlsChanged)
    def confirmedObjectCount(self) -> int:
        return self._objects_store.count()

    @Property(bool, notify=flightControlsChanged)
    def orbitAvailable(self) -> bool:
        return self._current_action_policy().can_open_orbit

    @Property("QVariantList", notify=confirmedObjectsChanged)
    def confirmedObjects(self) -> list[dict[str, object]]:
        return [
            {
                "object_id": obj.object_id,
                "class_id": obj.class_id,
                "confidence": obj.confidence,
                "lat": obj.lat,
                "lon": obj.lon,
                "track_id": obj.track_id,
                "display_index": index + 1,
                "selected": bool(obj.object_id == getattr(deps, "selected_object_id", None)),
            }
            for index, obj in enumerate(self._objects_store.all())
        ]

    @Property("QVariantList", notify=uavStatesChanged)
    def uavStates(self) -> list[UavState]:
        return list(self._uav_states.values())

    @Property(str, notify=flightSummaryChanged)
    def flightSummaryDuration(self) -> str:
        if not self._flight_summary or self._flight_summary.duration_s <= 0:
            return "0s"
        return self._format_duration(self._flight_summary.duration_s)

    @Property(str, notify=flightSummaryChanged)
    def flightSummaryDistance(self) -> str:
        if not self._flight_summary:
            return "0 m"
        return f"{self._flight_summary.distance_m:.0f} m"

    @Property(str, notify=flightSummaryChanged)
    def flightSummaryMinBattery(self) -> str:
        if not self._flight_summary or self._flight_summary.min_battery_percent is None:
            return "n/a"
        return f"{self._flight_summary.min_battery_percent:.1f}%"

    @Property(int, notify=flightSummaryChanged)
    def flightSummaryObjects(self) -> int:
        if not self._flight_summary:
            return 0
        return self._flight_summary.confirmed_objects

    @Property(QUrl, notify=mapUrlChanged)
    def mapUrl(self) -> QUrl:
        return self.map_bridge.url

    @Property(str, constant=True)
    def mapBridgeScript(self) -> str:
        return self.map_bridge.bridgeScript

    @Property(bool, notify=homePickModeChanged)
    def homePickModeEnabled(self) -> bool:
        return self._home_pick_mode

    @Property(bool, notify=objectSpawnModeChanged)
    def objectSpawnModeEnabled(self) -> bool:
        return self._object_spawn_mode

    @Property(bool, notify=mapRefreshNeededChanged)
    def mapRefreshNeeded(self) -> bool:
        return self._map_refresh_needed

    @Property(str, notify=backendChanged)
    def currentBackend(self) -> str:
        return self._backend

    @Property(str, notify=unrealVideoModeChanged)
    def unrealVideoMode(self) -> str:
        return self._unreal_video_mode

    @Property(float, notify=statsChanged)
    def fps(self) -> float:
        return self._fps

    @Property(float, notify=statsChanged)
    def latencyMs(self) -> float:
        return self._latency_ms

    @Property(float, notify=statsChanged)
    def detectionConfidence(self) -> float:
        return self._detection_conf

    @Property(str, notify=statsChanged)
    def currentBatteryText(self) -> str:
        if self._latest_telemetry and self._latest_telemetry.battery_percent is not None:
            return f"Battery: {float(self._latest_telemetry.battery_percent):.1f}%"
        return "Battery: --"

    @Property(bool, notify=statsChanged)
    def busAlive(self) -> bool:
        return self._bus_alive

    @Property(bool, notify=statsChanged)
    def groundStationEnabled(self) -> bool:
        return self._ground_station_enabled

    @Property(float, notify=statsChanged)
    def debugFlightProgress(self) -> float:
        return float(getattr(deps, "debug_flight_progress", 0.0))

    @Property(bool, notify=debugModeChanged)
    def debugMode(self) -> bool:
        if self._debug_sim is None:
            return False
        return self._debug_sim.telemetry_enabled()

    @Property(bool, notify=debugDetectorOrbitGuardChanged)
    def debugDisableDetectorDuringOrbit(self) -> bool:
        return self._debug_disable_detector_during_orbit

    @Property(bool, notify=bridgeModeChanged)
    def bridgeModeEnabled(self) -> bool:
        return self._bridge_mode_enabled

    @Property(str, notify=bridgeBatteryProfileChanged)
    def bridgeBatteryProfileId(self) -> str:
        return self._bridge_profile_id

    @Property(bool, notify=unsafeStartChanged)
    def allowUnsafeStart(self) -> bool:
        return self._allow_unsafe_start

    @Property(str, notify=routeBatteryChanged)
    def routeBatteryText(self) -> str:
        return self._route_battery_text

    @Property(str, notify=routeBatteryChanged)
    def routeBatteryRemainingText(self) -> str:
        return self._route_battery_remaining_text

    @Property(bool, notify=routeBatteryChanged)
    def routeBatteryWarning(self) -> bool:
        return self._route_battery_warning

    @Property(bool, notify=routeBatteryAdvisoryChanged)
    def routeBatteryAdvisoryVisible(self) -> bool:
        return self._route_battery_advisory_visible

    @Property(str, notify=routeBatteryAdvisoryChanged)
    def routeBatteryAdvisoryText(self) -> str:
        return self._route_battery_advisory_text

    @Property(bool, notify=routeBatteryAdvisoryChanged)
    def routeBatteryReturnHomeAvailable(self) -> bool:
        return self._route_battery_rtl_available

    @Property(bool, notify=orbitBatteryAdvisoryChanged)
    def orbitBatteryAdvisoryVisible(self) -> bool:
        return self._orbit_battery_advisory_visible

    @Property(str, notify=orbitBatteryAdvisoryChanged)
    def orbitBatteryAdvisoryText(self) -> str:
        return self._orbit_battery_advisory_text

    @Property(bool, notify=orbitBatteryAdvisoryChanged)
    def orbitBatteryReturnHomeAvailable(self) -> bool:
        return self._orbit_battery_rtl_available

    @Property(bool, notify=simCameraEnabledChanged)
    def simCameraEnabled(self) -> bool:
        if self._debug_sim is None:
            return False
        return self._debug_sim.camera_enabled()

    @Property(bool, notify=recoverableMissionChanged)
    def recoverableMissionAvailable(self) -> bool:
        return self._recoverable_mission is not None

    @Property(str, notify=recoverableMissionChanged)
    def recoverableMissionText(self) -> str:
        if self._recoverable_mission is None:
            return ""
        points = len(self._recoverable_mission.path)
        targets = len(self._recoverable_mission.confirmed_objects)
        return f"Restore last Unreal mission: {points} route points, {targets} targets."

    @Property(str, notify=linkStatusChanged)
    def unrealRuntimeStatus(self) -> str:
        return self._unreal_status

    @Property(str, constant=True)
    def tileCacheDir(self) -> str:
        return str(self._tile_cache_dir)

    # ---------- slots ---------- #
    @Slot()
    def cycleCamera(self) -> None:
        if self._camera_switcher is None:
            return
        self._camera_switcher()

    @Slot()
    def startDetector(self) -> None:
        self.det_vm.start()
        self._detector_running = True
        self._detector_forced_paused_by_orbit = False
        self.detectorRunningChanged.emit()

    @Slot()
    def stopDetector(self) -> None:
        self.det_vm.stop()
        self._detector_running = False
        if hasattr(self.det_vm, "reset"):
            try:
                self.det_vm.reset()
            except Exception:
                _log.debug("Failed to reset detector view-model", exc_info=True)
        try:
            self.video_bridge.clear_overlays()
        except Exception:
            _log.debug("Failed to clear video overlays after detector stop", exc_info=True)
        self._detection_conf = 0.0
        self._last_detection_ts = None
        self._update_stats()
        self.detectorRunningChanged.emit()

    def attach_unreal_link(self, link: UnrealLinkService) -> None:
        self._unreal_link = link
        self._unreal_link.set_camera_enabled(self._video_visible)

    def on_unreal_camera_info(self, payload: dict[str, object]) -> None:
        self._camera_info_ready = True
        self._projector_camera_params = CameraParams(
            fov_deg=float(payload.get("fov_deg", self._projector_camera_params.fov_deg)),
            mount_pitch_deg=float(
                payload.get("mount_pitch_deg", self._projector_camera_params.mount_pitch_deg)
            ),
            mount_yaw_deg=float(payload.get("mount_yaw_deg", self._projector_camera_params.mount_yaw_deg)),
            mount_roll_deg=float(
                payload.get("mount_roll_deg", self._projector_camera_params.mount_roll_deg)
            ),
        )
        if self._unreal_local_pipeline is None:
            return
        projector = getattr(self._unreal_local_pipeline, "projector", None)
        if projector is None or not hasattr(projector, "set_camera_params"):
            return
        try:
            projector.set_camera_params(self._projector_camera_params)  # type: ignore[attr-defined]
            _log.info(
                "Applied Unreal camera_info to projector: fov=%.2f mount(ypr)=(%.2f, %.2f, %.2f)",
                self._projector_camera_params.fov_deg,
                self._projector_camera_params.mount_yaw_deg,
                self._projector_camera_params.mount_pitch_deg,
                self._projector_camera_params.mount_roll_deg,
            )
            _log.debug(
                "Camera orientation source=unreal_camera_info (mount_pitch_deg=%.2f); local projection now uses runtime camera info",
                self._projector_camera_params.mount_pitch_deg,
            )
        except Exception as exc:  # noqa: BLE001
            _log.debug("Failed to apply camera_info to local projector: %s", exc)

    def on_camera_image(self, image: QImage) -> None:
        self._camera_monitor.on_frame()
        self._last_frame_ts = time.perf_counter()
        if not self._camera_available:
            self.set_camera_available(True)
        if (
            self._unreal_local_yolo_enabled
            and not self._detector_running
            and not self._unreal_local_detector_autostart_done
        ):
            self.startDetector()
            self._unreal_local_detector_autostart_done = True
            _log.info("Auto-started local YOLO detector on first Unreal camera frame")
        self._set_camera_status_detail("Live")
        self.video_bridge.update_qimage(image)
        self._feed_unreal_local_detector(image)
        self._update_stats()

    def _feed_unreal_local_detector(self, image: QImage) -> None:
        if not self._unreal_local_yolo_enabled:
            return
        if not self._detector_running:
            return
        if self._orbit_active:
            return
        if deps.frame_queue is None:
            return
        now = time.perf_counter()
        if (now - self._unreal_local_last_push_ts) < self._unreal_local_detect_interval_s:
            self._unreal_local_feed_skipped += 1
            self._maybe_log_unreal_local_feed(now)
            return
        try:
            frame_bgr = self._qimage_to_bgr_frame(image)
            captured_at = utc_now()
            frame_id = f"unreal_local:{captured_at.isoformat(timespec='microseconds')}"
            if self._latest_telemetry is not None:
                self._remember_unreal_local_telemetry(
                    frame_id,
                    self._latest_telemetry.model_copy(deep=True),
                )
            deps.frame_queue.put_nowait(
                {
                    "frame": frame_bgr,
                    "camera_id": "unreal_local",
                    "timestamp": captured_at,
                }
            )
            self._unreal_local_last_push_ts = now
            self._unreal_local_feed_pushed += 1
        except queue.Full:
            self._unreal_local_feed_dropped += 1
        except Exception as exc:  # noqa: BLE001
            _log.debug("Failed to convert/feed Unreal frame to local detector: %s", exc)
        self._maybe_log_unreal_local_feed(now)

    def _maybe_log_unreal_local_feed(self, now: float | None = None) -> None:
        if not self._unreal_local_yolo_enabled:
            return
        ts = time.perf_counter() if now is None else now
        if self._unreal_local_feed_window_ts <= 0.0:
            self._unreal_local_feed_window_ts = ts
            return
        elapsed = ts - self._unreal_local_feed_window_ts
        if elapsed < self._unreal_local_feed_log_window_s:
            return
        hz = self._unreal_local_feed_pushed / max(elapsed, 1e-3)
        _log.info(
            "Unreal local YOLO feed: pushed=%d dropped_queue_full=%d skipped_rate_limit=%d hz=%.2f target_hz=%.2f",
            self._unreal_local_feed_pushed,
            self._unreal_local_feed_dropped,
            self._unreal_local_feed_skipped,
            hz,
            self._unreal_local_detect_hz,
        )
        self._unreal_local_feed_window_ts = ts
        self._unreal_local_feed_pushed = 0
        self._unreal_local_feed_dropped = 0
        self._unreal_local_feed_skipped = 0

    @staticmethod
    def _qimage_to_bgr_frame(image: QImage) -> NDArray[np.uint8]:
        rgb = image.convertToFormat(QImage.Format.Format_RGB888)
        if rgb.isNull():
            raise ValueError("QImage is null")
        width = rgb.width()
        height = rgb.height()
        bytes_per_line = rgb.bytesPerLine()
        bits = rgb.bits()
        raw = bytes(bits)
        needed = bytes_per_line * height
        if len(raw) < needed:
            raise ValueError("QImage buffer shorter than expected")
        arr = np.frombuffer(raw[:needed], dtype=np.uint8).reshape((height, bytes_per_line))
        rgb_arr = arr[:, : width * 3].reshape((height, width, 3))
        return cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

    def _on_local_detection_batch(self, batch: object) -> None:
        if not self._unreal_local_yolo_enabled:
            return
        if self._unreal_local_pipeline is None:
            return
        if self._backend == "unreal" and not self._camera_info_ready:
            _log.debug(
                "Skipping local detection geo-confirmation: camera_info not ready (projection source of truth unavailable)"
            )
            return
        frame_meta = getattr(batch, "frame", None)
        if frame_meta is None:
            return
        try:
            frame_width = int(getattr(frame_meta, "width"))
            frame_height = int(getattr(frame_meta, "height"))
        except Exception:  # noqa: BLE001
            return
        if frame_width <= 0 or frame_height <= 0:
            return
        captured_at = self._coerce_datetime(getattr(frame_meta, "timestamp", None)) or utc_now()
        camera_id = str(getattr(frame_meta, "camera_id", None) or "unreal_local")
        # Important for K/N aggregation: frame_id must be unique per frame, not constant camera_id.
        frame_id = f"{camera_id}:{captured_at.isoformat(timespec='microseconds')}"
        telemetry = self._pop_unreal_local_telemetry(frame_id)
        if telemetry is None:
            telemetry = self._latest_telemetry
        if telemetry is None:
            return
        raw_detections: list[RawDetectionPayload] = []
        for det in getattr(batch, "detections", []) or []:
            bbox = self._extract_bbox_tuple(det)
            if bbox is None:
                continue
            det_ts = self._coerce_datetime(getattr(det, "timestamp", None)) or captured_at
            try:
                raw_detections.append(
                    RawDetectionPayload(
                        class_id=int(getattr(det, "class_id", getattr(det, "cls", 0)) or 0),
                        confidence=float(getattr(det, "confidence", getattr(det, "score", 0.0)) or 0.0),
                        bbox=bbox,
                        frame_id=frame_id,
                        timestamp=det_ts,
                        track_id=getattr(det, "track_id", None),
                    )
                )
            except Exception:  # noqa: BLE001
                continue
        if not raw_detections:
            return
        try:
            payload = DetectionBatchPayload(
                frame_id=frame_id,
                frame_width=frame_width,
                frame_height=frame_height,
                captured_at=captured_at,
                telemetry=telemetry,
                detections=raw_detections,
            )
            confirmed = self._unreal_local_pipeline.process_batch(payload)
            confirmed_count = len(confirmed or [])
            _log.info(
                "Unreal local YOLO batch frame_id=%s raw=%d confirmed=%d became_confirmed=%s",
                frame_id,
                len(raw_detections),
                confirmed_count,
                "yes" if confirmed_count > 0 else "no",
            )
            if confirmed_count > 0:
                _log.info(
                    "Unreal local YOLO confirmed classes=%s frame_id=%s",
                    [det.class_id for det in confirmed],
                    frame_id,
                )
        except Exception as exc:  # noqa: BLE001
            _log.debug("Unreal local YOLO pipeline processing failed: %s", exc)

    def process_backend_detections(self, objects: list[dict[str, object]]) -> None:
        if not objects:
            return
        if self._orbit_active or self._orbit_flow_state != OrbitFlowState.NORMAL_FLIGHT:
            return
        emitted = 0
        for obj in objects:
            lat = obj.get("lat")
            lon = obj.get("lon")
            try:
                lat_f = float(lat)
                lon_f = float(lon)
            except (TypeError, ValueError):
                continue
            class_id = self._extract_class_id(obj)
            ts = self._coerce_datetime(obj.get("timestamp")) or utc_now()
            if self._should_suppress_orbit_duplicate(lat_f, lon_f, class_id=class_id):
                continue
            updates = self._target_tracker.update(
                [
                    TargetObservation(
                        class_label=str(class_id),
                        lat=lat_f,
                        lon=lon_f,
                        timestamp=ts,
                        confidence=float(obj.get("confidence", 0.0) or 0.0),
                    )
                ]
            )
            if not updates or not updates[0].should_confirm:
                continue
            track = updates[0].track
            payload = {
                "object_id": f"track-{track.track_id}",
                "source_id": obj.get("source_id"),
                "class_id": class_id,
                "confidence": float(obj.get("confidence", 0.0) or 0.0),
                "lat": float(track.lat),
                "lon": float(track.lon),
                "track_id": int(track.track_id),
                "timestamp": ts,
            }
            bus.emit(Event.OBJECT_CONFIRMED_UI, payload)
            emitted += 1
        if emitted > 0:
            _log.info("Backend detections emitted stable targets=%d", emitted)

    @staticmethod
    def _extract_class_id(payload: dict[str, object]) -> int:
        raw = payload.get("class_id")
        if raw is None:
            raw = payload.get("cls")
        if raw is None:
            raw = payload.get("class")
        try:
            return int(raw)
        except Exception:
            text = str(raw or "").strip().lower()
            if text == "fire":
                return 1
            if text in ("human", "person"):
                return 2
            return 0

    @staticmethod
    def _extract_bbox_tuple(det: object) -> tuple[int, int, int, int] | None:
        bbox = getattr(det, "bbox", None)
        if isinstance(bbox, (tuple, list)) and len(bbox) == 4:
            try:
                return (
                    int(bbox[0]),
                    int(bbox[1]),
                    int(bbox[2]),
                    int(bbox[3]),
                )
            except Exception:  # noqa: BLE001
                return None
        if all(hasattr(det, k) for k in ("x1", "y1", "x2", "y2")):
            try:
                return (
                    int(getattr(det, "x1")),
                    int(getattr(det, "y1")),
                    int(getattr(det, "x2")),
                    int(getattr(det, "y2")),
                )
            except Exception:  # noqa: BLE001
                return None
        return None

    @staticmethod
    def _coerce_datetime(value: object) -> datetime | None:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                if text.endswith("Z"):
                    text = f"{text[:-1]}+00:00"
                dt = datetime.fromisoformat(text)
                if dt.tzinfo is not None:
                    return dt.replace(tzinfo=None)
                return dt
            except ValueError:
                return None
        return None

    def _remember_unreal_local_telemetry(self, frame_id: str, sample: TelemetrySample) -> None:
        self._unreal_local_telemetry_by_frame_id[frame_id] = sample
        self._unreal_local_frame_order.append(frame_id)
        while len(self._unreal_local_frame_order) > 64:
            stale_id = self._unreal_local_frame_order.popleft()
            self._unreal_local_telemetry_by_frame_id.pop(stale_id, None)

    def _pop_unreal_local_telemetry(self, frame_id: str) -> TelemetrySample | None:
        sample = self._unreal_local_telemetry_by_frame_id.pop(frame_id, None)
        if sample is None:
            return None
        try:
            self._unreal_local_frame_order.remove(frame_id)
        except ValueError:
            pass
        return sample

    def on_unreal_link_status(self, status: str) -> None:
        if status == self._unreal_status:
            return
        self._unreal_status = status
        self.linkStatusChanged.emit()
        if status == "waiting_for_route":
            if self._ignore_transient_unreal_waiting_for_route():
                return
            self._emit_warning(
                key="unreal_waiting_route",
                message="Unreal waiting for route; draw + confirm plan",
                severity="warn",
                cooldown_s=6,
            )
        elif status == "disconnected":
            self._emit_warning(
                key="unreal_disconnected",
                message="Unreal link disconnected",
                severity="warn",
                cooldown_s=8,
            )
            if self._backend == "unreal" and self._mission_state != MissionState.PREFLIGHT:
                self._stash_recoverable_mission()
                self._mission.abort_to_preflight("unreal_disconnected")
                self._pending_unreal_autostart = False
                self._active_path.set_normal()
                self.map_bridge.render_map()
                self.planConfirmedChanged.emit()
                self.flightControlsChanged.emit()
                self.toastRequested.emit("Unreal stopped. Restart sim and restore the saved mission.")
        elif status == "connected":
            self._emit_warning(
                key="unreal_connected",
                message="Unreal link connected",
                severity="info",
                cooldown_s=4,
            )

    def on_unreal_camera_status(self, status: str) -> None:
        mode_label = "H264" if self._unreal_video_mode == "h264_stream" else "JPEG"
        if status == "pyav_missing_jpeg_fallback":
            self._set_camera_status_detail("PyAV missing; using JPEG fallback (requested H264)")
            self.set_camera_available(False)
        elif status == "waiting_for_route":
            self._set_camera_status_detail(f"Waiting for route / drone not spawned ({mode_label})")
            self.set_camera_available(False)
        elif status == "disconnected":
            self._set_camera_status_detail(f"Camera stream disconnected ({mode_label})")
            self.set_camera_available(False)
        elif status == "paused":
            self._set_camera_status_detail(f"Camera paused ({mode_label})")
        elif status == "streaming":
            self._set_camera_status_detail(f"Live ({mode_label})")
        else:
            self._set_camera_status_detail(f"Camera not found ({mode_label})")

    def set_unreal_static_map(self, image_path: str, bounds: MapBounds) -> None:
        setattr(settings, "static_map_image_path", image_path)
        setattr(settings, "static_map_bounds", bounds.model_dump())
        setattr(settings, "map_provider", "static_image")
        _log.info(
            "Unreal static map set: image=%s bounds=(%.6f, %.6f)-(%.6f, %.6f)",
            image_path,
            bounds.lat_min,
            bounds.lon_min,
            bounds.lat_max,
            bounds.lon_max,
        )
        self.map_bridge.set_provider("static_image", offline=True, cache_dir="")
        self._set_map_refresh_needed(True)
        self.toastRequested.emit("Unreal map loaded")

    @Slot(str)
    def setBackend(self, backend: str) -> None:
        normalized = str(backend or "").strip().lower()
        aliases = {
            "unreal": "unreal",
            "mavlink": "mavlink",
            "stub": "stub",
            "custom": "custom",
            "client_bridge": "custom",
            "custom_sdk": "custom",
        }
        if normalized in aliases:
            normalized = aliases[normalized]
        if normalized not in ("unreal", "mavlink", "stub", "custom"):
            _log.warning("Backend switch rejected: unknown backend '%s'", backend)
            self.toastRequested.emit(f"Unknown backend: {backend}")
            return
        if normalized == self._backend:
            self.toastRequested.emit(f"Backend already set: {normalized}")
            return
        try:
            cfg_path = self._resolve_settings_path()
            raw = json.loads(cfg_path.read_text(encoding="utf-8"))
            raw["uav_backend"] = normalized
            raw["driver_type"] = normalized
            cfg_path.write_text(json.dumps(raw, indent=2, ensure_ascii=True), encoding="utf-8")
            setattr(settings, "uav_backend", normalized)
            setattr(settings, "driver_type", normalized)
            self._backend = normalized
            self.backendChanged.emit()
            _log.info("Backend switched to %s (settings: %s)", normalized, cfg_path)
            self.toastRequested.emit(f"Backend set to {normalized}. Restart module to apply.")
        except Exception as exc:  # noqa: BLE001
            _log.exception("Failed to switch backend to %s: %s", normalized, exc)
            self.toastRequested.emit(f"Failed to switch backend: {exc}")

    def _resolve_settings_path(self) -> Path:
        env_path = os.environ.get("FIRE_UAV_SETTINGS")
        if env_path:
            return Path(env_path).expanduser()
        import fire_uav

        pkg_root = Path(fire_uav.__file__).resolve().parent
        return pkg_root / "config" / "settings_default.json"

    @Slot(float)
    def setConfidence(self, value: float) -> None:
        self._confidence = float(value)
        self.det_vm.set_conf(self._confidence)
        self.confidenceChanged.emit()

    @Slot(bool)
    def setAutoOrbitEnabled(self, enabled: bool) -> None:
        flag = bool(enabled)
        if flag == self._auto_orbit_enabled:
            return
        self._auto_orbit_enabled = flag
        setattr(settings, "auto_orbit_enabled", flag)
        self.autoOrbitEnabledChanged.emit()

    @Slot(float)
    def setOrbitRadiusM(self, value: float) -> None:
        try:
            radius = float(value)
        except (TypeError, ValueError):
            return
        radius = max(1.0, radius)
        if abs(radius - self._orbit_radius_m) < 1e-6:
            return
        self._orbit_radius_m = radius
        setattr(settings, "orbit_radius_m", radius)
        self.orbitRadiusMChanged.emit()

    @Slot(bool)
    def setVideoVisible(self, visible: bool) -> None:
        self._video_visible = bool(visible)
        if self._unreal_link is not None:
            self._unreal_link.set_camera_enabled(self._video_visible)
        if not self._video_visible:
            try:
                self.video_bridge.clear_overlays()
            except Exception:
                _log.debug("Failed to clear overlays on tab switch", exc_info=True)
            if hasattr(self.det_vm, "reset"):
                try:
                    self.det_vm.reset()
                except Exception:
                    _log.debug("Failed to reset detector VM on tab switch", exc_info=True)

    @Slot(bool)
    def setDebugDisableDetectorDuringOrbit(self, enabled: bool) -> None:
        self._debug_disable_detector_during_orbit = bool(enabled)
        setattr(settings, "debug_disable_detector_during_orbit", self._debug_disable_detector_during_orbit)
        self.debugDetectorOrbitGuardChanged.emit()
        if self._orbit_active:
            return
        if not self._debug_disable_detector_during_orbit:
            self._restore_detector_after_orbit()

    @Slot(str)
    def selectConfirmedObject(self, object_id: str) -> None:
        target = self._objects_store.get(str(object_id))
        if target is None:
            return
        self._on_object_selected(target.object_id, float(target.lat), float(target.lon))

    @Slot()
    def restoreRecoverableMission(self) -> None:
        snapshot = self._recoverable_mission
        if snapshot is None or self._backend != "unreal" or self._unreal_link is None:
            return
        route = self._route_from_points(snapshot.path)
        if route is None:
            self.toastRequested.emit("Recoverable mission route is unavailable")
            return
        self._plan_vm.save_plan(list(snapshot.path))
        self._mission.confirm_plan(list(snapshot.path))
        if snapshot.home is not None:
            try:
                self._set_home_location(float(snapshot.home["lat"]), float(snapshot.home["lon"]))
            except Exception:
                pass
        self._clear_confirmed_objects()
        for obj in snapshot.confirmed_objects:
            bus.emit(Event.OBJECT_CONFIRMED_UI, dict(obj))
        if snapshot.selected_object_id:
            self._objects_store.set_selected(snapshot.selected_object_id)
            deps.selected_object_id = snapshot.selected_object_id
        if not self._send_unreal_route(route):
            self.toastRequested.emit("Failed to restore route to Unreal")
            return
        self._schedule_unreal_autostart()
        self._recoverable_mission = None
        self.recoverableMissionChanged.emit()
        self.planConfirmedChanged.emit()
        self.flightControlsChanged.emit()
        self.map_bridge.render_map()
        self.toastRequested.emit("Recoverable mission restored")

    @Slot()
    def discardRecoverableMission(self) -> None:
        if self._recoverable_mission is None:
            return
        self._recoverable_mission = None
        self.recoverableMissionChanged.emit()

    @Slot(str)
    def setUnrealVideoMode(self, mode: str) -> None:
        normalized = str(mode or "").strip().lower()
        if normalized not in ("jpeg_snapshots", "h264_stream"):
            self.toastRequested.emit(f"Unknown video mode: {mode}")
            return
        if normalized == self._unreal_video_mode:
            return
        try:
            cfg_path = self._resolve_settings_path()
            raw = json.loads(cfg_path.read_text(encoding="utf-8"))
            raw["unreal_video_mode"] = normalized
            cfg_path.write_text(json.dumps(raw, indent=2, ensure_ascii=True), encoding="utf-8")
            setattr(settings, "unreal_video_mode", normalized)
            self._unreal_video_mode = normalized
            self.unrealVideoModeChanged.emit()
            _log.info(
                "Unreal video mode updated in UI: %s (link_attached=%s)",
                normalized,
                self._unreal_link is not None,
            )
            if self._unreal_link is not None:
                self._unreal_link.set_camera_mode(normalized)
            else:
                self.toastRequested.emit("Unreal link not attached (mode saved only)")
            self.toastRequested.emit(f"Unreal video mode: {normalized}")
        except Exception as exc:  # noqa: BLE001
            _log.exception("Failed to update Unreal video mode: %s", exc)
            self.toastRequested.emit(f"Failed to update video mode: {exc}")

    @Slot(str)
    def handleMapConsole(self, message: str) -> None:
        self.map_bridge.handle_console(message)

    @Slot()
    def regenerateMap(self) -> None:
        self.map_bridge.render_map()

    @Slot()
    def refreshMapView(self) -> None:
        # WebEngine map updates are intentionally manual to avoid severe FPS degradation.
        self.map_bridge.render_map()
        self._set_map_refresh_needed(False)

    @Slot()
    def startHomePickMode(self) -> None:
        if self._home_pick_mode:
            return
        self._home_pick_mode = True
        self.homePickModeChanged.emit()
        self.toastRequested.emit("Click the map to set home; click current home again to clear it")

    @Slot()
    def stopHomePickMode(self) -> None:
        if not self._home_pick_mode:
            return
        self._home_pick_mode = False
        self.homePickModeChanged.emit()

    @Slot(float, float)
    def setHomeFromMap(self, lat: float, lon: float) -> None:
        _log.info("HOME_PICK click lat=%.6f lon=%.6f", lat, lon)
        current_home = getattr(deps, "home_location", None)
        if isinstance(current_home, dict):
            current_lat = current_home.get("lat")
            current_lon = current_home.get("lon")
            if current_lat is not None and current_lon is not None:
                try:
                    dist_m = haversine_m((float(lat), float(lon)), (float(current_lat), float(current_lon)))
                except (TypeError, ValueError):
                    dist_m = float("inf")
                if dist_m <= 5.0:
                    self.clearHomeLocation()
                    return
        self._set_home_location(lat, lon)
        self.stopHomePickMode()
        self._set_map_refresh_needed(True)
        self.toastRequested.emit(f"Home set: {lat:.6f}, {lon:.6f} - click Update map")

    @Slot()
    def clearHomeLocation(self) -> None:
        self.stopHomePickMode()
        if getattr(deps, "home_location", None) is None:
            self.toastRequested.emit("Home location not set")
            return
        deps.home_location = None
        setattr(settings, "home_lat", None)
        setattr(settings, "home_lon", None)
        self._clear_persisted_home_location()
        self._set_map_refresh_needed(True)
        _log.info("Home location cleared")
        self.toastRequested.emit("Home cleared - click Update map")

    @Slot()
    def startManualTargetMode(self) -> None:
        if self._object_spawn_mode:
            return
        self._object_spawn_mode = True
        self.objectSpawnModeChanged.emit()
        self.toastRequested.emit("Click the map to spawn a manual target")

    @Slot()
    def stopManualTargetMode(self) -> None:
        if not self._object_spawn_mode:
            return
        self._object_spawn_mode = False
        self.objectSpawnModeChanged.emit()

    @Slot(float, float)
    def spawnManualTargetAt(self, lat: float, lon: float) -> None:
        try:
            lat_val = float(lat)
            lon_val = float(lon)
        except (TypeError, ValueError):
            return
        _log.info("MANUAL_TARGET click lat=%.6f lon=%.6f", lat_val, lon_val)
        object_id = f"manual-{uuid.uuid4().hex[:8]}"
        payload = {
            "object_id": object_id,
            "class_id": 0,
            "confidence": 0.0,
            "lat": lat_val,
            "lon": lon_val,
            "track_id": None,
            "timestamp": utc_now(),
        }
        bus.emit(Event.OBJECT_CONFIRMED_UI, payload)
        self._objects_store.set_selected(object_id)
        deps.selected_object_id = object_id
        self.stopManualTargetMode()

    @Slot()
    def generatePath(self) -> None:
        self.map_bridge.generate_path()

    @Slot()
    def savePlan(self) -> None:
        self.map_bridge.save_plan()

    @Slot(str)
    def importGeoJson(self, filename: str) -> None:
        self.map_bridge.import_geojson(filename)

    @Slot(str)
    def importKml(self, filename: str) -> None:
        self.map_bridge.import_kml(filename)

    @Slot(str, bool, str)
    def setMapProvider(self, provider: str, offline: bool, cacheDir: str = "") -> None:
        self.map_bridge.set_provider(provider, offline=offline, cache_dir=cacheDir or None)

    @Slot()
    def confirmPlan(self) -> None:
        path = self._plan_vm.get_path()
        if not path:
            self.toastRequested.emit("Draw/generate a route first")
            return
        if not self._telemetry_available() and self._backend != "unreal":
            self.toastRequested.emit("Telemetry unavailable; enable Sim telemetry or connect UAV")
            return
        if self._maybe_prompt_route_battery_advisory(path, action="confirm_plan"):
            return
        if not self._mission.confirm_plan(path):
            self.toastRequested.emit("Route invalid; cannot confirm")
            return
        setattr(deps, "active_path_kind", "mission")
        setattr(deps, "debug_flight_progress", 0.0)
        self._active_path.set_normal()
        self._update_route_estimate(self._active_path.get_active_path())
        render_now = getattr(self.map_bridge, "_render_map_now", None)
        if callable(render_now):
            render_now()
        else:
            self.map_bridge.render_map()
        self.planConfirmedChanged.emit()
        self.flightControlsChanged.emit()
        if self._latest_telemetry:
            if not self._route_energy_ok(path):
                self._emit_warning(
                    key="battery_route_insufficient",
                    message="Insufficient battery for planned route",
                    severity="warn",
                    cooldown_s=8,
                )
        if self._backend == "unreal" and self._unreal_link is not None:
            self.startFlight(skip_checks=True)
            return
        self.toastRequested.emit("Plan confirmed")

    @Slot()
    def startFlight(self, *, skip_checks: bool = False, allow_unsafe_energy: bool = False) -> None:
        if not self._mission.plan_confirmed:
            self.toastRequested.emit("Confirm the plan first")
            return
        if not skip_checks and not self._link_monitor.is_link_ok():
            self._emit_warning(
                key="uav_link_missing",
                message="UAV link missing; cannot start flight",
                severity="warn",
                cooldown_s=6,
            )
            if not self._allow_unsafe_start:
                return
        if not skip_checks and not self._camera_monitor.is_camera_ok():
            self._emit_warning(
                key="camera_missing",
                message="Camera not ready; cannot start flight",
                severity="warn",
                cooldown_s=6,
            )
            if not self._allow_unsafe_start:
                return
        confirmed_path = list(self._mission.confirmed_plan or [])
        if (
            not allow_unsafe_energy
            and self._maybe_prompt_route_battery_advisory(confirmed_path, action="start_flight")
        ):
            return
        if self._backend == "unreal" and self._unreal_link is not None:
            route = self._route_from_points(confirmed_path)
            if route is None:
                self.toastRequested.emit("Route invalid; cannot send")
                return
            if not self._send_unreal_route(route):
                self.toastRequested.emit("Failed to send route to Unreal")
                return
        if self._mission.start_flight(skip_checks=skip_checks):
            self._commands_enabled = True
            if self._recoverable_mission is not None:
                self._recoverable_mission = None
                self.recoverableMissionChanged.emit()
            self._route_complete_announced = False
            self._rtl_forced = False
            self._rtl_route_sent = False
            deps.rtl_path = None
            deps.debug_orbit_path = None
            self._active_path.clear_overrides_on_new_flight()
            self.flightControlsChanged.emit()
            self.toastRequested.emit("Flight started")
            self.refreshMapView()

    @Slot()
    def editRoute(self) -> None:
        if self._mission_state != MissionState.IN_FLIGHT:
            return
        if self._latest_telemetry is None:
            self.toastRequested.emit("Telemetry unavailable; cannot enter route edit")
            return
        current_plan = self._mission.confirmed_plan or self._plan_vm.get_path() or []
        if len(current_plan) < 2:
            self.toastRequested.emit("Current route is too short to edit")
            return
        if self._backend == "unreal" and self._unreal_link is not None:
            if not self._try_unreal_hold_for_route_edit():
                self.toastRequested.emit("Failed to send HOLD to Unreal")
                return
        anchor = (float(self._latest_telemetry.lat), float(self._latest_telemetry.lon))
        locked_path, editable_tail = split_route_for_edit(current_plan, anchor)
        self._route_edit_original_plan = [(float(lat), float(lon)) for lat, lon in current_plan]
        self._staged_plan = list(editable_tail)
        self._route_edit_mode = True
        self._route_edit_anchor = anchor
        self._route_edit_locked_path = list(locked_path)
        deps.debug_orbit_path = None
        deps.rtl_path = None
        setattr(deps, "route_edit_anchor", {"lat": anchor[0], "lon": anchor[1]})
        setattr(deps, "route_edit_preview_path", list(editable_tail))
        setattr(deps, "route_edit_locked_path", list(locked_path))
        self.map_bridge.render_map()
        self.toastRequested.emit("Edit mode: remaining route is unlocked; remove or append points, then Apply")
        self.flightControlsChanged.emit()

    @Slot()
    def applyRouteEdits(self) -> None:
        if not self._route_edit_mode:
            return
        if not self._staged_plan:
            self.toastRequested.emit("Draw at least 1 point before apply")
            return
        pts = self._normalize_route_edit_points(self._staged_plan)
        if self._maybe_prompt_route_battery_advisory(pts, action="apply_route_edits"):
            return
        route = self._route_from_points(pts)
        if route is None:
            self.toastRequested.emit("Route needs at least 2 points")
            return
        resume_failed = False
        if self._backend == "unreal" and self._unreal_link is not None:
            if not self._send_unreal_route(route):
                self.toastRequested.emit("Failed to send updated route to Unreal")
                return
            if not self._try_unreal_resume_after_route_edit():
                resume_failed = True
        self._plan_vm.save_plan(pts)
        self._mission.confirm_plan(pts)
        deps.rtl_path = None
        deps.debug_orbit_path = None
        self._clear_route_edit_state()
        self._active_path.set_normal()
        self._update_route_estimate(self._active_path.get_active_path())
        self.map_bridge.render_map()
        self.planConfirmedChanged.emit()
        self.flightControlsChanged.emit()
        self.map_bridge.render_map()
        if resume_failed:
            self.toastRequested.emit("Route sent; resume manually in Unreal")
        else:
            self.toastRequested.emit("Route updates applied")

    @Slot()
    def cancelRouteEdits(self) -> None:
        if not self._route_edit_mode:
            return
        if self._route_edit_original_plan is not None:
            self._plan_vm.save_plan(list(self._route_edit_original_plan))
        resume_failed = False
        if self._backend == "unreal" and self._unreal_link is not None:
            if not self._try_unreal_resume_after_route_edit():
                resume_failed = True
        self._clear_route_edit_state()
        self._active_path.set_normal()
        deps.rtl_path = None
        deps.debug_orbit_path = None
        self.map_bridge.render_map()
        self.planConfirmedChanged.emit()
        self.flightControlsChanged.emit()
        if resume_failed:
            self.toastRequested.emit("Edit cancelled; resumed original route (resume manually in Unreal)")
        else:
            self.toastRequested.emit("Edit cancelled; resumed original route")

    @Slot()
    def returnToHome(self) -> None:
        if self._mission_state != MissionState.IN_FLIGHT:
            return
        if not self._current_action_policy().can_rtl:
            if not self._link_monitor.is_link_ok():
                self._emit_warning(
                    key="uav_link_missing",
                    message="UAV link missing; RTL unavailable",
                    severity="warn",
                    cooldown_s=6,
                )
            return
        self._initiate_rtl(reason="operator_rtl", user_message="Return-to-home initiated")

    @Slot()
    def sendRtlRoute(self) -> None:
        if not self._current_action_policy().can_send_rtl_route:
            if not self._link_monitor.is_link_ok():
                self._emit_warning(
                    key="uav_link_missing",
                    message="UAV link missing; RTL unavailable",
                    severity="warn",
                    cooldown_s=6,
                )
            return
        if self._send_rtl_route():
            self.toastRequested.emit("RTL route sent")

    @Slot()
    def completeLanding(self) -> None:
        if not self._current_action_policy().can_complete_landing:
            return
        if self._mission_state == MissionState.POSTFLIGHT:
            if self._backend == "unreal":
                self._send_unreal_despawn()
            self._mission.set_preflight("operator_postflight_confirm")
            self._mission.invalidate_plan("operator_postflight_confirm")
            self._pending_unreal_autostart = False
            self._rtl_route_sent = False
            deps.rtl_path = None
            deps.debug_orbit_path = None
            self._active_path.set_normal()
            self._clear_confirmed_objects()
            self.map_bridge.render_map()
            self.planConfirmedChanged.emit()
            self.flightControlsChanged.emit()
            self.toastRequested.emit("Mission completed and reset to idle")
            return
        self._mission.land_complete("operator_land")
        self._rtl_route_sent = False
        self.toastRequested.emit("Landing confirmed")

    @Slot()
    def abortToPreflight(self) -> None:
        if not self._current_action_policy().can_abort_to_preflight:
            return
        if self._backend == "unreal":
            self._send_unreal_despawn()
        self._mission.abort_to_preflight("operator_abort")
        self._pending_unreal_autostart = False
        deps.rtl_path = None
        deps.debug_orbit_path = None
        self._rtl_route_sent = False
        self._active_path.set_normal()
        self._clear_confirmed_objects()
        self.map_bridge.render_map()
        self.planConfirmedChanged.emit()
        self.flightControlsChanged.emit()
        self.toastRequested.emit("Mission aborted to preflight")

    @Slot()
    def backToPlanning(self) -> None:
        if self._backend == "unreal":
            self._send_unreal_despawn()
        self._mission.set_preflight("postflight_reset")
        self._mission.invalidate_plan("postflight_reset")
        self._pending_unreal_autostart = False
        deps.rtl_path = None
        deps.debug_orbit_path = None
        self._clear_confirmed_objects()
        self.planConfirmedChanged.emit()
        self.flightControlsChanged.emit()
        self.toastRequested.emit("Back to planning")

    @Slot()
    def orbitConfirmedObject(self) -> None:
        if self._objects_store.count() == 0:
            self.toastRequested.emit("No confirmed objects available")
            return
        target = self._objects_store.selected()
        if target is None and self._objects_store.count() > 1:
            self.toastRequested.emit("Select an object on the map first")
            return
        if target is None:
            target = self._objects_store.latest()
        if target is None:
            return
        self._orbit_targets([target], source="manual")

    @Slot("QVariantList")
    def orbitSelectedObjects(self, object_ids: list) -> None:
        ids = [str(obj_id) for obj_id in object_ids or [] if str(obj_id)]
        if not ids:
            self.toastRequested.emit("Select at least one object")
            return
        targets = [self._objects_store.get(object_id) for object_id in ids]
        targets = [target for target in targets if target is not None]
        if not targets:
            self.toastRequested.emit("Selected objects unavailable")
            return
        self._orbit_targets(self._order_targets_for_orbit(targets), source="manual")

    @Slot()
    def orbitAllConfirmedObjects(self) -> None:
        targets = self._objects_store.all()
        if not targets:
            self.toastRequested.emit("No confirmed objects available")
            return
        self._orbit_targets(self._order_targets_for_orbit(targets), source="manual")

    @Slot(str)
    def respondRouteBatteryAdvisory(self, action: str) -> None:
        action = str(action or "").strip().lower()
        pending_action = self._pending_route_battery_action
        pending_path = list(self._pending_route_battery_path)
        rtl_available = self._route_battery_rtl_available
        self._clear_route_battery_advisory()
        if action == "cancel":
            self.toastRequested.emit("Route action canceled")
            return
        if action == "rtl":
            if not rtl_available:
                self.toastRequested.emit("Battery is insufficient even for safe RTL")
                return
            self._initiate_rtl(reason="route_battery_advisory", user_message="Returning home for recharge")
            return
        if action != "proceed":
            return
        if pending_action == "confirm_plan":
            if pending_path:
                self._confirm_plan_after_battery_override(pending_path)
            return
        if pending_action == "start_flight":
            self.startFlight(allow_unsafe_energy=True)
            return
        if pending_action == "apply_route_edits":
            self._apply_route_edits_after_battery_override(pending_path)

    @Slot(str)
    def respondOrbitBatteryAdvisory(self, action: str) -> None:
        action = str(action or "").strip().lower()
        targets = list(self._pending_orbit_battery_targets)
        source = self._pending_orbit_battery_source
        route = self._pending_orbit_advisory_route
        base_route = self._pending_orbit_advisory_base_route
        target = self._pending_orbit_advisory_target
        self._clear_orbit_battery_advisory()
        if action == "cancel":
            self.toastRequested.emit("Orbit canceled")
            return
        if action == "rtl":
            if not self._orbit_battery_rtl_available:
                self.toastRequested.emit("Battery is insufficient even for safe RTL")
                return
            self._initiate_rtl(reason="orbit_battery_advisory", user_message="Returning home for recharge")
            return
        if action == "proceed":
            if not targets or route is None or base_route is None or target is None:
                self.toastRequested.emit("Orbit target unavailable")
                return
            self._stage_orbit_route(
                route=route,
                base_route=base_route,
                target=target,
                active_targets=[target],
                source=f"{source}_forced",
            )

    @Slot(bool)
    def setSimTelemetryEnabled(
        self,
        enabled: bool,
        *,
        notify_route_missing: bool = True,
        allow_without_route: bool = False,
    ) -> bool:
        if self._debug_sim is None:
            return False
        enabled = bool(enabled)
        was_enabled = self._debug_sim.telemetry_enabled()
        if (
            enabled
            and not allow_without_route
            and not (self._mission.confirmed_plan or self._plan_vm.get_path())
        ):
            if notify_route_missing:
                self.toastRequested.emit("Draw/generate a route first")
            return False
        if enabled and not was_enabled:
            self._debug_sim.reset_progress()
            deps.debug_flight_progress = 0.0
            if self._latest_telemetry is not None:
                source = getattr(self._latest_telemetry, "source", None)
                if source == "debug":
                    self._latest_telemetry = None
                    deps.latest_telemetry = None
        self._debug_sim.set_telemetry_enabled(enabled)
        deps.debug_flight_enabled = enabled
        deps.debug_map_manual_refresh = bool(enabled)
        self.debugModeChanged.emit()
        self.statsChanged.emit()
        return bool(self._debug_sim.telemetry_enabled())

    @Slot(bool)
    def setSimCameraEnabled(self, enabled: bool) -> None:
        if self._debug_sim is None:
            return
        self._debug_sim.set_camera_enabled(enabled)
        if enabled:
            self.set_camera_available(True)
        self.simCameraEnabledChanged.emit()
        self.statsChanged.emit()

    @Slot(bool)
    def setBridgeModeEnabled(self, enabled: bool) -> None:
        if self._bridge_mode_enabled == enabled:
            return
        self._bridge_mode_enabled = bool(enabled)
        if self._debug_sim is not None:
            self._debug_sim.set_bridge_mode(enabled)
        self.bridgeModeChanged.emit()
        self.flightControlsChanged.emit()
        if not enabled:
            _log.info("BRIDGE_MODE -> OFF (streams unchanged)")
            return
        _log.info("BRIDGE_MODE -> ON")
        if self._debug_sim is None:
            return
        telemetry_ok = self.setSimTelemetryEnabled(
            True,
            notify_route_missing=False,
            allow_without_route=False,
        )
        if not telemetry_ok:
            self.toastRequested.emit("bridge mode needs a route: draw/generate + confirm route first")
            _log.info("SIM_TELEMETRY blocked (bridge; route missing)")
        else:
            _log.info("SIM_TELEMETRY ON (bridge)")
        self.setSimCameraEnabled(True)
        _log.info("SIM_CAMERA ON (bridge)")

    @Slot(str)
    def setBridgeBatteryProfile(self, profile_id: str) -> None:
        if not profile_id:
            return
        profile = self._bridge_profiles.get(profile_id)
        if not profile:
            return
        self._bridge_profile_id = profile_id
        self._bridge_battery_wh = float(profile.get("battery_wh", self._bridge_battery_wh))
        self._bridge_speed_mps = float(profile.get("speed_mps", self._bridge_speed_mps))
        if self._debug_sim is not None:
            self._debug_sim.set_speed_mps(self._bridge_speed_mps)
        self.bridgeBatteryProfileChanged.emit()
        self._update_route_estimate()

    @Slot(bool)
    def setAllowUnsafeStart(self, enabled: bool) -> None:
        self._allow_unsafe_start = bool(enabled)
        self._mission.set_allow_unsafe_start(self._allow_unsafe_start)
        self._attempt_unreal_autostart()
        self.unsafeStartChanged.emit()
        self.flightControlsChanged.emit()
    @Slot()
    def spawnConfirmedObject(self) -> None:
        if self._debug_sim is None:
            return
        det = self._debug_sim.spawn_confirmed_object()
        self.toastRequested.emit(f"Spawned object at {det.lat:.4f}, {det.lon:.4f}")

    @Slot()
    def startFlightDebugMode(self) -> None:
        if self._debug_sim is not None:
            self._debug_sim.reset_progress()
            deps.debug_flight_progress = 0.0
            if self._latest_telemetry is not None:
                source = getattr(self._latest_telemetry, "source", None)
                if source == "debug":
                    self._latest_telemetry = None
                    deps.latest_telemetry = None
        self.setSimTelemetryEnabled(True)
        self.toastRequested.emit("Sim telemetry ON")

    @Slot()
    def stopFlightDebugMode(self) -> None:
        self.setSimTelemetryEnabled(False)
        self.toastRequested.emit("Sim telemetry OFF")

    @Slot()
    def resetFlightDebugProgress(self) -> None:
        if self._debug_sim is None:
            return
        self._debug_sim.reset_progress()
        deps.debug_flight_progress = 0.0
        self.statsChanged.emit()

    @Slot()
    def orbitTarget(self) -> None:
        if not self._plan_vm.get_path():
            self.toastRequested.emit("Draw/generate a route first")
            return
        self.map_bridge.generate_path()
        self.map_bridge.recomputeOrbitPreview()

    @Slot()
    def clearDebugTarget(self) -> None:
        self.map_bridge.clearDebugTarget()

    @Slot()
    def rebuildRoute(self) -> None:
        try:
            self._plan_vm.rebuild_route_from_current_geom(None)
            self.map_bridge.render_map()
            self.toastRequested.emit("Route rebuild triggered (debug stub)")
        except Exception as exc:  # noqa: BLE001
            self.toastRequested.emit(str(exc))

    @Slot()
    def recomputeOrbitPreview(self) -> None:
        self.map_bridge.recomputeOrbitPreview()

    @Slot(str)
    def showToast(self, message: str) -> None:
        self.toastRequested.emit(message or "Debug notification")

    # ---------- helpers ---------- #
    def on_frame(self, frame: NDArray[np.uint8]) -> None:
        now = time.perf_counter()
        if self._last_frame_ts is not None:
            dt = now - self._last_frame_ts
            if dt > 0:
                inst_fps = 1.0 / dt
                self._fps = 0.85 * self._fps + 0.15 * inst_fps if self._fps else inst_fps
        self._last_frame_ts = now
        self._camera_monitor.on_frame()
        if not self._camera_available:
            self.set_camera_available(True)
        self._update_stats()
        self.video_bridge.update_frame(frame)

    def on_telemetry(self, sample: TelemetrySample) -> None:
        self._latest_telemetry = sample
        deps.latest_telemetry = sample
        uav_id = self._resolve_uav_id(sample)
        self._telemetry_store.update(uav_id, sample)
        state = self._uav_states.get(uav_id)
        if state is None:
            state = UavState(uav_id)
            self._uav_states[uav_id] = state
            self.uavStatesChanged.emit()
        state.update_from_sample(sample)
        self._link_monitor.on_telemetry(sample)
        self._flight_recorder.record_telemetry(sample)
        self._handle_mission_progress_from_telemetry(sample)
        self._handle_battery(sample)
        self._maybe_restore_route_after_orbit(sample)
        self._update_route_estimate()
        self.statsChanged.emit()

    def set_camera_available(self, flag: bool) -> None:
        self._camera_available = flag
        self.cameraAvailableChanged.emit()

    def _on_detections(self, dets) -> None:  # noqa: ANN001
        det_list = getattr(dets, "detections", [])
        if not det_list:
            self._detection_conf = 0.0
            self._last_detection_ts = None
            self._update_stats()
            return
        best_conf = max(getattr(d, "score", getattr(d, "confidence", 0.0)) for d in det_list)
        bbox = getattr(det_list[0], "bbox", None)
        if bbox is None and all(hasattr(det_list[0], k) for k in ("x1", "y1", "x2", "y2")):
            bbox = (
                getattr(det_list[0], "x1"),
                getattr(det_list[0], "y1"),
                getattr(det_list[0], "x2"),
                getattr(det_list[0], "y2"),
            )
        stable_count = getattr(self.det_vm, "last_stable_count", 0)
        stable_conf = getattr(self.det_vm, "last_stable_conf", 0.0)
        eff_conf = stable_conf if stable_count > 0 else best_conf
        self._detection_conf = eff_conf
        _log.debug(
            "GUI detection event: raw=%d best=%.2f stable=%d stable_conf=%.2f bbox=%s",
            len(det_list),
            best_conf,
            stable_count,
            stable_conf,
            bbox,
        )
        if stable_count > 0:
            self.toastRequested.emit(f"Confirmed: {stable_count} (avg {stable_conf:.2f})")
        else:
            self.toastRequested.emit(f"Detections: {len(det_list)} (best {best_conf:.2f})")
        self._log_detection_event(det_list, eff_conf, bbox)

        now = time.perf_counter()
        self._last_detection_ts = now
        if self._last_frame_ts is not None:
            self._latency_ms = max(0.0, (now - self._last_frame_ts) * 1000)
        self._update_stats()

    def _append_log(self, line: str) -> None:
        self._logs.append(line)
        self._trim_logs()
        self.logsChanged.emit()

    def _on_object_confirmed(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        obj_id = str(payload.get("object_id", ""))
        lat = payload.get("lat")
        lon = payload.get("lon")
        try:
            _log.info(
                "OBJECT_CONFIRMED_UI emitted id=%s lat=%.6f lon=%.6f store_count=%d",
                obj_id or "n/a",
                float(lat),
                float(lon),
                self._objects_store.count(),
            )
        except Exception:
            _log.info(
                "OBJECT_CONFIRMED_UI emitted id=%s lat=%s lon=%s store_count=%d",
                obj_id or "n/a",
                lat,
                lon,
                self._objects_store.count(),
            )
        cls_id = int(payload.get("class_id", -1))
        conf = float(payload.get("confidence", 0.0))
        track_id = payload.get("track_id")
        track_str = f", track {track_id}" if track_id is not None else ""
        msg = f"Object {obj_id} (class {cls_id}{track_str}, conf {conf:.2f}) detected"
        self._write_confirmed_detection_json(payload)
        self.objectNotificationReceived.emit(obj_id, cls_id, conf, msg, track_id)

    def _auto_orbit_after_confirm(self, object_id: str) -> None:
        target = self._objects_store.get(object_id) if object_id else self._objects_store.latest()
        if target is None:
            return
        self._orbit_targets([target], source="auto")

    def _set_orbit_flow_state(self, state: OrbitFlowState, *, reason: str) -> None:
        if self._orbit_flow_state == state:
            return
        _log.info(
            "orbit_flow_state: %s -> %s (reason=%s)",
            self._orbit_flow_state.value,
            state.value,
            reason,
        )
        self._orbit_flow_state = state

    def _apply_reaction_slowdown(self) -> None:
        if self._reaction_speed_override_active:
            return
        if self._backend != "unreal" or self._unreal_link is None:
            return
        if not self._commands_enabled:
            return
        if self._unreal_link.send_command("SET_SPEED", {"speed_mps": float(self._reaction_slow_speed_mps)}):
            self._reaction_speed_override_active = True

    def _clear_reaction_slowdown(self) -> None:
        if not self._reaction_speed_override_active:
            return
        if self._backend == "unreal" and self._unreal_link is not None:
            self._unreal_link.send_command("CLEAR_VELOCITY_OVERRIDE")
        self._reaction_speed_override_active = False

    def _start_reaction_window(self, target: ConfirmedObject) -> None:
        self._reaction_target_id = target.object_id
        self._reaction_started_monotonic = time.monotonic()
        self._set_orbit_flow_state(
            OrbitFlowState.TARGET_DETECTED_REACTION_WINDOW,
            reason=f"target_confirmed:{target.object_id}",
        )
        self._apply_reaction_slowdown()

    def _finish_reaction_window(self, *, reason: str) -> None:
        self._reaction_target_id = None
        self._reaction_started_monotonic = 0.0
        self._clear_reaction_slowdown()
        self._set_orbit_flow_state(OrbitFlowState.ROUTE_RESUME, reason=reason)
        self._set_orbit_flow_state(OrbitFlowState.NORMAL_FLIGHT, reason=f"{reason}:resume_done")

    def _tick_reaction_window(self) -> None:
        if self._orbit_flow_state != OrbitFlowState.TARGET_DETECTED_REACTION_WINDOW:
            return
        if self._reaction_started_monotonic <= 0.0:
            return
        elapsed = time.monotonic() - self._reaction_started_monotonic
        if elapsed < self._reaction_window_s:
            return
        target = self._objects_store.get(self._reaction_target_id or "")
        if self._auto_orbit_enabled and target is not None:
            self._orbit_targets([target], source="auto_timeout")
            return
        self._finish_reaction_window(reason="reaction_timeout_continue")

    def _queue_pending_orbit_target(self, target: ConfirmedObject, *, source: str) -> None:
        if target.object_id in self._pending_orbit_ids:
            return
        self._pending_orbit_ids.add(target.object_id)
        self._pending_orbit_queue.append((target.object_id, source))

    def _dequeue_pending_orbit_targets(self) -> list[tuple[ConfirmedObject, str]]:
        queued: list[tuple[ConfirmedObject, str]] = []
        while self._pending_orbit_queue:
            object_id, source = self._pending_orbit_queue.pop(0)
            self._pending_orbit_ids.discard(object_id)
            target = self._objects_store.get(object_id)
            if target is not None:
                queued.append((target, source))
        return queued

    def _pop_next_pending_orbit_target(
        self,
        sample: TelemetrySample | None = None,
    ) -> tuple[ConfirmedObject, str] | None:
        queued = self._dequeue_pending_orbit_targets()
        if not queued:
            return None
        if sample is None:
            target, source = queued[0]
            for queued_target, queued_source in queued[1:]:
                self._queue_pending_orbit_target(queued_target, source=queued_source)
            return target, source
        current = (float(sample.lat), float(sample.lon))
        best_idx = min(
            range(len(queued)),
            key=lambda idx: haversine_m(current, (float(queued[idx][0].lat), float(queued[idx][0].lon))),
        )
        target, source = queued.pop(best_idx)
        for queued_target, queued_source in queued:
            self._queue_pending_orbit_target(queued_target, source=queued_source)
        return target, source

    def attach_debug_sim(self, sim: DebugSimulationService) -> None:
        self._debug_sim = sim
        self._debug_sim.set_speed_mps(self._bridge_speed_mps)

    def get_confirmed_path(self) -> list[tuple[float, float]]:
        return self._mission.confirmed_plan or self._plan_vm.get_path()

    def get_active_path_for_sim(self) -> list[tuple[float, float]]:
        path = list(self._active_path.get_active_path())
        telemetry = self._latest_telemetry
        if (
            telemetry is None
            or not path
            or self._mission_state != MissionState.IN_FLIGHT
            or self._active_path.mode == ActivePathMode.ORBIT
        ):
            return path
        current = (float(telemetry.lat), float(telemetry.lon))
        if haversine_m(current, path[0]) <= 2.0:
            return path
        route = [Waypoint(lat=lat, lon=lon, alt=float(telemetry.alt)) for lat, lon in path]
        nearest_idx = self._nearest_waypoint_index(float(telemetry.lat), float(telemetry.lon), route)
        nearest_dist = haversine_m(current, path[nearest_idx])
        if nearest_dist > 250.0:
            return path
        remaining_path = path[nearest_idx:]
        if nearest_dist <= 2.0:
            return remaining_path
        return [current, *remaining_path]

    def _on_active_path_changed(self, _mode: ActivePathMode) -> None:
        setattr(
            deps,
            "active_path_kind",
            "maneuver" if self._active_path.mode == ActivePathMode.ORBIT else "mission",
        )
        self._update_route_estimate(self._active_path.get_active_path())
        try:
            self.map_bridge.render_map()
        except Exception:
            pass
        self.flightControlsChanged.emit()

    def _on_status_tick(self) -> None:
        self._link_monitor.check()
        self._camera_monitor.check()
        self._mission.refresh_readiness("status_tick")
        self._tick_reaction_window()
        self.flightControlsChanged.emit()

    def _schedule_unreal_autostart(self) -> None:
        if self._backend != "unreal":
            return
        self._pending_unreal_autostart = True
        self._attempt_unreal_autostart()

    def _attempt_unreal_autostart(self) -> None:
        if not self._pending_unreal_autostart:
            return
        if self._mission_state == MissionState.IN_FLIGHT:
            self._pending_unreal_autostart = False
            return
        if self._mission_state not in (MissionState.PREFLIGHT, MissionState.READY):
            self._pending_unreal_autostart = False
            return
        if not self._mission.plan_confirmed:
            self._pending_unreal_autostart = False
            return
        link_ok = self._link_monitor.is_link_ok()
        telemetry_seen = self._latest_telemetry is not None
        if not (link_ok or telemetry_seen):
            if not self._allow_unsafe_start:
                return
        self.startFlight(skip_checks=True)
        if self._mission_state == MissionState.IN_FLIGHT:
            self._pending_unreal_autostart = False

    def _on_mission_state(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        state_raw = payload.get("state")
        try:
            new_state = MissionState(str(state_raw))
        except Exception:
            return
        prev_state = self._mission_state
        if new_state == prev_state:
            return
        self._mission_state = new_state
        deps.mission_state = self._mission_state.value
        if prev_state == MissionState.READY and new_state == MissionState.IN_FLIGHT:
            _log.info("missionState transition: %s -> %s", prev_state.value, new_state.value)
        if new_state in (MissionState.PREFLIGHT, MissionState.READY):
            self._route_complete_announced = False
            self._clear_manual_orbit_state()
            self._clear_reaction_slowdown()
            self._reaction_target_id = None
            self._reaction_started_monotonic = 0.0
            self._pending_orbit_queue = []
            self._pending_orbit_ids = set()
            self._set_orbit_flow_state(OrbitFlowState.NORMAL_FLIGHT, reason="mission_state_reset")
            self._clear_route_edit_state()
            self._commands_enabled = True
            self._rtl_forced = False
            self._rtl_route_sent = False
            deps.rtl_path = None
            deps.debug_orbit_path = None
            deps.selected_object_id = None
            deps.debug_target = None
            if new_state == MissionState.PREFLIGHT:
                self._flight_summary = None
                self.flightSummaryChanged.emit()
        if new_state == MissionState.IN_FLIGHT:
            self._pending_unreal_autostart = False
            self._commands_enabled = self._link_monitor.is_link_ok()
        if new_state == MissionState.RTL:
            self._commands_enabled = self._link_monitor.is_link_ok()
        if new_state == MissionState.POSTFLIGHT:
            self._route_complete_announced = True
            self._clear_manual_orbit_state()
            self._clear_reaction_slowdown()
            self._reaction_target_id = None
            self._reaction_started_monotonic = 0.0
            self._pending_orbit_queue = []
            self._pending_orbit_ids = set()
            self._set_orbit_flow_state(OrbitFlowState.NORMAL_FLIGHT, reason="mission_postflight")
            self._clear_route_edit_state()
            deps.rtl_path = None
            deps.debug_orbit_path = None
        if new_state in (MissionState.PREFLIGHT, MissionState.READY, MissionState.POSTFLIGHT):
            self._active_path.set_normal()
        self.missionStateChanged.emit()
        self.flightControlsChanged.emit()

    def _on_link_status(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        status_raw = payload.get("status")
        try:
            self._link_status = LinkStatus(str(status_raw))
        except Exception:
            return
        if self._mission_state == MissionState.IN_FLIGHT:
            if self._link_status == LinkStatus.DISCONNECTED:
                self._commands_enabled = False
                self.toastRequested.emit("Link lost; commands disabled")
            elif self._link_status == LinkStatus.CONNECTED:
                self._commands_enabled = True
        if self._link_status == LinkStatus.DISCONNECTED:
            self._emit_warning(
                key="uav_link_missing",
                message="UAV link missing / telemetry lost",
                severity="warn",
                cooldown_s=6,
            )
        elif self._link_status == LinkStatus.DEGRADED:
            self._emit_warning(
                key="uav_link_stale",
                message="Telemetry stale; check UAV link",
                severity="warn",
                cooldown_s=6,
            )
        self._mission.refresh_readiness("link_status")
        self._attempt_unreal_autostart()
        self.linkStatusChanged.emit()
        self.flightControlsChanged.emit()

    def _on_camera_status(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        status_raw = payload.get("status")
        try:
            self._camera_status = CameraStatus(str(status_raw))
        except Exception:
            return
        if self._camera_status == CameraStatus.NOT_READY:
            self._emit_warning(
                key="camera_missing",
                message="Camera not ready / not streaming",
                severity="warn",
                cooldown_s=6,
            )
            self.set_camera_available(False)
        elif self._camera_status == CameraStatus.READY:
            self.set_camera_available(True)
        self._mission.refresh_readiness("camera_status")
        self.cameraStatusChanged.emit()
        self.flightControlsChanged.emit()

    def _on_capabilities_updated(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        caps = payload.get("capabilities")
        if hasattr(caps, "model_dump"):
            caps = caps.model_dump()
        if not isinstance(caps, dict):
            return
        self._capabilities.update(caps)
        self.flightControlsChanged.emit()

    def _on_warning_toast(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        key = str(payload.get("key", "warning"))
        message = str(payload.get("message", "Warning"))
        cooldown_s = float(payload.get("cooldown_s", 5))
        if self._toast_dedupe.should_show(key, cooldown_s):
            self.toastRequested.emit(message)

    def _on_flight_session_ended(self, _payload: object) -> None:
        summary = getattr(self._flight_recorder, "last_summary", None)
        if summary is None:
            return
        self._flight_summary = summary
        self.flightSummaryChanged.emit()

    def _on_plan_changed(self, pts: list) -> None:
        path = [(float(lat), float(lon)) for lat, lon in pts]
        if self._mission_state == MissionState.IN_FLIGHT:
            if not self._route_edit_mode:
                self.toastRequested.emit("Press Edit route first")
                restore = self._mission.confirmed_plan or self._route_edit_original_plan or []
                if restore:
                    self._plan_vm.save_plan([(float(lat), float(lon)) for lat, lon in restore])
                else:
                    deps.plan_data = {"path": []}
                self.map_bridge.render_map()
                self.flightControlsChanged.emit()
                return
            normalized = self._normalize_route_edit_points(path)
            self._staged_plan = normalized
            setattr(deps, "route_edit_preview_path", list(normalized))
            self.flightControlsChanged.emit()
            return
        self._mission.invalidate_plan("plan_changed")
        self._pending_unreal_autostart = False
        deps.rtl_path = None
        self.planConfirmedChanged.emit()
        self.flightControlsChanged.emit()
        self._update_route_estimate(path)

    def _on_object_selected(self, object_id: str, lat: float, lon: float) -> None:
        deps.selected_object_id = object_id
        self._objects_store.set_selected(object_id)
        deps.debug_target = {"lat": lat, "lon": lon}
        self._on_objects_changed()
        self.map_bridge.render_map()
        self.flightControlsChanged.emit()

    def _on_objects_changed(self) -> None:
        all_objects = self._objects_store.all()
        current_ids = {obj.object_id for obj in all_objects}
        new_ids = current_ids - self._known_confirmed_ids
        self._known_confirmed_ids = current_ids
        if (
            self._unreal_local_yolo_enabled
            and not self._unreal_local_first_confirmed_logged
            and self._objects_store.count() > 0
        ):
            self._unreal_local_first_confirmed_logged = True
            _log.info("Unreal local YOLO first confirmed target received")
        if new_ids:
            was_needed = self._map_refresh_needed
            self._set_map_refresh_needed(True)
            if not was_needed or self._toast_dedupe.should_show("map_refresh_needed_object", 4.0):
                self.toastRequested.emit("UAV detected object - click Update map to show on map")
            new_targets = [obj for obj in all_objects if obj.object_id in new_ids]
            if self._orbit_active or self._orbit_flow_state == OrbitFlowState.ORBIT_ACTIVE:
                for target in new_targets:
                    self._queue_pending_orbit_target(target, source="auto")
                _log.debug(
                    "Orbit active guard: queued new targets during orbit (count=%d)",
                    len(new_targets),
                )
            elif (
                self._mission_state == MissionState.IN_FLIGHT
                and self._commands_enabled
                and not self._route_edit_mode
                and new_targets
            ):
                ts_targets = [obj for obj in new_targets if obj.timestamp is not None]
                target = (
                    max(ts_targets, key=lambda obj: obj.timestamp or datetime.min)
                    if ts_targets
                    else new_targets[-1]
                )
                self._start_reaction_window(target)
                if self._auto_orbit_enabled:
                    self._orbit_targets([target], source="auto")
        deps.confirmed_objects = [
            {
                "object_id": obj.object_id,
                "class_id": obj.class_id,
                "confidence": obj.confidence,
                "lat": obj.lat,
                "lon": obj.lon,
                "track_id": obj.track_id,
                "display_index": index + 1,
                "selected": bool(obj.object_id == getattr(deps, "selected_object_id", None)),
            }
            for index, obj in enumerate(all_objects)
        ]
        self.confirmedObjectsChanged.emit()
        self.flightControlsChanged.emit()
        _log.info("OBJECT_CONFIRMED_UI store count=%d", self._objects_store.count())

    def _set_map_refresh_needed(self, needed: bool) -> None:
        flag = bool(needed)
        if flag == self._map_refresh_needed:
            return
        self._map_refresh_needed = flag
        self.mapRefreshNeededChanged.emit()

    def _set_home_location(self, lat: float, lon: float) -> None:
        try:
            lat_val = float(lat)
            lon_val = float(lon)
        except (TypeError, ValueError):
            return
        deps.home_location = {"lat": lat_val, "lon": lon_val}
        setattr(settings, "home_lat", lat_val)
        setattr(settings, "home_lon", lon_val)
        self._persist_home_location(lat_val, lon_val)
        _log.info("HOME_PICK stored lat=%.6f lon=%.6f", lat_val, lon_val)

    def _load_persisted_home_location(self) -> None:
        path = self._home_persist_path
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            _log.warning("Failed to read persisted home location")
            return
        lat = payload.get("lat")
        lon = payload.get("lon")
        try:
            lat_val = float(lat)
            lon_val = float(lon)
        except (TypeError, ValueError):
            return
        deps.home_location = {"lat": lat_val, "lon": lon_val}
        setattr(settings, "home_lat", lat_val)
        setattr(settings, "home_lon", lon_val)
        _log.info("HOME_PICK loaded lat=%.6f lon=%.6f", lat_val, lon_val)

    def _persist_home_location(self, lat: float, lon: float) -> None:
        try:
            self._home_persist_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"lat": float(lat), "lon": float(lon)}
            self._home_persist_path.write_text(json.dumps(payload, indent=2))
        except Exception:  # noqa: BLE001
            _log.exception("Failed to persist home location")

    def _clear_persisted_home_location(self) -> None:
        try:
            if self._home_persist_path.exists():
                self._home_persist_path.unlink()
        except Exception:  # noqa: BLE001
            _log.exception("Failed to delete persisted home location")

    def _current_action_policy(self) -> MissionActionPolicy:
        telemetry_available = self._telemetry_available()
        at_home = self._is_at_home()
        selected = self._objects_store.selected()
        selected_id = selected.object_id if selected else None
        supports_waypoints = bool(self._capabilities.get("supports_waypoints", True))
        supports_orbit = bool(self._capabilities.get("supports_orbit", True))
        supports_rtl = bool(self._capabilities.get("supports_rtl", True))
        policy = MissionActionPolicy.evaluate(
            mission_state=self._mission_state,
            link_ok=self._link_monitor.is_link_ok(),
            camera_ok=self._camera_monitor.is_camera_ok(),
            commands_enabled=self._commands_enabled,
            has_confirmed_plan=self._mission.plan_confirmed,
            active_path_mode=self._active_path.mode,
            confirmed_object_count=self._objects_store.count(),
            selected_object_id=selected_id,
            route_edit_mode=self._route_edit_mode,
            allow_unsafe_start=self._allow_unsafe_start,
            supports_waypoints=supports_waypoints,
            supports_orbit=supports_orbit,
            supports_rtl=supports_rtl,
            telemetry_available=telemetry_available,
            at_home=at_home,
        )
        self._log_orbit_availability_change(
            policy=policy,
            telemetry_available=telemetry_available,
            selected_object_id=selected_id,
            supports_orbit=supports_orbit,
        )
        return policy

    def _log_orbit_availability_change(
        self,
        *,
        policy: MissionActionPolicy,
        telemetry_available: bool,
        selected_object_id: str | None,
        supports_orbit: bool,
    ) -> None:
        state = (policy.can_open_orbit, policy.can_orbit)
        if self._last_orbit_availability == state:
            return
        self._last_orbit_availability = state

        open_blockers: list[str] = []
        orbit_blockers: list[str] = []
        confirmed_count = self._objects_store.count()
        in_flight = self._mission_state == MissionState.IN_FLIGHT
        if not in_flight:
            open_blockers.append("mission_not_in_flight")
            orbit_blockers.append("mission_not_in_flight")
        if not self._commands_enabled:
            open_blockers.append("commands_disabled")
            orbit_blockers.append("commands_disabled")
        if not telemetry_available:
            open_blockers.append("telemetry_unavailable")
            orbit_blockers.append("telemetry_unavailable")
        if confirmed_count <= 0:
            open_blockers.append("no_confirmed_targets")
            orbit_blockers.append("no_confirmed_targets")
        if not supports_orbit:
            open_blockers.append("orbit_not_supported")
            orbit_blockers.append("orbit_not_supported")
        if confirmed_count > 1 and selected_object_id is None:
            orbit_blockers.append("selection_required_for_multiple_targets")

        _log.info(
            "Orbit availability changed: can_open_orbit=%s can_orbit=%s open_blockers=%s orbit_blockers=%s "
            "context={mission_state=%s,commands_enabled=%s,telemetry_available=%s,confirmed_object_count=%d,"
            "selected_object=%s,supports_orbit=%s}",
            policy.can_open_orbit,
            policy.can_orbit,
            open_blockers or ["ready"],
            orbit_blockers or ["ready"],
            self._mission_state.value,
            self._commands_enabled,
            telemetry_available,
            confirmed_count,
            selected_object_id or "none",
            supports_orbit,
        )

    def _normalize_route_edit_points(self, pts: list[tuple[float, float]] | None) -> list[tuple[float, float]]:
        normalized = [(float(lat), float(lon)) for lat, lon in (pts or [])]
        if self._route_edit_anchor is not None:
            if not normalized:
                normalized = [self._route_edit_anchor]
            elif haversine_m(self._route_edit_anchor, normalized[0]) > 1.0:
                normalized = [self._route_edit_anchor, *normalized]
            else:
                normalized[0] = self._route_edit_anchor
        normalized = dedupe_path(normalized)
        if self._route_edit_anchor is not None and len(normalized) < 2:
            normalized = [self._route_edit_anchor, self._route_edit_anchor]
        return normalized

    def _clear_route_edit_state(self) -> None:
        self._route_edit_mode = False
        self._hold_sent_for_edit = False
        self._staged_plan = None
        self._route_edit_anchor = None
        self._route_edit_original_plan = None
        self._route_edit_locked_path = None
        setattr(deps, "route_edit_anchor", None)
        setattr(deps, "route_edit_preview_path", None)
        setattr(deps, "route_edit_locked_path", None)

    def _resolve_landing_target(self) -> WorldCoord | None:
        home = getattr(deps, "home_location", None)
        if isinstance(home, dict):
            lat = home.get("lat")
            lon = home.get("lon")
            if lat is not None and lon is not None:
                try:
                    return WorldCoord(lat=float(lat), lon=float(lon))
                except (TypeError, ValueError):
                    return None
        path = self._mission.confirmed_plan or self._plan_vm.get_path()
        route = self._route_from_points(path) or Route(version=1, waypoints=[], active_index=None)
        return resolve_base_location(settings, route, self._latest_telemetry)

    def _is_at_home(self) -> bool:
        if self._latest_telemetry is None:
            return False
        target = self._resolve_landing_target()
        if target is None:
            return False
        try:
            dist = haversine_m(
                (self._latest_telemetry.lat, self._latest_telemetry.lon),
                (target.lat, target.lon),
            )
        except Exception:  # noqa: BLE001
            return False
        return dist <= 30.0

    def _telemetry_available(self) -> bool:
        if self._backend == "unreal":
            return True
        if self._link_monitor.is_link_ok():
            return True
        if self._debug_sim is not None and self._debug_sim.telemetry_enabled():
            return True
        return False

    def _send_unreal_route(
        self,
        route: Route,
        *,
        orbit_target: tuple[float, float] | None = None,
        orbit_target_alt: float | None = None,
    ) -> bool:
        if self._unreal_link is None:
            return False
        wps = [{"lat": wp.lat, "lon": wp.lon, "alt": wp.alt} for wp in route.waypoints]
        payload = {
            "type": "route",
            "uav_id": self._unreal_uav_id,
            "version": route.version,
            "waypoints": wps,
            "active_index": route.active_index if route.active_index is not None else 0,
        }
        if orbit_target is not None:
            payload["orbit_target"] = {
                "lat": float(orbit_target[0]),
                "lon": float(orbit_target[1]),
            }
            if orbit_target_alt is not None and math.isfinite(float(orbit_target_alt)):
                payload["orbit_target"]["alt"] = float(orbit_target_alt)
        sent = self._unreal_link.send_route(payload)
        if sent:
            self._last_unreal_route_sent_monotonic = time.monotonic()
        return sent

    def _ignore_transient_unreal_waiting_for_route(self) -> bool:
        return self._within_unreal_route_send_grace()

    def _within_unreal_route_send_grace(self) -> bool:
        if self._backend != "unreal":
            return False
        last_sent = float(getattr(self, "_last_unreal_route_sent_monotonic", 0.0) or 0.0)
        if last_sent <= 0.0:
            return False
        grace_s = max(0.0, float(getattr(self, "_unreal_waiting_route_grace_s", 2.5) or 0.0))
        return (time.monotonic() - last_sent) <= grace_s

    def _recover_from_failed_orbit_attempt(self, *, reason: str, user_message: str) -> None:
        _log.warning("Orbit attempt failed: %s", reason)
        if self._orbit_active:
            return
        self._reaction_target_id = None
        self._reaction_started_monotonic = 0.0
        self._clear_reaction_slowdown()
        self._pending_orbit_queue = []
        self._pending_orbit_ids.clear()
        setattr(deps, "debug_orbit_targets", [])
        setattr(deps, "debug_orbit_sequence", [])
        setattr(deps, "debug_orbit_preview_paths", [])
        self._set_orbit_flow_state(OrbitFlowState.NORMAL_FLIGHT, reason=f"orbit_failed:{reason}")
        self.toastRequested.emit(user_message)

    def _resolve_orbit_target_altitude_m(self) -> float | None:
        if self._latest_telemetry is None:
            return None
        alt_agl = getattr(self._latest_telemetry, "alt_agl", None)
        if alt_agl is None:
            return None
        try:
            return max(0.0, float(self._latest_telemetry.alt) - float(alt_agl))
        except (TypeError, ValueError):
            return None

    def _try_unreal_hold_for_route_edit(self) -> bool:
        if self._backend != "unreal" or self._unreal_link is None:
            return True
        if self._hold_sent_for_edit:
            return True
        ok = self._unreal_link.send_command("HOLD")
        if ok:
            self._hold_sent_for_edit = True
        return ok

    def _try_unreal_resume_after_route_edit(self) -> bool:
        if self._backend != "unreal" or self._unreal_link is None:
            return True
        if self._unreal_link.send_command("RESUME"):
            return True
        return self._unreal_link.send_command("CONTINUE")

    def _emit_warning(self, *, key: str, message: str, severity: str, cooldown_s: float) -> None:
        bus.emit(
            Event.WARNING_TOAST,
            {
                "key": key,
                "message": message,
                "severity": severity,
                "cooldown_s": cooldown_s,
            },
        )

    def _clear_route_battery_advisory(self) -> None:
        changed = (
            self._route_battery_advisory_visible
            or bool(self._route_battery_advisory_text)
            or self._route_battery_rtl_available
            or bool(self._pending_route_battery_action)
            or bool(self._pending_route_battery_path)
        )
        self._route_battery_advisory_visible = False
        self._route_battery_advisory_text = ""
        self._route_battery_rtl_available = False
        self._pending_route_battery_action = ""
        self._pending_route_battery_path = []
        if changed:
            self.routeBatteryAdvisoryChanged.emit()

    def _clear_orbit_battery_advisory(self) -> None:
        changed = (
            self._orbit_battery_advisory_visible
            or bool(self._orbit_battery_advisory_text)
            or self._orbit_battery_rtl_available
            or bool(self._pending_orbit_battery_targets)
        )
        self._orbit_battery_advisory_visible = False
        self._orbit_battery_advisory_text = ""
        self._orbit_battery_rtl_available = False
        self._pending_orbit_battery_targets = []
        self._pending_orbit_battery_source = "manual"
        self._pending_orbit_advisory_route = None
        self._pending_orbit_advisory_base_route = None
        self._pending_orbit_advisory_target = None
        if changed:
            self.orbitBatteryAdvisoryChanged.emit()

    def _route_energy_summary(
        self,
        path: list[tuple[float, float]],
        *,
        telemetry: TelemetrySample | None = None,
    ) -> dict[str, float | bool] | None:
        sample = telemetry or self._latest_telemetry
        if sample is None:
            fallback_lat, fallback_lon = (0.0, 0.0)
            if path:
                fallback_lat, fallback_lon = path[0]
            else:
                center = getattr(settings, "map_center", None)
                if isinstance(center, (list, tuple)) and len(center) >= 2:
                    try:
                        fallback_lat, fallback_lon = float(center[0]), float(center[1])
                    except (TypeError, ValueError):
                        pass
            sample = TelemetrySample(
                lat=fallback_lat,
                lon=fallback_lon,
                alt=120.0,
                battery=1.0,
                battery_percent=100.0,
                timestamp=utc_now(),
            )
        route = self._route_from_points(path)
        if route is None:
            return None
        estimate = self._estimate_route_energy(route, telemetry=sample)
        if estimate is None:
            return None
        available = sample.battery_percent
        if available is None:
            available = max(0.0, min(100.0, sample.battery * 100.0))
        return {
            "can_complete": bool(estimate.can_complete),
            "required_percent": float(estimate.required_percent),
            "margin_percent": float(estimate.margin_percent),
            "available_percent": float(available),
            "reserved_percent": float(getattr(settings, "min_return_percent", 20.0) or 0.0),
        }

    def _rtl_route_available(self) -> bool:
        route = self._build_rtl_route(silent=True)
        if route is None:
            return False
        estimate = self._estimate_route_energy(route)
        return bool(estimate.can_complete) if estimate is not None else False

    def _show_route_battery_advisory(
        self,
        *,
        action: str,
        path: list[tuple[float, float]],
        summary: dict[str, float | bool],
    ) -> None:
        required = float(summary.get("required_percent", 0.0) or 0.0)
        margin = float(summary.get("margin_percent", 0.0) or 0.0)
        available = float(summary.get("available_percent", 0.0) or 0.0)
        reserved = float(summary.get("reserved_percent", 0.0) or 0.0)
        self._pending_route_battery_action = action
        self._pending_route_battery_path = list(path)
        self._route_battery_rtl_available = self._rtl_route_available()
        self._route_battery_advisory_visible = True
        self._route_battery_advisory_text = (
            "Route may not be completed safely. "
            f"Need about {required:.1f}% plus reserve {reserved:.1f}%; "
            f"available {available:.1f}%, margin {margin:.1f}%."
        )
        self.routeBatteryAdvisoryChanged.emit()

    def _maybe_prompt_route_battery_advisory(self, path: list[tuple[float, float]], *, action: str) -> bool:
        summary = self._route_energy_summary(path)
        if summary is None or bool(summary["can_complete"]):
            return False
        self._emit_warning(
            key="battery_route_insufficient",
            message="Insufficient battery for planned route",
            severity="warn",
            cooldown_s=8,
        )
        self._show_route_battery_advisory(action=action, path=path, summary=summary)
        return True

    def _confirm_plan_after_battery_override(self, path: list[tuple[float, float]]) -> None:
        if not self._mission.confirm_plan(path):
            self.toastRequested.emit("Route invalid; cannot confirm")
            return
        setattr(deps, "active_path_kind", "mission")
        setattr(deps, "debug_flight_progress", 0.0)
        self._active_path.set_normal()
        self._update_route_estimate(self._active_path.get_active_path())
        render_now = getattr(self.map_bridge, "_render_map_now", None)
        if callable(render_now):
            render_now()
        else:
            self.map_bridge.render_map()
        self.planConfirmedChanged.emit()
        self.flightControlsChanged.emit()
        if self._backend == "unreal" and self._unreal_link is not None:
            self.startFlight(skip_checks=True, allow_unsafe_energy=True)
            return
        self.toastRequested.emit("Plan confirmed")

    def _apply_route_edits_after_battery_override(self, pts: list[tuple[float, float]]) -> None:
        route = self._route_from_points(pts)
        if route is None:
            self.toastRequested.emit("Route needs at least 2 points")
            return
        resume_failed = False
        if self._backend == "unreal" and self._unreal_link is not None:
            if not self._send_unreal_route(route):
                self.toastRequested.emit("Failed to send updated route to Unreal")
                return
            if not self._try_unreal_resume_after_route_edit():
                resume_failed = True
        self._plan_vm.save_plan(pts)
        self._mission.confirm_plan(pts)
        deps.rtl_path = None
        deps.debug_orbit_path = None
        self._clear_route_edit_state()
        self._active_path.set_normal()
        self._update_route_estimate(self._active_path.get_active_path())
        self.map_bridge.render_map()
        self.planConfirmedChanged.emit()
        self.flightControlsChanged.emit()
        self.map_bridge.render_map()
        if resume_failed:
            self.toastRequested.emit("Route sent; resume manually in Unreal")
        else:
            self.toastRequested.emit("Route updates applied")

    def _set_camera_status_detail(self, message: str) -> None:
        if message == self._camera_status_detail:
            return
        self._camera_status_detail = message
        self.cameraStatusDetailChanged.emit()

    def _update_route_estimate(self, path: list[tuple[float, float]] | None = None) -> None:
        if path is None:
            path = self._active_path.get_active_path()
        max_distance_m = self._resolve_max_distance_m()
        fallback_lat, fallback_lon = (0.0, 0.0)
        if path:
            fallback_lat, fallback_lon = path[-1]
        else:
            center = getattr(settings, "map_center", None)
            if isinstance(center, (list, tuple)) and len(center) >= 2:
                try:
                    fallback_lat, fallback_lon = float(center[0]), float(center[1])
                except (TypeError, ValueError):
                    pass
        available = 100.0
        if self._latest_telemetry and self._latest_telemetry.battery_percent is not None:
            available = float(self._latest_telemetry.battery_percent)
        telemetry = self._latest_telemetry or TelemetrySample(
            lat=fallback_lat,
            lon=fallback_lon,
            alt=120.0,
            battery=1.0,
            battery_percent=available,
            timestamp=utc_now(),
        )
        base_route = self._route_from_points(path) or Route(version=1, waypoints=[], active_index=None)
        base = self._resolve_base_location(base_route, telemetry)
        deps.route_stats = {
            "max_distance_m": max_distance_m,
            "available_percent": available,
            "reserved_percent": float(getattr(settings, "min_return_percent", 20.0) or 0.0),
            "base": [base.lon, base.lat] if base else None,
            "clamp_enabled": self._mission_state in (MissionState.IN_FLIGHT, MissionState.RTL),
        }
        if len(path) < 2:
            self._route_battery_text = "Route: --"
            self._route_battery_remaining_text = "Remaining: --"
            self._route_battery_warning = False
            self.routeBatteryChanged.emit()
            return

        if max_distance_m <= 0:
            self._route_battery_text = "Route: n/a"
            self._route_battery_remaining_text = "Remaining: n/a"
            self._route_battery_warning = False
            self.routeBatteryChanged.emit()
            return

        route_distance = 0.0
        for a, b in zip(path[:-1], path[1:]):
            route_distance += haversine_m(a, b)
        route_percent = (route_distance / max_distance_m) * 100.0

        summary = self._route_energy_summary(path, telemetry=telemetry)
        required_percent = route_percent
        remaining = available - route_percent
        if summary is not None:
            required_percent = float(summary.get("required_percent", route_percent) or route_percent)
            reserved = float(summary.get("reserved_percent", 0.0) or 0.0)
            remaining = float(summary.get("margin_percent", available - route_percent - reserved) or 0.0)

        if not self._link_monitor.is_link_ok():
            self._emit_warning(
                key="no_uav_stub_battery",
                message="UAV link missing; using stub battery model",
                severity="warn",
                cooldown_s=12,
            )

        if remaining < 0:
            if not self._route_battery_warning:
                self._emit_warning(
                    key="route_energy_insufficient",
                    message="Insufficient battery for return to home",
                    severity="warn",
                    cooldown_s=8,
                )
            self._route_battery_warning = True
        else:
            self._route_battery_warning = False

        self._route_battery_remaining_text = f"Remaining after route+reserve: {remaining:.1f}%"
        self._route_battery_text = (
            f"Route+RTL: {required_percent:.1f}%"
            if required_percent > 0
            else (f"Route: {route_percent:.1f}%" if route_percent > 0 else "Route: --")
        )
        self.routeBatteryChanged.emit()

    def _orbit_targets(
        self,
        targets: list[ConfirmedObject],
        *,
        source: str = "manual",
        allow_unsafe: bool = False,
    ) -> None:
        if self._mission_state != MissionState.IN_FLIGHT:
            return
        if not self._commands_enabled:
            self.toastRequested.emit("Link lost; commands disabled")
            return
        self._clear_orbit_battery_advisory()
        if not self._latest_telemetry:
            self.toastRequested.emit("No telemetry available for orbit")
            return
        if self._route_edit_mode:
            self.toastRequested.emit("Apply or cancel route edit before orbit")
            return
        if not targets:
            self.toastRequested.emit("No confirmed objects available")
            return
        if self._orbit_active:
            for target in targets:
                self._queue_pending_orbit_target(target, source=source)
            _log.info("Orbit start skipped: orbit already active; queued=%d", len(targets))
            return

        valid_targets: list[ConfirmedObject] = []
        for target in targets:
            try:
                lat = float(target.lat)
                lon = float(target.lon)
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(lat) and math.isfinite(lon)):
                continue
            valid_targets.append(target)
        if not valid_targets:
            self._recover_from_failed_orbit_attempt(
                reason=f"invalid_target:{source}",
                user_message="No valid orbit targets",
            )
            return
        for target in valid_targets[1:]:
            self._queue_pending_orbit_target(target, source=source)
        preview_targets: list[ConfirmedObject] = []
        preview_target_ids: set[str] = set()
        for target in valid_targets:
            if target.object_id in preview_target_ids:
                continue
            preview_targets.append(target)
            preview_target_ids.add(target.object_id)
        for object_id, _queued_source in self._pending_orbit_queue:
            queued_target = self._objects_store.get(object_id)
            if queued_target is not None and queued_target.object_id not in preview_target_ids:
                preview_targets.append(queued_target)
                preview_target_ids.add(queued_target.object_id)
        setattr(
            deps,
            "debug_orbit_targets",
            [(float(target.lat), float(target.lon)) for target in preview_targets],
        )
        if self._latest_telemetry is not None:
            setattr(
                deps,
                "debug_orbit_sequence",
                [
                    (float(self._latest_telemetry.lat), float(self._latest_telemetry.lon)),
                    *[(float(target.lat), float(target.lon)) for target in preview_targets],
                ],
            )
        advisory_targets = list(preview_targets)
        valid_targets = valid_targets[:1]

        base_path = self._mission.confirmed_plan or self._plan_vm.get_path()
        base_route = self._route_from_points(base_path)
        if base_route is None:
            self._recover_from_failed_orbit_attempt(
                reason=f"missing_base_route:{source}",
                user_message="Confirm a route before orbiting",
            )
            return
        if self._latest_telemetry is not None and base_route.waypoints:
            base_route.active_index = self._nearest_rejoin_segment_index(
                float(self._latest_telemetry.lat),
                float(self._latest_telemetry.lon),
                base_route.waypoints,
            )
        setattr(
            deps,
            "debug_orbit_preview_paths",
            self._build_orbit_preview_paths(
                targets=preview_targets,
                current_state=self._latest_telemetry,
                base_route=base_route,
            ),
        )
        self._orbit_resume_route = base_route.model_copy(deep=True)

        state = self._latest_telemetry
        target = valid_targets[0]
        route = self._plan_vm._route_planner.plan_maneuver(
            current_state=state,
            target_lat=float(target.lat),
            target_lon=float(target.lon),
            base_route=base_route,
            allow_unsafe=allow_unsafe,
        )
        unsafe_route = self._plan_vm._route_planner.plan_maneuver(
            current_state=state,
            target_lat=float(target.lat),
            target_lon=float(target.lon),
            base_route=base_route,
            allow_unsafe=True,
        )
        if route is None:
            if unsafe_route is not None and not allow_unsafe:
                queued_targets = [target]
                for object_id, _queued_source in self._pending_orbit_queue:
                    queued_target = self._objects_store.get(object_id)
                    if queued_target is not None:
                        queued_targets.append(queued_target)
                self._show_orbit_battery_advisory(
                    targets=queued_targets,
                    source=source,
                    orbit_route=unsafe_route,
                    base_route=base_route,
                    target=target,
                )
                return
            self._emit_warning(
                key="battery_orbit_insufficient",
                message="Insufficient battery for orbit",
                severity="warn",
                cooldown_s=8,
            )
            self._recover_from_failed_orbit_attempt(
                reason=f"planner_rejected:{source}",
                user_message="Orbit unavailable: energy/path check failed",
            )
            return
        advisory_route = route
        if len(advisory_targets) > 1:
            chain_route = self._build_orbit_chain_route(
                targets=advisory_targets,
                current_state=state,
                base_route=base_route,
            )
            if chain_route is not None:
                advisory_route = chain_route
        if not allow_unsafe and self._orbit_requires_battery_prompt(advisory_route):
            queued_targets = [target]
            for object_id, _queued_source in self._pending_orbit_queue:
                queued_target = self._objects_store.get(object_id)
                if queued_target is not None:
                    queued_targets.append(queued_target)
            self._show_orbit_battery_advisory(
                targets=queued_targets,
                source=source,
                orbit_route=advisory_route,
                base_route=base_route,
                target=target,
            )
            return
        self._stage_orbit_route(
            route=route,
            base_route=base_route,
            target=target,
            active_targets=valid_targets,
            source=source,
        )

    def _nearest_waypoint_index(self, lat: float, lon: float, route: list[Waypoint]) -> int:
        nearest_idx = 0
        nearest_dist = float("inf")
        for idx, wp in enumerate(route):
            dist = haversine_m((lat, lon), (wp.lat, wp.lon))
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_idx = idx
        return nearest_idx

    def _nearest_rejoin_segment_index(self, lat: float, lon: float, route: list[Waypoint]) -> int:
        if len(route) < 2:
            return 0

        scale_lat = 111_320.0
        scale_lon = 111_320.0 * math.cos(math.radians(float(lat)))

        def _to_local(wp: Waypoint) -> tuple[float, float]:
            return (
                (float(wp.lon) - float(lon)) * scale_lon,
                (float(wp.lat) - float(lat)) * scale_lat,
            )

        best_idx = 0
        best_dist = float("inf")

        for idx in range(len(route) - 1):
            start_wp = route[idx]
            end_wp = route[idx + 1]
            sx, sy = _to_local(start_wp)
            ex, ey = _to_local(end_wp)
            dx = ex - sx
            dy = ey - sy
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq <= 1e-6:
                candidate = start_wp
            else:
                t = max(0.0, min(1.0, (-(sx * dx + sy * dy)) / seg_len_sq))
                candidate = Waypoint(
                    lat=float(start_wp.lat) + (float(end_wp.lat) - float(start_wp.lat)) * t,
                    lon=float(start_wp.lon) + (float(end_wp.lon) - float(start_wp.lon)) * t,
                    alt=float(start_wp.alt) + (float(end_wp.alt) - float(start_wp.alt)) * t,
                )
            dist = haversine_m((lat, lon), (candidate.lat, candidate.lon))
            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        return best_idx

    def _maybe_restore_route_after_orbit(self, sample: TelemetrySample) -> None:
        if not self._orbit_active:
            return
        completion_wp = self._orbit_route_end_wp or self._orbit_rejoin_wp
        if completion_wp is None:
            return

        dist_m = haversine_m(
            (float(sample.lat), float(sample.lon)),
            (completion_wp.lat, completion_wp.lon),
        )
        if dist_m > self._orbit_rejoin_threshold_m:
            self._orbit_rejoin_close_hits = 0
            return

        self._orbit_rejoin_close_hits += 1
        if self._orbit_rejoin_close_hits < 2:
            return
        if self._orbit_started_monotonic > 0.0:
            elapsed = time.monotonic() - self._orbit_started_monotonic
            if elapsed < self._orbit_min_complete_time_s:
                return

        queued = self._pop_next_pending_orbit_target(sample)
        next_target: ConfirmedObject | None = None
        next_source: str | None = None
        if queued is not None:
            queued_target, queued_source = queued
            if queued_source != "auto" or self._auto_orbit_enabled:
                next_target = queued_target
                next_source = queued_source
        self._mark_pending_orbit_targets_orbited()
        if self._backend == "unreal" and self._unreal_link is not None:
            resume_route = self._orbit_resume_route
            if next_target is None:
                self._unreal_link.send_command("ORBIT_STOP")
            if next_target is None:
                resume_route = self._build_final_resume_route(sample) or resume_route
            if next_target is None and resume_route is not None and resume_route.waypoints:
                if not self._send_unreal_route(resume_route):
                    self.toastRequested.emit("Orbit complete; failed to resume mission route")
                    return
        self._clear_manual_orbit_state(clear_preview=next_target is None)
        deps.debug_orbit_path = None
        deps.debug_target = None
        deps.selected_object_id = None
        self._objects_store.set_selected(None)
        self._on_objects_changed()
        self._active_path.set_normal()
        if next_target is None:
            self._set_orbit_flow_state(OrbitFlowState.ROUTE_RESUME, reason="orbit_rejoin_reached")
            self._set_orbit_flow_state(OrbitFlowState.NORMAL_FLIGHT, reason="route_resume_complete")
            self.map_bridge.render_map()
        else:
            self._set_orbit_flow_state(OrbitFlowState.NORMAL_FLIGHT, reason="orbit_chain_next_target")
        if next_target is None:
            self.toastRequested.emit("Orbit complete; resumed mission route")
        else:
            self.toastRequested.emit("Orbit complete; moving to next target")
            self._orbit_targets([next_target], source=f"queued_{next_source}")

    def _try_unreal_resume_before_route_send(self) -> None:
        if self._backend != "unreal" or self._unreal_link is None:
            return
        if self._unreal_link.send_command("RESUME"):
            return
        self._unreal_link.send_command("CONTINUE")

    def _clear_manual_orbit_state(self, *, clear_preview: bool = True) -> None:
        self._orbit_active = False
        self._orbit_rejoin_wp = None
        self._orbit_rejoin_close_hits = 0
        self._orbit_target_track_ids = set()
        self._orbit_target_centers = []
        self._orbit_route_end_wp = None
        self._orbit_resume_route = None
        self._orbit_resume_min_index = None
        self._orbit_started_monotonic = 0.0
        self._orbit_min_complete_time_s = 0.0
        if clear_preview:
            setattr(deps, "debug_orbit_targets", [])
            setattr(deps, "debug_orbit_sequence", [])
            setattr(deps, "debug_orbit_preview_paths", [])
        self._restore_detector_after_orbit()

    def _build_final_resume_route(self, sample: TelemetrySample) -> Route | None:
        base_route = None
        if self._orbit_resume_route is not None and self._orbit_resume_route.waypoints:
            base_route = self._orbit_resume_route.model_copy(deep=True)
        else:
            base_path = self._mission.confirmed_plan or self._plan_vm.get_path()
            base_route = self._route_from_points(base_path)
        if base_route is None or not base_route.waypoints:
            return None
        if base_route.active_index is None:
            base_route.active_index = self._nearest_rejoin_segment_index(
                float(sample.lat),
                float(sample.lon),
                base_route.waypoints,
            )
        exit_wp = Waypoint(
            lat=float(sample.lat),
            lon=float(sample.lon),
            alt=float(getattr(sample, "alt", self._latest_telemetry.alt if self._latest_telemetry else 120.0) or 120.0),
        )
        planner = getattr(self._plan_vm, "_route_planner", None)
        rejoin_path: list[Waypoint] | None = None
        try:
            if planner is not None and hasattr(planner, "plan_rejoin"):
                rejoin_path = planner.plan_rejoin(exit_wp, base_route)
            else:
                rejoin_path = build_rejoin(exit_wp, base_route)
        except Exception:
            _log.debug("Failed to build final orbit resume route", exc_info=True)
            return None
        if not rejoin_path:
            return None
        return Route(
            version=base_route.version if base_route.version is not None else 1,
            waypoints=list(rejoin_path),
            active_index=0,
        )

    def _extract_orbit_segment(
        self,
        route: Route,
        base_route: Route,
    ) -> tuple[list[Waypoint], Waypoint | None]:
        waypoints = list(route.waypoints or [])
        if not waypoints:
            return [], None

        base_waypoints = list(base_route.waypoints or [])
        start_idx = 0
        if base_waypoints and base_route.active_index is not None:
            start_idx = max(0, min(int(base_route.active_index), len(base_waypoints) - 1))

        rejoin_start_idx: int | None = None
        for base_idx in range(start_idx, len(base_waypoints)):
            base_suffix = base_waypoints[base_idx:]
            suffix_len = len(base_suffix)
            if suffix_len <= 0 or suffix_len > len(waypoints):
                continue
            route_idx = len(waypoints) - suffix_len
            route_suffix = waypoints[route_idx:]
            if not all(self._same_waypoint(a, b) for a, b in zip(route_suffix, base_suffix)):
                continue
            if suffix_len == 1 and (
                route_idx <= 0 or not self._same_waypoint(waypoints[route_idx - 1], waypoints[route_idx])
            ):
                continue
            if route_idx == 0:
                continue
            if all(self._same_waypoint(a, b) for a, b in zip(route_suffix, base_suffix)):
                rejoin_start_idx = route_idx
                break

        if rejoin_start_idx is None or rejoin_start_idx <= 0:
            return waypoints, waypoints[-1]

        orbit_segment = waypoints[:rejoin_start_idx]
        return orbit_segment, orbit_segment[-1]

    def _extract_resume_waypoint_index(self, route: Route, base_route: Route) -> int | None:
        waypoints = list(route.waypoints or [])
        base_waypoints = list(base_route.waypoints or [])
        if not waypoints or not base_waypoints:
            return None

        start_idx = 0
        if base_route.active_index is not None:
            start_idx = max(0, min(int(base_route.active_index), len(base_waypoints) - 1))

        for base_idx in range(start_idx, len(base_waypoints)):
            base_suffix = base_waypoints[base_idx:]
            suffix_len = len(base_suffix)
            if suffix_len <= 0 or suffix_len > len(waypoints):
                continue
            route_idx = len(waypoints) - suffix_len
            route_suffix = waypoints[route_idx:]
            if all(self._same_waypoint(a, b) for a, b in zip(route_suffix, base_suffix)):
                return base_idx
        return None

    def _build_orbit_preview_paths(
        self,
        *,
        targets: list[ConfirmedObject],
        current_state: TelemetrySample | None,
        base_route: Route,
    ) -> list[list[tuple[float, float]]]:
        if current_state is None or not targets or not base_route.waypoints:
            return []
        planner = getattr(self._plan_vm, "_route_planner", None)
        if planner is None or not hasattr(planner, "plan_maneuver"):
            return []

        preview_paths: list[list[tuple[float, float]]] = []
        simulated_state = current_state.model_copy(deep=True)
        for target in targets:
            try:
                route = planner.plan_maneuver(
                    current_state=simulated_state,
                    target_lat=float(target.lat),
                    target_lon=float(target.lon),
                    base_route=base_route,
                    allow_unsafe=True,
                )
            except Exception:
                _log.debug("Failed to build orbit preview path", exc_info=True)
                break
            if route is None or not route.waypoints:
                simulated_state = self._telemetry_from_target(simulated_state, target)
                continue

            orbit_segment, orbit_completion_wp = self._extract_orbit_segment(route, base_route)
            segment = orbit_segment or list(route.waypoints)
            coords = dedupe_path([(float(wp.lat), float(wp.lon)) for wp in segment], threshold_m=0.5)
            if len(coords) >= 2:
                preview_paths.append(coords)

            completion_wp = orbit_completion_wp or segment[-1]
            simulated_state = self._telemetry_from_waypoint(simulated_state, completion_wp)
        return preview_paths

    def _telemetry_from_waypoint(self, sample: TelemetrySample, waypoint: Waypoint) -> TelemetrySample:
        return sample.model_copy(
            update={
                "lat": float(waypoint.lat),
                "lon": float(waypoint.lon),
                "alt": float(waypoint.alt),
                "timestamp": utc_now(),
            }
        )

    def _telemetry_from_target(self, sample: TelemetrySample, target: ConfirmedObject) -> TelemetrySample:
        return sample.model_copy(
            update={
                "lat": float(target.lat),
                "lon": float(target.lon),
                "timestamp": utc_now(),
            }
        )

    def _estimate_orbit_min_complete_time_s(self, orbit_segment: list[Waypoint]) -> float:
        if len(orbit_segment) < 2:
            return 0.0
        distance_m = 0.0
        for a, b in zip(orbit_segment, orbit_segment[1:]):
            distance_m += haversine_m((float(a.lat), float(a.lon)), (float(b.lat), float(b.lon)))
        cruise_speed = max(0.5, float(getattr(settings, "cruise_speed_mps", 12.0) or 12.0))
        # Require most of the orbit segment to be flown before allowing route resume.
        return max(3.0, (distance_m / cruise_speed) * 0.6)

    @staticmethod
    def _same_waypoint(a: Waypoint, b: Waypoint) -> bool:
        return (
            haversine_m((float(a.lat), float(a.lon)), (float(b.lat), float(b.lon))) <= 0.5
            and abs(float(a.alt) - float(b.alt)) <= 0.5
        )

    def _set_pending_orbit_targets(self, targets: list[ConfirmedObject]) -> None:
        self._orbit_target_track_ids = set()
        self._orbit_target_centers = []
        for target in targets:
            tid = self._coerce_track_id(target.track_id)
            if tid is not None:
                self._orbit_target_track_ids.add(tid)
            self._orbit_target_centers.append((float(target.lat), float(target.lon)))

    def _order_targets_for_orbit(self, targets: list[ConfirmedObject]) -> list[ConfirmedObject]:
        if len(targets) <= 1:
            return list(targets)
        remaining = list(targets)
        ordered: list[ConfirmedObject] = []
        telemetry = self._latest_telemetry
        if telemetry is not None:
            current = (float(telemetry.lat), float(telemetry.lon))
        else:
            first = remaining[0]
            current = (float(first.lat), float(first.lon))
        while remaining:
            next_target = min(
                remaining,
                key=lambda target: haversine_m(current, (float(target.lat), float(target.lon))),
            )
            ordered.append(next_target)
            remaining.remove(next_target)
            current = (float(next_target.lat), float(next_target.lon))
        return ordered

    def _activate_orbit_target_lock(self) -> None:
        for track_id in self._orbit_target_track_ids:
            self._target_tracker.mark_in_orbit(track_id)
        for lat, lon in self._orbit_target_centers:
            self._target_tracker.add_suppression_zone(lat=lat, lon=lon)

    def _mark_pending_orbit_targets_orbited(self) -> None:
        for track_id in self._orbit_target_track_ids:
            self._target_tracker.mark_orbited(track_id)
        for lat, lon in self._orbit_target_centers:
            self._target_tracker.add_suppression_zone(lat=lat, lon=lon)

    def _mark_targets_orbited(self, targets: list[ConfirmedObject]) -> None:
        for target in targets:
            tid = self._coerce_track_id(target.track_id)
            if tid is not None:
                self._target_tracker.mark_orbited(tid)
            self._target_tracker.add_suppression_zone(lat=float(target.lat), lon=float(target.lon))

    def _apply_detector_pause_for_orbit(self) -> None:
        if not self._detector_running:
            return
        self.stopDetector()
        self._detector_forced_paused_by_orbit = True

    def _restore_detector_after_orbit(self) -> None:
        if not self._detector_forced_paused_by_orbit:
            return
        if self._detector_running:
            self._detector_forced_paused_by_orbit = False
            return
        self.startDetector()
        self._detector_forced_paused_by_orbit = False

    def _should_suppress_orbit_duplicate(self, lat: float, lon: float, *, class_id: int) -> bool:
        if class_id != 1:
            return False
        if not self._orbit_target_centers:
            return False
        for center in self._orbit_target_centers:
            if haversine_m(center, (lat, lon)) <= 20.0:
                return True
        return False

    def _stash_recoverable_mission(self) -> None:
        path = list(self._mission.confirmed_plan or self._plan_vm.get_path() or [])
        if len(path) < 2:
            return
        home = getattr(deps, "home_location", None)
        snapshot = RecoverableMissionSnapshot(
            path=[(float(lat), float(lon)) for lat, lon in path],
            home=(
                {"lat": float(home["lat"]), "lon": float(home["lon"])}
                if isinstance(home, dict) and home.get("lat") is not None and home.get("lon") is not None
                else None
            ),
            confirmed_objects=[dict(item) for item in getattr(deps, "confirmed_objects", []) or []],
            selected_object_id=str(getattr(deps, "selected_object_id", "") or "") or None,
        )
        self._recoverable_mission = snapshot
        self.recoverableMissionChanged.emit()

    @staticmethod
    def _coerce_track_id(value: object) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _clear_confirmed_objects(self) -> None:
        self._objects_store.clear()
        self._target_tracker.reset()
        self._known_confirmed_ids.clear()
        self._pending_orbit_queue = []
        self._pending_orbit_ids.clear()
        self._orbit_target_track_ids.clear()
        self._orbit_target_centers = []
        self._reaction_target_id = None
        self._unreal_local_telemetry_by_frame_id.clear()
        self._unreal_local_frame_order.clear()
        deps.selected_object_id = None
        setattr(deps, "debug_orbit_targets", [])
        setattr(deps, "debug_orbit_sequence", [])
        setattr(deps, "debug_orbit_preview_paths", [])

    def _handle_battery(self, sample: TelemetrySample) -> None:
        if self._mission_state not in (MissionState.IN_FLIGHT, MissionState.READY):
            return
        if self._rtl_forced:
            return
        suppress_runtime_route_check = (
            self._mission_state == MissionState.IN_FLIGHT and self._within_unreal_route_send_grace()
        )
        remaining_route = (
            self._build_remaining_active_route(sample)
            if self._mission_state == MissionState.IN_FLIGHT and not suppress_runtime_route_check
            else None
        )
        if remaining_route is not None:
            base_path = list(self._mission.confirmed_plan or self._plan_vm.get_path() or [])
            mission_base_route = self._route_from_points(base_path) or remaining_route
            estimate = self._estimate_route_energy(
                remaining_route,
                telemetry=sample,
                base_route=mission_base_route,
            )
            if estimate is not None and not estimate.can_complete:
                self._rtl_forced = True
                self._emit_warning(
                    key="battery_route_runtime_insufficient",
                    message="Battery is no longer sufficient for the remaining route; forcing return-to-home",
                    severity="error",
                    cooldown_s=8,
                )
                self._initiate_rtl(
                    reason="battery_runtime_insufficient",
                    user_message="Remaining route is unsafe; returning home",
                )
                return
        try:
            critical = self._energy_model.is_critical(sample)
        except Exception:
            critical = False
        if not critical:
            return
        self._rtl_forced = True
        bus.emit(Event.BATTERY_CRITICAL, {"battery_percent": sample.battery_percent})
        self._emit_warning(
            key="battery_critical",
            message="Critical battery; forcing return-to-home",
            severity="error",
            cooldown_s=10,
        )
        if self._mission_state == MissionState.IN_FLIGHT:
            self._initiate_rtl(reason="battery_critical", user_message="Critical battery; returning home")

    def _handle_mission_progress_from_telemetry(self, sample: TelemetrySample) -> None:
        mode = str(getattr(sample, "flight_mode", None) or getattr(sample, "status", None) or "").upper()
        if not mode:
            return
        if mode == "WAITING_FOR_ROUTE":
            if self._ignore_transient_unreal_waiting_for_route():
                return
            if self._mission_state != MissionState.PREFLIGHT:
                self._stash_recoverable_mission()
                self._mission.abort_to_preflight("sim_waiting_for_route")
            return
        if mode not in ("MISSION_COMPLETE", "ROUTE_COMPLETE", "COMPLETED"):
            return
        if self._route_complete_announced:
            return
        self._route_complete_announced = True
        if self._mission_state in (MissionState.IN_FLIGHT, MissionState.RTL):
            self._mission.land_complete("route_complete")
        self.toastRequested.emit("Маршрут завершён")

    def _route_energy_ok(self, pts: list[tuple[float, float]]) -> bool:
        if not self._latest_telemetry:
            return True
        route = self._route_from_points(pts)
        if route is None:
            return True
        estimate = self._estimate_route_energy(route)
        if estimate is None:
            return True
        return bool(estimate.can_complete)

    def _estimate_route_energy(
        self,
        route: Route,
        telemetry: TelemetrySample | None = None,
        *,
        base_route: Route | None = None,
    ):
        sample = telemetry or self._latest_telemetry
        if sample is None:
            return None
        base = self._resolve_base_location(base_route or route, sample)
        if base is None:
            return None
        try:
            return self._energy_model.estimate_route_feasibility(sample, route, base)
        except Exception:
            return None

    def _orbit_requires_battery_prompt(self, route: Route) -> bool:
        estimate = self._estimate_route_energy(route)
        if estimate is None:
            return False
        low_margin_percent = float(getattr(settings, "orbit_low_margin_percent", 5.0) or 5.0)
        if not estimate.can_complete:
            return True
        return estimate.margin_percent <= low_margin_percent

    def _build_orbit_chain_route(
        self,
        *,
        targets: list[ConfirmedObject],
        current_state: TelemetrySample,
        base_route: Route,
    ) -> Route | None:
        planner = getattr(self._plan_vm, "_route_planner", None)
        if planner is None or not targets:
            return None
        current_sample = current_state.model_copy(deep=True)
        chain_waypoints: list[Waypoint] = []
        for target in targets:
            route = planner.plan_maneuver(
                current_state=current_sample,
                target_lat=float(target.lat),
                target_lon=float(target.lon),
                base_route=base_route,
                allow_unsafe=True,
            )
            if route is None or not route.waypoints:
                return None
            segment = [wp.model_copy(deep=True) for wp in route.waypoints]
            if chain_waypoints and segment:
                segment = segment[1:]
            chain_waypoints.extend(segment)
            last_wp = route.waypoints[-1]
            current_sample = current_sample.model_copy(
                update={"lat": last_wp.lat, "lon": last_wp.lon, "alt": last_wp.alt}
            )
        return Route(
            version=base_route.version or 1,
            waypoints=chain_waypoints,
            active_index=0 if chain_waypoints else None,
        )

    def _show_orbit_battery_advisory(
        self,
        *,
        targets: list[ConfirmedObject],
        source: str,
        orbit_route: Route,
        base_route: Route,
        target: ConfirmedObject,
    ) -> None:
        estimate = self._estimate_route_energy(orbit_route)
        rtl_available = self._rtl_route_available()
        self._pending_orbit_battery_targets = list(targets)
        self._pending_orbit_battery_source = source
        self._pending_orbit_advisory_route = orbit_route.model_copy(deep=True)
        self._pending_orbit_advisory_base_route = base_route.model_copy(deep=True)
        self._pending_orbit_advisory_target = target
        self._orbit_battery_rtl_available = rtl_available
        self._orbit_battery_advisory_visible = True
        if estimate is not None:
            self._orbit_battery_advisory_text = (
                "Orbit may not be completed safely. "
                f"Need about {estimate.required_percent:.1f}% plus reserve; "
                f"margin {estimate.margin_percent:.1f}%."
            )
        else:
            self._orbit_battery_advisory_text = (
                "Orbit may not be completed safely with current battery."
            )
        self.orbitBatteryAdvisoryChanged.emit()

    def _stage_orbit_route(
        self,
        *,
        route: Route,
        base_route: Route,
        target: ConfirmedObject,
        active_targets: list[ConfirmedObject],
        source: str,
    ) -> None:
        if not route.waypoints:
            self._recover_from_failed_orbit_attempt(
                reason=f"empty_route:{source}",
                user_message="Orbit route unavailable",
            )
            return
        rejoin_wp: Waypoint | None = None
        last_wp = route.waypoints[-1]
        if base_route.waypoints:
            rejoin_idx = self._nearest_waypoint_index(last_wp.lat, last_wp.lon, base_route.waypoints)
            rejoin_wp = base_route.waypoints[rejoin_idx].model_copy(deep=True)

        orbit_segment, orbit_completion_wp = self._extract_orbit_segment(route, base_route)
        self._orbit_resume_min_index = self._extract_resume_waypoint_index(route, base_route)
        if self._orbit_resume_min_index is not None and base_route.waypoints:
            rejoin_wp = base_route.waypoints[self._orbit_resume_min_index].model_copy(deep=True)
        if not orbit_segment:
            orbit_segment = list(route.waypoints)
        orbit_route = Route(version=1, waypoints=list(orbit_segment), active_index=0)
        orbit_target = (float(target.lat), float(target.lon))
        if self._backend == "unreal" and self._unreal_link is not None:
            self._try_unreal_resume_before_route_send()
            if not self._send_unreal_route(
                orbit_route,
                orbit_target=orbit_target,
                orbit_target_alt=self._resolve_orbit_target_altitude_m(),
            ):
                self._recover_from_failed_orbit_attempt(
                    reason=f"route_send_failed:{source}",
                    user_message="Failed to send orbit to Unreal",
                )
                return

        self._reaction_target_id = None
        self._reaction_started_monotonic = 0.0
        self._clear_reaction_slowdown()
        self._orbit_active = True
        self._orbit_rejoin_wp = rejoin_wp
        self._orbit_route_end_wp = (
            orbit_completion_wp.model_copy(deep=True) if orbit_completion_wp is not None else None
        )
        self._orbit_started_monotonic = time.monotonic()
        self._orbit_min_complete_time_s = self._estimate_orbit_min_complete_time_s(orbit_segment)
        self._orbit_rejoin_close_hits = 0
        self._set_pending_orbit_targets(active_targets)
        self._activate_orbit_target_lock()
        self._apply_detector_pause_for_orbit()
        self._set_orbit_flow_state(OrbitFlowState.ORBIT_ACTIVE, reason=f"orbit_started:{source}")

        deps.debug_target = {"lat": orbit_target[0], "lon": orbit_target[1]}
        deps.debug_orbit_path = [(wp.lat, wp.lon) for wp in orbit_segment]
        self._active_path.set_orbit(deps.debug_orbit_path)
        self.map_bridge.render_map()
        self.toastRequested.emit("Orbit route staged")

    def _route_from_points(self, pts: list[tuple[float, float]]) -> Route | None:
        clean_pts = dedupe_path(pts)
        if len(clean_pts) < 2:
            return None
        alt = self._latest_telemetry.alt if self._latest_telemetry else 120.0
        waypoints = [Waypoint(lat=lat, lon=lon, alt=alt) for lat, lon in clean_pts]
        return Route(version=1, waypoints=waypoints, active_index=0)

    def _resolve_max_distance_m(self) -> float:
        max_range = getattr(self._energy_model, "max_range_m", None)
        if callable(max_range):
            try:
                resolved = float(max_range())
                if resolved > 0:
                    return resolved
            except Exception:
                pass
        max_distance_m = float(getattr(settings, "max_flight_distance_m", 15000.0) or 0.0)
        if max_distance_m > 0:
            return max_distance_m
        battery_wh = self._bridge_battery_wh or float(getattr(settings, "battery_wh", 4500.0) or 4500.0)
        cruise_speed = float(getattr(settings, "cruise_speed_mps", 12.0) or 12.0)
        power_w = float(getattr(settings, "power_cruise_w", 45.0) or 45.0)
        if power_w <= 0 or cruise_speed <= 0:
            return float(getattr(settings, "max_flight_distance_m", 15000.0) or 0.0)
        return (battery_wh / power_w) * cruise_speed * 3600.0

    def _resolve_base_location(self, route: Route, telemetry: TelemetrySample) -> WorldCoord | None:
        resolved = resolve_base_location(settings, route, telemetry)
        if resolved is not None:
            return resolved
        return WorldCoord(lat=telemetry.lat, lon=telemetry.lon)

    def _resolve_uav_id(self, sample: TelemetrySample) -> str:
        source = getattr(sample, "source", None)
        if source:
            return str(source)
        return str(getattr(settings, "uav_id", None) or "uav")

    def _build_remaining_active_route(self, sample: TelemetrySample) -> Route | None:
        path = self._active_path.get_active_path()
        if len(path) < 2:
            return None
        remaining_points = list(path)
        if self._active_path.mode != ActivePathMode.ORBIT:
            nearest_idx = self._nearest_waypoint_index(
                float(sample.lat),
                float(sample.lon),
                [Waypoint(lat=lat, lon=lon, alt=float(sample.alt)) for lat, lon in path],
            )
            remaining_points = path[nearest_idx:]
        waypoints = [Waypoint(lat=float(sample.lat), lon=float(sample.lon), alt=float(sample.alt))]
        for lat, lon in remaining_points:
            waypoints.append(Waypoint(lat=lat, lon=lon, alt=float(sample.alt)))
        return Route(version=1, waypoints=waypoints, active_index=0 if waypoints else None)

    def _send_rtl_route(self) -> bool:
        route = self._build_rtl_route()
        if route is None:
            return False
        if self._backend == "unreal" and self._unreal_link is not None:
            if self._orbit_active:
                self._unreal_link.send_command("ORBIT_STOP")
            if not self._send_unreal_route(route):
                return False
        self._rtl_route_sent = True
        deps.rtl_path = [(wp.lat, wp.lon) for wp in route.waypoints]
        deps.debug_orbit_path = None
        if hasattr(self._active_path, "set_rtl"):
            self._active_path.set_rtl(deps.rtl_path)
        self.map_bridge.render_map()
        return True

    def _initiate_rtl(self, *, reason: str, user_message: str) -> bool:
        self._clear_route_battery_advisory()
        self._clear_orbit_battery_advisory()
        self._pending_orbit_queue = []
        self._pending_orbit_ids.clear()
        if self._orbit_active:
            self._clear_manual_orbit_state()
        deps.debug_orbit_path = None
        deps.debug_target = None
        deps.selected_object_id = None
        self._objects_store.set_selected(None)
        self._on_objects_changed()
        if not self._send_rtl_route():
            self.toastRequested.emit("Failed to send RTL route")
            return False
        trigger_rtl = getattr(self._mission, "trigger_rtl", None)
        if callable(trigger_rtl):
            trigger_rtl(reason)
        self.flightControlsChanged.emit()
        self.toastRequested.emit(user_message)
        return True

    def _send_unreal_despawn(self) -> bool:
        if self._backend != "unreal" or self._unreal_link is None:
            return False
        return self._unreal_link.send_command("DESPAWN")

    def _build_rtl_route(self, *, silent: bool = False) -> Route | None:
        if self._latest_telemetry is None:
            if not silent:
                self.toastRequested.emit("No telemetry available for RTL")
            return None
        base_path = list(self._mission.confirmed_plan or self._plan_vm.get_path() or [])
        base_route = self._route_from_points(base_path) or Route(version=1, waypoints=[], active_index=None)
        base = self._resolve_base_location(base_route, self._latest_telemetry)
        if base is None:
            if not silent:
                self.toastRequested.emit("Home location unavailable")
            return None
        alt = self._latest_telemetry.alt
        waypoints = [
            Waypoint(lat=self._latest_telemetry.lat, lon=self._latest_telemetry.lon, alt=alt),
            Waypoint(lat=base.lat, lon=base.lon, alt=alt),
        ]
        route = Route(version=1, waypoints=waypoints, active_index=0)
        try:
            estimate = self._energy_model.estimate_route_feasibility(self._latest_telemetry, route, base)
            if not estimate.can_complete:
                self._emit_warning(
                    key="battery_rtl_insufficient",
                    message="Battery low for RTL; attempting best-effort return",
                    severity="error",
                    cooldown_s=10,
                )
        except Exception:
            pass
        return route

    def _can_start_flight(self) -> bool:
        if self._mission_state not in (MissionState.PREFLIGHT, MissionState.READY):
            return False
        if not self._mission.plan_confirmed:
            return False
        if self._allow_unsafe_start:
            return True
        return self._link_monitor.is_link_ok() and self._camera_monitor.is_camera_ok()

    def _can_orbit(self) -> bool:
        return self._current_action_policy().can_orbit

    @staticmethod
    def _format_duration(duration_s: float) -> str:
        mins, secs = divmod(int(duration_s), 60)
        if mins:
            return f"{mins}m {secs}s"
        return f"{secs}s"

    @staticmethod
    def _load_bridge_profiles() -> dict[str, dict[str, object]]:
        return {
            "mavic3": {"label": "DJI Mavic 3 (77 Wh)", "battery_wh": 77.0, "speed_mps": 15.0},
            "mini4": {"label": "DJI Mini 4 Pro (29 Wh)", "battery_wh": 29.0, "speed_mps": 12.0},
            "matrice30": {"label": "DJI Matrice 30T (263 Wh)", "battery_wh": 263.0, "speed_mps": 16.0},
            "autel_evo": {"label": "Autel EVO II (82 Wh)", "battery_wh": 82.0, "speed_mps": 16.0},
            "skydio_x2": {"label": "Skydio X2 (49 Wh)", "battery_wh": 49.0, "speed_mps": 13.0},
        }

    def _update_stats(self) -> None:
        now = time.perf_counter()
        if self._last_detection_ts is not None and (now - self._last_detection_ts) > 2.0:
            self._detection_conf = 0.0
        camera_age = self._camera_monitor.age_s()
        if camera_age is None or camera_age > 2.0:
            self._fps = 0.0
        else:
            try:
                self._fps = max(0.0, float(self._camera_monitor.fps))
            except Exception:
                self._fps = 0.0
        bus_alive = self._last_detection_ts is not None and (now - self._last_detection_ts) < 2.0
        if bus_alive != self._bus_alive:
            self._bus_alive = bus_alive
        self.statsChanged.emit()

    def _trim_logs(self) -> None:
        if len(self._logs) > 400:
            self._logs = self._logs[-400:]

    def _load_log_history(self, limit: int = 200) -> None:
        try:
            if not self._log_history_path.exists():
                return
            lines = self._log_history_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if limit:
                lines = lines[-limit:]
            self._logs.extend(lines)
            self._trim_logs()
            if lines:
                self.logsChanged.emit()
        except Exception:  # noqa: BLE001
            _log.exception("Failed to preload log history")

    # ---------- detection logging helpers ---------- #
    def _log_detection_event(
        self, detections: list[object], best_conf: float, bbox: tuple[int, int, int, int] | None
    ) -> None:
        if not detections:
            return
        now = time.monotonic()
        classes = [
            int(getattr(d, "class_id", getattr(d, "cls", -1)))
            for d in detections
            if hasattr(d, "class_id") or hasattr(d, "cls")
        ]
        class_sig = tuple(classes)
        sig = (len(detections), class_sig, bbox)
        log_now = True
        if self._last_detection_log_ts is not None and self._last_detection_sig == sig:
            if (now - self._last_detection_log_ts) < 2.0:
                log_now = False
        if log_now:
            self._last_detection_log_ts = now
            self._last_detection_sig = sig
        payload = make_raw_detection_log_entry(
            timestamp=utc_iso_z(),
            count=len(detections),
            best_confidence=round(float(best_conf), 4),
            best_bbox=list(bbox) if bbox else None,
            classes=classes,
            detections=[self._serialize_detection(d) for d in detections],
            telemetry=self._latest_telemetry,
        )
        if log_now:
            _log.warning(
                "Detection | count=%d best=%.2f classes=%s bbox=%s",
                len(detections),
                best_conf,
                classes or "n/a",
                bbox,
            )
        self._write_detection_json(payload)

    def _serialize_detection(self, det: object) -> dict[str, object]:
        bbox = getattr(det, "bbox", None)
        if bbox is None and all(hasattr(det, k) for k in ("x1", "y1", "x2", "y2")):
            bbox = (
                getattr(det, "x1"),
                getattr(det, "y1"),
                getattr(det, "x2"),
                getattr(det, "y2"),
            )
        ts = getattr(det, "timestamp", None)
        ts_iso = None
        if hasattr(ts, "isoformat"):
            try:
                ts_iso = ts.isoformat()
            except Exception:  # noqa: BLE001
                ts_iso = None
        return {
            "class_id": getattr(det, "class_id", getattr(det, "cls", None)),
            "confidence": float(getattr(det, "confidence", getattr(det, "score", 0.0))),
            "bbox": list(bbox) if bbox else None,
            "camera_id": getattr(det, "camera_id", None),
            "timestamp": ts_iso,
        }

    def _write_detection_json(self, payload: dict[str, object]) -> None:
        try:
            self._det_json_path.parent.mkdir(parents=True, exist_ok=True)
            with self._det_json_path.open("a", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
                fh.write("\n")
        except Exception:  # noqa: BLE001
            _log.exception("Failed to write detection JSON")

    def _write_confirmed_detection_json(self, payload: dict[str, object]) -> None:
        try:
            entry = make_confirmed_detection_log_entry(payload, telemetry=self._latest_telemetry)
            self._confirmed_det_json_path.parent.mkdir(parents=True, exist_ok=True)
            with self._confirmed_det_json_path.open("a", encoding="utf-8") as fh:
                json.dump(entry, fh, ensure_ascii=False, default=str)
                fh.write("\n")
        except Exception:  # noqa: BLE001
            _log.exception("Failed to write confirmed detection JSON")


# --------------------------------------------------------------------------- #
#                               MainWindow facade
# --------------------------------------------------------------------------- #
class MainWindow(QObject):
    """QML-driven UI facade that keeps the existing lifecycle wiring."""

    def __init__(self, *, qml_file: str = "main.qml") -> None:
        super().__init__()
        self.det_vm = DetectorVM()
        self.plan_vm = PlannerVM()
        self._backend = resolve_driver_type(settings)
        self._use_unreal = self._backend == "unreal"
        self._unreal_detection_source = str(
            getattr(settings, "unreal_detection_source", "local_yolo") or "local_yolo"
        ).lower()
        if self._unreal_detection_source not in ("backend", "local_yolo", "both"):
            self._unreal_detection_source = "local_yolo"
        self._unreal_local_yolo_enabled = self._use_unreal and self._unreal_detection_source in (
            "local_yolo",
            "both",
        )
        self._unreal_forwarded_object_seen_ts: dict[str, float] = {}
        self._frame_q = deps.frame_queue
        self._cam_fps = getattr(settings, "camera_fps", 30)
        self._camera_index = 0
        self._camera_candidates: list[int] = [] if self._use_unreal else self._probe_cameras()

        # Optional camera/detector threads
        if not self._use_unreal:
            try:
                self.cam_thr = deps.get_camera()
                self.det_thr = deps.get_detector()
                self._have_camera = True
                self._camera_index = getattr(self.cam_thr, "index", 0)
                if self._camera_index not in self._camera_candidates:
                    self._camera_candidates.insert(0, self._camera_index)
                self.cam_thr.frame.connect(self._on_frame)
                self.cam_thr.error.connect(lambda msg: self.app.toastRequested.emit(msg))
            except RuntimeError:
                self.cam_thr = None
                self.det_thr = None
                self._have_camera = False
        else:
            self.cam_thr = None
            if self._unreal_local_yolo_enabled:
                try:
                    self.det_thr = deps.get_detector()
                    _log.info(
                        "Unreal mode local YOLO detector thread created (source=%s)",
                        self._unreal_detection_source,
                    )
                except RuntimeError:
                    self.det_thr = None
                    _log.warning(
                        "Unreal local YOLO requested but detect_factory is not configured (source=%s)",
                        self._unreal_detection_source,
                    )
            else:
                self.det_thr = None
            self._have_camera = False

        # Register components before QML triggers APP_START so start_all() sees them.
        deps.get_lifecycle().register(self.cam_thr, self.det_thr)

        # Bridges exposed to QML
        self._video_provider = VideoFrameProvider()
        self._video_bridge = VideoBridge(self._video_provider)
        self._map_bridge = MapBridge(self.plan_vm)
        self.app = AppController(
            self.det_vm,
            self._map_bridge,
            self._video_bridge,
            camera_available=self._have_camera,
            camera_switcher=self._cycle_camera,
        )
        self._debug_sim: DebugSimulationService | None = None
        self._unreal_link: UnrealLinkService | None = None
        if not self._use_unreal:
            self._debug_sim = DebugSimulationService(
                route_provider=self.app.get_active_path_for_sim,
                telemetry_callback=self.app.on_telemetry,
                frame_callback=self._on_frame,
            )
            self.app.attach_debug_sim(self._debug_sim)
        else:
            base_url = str(getattr(settings, "unreal_base_url", "http://127.0.0.1:9000"))
            uav_id = str(getattr(settings, "uav_id", None) or "sim")
            camera_mode = str(
                getattr(settings, "unreal_video_mode", "h264_stream") or "h264_stream"
            ).lower()
            video_endpoint = str(getattr(settings, "unreal_video_endpoint", "/sim/v1/video.ts"))
            video_fps = float(getattr(settings, "unreal_video_target_fps", 15.0) or 15.0)
            video_reconnect_s = float(
                getattr(settings, "unreal_video_reconnect_s", 1.0) or 1.0
            )
            telemetry_hz = float(getattr(settings, "unreal_telemetry_hz", 6.0) or 6.0)
            detections_hz = float(getattr(settings, "unreal_detections_hz", 1.0) or 1.0)
            camera_hz = float(getattr(settings, "unreal_camera_hz", 8.0) or 8.0)
            if self._unreal_detection_source == "local_yolo":
                detections_hz = 0.0
            unreal_detections_cb = (
                self._on_unreal_detections if self._unreal_detection_source in ("backend", "both") else None
            )
            self._unreal_link = UnrealLinkService(
                base_url=base_url,
                uav_id=uav_id,
                telemetry_hz=telemetry_hz,
                detections_hz=detections_hz,
                camera_hz=camera_hz,
                camera_mode=camera_mode,
                video_endpoint=video_endpoint,
                video_target_fps=video_fps,
                video_reconnect_s=video_reconnect_s,
                on_telemetry=self.app.on_telemetry,
                on_camera_frame=self.app.on_camera_image,
                on_detections=unreal_detections_cb,
                on_link_status=self.app.on_unreal_link_status,
                on_camera_status=self.app.on_unreal_camera_status,
                on_camera_info=self.app.on_unreal_camera_info,
                on_map_ready=self.app.set_unreal_static_map,
                on_warning=lambda msg: self.app.toastRequested.emit(msg),
            )
            self.app.attach_unreal_link(self._unreal_link)
            self._unreal_link.start()

        # QML engine + context
        self._engine = QQmlApplicationEngine()
        self._engine.addImageProvider("video", self._video_provider)
        self._engine.rootContext().setContextProperty("app", self.app)

        qml_path = Path(__file__).resolve().parents[1] / "qml" / qml_file
        _log.info("Loading QML UI from %s", qml_path)
        self._engine.load(QUrl.fromLocalFile(str(qml_path)))
        if not self._engine.rootObjects():
            raise RuntimeError("Failed to load QML UI")
        self._window = self._engine.rootObjects()[0]

    # ---------- facade API ---------- #
    def show(self) -> None:
        if self._window is not None:
            self._window.show()

    # ---------- slots from threads ---------- #
    def _on_frame(self, frame: NDArray[np.uint8]) -> None:
        if self._frame_q is not None:
            try:
                self._frame_q.put_nowait(frame)
            except Exception:
                pass
        self.app.on_frame(frame)

    def _on_unreal_detections(self, batch: object, objects: list[dict[str, object]]) -> None:
        """
        Unreal detections do not include image-space bounding boxes.
        They must NOT go through the local detector pipeline (Event.DETECTION),
        otherwise QML/DetectorVM may crash on bbox=None.
        """
        # Do NOT: bus.emit(Event.DETECTION, batch)

        # Still push confirmed objects (map targets, list, orbit, etc.)
        if self._unreal_detection_source == "local_yolo":
            return
        self.app.process_backend_detections(objects)

    def stop_services(self) -> None:
        if self._unreal_link is not None:
            try:
                self._unreal_link.stop()
            except Exception:
                pass

    # ---------- camera helpers ---------- #
    def _probe_cameras(self, limit: int = 5) -> list[int]:
        found: list[int] = []
        for idx in range(limit):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                found.append(idx)
            cap.release()
        return found

    def _camera_available(self, index: int) -> bool:
        cap = cv2.VideoCapture(index)
        ok = cap.isOpened()
        cap.release()
        return ok

    def _cycle_camera(self) -> bool:
        # Refresh candidate list each time to catch hotplug devices.
        self._camera_candidates = self._probe_cameras()
        if len(self._camera_candidates) <= 1:
            self.app.toastRequested.emit("No alternate camera found")
            return False

        current = (
            self._camera_index
            if self._camera_index in self._camera_candidates
            else self._camera_candidates[0]
        )
        next_idx = self._camera_candidates[
            (self._camera_candidates.index(current) + 1) % len(self._camera_candidates)
        ]
        if next_idx == current and len(self._camera_candidates) == 1:
            self.app.toastRequested.emit("No alternate camera found")
            return False

        return self._switch_camera(next_idx)

    def _switch_camera(self, index: int) -> bool:
        if not self._camera_available(index):
            self._have_camera = False
            self.app.set_camera_available(False)
            self.app.toastRequested.emit(f"Camera #{index} is not available")
            return False

        old_cam = getattr(self, "cam_thr", None)
        if old_cam is not None:
            try:
                old_cam.stop()
                old_cam.join(timeout=2.0)
            except Exception:  # noqa: BLE001
                pass

        try:
            new_cam = CameraThread(index=index, fps=self._cam_fps, out_queue=self._frame_q)
        except Exception as exc:  # noqa: BLE001
            self.cam_thr = None
            self._have_camera = False
            self.app.set_camera_available(False)
            self.app.toastRequested.emit(f"Failed to init camera #{index}: {exc}")
            return False

        new_cam.frame.connect(self._on_frame)
        new_cam.error.connect(lambda msg: self.app.toastRequested.emit(msg))
        deps.get_lifecycle().register(new_cam)
        new_cam.start()

        self.cam_thr = new_cam
        self._camera_index = index
        self._have_camera = True
        self.app.set_camera_available(True)
        self.app.toastRequested.emit(f"Camera switched to #{index}")
        return True


