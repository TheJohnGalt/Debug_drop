from __future__ import annotations

from queue import Queue
from typing import TYPE_CHECKING, Any, Callable, Optional

from fire_uav.services.lifecycle.manager import LifecycleManager

if TYPE_CHECKING:  # imports only for type checking to avoid circular deps
    from fire_uav.services.components.camera import CameraThread
    from fire_uav.services.components.detect import DetectThread
    from fire_uav.services.components.relay_component import RelayThread

# ────────── очереди ────────── #
frame_queue: Optional[Queue] = None  # кадры camera → detector
dets_queue: Optional[Queue] = None  # детекции detector → GUI/API

# ────────── фабрики компонентов ────────── #
camera_factory: Callable[[], "CameraThread"] | None = None
detect_factory: Callable[[], "DetectThread"] | None = None
relay_factory: Callable[[], "RelayThread"] | None = None
plan_widget_factory: Callable[..., object] | None = None

# ────────── lifecycle ────────── #
lifecycle_manager: "LifecycleManager | None" = None

# ────────── API-shared data ────────── #
plan_data: Any | None = None  # хранит JSON-план (List[waypoints])
last_detection: Any | None = None  # последняя пачка детекций

# ────────── GUI/debug-only shared state ────────── #
# Используется только визуальными отладочными функциями (маршрут/орбита) и
# не затрагивает пайплайн детекции.
debug_target: dict | None = None  # {"lat": float, "lon": float}
debug_orbit_path: list[tuple[float, float]] | None = None  # [(lat, lon), ...]
debug_orbit_preview_paths: list[list[tuple[float, float]]] = []  # preview chain per orbit target
debug_flight_enabled: bool = False  # флаг режима "flight debug" в GUI
debug_flight_progress: float = 0.0  # debug progress 0..1
debug_flight_completed: bool = False  # debug flight reached end of route
debug_map_manual_refresh: bool = False  # вручную обновляем позицию дрона на карте
rtl_path: list[tuple[float, float]] | None = None  # RTL path override for map display
confirmed_objects: list[dict] = []  # cached confirmed objects for map overlays
selected_object_id: str | None = None  # selected confirmed object for orbit actions
latest_telemetry: Any | None = None  # last telemetry sample for map/status
route_stats: dict | None = None  # cached route battery stats for map overlays
home_location: dict | None = None  # {"lat": float, "lon": float} for RTL/base
mission_state: str | None = None  # cached mission state for UI/map


# ────────── helpers ────────── #
def get_camera() -> "CameraThread":
    if camera_factory is None:
        raise RuntimeError("camera_factory not configured")
    return camera_factory()


def get_detector() -> "DetectThread":
    if detect_factory is None:
        raise RuntimeError("detect_factory not configured")
    return detect_factory()

def get_realy() -> "RelayThread":
    if relay_factory is None:
        print("relay_factory not configured")
        raise RuntimeError("relay_factory not configured")
    return relay_factory()

def get_lifecycle() -> "LifecycleManager":
    if lifecycle_manager is None:
        raise RuntimeError("lifecycle_manager not configured")
    return lifecycle_manager
