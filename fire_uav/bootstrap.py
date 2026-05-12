# mypy: ignore-errors
from __future__ import annotations

import logging
from queue import Queue

import cv2

import fire_uav.infrastructure.providers as deps
from fire_uav.config import settings
from fire_uav.module_core.drivers.registry import resolve_driver_type
from fire_uav.services.bus import Event, bus
from fire_uav.services.components.camera import CameraThread
from fire_uav.services.components.detect import DetectThread
from fire_uav.services.lifecycle.manager import LifecycleManager

_log = logging.getLogger(__name__)


def _camera_or_fallback_frame_available(index: int | str = 0) -> bool:
    cap = cv2.VideoCapture(index)
    ok = cap.isOpened()
    cap.release()

    _fallback_frame = cv2.imread("fire_uav/utils/debug/debug_img/fallback_frame.png")
    if _fallback_frame is None:
        ok = False
    else:
        ok = True

    return ok


def init_module_core(*, fps: int = 30) -> None:
    if deps.lifecycle_manager is not None:
        return

    deps.frame_queue = Queue(maxsize=5)
    deps.dets_queue = Queue(maxsize=5)

    if _camera_or_fallback_frame_available():
        deps.camera_factory = lambda: CameraThread(
            index=0,
            fps=fps,
            out_queue=deps.frame_queue,
        )
        deps.detect_factory = lambda: DetectThread(
            in_q=deps.frame_queue,
            out_q=deps.dets_queue,
        )
    else:
        deps.camera_factory = None
        deps.detect_factory = None

        _log.warning("Camera not found — GUI will start without live feed")

    print(deps.camera_factory, deps.camera_factory)

    deps.lifecycle_manager = LifecycleManager()
    detect_thread = deps.get_detector()
    camera_thread = deps.get_camera()
    deps.lifecycle_manager.register(detect_thread, camera_thread)

    bus.subscribe(Event.APP_START, lambda *_: deps.get_lifecycle().start_all())
    bus.subscribe(Event.APP_STOP, lambda *_: deps.get_lifecycle().stop_all())


def init_ground_core(*, fps: int = 30) -> None:
    if deps.lifecycle_manager is not None:
        return

    from fire_uav.ground_app.gui.camera_qt import CameraThread as QtCameraThread

    deps.frame_queue = Queue(maxsize=5)
    deps.dets_queue = Queue(maxsize=5)

    if resolve_driver_type(settings) == "unreal":
        deps.camera_factory = None
        detection_source = str(
            getattr(settings, "unreal_detection_source", "local_yolo") or "local_yolo"
        ).lower()
        if detection_source in ("local_yolo", "both"):
            deps.detect_factory = lambda: DetectThread(
                in_q=deps.frame_queue,
                out_q=deps.dets_queue,
            )
            _log.info(
                "Unreal backend active - local YOLO detector enabled (source=%s)",
                detection_source,
            )
        else:
            deps.detect_factory = None
            _log.info("Unreal backend active - skipping local detector (source=%s)", detection_source)
        deps.lifecycle_manager = LifecycleManager()
        bus.subscribe(Event.APP_START, lambda *_: deps.get_lifecycle().start_all())
        bus.subscribe(Event.APP_STOP, lambda *_: deps.get_lifecycle().stop_all())
        return

    if _camera_available():
        deps.camera_factory = lambda: QtCameraThread(
            index=0,
            fps=fps,
            out_queue=deps.frame_queue,
        )
        deps.detect_factory = lambda: DetectThread(
            in_q=deps.frame_queue,
            out_q=deps.dets_queue,
        )
    else:
        deps.camera_factory = None
        deps.detect_factory = None
        _log.warning("Camera not found - GUI will start without live feed")

    deps.lifecycle_manager = LifecycleManager()
    bus.subscribe(Event.APP_START, lambda *_: deps.get_lifecycle().start_all())
    bus.subscribe(Event.APP_STOP, lambda *_: deps.get_lifecycle().stop_all())


def init_core(*, fps: int = 30) -> None:
    """Backward-compatible alias for the headless module init."""
    init_module_core(fps=fps)
