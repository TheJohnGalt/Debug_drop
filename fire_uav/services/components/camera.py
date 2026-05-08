# mypy: ignore-errors
"""Headless camera worker that captures frames and pushes them to a queue."""

from __future__ import annotations

import logging
import time
from typing import Callable, Final

import cv2
import numpy as np

from fire_uav.config.settings import settings
from fire_uav.services.components.base import ManagedComponent, State
from fire_uav.services.metrics import fps_gauge

LOG: Final = logging.getLogger("camera")

FrameCallback = Callable[[np.ndarray], None]
ErrorCallback = Callable[[str], None]


class CameraThread(ManagedComponent):
    """Managed camera capture worker without Qt dependencies."""

    _LOG_EVERY: Final[float] = 5.0

    def __init__(
        self,
        index: int | str = 0,
        fps: int = 30,
        out_queue=None,
        *,
        on_frame: FrameCallback | None = None,
        on_error: ErrorCallback | None = None,
    ) -> None:
        super().__init__(name="CameraThread")
        self.index = index
        self.fps = fps
        self._q = out_queue
        self._on_frame = on_frame
        self._on_error = on_error
        self._sleep = 1 / fps if fps else 0
        self._stat_ts = time.perf_counter()
        self._stat_frames = 0

    def _emit_frame(self, frame: np.ndarray) -> None:
        if self._on_frame is not None:
            try:
                self._on_frame(frame)
            except Exception:  # noqa: BLE001
                pass
        if self._q is not None:
            try:
                self._q.put_nowait(frame)
            except Exception:  # noqa: BLE001
                pass

    def _emit_error(self, message: str) -> None:
        if self._on_error is not None:
            try:
                self._on_error(message)
            except Exception:  # noqa: BLE001
                pass
        else:
            LOG.warning(message)



    def loop(self) -> None:

        if settings.gstreamer_pipline is not None:
            cap = cv2.VideoCapture(settings.gstreamer_pipline, cv2.CAP_GSTREAMER)
        else:
            cap = cv2.VideoCapture(self.index)
        
        if not cap.isOpened():
            self._emit_error("Camera not opened")
            return

        while not self._stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                self._emit_error("Failed to read frame")
                break

            self._emit_frame(frame)

            fps_gauge.inc()
            self._stat_frames += 1
            now = time.perf_counter()
            if now - self._stat_ts >= self._LOG_EVERY:
                fps = self._stat_frames / (now - self._stat_ts)
                LOG.info("Camera FPS: %.1f", fps)
                self._stat_ts, self._stat_frames = now, 0

            if self._sleep:
                time.sleep(self._sleep)

        cap.release()
        self.state = State.STOPPED

    def stop(self) -> None:
        super().stop()
        self.state = State.STOPPED


