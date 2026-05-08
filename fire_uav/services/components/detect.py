# mypy: ignore-errors
# fire_uav/services/components/detect.py

from __future__ import annotations

import logging
import queue
import time
from datetime import datetime
from typing import Final

import numpy as np
from numpy.typing import NDArray

import fire_uav.infrastructure.providers as deps
from fire_uav.config.settings import settings
from fire_uav.domain.detect.detection import DetectionEngine
from fire_uav.module_core.schema import DetectionsBatch
from fire_uav.services.bus import Event, bus
from fire_uav.services.components.base import ManagedComponent, State
from fire_uav.services.metrics import detect_latency, fps_gauge, queue_size

LOG: Final = logging.getLogger("detect")

Frame = NDArray[np.uint8]
FrameItem = Frame | dict[str, object]
_SLEEP = 0.05
_STAT_EVERY = 1.0  # seconds


class DetectThread(ManagedComponent):
    """YOLO detector + latency and queue-size instrumentation."""

    def __init__(
        self,
        in_q: queue.Queue[FrameItem | None],
        out_q: queue.Queue[DetectionsBatch],
    ) -> None:
        super().__init__(name="DetectThread")
        self._in_q = in_q
        self._out_q = out_q
        self._engine = None
        self._disabled_reason: str | None = None
        try:
            self._engine = DetectionEngine(
                wanted_classes=settings.yolo_classes,
                conf_threshold=settings.yolo_conf,
                iou_threshold=settings.yolo_iou,
            )
        except Exception as exc:  # noqa: BLE001
            self._disabled_reason = str(exc)
        self._stat_ts = time.perf_counter()
        self._last_frame_ts = time.perf_counter()
        self._paused = False
        bus.subscribe(Event.DETECTOR_START, lambda *_: self.resume_processing())
        bus.subscribe(Event.DETECTOR_STOP, lambda *_: self.pause_processing())

    def pause_processing(self) -> None:
        self._paused = True

    def resume_processing(self) -> None:
        self._paused = False

    def loop(self) -> None:
        if self._engine is None:
            reason = self._disabled_reason or "unknown error"
            LOG.warning("Detection disabled: %s", reason)
            return
        LOG.info("Detector thread started")
        while self.state is State.RUNNING:
            try:
                frame_item = self._in_q.get(timeout=_SLEEP)
            except queue.Empty:
                frame_item = None

            if frame_item is None:
                continue
            if self._paused:
                continue

            frame = frame_item
            camera_id = "cam0"
            captured_at: datetime | None = None
            if isinstance(frame_item, dict):
                frame = frame_item.get("frame")
                raw_camera_id = frame_item.get("camera_id")
                raw_timestamp = frame_item.get("timestamp")
                if isinstance(raw_camera_id, str) and raw_camera_id.strip():
                    camera_id = raw_camera_id.strip()
                if isinstance(raw_timestamp, datetime):
                    captured_at = raw_timestamp
            if frame is None or not isinstance(frame, np.ndarray):
                continue

            # track queue depth
            queue_size.set(self._in_q.qsize())

            # Quick FPS estimate from capture cadence
            now = time.perf_counter()
            dt = now - self._last_frame_ts
            if dt > 0:
                current_fps = 1.0 / dt
                try:
                    prev = fps_gauge._value.get()  # type: ignore[attr-defined]
                except Exception:
                    prev = current_fps
                fps_gauge.set(0.8 * prev + 0.2 * current_fps)
            self._last_frame_ts = now

            # measure inference latency
            with detect_latency.time():
                if hasattr(self._engine, "detect"):
                    batch = self._engine.detect(frame)
                else:
                    batch = self._engine.infer(frame, camera_id=camera_id, return_batch=True)
            if captured_at is not None and hasattr(batch, "frame"):
                try:
                    batch.frame.timestamp = captured_at
                except Exception:
                    pass

            deps.last_detection = batch
            dets = getattr(batch, "detections", [])
            count = len(dets)
            best = 0.0
            if count:
                best = max(
                    (getattr(d, "confidence", getattr(d, "score", 0.0)) for d in dets),
                    default=0.0,
                )
                bbox = getattr(dets[0], "bbox", None)
                if bbox is None and all(hasattr(dets[0], k) for k in ("x1", "y1", "x2", "y2")):
                    bbox = (
                        getattr(dets[0], "x1"),
                        getattr(dets[0], "y1"),
                        getattr(dets[0], "x2"),
                        getattr(dets[0], "y2"),
                    )
                LOG.debug(
                    "Detections: count=%d best=%.2f bbox=%s",
                    count,
                    best,
                    bbox,
                )

            try:
                self._out_q.put_nowait(batch)
            except queue.Full:
                LOG.debug("Output queue full - dropping detection")

            bus.emit(Event.DETECTION, batch)


            # periodic debug
            now = time.perf_counter()
            if now - self._stat_ts >= _STAT_EVERY:
                LOG.debug(
                    "Detector heartbeat: frames_q=%d dets_q=%d last_batch=%d best=%.2f",
                    self._in_q.qsize(),
                    self._out_q.qsize(),
                    count,
                    best,
                )
                self._stat_ts = now

        LOG.info("Detector thread stopped")

    def stop(self) -> None:
        self.state = State.STOPPED
        try:
            self._in_q.put_nowait(None)
        except Exception:
            pass
