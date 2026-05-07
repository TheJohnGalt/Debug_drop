from __future__ import annotations

import logging
import time
# костыль, так как jetson не работает с python 3.11
try:
    from enum import StrEnum
except ImportError:
    from enum import Enum
    class StrEnum(str, Enum):
        pass

from fire_uav.services.bus import Event, bus

log = logging.getLogger(__name__)


class CameraStatus(StrEnum):
    READY = "READY"
    NOT_READY = "NOT_READY"


class CameraMonitor:
    def __init__(self, *, stale_after_s: float = 2.0) -> None:
        self._stale_after = stale_after_s
        self._last_frame_ts: float | None = None
        self._seen_frame = False
        self._status = CameraStatus.NOT_READY
        self._fps = 0.0

    @property
    def status(self) -> CameraStatus:
        return self._status

    @property
    def fps(self) -> float:
        return self._fps

    def on_frame(self) -> None:
        now = time.monotonic()
        if self._last_frame_ts is not None:
            dt = max(1e-3, now - self._last_frame_ts)
            inst = 1.0 / dt
            self._fps = 0.8 * self._fps + 0.2 * inst if self._fps else inst
        self._last_frame_ts = now
        self._seen_frame = True
        self._update_status()

    def on_status(
        self,
        *,
        available: bool,
        streaming: bool,
        fps: float | None = None,
        last_frame_age_s: float | None = None,
    ) -> None:
        now = time.monotonic()
        if available and streaming:
            if last_frame_age_s is not None:
                self._last_frame_ts = max(0.0, now - float(last_frame_age_s))
            else:
                self._last_frame_ts = now
            self._seen_frame = True
        else:
            self._last_frame_ts = None
            self._seen_frame = False
        if fps is not None:
            try:
                self._fps = float(fps)
            except (TypeError, ValueError):
                pass
        self._update_status()

    def check(self) -> None:
        self._update_status()

    def is_camera_ok(self) -> bool:
        return self._status == CameraStatus.READY

    def age_s(self) -> float | None:
        if self._last_frame_ts is None:
            return None
        return max(0.0, time.monotonic() - self._last_frame_ts)

    # ------------------------------------------------------------------ #
    def _update_status(self) -> None:
        age = self.age_s()
        if age is None or age > self._stale_after or not self._seen_frame:
            new_status = CameraStatus.NOT_READY
        else:
            new_status = CameraStatus.READY

        if new_status == self._status:
            return
        self._status = new_status
        bus.emit(
            Event.CAMERA_STATUS_CHANGED,
            {"status": new_status.value, "age_s": age, "fps": self._fps},
        )
        log.info("Camera status -> %s (age=%s, fps=%.1f)", new_status.value, age, self._fps)


__all__ = ["CameraMonitor", "CameraStatus"]
