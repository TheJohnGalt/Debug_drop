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

from fire_uav.module_core.schema import TelemetrySample
from fire_uav.services.bus import Event, bus

log = logging.getLogger(__name__)


class LinkStatus(StrEnum):
    CONNECTED = "CONNECTED"
    DEGRADED = "DEGRADED"
    DISCONNECTED = "DISCONNECTED"


class LinkMonitor:
    def __init__(
        self,
        *,
        connected_timeout_s: float = 2.0,
        degraded_timeout_s: float = 5.0,
    ) -> None:
        self._connected_timeout = connected_timeout_s
        self._degraded_timeout = degraded_timeout_s
        self._last_seen: float | None = None
        self._status = LinkStatus.DISCONNECTED

    @property
    def status(self) -> LinkStatus:
        return self._status

    def on_telemetry(self, sample: TelemetrySample) -> None:
        self._last_seen = time.monotonic()
        self._update_status()

    def check(self) -> None:
        self._update_status()

    def is_link_ok(self) -> bool:
        return self._status == LinkStatus.CONNECTED

    def age_s(self) -> float | None:
        if self._last_seen is None:
            return None
        return max(0.0, time.monotonic() - self._last_seen)

    # ------------------------------------------------------------------ #
    def _update_status(self) -> None:
        age = self.age_s()
        if age is None:
            new_status = LinkStatus.DISCONNECTED
        elif age <= self._connected_timeout:
            new_status = LinkStatus.CONNECTED
        elif age <= self._degraded_timeout:
            new_status = LinkStatus.DEGRADED
        else:
            new_status = LinkStatus.DISCONNECTED

        if new_status == self._status:
            return
        self._status = new_status
        bus.emit(
            Event.UAV_LINK_STATUS_CHANGED,
            {"status": new_status.value, "age_s": age},
        )
        log.info("UAV link status -> %s (age=%s)", new_status.value, age)


__all__ = ["LinkMonitor", "LinkStatus"]
