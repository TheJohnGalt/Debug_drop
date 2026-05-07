from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
# костыль, так как jetson не работает с python 3.11
try:
    from enum import StrEnum
except ImportError:
    from enum import Enum
    class StrEnum(str, Enum):
        pass
from typing import Iterable, Sequence

from fire_uav.services.bus import Event, bus
from fire_uav.utils.time import utc_iso_z, utc_now

log = logging.getLogger(__name__)


class MissionState(StrEnum):
    PREFLIGHT = "PREFLIGHT"
    READY = "READY"
    IN_FLIGHT = "IN_FLIGHT"
    RTL = "RTL"
    POSTFLIGHT = "POSTFLIGHT"


@dataclass(slots=True)
class PlanSnapshot:
    points: list[tuple[float, float]]
    confirmed_at: datetime


class MissionStateMachine:
    def __init__(self, *, link_monitor=None, camera_monitor=None) -> None:
        self._state = MissionState.PREFLIGHT
        self._plan: PlanSnapshot | None = None
        self._link_monitor = link_monitor
        self._camera_monitor = camera_monitor
        self._session_active = False
        self._allow_unsafe_start = False

    @property
    def current_state(self) -> MissionState:
        return self._state

    @property
    def plan_confirmed(self) -> bool:
        return self._plan is not None

    @property
    def confirmed_plan(self) -> list[tuple[float, float]] | None:
        if not self._plan:
            return None
        return list(self._plan.points)

    def set_preflight(self, reason: str | None = None) -> None:
        self._set_state(MissionState.PREFLIGHT, reason=reason or "preflight")

    def confirm_plan(self, pts: Iterable[tuple[float, float]]) -> bool:
        path = list(pts)
        if not self._is_valid_plan(path):
            log.warning("Plan confirmation rejected: invalid route.")
            return False
        self._plan = PlanSnapshot(points=path, confirmed_at=utc_now())
        self._refresh_ready_state(reason="plan_confirmed")
        return True

    def invalidate_plan(self, reason: str | None = None) -> None:
        if self._plan is None:
            return
        self._plan = None
        if self._state in (MissionState.READY, MissionState.PREFLIGHT):
            self._set_state(MissionState.PREFLIGHT, reason=reason or "plan_invalidated")

    def start_flight(self, *, skip_checks: bool = False) -> bool:
        if not self.plan_confirmed:
            return False
        if not skip_checks and not self._allow_unsafe_start:
            if not self._is_link_ok():
                return False
            if not self._is_camera_ok():
                return False
        self._session_active = True
        self._set_state(MissionState.IN_FLIGHT, reason="start_flight")
        bus.emit(
            Event.FLIGHT_SESSION_STARTED,
            {
                "started_at": utc_iso_z(),
                "plan": self.confirmed_plan or [],
            },
        )
        return True

    def set_allow_unsafe_start(self, allowed: bool) -> None:
        self._allow_unsafe_start = bool(allowed)

    def trigger_rtl(self, reason: str | None = None) -> None:
        if self._state not in (MissionState.IN_FLIGHT, MissionState.READY):
            return
        self._set_state(MissionState.RTL, reason=reason or "rtl")

    def land_complete(self, reason: str | None = None) -> None:
        if not self._session_active:
            self._set_state(MissionState.POSTFLIGHT, reason=reason or "land_complete")
            return
        self._session_active = False
        self._set_state(MissionState.POSTFLIGHT, reason=reason or "land_complete")
        bus.emit(
            Event.FLIGHT_SESSION_ENDED,
            {"ended_at": utc_iso_z(), "reason": reason or "land_complete"},
        )

    def abort_to_preflight(self, reason: str | None = None) -> None:
        if self._session_active:
            self._session_active = False
            bus.emit(
                Event.FLIGHT_SESSION_ENDED,
                {"ended_at": utc_iso_z(), "reason": reason or "abort"},
            )
        self._plan = None
        self._set_state(MissionState.PREFLIGHT, reason=reason or "abort")

    def refresh_readiness(self, reason: str | None = None) -> None:
        self._refresh_ready_state(reason=reason or "readiness_update")

    # ------------------------------------------------------------------ #
    def _refresh_ready_state(self, reason: str) -> None:
        if self._state in (MissionState.IN_FLIGHT, MissionState.RTL, MissionState.POSTFLIGHT):
            return
        ready = self.plan_confirmed and self._is_link_ok() and self._is_camera_ok()
        target = MissionState.READY if ready else MissionState.PREFLIGHT
        self._set_state(target, reason=reason)

    def _is_valid_plan(self, pts: Sequence[tuple[float, float]]) -> bool:
        return len(pts) >= 2

    def _is_link_ok(self) -> bool:
        if self._link_monitor is None:
            return False
        try:
            return bool(self._link_monitor.is_link_ok())
        except Exception:  # noqa: BLE001
            return False

    def _is_camera_ok(self) -> bool:
        if self._camera_monitor is None:
            return False
        try:
            return bool(self._camera_monitor.is_camera_ok())
        except Exception:  # noqa: BLE001
            return False

    def _set_state(self, new_state: MissionState, reason: str | None = None) -> None:
        if new_state == self._state:
            return
        self._state = new_state
        bus.emit(
            Event.MISSION_STATE_CHANGED,
            {"state": new_state.value, "reason": reason or "state_change"},
        )


__all__ = ["MissionState", "MissionStateMachine", "PlanSnapshot"]
