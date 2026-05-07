from __future__ import annotations

# костыль, так как jetson не работает с python 3.11
try:
    from enum import StrEnum
except ImportError:
    from enum import Enum
    class StrEnum(str, Enum):
        pass
from typing import Callable

from fire_uav.services.mission.state import MissionState


class ActivePathMode(StrEnum):
    NORMAL = "NORMAL"
    ORBIT = "ORBIT"
    RTL = "RTL"


class ActivePathController:
    def __init__(
        self,
        *,
        confirmed_path_provider: Callable[[], list[tuple[float, float]]],
        draft_path_provider: Callable[[], list[tuple[float, float]]],
        mission_state_provider: Callable[[], MissionState],
        plan_confirmed_provider: Callable[[], bool],
        on_change: Callable[[ActivePathMode], None] | None = None,
    ) -> None:
        self._confirmed_path_provider = confirmed_path_provider
        self._draft_path_provider = draft_path_provider
        self._mission_state_provider = mission_state_provider
        self._plan_confirmed_provider = plan_confirmed_provider
        self._on_change = on_change
        self._mode = ActivePathMode.NORMAL
        self._orbit_path: list[tuple[float, float]] | None = None
        self._rtl_path: list[tuple[float, float]] | None = None

    @property
    def mode(self) -> ActivePathMode:
        return self._mode

    def get_active_path(self) -> list[tuple[float, float]]:
        if self._mode == ActivePathMode.RTL and self._rtl_path:
            return list(self._rtl_path)
        if self._mode == ActivePathMode.ORBIT and self._orbit_path:
            return list(self._orbit_path)
        if self._plan_confirmed_provider():
            return list(self._confirmed_path_provider() or [])
        if self._mission_state_provider() == MissionState.PREFLIGHT:
            return list(self._draft_path_provider() or [])
        return list(self._confirmed_path_provider() or [])

    def set_normal(self) -> None:
        changed = self._mode != ActivePathMode.NORMAL or self._orbit_path or self._rtl_path
        self._mode = ActivePathMode.NORMAL
        self._orbit_path = None
        self._rtl_path = None
        self._notify_if_changed(changed)

    def set_orbit(self, pts: list[tuple[float, float]] | None) -> None:
        path = list(pts or [])
        changed = self._mode != ActivePathMode.ORBIT or path != (self._orbit_path or [])
        self._mode = ActivePathMode.ORBIT if path else ActivePathMode.NORMAL
        self._orbit_path = path or None
        self._notify_if_changed(changed)

    def set_rtl(self, pts: list[tuple[float, float]] | None) -> None:
        path = list(pts or [])
        changed = self._mode != ActivePathMode.RTL or path != (self._rtl_path or [])
        self._mode = ActivePathMode.RTL if path else ActivePathMode.NORMAL
        self._rtl_path = path or None
        self._notify_if_changed(changed)

    def clear_overrides_on_new_flight(self) -> None:
        self.set_normal()

    def _notify_if_changed(self, changed: bool) -> None:
        if not changed or self._on_change is None:
            return
        self._on_change(self._mode)


__all__ = ["ActivePathController", "ActivePathMode"]
