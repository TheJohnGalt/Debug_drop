"""
Лёгкая pub-sub шина c минимальными зависимостями.

▪ Поддерживает любое количество подписчиков на строковый event-key
▪ Логирует все события (debug-уровень)
▪ Перехватывает исключения подписчиков, чтобы не ронять приложение
"""

from __future__ import annotations

import logging
from collections import defaultdict

# костыль, так как jetson не работает с python 3.11
try:
    from enum import StrEnum
except ImportError:
    from enum import Enum
    class StrEnum(str, Enum):
        pass

from typing import Any, DefaultDict, List, Protocol, runtime_checkable

log = logging.getLogger("bus")


class Event(StrEnum):
    """Строго перечисляем public-события приложения."""

    # управление
    APP_START = "app_start"
    APP_STOP = "app_stop"
    MODULE_UNHEALTHY = "module_unhealthy"
    BATTERY_CRITICAL = "battery_critical"
    MISSION_STATE_CHANGED = "mission_state_changed"
    UAV_LINK_STATUS_CHANGED = "uav_link_status_changed"
    CAMERA_STATUS_CHANGED = "camera_status_changed"
    CAPABILITIES_UPDATED = "capabilities_updated"
    COMMAND_ACK = "command_ack"
    WARNING_TOAST = "warning_toast"
    FLIGHT_SESSION_STARTED = "flight_session_started"
    FLIGHT_SESSION_ENDED = "flight_session_ended"

    # детектор
    DETECTION = "detection"
    CONF_CHANGE = "conf_change"
    DETECTOR_START = "detector_start"
    DETECTOR_STOP = "detector_stop"
    OBJECT_CONFIRMED_UI = "object_confirmed_ui"

    # прочее можно добавлять по мере надобности


@runtime_checkable
class _Callback(Protocol):
    def __call__(self, payload: Any | None) -> None: ...  # noqa: D401, E701


class _EventBus:
    """Singleton-шина: subscribe/emit."""

    def __init__(self) -> None:
        self._subs: DefaultDict[str, List[_Callback]] = defaultdict(list)

    # ---------------- subscribe / emit ---------------- #
    def subscribe(self, event: Event | str, cb: _Callback, /) -> None:
        """Регистрируем callback. Дубликаты не добавляем."""
        key = str(event)
        if cb not in self._subs[key]:
            self._subs[key].append(cb)
            log.debug("SUB   %-12s  %s", key, cb)

    def emit(self, event: Event | str, payload: Any | None = None, /) -> None:
        """Вызываем подписчиков; ошибки логируем, но не пробрасываем."""
        key = str(event)
        log.debug("EMIT  %-12s  %s", key, type(payload).__name__)
        for cb in list(self._subs.get(key, [])):
            try:
                cb(payload)
            except Exception:  # noqa: BLE001
                log.exception("Error in subscriber %s for event %s", cb, key)


# глобальный экземпляр
bus: _EventBus = _EventBus()
