"""Единая точка управления компонентами, работающими в потоках."""

from __future__ import annotations

import logging
import threading
from threading import Event
from typing import List

from fire_uav.services.components.base import ManagedComponent, State

_log = logging.getLogger(__name__)


class LifecycleManager:
    """
    Регистрирует все `ManagedComponent`-ы и обеспечивает
    согласованный `start_all()` / `stop_all()` с graceful-join.
    """

    def __init__(self) -> None:
        self._components: List[ManagedComponent] = []
        self._stop_event = Event()

    # ─────────────────── регистрация ─────────────────── #
    def register(self, *components: ManagedComponent | None) -> None:
        for c in components:
            if c is None or c in self._components:
                continue
            self._components.append(c)
            _log.debug("Registered component %s", c.name)
            print("LifecycleManager Registered component %s", c.name)

    # ───────────────── start / stop ───────────────────── #
    def start_all(self) -> None:
        for c in self._components:
            if c.state is State.IDLE:
                _log.info("Starting %s", c.name)
                c.start()

    def stop_all(self, join: bool = True, timeout: float | None = None) -> None:
        for c in self._components:
            _log.info("Stopping %s", c.name)
            c.stop()
        if join:
            for c in self._components:
                if c.ident is None:
                    _log.debug("Skipping join for non-started component %s", c.name)
                    continue
                if c.ident == threading.get_ident():
                    _log.debug("Skipping self-join for component %s", c.name)
                    continue
                _log.debug("Joining %s", c.name)
                c.join(timeout)

    # ─────────────────── диагностика ─────────────────── #
    def states(self) -> list[tuple[str, State]]:
        return [(c.name, c.state) for c in self._components]
