"""
Managed FastAPI relay thread.

Запускает FastAPI-приложение внутри ManagedComponent,
чтобы relay можно было стартовать/останавливать через общий pipeline.
"""

from __future__ import annotations

import logging
from typing import Final

import uvicorn

from fire_uav.services.components.base import ManagedComponent, State

from fire_uav.services.telemetry.relay import app, LOCAL_HOST, LOCAL_PORT

LOG: Final = logging.getLogger("relay")

class RelayThread(ManagedComponent):
    """Managed FastAPI relay worker."""

    def __init__(
        self,
        *,
        host: str = LOCAL_HOST,
        port: int = LOCAL_PORT,
        log_level: str = "info",
    ) -> None:
        super().__init__(name="RelayThread")

        self.host = host
        self.port = port
        self.log_level = log_level

        self._server: uvicorn.Server | None = None

    def loop(self) -> None:
        config = uvicorn.Config(
            app=app,
            host=self.host,
            port=self.port,
            log_level=self.log_level,
            lifespan="on",
        )
        
        self._server = uvicorn.Server(config)

        LOG.info("Starting relay FastAPI server on %s:%s", self.host, self.port)

        self._server.run()

        LOG.info("Relay FastAPI server stopped")
        self.state = State.STOPPED

    def stop(self) -> None:
        super().stop()

        if self._server is not None:
            LOG.info("Stopping relay FastAPI server")
            self._server.should_exit = True