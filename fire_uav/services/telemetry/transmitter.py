"""
Очень простой TCP/UDP передатчик (строки JSON с разделителем `\\n`).
"""

from __future__ import annotations

import os
import json
import logging
import socket
from typing import Any


_log = logging.getLogger(__name__)


class Transmitter:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9000,
        *,
        udp: bool = False,
        timeout_s: float = 3.0,
    ) -> None:
        self.addr = (host, port)
        self.udp = udp
        self._timeout = timeout_s
        if udp:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.settimeout(timeout_s)
        else:
            # TCP: быстрый connect с таймаутом
            self.sock = socket.create_connection(self.addr, timeout=timeout_s)
        _log.info(
            "Transmitter connected %s:%d (%s, timeout=%.1fs)",
            host,
            port,
            "udp" if udp else "tcp",
            timeout_s,
        )

    def send(self, obj: Any) -> None:
        data = (json.dumps(obj) + "\n").encode()
        print("data sended")
        if self.udp:
            self.sock.sendto(data, self.addr)
        else:
            self.sock.sendall(data)

    def close(self) -> None:
        self.sock.close()
