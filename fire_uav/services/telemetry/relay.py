from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime
from typing import Literal

import httpx
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from fire_uav.module_core.contract.v1 import CommandV1, RouteV1
from fire_uav.services.bus import bus

# to do: доделать релей и слить втеки с наземной станцией !!!

LOCAL_HOST = "0.0.0.0"
LOCAL_PORT = 6000
REMOTE_BASE_URL = "http://192.168.0.24:5000"
REQUEST_TIMEOUT_S = 3.0
RETRY_DELAY_S = 2.0

# to do: перенести DetectionV1 в contract.v1 
class DetectionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol_version: Literal[1] = 1
    uav_id: str
    timestamp: datetime
    class_id: int = Field(..., ge=0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    alt: float | None = None
    frame_id: str | None = None
    track_id: int | None = None
    object_id: str | None = None


class _Outbox:
    def __init__(self) -> None:
        self._items: deque[BaseModel] = deque()
        self._lock = threading.Lock()

    def put(self, msg: BaseModel) -> int:
        with self._lock:
            self._items.append(msg)
            return len(self._items)

    def peek(self) -> BaseModel | None:
        with self._lock:
            return self._items[0] if self._items else None

    def ack(self, msg: BaseModel) -> None:
        with self._lock:
            if self._items and self._items[0] is msg:
                self._items.popleft()

    def size(self) -> int:
        with self._lock:
            return len(self._items)


app = FastAPI(title="fire-uav relay", version="1.0.0")

_detection_outbox = _Outbox()
_route_outbox = _Outbox()
_command_outbox = _Outbox()
_sender_threads_started = False


def _post_json(path: str, msg: BaseModel) -> bool:
    url = f"{REMOTE_BASE_URL.rstrip('/')}{path}"
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT_S) as client:
            resp = client.post(url, json=msg.model_dump(mode="json"))
            resp.raise_for_status()
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[relay] send failed path={path}: {exc}", flush=True)
        return False


def _sender_loop(name: str, outbox: _Outbox, remote_path: str) -> None:
    while True:
        msg = outbox.peek()
        if msg is None:
            time.sleep(0.1)
            continue

        if _post_json(remote_path, msg):
            outbox.ack(msg)
            print(f"[relay] ACK {name} queue={outbox.size()}", flush=True)
            continue

        time.sleep(RETRY_DELAY_S)


def _start_sender_threads() -> None:
    global _sender_threads_started
    if _sender_threads_started:
        return
    _sender_threads_started = True
    threads = (
        ("detection", _detection_outbox, "/link/v1/receive_detection"),
        ("route", _route_outbox, "/link/v1/receive_route"),
        ("command", _command_outbox, "/link/v1/receive_command"),
    )
    for name, outbox, remote_path in threads:
        threading.Thread(
            target=_sender_loop,
            args=(name, outbox, remote_path),
            daemon=True,
        ).start()


@app.on_event("startup")
def on_startup() -> None:
    _start_sender_threads()


@app.post("/link/v1/send_detection")
def send_detection(msg: DetectionV1) -> dict[str, object]:
    queue_size = _detection_outbox.put(msg)
    return {"status": "queued", "queue_size": queue_size}


@app.post("/link/v1/receive_detection")
def receive_detection(msg: DetectionV1) -> dict[str, str]:
    bus.emit("detection_received", msg.model_dump(mode="json"))
    return {"status": "ACK"}


@app.post("/link/v1/send_route")
def send_route(msg: RouteV1) -> dict[str, object]:
    queue_size = _route_outbox.put(msg)
    return {"status": "queued", "queue_size": queue_size}


@app.post("/link/v1/receive_route")
def receive_route(msg: RouteV1) -> dict[str, str]:
    bus.emit("route_sent", msg.model_dump(mode="json"))
    return {"status": "ACK"}


@app.post("/link/v1/send_command")
def send_command(msg: CommandV1) -> dict[str, object]:
    queue_size = _command_outbox.put(msg)
    return {"status": "queued", "queue_size": queue_size}


@app.post("/link/v1/receive_command")
def receive_command(msg: CommandV1) -> dict[str, str]:
    bus.emit("command_received", msg.model_dump(mode="json"))
    print("receive_command")
    return {"status": "ACK"}

@app.get("/link/v1/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "outbox": {
            "detections": _detection_outbox.size(),
            "routes": _route_outbox.size(),
            "commands": _command_outbox.size(),
        },
    }
