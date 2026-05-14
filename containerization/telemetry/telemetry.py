from __future__ import annotations

import asyncio
import json
import math
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import serial
from fastapi import FastAPI
from pydantic import BaseModel
from pymavlink import mavutil

from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

TYPE_IS_UAV = False

MAVLINK_CONNECTION: mavutil.mavfile | None = None
MODEM_SERIAL: serial.Serial | None = None

LAST_MODEM_DATA = None
LAST_UAV_DATA = None

LAST_ROUTE_ID = None
ROUTE_IN_PROGRESS = False
MISSION_TIMEOUT_SEC = 30

CONFIG_PATH = Path(__file__).parent / "telemetry_config.json"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

UAV_ID = CONFIG["uav_id"]

MAVLINK_DEVICE = CONFIG["mavlink_device"]
MAVLINK_BAUDRATE = CONFIG["mavlink_baudrate"]

MODEM_DEVICE = CONFIG["modem_device"]
MODEM_BAUDRATE = CONFIG["modem_baudrate"]

MAVLINK_POSITION_HZ = CONFIG["mavlink_position_hz"]
MAVLINK_ATTITUDE_HZ = CONFIG["mavlink_attitude_hz"]
MAVLINK_BATTERY_HZ = CONFIG["mavlink_battery_hz"]

MISSION_TIMEOUT_SEC = CONFIG["mission_timeout_sec"]

app = FastAPI(title="Telemetry Service")

class ContractBaseV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol_version: Literal[1] = 1
    uav_id: str
    timestamp: datetime

class CommandV1(ContractBaseV1):
    command_id: str
    type: str
    params: dict

class WaypointV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    alt: float
    speed_mps: float | None = None
    loiter_radius_m: float | None = None
    action: str | None = None

class RouteV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol_version: Literal[1] = 1
    uav_id: str
    route_id: str
    waypoints: list[WaypointV1]
    mode: str
    created_at: datetime

class TelemetryData(BaseModel):
    protocol_version: int = 1
    uav_id: str
    timestamp: str

    lat: float
    lon: float
    alt: float

    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None

    ground_speed_mps: Optional[float] = None
    vertical_speed_mps: Optional[float] = None

    battery_percent: Optional[float] = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def start() -> None:
    """
    Инициализация источников телеметрии.
    """
    global TYPE_IS_UAV
    global MAVLINK_CONNECTION
    global MODEM_SERIAL

    TYPE_IS_UAV = resolve_telemetry_type()

    if TYPE_IS_UAV:
        MAVLINK_CONNECTION = mavutil.mavlink_connection(
            MAVLINK_DEVICE,
            baud=MAVLINK_BAUDRATE,
        )

        MAVLINK_CONNECTION.wait_heartbeat(timeout=5)

        def set_message_interval(message_id: int, frequency_hz: float) -> None:
            interval_us = int(1_000_000 / frequency_hz)

            MAVLINK_CONNECTION.mav.command_long_send(
                MAVLINK_CONNECTION.target_system,
                MAVLINK_CONNECTION.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0,
                message_id,
                interval_us,
                0,
                0,
                0,
                0,
                0,
            )

        set_message_interval(
            mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
            MAVLINK_POSITION_HZ,
        )

        set_message_interval(
            mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,
            MAVLINK_ATTITUDE_HZ,
        )

        set_message_interval(
            mavutil.mavlink.MAVLINK_MSG_ID_BATTERY_STATUS,
            MAVLINK_BATTERY_HZ,
        )

    else:
        try:
            MODEM_SERIAL = serial.Serial(
                MODEM_DEVICE,
                MODEM_BAUDRATE,
                timeout=1,
            )

            MODEM_SERIAL.write(b"AT+CGNSPWR=1\r")
            MODEM_SERIAL.flush()
        except:
            pass


def send_mission(waypoints) -> bool:
    """
    Отправка миссии в автопилот через MAVLink.
    """
    global MAVLINK_CONNECTION

    if MAVLINK_CONNECTION is None:
        return False

    try:
        count = len(waypoints)

        MAVLINK_CONNECTION.mav.mission_count_send(
            MAVLINK_CONNECTION.target_system,
            MAVLINK_CONNECTION.target_component,
            count,
            0,
        )

        for seq, wp in enumerate(waypoints):
            MAVLINK_CONNECTION.mav.mission_item_int_send(
                MAVLINK_CONNECTION.target_system,
                MAVLINK_CONNECTION.target_component,
                seq,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                0,
                1,
                0,
                0,
                0,
                0,
                int(wp.lat * 1e7),
                int(wp.lon * 1e7),
                float(wp.alt),
            )

        ack = MAVLINK_CONNECTION.recv_match(
            type="MISSION_ACK",
            blocking=True,
            timeout=MISSION_TIMEOUT_SEC,
        )

        return ack is not None

    except Exception:
        return False

def send_command(command: CommandV1) -> bool:
    """
    Отправка простой команды в автопилот через MAVLink.
    """
    global MAVLINK_CONNECTION

    if MAVLINK_CONNECTION is None:
        return False

    try:
        command_type = command.type.value if hasattr(command.type, "value") else command.type
        params = command.params or {}

        if command_type == "START":
            mav_cmd = mavutil.mavlink.MAV_CMD_MISSION_START
            p1 = float(params.get("first_item", 0))
            p2 = float(params.get("last_item", 0))
            p3 = p4 = p5 = p6 = p7 = 0.0

        elif command_type == "ABORT":
            mav_cmd = mavutil.mavlink.MAV_CMD_DO_PAUSE_CONTINUE
            p1 = 0.0
            p2 = p3 = p4 = p5 = p6 = p7 = 0.0

        elif command_type == "RTL":
            mav_cmd = mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH
            p1 = p2 = p3 = p4 = p5 = p6 = p7 = 0.0

        elif command_type == "ORBIT":
            mav_cmd = mavutil.mavlink.MAV_CMD_DO_ORBIT
            p1 = float(params.get("radius_m", math.nan))
            p2 = float(params.get("velocity_mps", math.nan))
            p3 = float(params.get("yaw_behavior", math.nan))
            p4 = float(params.get("orbits_rad", 0.0))
            p5 = float(params.get("lat", math.nan))
            p6 = float(params.get("lon", math.nan))
            p7 = float(params.get("alt", math.nan))

        elif command_type == "APPLY_ROUTE":
            mav_cmd = mavutil.mavlink.MAV_CMD_MISSION_START
            p1 = float(params.get("first_item", 0))
            p2 = float(params.get("last_item", 0))
            p3 = p4 = p5 = p6 = p7 = 0.0

        elif command_type == "LAND":
            mav_cmd = mavutil.mavlink.MAV_CMD_NAV_LAND
            p1 = float(params.get("abort_alt_m", 0.0))
            p2 = float(params.get("precision_land_mode", 0.0))
            p3 = 0.0
            p4 = float(params.get("yaw_deg", math.nan))
            p5 = float(params.get("lat", math.nan))
            p6 = float(params.get("lon", math.nan))
            p7 = float(params.get("alt", math.nan))

        else:
            return False

        MAVLINK_CONNECTION.mav.command_long_send(
            MAVLINK_CONNECTION.target_system,
            MAVLINK_CONNECTION.target_component,
            mav_cmd,
            0,
            p1,
            p2,
            p3,
            p4,
            p5,
            p6,
            p7,
        )

        ack = MAVLINK_CONNECTION.recv_match(
            type="COMMAND_ACK",
            blocking=True,
            timeout=MISSION_TIMEOUT_SEC,
        )

        return ack is not None

    except Exception:
        return False

def resolve_telemetry_type() -> bool:
    """
    Проверка доступности MAVLink подключения.
    """
    try:
        conn = mavutil.mavlink_connection(
            MAVLINK_DEVICE,
            baud=MAVLINK_BAUDRATE,
        )

        hb = conn.wait_heartbeat(timeout=3)

        if hb is not None:
            conn.close()
            return True

    except Exception:
        pass

    return False


def pull_uav_telemetry() -> Optional[TelemetryData]:
    """
    Получение телеметрии с БПЛА.
    """
    global MAVLINK_CONNECTION

    if MAVLINK_CONNECTION is None:
        return None

    lat = None
    lon = None
    alt = None

    yaw = None
    pitch = None
    roll = None

    ground_speed = None
    vertical_speed = None

    battery = None

    try:
        msg = MAVLINK_CONNECTION.recv_match(
            type="GLOBAL_POSITION_INT",
            blocking=True,
            timeout=1,
        )

        if msg:
            lat = msg.lat / 1e7
            lon = msg.lon / 1e7
            alt = msg.alt / 1000.0

            vx = msg.vx / 100.0
            vy = msg.vy / 100.0
            vz = msg.vz / 100.0

            ground_speed = math.sqrt(vx * vx + vy * vy)
            vertical_speed = -vz

        attitude = MAVLINK_CONNECTION.recv_match(
            type="ATTITUDE",
            blocking=False,
        )

        if attitude:
            roll = math.degrees(attitude.roll)
            pitch = math.degrees(attitude.pitch)
            yaw = math.degrees(attitude.yaw)

        battery_msg = MAVLINK_CONNECTION.recv_match(
            type="BATTERY_STATUS",
            blocking=False,
        )

        if battery_msg:
            battery = battery_msg.battery_remaining

        if lat is None or lon is None or alt is None:
            return None

        return TelemetryData(
            uav_id=UAV_ID,
            timestamp=utc_now_iso(),

            lat=lat,
            lon=lon,
            alt=alt,

            yaw=yaw,
            pitch=pitch,
            roll=roll,

            ground_speed_mps=ground_speed,
            vertical_speed_mps=vertical_speed,

            battery_percent=battery,
        )

    except Exception:
        return None

def pull_modem_telemetry() -> Optional[TelemetryData]:
    return TelemetryData(
        uav_id=UAV_ID,
        timestamp=utc_now_iso(),

        lat=56.03161,
        lon=92.948316,
        alt=30.0,
    )

# временно закоментировано
# def pull_modem_telemetry() -> Optional[TelemetryData]:
#     """
#     Получение GNSS телеметрии с модема.
#     """
#     global MODEM_SERIAL

#     if MODEM_SERIAL is None:
#         return None

#     try:
#         MODEM_SERIAL.write(b"AT+CGNSINF\r")
#         MODEM_SERIAL.flush()

#         lines = MODEM_SERIAL.readlines()

#         for raw in lines:
#             line = raw.decode(errors="ignore").strip()

#             if "+CGNSINF:" not in line:
#                 continue

#             data = line.split(",")

#             lat = float(data[3])
#             lon = float(data[4])
#             alt = float(data[5])

#             return TelemetryData(
#                 uav_id=UAV_ID,
#                 timestamp=utc_now_iso(),

#                 lat=lat,
#                 lon=lon,
#                 alt=alt,
#             )

#     except Exception:
#         return None

#     return None


def telemetry_worker() -> None:
    """
    Фоновое обновление телеметрии.
    """
    global LAST_MODEM_DATA
    global LAST_UAV_DATA

    while True:

        if TYPE_IS_UAV:
            LAST_UAV_DATA = pull_uav_telemetry()
        else:
            LAST_MODEM_DATA = pull_modem_telemetry()

        asyncio.run(asyncio.sleep(0.1))


@app.get("/telemetry")
async def get_telemetry():

    if TYPE_IS_UAV:
        if LAST_UAV_DATA is not None:
            return LAST_UAV_DATA.model_dump()

    else:
        if LAST_MODEM_DATA is not None:
            return LAST_MODEM_DATA.model_dump(
                exclude_none=True
            )

    return {
        "protocol_version": 1,
        "uav_id": UAV_ID,
        "timestamp": utc_now_iso(),
        "lat": 0.0,
        "lon": 0.0,
        "alt": 0.0,
    }

@app.post("/mission")
async def upload_mission(mission: RouteV1):

    global LAST_ROUTE_ID
    global ROUTE_IN_PROGRESS

    if TYPE_IS_UAV is False:
        return {"ok": False, "error": "MAVLink not available"}

    if MAVLINK_CONNECTION is None:
        return {"ok": False, "error": "no connection"}

    ROUTE_IN_PROGRESS = True
    LAST_ROUTE_ID = mission.route_id

    success = send_mission(mission.waypoints)

    ROUTE_IN_PROGRESS = False

    if not success:
        return {
            "ok": False,
            "route_id": mission.route_id,
            "status": "failed",
        }

    return {
        "ok": True,
        "route_id": mission.route_id,
        "status": "uploaded",
    }

@app.post("/command")
async def receive_command(command: CommandV1):

    if TYPE_IS_UAV is False:
        return {"ok": False, "error": "MAVLink not available"}

    if MAVLINK_CONNECTION is None:
        return {"ok": False, "error": "no connection"}

    success = send_command(command)

    if not success:
        return {
            "ok": False,
            "command_id": command.command_id,
            "status": "failed",
        }

    return {
        "ok": True,
        "command_id": command.command_id,
        "status": "sent",
    }

@app.on_event("startup")
async def startup_event():
    start()
    thread = threading.Thread( target=telemetry_worker, daemon=True, )
    thread.start()