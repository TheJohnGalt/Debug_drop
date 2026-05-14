"""Driver registry for selecting UAV backends."""

from __future__ import annotations

from fire_uav.module_core.adapters import (
    CustomSdkUavAdapter,
    MavlinkUavAdapter,
    StubUavAdapter,
    UnrealSimUavAdapter,
    ModuleAdapter,
)
from fire_uav.module_core.interfaces.uav_driver import UavDriver


def resolve_driver_type(cfg) -> str:
    driver_type = (getattr(cfg, "driver_type", "") or "").strip().lower()
    if not driver_type:
        driver_type = (getattr(cfg, "uav_backend", "") or "").strip().lower()
    return driver_type or "stub"


def create_driver(cfg, *, logger=None) -> UavDriver:
    driver_type = resolve_driver_type(cfg)
    if driver_type in ("stub", "dummy", "fake"):
        lat, lon = 56.02, 92.90
        center = getattr(cfg, "map_center", None)
        if isinstance(center, (list, tuple)) and len(center) >= 2:
            try:
                lat, lon = float(center[0]), float(center[1])
            except (TypeError, ValueError):
                pass
        return StubUavAdapter(default_lat=lat, default_lon=lon, logger=logger)
    if driver_type == "mavlink":
        return MavlinkUavAdapter(cfg.mavlink_connection_string, logger=logger)
    if driver_type == "unreal":
        return UnrealSimUavAdapter(
            base_url=cfg.unreal_base_url,
            uav_id=getattr(cfg, "uav_id", None) or "sim",
            logger=logger,
        )
    if driver_type in ("custom", "client_bridge"):
        return CustomSdkUavAdapter(cfg.custom_sdk_config, logger=logger)
    if driver_type in ("module_adapter"):
        return ModuleAdapter(telemetry_base_url="http://127.0.0.1:8000", logger=logger)

    raise ValueError(f"Unknown driver_type: {driver_type}")


__all__ = ["create_driver", "resolve_driver_type"]
