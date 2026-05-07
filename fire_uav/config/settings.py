"""
Настройки проекта: единый объект `settings`, который подхватывает значения из JSON
и предоставляет их всем слоям приложения.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal

from fire_uav.module_core.settings_loader import Settings as _SettingsDict, load_settings as _load_settings
from fire_uav.core.protocol import MapBounds


@dataclass
class Settings:
    # Runtime role/adapter
    profile: Literal["dev", "demo", "jetson"] = "dev"
    role: str = "ground"
    uav_backend: str = "unreal"
    driver_type: str = ""
    mavlink_connection_string: str = "udp:127.0.0.1:14550"
    unreal_base_url: str = "http://127.0.0.1:9000"
    unreal_video_mode: str = "h264_stream"
    unreal_video_endpoint: str = "/sim/v1/video.ts"
    unreal_video_target_fps: float = 60.0
    unreal_video_reconnect_s: float = 1.0
    unreal_camera_hz: float = 60.0
    unreal_telemetry_hz: float = 60.0
    unreal_detections_hz: float = 1.0
    unreal_detection_source: str = "local_yolo"
    unreal_local_detect_hz: float = 5.0
    camera_fov_deg: float = 82.1
    camera_mount_pitch_deg: float = 90.0
    camera_mount_yaw_deg: float = 0.0
    camera_mount_roll_deg: float = 0.0
    custom_sdk_config: dict = field(default_factory=dict)
    use_native_core: bool = False
    use_ortools: bool = True
    uav_id: str | None = None
    notifications_dir: Path = Path("data/notifications")
    bbox_smooth_alpha: float = 0.25
    bbox_smooth_max_dist_px: float = 60.0
    track_iou_threshold: float = 0.25
    track_max_age_seconds: float = 2.0
    track_min_hits: int = 2
    track_max_missed: int = 10
    track_max_center_distance_px: float = 80.0
    visualizer_enabled: bool = True
    visualizer_url: str = "http://127.0.0.1:8000"
    log_level: str = "WARNING"
    log_to_file: bool = False
    log_file: Path = Path("data/logs/fire_uav.log")
    log_max_bytes: int = 5_000_000
    log_backup_count: int = 3
    health_host: str = "127.0.0.1"
    health_port: int = 8079
    watchdog_interval_sec: float = 5.0
    no_telemetry_timeout_sec: float = 10.0
    no_detection_timeout_sec: float = 60.0
    watchdog_expect_detections: bool = False

    # Параметры YOLO
    yolo_model: str = "data/models/best_yolo11.pt"
    yolo_conf: float = 0.15
    yolo_iou: float = 0.45
    yolo_classes: List[int] = field(default_factory=lambda: [0, 1])

    # Общие пути
    output_root: Path = Path("data/outputs")

    # Отправка на наземную станцию
    ground_station_host: str = "192.168.0.24"
    ground_station_port: int = 9000
    ground_station_udp: bool = False
    ground_station_enabled: bool = False

    # Агрегация по кадрам / телеметрия
    agg_window: int = 3
    agg_votes_required: int = 1
    agg_min_confidence: float = 0.4
    agg_max_distance_m: float = 60.0
    agg_ttl_seconds: float = 8.0
    dedup_bbox_center_distance_px: float = 120.0
    dedup_geo_distance_m: float = 45.0
    match_radius_m: float = 35.0
    object_registry_match_radius_m: float = 45.0
    suppression_radius_m: float = 30.0
    suppression_ttl_s: float = 180.0
    stable_frames_n: int = 1

    # ------------------------ #
    map_provider: str = "openlayers_de"
    static_map_image_path: str | None = None
    static_map_bounds: dict | None = None
    map_center: list[float] | None = field(default_factory=lambda: [56.02, 92.9])
    home_lat: float | None = None
    home_lon: float | None = None
    base_lat: float | None = None
    base_lon: float | None = None
    gsd_cm: float = 3.0
    side_overlap: float = 0.7
    front_overlap: float = 0.8
    auto_orbit_enabled: bool = False
    orbit_radius_m: float = 50.0
    orbit_points_per_circle: int = 12
    orbit_loops: int = 1
    maneuver_alt_m: float | None = None
    cruise_speed_mps: float = 12.0
    power_cruise_w: float = 45.0
    battery_wh: float = 77.0
    no_fly_geojson: str = "data/no_fly_zones.geojson"
    max_flight_distance_m: float = 0.0
    min_return_percent: float = 20.0
    critical_battery_percent: float = 10.0

    @classmethod
    def from_dict(cls, data: _SettingsDict) -> "Settings":
        """Собрать объект настроек из dict (например, из JSON-файла)."""
        defaults = cls()
        bounds_raw = data.get("static_map_bounds", defaults.static_map_bounds)
        static_bounds = None
        if isinstance(bounds_raw, MapBounds):
            static_bounds = bounds_raw.model_dump()
        elif isinstance(bounds_raw, dict):
            try:
                static_bounds = MapBounds(**bounds_raw).model_dump()
            except Exception:
                static_bounds = None
        return cls(
            profile=data.get("profile", defaults.profile),
            role=data.get("role", defaults.role),
            uav_backend=data.get("uav_backend", defaults.uav_backend),
            driver_type=data.get("driver_type", defaults.driver_type),
            mavlink_connection_string=data.get(
                "mavlink_connection_string", defaults.mavlink_connection_string
            ),
            unreal_base_url=data.get("unreal_base_url", defaults.unreal_base_url),
            unreal_video_mode=str(data.get("unreal_video_mode", defaults.unreal_video_mode)),
            unreal_video_endpoint=str(
                data.get("unreal_video_endpoint", defaults.unreal_video_endpoint)
            ),
            unreal_video_target_fps=float(
                data.get("unreal_video_target_fps", defaults.unreal_video_target_fps)
            ),
            unreal_video_reconnect_s=float(
                data.get("unreal_video_reconnect_s", defaults.unreal_video_reconnect_s)
            ),
            unreal_camera_hz=float(data.get("unreal_camera_hz", defaults.unreal_camera_hz)),
            unreal_telemetry_hz=float(
                data.get("unreal_telemetry_hz", defaults.unreal_telemetry_hz)
            ),
            unreal_detections_hz=float(
                data.get("unreal_detections_hz", defaults.unreal_detections_hz)
            ),
            unreal_detection_source=str(
                data.get("unreal_detection_source", defaults.unreal_detection_source)
            ),
            unreal_local_detect_hz=float(
                data.get("unreal_local_detect_hz", defaults.unreal_local_detect_hz)
            ),
            camera_fov_deg=float(data.get("camera_fov_deg", defaults.camera_fov_deg)),
            camera_mount_pitch_deg=float(
                data.get("camera_mount_pitch_deg", defaults.camera_mount_pitch_deg)
            ),
            camera_mount_yaw_deg=float(
                data.get("camera_mount_yaw_deg", defaults.camera_mount_yaw_deg)
            ),
            camera_mount_roll_deg=float(
                data.get("camera_mount_roll_deg", defaults.camera_mount_roll_deg)
            ),
            custom_sdk_config=data.get("custom_sdk_config", defaults.custom_sdk_config),
            use_native_core=bool(data.get("use_native_core", defaults.use_native_core)),
            use_ortools=bool(data.get("use_ortools", defaults.use_ortools)),
            uav_id=data.get("uav_id", defaults.uav_id),
            notifications_dir=Path(data.get("notifications_dir", defaults.notifications_dir)),
            bbox_smooth_alpha=float(data.get("bbox_smooth_alpha", defaults.bbox_smooth_alpha)),
            bbox_smooth_max_dist_px=float(
                data.get("bbox_smooth_max_dist_px", defaults.bbox_smooth_max_dist_px)
            ),
            track_iou_threshold=float(data.get("track_iou_threshold", defaults.track_iou_threshold)),
            track_max_age_seconds=float(
                data.get("track_max_age_seconds", defaults.track_max_age_seconds)
            ),
            track_min_hits=int(data.get("track_min_hits", defaults.track_min_hits)),
            track_max_missed=int(data.get("track_max_missed", defaults.track_max_missed)),
            track_max_center_distance_px=float(
                data.get("track_max_center_distance_px", defaults.track_max_center_distance_px)
            ),
            visualizer_enabled=bool(data.get("visualizer_enabled", defaults.visualizer_enabled)),
            visualizer_url=data.get("visualizer_url", defaults.visualizer_url),
            log_level=data.get("log_level", defaults.log_level),
            log_to_file=bool(data.get("log_to_file", defaults.log_to_file)),
            log_file=Path(data.get("log_file", defaults.log_file)),
            log_max_bytes=int(data.get("log_max_bytes", defaults.log_max_bytes)),
            log_backup_count=int(data.get("log_backup_count", defaults.log_backup_count)),
            health_host=data.get("health_host", defaults.health_host),
            health_port=int(data.get("health_port", defaults.health_port)),
            watchdog_interval_sec=float(
                data.get("watchdog_interval_sec", defaults.watchdog_interval_sec)
            ),
            no_telemetry_timeout_sec=float(
                data.get("no_telemetry_timeout_sec", defaults.no_telemetry_timeout_sec)
            ),
            no_detection_timeout_sec=float(
                data.get("no_detection_timeout_sec", defaults.no_detection_timeout_sec)
            ),
            watchdog_expect_detections=bool(
                data.get("watchdog_expect_detections", defaults.watchdog_expect_detections)
            ),
            yolo_model=data.get("yolo_model", defaults.yolo_model),
            yolo_conf=float(data.get("yolo_conf", defaults.yolo_conf)),
            yolo_iou=float(data.get("yolo_iou", defaults.yolo_iou)),
            yolo_classes=list(data.get("yolo_classes", defaults.yolo_classes)),
            output_root=Path(data.get("output_root", defaults.output_root)),
            ground_station_host=data.get("ground_station_host", defaults.ground_station_host),
            ground_station_port=int(data.get("ground_station_port", defaults.ground_station_port)),
            ground_station_udp=bool(data.get("ground_station_udp", defaults.ground_station_udp)),
            ground_station_enabled=bool(
                data.get("ground_station_enabled", defaults.ground_station_enabled)
            ),
            agg_window=int(data.get("agg_window", defaults.agg_window)),
            agg_votes_required=int(data.get("agg_votes_required", defaults.agg_votes_required)),
            agg_min_confidence=float(data.get("agg_min_confidence", defaults.agg_min_confidence)),
            agg_max_distance_m=float(data.get("agg_max_distance_m", defaults.agg_max_distance_m)),
            agg_ttl_seconds=float(data.get("agg_ttl_seconds", defaults.agg_ttl_seconds)),
            dedup_bbox_center_distance_px=float(
                data.get("dedup_bbox_center_distance_px", defaults.dedup_bbox_center_distance_px)
            ),
            dedup_geo_distance_m=float(
                data.get("dedup_geo_distance_m", defaults.dedup_geo_distance_m)
            ),
            match_radius_m=float(data.get("match_radius_m", defaults.match_radius_m)),
            object_registry_match_radius_m=float(
                data.get("object_registry_match_radius_m", defaults.object_registry_match_radius_m)
            ),
            suppression_radius_m=float(
                data.get("suppression_radius_m", defaults.suppression_radius_m)
            ),
            suppression_ttl_s=float(data.get("suppression_ttl_s", defaults.suppression_ttl_s)),
            stable_frames_n=int(data.get("stable_frames_n", defaults.stable_frames_n)),
            map_provider=str(data.get("map_provider", defaults.map_provider)),
            static_map_image_path=data.get("static_map_image_path", defaults.static_map_image_path),
            static_map_bounds=static_bounds,
            map_center=data.get("map_center", defaults.map_center),
            home_lat=(
                None if data.get("home_lat", defaults.home_lat) is None else float(data.get("home_lat"))
            ),
            home_lon=(
                None if data.get("home_lon", defaults.home_lon) is None else float(data.get("home_lon"))
            ),
            base_lat=(
                None if data.get("base_lat", defaults.base_lat) is None else float(data.get("base_lat"))
            ),
            base_lon=(
                None if data.get("base_lon", defaults.base_lon) is None else float(data.get("base_lon"))
            ),
            gsd_cm=float(data.get("gsd_cm", defaults.gsd_cm)),
            side_overlap=float(data.get("side_overlap", defaults.side_overlap)),
            front_overlap=float(data.get("front_overlap", defaults.front_overlap)),
            auto_orbit_enabled=bool(data.get("auto_orbit_enabled", defaults.auto_orbit_enabled)),
            orbit_radius_m=float(data.get("orbit_radius_m", defaults.orbit_radius_m)),
            orbit_points_per_circle=int(
                data.get("orbit_points_per_circle", defaults.orbit_points_per_circle)
            ),
            orbit_loops=int(data.get("orbit_loops", defaults.orbit_loops)),
            maneuver_alt_m=(
                None
                if data.get("maneuver_alt_m", defaults.maneuver_alt_m) is None
                else float(data.get("maneuver_alt_m"))
            ),
            cruise_speed_mps=float(data.get("cruise_speed_mps", defaults.cruise_speed_mps)),
            power_cruise_w=float(data.get("power_cruise_w", defaults.power_cruise_w)),
            battery_wh=float(data.get("battery_wh", defaults.battery_wh)),
            no_fly_geojson=str(data.get("no_fly_geojson", defaults.no_fly_geojson)),
            max_flight_distance_m=float(
                data.get("max_flight_distance_m", defaults.max_flight_distance_m)
            ),
            min_return_percent=float(data.get("min_return_percent", defaults.min_return_percent)),
            critical_battery_percent=float(
                data.get("critical_battery_percent", defaults.critical_battery_percent)
            ),
        )


def _apply_profile_overrides(data: Dict[str, Any], profile: str) -> Dict[str, Any]:
    """Apply profile-specific defaults unless explicitly overridden."""
    profile = profile.lower()
    overrides: Dict[str, Dict[str, Any]] = {
        "dev": {"log_level": "DEBUG", "visualizer_enabled": True},
        "demo": {"log_level": "INFO", "visualizer_enabled": True},
        "jetson": {"log_level": "INFO", "visualizer_enabled": False, "use_native_core": True},
    }
    for key, value in overrides.get(profile, {}).items():
        data.setdefault(key, value)
    return data


def load_settings() -> Settings:
    raw = _load_settings()
    profile = str(os.environ.get("FIRE_UAV_PROFILE", raw.get("profile", "dev"))).lower()
    raw["profile"] = profile
    raw = _apply_profile_overrides(raw, profile)
    return Settings.from_dict(raw)


settings = load_settings()

__all__ = ["Settings", "settings", "load_settings"]
