from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Callable, List, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

from fire_uav.config import settings
from fire_uav.domain.video.camera import CameraParams
from fire_uav.module_core.detections.aggregator import DetectionAggregator, DetectionEvent
from fire_uav.module_core.geometry import haversine_m
from fire_uav.module_core.detections.manager import ObjectNotificationManager
from fire_uav.module_core.detections.notifications import JsonNotificationWriter
from fire_uav.module_core.detections.registry import ObjectRegistry
from fire_uav.module_core.detections.smoothing import build_smoother
from fire_uav.module_core.factories import get_geo_projector
from fire_uav.module_core.interfaces.geo import IGeoProjector
from fire_uav.module_core.schema import GeoDetection, TelemetrySample, WorldCoord
from fire_uav.services.targets.target_tracker import TargetObservation, TargetTracker
from fire_uav.services.telemetry.transmitter import Transmitter
from fire_uav.utils.time import utc_now

logger = logging.getLogger(__name__)


class RawDetectionPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    class_id: int = Field(..., ge=0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: Tuple[int, int, int, int]
    frame_id: str
    timestamp: datetime
    track_id: int | None = None


class DetectionBatchPayload(BaseModel):
    frame_id: str
    frame_width: int = Field(..., gt=0)
    frame_height: int = Field(..., gt=0)
    captured_at: datetime
    telemetry: TelemetrySample
    detections: List[RawDetectionPayload]


class DetectionPipeline:
    """
    Связывает сырые детекции модели с телеметрией, выполняет агрегацию
    и отправляет подтверждённые цели на наземную станцию.
    """

    def __init__(
        self,
        *,
        aggregator: DetectionAggregator | None = None,
        projector: IGeoProjector | None = None,
        transmitter: Transmitter | None = None,
        camera_params: CameraParams | None = None,
        visualizer_adapter=None,
        loop=None,
        detection_callback: Callable[[datetime], None] | None = None,
        target_tracker: TargetTracker | None = None,
    ) -> None:
        self.aggregator = aggregator or DetectionAggregator(
            window=settings.agg_window,
            votes_required=settings.agg_votes_required,
            min_confidence=settings.agg_min_confidence,
            max_distance_m=settings.agg_max_distance_m,
            ttl_seconds=settings.agg_ttl_seconds,
        )
        resolved_camera = camera_params
        # Prefer native geo projector when available; falls back to Python implementation.
        self.projector = projector or get_geo_projector(settings, camera=resolved_camera)
        if camera_params is not None and hasattr(self.projector, "set_camera_params"):
            try:
                self.projector.set_camera_params(camera_params)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                logger.debug("Failed to apply explicit camera params to projector", exc_info=True)
        self.transmitter = transmitter
        self._smoother = build_smoother(settings)
        self._registry = ObjectRegistry(
            spatial_match_radius_m=float(
                getattr(settings, "object_registry_match_radius_m", 90.0) or 90.0
            )
        )
        notifications_dir = Path(getattr(settings, "notifications_dir", "data/notifications"))
        self._notification_manager = ObjectNotificationManager(
            registry=self._registry,
            writer=JsonNotificationWriter(notifications_dir),
            logger=logger,
            uav_id=getattr(settings, "uav_id", None),
        )
        self._target_tracker = target_tracker or TargetTracker(
            match_radius_m=float(getattr(settings, "match_radius_m", 30.0) or 30.0),
            suppression_radius_m=float(getattr(settings, "suppression_radius_m", 60.0) or 60.0),
            suppression_ttl_s=float(getattr(settings, "suppression_ttl_s", 180.0) or 180.0),
            stable_frames_n=int(getattr(settings, "stable_frames_n", 1) or 1),
        )
        self._lock = Lock()
        self._visualizer = visualizer_adapter
        self._loop = loop
        self._detection_callback = detection_callback

    def process_batch(self, payload: DetectionBatchPayload) -> List[GeoDetection]:
        if not payload.detections:
            return []

        projected_events: list[tuple[DetectionEvent, tuple[float, float, float, float]]] = []
        smoothed = self._smoother.assign_and_smooth(payload.detections)
        for det, smoothed_bbox, track_id in smoothed:
            projected = self.projector.project_bbox_to_ground(
                payload.telemetry,
                smoothed_bbox,
                payload.frame_width,
                payload.frame_height,
            )
            if projected is None:
                logger.debug(
                    "Detection projection skipped: frame=%s bbox=%s track_id=%s reason=invalid_ground_intersection",
                    det.frame_id or payload.frame_id,
                    smoothed_bbox,
                    track_id,
                )
                continue
            lat, lon = projected
            coord = WorldCoord(lat=lat, lon=lon)
            projected_events.append(
                (
                    DetectionEvent(
                        class_id=det.class_id,
                        confidence=det.confidence,
                        location=coord,
                        frame_id=det.frame_id or payload.frame_id,
                        timestamp=det.timestamp or payload.captured_at,
                        track_id=track_id,
                    ),
                    tuple(float(v) for v in smoothed_bbox),
                )
            )

        if self.aggregator is None:
            return []

        events = self._dedupe_projected_events(projected_events)
        with self._lock:
            aggregated = self.aggregator.add_many(events)

        confirmed = self._collect_stable_confirmations(aggregated)
        if confirmed:
            for det in confirmed:
                self._notification_manager.handle_confirmed_detection(det)
                self._publish_visualizer(det)
            if self._detection_callback:
                self._detection_callback(confirmed[-1].timestamp or utc_now())
        self._transmit(confirmed)
        return confirmed

    def _dedupe_projected_events(
        self,
        projected_events: Sequence[tuple[DetectionEvent, tuple[float, float, float, float]]],
    ) -> list[DetectionEvent]:
        if not projected_events:
            return []
        merge_px = float(getattr(settings, "dedup_bbox_center_distance_px", 72.0) or 72.0)
        merge_geo_m = float(getattr(settings, "dedup_geo_distance_m", 45.0) or 45.0)
        kept: list[tuple[DetectionEvent, tuple[float, float, float, float]]] = []
        for event, bbox in sorted(
            projected_events,
            key=lambda item: float(item[0].confidence),
            reverse=True,
        ):
            duplicate = False
            for prev_event, prev_bbox in kept:
                if prev_event.class_id != event.class_id:
                    continue
                if self._bbox_center_distance_px(prev_bbox, bbox) > merge_px:
                    continue
                dist_m = haversine_m(
                    (prev_event.location.lat, prev_event.location.lon),
                    (event.location.lat, event.location.lon),
                )
                if dist_m <= merge_geo_m:
                    duplicate = True
                    break
            if not duplicate:
                kept.append((event, bbox))
        return [event for event, _bbox in kept]

    @staticmethod
    def _bbox_center_distance_px(
        a: tuple[float, float, float, float],
        b: tuple[float, float, float, float],
    ) -> float:
        ax = (a[0] + a[2]) * 0.5
        ay = (a[1] + a[3]) * 0.5
        bx = (b[0] + b[2]) * 0.5
        by = (b[1] + b[3]) * 0.5
        return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

    # ------------------------------------------------------------------ #
    def _transmit(self, detections: Sequence[GeoDetection]) -> None:
        if not self.transmitter or not detections:
            return
        for det in detections:
            payload = {
                "class_id": det.class_id,
                "confidence": det.confidence,
                "lat": det.lat,
                "lon": det.lon,
                "timestamp": det.timestamp.isoformat(),
                "frame": det.frame_id,
            }
            try:
                self.transmitter.send(payload)
                logger.info(
                    "Sent to ground station: cls=%s conf=%.2f lat=%.6f lon=%.6f",
                    det.class_id,
                    det.confidence,
                    det.lat,
                    det.lon,
                )
            except Exception:  # noqa: BLE001
                logger.exception("Failed to transmit detection")

    def _publish_visualizer(self, det: GeoDetection) -> None:
        if not self._visualizer:
            return
        if self._loop:
            asyncio.run_coroutine_threadsafe(self._visualizer.publish_object(det), self._loop)
        else:
            logger.debug("Visualizer adapter provided without event loop; skipping publish_object.")

    def _collect_stable_confirmations(self, detections: Sequence[GeoDetection]) -> list[GeoDetection]:
        if not detections:
            return []
        confirmed: list[GeoDetection] = []
        with self._lock:
            for det in detections:
                updates = self._target_tracker.update(
                    [
                        TargetObservation(
                            class_label=str(det.class_id),
                            lat=det.lat,
                            lon=det.lon,
                            timestamp=det.timestamp,
                            confidence=det.confidence,
                        )
                    ]
                )
                if not updates or not updates[0].should_confirm:
                    continue
                update = updates[0]
                confirmed.append(
                    GeoDetection(
                        class_id=det.class_id,
                        confidence=det.confidence,
                        lat=update.track.lat,
                        lon=update.track.lon,
                        alt=det.alt,
                        timestamp=det.timestamp,
                        frame_id=det.frame_id,
                        track_id=update.track.track_id,
                    )
                )
        return confirmed
