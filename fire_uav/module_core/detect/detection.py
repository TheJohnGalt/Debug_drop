"""
Обёртка над Ultralytics-YOLO v8: инференс и приведение результатов
к pydantic-моделям проекта.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Sequence, cast

import numpy as np
from numpy.typing import NDArray

from fire_uav.config.settings import settings
from fire_uav.domain.video.camera import CameraParams
from fire_uav.module_core.schema import Detection, DetectionsBatch, FrameMeta

if TYPE_CHECKING:
    from ultralytics.engine.results import Results

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from ultralytics import YOLO as _YOLO
except ImportError:  # pragma: no cover
    _YOLO = None

YOLO: Any | None = _YOLO

_log = logging.getLogger(__name__)


class DetectionEngine:
    """YOLO-детектор, возвращающий pydantic-объекты."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        wanted_classes: Sequence[int] | None = None,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
        device: str | None = None,
    ) -> None:
        if YOLO is None:  # pragma: no cover
            raise RuntimeError("Install `ultralytics` to use DetectionEngine")
        if torch is None:  # pragma: no cover
            raise RuntimeError("Install `torch` to use DetectionEngine")

        model_path = model_path or settings.yolo_model
        conf_threshold = conf_threshold or settings.yolo_conf
        iou_threshold = iou_threshold or settings.yolo_iou
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._yolo = YOLO(str(model_path))
        self._yolo.overrides |= {
            "conf": conf_threshold,
            "iou": iou_threshold,
            "device": device,
        }
        self._wanted = set(wanted_classes or settings.yolo_classes)
        _log.info(
            "YOLO %s loaded (device=%s, conf=%.2f, classes=%s)",
            model_path,
            device,
            conf_threshold,
            sorted(self._wanted) if self._wanted else "ALL",
        )

    # ------------------------------------------------------------------ #
    def infer(
        self,
        frame_bgr: NDArray[np.uint8],
        *,
        camera_id: str = "cam0",
        cam_params: CameraParams | None = None,  # резерв
        return_batch: bool = False,
    ) -> List[Detection] | DetectionsBatch:
        """Запуск модели и упаковка результата в pydantic-модели."""
        h, w = frame_bgr.shape[:2]
        results = cast(List["Results"], self._yolo(frame_bgr, verbose=False))

        detections: List[Detection] = []
        for r in results:
            for cls, conf, xyxy in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy):
                cls_id = int(cls)
                if self._wanted and cls_id not in self._wanted:
                    continue
                x1, y1, x2, y2 = map(int, xyxy)
                detections.append(
                    Detection(
                        camera_id=camera_id,
                        class_id=cls_id,
                        confidence=float(conf),
                        bbox=(x1, y1, x2, y2),
                    )
                )

        if return_batch:
            meta = FrameMeta(camera_id=camera_id, width=w, height=h)
            return DetectionsBatch(frame=meta, detections=detections)
        return detections
