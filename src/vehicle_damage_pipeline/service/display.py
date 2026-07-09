from __future__ import annotations

import json
from typing import Any

DETECTION_TABLE_HEADERS = ["class_name", "confidence", "bbox_xyxy", "mask_points"]


def _mask_point_count(detection: dict[str, Any]) -> int:
    polygon = detection.get("mask_polygon")
    return len(polygon) if isinstance(polygon, list) else 0


def build_detection_table(prediction: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for detection in prediction.get("detections", []):
        rows.append(
            [
                str(detection.get("class_name", "unknown")),
                float(detection.get("confidence", 0.0)),
                str(detection.get("bbox_xyxy", [])),
                _mask_point_count(detection),
            ]
        )
    return rows


def build_public_prediction_summary(prediction: dict[str, Any]) -> dict[str, Any]:
    public_detections = []
    for detection in prediction.get("detections", []):
        public_detections.append(
            {
                "class_id": detection.get("class_id"),
                "class_name": detection.get("class_name"),
                "confidence": detection.get("confidence"),
                "bbox_xyxy": detection.get("bbox_xyxy", []),
                "mask_points": _mask_point_count(detection),
            }
        )
    return {
        "image_name": prediction.get("image_name"),
        "model_version": prediction.get("model_version"),
        "latency_ms": prediction.get("latency_ms"),
        "summary": prediction.get("summary"),
        "detections": public_detections,
    }


def build_debug_prediction_json(prediction: dict[str, Any]) -> str:
    return json.dumps(prediction, ensure_ascii=False, indent=2)
