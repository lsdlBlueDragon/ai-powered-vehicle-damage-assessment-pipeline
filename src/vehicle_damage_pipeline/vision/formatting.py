from __future__ import annotations

from typing import Any, Iterable


def _round_float(value: Any, digits: int) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    return value


def _round_sequence(values: Iterable[Any], digits: int) -> list[Any]:
    return [_round_float(value, digits) for value in values]


def normalize_detection(detection: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "class_id": int(detection["class_id"]),
        "class_name": str(detection["class_name"]),
        "confidence": round(float(detection["confidence"]), 3),
        "bbox_xyxy": _round_sequence(detection.get("bbox_xyxy", []), 1),
    }
    polygon = detection.get("mask_polygon")
    if polygon is not None:
        normalized["mask_polygon"] = [
            _round_sequence(point, 1) for point in polygon
        ]
    return normalized


def summarize_detections(detections: list[dict[str, Any]]) -> str:
    if not detections:
        return "No vehicle damage detections above threshold."
    counts: dict[str, int] = {}
    for detection in detections:
        label = str(detection["class_name"])
        counts[label] = counts.get(label, 0) + 1
    return ", ".join(f"{count} {label}" for label, count in sorted(counts.items()))


def build_prediction_response(
    *,
    image_name: str,
    detections: list[dict[str, Any]],
    latency_ms: float,
    model_version: str,
) -> dict[str, Any]:
    normalized = [normalize_detection(detection) for detection in detections]
    return {
        "image_name": image_name,
        "model_version": model_version,
        "latency_ms": round(float(latency_ms), 2),
        "detections": normalized,
        "summary": summarize_detections(normalized),
    }
