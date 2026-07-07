from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from vehicle_damage_pipeline.vision.formatting import build_prediction_response


CLASS_NAMES = {
    0: "dent",
    1: "scratch",
    2: "crack",
    3: "glass shatter",
    4: "lamp broken",
    5: "tire flat",
}


def _extract_detections(result: Any) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    if boxes is None:
        return detections
    xyxy = boxes.xyxy.cpu().tolist() if getattr(boxes, "xyxy", None) is not None else []
    confs = boxes.conf.cpu().tolist() if getattr(boxes, "conf", None) is not None else []
    classes = boxes.cls.cpu().tolist() if getattr(boxes, "cls", None) is not None else []
    polygons = []
    if masks is not None and getattr(masks, "xy", None) is not None:
        polygons = [
            [[float(x), float(y)] for x, y in polygon.tolist()]
            for polygon in masks.xy
        ]
    for index, box in enumerate(xyxy):
        class_id = int(classes[index])
        detections.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES.get(class_id, str(class_id)),
                "confidence": float(confs[index]),
                "bbox_xyxy": [float(v) for v in box],
                "mask_polygon": polygons[index] if index < len(polygons) else None,
            }
        )
    return detections


def predict_images(
    *,
    weights: str | Path,
    source: str | Path | list[str | Path],
    output_dir: str | Path,
    imgsz: int = 1024,
    conf: float = 0.25,
    iou: float = 0.7,
) -> list[dict[str, Any]]:
    from ultralytics import YOLO

    weights = Path(weights)
    output_dir = Path(output_dir)
    if not weights.exists():
        raise FileNotFoundError(f"Missing weights: {weights}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    source_arg = [str(p) for p in source] if isinstance(source, list) else str(source)
    started = time.perf_counter()
    results = model.predict(
        source=source_arg,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
    )
    latency_ms = (time.perf_counter() - started) * 1000 / max(len(results), 1)
    responses = []
    for result in results:
        image_path = Path(getattr(result, "path", "image"))
        responses.append(
            build_prediction_response(
                image_name=image_path.name,
                detections=_extract_detections(result),
                latency_ms=latency_ms,
                model_version=weights.stem,
            )
        )
    (output_dir / "prediction_summary.json").write_text(
        json.dumps(responses, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return responses


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO vehicle damage prediction.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    args = parser.parse_args()
    result = predict_images(
        weights=args.weights,
        source=args.source,
        output_dir=args.output,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
