from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from vehicle_damage_pipeline.vision.predict import predict_images


class DamagePredictor:
    def __init__(self, weights: str | Path | None = None, output_dir: str | Path = "runs/service_predict"):
        self.weights = Path(weights) if weights else None
        self.output_dir = Path(output_dir)

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok" if self.weights and self.weights.exists() else "missing_weights",
            "weights": str(self.weights) if self.weights else None,
            "output_dir": str(self.output_dir),
        }

    def predict_file(self, image_path: str | Path) -> dict[str, Any]:
        if not self.weights or not self.weights.exists():
            raise FileNotFoundError("YOLO weights are not configured or missing.")
        results = predict_images(weights=self.weights, source=image_path, output_dir=self.output_dir)
        return results[0] if results else {}

    def predict_bytes(self, image_bytes: bytes, suffix: str = ".jpg") -> dict[str, Any]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(image_bytes)
            path = Path(handle.name)
        try:
            return self.predict_file(path)
        finally:
            try:
                path.unlink()
            except OSError:
                pass


def generate_assessment_report(prediction: dict[str, Any]) -> str:
    detections = prediction.get("detections", [])
    if not detections:
        return "未检测到高于阈值的车辆损伤；建议结合人工复核和更多视角图片。"
    lines = ["车辆损伤自动检测摘要："]
    for item in detections:
        lines.append(
            f"- {item['class_name']}: confidence={item['confidence']:.3f}, bbox={item['bbox_xyxy']}"
        )
    lines.append("该输出仅用于辅助评估，不构成生产级保险定损结论。")
    return "\n".join(lines)
