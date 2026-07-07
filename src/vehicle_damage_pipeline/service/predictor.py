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


CLASS_REPORT_GUIDANCE = {
    "dent": {
        "zh": "凹陷",
        "meaning": "车身钣金区域可能存在受力变形或局部内凹。",
        "advice": "建议人工复核凹陷边界、漆面是否开裂，并评估钣金修复与补漆需求。",
    },
    "scratch": {
        "zh": "划痕",
        "meaning": "车漆表面可能存在线状或片状擦伤。",
        "advice": "建议检查划痕深度，区分表层抛光可修复与需要补漆的情况。",
    },
    "crack": {
        "zh": "裂纹",
        "meaning": "局部材料可能出现裂开、断裂或细长破损。",
        "advice": "建议人工确认裂纹是否贯穿结构件，并结合近距离图片复核。",
    },
    "glass shatter": {
        "zh": "玻璃破碎",
        "meaning": "车窗或玻璃区域可能存在破碎、裂片或大面积损坏。",
        "advice": "建议优先检查安全风险，并确认是否需要更换玻璃组件。",
    },
    "lamp broken": {
        "zh": "灯具破损",
        "meaning": "车灯区域可能存在外壳破裂、缺失或灯组损坏。",
        "advice": "建议检查灯具功能、密封性和外壳完整性。",
    },
    "tire flat": {
        "zh": "轮胎亏气",
        "meaning": "轮胎形态可能显示明显亏气或变形。",
        "advice": "建议检查胎压、胎壁损伤和是否存在扎钉漏气。",
    },
}


def _confidence_band(confidence: float) -> str:
    if confidence >= 0.75:
        return "高置信度"
    if confidence >= 0.5:
        return "中等置信度"
    return "较低置信度"


def generate_assessment_report(prediction: dict[str, Any]) -> str:
    detections = prediction.get("detections", [])
    if not detections:
        return (
            "未检测到高于阈值的车辆损伤。建议结合人工复核、更多角度图片和更高分辨率局部图像继续判断。"
        )

    lines = [
        f"检测到 {len(detections)} 处疑似车辆损伤。以下结论由视觉模型自动生成，适合作为初步筛查参考："
    ]
    for index, item in enumerate(detections, start=1):
        class_name = str(item.get("class_name", "unknown"))
        confidence = float(item.get("confidence", 0.0))
        bbox = item.get("bbox_xyxy", [])
        guidance = CLASS_REPORT_GUIDANCE.get(
            class_name,
            {
                "zh": class_name,
                "meaning": "图像中存在疑似异常损伤区域。",
                "advice": "建议人工结合原图和更多角度图片复核。",
            },
        )
        lines.extend(
            [
                "",
                f"{index}. {guidance['zh']}（{class_name}，{_confidence_band(confidence)}，confidence={confidence:.3f}）",
                f"   - 文字说明：{guidance['meaning']}",
                f"   - 建议动作：{guidance['advice']}",
                f"   - 结构化定位：bbox={bbox}",
            ]
        )
    lines.append("")
    lines.append("该输出仅用于辅助评估，不构成生产级保险定损结论。")
    return "\n".join(lines)
