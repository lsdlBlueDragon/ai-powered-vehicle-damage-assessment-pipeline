from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Callable

from vehicle_damage_pipeline.llm.adapter import adapter_status
from vehicle_damage_pipeline.llm.qwen_reporter import DEFAULT_QWEN_MODEL_ID, build_assessment_report_messages, generate_qwen_text
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
        "meaning": "模型检测到疑似凹陷类别候选。",
    },
    "scratch": {
        "zh": "划痕",
        "meaning": "模型检测到疑似划痕类别候选。",
    },
    "crack": {
        "zh": "裂纹",
        "meaning": "模型检测到疑似裂纹类别候选。",
    },
    "glass shatter": {
        "zh": "玻璃破碎",
        "meaning": "模型检测到疑似玻璃破碎类别候选。",
    },
    "lamp broken": {
        "zh": "灯具破损",
        "meaning": "模型检测到疑似灯具破损类别候选。",
    },
    "tire flat": {
        "zh": "轮胎亏气",
        "meaning": "模型检测到疑似轮胎亏气类别候选。",
    },
}

ASSESSMENT_FORBIDDEN_PHRASES = [
    "车身左侧",
    "车身右侧",
    "左侧中部",
    "右侧中部",
    "车头",
    "车尾",
    "从车头延伸至车尾",
    "面积较大",
    "面积较小",
    "严重",
    "轻微",
    "钣金",
    "补漆",
    "维修",
    "更换",
    "费用",
    "理赔",
    "责任",
    "无需复核",
]


def _confidence_band(confidence: float) -> str:
    if confidence >= 0.75:
        return "高置信度"
    if confidence >= 0.5:
        return "中等置信度"
    return "较低置信度"


def generate_template_assessment_report(prediction: dict[str, Any]) -> str:
    detections = prediction.get("detections", [])
    if not detections:
        return "当前阈值下未检测到高于阈值的高置信度车辆损伤候选。建议人工复核，并结合更多角度图片和更高分辨率局部图像继续判断。"

    lines = [f"检测到 {len(detections)} 处疑似车辆损伤。以下结论由视觉模型自动生成，适合作为初步筛查参考："]
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
                "   - 人工复核：建议人工复核，并结合原图、更多角度图片和局部高清图确认该候选。",
                f"   - 结构化定位：bbox={bbox}",
            ]
        )
    lines.append("")
    lines.append("该输出仅用于辅助筛查和人工复核前参考。")
    return "\n".join(lines)


def evaluate_assessment_report(prediction: dict[str, Any], report: str) -> dict[str, Any]:
    text = report.lower()
    forbidden = [phrase for phrase in ASSESSMENT_FORBIDDEN_PHRASES if phrase.lower() in text]
    detections = prediction.get("detections", [])
    class_results = []
    confidence_results = []
    bbox_results = []
    if not detections:
        has_review = "人工复核" in report
        return {
            "required_fields": {
                "review_advice": has_review,
                "classes": [],
                "confidences": [],
                "bboxes": [],
            },
            "forbidden_phrases": forbidden,
            "passed": has_review and not forbidden,
        }
    for item in detections:
        class_name = str(item.get("class_name", ""))
        guidance = CLASS_REPORT_GUIDANCE.get(class_name, {"zh": class_name})
        confidence = float(item.get("confidence", 0.0))
        bbox = item.get("bbox_xyxy", [])
        class_results.append(class_name in report or str(guidance["zh"]) in report)
        confidence_results.append(f"{confidence:.3f}" in report or f"{confidence:.2f}" in report)
        bbox_results.append(str(bbox) in report)
    return {
        "required_fields": {
            "review_advice": "人工复核" in report,
            "classes": class_results,
            "confidences": confidence_results,
            "bboxes": bbox_results,
        },
        "forbidden_phrases": forbidden,
        "passed": all(class_results)
        and all(confidence_results)
        and all(bbox_results)
        and "人工复核" in report
        and not forbidden,
    }


def _normalize_report_backend(backend: str) -> str:
    normalized = backend.lower().strip()
    if normalized in {"template", "deterministic", "no-qwen"}:
        return "template"
    if normalized in {"qwen", "qwen_adapter", "qwen-adapter"}:
        return "qwen"
    raise ValueError(f"Unsupported assessment report backend: {backend}")


def generate_assessment_report(
    prediction: dict[str, Any],
    *,
    backend: str = "qwen",
    adapter_dir: str | Path | None = None,
    model_id: str = DEFAULT_QWEN_MODEL_ID,
    language: str = "Chinese",
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    load_in_4bit: bool = True,
    fallback_to_template: bool = True,
    qwen_generate_fn: Callable[..., str] | None = None,
) -> str:
    selected_backend = _normalize_report_backend(backend)
    if selected_backend == "template":
        return generate_template_assessment_report(prediction)

    status = adapter_status(adapter_dir) if adapter_dir else {"complete": False, "missing_files": ["adapter_dir"]}
    if not status["complete"]:
        if fallback_to_template:
            return generate_template_assessment_report(prediction)
        raise FileNotFoundError(f"Qwen assessment adapter is incomplete: {status}")

    messages = build_assessment_report_messages(prediction, language=language)
    generator = qwen_generate_fn or generate_qwen_text
    try:
        text = generator(
            messages=messages,
            model_id=model_id,
            adapter_dir=adapter_dir,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            load_in_4bit=load_in_4bit,
        ).strip()
    except Exception:
        if fallback_to_template:
            return generate_template_assessment_report(prediction)
        raise
    if text and evaluate_assessment_report(prediction, text)["passed"]:
        return text
    if fallback_to_template:
        return generate_template_assessment_report(prediction)
    return text
