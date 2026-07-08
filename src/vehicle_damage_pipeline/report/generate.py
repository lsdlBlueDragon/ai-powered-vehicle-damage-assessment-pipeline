from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from vehicle_damage_pipeline.eval.llm_eval import evaluate_report
from vehicle_damage_pipeline.llm.adapter import adapter_status, is_report_adapter_complete, report_adapter_dir
from vehicle_damage_pipeline.llm.qwen_reporter import DEFAULT_QWEN_MODEL_ID, build_project_report_messages, generate_qwen_text


@dataclass(frozen=True)
class GeneratedReport:
    text: str
    metadata: dict[str, Any]


def _metric(context: dict[str, Any], key: str) -> float:
    return float(context["test_metrics"][key])


def generate_template_report(context: dict[str, Any], language: str = "Chinese") -> str:
    box_map50 = _metric(context, "metrics/mAP50(B)")
    box_map = _metric(context, "metrics/mAP50-95(B)")
    mask_map50 = _metric(context, "metrics/mAP50(M)")
    mask_map = _metric(context, "metrics/mAP50-95(M)")
    if language.lower().startswith("en"):
        return (
            "# AI-Powered Vehicle Damage Assessment Pipeline\n\n"
            "## Project Overview\n"
            "This project builds an engineering pipeline for vehicle damage detection, instance segmentation, "
            "structured prediction output, and grounded report generation.\n\n"
            "## Dataset And Task\n"
            "The vision task uses CarDD-style vehicle damage images and focuses on detection plus instance segmentation.\n\n"
            "## Method\n"
            "The detector is YOLO11n-seg. The report layer uses structured experiment context and can run either a "
            "Qwen2.5-7B LoRA adapter or this deterministic template fallback.\n\n"
            "## Results\n"
            f"On the test split, box mAP50 is {box_map50:.3f}, box mAP50-95 is {box_map:.3f}, "
            f"mask mAP50 is {mask_map50:.3f}, and mask mAP50-95 is {mask_map:.3f}.\n\n"
            "## RAG/LLM Evaluation\n"
            "The report is checked for metric grounding, required section coverage, and forbidden exaggerated claims.\n\n"
            "## Limitations And Next Steps\n"
            "This is not a production-ready insurance assessment system and does not claim SOTA performance.\n"
        )
    return (
        "# AI-Powered Vehicle Damage Assessment Pipeline\n\n"
        "## 项目概览\n"
        "本项目构建车辆损伤检测、实例分割、结构化推理输出和可信报告生成的 AI 工程闭环。"
        "主训练流程在 Colab 运行，大文件保存在 Google Drive，GitHub 保留代码、配置、文档和可复现实验入口。\n\n"
        "## 数据与任务\n"
        "数据集为 CarDD 风格车辆损伤图像，任务包含损伤目标检测与实例分割，类别包括 dent、scratch、crack、"
        "glass shatter、lamp broken 和 tire flat。\n\n"
        "## 方法\n"
        "视觉模型采用 YOLO11n-seg，输出检测框、类别、置信度和 mask。报告层默认使用 Qwen2.5-7B-Instruct 的 "
        "LoRA adapter 生成，也支持 `--no-qwen` 切换到确定性模板，便于低显存或离线演示。\n\n"
        "## 测试结果\n"
        f"测试集 box mAP50 为 {box_map50:.3f}，box mAP50-95 为 {box_map:.3f}；"
        f"mask mAP50 为 {mask_map50:.3f}，mask mAP50-95 为 {mask_map:.3f}。\n\n"
        "## RAG/LLM 评估\n"
        "报告评估关注检索覆盖、指标是否与 context JSON 一致、章节覆盖率，以及是否出现 SOTA 或生产级保险定损等夸大声明。\n\n"
        "## 局限性与下一步\n"
        "该项目不是生产级保险定损系统，也不声明 SOTA。后续优化重点包括更强的 per-class 评估、更多真实图片测试、"
        "Qwen 单图报告质量评估，以及服务端显存管理。\n"
    )


generate_deterministic_report = generate_template_report


def _normalize_backend(backend: str) -> str:
    normalized = backend.lower().strip()
    if normalized in {"template", "deterministic", "no-qwen"}:
        return "template"
    if normalized in {"qwen", "qwen_adapter", "qwen-adapter"}:
        return "qwen"
    raise ValueError(f"Unsupported report backend: {backend}")


def generate_report_from_context(
    context: dict[str, Any],
    *,
    language: str = "Chinese",
    backend: str = "qwen",
    adapter_dir: str | Path | None = None,
    drive_root: str | Path | None = None,
    model_id: str = DEFAULT_QWEN_MODEL_ID,
    max_new_tokens: int = 900,
    temperature: float = 0.2,
    load_in_4bit: bool = True,
    fallback_to_template: bool = True,
    qwen_generate_fn: Callable[..., str] | None = None,
) -> GeneratedReport:
    selected_backend = _normalize_backend(backend)
    if selected_backend == "template":
        return GeneratedReport(
            text=generate_template_report(context, language=language),
            metadata={"backend": "template", "requested_backend": backend, "language": language},
        )

    resolved_adapter = Path(adapter_dir) if adapter_dir else None
    if resolved_adapter is None and drive_root is not None:
        resolved_adapter = report_adapter_dir(drive_root)
    status = adapter_status(resolved_adapter) if resolved_adapter else {"complete": False, "missing_files": ["adapter_dir"]}

    if not status["complete"]:
        if not fallback_to_template:
            raise FileNotFoundError(f"Qwen report adapter is incomplete: {status}")
        return GeneratedReport(
            text=generate_template_report(context, language=language),
            metadata={
                "backend": "template",
                "requested_backend": "qwen",
                "fallback_reason": "qwen_adapter_incomplete",
                "adapter_status": status,
                "language": language,
            },
        )

    messages = build_project_report_messages(context, language=language)
    generator = qwen_generate_fn or generate_qwen_text
    text = generator(
        messages=messages,
        model_id=model_id,
        adapter_dir=resolved_adapter,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        load_in_4bit=load_in_4bit,
    )
    validation = evaluate_report(context, text)
    if not validation["passed"]:
        if not fallback_to_template:
            raise ValueError(f"Qwen report failed validation: {validation}")
        return GeneratedReport(
            text=generate_template_report(context, language=language),
            metadata={
                "backend": "template",
                "requested_backend": "qwen",
                "fallback_reason": "qwen_report_validation_failed",
                "qwen_backend": "qwen_adapter",
                "adapter_dir": str(resolved_adapter),
                "adapter_complete": is_report_adapter_complete(resolved_adapter),
                "model_id": model_id,
                "language": language,
                "qwen_validation": validation,
            },
        )
    return GeneratedReport(
        text=text,
        metadata={
            "backend": "qwen_adapter",
            "requested_backend": "qwen",
            "adapter_dir": str(resolved_adapter),
            "adapter_complete": is_report_adapter_complete(resolved_adapter),
            "model_id": model_id,
            "language": language,
            "qwen_validation": validation,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a grounded project report from context JSON.")
    parser.add_argument("--context", required=True)
    parser.add_argument("--output")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--backend", choices=["qwen", "template"], default="qwen")
    parser.add_argument("--no-qwen", action="store_true", help="Use the deterministic template report backend.")
    parser.add_argument("--adapter-dir")
    parser.add_argument("--drive-root")
    parser.add_argument("--model-id", default=DEFAULT_QWEN_MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=900)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--no-template-fallback", action="store_true")
    parser.add_argument("--metadata-output")
    args = parser.parse_args()

    context_path = Path(args.context)
    context = json.loads(context_path.read_text(encoding="utf-8"))
    backend = "template" if args.no_qwen else args.backend
    drive_root = args.drive_root
    if drive_root is None and context_path.parent.name == "reports":
        drive_root = str(context_path.parent.parent)
    result = generate_report_from_context(
        context,
        language=args.language,
        backend=backend,
        adapter_dir=args.adapter_dir,
        drive_root=drive_root,
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        load_in_4bit=not args.no_4bit,
        fallback_to_template=not args.no_template_fallback,
    )
    output = Path(args.output) if args.output else context_path.parent / "qwen7b_final_report.md"
    output.write_text(result.text, encoding="utf-8")
    metadata_output = Path(args.metadata_output) if args.metadata_output else output.with_suffix(".metadata.json")
    metadata_output.write_text(json.dumps(result.metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Report saved:", output)
    print("Report metadata saved:", metadata_output)
    print(json.dumps(result.metadata, ensure_ascii=False, indent=2))
    print(result.text)


if __name__ == "__main__":
    main()
