from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _metric(context: dict[str, Any], key: str) -> float:
    return float(context["test_metrics"][key])


def generate_deterministic_report(context: dict[str, Any], language: str = "Chinese") -> str:
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
            "## Results\n"
            f"On the test split, box mAP50 is {box_map50:.3f}, box mAP50-95 is {box_map:.3f}, "
            f"mask mAP50 is {mask_map50:.3f}, and mask mAP50-95 is {mask_map:.3f}.\n\n"
            "## Limitations\n"
            "This is not a production-ready insurance assessment system and does not claim SOTA performance.\n"
        )
    return (
        "# AI-Powered Vehicle Damage Assessment Pipeline\n\n"
        "## 项目概览\n"
        "本项目构建车辆损伤检测、实例分割、结构化预测输出和可信报告生成的 AI 工程闭环。"
        "主训练流程在 Colab 运行，大文件保存在 Google Drive，GitHub 仅保留代码、配置和文档。\n\n"
        "## 数据集与任务\n"
        "数据集为 CarDD，任务包含车辆损伤检测与实例分割，类别包括 dent、scratch、crack、"
        "glass shatter、lamp broken 和 tire flat。\n\n"
        "## 方法\n"
        "视觉模型采用 YOLO11n-seg，输出检测框、类别、置信度和 mask。报告模块使用结构化 context "
        "约束生成内容，并通过评估脚本检查指标一致性和夸大声明。\n\n"
        "## 结果\n"
        f"测试集 box mAP50 为 {box_map50:.3f}，box mAP50-95 为 {box_map:.3f}；"
        f"mask mAP50 为 {mask_map50:.3f}，mask mAP50-95 为 {mask_map:.3f}。\n\n"
        "## 局限性\n"
        "该项目不是生产级保险定损系统，也不声明 SOTA。当前 crack、scratch、dent 等细粒度损伤仍是主要优化方向。\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a grounded project report from context JSON.")
    parser.add_argument("--context", required=True)
    parser.add_argument("--output")
    parser.add_argument("--language", default="Chinese")
    args = parser.parse_args()
    context_path = Path(args.context)
    context = json.loads(context_path.read_text(encoding="utf-8"))
    report = generate_deterministic_report(context, language=args.language)
    output = Path(args.output) if args.output else context_path.parent / "qwen7b_final_report.md"
    output.write_text(report, encoding="utf-8")
    print("Report saved:", output)
    print(report)


if __name__ == "__main__":
    main()
