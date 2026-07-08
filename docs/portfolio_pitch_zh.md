# 简历与面试项目包装

## 推荐项目名

```text
AI-Powered Vehicle Damage Assessment Pipeline
车辆损伤检测、实例分割与自动化报告生成系统
```

## 一句话

基于 CarDD 和 YOLO11n-seg 构建车辆损伤检测与实例分割工程管线，输出结构化预测 JSON，并接入 Qwen2.5-7B QLoRA 报告模块与 RAG/LLM 评估，形成从图像到报告的 AI 工程闭环。

## 简历 Bullet

- 构建车辆损伤检测与自动化报告生成系统，基于 YOLO11n-seg 完成 CarDD 数据集实例分割训练、ONNX 导出与推理服务封装，测试集达到 box mAP50 0.6746、mask mAP50 0.6712。
- 将 Colab notebook 改造为工程化 CLI runbook，支持数据转换、训练、推理、报告上下文构建和 LLM 评估的 `python -m` 调用。
- 设计 FastAPI 与 Gradio 推理入口，输出损伤类别、置信度、bbox、mask polygon、推理耗时和结构化报告文本。
- 建立轻量 RAG/LLM 评估体系，检查报告指标一致性、检索覆盖、章节完整性和夸大声明，降低 LLM 报告幻觉风险。

## 面试定位

这个项目重点不是 SOTA，而是 AI 工程能力：数据处理、训练复现、推理封装、结构化输出、评估、文档化和边界意识。
