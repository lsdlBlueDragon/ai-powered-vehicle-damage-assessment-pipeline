# CarDD Resume Interview Next Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将已完成的 CarDD YOLO11n-seg 训练与 demo 结果整理成可投递简历、可面试讲解、可继续扩展 QLoRA 报告模块的完整交付。

**Architecture:** 当前仓库保留轻量 notebook、配置和文档，Google Drive 保存 CarDD 数据、训练权重、评估指标和 demo 输出。下一阶段不改动视觉训练主流程，重点固化证据、补齐中文材料、运行 QLoRA 报告模块，并把风险点整理成可解释的面试答案。

**Tech Stack:** Python, Ultralytics YOLO11, PyTorch, OpenCV, COCO, YOLO Segmentation, Google Colab, Google Drive, ONNX, Qwen2.5-7B-Instruct, PEFT, QLoRA, bitsandbytes, Markdown.

---

### Task 1: 冻结 YOLO 结果证据

**Files:**
- Read: `CarDD_YOLO11/reports/yolo11n_seg_test_metrics.json` in Google Drive
- Read: `CarDD_YOLO11/runs/train/yolo11n_seg/results.csv` in Google Drive
- Read: `CarDD_YOLO11/runs/predict/demo/` in Google Drive
- Modify: `docs/results_summary.md`

- [ ] **Step 1: 在 Drive 中确认核心产物存在**

在 Colab 或 Drive 文件面板确认：

```text
CarDD_YOLO11/runs/train/yolo11n_seg/weights/best.pt
CarDD_YOLO11/runs/train/yolo11n_seg/weights/last.pt
CarDD_YOLO11/runs/train/yolo11n_seg/results.csv
CarDD_YOLO11/exports/best.onnx
CarDD_YOLO11/reports/yolo11n_seg_test_metrics.json
CarDD_YOLO11/reports/yolo11n_seg_run_summary.json
CarDD_YOLO11/runs/predict/demo/
```

Expected: all paths exist.

- [ ] **Step 2: 对照指标文档**

打开 `docs/results_summary.md`，确认测试指标为：

```text
Box mAP50: 0.644
Box mAP50-95: 0.488
Mask mAP50: 0.638
Mask mAP50-95: 0.473
```

Expected: 与 Drive JSON 一致。

- [ ] **Step 3: 选择展示图**

从 `CarDD_YOLO11/runs/predict/demo/` 选择 4 到 8 张包含不同类别和不同难度的可视化图。

Expected: 至少包含 scratch、dent、lamp broken、crack 中的 3 类。

### Task 2: 修正文档展示瑕疵

**Files:**
- Modify: `docs/drive_layout.md`
- Modify: `docs/colab_operation_guide.md` if Drive layout text changes

- [ ] **Step 1: 修复 `docs/drive_layout.md` 目录树乱码**

将乱码目录树替换为：

```text
CarDD_YOLO11/
|-- data_raw/
|   `-- CarDD_release.zip
|-- data_coco/
|   `-- extracted-or-prepared-coco-style-cardd
|-- data_yolo/
|   |-- images/
|   |-- labels/
|   |-- cardd.yaml
|   `-- data_ready.json
|-- runs/
|   |-- train/
|   |-- val/
|   `-- predict/
|-- exports/
|-- reports/
`-- backups/
```

Expected: Markdown 预览中不再出现乱码字符。

- [ ] **Step 2: 检查隐私清单**

打开 `docs/privacy_checklist.md`，确认仓库没有：

```text
CarDD_release.zip
data_yolo/
*.pt
*.onnx
llm_adapters/
private Drive links
tokens
```

Expected: GitHub 只保留轻量文件。

### Task 3: 完成 Qwen7B QLoRA 报告模块

**Files:**
- Run: `notebooks/03_finetune_qwen7b_report_lora.ipynb`
- Run: `notebooks/04_generate_llm_report_qwen7b.ipynb`
- Modify: `notebooks/04_generate_llm_report_qwen7b.ipynb` if Chinese output is required

- [ ] **Step 1: 运行 notebook 03**

Colab Runtime:

```text
GPU, L4 or better
```

Expected outputs:

```text
CarDD_YOLO11/llm_adapters/qwen2_5_7b_cardd_report_lora/
CarDD_YOLO11/reports/qwen7b_report_sft_data.jsonl
CarDD_YOLO11/reports/qwen7b_report_sft_metadata.json
```

- [ ] **Step 2: 如需中文报告，修改 notebook 04**

将：

```python
REPORT_LANGUAGE = 'English'
```

改为：

```python
REPORT_LANGUAGE = 'Chinese'
```

Expected: 最终报告以中文生成。

- [ ] **Step 3: 运行 notebook 04**

Expected outputs:

```text
CarDD_YOLO11/reports/qwen7b_report_context.json
CarDD_YOLO11/reports/qwen7b_final_report.md
```

- [ ] **Step 4: 人工校验报告**

检查 `qwen7b_final_report.md`：

```text
不得编造指标。
不得声称 SOTA。
必须说明大文件保存在 Google Drive。
必须包含 limitations and future work。
```

Expected: 报告内容与真实 JSON、CSV、demo 输出一致。

### Task 4: 准备简历与面试材料

**Files:**
- Read: `docs/cardd_resume_interview_report_zh.md`
- Create outside repository if desired: resume/project-slides/interview-notes

- [ ] **Step 1: 采用简历 bullet**

优先使用：

```text
复现 CarDD 车辆损伤实例分割任务，基于 YOLO11n-seg 完成 COCO 到 YOLO segmentation 转换、Colab Drive 断点续训、测试评估、ONNX 导出和 demo 推理，测试集达到 box mAP50 0.644、mask mAP50 0.638。
```

Expected: 简历中出现模型、任务、工程闭环和量化指标。

- [ ] **Step 2: 准备 60 秒讲稿**

使用 `docs/cardd_resume_interview_report_zh.md` 中的 60 秒项目介绍。

Expected: 能在 60 秒内讲清楚任务、方法、结果和工程亮点。

- [ ] **Step 3: 准备风险回答**

至少准备以下问题：

```text
为什么选 YOLO11n-seg？
为什么 crack 类效果差？
mAP50 和 mAP50-95 区别是什么？
数据转换有什么风险？
Qwen7B QLoRA 模块的价值是什么？
```

Expected: 每个问题都有 30 到 60 秒答案。

### Task 5: 可选精度增强实验

**Files:**
- Modify: `notebooks/01_train_cardd_yolo11_seg.ipynb` if a new experiment is started
- Modify: `docs/results_summary.md` after the new experiment completes

- [ ] **Step 1: 复制实验 run name**

将 notebook 01 中：

```python
BASE_MODEL = 'yolo11n-seg.pt'
RUN_NAME = 'yolo11n_seg'
```

改为：

```python
BASE_MODEL = 'yolo11s-seg.pt'
RUN_NAME = 'yolo11s_seg'
```

Expected: 新实验不会覆盖当前 `yolo11n_seg` 结果。

- [ ] **Step 2: 保持相同评估流程**

继续使用：

```text
imgsz: 1024
batch: 7 or lower if OOM
split: test when available
```

Expected: 可以和 YOLO11n-seg 公平对比。

- [ ] **Step 3: 更新结果摘要**

在 `docs/results_summary.md` 添加新模型结果表。

Expected: 至少记录 box mAP50、box mAP50-95、mask mAP50、mask mAP50-95、训练时长、GPU。

