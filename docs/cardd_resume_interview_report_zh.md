# CarDD YOLO11 Segmentation 项目中文报告与操作指南

更新时间：2026-07-08

本报告面向简历、面试和项目收尾展示，基于当前仓库文件、现有文档和 Colab runbook 的实际结构整理。当前主流程包括：

- `notebooks/01_train_cardd_yolo11_seg.ipynb`
- `notebooks/02_demo_cardd_yolo11_seg.ipynb`
- `notebooks/03_finetune_qwen7b_report_lora.ipynb`
- `notebooks/04_generate_llm_report_qwen7b.ipynb`
- `notebooks/06_colab_qwen_report_eval_full_workflow.ipynb`

需要注意：仓库内的 notebook 是干净模板版，`execution_count` 和输出区均为空。真实训练结果、模型权重、指标 JSON、ONNX、demo 可视化图应以 Google Drive 下 `CarDD_YOLO11/` 的产物为准。

## 1. 项目定位

项目名称建议：

```text
CarDD 车辆损伤检测与实例分割复现项目
```

一句话概述：

```text
基于 Ultralytics YOLO11n-seg 在 CarDD 数据集上完成车辆损伤检测与实例分割复现，覆盖 COCO 标注到 YOLO segmentation 格式转换、Colab Drive 持久化训练、断点续训、测试集评估、ONNX 导出和 demo 推理展示。
```

简历定位：

- 这是一个完整的计算机视觉工程复现项目，不只是跑通模型。
- 核心亮点是从数据转换、训练、评估、可视化、部署导出到实验报告生成的端到端闭环。
- 结果可量化：YOLO11n-seg 训练 100 epochs，测试集 box mAP50 为 0.6746，mask mAP50 为 0.6712。
- 工程可靠性可讲：Google Drive 持久化产物、`last.pt` 断点续训、数据转换幂等标记、GitHub 只保留轻量代码和文档。

## 2. 当前项目状态

已完成：

- CarDD 数据集准备与 COCO 风格标注解析。
- COCO instance segmentation 标注转换为 YOLO segmentation label。
- YOLO11n-seg 训练 100 epochs。
- 测试集评估和指标保存。
- `best.pt`、`last.pt`、`best.onnx` 等产物保存到 Drive。
- demo 推理，可输出类别、检测框、置信度和 mask 可视化。
- README、复现计划、Colab 操作指南、结果摘要已初步整理。

待完成或可增强：

- 运行 `03_finetune_qwen7b_report_lora.ipynb`，生成 Qwen2.5-7B-Instruct 的 QLoRA adapter。
- 运行 `04_generate_llm_report_qwen7b.ipynb`，基于实验产物生成最终 Markdown 报告。
- 如需中文 LLM 报告，需要把 notebook 04 中的 `REPORT_LANGUAGE = 'English'` 改为 `REPORT_LANGUAGE = 'Chinese'`。
- 修正 `docs/drive_layout.md` 中目录树的乱码字符。
- 将最终 demo 图片、训练曲线、混淆矩阵或 PR 曲线整理成面试展示材料。

## 3. 仓库结构与职责

```text
.
|-- README.md
|-- requirements-colab.txt
|-- configs/
|   `-- cardd_yolo.yaml
|-- docs/
|   |-- colab_operation_guide.md
|   |-- drive_layout.md
|   |-- llm_report_module_plan.md
|   |-- privacy_checklist.md
|   |-- reproduction_plan.md
|   `-- results_summary.md
`-- notebooks/
    |-- 01_train_cardd_yolo11_seg.ipynb
    |-- 02_demo_cardd_yolo11_seg.ipynb
    |-- 03_finetune_qwen7b_report_lora.ipynb
    `-- 04_generate_llm_report_qwen7b.ipynb
```

本地 GitHub 仓库只存轻量文件。以下大文件应保留在 Google Drive，不进入 Git：

- CarDD 数据压缩包和解压图像。
- YOLO 转换后的 `data_yolo/`。
- 训练输出、权重、实验日志。
- ONNX 导出模型。
- Qwen LoRA adapter。
- 私人 Drive 链接、授权表、token 和个人信息。

## 4. 数据与标注处理技术细节

数据集：

- CarDD，面向车辆损伤检测与分割。
- 当前类别配置为 6 类：`dent`、`scratch`、`crack`、`glass shatter`、`lamp broken`、`tire flat`。

原始格式：

- notebook 01 会在 `data_raw/` 或 `data_coco/` 中查找 COCO 风格 JSON 标注。
- 通过文件名推断 split：包含 `train`、`val`、`valid`、`test` 的 JSON 分别映射到训练、验证、测试集。

转换逻辑：

- 根据 COCO `categories` 建立类别 ID 到 YOLO 类别索引的映射。
- 根据 `images` 和 `annotations` 建立 image_id 到 annotations 的映射。
- 对每个实例的 `segmentation` polygon 做归一化：
  - x 坐标除以图像宽度。
  - y 坐标除以图像高度。
  - 坐标裁剪到 `[0, 1]`。
- 输出 YOLO segmentation label：

```text
class_id x1 y1 x2 y2 x3 y3 ...
```

工程设计：

- 转换后的图片写入 `data_yolo/images/<split>/`。
- 转换后的 label 写入 `data_yolo/labels/<split>/`。
- `data_yolo/cardd.yaml` 作为 Ultralytics 数据配置。
- `data_yolo/data_ready.json` 标记数据集整体转换完成。
- `_split_ready/<split>.json` 标记每个 split 已转换，避免 Colab 中断后重复处理。

需要面试时主动说明的限制：

- 当前转换逻辑对一个 annotation 中的多个 polygon 取最长 polygon，即 `max(seg, key=len)`。这简化了 YOLO label 生成，但如果某个实例有多个分离区域，可能损失小区域信息。可作为未来改进点：保留多 polygon、合并 mask 或使用 RLE 到 polygon 的更完整转换。

## 5. 模型训练技术细节

默认模型：

```text
yolo11n-seg.pt
```

选择理由：

- `n` 版本小，适合 Colab L4 快速复现。
- 支持检测框和实例 mask 双输出。
- 更适合把项目做成完整闭环，而不是追求单点最高性能。

训练配置：

```text
epochs: 100
imgsz: 1024
batch: 7
optimizer: AdamW
amp: True
overlap_mask: True
multi_scale: True
mosaic: 1.0
mixup: 0.1
copy_paste: 0.1
patience: 30
save_period: 1
workers: 2
seed: 42
```

`batch=7` 的关键原因：

- 文档中记录训练集为 2817 张图片。
- 如果使用某些 batch size，最后一个 batch 可能只剩 1 张图。
- 分割训练中 BatchNorm 等模块可能触发 `Expected more than 1 value per channel when training`。
- 设置 `batch=7` 可以避免 singleton final batch，提高 Colab 训练稳定性。

断点续训策略：

- 如果 `runs/train/yolo11n_seg/weights/last.pt` 存在，notebook 会从 `last.pt` 加载权重。
- 默认 `STRICT_RESUME = False`，会加载 checkpoint 权重但使用当前 notebook 中更安全的训练参数。
- 如果设置 `STRICT_RESUME = True`，则使用 Ultralytics 的严格 resume 逻辑，可能继承旧的 batch 参数。

训练产物：

```text
CarDD_YOLO11/runs/train/yolo11n_seg/weights/best.pt
CarDD_YOLO11/runs/train/yolo11n_seg/weights/last.pt
CarDD_YOLO11/runs/train/yolo11n_seg/results.csv
CarDD_YOLO11/runs/train/yolo11n_seg/results.png
```

## 6. 评估、导出与 Demo 技术细节

评估逻辑：

- 优先使用 `best.pt`。
- 如果不存在 `best.pt`，回退到 `last.pt`。
- 如果 `data_yolo/images/test` 存在，则在 test split 上评估，否则在 val split 上评估。
- 评估结果写入：

```text
CarDD_YOLO11/reports/yolo11n_seg_test_metrics.json
```

导出逻辑：

- 使用 Ultralytics `model.export(format='onnx', imgsz=1024)`。
- 导出产物复制到：

```text
CarDD_YOLO11/exports/best.onnx
```

Demo 逻辑：

- notebook 02 优先读取 `best.pt`，缺失时回退 `last.pt`。
- 如果 `demo_images/` 中有自定义图片，优先对这些图片推理。
- 否则从 test、val、train 中抽样。
- 推理参数：

```text
imgsz: 1024
conf: 0.25
iou: 0.7
save_txt: True
save_conf: True
```

Demo 输出：

```text
CarDD_YOLO11/runs/predict/demo/
```

当前记录的 demo 结果：

```text
image0: 1 scratch
image1: no detections
image2: 1 lamp broken
image3: 1 dent, 1 scratch
image4: 1 scratch
image5: 2 cracks
image6: 1 scratch
image7: no detections
```

## 7. 实验结果摘要

训练设置：

```text
Model: YOLO11n-seg
Dataset: CarDD
Epochs: 100
Image size: 1024
Batch size: 7
GPU: Colab L4
Train time: about 4.1 hours
```

验证集结果：

验证集详细指标保留在 Drive 的训练日志和 `results.csv` 中；当前面向简历和公开展示的主结果统一采用下面的最新测试集指标。

测试集结果：

```text
Box  precision: 0.6717
Box  recall:    0.6374
Box  mAP50:     0.6746
Box  mAP50-95:  0.5111

Mask precision: 0.6795
Mask recall:    0.6242
Mask mAP50:     0.6712
Mask mAP50-95:  0.4917
```

各类别 test mAP50：

```text
Class          Box mAP50    Mask mAP50
dent             0.492         0.496
scratch          0.511         0.475
crack            0.211         0.219
glass shatter    0.978         0.978
lamp broken      0.790         0.778
tire flat        0.882         0.882
```

结果解读：

- `glass shatter`、`tire flat`、`lamp broken` 表现最好，视觉形态更明显、边界更清晰。
- `crack` 表现最弱，原因通常包括细长目标、纹理不明显、标注边界难、与划痕或车身纹理混淆。
- `dent` 和 `scratch` 居中但仍有提升空间，主要难点是尺度小、边界模糊、损伤形态变化大。
- mask mAP50 和 box mAP50 接近，说明模型不仅能定位，也能较稳定地产生实例分割轮廓。
- mAP50-95 明显低于 mAP50，说明在更严格 IoU 阈值下，精细边界和定位质量仍有改进空间。

## 8. Qwen7B QLoRA 报告模块规划

notebook 03 和 04 已准备，但当前阶段尚未说明已完整运行。

目标：

- 使用 Qwen2.5-7B-Instruct 生成实验报告。
- 通过 QLoRA 做轻量监督微调，让模型更适合从结构化实验指标生成技术报告。
- 输入只使用已有指标、训练 CSV、demo 摘要和少量报告风格样例，避免凭空编造。

默认方案：

```text
Base model: Qwen/Qwen2.5-7B-Instruct
Fine-tuning: PEFT LoRA + bitsandbytes 4-bit QLoRA
Open dataset: GEM/totto
Open train examples: 1200
Open eval examples: 120
Max sequence length: 1024
Epochs: 1
Per-device batch: 1
Gradient accumulation: 8
Learning rate: 2e-4
```

LoRA 配置：

```text
r: 16
lora_alpha: 32
lora_dropout: 0.05
bias: none
target_modules:
  q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

预期输出：

```text
CarDD_YOLO11/llm_adapters/qwen2_5_7b_cardd_report_lora/
CarDD_YOLO11/reports/qwen7b_report_sft_data.jsonl
CarDD_YOLO11/reports/qwen7b_report_sft_metadata.json
CarDD_YOLO11/reports/qwen7b_report_context.json
CarDD_YOLO11/reports/qwen7b_final_report.md
```

面试中建议的表达：

- 这个 LLM 模块不是为了提升检测精度，而是把实验结果自动转成结构化报告。
- `GEM/totto` 的作用是提供 data-to-text 训练信号，训练模型从结构化证据生成忠实文本。
- 少量 CarDD-specific examples 用于对齐本项目报告格式。
- 生成 prompt 中明确要求只使用 context JSON，避免模型编造指标。

## 9. 简历写法建议

项目标题：

```text
CarDD 车辆损伤检测与实例分割复现 | YOLO11 + Colab + QLoRA
```

技术栈：

```text
Python, Ultralytics YOLO11, PyTorch, OpenCV, COCO, YOLO Segmentation, Google Colab, Google Drive, ONNX, Qwen2.5-7B, PEFT, QLoRA, bitsandbytes
```

中文简历 bullet 建议：

- 基于 CarDD 数据集完成车辆损伤检测与实例分割复现，构建从 COCO 标注解析、YOLO segmentation 数据转换、模型训练、评估、ONNX 导出到 demo 可视化的端到端流程。
- 使用 Ultralytics YOLO11n-seg 在 Colab L4 上训练 100 epochs，设置 Drive 持久化训练目录与 `last.pt` 断点续训机制，解决 Colab 中断和训练产物丢失问题。
- 针对分割训练中 singleton batch 风险，将 batch size 调整为 7，并结合 AdamW、AMP、multi-scale、mosaic、mixup、copy-paste 等策略完成稳定训练。
- 在测试集上取得 box mAP50 0.6746、mask mAP50 0.6712，完成按类别分析，发现 `glass shatter`、`tire flat`、`lamp broken` 表现较优，`crack`、`scratch`、`dent` 为主要优化方向。
- 设计 Qwen2.5-7B-Instruct QLoRA 报告生成模块，将实验指标、训练日志和 demo 输出整理为结构化 context，生成可复用的 Markdown 实验报告。

如果简历篇幅很短，可以压缩成 2 条：

- 复现 CarDD 车辆损伤实例分割任务，基于 YOLO11n-seg 完成 COCO 到 YOLO segmentation 转换、Colab Drive 断点续训、测试评估、ONNX 导出和 demo 推理，测试集达到 box mAP50 0.6746、mask mAP50 0.6712。
- 构建实验报告自动化模块，使用 Qwen2.5-7B-Instruct + QLoRA 将结构化指标、训练日志和 demo 结果生成 Markdown 报告，并通过 prompt 约束降低指标编造风险。

不建议写法：

- 不要写“达到 SOTA”，当前项目没有和论文原始方法或其他方法做严格横向对比。
- 不要写“部署到生产”，当前只完成 ONNX 导出，没有完整服务化压测。
- 不要写“独立构建大模型”，实际是 QLoRA 轻量微调和报告生成。

## 10. 面试问答准备

### 10.1 60 秒项目介绍

这个项目是一个 CarDD 车辆损伤检测与实例分割的工程复现。我没有只停留在训练模型，而是把完整流程做成了 Colab 可复现 pipeline：首先解析 CarDD 的 COCO 风格标注，把 polygon 转成 YOLO segmentation 格式；然后用 YOLO11n-seg 在 Colab L4 上训练 100 个 epoch，并把数据、checkpoint、评估结果和 demo 输出都持久化到 Google Drive，支持中断后从 `last.pt` 继续。最终测试集 box mAP50 是 0.6746，mask mAP50 是 0.6712，并完成了 ONNX 导出和 demo 可视化。后续我还设计了 Qwen2.5-7B 的 QLoRA 报告模块，用结构化实验指标自动生成 Markdown 报告。

### 10.2 为什么选 YOLO11n-seg，而不是更大的模型

可以回答：

```text
这个项目的目标是快速、完整、可复现地打通车辆损伤实例分割流程，所以我优先选择 YOLO11n-seg。它足够小，适合 Colab L4，训练成本可控，同时支持 box 和 mask 两类输出。后续如果目标从复现闭环转向提升精度，可以把模型替换成 yolo11s-seg 或更大版本，再做输入分辨率、类别重采样和增强策略调参。
```

### 10.3 mAP50 和 mAP50-95 怎么解释

可以回答：

```text
mAP50 是 IoU 阈值为 0.5 时的平均精度，主要反映模型能否大致定位目标。mAP50-95 是从 0.5 到 0.95 多个 IoU 阈值的平均，更严格，更能体现边界和定位质量。当前测试集 box mAP50 是 0.6746，box mAP50-95 是 0.5111；mask mAP50 是 0.6712，mask mAP50-95 是 0.4917，说明模型已经具备可用定位能力，但精细定位和 mask 边界仍有提升空间。
```

### 10.4 为什么 crack 类效果差

可以回答：

```text
crack 通常目标更细、更长、更依赖局部纹理，和车身反光、划痕、阴影容易混淆。实例分割还要求 mask 边界准确，所以它比 glass shatter、tire flat 这类形态明显的类别更难。后续可以从更高输入分辨率、类别重采样、hard example mining、针对细小目标的数据增强和更大模型几个方向改进。
```

### 10.5 数据转换有什么技术难点

可以回答：

```text
CarDD 原始标注是 COCO 风格，YOLO segmentation 需要每个实例一行 polygon 坐标，并且坐标要归一化到 0 到 1。我实现时需要建立 category_id 到 class index 的映射、image_id 到 annotations 的映射，同时处理 split、图片路径索引、缺失图片、空 segmentation 和小于 3 个点的无效 polygon。为了适应 Colab 中断，我还给每个 split 写了 ready 标记，避免重复转换。
```

### 10.6 项目里最能体现工程能力的点

可以回答：

```text
我觉得主要有三个点。第一是把数据转换、训练、评估、导出、demo 拆成可复现 notebook，并且所有大产物都落到 Drive。第二是对 Colab 中断、batch size 导致的 singleton batch、checkpoint resume 这些实际问题做了处理。第三是没有把模型结果停留在数字上，而是整理了 per-class analysis、demo 输出和报告生成模块，方便复盘模型优势和缺陷。
```

### 10.7 QLoRA 报告模块的价值是什么

可以回答：

```text
它不是为了提升视觉模型精度，而是为了提升实验总结效率。输入是 metrics JSON、results.csv、demo 文件列表和 per-class 结果，模型被要求只依据这些结构化证据写报告。这样可以把视觉实验从训练到分析报告串成闭环，也能展示我对多模态或 AI 工程自动化的理解。
```

## 11. 当前阶段操作指南

你现在已经完成 notebook 01 和 02，建议按下面顺序收尾。

### 11.1 先冻结已完成产物

在 Google Drive 确认以下文件存在：

```text
CarDD_YOLO11/runs/train/yolo11n_seg/weights/best.pt
CarDD_YOLO11/runs/train/yolo11n_seg/weights/last.pt
CarDD_YOLO11/runs/train/yolo11n_seg/results.csv
CarDD_YOLO11/runs/train/yolo11n_seg/results.png
CarDD_YOLO11/exports/best.onnx
CarDD_YOLO11/reports/yolo11n_seg_test_metrics.json
CarDD_YOLO11/reports/yolo11n_seg_run_summary.json
CarDD_YOLO11/runs/predict/demo/
```

建议把以下内容单独截图或下载备份：

- `results.png`
- 测试集 PR 曲线或 confusion matrix，如果 Ultralytics 已生成。
- 4 到 8 张 demo 可视化图。
- `yolo11n_seg_test_metrics.json`
- `results.csv`

### 11.2 检查 GitHub 仓库是否干净

当前仓库应只提交：

```text
README.md
requirements-colab.txt
configs/
docs/
notebooks/
```

不要提交：

```text
*.pt
*.onnx
CarDD_release.zip
data_yolo/
runs/
llm_adapters/
私人链接、token、授权表、个人信息
```

### 11.3 如果要生成中文最终报告

运行 notebook 03 前：

- 使用 GPU Runtime，建议 L4 或更好。
- 确认 notebook 01 和 02 的产物已经在 Drive。
- 确认 Hugging Face 可以下载 `Qwen/Qwen2.5-7B-Instruct`。

运行 notebook 04 前：

- 如果已经运行 notebook 03，会加载 LoRA adapter。
- 如果没有 adapter，会回退到 base Qwen2.5-7B-Instruct。
- 如果想生成中文报告，把：

```python
REPORT_LANGUAGE = 'English'
```

改为：

```python
REPORT_LANGUAGE = 'Chinese'
```

### 11.4 如果只准备面试展示

最小展示包建议包含：

- 一页项目结构图。
- 一页数据转换流程图。
- 一页训练配置和断点续训说明。
- 一页总体指标。
- 一页 per-class 结果分析。
- 一页 demo 图片对比。
- 一页局限性与未来优化。

## 12. 下一阶段计划

### 阶段 A：结果证据固化

目标：确保所有简历数字都有可追溯文件。

任务：

- 检查 Drive 中 `yolo11n_seg_test_metrics.json` 的指标与 `docs/results_summary.md` 一致。
- 检查 `results.csv`、`results.png`、demo 输出目录存在。
- 为 demo 选择 4 到 8 张效果有代表性的图片。
- 将指标、路径和截图整理成一个面试展示文件夹。

### 阶段 B：中文报告与简历材料

目标：把项目包装成可以投递和讲解的材料。

任务：

- 使用本报告中的简历 bullet 更新简历。
- 准备 60 秒、3 分钟、8 分钟三个版本的项目讲解。
- 准备常见面试问答。
- 修正文档中 Drive layout 的乱码。

### 阶段 C：Qwen7B 报告模块

目标：完成“视觉模型结果到技术报告”的自动化闭环。

任务：

- 运行 notebook 03，保存 LoRA adapter。
- 运行 notebook 04，生成最终报告。
- 如需中文输出，把 `REPORT_LANGUAGE` 改为 `Chinese`。
- 检查生成报告是否严格使用真实指标，没有编造。

### 阶段 D：精度提升实验

目标：给面试准备“如果继续优化你会怎么做”的答案。

可选实验：

- 将模型换成 `yolo11s-seg.pt`。
- 继续使用 `imgsz=1024`，观察显存和训练时间。
- 针对 `crack`、`scratch`、`dent` 做类别重采样或 hard example 分析。
- 尝试关闭或调整 `mixup`、`copy_paste`，观察细粒度 mask 的影响。
- 记录每个实验的 mAP50、mAP50-95、训练时长和失败样例。

### 阶段 E：部署与展示增强

目标：从复现项目升级成可演示系统。

可选方向：

- 使用 ONNX Runtime 写一个轻量推理脚本。
- 做一个 Gradio demo，上传车辆图片后显示损伤类别、置信度、box 和 mask。
- 加入错误案例页面，展示模型局限性和改进思路。

## 13. 当前项目风险与清理项

风险或小问题：

- 本地 notebook 没有执行输出，提交 GitHub 很干净，但面试展示要依赖 Drive 产物或截图。
- `docs/drive_layout.md` 的目录树出现编码乱码，建议发布前修复。
- notebook 04 默认 `REPORT_LANGUAGE = 'English'`，如果目标是中文报告需要手动改。
- per-class test mAP50 在 notebook 03 和 04 中是硬编码记录，最好在最终说明中注明来源是本次测试结果摘要。
- 数据转换当前只保留最长 polygon，未来可以增强为完整多 polygon 支持。

推荐优先级：

1. 先备份 Drive 结果和 demo 图。
2. 修复文档乱码与中文报告输出设置。
3. 运行 notebook 03 和 04，完成 QLoRA 报告模块。
4. 整理简历 bullet 和面试讲稿。
5. 再考虑更大模型或部署 demo。

## 14. 最终交付清单

面向 GitHub：

- README 完整。
- notebooks 可按顺序运行。
- docs 包含复现计划、操作指南、结果摘要、隐私检查。
- 不含大文件和私人信息。

面向简历：

- 项目标题清晰。
- 技术栈完整。
- 有量化指标。
- 有工程难点。
- 有结果分析和未来优化。

面向面试：

- 能讲清楚数据转换。
- 能讲清楚 YOLO segmentation 输出格式。
- 能解释 mAP50 与 mAP50-95。
- 能解释为什么某些类别难。
- 能讲清楚 Colab Drive 断点续训。
- 能讲清楚 QLoRA 报告模块的边界和价值。
