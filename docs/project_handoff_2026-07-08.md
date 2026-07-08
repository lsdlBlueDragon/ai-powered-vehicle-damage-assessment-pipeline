# AI-Powered Vehicle Damage Assessment Pipeline 交接文档

更新时间：2026-07-08

本文档用于新开对话或交接给新的 agent。阅读目标不是重新立项，而是快速接上当前项目状态：从最初的 CarDD + YOLO11 notebook 复现，如何转成面向简历和面试的 AI 工程项目；哪些已经完成；哪些证据已经跑出来；当前最需要修复的问题是什么；以及下一阶段应该如何收束。

## 1. 当前项目一句话

项目名称：**AI-Powered Vehicle Damage Assessment Pipeline**

当前定位：用 YOLO11 实例分割识别车辆损伤，输出结构化检测结果；用 Qwen2.5-7B LoRA 作为常驻报告生成方案，把结构化结果转成受约束的中文辅助评估报告；再用轻量 RAG/LLM evaluation 检查报告是否忠实于检测上下文，并提供 FastAPI/Gradio demo。

必须强调：

- 不是 SOTA 声明。
- 不是生产级保险定损系统。
- 不做维修报价、理赔决策或安全结论。
- LLM 只能基于结构化检测结果和项目上下文生成报告，不能自行补充严重程度、维修建议、事故责任、费用、是否无需复核等结论。

## 2. 关键路径

本地仓库：

```text
C:\Users\90553\Desktop\Wireless Simulation\QLoRA_project
```

Google Drive 桌面镜像：

```text
G:\我的云端硬盘\CarDD_YOLO11
```

Colab Drive root：

```text
/content/drive/MyDrive/CarDD_YOLO11
```

GitHub 新作品集仓库：

```text
https://github.com/lsdlBlueDragon/ai-powered-vehicle-damage-assessment-pipeline.git
```

Git remote 约定：

```text
origin     旧复现仓库
portfolio  新作品集仓库
```

最新已知提交：

```text
ec8fb9e docs: add Qwen Colab report adapter runbook
2c70f99 feat: make Qwen reports the default backend
7bdceb0 feat: improve single-image assessment narrative
2e42c29 feat: add Colab public Gradio demo launcher
ba61e6a feat: build vehicle damage assessment pipeline
```

## 3. 目标变化全过程

### 阶段 A：原始复现目标

最初目标是完成 CarDD 车辆损伤数据集上的 YOLO11 segmentation notebook 复现：

- 准备 CarDD 数据。
- 转换为 YOLO segmentation 格式。
- 在 Colab 训练 YOLO11n-seg。
- 运行 demo 预测。
- 导出 ONNX。
- 记录测试集指标。

核心文件：

```text
01_train_cardd_yolo11_seg.ipynb
02_demo_cardd_yolo11_seg.ipynb
reports/yolo11n_seg_test_metrics.json
runs/train/yolo11n_seg/weights/best.pt
exports/best.onnx
runs/predict/demo/
```

### 阶段 B：简历和面试导向

用户提出“当前项目是否适合作为简历面试方向”。结论是：仅有 notebook 复现不够专业，但车辆损伤检测方向本身适合作为 AI 工程作品集。需要把叙述从“我跑了一个模型”升级为：

- 车辆损伤检测与实例分割。
- 结构化结果输出。
- LLM 报告生成。
- 服务化推理。
- 可复现 Colab 工作流。
- 评估与限制说明。

这一步的核心变化是从“算法复现”转成“端到端 AI 工程系统”。

### 阶段 C：工程化改造计划

用户要求把 GitHub 仓库改为 `AI-Powered Vehicle Damage Assessment Pipeline`，并按工程项目改造。已形成总计划：

- 新建 `src/vehicle_damage_pipeline/` Python package。
- 主流程仍在 Colab 跑，但 notebook 内用 `!python -m ...` 调 CLI。
- 增加 FastAPI 和 Gradio。
- 增加 RAG/LLM evaluation。
- 增加 model card、data card、experiment card、API usage、portfolio pitch。
- Drive 保留大文件，GitHub 只放代码、文档、轻量配置和示例。

### 阶段 D：四个 notebook 与 runbook

原先已有：

```text
01_train_cardd_yolo11_seg.ipynb
02_demo_cardd_yolo11_seg.ipynb
03_finetune_qwen7b_report_lora.ipynb
04_generate_llm_report_qwen7b.ipynb
```

后来新增 Colab 总 runbook：

```text
00_colab_ai_powered_vehicle_damage_pipeline_runbook.ipynb
```

再后来，为 Qwen 报告 adapter 专门新增：

```text
05_colab_qwen_report_adapter_workflow.ipynb
```

当前项目建议把 `05` 当作 Qwen 报告层的主 notebook。YOLO 已经训练完成，后续不应重复跑 YOLO 微调，除非用户明确要求。

### 阶段 E：推理服务和 demo

已增加：

- FastAPI `/health`、`/predict`、`/report`。
- Gradio demo。
- Colab public Gradio launcher。
- 专用 demo 脚本，避免 Colab 中只出现 `127.0.0.1:7860` 导致用户无法打开。

需要注意：Colab 不能直接访问本地 URL，Gradio 必须 `share=True` 或使用 Colab 专用 launcher 生成公网链接。

### 阶段 F：Qwen 作为常驻报告方案

用户确认：Qwen 作为默认报告输出方案，但可以选择不用 Qwen。

已做：

- Qwen2.5-7B-Instruct + LoRA adapter workflow。
- 如果 adapter 已完整存在，Colab 重连后应跳过训练并直接加载 adapter。
- 报告生成默认走 Qwen adapter。
- 无 Qwen 或 Qwen 输出不合格时，应降级到受控模板。

当前关键问题：Qwen 确实加载并生成了内容，但报告没有覆盖所有评估要求，导致 `llm_eval_summary` 失败；单图 demo 报告还有编造/过度推断问题。

## 4. 已完成的主要工作

| 模块 | 状态 | 证据/备注 |
| --- | --- | --- |
| CarDD YOLO11 训练 | 已完成 | 不要重复训练，除非明确要求 |
| YOLO demo/ONNX | 已完成 | Drive 中保留预测输出和导出文件 |
| 工程化 Python package | 已完成基础结构 | `src/vehicle_damage_pipeline/` |
| CLI 化 Colab 调用 | 已完成主结构 | notebook 使用 `!python -m ...` |
| FastAPI 推理服务 | 已完成基础版 | `/health`、`/predict`、`/report` |
| Gradio demo | 已完成基础版 | 需要继续改展示质量 |
| Colab public demo launcher | 已完成 | 避免本地 URL 无法打开 |
| Qwen adapter workflow | 已完成基础版 | adapter 存在时可跳过训练 |
| Qwen 默认报告方案 | 已完成基础版 | 但报告质量约束还不够 |
| RAG/LLM eval | 已完成基础版 | 当前 `llm_eval_summary` fail |
| 备份机制 | 已完成基础版 | 最新 notebook 运行结果备份显示 `missing: []` |
| GitHub 新仓库 | 已建立并推送 | `portfolio` remote |

## 5. 当前最新运行结果

来自用户提供的：

```text
C:\Users\90553\Downloads\05_colab_qwen_report_adapter_workflow.ipynb
C:\Users\90553\Downloads\demo1.pdf
C:\Users\90553\Downloads\demo2.pdf
```

### 5.1 Notebook 运行情况

当前 notebook 共有 16 个 cell。已观察到：

- 主要 cell 已运行。
- Gradio cell 停留在服务运行状态是正常现象。
- 测试 cell 显示：`15 passed in 10.71s`。
- Qwen adapter 检查通过，adapter 文件完整。
- Qwen 模型实际加载，日志显示加载约 `15.2G` 权重。
- 备份 cell 显示 `missing: []`。

### 5.2 最新 YOLO 指标

当前更优指标来自最新 context，不应继续使用早期较低指标作为主要展示值。

```text
box precision:  0.6717414519451572
box recall:     0.6373781613844642
box mAP50:      0.6745857662514867
box mAP50-95:   0.5111031193401403

mask precision: 0.6795125959015245
mask recall:    0.6241890394653345
mask mAP50:     0.6711594915715345
mask mAP50-95:  0.49173212749837997
```

对外展示可写成：

```text
Box mAP50:  0.6746
Mask mAP50: 0.6712
```

旧指标：

```text
Box mAP50:  early lower value, replaced by the latest context metric
Mask mAP50: early lower value, replaced by the latest context metric
```

旧指标仍可能出现在 README、docs、tests、eval query 中，需要统一更新。

### 5.3 Qwen 报告生成情况

Qwen backend 已工作：

```text
backend: qwen_adapter
model: Qwen/Qwen2.5-7B-Instruct
adapter complete: true
```

但生成内容只覆盖了部分章节，例如项目概览、训练设置，没有完整覆盖“结果”“局限性”等评估要求。因此不能把“Qwen 已能加载”误判为“报告层完成”。

### 5.4 LLM evaluation 结果

当前 `llm_eval_summary` 失败：

```text
passed: false
retrieval recall@5: 1.0
box_map50: false
mask_map50: false
required section 项目概览: true
required section 结果: false
required section 局限性: false
```

解读：

- 检索层基本可用。
- 失败主要在报告生成层和 evaluation 规则不一致。
- eval 仍检查早期旧指标，而 context 已经是新指标。
- Qwen 输出没有强制插入准确指标和必需章节。

下一步第一优先级就是让 `llm_eval_summary.json` 通过。

## 6. PDF demo 当前问题

### demo1.pdf

文件：

```text
C:\Users\90553\Downloads\demo1.pdf
```

观察结果：

- 2 页。
- 示例图 `000040.jpg` 未检测到高置信度损伤。
- 文案出现“无需复核”倾向，过强。

建议改为：

```text
当前阈值下未发现高置信度损伤候选；建议结合原图、更多角度照片和人工复核判断。
```

禁止写：

```text
无需复核
确认无损伤
车辆状态正常
```

### demo2.pdf

文件：

```text
C:\Users\90553\Downloads\demo2.pdf
```

观察结果：

- 约 35 页，主要原因是 raw `mask_polygon` JSON 被直接展示，导致 PDF 极长。
- 检测到 `000033.jpg`：

```text
class: dent
confidence: 0.68
bbox: [282.0, 244.1, 621.1, 529.7]
```

严重问题：

Qwen 报告出现过度推断，例如：

```text
位于车身左侧中部区域，从车头延伸至车尾
覆盖整个车身中部
面积较大且严重
```

这些信息并不能从当前 structured JSON 可靠推出，属于 LLM 编造/过度解释。

建议：

- PDF demo 控制在 1-3 页。
- 主视图展示 annotated image、检测表格、短报告。
- full mask polygon 放到折叠区，不进入 PDF 主导出内容。
- 对单图报告增加“不能推断严重程度、不能推断车身部位、不能推断维修建议”的硬约束。

## 7. LLM 报告必须遵守的限制

所有 Qwen/LLM 报告都必须满足：

1. 事实来源只能是结构化检测 JSON、指标 context、项目文档片段。
2. 数值指标必须由程序插入，不能让 LLM 自己生成。
3. 类别、confidence、bbox 必须与检测 JSON 一致。
4. 不得声称 SOTA。
5. 不得声称 production-ready。
6. 不得给出保险定损、维修报价、赔付建议。
7. 不得根据单张图断言“无需复核”。
8. 不得编造车身部位、损伤严重程度、损伤面积比例。
9. 不得用“严重”“轻微”“大面积”等词，除非有明确计算规则支持。
10. 必须包含免责声明。

标准中文免责声明：

```text
该输出仅用于辅助评估，不构成生产级保险定损结论。
```

建议英文免责声明：

```text
This output is for auxiliary assessment only and is not a production-grade insurance damage appraisal.
```

推荐单图报告模板：

```text
车辆损伤自动检测摘要：
- 检测到 dent 候选区域，置信度 0.680，位置框 [282.0, 244.1, 621.1, 529.7]。
- 当前输出仅反映模型在该图像上的检测结果；损伤严重程度、维修方案和保险结论需要结合更多图像与人工复核。

该输出仅用于辅助评估，不构成生产级保险定损结论。
```

不建议让 Qwen 直接自由生成最终文本。更稳妥的方式：

1. 程序生成固定骨架和硬指标。
2. Qwen 只润色“项目介绍/限制说明”等非数值段落。
3. 生成后跑 validator。
4. 若 validator 失败，自动回退模板。

## 8. 与外部项目和行业方向的对比

### 8.1 CarDD 数据集定位

CarDD 官方页面介绍其为车辆损伤检测与分割数据集，包含约 4,000 张高分辨率图像、9,000 多个损伤实例和 6 类损伤类型，并需要填写 license agreement 才能下载数据。

对本项目的意义：

- 使用 CarDD 是合理的，因为它和“车辆损伤检测/实例分割”目标高度一致。
- GitHub 不应上传原始数据集、压缩包或大权重。
- 简历中可以写“基于公开研究数据集 CarDD 构建工程化 pipeline”，不要写成生产业务数据。

来源：https://cardd-ustc.github.io/

### 8.2 YOLO segmentation 工程方向

Ultralytics segmentation 文档把实例分割定义为对图像对象进行类别标注并生成 mask/contour，支持 train、val、predict、export，并能输出 boxes、class ids、masks、polygons 等结果。

对本项目的意义：

- 当前 YOLO11-seg + bbox + mask polygon 的结构化输出是主流且合理的工程方案。
- 作品集应突出“检测结果结构化、服务化、可评估”，而不是只展示一张预测图。
- mask polygon 是工程输出，不适合全部直接暴露给普通 demo 用户。

来源：https://docs.ultralytics.com/tasks/segment/

### 8.3 PEFT/LoRA 与 Qwen adapter

Hugging Face PEFT 文档强调，PEFT 通过只训练少量参数来适配大模型，降低计算和存储成本。LoRA 通过插入小型可训练低秩矩阵、冻结基础模型权重来减少训练内存和时间。

对本项目的意义：

- Qwen2.5-7B + LoRA adapter 是合理的报告生成方向，适合展示“视觉模型 + LLM 应用”的组合能力。
- 但是 LoRA 微调不能用来掩盖事实约束问题。真正专业的部分是“受控生成 + eval + fallback”，不是单纯加载 7B 模型。

来源：

- https://huggingface.co/docs/peft/en/index
- https://huggingface.co/docs/peft/en/task_guides/lora_based_methods

### 8.4 RAG/LLM evaluation

RAGAS 文档列出了 RAG 评估常见指标，如 context precision、context recall、response relevancy、faithfulness、factual correctness、string presence 等。

对本项目的意义：

- 当前轻量 eval 方向是对的：检查检索、数值一致性、章节覆盖、禁止夸大声明。
- 暂时不需要引入完整 RAGAS 依赖，避免增加复杂度。
- 可以在 README 中说明“inspired by common RAG evaluation dimensions”，但当前实现是轻量规则评估。

来源：https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/

### 8.5 CrashCar101 等相近方向

CrashCar101 探索合成损伤车辆数据，用于车辆部件和损伤分割。

对本项目的意义：

- 说明该方向确实有研究和工程价值。
- 但当前阶段不建议加入合成数据生成，容易扩大范围。
- 可以作为未来工作：扩充数据、改善少样本损伤类别、提高泛化。

来源：https://arxiv.org/abs/2311.06536

## 9. 当前项目是否适合作为简历/面试项目

结论：适合，但前提是把报告层和 demo 层收束好。

适合点：

- 有完整视觉训练链路：数据准备、YOLO segmentation、metrics、demo、ONNX。
- 有工程封装：CLI、package、FastAPI、Gradio。
- 有 LLM 应用：Qwen adapter、报告生成。
- 有评估意识：RAG/LLM eval、禁止夸大、数值一致性检查。
- 有部署意识：Colab/Drive 分离、GitHub-safe artifact 管理。
- 面试中能讲清 trade-off：为什么不追求 SOTA，为什么报告要受控，为什么需要 fallback。

当前不足：

- `llm_eval_summary` 失败会削弱“评估闭环”说服力。
- README/docs 中旧指标未统一，会显得不严谨。
- PDF demo 太长且暴露 raw JSON，不适合展示。
- 单图 Qwen 报告会编造 unsupported claim，必须修。
- Qwen 7B 加载重，demo 不能依赖每次都从零加载大模型。

总体建议：

项目现在不需要继续加功能，而要做“收束和打磨”。最有价值的专业性提升不是再加一个模型，而是让现有 pipeline 稳定、可解释、可验证、可展示。

## 10. 下一阶段总计划

### P0：修复 LLM evaluation

目标：

```text
reports/llm_eval_summary.json
passed: true
```

任务：

1. 更新 eval 中旧指标，改为最新 context 值。
2. 修改 `src/vehicle_damage_pipeline/report/generate.py`。
3. 用程序插入准确指标和固定章节。
4. Qwen 只负责润色非关键段落。
5. 增加 report validator。
6. 失败时自动 fallback 到模板报告。
7. 更新相关 tests。

预期输出：

```text
reports/qwen7b_final_report.md
reports/llm_eval_summary.json
reports/llm_eval_summary.md
```

### P1：更新指标与公开文档

目标：所有公开文件统一为最新指标。

任务：

1. 搜索并替换早期旧指标。
2. README 添加 latest results 表。
3. `docs/model_card.md`、`docs/experiment_card.md`、`docs/results_summary.md` 同步。
4. 明确写“不追求 SOTA、非生产定损”。

展示指标建议：

```text
Box mAP50: 0.6746
Mask mAP50: 0.6712
```

### P2：修复 Gradio/PDF demo

目标：demo 可直接用于面试展示。

任务：

1. PDF 主页面只展示 annotated image、检测表格、短报告。
2. full JSON 和 mask polygon 放折叠区。
3. 单图报告禁止编造位置、严重程度、维修方案。
4. no-damage case 改成“当前阈值下未发现高置信度候选，建议人工复核”。
5. 导出 demo PDF 控制在 1-3 页。

预期：

```text
demo1.pdf: clean no-high-confidence-damage example
demo2.pdf: clean dent example without hallucinated severity/location
```

### P3：轻量稳定性优化

只在 P0-P2 完成后做：

1. Qwen singleton/cached pipeline，避免重复加载。
2. 记录 `qwen_latency_ms`。
3. 为 3 个单图 demo 增加小型 eval：
   - no high-confidence damage
   - dent
   - difficult crack/scratch case

暂不建议做：

- 新增检测架构。
- 新增向量数据库。
- 云部署。
- OCR/理赔单解析。
- 维修报价。
- 合成数据生成。
- 多 LLM 对比。

## 11. 新开对话建议提示词

可以把下面内容直接发给新对话：

```text
继续 AI-Powered Vehicle Damage Assessment Pipeline 项目。
请先阅读 docs/project_handoff_2026-07-08.md。

当前优先级：
1. 修复 Qwen 报告生成和 eval，使 reports/llm_eval_summary.json 通过。
2. 把早期旧指标统一更新为最新 context 指标：box mAP50 0.6746，mask mAP50 0.6712。
3. 修复 Gradio/PDF demo：不要在主页面输出完整 mask polygon；单图 Qwen 报告不得编造损伤严重程度、车身部位、维修建议；必须包含免责声明。

限制：
- 不要重新训练 YOLO。
- 不要删除 Drive 结果。
- 不要把 CarDD 原始数据、权重、adapter、ONNX、私有 Drive 链接提交到 GitHub。
- 不要增加新模型架构、向量数据库、云部署、OCR 或维修报价模块。

本地仓库：
C:\Users\90553\Desktop\Wireless Simulation\QLoRA_project

Drive：
G:\我的云端硬盘\CarDD_YOLO11

GitHub remote：
portfolio
```

## 12. 交接红线

- 不要重跑 YOLO 训练。
- 不要删除 Drive artifacts。
- 不要提交 CarDD 原始数据、权重、adapter、ONNX 或私有链接。
- 不要声称 SOTA。
- 不要声称 production-ready insurance assessment。
- 不要让 Qwen 自由生成最终事实段落。
- 不要把 demo PDF 中 full mask polygon dump 当作最终展示材料。
- 不要在 no-damage case 中写“无需复核”。
- 不要让 LLM 生成维修费用、责任判断、理赔建议。

## 13. 当前最短行动路线

如果下一轮只做一件事，做 P0：

1. 修改 report generator，使固定章节和指标由程序保证。
2. 修改 eval，使指标检查使用最新 context。
3. 重新运行 notebook 05 的报告生成和 eval cell。
4. 确认 `llm_eval_summary` 通过。

如果下一轮做两件事，加上 P2：

1. 修 Gradio/PDF 展示。
2. 用 `000033.jpg` 和 no-damage 示例各导出一个干净 demo。

完成这两步后，项目就更像一个可讲、可展示、可防守的 AI Engineer 作品集，而不是 notebook 堆叠。

## 14. 2026-07-08 最终收口状态更新

本节为最新状态，优先级高于上方历史计划。

已经完成：

1. Task 1 报告链路已具备可用工程闭环：
   - Qwen LoRA adapter 文件在 Drive 中完整存在。
   - Colab public demo 页面显示报告层默认使用 Qwen LoRA adapter。
   - 用户补充说明：notebook 中的 `^C` 是为了终止外部链接测试/展示阶段，并不代表 Qwen 加载和报告生成失败。
   - 06 notebook 对旧 metadata、Qwen validation fallback、template fallback 均做兼容。
2. Task 2 eval 已跑通：
   - `reports/llm_eval_summary.json` 最终 `passed: true`。
   - report eval：指标 grounding、必要章节、forbidden claims 检查均通过。
   - retrieval eval：`recall_at_5: 1.0`。
   - 最新公开指标检查为 box mAP50 `0.6746`、mask mAP50 `0.6712`。
3. 公开指标已统一：
   - README、model card、experiment card、results summary、中文面试报告和 eval 检查均使用最新 test metrics。
4. 当前 canonical Colab runbook：
   - 不再继续使用 05 作为新流程入口。
   - 使用 `notebooks/06_colab_qwen_report_eval_full_workflow.ipynb` 跑 Task 1+2 报告/eval 全流程。
   - 06 已包含 PYTHONPATH 修复、retrieval evidence、eval-safe fallback、metadata 兼容和最终备份 cell。
5. 最新 Colab 证据备份：
   - Drive 备份目录：`CarDD_YOLO11/backups/qwen_report_eval_task12_20260708_085511`
   - Drive 备份压缩包：`CarDD_YOLO11/backups/qwen_report_eval_task12_20260708_085511.zip`
   - manifest 中 `missing: []`。
6. demo PDF 证据：
   - `demo1.pdf`：public Gradio no-damage 示例，包含外部 gradio.live 链接和报告摘要。
   - `demo2.pdf`：public Gradio dent 示例，包含检测 JSON、mask polygon 和自然语言摘要。
   - 注意：demo2 仍较长并暴露完整 polygon，适合作为调试/证据材料；若要作为简历展示材料，建议截图或另导出 1-3 页精简版。

按最初目标重新评估：

该项目已经达到“完整可用、面向简历和面试的工程项目”标准。推荐展示口径是：

```text
基于 CarDD 的车辆损伤实例分割工程项目，使用 YOLO11n-seg 完成训练、测试评估、Colab/Drive 断点恢复、demo 推理、Qwen LoRA 报告层、LLM/RAG eval 和证据备份，形成可复现、可验证、可讲解的 AI Engineer 作品集闭环。
```

边界表述：

- 可以说项目包含 Qwen LoRA 报告层和 eval/fallback 机制。
- 可以说 public Colab demo 已生成报告摘要和外部链接。
- 不要把项目描述成生产级保险定损系统。
- 不要声称 SOTA。
- 不要把长 polygon PDF 当作最终精修展示页。
