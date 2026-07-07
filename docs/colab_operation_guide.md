# Colab Operation Guide

This guide explains the exact Colab workflow for reproducing the CarDD YOLO11 segmentation project and generating the Qwen7B report.

## 0. Before Running

Required:

- A Google account with Drive access.
- Colab GPU runtime. L4 or better is recommended.
- Authorized access to the CarDD dataset.
- The notebooks in this repository, or the synced copies under:

```text
CarDD_YOLO11/notebooks/
```

Do not upload dataset archives, model weights, generated reports, or signed license forms to GitHub.

## 1. Drive Layout

All large files should live under:

```text
/content/drive/MyDrive/CarDD_YOLO11/
```

Expected folders:

```text
CarDD_YOLO11/
|-- data_raw/
|-- data_coco/
|-- data_yolo/
|-- runs/
|-- exports/
|-- reports/
|-- llm_adapters/
|-- demo_images/
`-- notebooks/
```

The notebooks create missing folders automatically.

## 2. Notebook 01: Train and Evaluate YOLO11 Segmentation

Notebook:

```text
notebooks/01_train_cardd_yolo11_seg.ipynb
```

Runtime:

```text
GPU, L4 or better
```

Run cells in order:

1. Mount Drive.
2. Install and import dependencies.
3. Configure paths.
4. Download CarDD to Drive.
5. Prepare dataset.
6. Quick label visualization.
7. Train or resume.
8. Evaluate and export.
9. Final backup summary.

Important behavior:

- `CarDD_release.zip` is downloaded to `data_raw/` if missing.
- COCO annotations are converted to YOLO segmentation format under `data_yolo/`.
- Dataset conversion is resumable using `data_yolo/_split_ready/`.
- Training saves directly to Drive.
- If `last.pt` exists, training continues from it using safe current arguments.
- Default batch is `7` to avoid a singleton final batch on the 2817-image train split.

Expected key outputs:

```text
CarDD_YOLO11/runs/train/yolo11n_seg/weights/best.pt
CarDD_YOLO11/runs/train/yolo11n_seg/weights/last.pt
CarDD_YOLO11/exports/best.onnx
CarDD_YOLO11/reports/yolo11n_seg_test_metrics.json
CarDD_YOLO11/reports/yolo11n_seg_run_summary.json
```

Common issue:

```text
CUDA out of memory
```

Recommended response:

- Keep `BATCH = 7`.
- If OOM still happens, set `multi_scale=False` before reducing image size.

Common issue:

```text
Expected more than 1 value per channel when training
```

Cause:

- Final batch size became 1.

Recommended response:

- Use the current notebook with `BATCH = 7`.
- Do not use strict resume with old `batch=8` arguments.

## 3. Notebook 02: Demo Inference

Notebook:

```text
notebooks/02_demo_cardd_yolo11_seg.ipynb
```

Runtime:

```text
GPU recommended, CPU acceptable for a small demo
```

Run after notebook 01.

Behavior:

- Loads `best.pt`, falling back to `last.pt`.
- Uses images from `demo_images/` if present.
- Otherwise samples prepared test/validation images.
- Saves visual predictions and label txt files.

Expected output:

```text
CarDD_YOLO11/runs/predict/demo/
```

## 4. Notebook 03: Qwen7B Report-Direction QLoRA Fine-Tuning

Notebook:

```text
notebooks/03_finetune_qwen7b_report_lora.ipynb
```

Runtime:

```text
GPU, L4 or better
```

Run after notebooks 01 and 02.

Purpose:

- Fine-tune `Qwen/Qwen2.5-7B-Instruct` toward faithful report-style generation.
- Use `GEM/totto` as the open data-to-text training source.
- Add a few CarDD-specific report-format examples from the actual project metrics.
- Save only the LoRA adapter to Drive.
- Reuse the existing Drive adapter after Colab reconnect. The default command skips training when the adapter is complete; use `--force-retrain` only when intentionally rebuilding it.

Expected outputs:

```text
CarDD_YOLO11/llm_adapters/qwen2_5_7b_cardd_report_lora/
CarDD_YOLO11/reports/qwen7b_report_sft_data.jsonl
CarDD_YOLO11/reports/qwen7b_report_sft_metadata.json
```

Notes:

- The base Qwen model is downloaded through Hugging Face cache in Colab.
- The base model is not committed to GitHub.
- The LoRA adapter should stay in Drive because it is a generated model artifact.
- Adapter completeness is checked by `adapter_config.json`, `adapter_model.safetensors`, `tokenizer_config.json`, and `tokenizer.json`.

## 5. Notebook 04: Generate Final Qwen7B Report

Notebook:

```text
notebooks/04_generate_llm_report_qwen7b.ipynb
```

Runtime:

```text
GPU recommended
```

Run after notebook 03. The default report backend uses the fine-tuned Qwen LoRA adapter saved in Drive. For low-memory or offline demonstration, add `--no-qwen` to use the deterministic template backend.

Behavior:

- Reads metrics, run summary, training CSV, and demo output from Drive.
- Builds a report context JSON.
- Loads the LoRA adapter by default and records report backend metadata.
- Generates a final Markdown report.

Expected outputs:

```text
CarDD_YOLO11/reports/qwen7b_report_context.json
CarDD_YOLO11/reports/qwen7b_final_report.md
CarDD_YOLO11/reports/qwen7b_final_report.metadata.json
```

## 6. Final GitHub Deliverables

GitHub should contain:

```text
README.md
requirements-colab.txt
configs/cardd_yolo.yaml
docs/
notebooks/
```

GitHub should not contain:

```text
CarDD_release.zip
data_yolo/
*.pt
*.onnx
llm_adapters/
signed license forms
private Drive links
personal tokens or credentials
```

## 7. Recommended Execution Order

For the complete project:

```text
01_train_cardd_yolo11_seg.ipynb
02_demo_cardd_yolo11_seg.ipynb
03_finetune_qwen7b_report_lora.ipynb
04_generate_llm_report_qwen7b.ipynb
```

For a quick visual-only reproduction:

```text
01_train_cardd_yolo11_seg.ipynb
02_demo_cardd_yolo11_seg.ipynb
```

For report generation after existing visual results:

```text
03_finetune_qwen7b_report_lora.ipynb
04_generate_llm_report_qwen7b.ipynb
```
