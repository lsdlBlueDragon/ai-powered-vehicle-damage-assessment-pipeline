# Qwen7B Report Module Plan

## Goal

Add a lightweight LLM report generation module for the CarDD reproduction project.

The module should read existing Drive-backed experiment artifacts and generate a polished Markdown report. It does not retrain the detector and does not upload model weights to GitHub.

## Mainstream Options Checked

1. Transformers direct inference
   - Load `Qwen/Qwen2.5-7B-Instruct` with `AutoModelForCausalLM` and `AutoTokenizer`.
   - Use `tokenizer.apply_chat_template(...)`.
   - Best fit for a simple Colab notebook module.

2. Transformers with bitsandbytes 4-bit quantization
   - Reduces VRAM footprint on Colab L4.
   - Good default for one-off report generation.
   - Uses `BitsAndBytesConfig(load_in_4bit=True)`.

3. AWQ model
   - Uses `Qwen/Qwen2.5-7B-Instruct-AWQ`.
   - Good for lower memory inference, but adds extra dependency/version risk.
   - Keep as a later option if bitsandbytes is unstable.

4. vLLM or SGLang service
   - Strong for serving many requests.
   - Overkill for generating one project report inside Colab.

## Chosen Design

Use a third Colab notebook:

```text
notebooks/03_generate_llm_report_qwen7b.ipynb
```

Default model:

```text
Qwen/Qwen2.5-7B-Instruct
```

Default loading:

```text
Transformers + bitsandbytes 4-bit
```

Inputs:

```text
CarDD_YOLO11/reports/yolo11n_seg_test_metrics.json
CarDD_YOLO11/reports/yolo11n_seg_run_summary.json
CarDD_YOLO11/runs/train/yolo11n_seg/results.csv
CarDD_YOLO11/runs/predict/demo/
```

Outputs:

```text
CarDD_YOLO11/reports/qwen7b_final_report.md
CarDD_YOLO11/reports/qwen7b_report_context.json
```

## Report Contents

The generated report should include:

- Project background.
- Dataset and task.
- Reproduction pipeline.
- Training setup.
- Validation and test metrics.
- Per-class analysis.
- Demo inference summary.
- Limitations.
- Possible improvements.

## Execution Notes

- Run after notebooks 01 and 02.
- Use a GPU Colab runtime.
- If 4-bit model loading fails, restart runtime and rerun the notebook.
- Model cache remains in the Colab/Hugging Face cache; generated reports are saved to Drive.

