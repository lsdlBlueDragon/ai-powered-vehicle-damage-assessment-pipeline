# Qwen7B Report Module Plan

## Goal

Add a lightweight LLM fine-tuning and report generation module for the CarDD reproduction project.

The module should fine-tune a Qwen 7B model toward report-style generation, then read existing Drive-backed experiment artifacts and generate a polished Markdown report. It does not retrain the detector and does not upload model weights to GitHub.

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

## Open Dataset Choice

There are public datasets for report-like generation, but a perfect "ML experiment report" instruction dataset is not a common standard benchmark.

Recommended open training signal:

- `GEM/totto`: table-to-text, 138k rows on Hugging Face, licensed cc-by-sa-3.0. It trains faithful conversion from structured evidence into concise text.
- `GEM/web_nlg`: data-to-text from RDF-like structured inputs, useful as an alternative.
- `SciGen`: scientific table-to-text with numerical reasoning, closer to research reporting but less convenient than Hugging Face-hosted ToTTo for a fast Colab workflow.
- `e2e_nlg`: data-to-text in the restaurant domain, useful for NLG mechanics but less aligned with technical reports.

Chosen default: `GEM/totto` plus a few CarDD-specific report-format examples built from the actual experiment metrics.

## Chosen Design

Use two Qwen-related Colab notebooks:

```text
notebooks/03_finetune_qwen7b_report_lora.ipynb
notebooks/04_generate_llm_report_qwen7b.ipynb
```

Default model:

```text
Qwen/Qwen2.5-7B-Instruct
```

Default fine-tuning:

```text
Transformers + PEFT LoRA + bitsandbytes 4-bit QLoRA
```

Inputs:

```text
GEM/totto
CarDD_YOLO11/reports/yolo11n_seg_test_metrics.json
CarDD_YOLO11/reports/yolo11n_seg_run_summary.json
CarDD_YOLO11/runs/train/yolo11n_seg/results.csv
CarDD_YOLO11/runs/predict/demo/
```

Outputs:

```text
CarDD_YOLO11/llm_adapters/qwen2_5_7b_cardd_report_lora/
CarDD_YOLO11/reports/qwen7b_report_sft_data.jsonl
CarDD_YOLO11/reports/qwen7b_report_sft_metadata.json
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
- Run notebook 03 on a GPU Colab runtime to create the LoRA adapter.
- Run notebook 04 after notebook 03; it falls back to the base model if the adapter does not exist.
- If 4-bit model loading fails, restart runtime and rerun the notebook.
- Model cache remains in the Colab/Hugging Face cache; generated reports are saved to Drive.
