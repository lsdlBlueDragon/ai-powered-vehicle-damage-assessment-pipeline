# AI-Powered Vehicle Damage Assessment Pipeline

End-to-end AI engineering project for vehicle damage detection, instance segmentation, structured inference output, grounded report generation, and RAG/LLM evaluation.

The project uses CarDD as the vision dataset and YOLO11 segmentation as the detector. Large datasets, checkpoints, adapters, and generated artifacts live in Google Drive; GitHub contains the lightweight engineering code, Colab runbooks, configs, tests, and documentation.

## What This Builds

```text
Vehicle image
  -> YOLO11n-seg damage detection and instance segmentation
  -> structured prediction JSON
  -> Qwen LoRA-backed Chinese/English project and image-level reports
  -> RAG/LLM evaluation for metric consistency, retrieval, and forbidden claims
  -> FastAPI + Gradio inference demo
```

This is not a SOTA claim and not a production insurance assessment system. It is a portfolio-oriented AI engineering pipeline focused on reproducibility, service boundaries, evaluation, and honest limitations.

## Current Results

Default completed experiment:

```text
Model: YOLO11n-seg
Dataset: CarDD
Epochs: 100
Image size: 1024
Batch size: 7
GPU: Colab L4
Training time: about 4.1 hours
```

Test metrics:

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

The strongest classes are `glass shatter`, `tire flat`, and `lamp broken`. The harder classes are `crack`, `scratch`, and `dent`.

## Repository Layout

```text
configs/                         Dataset config templates
docs/                            Model/data cards, runbooks, evaluation notes
notebooks/                       Thin Colab runbooks using !python -m commands
src/vehicle_damage_pipeline/     Engineering package
tests/                           Lightweight behavior tests
requirements-colab.txt           Colab dependency list
pyproject.toml                   Local package metadata
```

## Colab Workflow

Use Google Drive root:

```text
/content/drive/MyDrive/CarDD_YOLO11
```

Run the notebooks in order:

```text
01_train_cardd_yolo11_seg.ipynb
02_demo_cardd_yolo11_seg.ipynb
03_finetune_qwen7b_report_lora.ipynb
04_generate_llm_report_qwen7b.ipynb
06_colab_qwen_report_eval_full_workflow.ipynb
07_colab_final_portfolio_validation.ipynb
08_colab_public_url_demo_test.ipynb
```

Use `06_colab_qwen_report_eval_full_workflow.ipynb` as the current Task 1+2 report/eval runbook. It syncs the latest lightweight GitHub code into Drive, reuses the existing Qwen LoRA adapter with `--skip-if-complete`, checks the latest public metrics, generates the validated report, runs the LLM/RAG eval, and backs up the evidence bundle.

Use `07_colab_final_portfolio_validation.ipynb` after the portfolio polish changes. It verifies package tests, project report eval, single-image report safety, compact demo display helpers, and final lightweight evidence backup against existing Drive artifacts.

Use `08_colab_public_url_demo_test.ipynb` to rehearse the public Gradio demo flow. It supports uploaded images or direct online image URLs, and keeps the full prediction JSON in a collapsed, print-hidden debug panel so damaged-image masks do not make printed/PDF demo pages too long.

The notebooks are now runbooks. They install this package and call the CLI:

```bash
!python -m vehicle_damage_pipeline.data.prepare_cardd --drive-root /content/drive/MyDrive/CarDD_YOLO11
!python -m vehicle_damage_pipeline.vision.train_yolo --drive-root /content/drive/MyDrive/CarDD_YOLO11 --model yolo11n-seg.pt
!python -m vehicle_damage_pipeline.vision.predict --weights /content/drive/MyDrive/CarDD_YOLO11/runs/train/yolo11n_seg/weights/best.pt --source /content/drive/MyDrive/CarDD_YOLO11/demo_images --output /content/drive/MyDrive/CarDD_YOLO11/runs/predict/demo
!python -m vehicle_damage_pipeline.report.build_context --drive-root /content/drive/MyDrive/CarDD_YOLO11
!python -m vehicle_damage_pipeline.llm.finetune_report_lora --drive-root /content/drive/MyDrive/CarDD_YOLO11 --skip-if-complete --train-examples 1200 --eval-examples 120 --epochs 1 --max-seq-length 1024
!python -m vehicle_damage_pipeline.report.generate --context /content/drive/MyDrive/CarDD_YOLO11/reports/qwen7b_report_context.json --language Chinese --backend qwen --adapter-dir /content/drive/MyDrive/CarDD_YOLO11/llm_adapters/qwen2_5_7b_cardd_report_lora
!python -m vehicle_damage_pipeline.eval.run_llm_eval --context /content/drive/MyDrive/CarDD_YOLO11/reports/qwen7b_report_context.json --report /content/drive/MyDrive/CarDD_YOLO11/reports/qwen7b_final_report.md --knowledge-root /content/drive/MyDrive/CarDD_YOLO11 --output-json /content/drive/MyDrive/CarDD_YOLO11/reports/llm_eval_summary.json --output-markdown /content/drive/MyDrive/CarDD_YOLO11/reports/llm_eval_summary.md
```

`report.generate` and the Gradio/API report layer use the Qwen LoRA adapter by default. Add `--no-qwen` for deterministic template reports on low-memory or offline runs. The LoRA training CLI is reconnect-safe: when a complete adapter is already present in Drive, it writes metadata and skips training unless `--force-retrain` is provided.

## Reproducibility Levels

| Level | Required access | What you can run |
| --- | --- | --- |
| Level 1 | No CarDD data, no weights, no Drive artifacts | Install the package, run unit tests, inspect code, and read docs/runbooks. |
| Level 2 | Authorized CarDD dataset access | Convert annotations, train/evaluate YOLO11n-seg, export ONNX, and regenerate metrics. |
| Level 3 | Private Drive artifacts from this project | Run demo inference, Qwen adapter reports, RAG/LLM eval, and evidence backup without retraining. |

The public GitHub repository is intentionally Level 1 by default. Do not publish the private `CarDD_YOLO11` Drive root because it contains licensed data, weights, adapters, ONNX exports, runs, and backups. A public artifact folder must be curated separately and contain only lightweight demo screenshots/PDFs, report/eval summaries, docs, and notebooks.

Install options:

```bash
pip install -e ".[dev]"
pip install -e ".[vision,service,dev]"
pip install -e ".[vision,llm,service,dev]"
```

## Inference Service

FastAPI:

```bash
set VEHICLE_DAMAGE_WEIGHTS=<Google Drive desktop mirror>\CarDD_YOLO11\runs\train\yolo11n_seg\weights\best.pt
set VEHICLE_DAMAGE_QWEN_ADAPTER_DIR=<Google Drive desktop mirror>\CarDD_YOLO11\llm_adapters\qwen2_5_7b_cardd_report_lora
uvicorn vehicle_damage_pipeline.service.api:app --host 0.0.0.0 --port 8000
```

Endpoints:

```text
GET  /health
POST /predict
POST /report
```

Gradio:

```bash
python -m vehicle_damage_pipeline.service.gradio_app --weights "<Google Drive desktop mirror>\CarDD_YOLO11\runs\train\yolo11n_seg\weights\best.pt" --adapter-dir "<Google Drive desktop mirror>\CarDD_YOLO11\llm_adapters\qwen2_5_7b_cardd_report_lora"
```

## RAG/LLM Evaluation

The evaluation module checks:

- retrieval coverage over README, docs, report JSON, run summaries, and demo labels;
- grounded metric mentions against `qwen7b_report_context.json`;
- required report sections;
- forbidden claims such as SOTA or production-ready insurance assessment statements.

Outputs:

```text
reports/llm_eval_summary.json
reports/llm_eval_summary.md
```

## GitHub Safety

Do not commit:

```text
CarDD_release.zip
data_raw/
data_coco/
data_yolo/
runs/
*.pt
*.onnx
llm_adapters/
private Drive links
tokens or credentials
```

Keep those artifacts in Google Drive.

## Citation

```bibtex
@article{wang2023cardd,
  title={CarDD: A New Dataset for Vision-Based Car Damage Detection},
  author={Wang, Xinkuang and Li, Wenjing and Wu, Zhongcheng},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  volume={24},
  number={7},
  pages={7202--7214},
  year={2023},
  doi={10.1109/TITS.2023.3258480}
}
```
