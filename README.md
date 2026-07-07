# CarDD YOLO11 Segmentation Reproduction

CarDD car damage instance segmentation reproduction using Ultralytics YOLO11 on Colab.

This repository keeps only lightweight, GitHub-friendly files. The CarDD dataset, checkpoints, training runs, and exported models should live in Google Drive, not in GitHub.

## Project Goal

Build a fast but complete reproduction project for CarDD:

- Convert CarDD COCO-style instance segmentation annotations to YOLO segmentation format.
- Fine-tune a small YOLO11 segmentation model for car damage detection and mask segmentation.
- Save all Colab training outputs to Google Drive.
- Resume training automatically from Drive after Colab disconnects.
- Run an end-to-end demo with predicted classes, boxes, confidence scores, and masks.

## Repository Layout

```text
.
|-- configs/
|   `-- cardd_yolo.yaml
|-- docs/
|   |-- drive_layout.md
|   |-- llm_report_module_plan.md
|   |-- privacy_checklist.md
|   |-- reproduction_plan.md
|   `-- results_summary.md
|-- notebooks/
|   |-- 01_train_cardd_yolo11_seg.ipynb
|   |-- 02_demo_cardd_yolo11_seg.ipynb
|   |-- 03_finetune_qwen7b_report_lora.ipynb
|   `-- 04_generate_llm_report_qwen7b.ipynb
|-- requirements-colab.txt
|-- .gitignore
`-- README.md
```

## Drive Layout

Use this Google Drive path in Colab:

```text
/content/drive/MyDrive/CarDD_YOLO11/
```

An optional local Google Drive desktop mirror can be:

```text
<Google Drive desktop mirror>/CarDD_YOLO11/
```

Large files belong there, not in GitHub.

## Notebooks

Run in this order:

1. `notebooks/01_train_cardd_yolo11_seg.ipynb`
   - Mounts Drive.
   - Installs dependencies.
   - Downloads the authorized CarDD zip to Drive if missing.
   - Extracts and converts COCO annotations to YOLO segmentation.
   - Trains YOLO11 segmentation.
   - Loads `last.pt` and continues with safe training arguments if interrupted.
   - Evaluates and saves metrics.

2. `notebooks/02_demo_cardd_yolo11_seg.ipynb`
   - Loads `best.pt` from Drive.
   - Runs inference on demo images or test samples.
   - Saves visualized predictions to Drive.

3. `notebooks/03_finetune_qwen7b_report_lora.ipynb`
   - Uses a small open `GEM/totto` data-to-text slice plus CarDD-specific report examples.
   - Fine-tunes Qwen2.5-7B-Instruct with QLoRA.
   - Saves only the LoRA adapter under `CarDD_YOLO11/llm_adapters/`.

4. `notebooks/04_generate_llm_report_qwen7b.ipynb`
   - Loads experiment metrics and demo summaries from Drive.
   - Uses the fine-tuned Qwen2.5-7B-Instruct LoRA adapter when available.
   - Saves the generated report under `CarDD_YOLO11/reports/`.

## Results

Default run:

```text
Model: YOLO11n-seg
Epochs: 100
GPU: Colab L4
Train time: about 4.1 hours
```

Test set metrics:

```text
Box  precision: 0.697
Box  recall:    0.592
Box  mAP50:     0.644
Box  mAP50-95:  0.488

Mask precision: 0.703
Mask recall:    0.592
Mask mAP50:     0.638
Mask mAP50-95:  0.473
```

See [docs/results_summary.md](docs/results_summary.md) for per-class results and artifact locations.

## Data Requirement

CarDD requires following the dataset provider's license process. After access is approved, the training notebook can download the official `CarDD_release.zip` directly into Google Drive.

The expected Drive location is:

```text
CarDD_YOLO11/data_raw/
```

You can also put an already extracted COCO-style dataset under:

```text
CarDD_YOLO11/data_coco/
```

The training notebook will search these locations.

## Default Model

The default model is:

```text
yolo11n-seg.pt
```

This keeps the reproduction fast while still producing a complete detection and segmentation pipeline. You can switch to `yolo11s-seg.pt` or a larger model inside the training notebook.

## Citation

If you use CarDD, cite the dataset paper:

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
