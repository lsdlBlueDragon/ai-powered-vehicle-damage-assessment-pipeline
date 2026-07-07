# Drive Layout

Use Google Drive for all files that must survive Colab disconnects.

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
|-- llm_adapters/
|-- external_data/
|-- notebooks/
|-- docs/
`-- backups/
```

Colab path:

```text
/content/drive/MyDrive/CarDD_YOLO11/
```

The GitHub repository should contain code, configs, notebooks, and docs only. Dataset files, model weights, adapters, ONNX exports, and generated runs remain in Drive.

## Backup Strategy

- YOLO checkpoints are saved directly under `runs/train/`.
- The latest evaluation metrics and generated reports are saved under `reports/`.
- Lightweight evidence backups should copy notebooks, reports, demo outputs, ONNX, docs, and configs only.
- Backups should not include `data_raw`, `data_coco`, `data_yolo`, `*.pt`, or LoRA checkpoint directories.
