# Drive Layout

Use Google Drive for all files that should survive Colab disconnects.

```text
CarDD_YOLO11/
├── data_raw/
│   └── put-authorized-cardd-zip-or-files-here
├── data_coco/
│   └── extracted-or-prepared-coco-style-cardd
├── data_yolo/
│   ├── images/
│   ├── labels/
│   ├── cardd.yaml
│   └── data_ready.json
├── runs/
│   ├── train/
│   ├── val/
│   └── predict/
├── exports/
├── reports/
└── backups/
```

Windows Google Drive mirror:

```text
G:\我的云端硬盘\CarDD_YOLO11\
```

Colab path:

```text
/content/drive/MyDrive/CarDD_YOLO11/
```

## Backup Strategy

- Training uses `project=/content/drive/MyDrive/CarDD_YOLO11/runs/train`.
- Checkpoints are saved directly to Drive.
- If `runs/train/yolo11n_seg/weights/last.pt` exists, the training notebook resumes from it.
- Evaluation, prediction images, CSV files, plots, and exported models are also written to Drive.

