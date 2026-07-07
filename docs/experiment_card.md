# Experiment Card

## Experiment

```text
Name: yolo11n_seg
Model: YOLO11n-seg
Epochs: 100
Image size: 1024
Batch size: 7
Optimizer: AdamW
GPU: Colab L4
Training time: about 4.1 hours
```

## Engineering Choices

- Batch size `7` avoids singleton final batches in the 2817-image training split.
- Checkpoints are saved directly to Google Drive.
- Resume behavior loads `last.pt` while keeping safe current training arguments by default.
- ONNX export is produced for deployment-oriented demonstration.

## Results

```text
Box mAP50:     0.644
Box mAP50-95:  0.488
Mask mAP50:    0.638
Mask mAP50-95: 0.473
```

## Follow-Up Experiments

- Try `yolo11s-seg.pt` under a separate run name.
- Add focused crack/scratch/dent error analysis.
- Compare inference latency between PyTorch and ONNX Runtime.
- Add confidence calibration and threshold sweep.
