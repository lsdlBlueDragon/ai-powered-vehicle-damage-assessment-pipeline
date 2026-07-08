# Model Card

## Model

- Vision model: `YOLO11n-seg`
- Task: vehicle damage detection and instance segmentation
- Dataset: CarDD
- Output: class labels, confidence scores, bounding boxes, segmentation masks, and structured JSON

## Intended Use

This model is intended for portfolio demonstration, research reproduction, and AI engineering workflow practice. It can assist with visual inspection demos but is not validated for production insurance claim decisions.

## Metrics

Test split:

```text
Box precision:  0.6717
Box recall:     0.6374
Box mAP50:      0.6746
Box mAP50-95:   0.5111
Mask precision: 0.6795
Mask recall:    0.6242
Mask mAP50:     0.6712
Mask mAP50-95:  0.4917
```

## Strengths

- Stronger performance on visually distinctive categories: `glass shatter`, `tire flat`, and `lamp broken`.
- Produces both detection and mask outputs.
- Small enough for practical Colab reproduction.

## Limitations

- Weaker performance on fine-grained damage categories such as `crack`, `scratch`, and `dent`.
- Not calibrated for real insurance workflows.
- Results depend on CarDD distribution and may not generalize to all vehicle images.
- Does not claim SOTA performance.

## Artifacts

Large artifacts stay in Google Drive:

```text
runs/train/yolo11n_seg/weights/best.pt
runs/train/yolo11n_seg/weights/last.pt
exports/best.onnx
reports/yolo11n_seg_test_metrics.json
```
