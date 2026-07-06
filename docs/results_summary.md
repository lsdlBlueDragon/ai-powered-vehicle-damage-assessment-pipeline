# Results Summary

## Training Setup

- Model: `yolo11n-seg.pt`
- Task: car damage instance segmentation
- Dataset: CarDD, converted from COCO-style annotations to YOLO segmentation format
- Epochs: 100
- Image size: 1024
- Batch size: 7
- GPU: Colab L4
- Training time: about 4.1 hours

The training notebook uses Drive-backed checkpoints and avoids singleton final batches during segmentation training.

## Validation Result

Final validation on `best.pt`:

```text
Box  precision: 0.659
Box  recall:    0.609
Box  mAP50:     0.631
Box  mAP50-95:  0.482

Mask precision: 0.661
Mask recall:    0.605
Mask mAP50:     0.626
Mask mAP50-95:  0.473
```

## Test Result

Final test metrics:

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

Per-class test mAP50:

```text
Class          Box mAP50    Mask mAP50
dent             0.492         0.496
scratch          0.511         0.475
crack            0.211         0.219
glass shatter    0.978         0.978
lamp broken      0.790         0.778
tire flat        0.882         0.882
```

The model performs best on `glass shatter`, `tire flat`, and `lamp broken`. The harder classes are `crack`, `scratch`, and `dent`, which is expected because these categories are visually finer and more ambiguous.

## Demo Output

The demo notebook sampled 8 test images and saved visualized predictions under:

```text
CarDD_YOLO11/runs/predict/demo/
```

Demo detections:

```text
image0: 1 scratch
image1: no detections
image2: 1 lamp broken
image3: 1 dent, 1 scratch
image4: 1 scratch
image5: 2 cracks
image6: 1 scratch
image7: no detections
```

## Saved Artifacts

Keep these Drive artifacts:

```text
CarDD_YOLO11/runs/train/yolo11n_seg/weights/best.pt
CarDD_YOLO11/runs/train/yolo11n_seg/weights/last.pt
CarDD_YOLO11/exports/best.onnx
CarDD_YOLO11/reports/yolo11n_seg_test_metrics.json
CarDD_YOLO11/reports/yolo11n_seg_run_summary.json
CarDD_YOLO11/runs/train/yolo11n_seg/results.csv
CarDD_YOLO11/runs/train/yolo11n_seg/results.png
CarDD_YOLO11/runs/val/yolo11n_seg_test/
CarDD_YOLO11/runs/predict/demo/
```

Intermediate `epoch*.pt` checkpoints are not required after training is complete.
