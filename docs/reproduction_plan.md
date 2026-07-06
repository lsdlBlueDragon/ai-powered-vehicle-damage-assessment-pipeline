# Reproduction Plan

## Assumptions

- The CarDD dataset has been obtained legally from the official provider.
- Colab uses an L4 GPU or better.
- Large files are stored in Google Drive, not GitHub.
- The main reproduction route uses Ultralytics YOLO11 segmentation rather than the original MMDetection route.

## Scope

This project reproduces a practical CarDD instance segmentation workflow:

- Dataset preparation.
- COCO-to-YOLO segmentation conversion.
- Small-model YOLO11 segmentation fine-tuning.
- Automatic Drive checkpointing and resume.
- Validation metrics and visual outputs.
- Demo inference notebook.

## Tasks

1. Prepare Drive layout.
   - Verify: required folders exist under `CarDD_YOLO11`.

2. Prepare CarDD data.
   - Verify: annotation JSON files are detected.
   - Verify: converted YOLO labels and image folders exist.
   - Verify: random label visualization looks correct.

3. Train YOLO11 segmentation.
   - Verify: `last.pt`, `best.pt`, `results.csv`, and plots are saved under Drive.
   - Verify: rerunning the training notebook resumes from `last.pt`.

4. Evaluate.
   - Verify: bbox and mask metrics are saved.
   - Verify: per-class results are available from Ultralytics outputs.

5. Demo.
   - Verify: selected images produce class labels, boxes, confidence scores, and masks.
   - Verify: visualized predictions are saved under Drive.

## Default Training Choice

Use `yolo11n-seg.pt` by default. It is small enough for quick reproduction and still supports a complete instance segmentation pipeline. For stronger results, change the notebook variable to `yolo11s-seg.pt` or larger.

## Current Status

- Dataset preparation is complete.
- YOLO11n-seg training is complete for 100 epochs.
- Test evaluation is complete.
- ONNX export is complete.
- Demo inference is complete.
- Final lightweight repository documentation is complete.
