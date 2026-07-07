# Data Card

## Dataset

CarDD is a vehicle damage dataset for vision-based car damage detection and segmentation.

## Classes

```text
dent
scratch
crack
glass shatter
lamp broken
tire flat
```

## Processing

The pipeline converts COCO-style instance segmentation annotations into YOLO segmentation labels:

```text
class_id x1 y1 x2 y2 ...
```

Coordinates are normalized to `[0, 1]`. The conversion is resumable through per-split ready markers under `data_yolo/_split_ready/`.

## Governance

CarDD must be obtained through the provider's authorized process. Dataset archives, extracted images, and license forms must not be committed to GitHub.

## Known Data Risks

- Fine-grained categories such as cracks and scratches are visually ambiguous.
- Some damage masks can have complex boundaries.
- Current conversion keeps the longest polygon when multiple polygons appear for one annotation.
