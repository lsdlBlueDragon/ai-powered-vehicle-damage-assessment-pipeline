# API Usage

## Start FastAPI

```bash
set VEHICLE_DAMAGE_WEIGHTS=<Google Drive desktop mirror>\CarDD_YOLO11\runs\train\yolo11n_seg\weights\best.pt
uvicorn vehicle_damage_pipeline.service.api:app --host 0.0.0.0 --port 8000
```

## Health

```bash
curl http://localhost:8000/health
```

## Predict

```bash
curl -X POST http://localhost:8000/predict ^
  -F "file=@demo.jpg"
```

Response shape:

```json
{
  "image_name": "demo.jpg",
  "model_version": "best",
  "latency_ms": 42.12,
  "detections": [
    {
      "class_id": 1,
      "class_name": "scratch",
      "confidence": 0.877,
      "bbox_xyxy": [10.2, 20.6, 100.8, 120.1],
      "mask_polygon": [[10.0, 20.0], [100.0, 20.0]]
    }
  ],
  "summary": "1 scratch",
  "report": "..."
}
```

## Gradio

```bash
python -m vehicle_damage_pipeline.service.gradio_app --weights "<Google Drive desktop mirror>\CarDD_YOLO11\runs\train\yolo11n_seg\weights\best.pt"
```
