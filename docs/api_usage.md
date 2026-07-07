# API Usage

## Start FastAPI

```bash
set VEHICLE_DAMAGE_WEIGHTS=<Google Drive desktop mirror>\CarDD_YOLO11\runs\train\yolo11n_seg\weights\best.pt
set VEHICLE_DAMAGE_QWEN_ADAPTER_DIR=<Google Drive desktop mirror>\CarDD_YOLO11\llm_adapters\qwen2_5_7b_cardd_report_lora
set VEHICLE_DAMAGE_REPORT_BACKEND=qwen
uvicorn vehicle_damage_pipeline.service.api:app --host 0.0.0.0 --port 8000
```

Use `set VEHICLE_DAMAGE_REPORT_BACKEND=template` when you want to run without Qwen.

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
python -m vehicle_damage_pipeline.service.gradio_app --weights "<Google Drive desktop mirror>\CarDD_YOLO11\runs\train\yolo11n_seg\weights\best.pt" --adapter-dir "<Google Drive desktop mirror>\CarDD_YOLO11\llm_adapters\qwen2_5_7b_cardd_report_lora"
```

Add `--no-qwen` to use deterministic template reports.
