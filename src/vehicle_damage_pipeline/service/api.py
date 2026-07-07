from __future__ import annotations

import os

from vehicle_damage_pipeline.service.predictor import DamagePredictor, generate_assessment_report


WEIGHTS = os.environ.get("VEHICLE_DAMAGE_WEIGHTS")
OUTPUT_DIR = os.environ.get("VEHICLE_DAMAGE_OUTPUT_DIR", "runs/service_predict")
predictor = DamagePredictor(weights=WEIGHTS, output_dir=OUTPUT_DIR)

try:
    from fastapi import FastAPI, File, HTTPException, UploadFile
except Exception as exc:  # pragma: no cover - import-time guidance for missing deps
    raise RuntimeError("Install fastapi and python-multipart to run the API service.") from exc


app = FastAPI(title="AI-Powered Vehicle Damage Assessment Pipeline")


@app.get("/health")
def health() -> dict[str, object]:
    return predictor.health()


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, object]:
    try:
        result = predictor.predict_bytes(await file.read(), suffix=os.path.splitext(file.filename or "image.jpg")[1] or ".jpg")
        result["report"] = generate_assessment_report(result)
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/report")
def report(prediction: dict[str, object]) -> dict[str, str]:
    return {"report": generate_assessment_report(prediction)}
