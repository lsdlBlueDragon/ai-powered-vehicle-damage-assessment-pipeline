from vehicle_damage_pipeline.vision.formatting import build_prediction_response


def test_prediction_response_is_json_ready_and_contains_runtime_metadata():
    response = build_prediction_response(
        image_name="demo.jpg",
        detections=[
            {
                "class_id": 1,
                "class_name": "scratch",
                "confidence": 0.87654,
                "bbox_xyxy": [10.2, 20.6, 100.8, 120.1],
                "mask_polygon": [[10.0, 20.0], [100.0, 20.0], [100.0, 120.0]],
            }
        ],
        latency_ms=42.1234,
        model_version="yolo11n_seg_best",
    )

    assert response["image_name"] == "demo.jpg"
    assert response["model_version"] == "yolo11n_seg_best"
    assert response["latency_ms"] == 42.12
    assert response["detections"][0]["confidence"] == 0.877
    assert response["detections"][0]["bbox_xyxy"] == [10.2, 20.6, 100.8, 120.1]
    assert response["detections"][0]["mask_polygon"][0] == [10.0, 20.0]


def test_prediction_response_handles_no_detections():
    response = build_prediction_response(
        image_name="empty.jpg",
        detections=[],
        latency_ms=5,
        model_version="missing",
    )

    assert response["detections"] == []
    assert response["summary"] == "No vehicle damage detections above threshold."
