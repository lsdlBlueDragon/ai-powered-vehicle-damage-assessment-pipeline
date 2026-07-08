import json


def _prediction():
    return {
        "image_name": "000033.jpg",
        "model_version": "best",
        "latency_ms": 123.45,
        "detections": [
            {
                "class_id": 0,
                "class_name": "dent",
                "confidence": 0.68,
                "bbox_xyxy": [282.0, 244.1, 621.1, 529.7],
                "mask_polygon": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            }
        ],
        "summary": "1 dent",
    }


def test_public_prediction_summary_hides_full_mask_polygon():
    from vehicle_damage_pipeline.service.display import build_public_prediction_summary

    summary = build_public_prediction_summary(_prediction())
    text = json.dumps(summary, ensure_ascii=False)

    assert summary["image_name"] == "000033.jpg"
    assert summary["detections"][0]["mask_points"] == 3
    assert "mask_polygon" not in text


def test_detection_table_contains_compact_fields():
    from vehicle_damage_pipeline.service.display import build_detection_table

    rows = build_detection_table(_prediction())

    assert rows == [
        {
            "class_name": "dent",
            "confidence": 0.68,
            "bbox_xyxy": "[282.0, 244.1, 621.1, 529.7]",
            "mask_points": 3,
        }
    ]


def test_debug_prediction_json_keeps_full_polygon():
    from vehicle_damage_pipeline.service.display import build_debug_prediction_json

    debug_json = build_debug_prediction_json(_prediction())
    parsed = json.loads(debug_json)

    assert parsed["detections"][0]["mask_polygon"] == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
