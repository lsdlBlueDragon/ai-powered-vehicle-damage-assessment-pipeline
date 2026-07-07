from vehicle_damage_pipeline.service.predictor import generate_assessment_report


def test_assessment_report_adds_plain_language_explanation():
    report = generate_assessment_report(
        {
            "image_name": "000033.jpg",
            "detections": [
                {
                    "class_name": "dent",
                    "confidence": 0.68,
                    "bbox_xyxy": [282.0, 244.1, 621.1, 529.7],
                }
            ],
        }
    )

    assert "检测到 1 处疑似车辆损伤" in report
    assert "凹陷" in report
    assert "中等置信度" in report
    assert "钣金" in report
    assert "建议人工复核" in report
    assert "bbox=[282.0, 244.1, 621.1, 529.7]" in report


def test_assessment_report_handles_empty_detection_in_plain_language():
    report = generate_assessment_report({"detections": []})

    assert "未检测到高于阈值" in report
    assert "更多角度" in report
