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
    assert "建议人工复核" in report
    assert "bbox=[282.0, 244.1, 621.1, 529.7]" in report
    assert "钣金" not in report
    assert "补漆" not in report
    assert "严重" not in report
    assert "车身左侧" not in report


def test_assessment_report_handles_empty_detection_in_plain_language():
    report = generate_assessment_report({"detections": []})

    assert "未检测到高于阈值" in report
    assert "人工复核" in report
    assert "无需复核" not in report


def test_assessment_report_rejects_qwen_hallucinated_details():
    from pathlib import Path

    from vehicle_damage_pipeline.service.predictor import evaluate_assessment_report

    root = Path(__file__).resolve().parent / "_tmp_qwen_report_workflow" / "assessment_hallucination"
    adapter = root / "llm_adapters" / "qwen2_5_7b_cardd_report_lora"
    adapter.mkdir(parents=True, exist_ok=True)
    for name in [
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer_config.json",
        "tokenizer.json",
    ]:
        (adapter / name).write_text("{}", encoding="utf-8")

    prediction = {
        "image_name": "000033.jpg",
        "detections": [
            {
                "class_name": "dent",
                "confidence": 0.68,
                "bbox_xyxy": [282.0, 244.1, 621.1, 529.7],
            }
        ],
    }
    hallucinated = "凹陷位于车身左侧中部，从车头延伸至车尾，面积较大且严重，建议钣金修复。"

    report = generate_assessment_report(
        prediction,
        backend="qwen",
        adapter_dir=adapter,
        qwen_generate_fn=lambda **_: hallucinated,
    )
    validation = evaluate_assessment_report(prediction, report)

    assert validation["passed"] is True
    for phrase in ["车身左侧中部", "从车头延伸至车尾", "面积较大", "严重", "钣金修复"]:
        assert phrase not in report
