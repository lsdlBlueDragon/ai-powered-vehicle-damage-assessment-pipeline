import shutil
from pathlib import Path


TMP_ROOT = Path(__file__).resolve().parent / "_tmp_qwen_report_workflow"


def _fresh_case(name: str) -> Path:
    root = TMP_ROOT / name
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    return root


def _make_complete_adapter(root: Path) -> Path:
    adapter = root / "llm_adapters" / "qwen2_5_7b_cardd_report_lora"
    adapter.mkdir(parents=True, exist_ok=True)
    for name in [
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer_config.json",
        "tokenizer.json",
    ]:
        (adapter / name).write_text("{}", encoding="utf-8")
    return adapter


def _context() -> dict:
    return {
        "project": {"name": "AI-Powered Vehicle Damage Assessment Pipeline"},
        "dataset": {"name": "CarDD"},
        "test_metrics": {
            "metrics/mAP50(B)": 0.6745857662514867,
            "metrics/mAP50-95(B)": 0.5111031193401403,
            "metrics/mAP50(M)": 0.6711594915715345,
            "metrics/mAP50-95(M)": 0.49173212749837997,
        },
    }


def test_report_adapter_status_requires_core_files():
    from vehicle_damage_pipeline.llm.adapter import adapter_status, is_report_adapter_complete

    root = _fresh_case("adapter_status")
    adapter = root / "llm_adapters" / "qwen2_5_7b_cardd_report_lora"
    adapter.mkdir(parents=True)

    status = adapter_status(adapter)
    assert status["complete"] is False
    assert "adapter_model.safetensors" in status["missing_files"]
    assert is_report_adapter_complete(adapter) is False

    _make_complete_adapter(root)
    status = adapter_status(adapter)
    assert status["complete"] is True
    assert status["missing_files"] == []
    assert is_report_adapter_complete(adapter) is True


def test_finetune_skips_existing_complete_adapter_without_training_imports():
    from vehicle_damage_pipeline.llm.finetune_report_lora import finetune_report_lora

    root = _fresh_case("skip_existing_adapter")
    adapter = _make_complete_adapter(root)
    result = finetune_report_lora(drive_root=root, model_id="local-test-model")

    assert result == adapter
    metadata_path = root / "reports" / "qwen7b_report_sft_metadata.json"
    assert metadata_path.exists()
    metadata = metadata_path.read_text(encoding="utf-8")
    assert "skipped_existing_adapter" in metadata
    assert "local-test-model" in metadata


def test_project_report_defaults_to_qwen_adapter_and_can_use_template():
    from vehicle_damage_pipeline.eval.llm_eval import evaluate_report
    from vehicle_damage_pipeline.report.generate import generate_report_from_context

    root = _fresh_case("report_backend")
    adapter = _make_complete_adapter(root)
    valid_qwen_text = (
        "# Qwen report\n\n"
        "## 项目概览\n"
        "项目使用 CarDD 数据集构建车辆损伤检测与实例分割流程。\n\n"
        "## 结果\n"
        "测试集 box mAP50 0.675，mask mAP50 0.671。\n\n"
        "## 局限性\n"
        "该项目不是生产级保险定损系统，也不声称 SOTA。"
    )

    qwen_result = generate_report_from_context(
        _context(),
        adapter_dir=adapter,
        qwen_generate_fn=lambda **_: valid_qwen_text,
    )
    assert qwen_result.metadata["backend"] == "qwen_adapter"
    assert qwen_result.metadata["qwen_validation"]["passed"] is True
    assert qwen_result.metadata["final_validation"]["passed"] is True
    assert qwen_result.text.startswith("# Qwen report")

    template_result = generate_report_from_context(_context(), backend="template")
    assert template_result.metadata["backend"] == "template"
    assert template_result.metadata["final_validation"]["passed"] is True
    assert "box mAP50" in template_result.text
    assert "0.675" in template_result.text
    assert evaluate_report(_context(), template_result.text)["passed"] is True


def test_project_report_falls_back_when_qwen_output_fails_eval():
    from vehicle_damage_pipeline.report.generate import generate_report_from_context

    root = _fresh_case("report_validation_fallback")
    adapter = _make_complete_adapter(root)
    context = _context()

    result = generate_report_from_context(
        context,
        adapter_dir=adapter,
        qwen_generate_fn=lambda **_: "# Qwen report\n\nOnly project overview.",
    )

    assert result.metadata["backend"] == "template"
    assert result.metadata["requested_backend"] == "qwen"
    assert result.metadata["fallback_reason"] == "qwen_report_validation_failed"
    assert result.metadata["qwen_validation"]["passed"] is False
    assert result.metadata["final_validation"]["passed"] is True
    assert "box mAP50 为 0.675" in result.text
    assert "mask mAP50 为 0.671" in result.text
    assert "## 结果" in result.text
    assert "## 局限性" in result.text


def test_project_report_fallback_template_avoids_forbidden_claims():
    from vehicle_damage_pipeline.eval.llm_eval import evaluate_report
    from vehicle_damage_pipeline.report.generate import generate_report_from_context

    result = generate_report_from_context(_context(), backend="template")
    validation = evaluate_report(_context(), result.text)

    assert validation["passed"] is True
    assert validation["quality"]["forbidden_claims"] == []
    assert "SOTA" not in result.text
    assert "生产级保险定损" not in result.text


def test_assessment_report_can_use_qwen_backend_and_template_fallback():
    from vehicle_damage_pipeline.service.predictor import generate_assessment_report

    root = _fresh_case("assessment_backend")
    adapter = _make_complete_adapter(root)
    prediction = {
        "detections": [
            {
                "class_name": "dent",
                "confidence": 0.68,
                "bbox_xyxy": [282.0, 244.1, 621.1, 529.7],
            }
        ]
    }

    qwen_report = generate_assessment_report(
        prediction,
        backend="qwen",
        adapter_dir=adapter,
        qwen_generate_fn=lambda **_: (
            "Qwen 单图报告：检测到 dent/凹陷，confidence=0.680，"
            "bbox=[282.0, 244.1, 621.1, 529.7]。建议人工复核。"
        ),
    )
    assert qwen_report.startswith("Qwen 单图报告")

    template_report = generate_assessment_report(prediction, backend="template")
    assert "凹陷" in template_report
    assert "bbox=[282.0, 244.1, 621.1, 529.7]" in template_report
