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
            "metrics/mAP50(B)": 0.644,
            "metrics/mAP50-95(B)": 0.311,
            "metrics/mAP50(M)": 0.638,
            "metrics/mAP50-95(M)": 0.287,
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
    from vehicle_damage_pipeline.report.generate import generate_report_from_context

    root = _fresh_case("report_backend")
    adapter = _make_complete_adapter(root)

    qwen_result = generate_report_from_context(
        _context(),
        adapter_dir=adapter,
        qwen_generate_fn=lambda **_: "# Qwen report\n\nbox mAP50 0.644, mask mAP50 0.638",
    )
    assert qwen_result.metadata["backend"] == "qwen_adapter"
    assert qwen_result.text.startswith("# Qwen report")

    template_result = generate_report_from_context(_context(), backend="template")
    assert template_result.metadata["backend"] == "template"
    assert "box mAP50" in template_result.text
    assert "0.644" in template_result.text


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
        qwen_generate_fn=lambda **_: "Qwen 单图报告：检测到右前翼子板疑似凹陷，建议人工复核。",
    )
    assert qwen_report.startswith("Qwen 单图报告")

    template_report = generate_assessment_report(prediction, backend="template")
    assert "凹陷" in template_report
    assert "bbox=[282.0, 244.1, 621.1, 529.7]" in template_report
