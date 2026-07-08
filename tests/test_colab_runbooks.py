import json
from pathlib import Path


def _notebook_text(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_07_final_portfolio_validation_notebook_exists_and_checks_required_flows():
    path = Path("notebooks/07_colab_final_portfolio_validation.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    text = _notebook_text(path)
    question_triplet = "?" * 3

    assert len(notebook["cells"]) >= 10
    assert question_triplet not in text
    assert "最终作品集验证" in text
    assert "挂载 Drive" in text
    assert "单图报告安全" in text
    assert "证据备份" in text
    assert "final_portfolio_validation_" in text
    assert "final_validation" in text
    assert "Stale sync or old GitHub commit" in text
    assert "ensure_vehicle_damage_package" in text
    assert "sys.path.insert(0, str(src_root))" in text
    assert "evaluate_assessment_report" in text
    assert "build_public_prediction_summary" in text
    assert "build_detection_table" in text
    assert "mask_polygon" in text
    assert "missing" in text


def test_key_python_files_do_not_contain_mojibake_markers():
    files = [
        Path("src/vehicle_damage_pipeline/eval/llm_eval.py"),
        Path("src/vehicle_damage_pipeline/report/generate.py"),
        Path("src/vehicle_damage_pipeline/service/predictor.py"),
        Path("src/vehicle_damage_pipeline/llm/qwen_reporter.py"),
        Path("tests/test_llm_eval.py"),
        Path("tests/test_qwen_report_workflow.py"),
        Path("tests/test_assessment_report.py"),
    ]
    question_triplet = "?" * 3
    bad_markers = [
        "\u6924\u572d\u6d30",
        "\u7f01\u64b4\u7049",
        "\u705e\u20ac",
        "\u59ab\u20ac",
        "\u9351\u5f52",
        "\u9422\u71b6\u9a87",
        "\u675e\ufe41\u97e9",
        "\u5be4\u9e3f",
        "\u93c3\u72bb\u6e36",
        "\u95bd\uff49\u567e",
    ]
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert question_triplet not in text
        for marker in bad_markers:
            assert marker not in text, f"{path} still contains mojibake marker {marker!r}"
