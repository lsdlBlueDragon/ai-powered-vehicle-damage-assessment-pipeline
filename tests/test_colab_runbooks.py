import json
from pathlib import Path


def test_07_final_portfolio_validation_notebook_exists_and_checks_required_flows():
    path = Path("notebooks/07_colab_final_portfolio_validation.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert len(notebook["cells"]) >= 10
    assert "final_portfolio_validation_" in text
    assert "final_validation" in text
    assert "evaluate_assessment_report" in text
    assert "build_public_prediction_summary" in text
    assert "build_detection_table" in text
    assert "mask_polygon" in text
    assert "missing" in text
