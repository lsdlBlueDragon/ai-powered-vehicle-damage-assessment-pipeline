import tomllib
from pathlib import Path


def test_pyproject_declares_runtime_extras():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]

    assert "ultralytics>=8.3.0" in extras["vision"]
    assert "opencv-python" in extras["vision"]
    assert "transformers>=4.51.0" in extras["llm"]
    assert "peft" in extras["llm"]
    assert "fastapi" in extras["service"]
    assert "gradio" in extras["service"]
    assert "pytest" in extras["dev"]
