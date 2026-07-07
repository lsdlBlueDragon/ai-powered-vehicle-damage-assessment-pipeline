from __future__ import annotations

from pathlib import Path
from typing import Any

from vehicle_damage_pipeline.paths import project_paths


REPORT_ADAPTER_NAME = "qwen2_5_7b_cardd_report_lora"
REQUIRED_REPORT_ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer_config.json",
    "tokenizer.json",
)


def report_adapter_dir(drive_root: str | Path) -> Path:
    return project_paths(drive_root)["llm_adapters"] / REPORT_ADAPTER_NAME


def adapter_status(adapter_dir: str | Path) -> dict[str, Any]:
    adapter_path = Path(adapter_dir)
    missing_files = [name for name in REQUIRED_REPORT_ADAPTER_FILES if not (adapter_path / name).is_file()]
    present_files = [name for name in REQUIRED_REPORT_ADAPTER_FILES if (adapter_path / name).is_file()]
    total_size = 0
    if adapter_path.exists():
        total_size = sum(path.stat().st_size for path in adapter_path.rglob("*") if path.is_file())
    return {
        "adapter_dir": str(adapter_path),
        "exists": adapter_path.exists(),
        "complete": adapter_path.exists() and not missing_files,
        "required_files": list(REQUIRED_REPORT_ADAPTER_FILES),
        "present_files": present_files,
        "missing_files": missing_files,
        "total_size_bytes": total_size,
    }


def is_report_adapter_complete(adapter_dir: str | Path) -> bool:
    return bool(adapter_status(adapter_dir)["complete"])
