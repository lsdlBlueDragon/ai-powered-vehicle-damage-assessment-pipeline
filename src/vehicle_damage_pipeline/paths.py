from __future__ import annotations

from pathlib import Path


DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive/CarDD_YOLO11")


def project_paths(drive_root: str | Path = DEFAULT_DRIVE_ROOT) -> dict[str, Path]:
    root = Path(drive_root)
    return {
        "drive_root": root,
        "data_raw": root / "data_raw",
        "data_coco": root / "data_coco",
        "data_yolo": root / "data_yolo",
        "runs": root / "runs",
        "reports": root / "reports",
        "exports": root / "exports",
        "backups": root / "backups",
        "demo_images": root / "demo_images",
        "llm_adapters": root / "llm_adapters",
        "external_data": root / "external_data",
    }


def ensure_project_dirs(drive_root: str | Path = DEFAULT_DRIVE_ROOT) -> dict[str, Path]:
    paths = project_paths(drive_root)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
