from __future__ import annotations

from pathlib import Path


BLOCKED_PARTS = {
    "data_raw",
    "data_coco",
    "data_yolo",
    "weights",
    "checkpoint-26",
}
BLOCKED_SUFFIXES = {".pt", ".pth", ".zip", ".tar", ".gz", ".7z"}
ALLOWED_ONNX = "exports/best.onnx"


def _normalized(path: Path) -> str:
    return path.as_posix().lower()


def should_include_backup_file(path: Path) -> bool:
    normalized = _normalized(path)
    if normalized == ALLOWED_ONNX:
        return True
    if any(part.lower() in BLOCKED_PARTS for part in path.parts):
        return False
    if path.suffix.lower() in BLOCKED_SUFFIXES:
        return False
    return True


def build_manifest_entries(
    *,
    files: list[Path],
    source_root: Path,
    backup_root: Path,
) -> list[dict[str, object]]:
    entries = []
    for file in files:
        relative = file.relative_to(source_root)
        restricted = not should_include_backup_file(relative)
        entries.append(
            {
                "relative_path": relative.as_posix(),
                "source": str(file),
                "destination": str(backup_root / relative),
                "size_bytes": file.stat().st_size,
                "restricted": restricted,
            }
        )
    return entries
