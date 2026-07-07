from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from vehicle_damage_pipeline.backup.manifest import build_manifest_entries, should_include_backup_file


def _copy_tree_files(source_root: Path, backup_root: Path, patterns: list[str]) -> list[Path]:
    copied: list[Path] = []
    for pattern in patterns:
        for src in source_root.glob(pattern):
            if not src.is_file():
                continue
            relative = src.relative_to(source_root)
            if not should_include_backup_file(relative):
                continue
            dst = backup_root / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(src)
    return copied


def create_lightweight_backup(
    *,
    repo_root: str | Path,
    drive_root: str | Path,
    backup_root: str | Path,
    downloads_root: str | Path | None = None,
) -> dict[str, object]:
    repo_root = Path(repo_root)
    drive_root = Path(drive_root)
    backup_root = Path(backup_root)
    backup_root.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    copied.extend(
        _copy_tree_files(
            repo_root,
            backup_root / "repo_snapshot_docs",
            ["README.md", "requirements-colab.txt", "configs/*", "docs/**/*.md"],
        )
    )
    copied.extend(_copy_tree_files(drive_root, backup_root, ["reports/*", "runs/predict/demo/*", "runs/predict/demo/labels/*", "exports/best.onnx"]))
    if downloads_root:
        downloads = Path(downloads_root)
        notebook_dir = backup_root / "notebooks_ran"
        notebook_dir.mkdir(parents=True, exist_ok=True)
        for name in [
            "01_train_cardd_yolo11_seg.ipynb",
            "02_demo_cardd_yolo11_seg.ipynb",
            "03_finetune_qwen7b_report_lora.ipynb",
            "04_generate_llm_report_qwen7b.ipynb",
        ]:
            candidates = [downloads / name, drive_root / "notebooks" / name, repo_root / "notebooks" / name]
            src = next((candidate for candidate in candidates if candidate.exists()), None)
            if src:
                shutil.copy2(src, notebook_dir / name)
                copied.append(src)

    entries = []
    for src in copied:
        source_base = src.parents[len(src.parents) - 1] if False else src.parent
        entries.append(
            {
                "relative_path": src.name,
                "source": str(src),
                "destination": "see backup directory",
                "size_bytes": src.stat().st_size,
                "restricted": False,
            }
        )
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(repo_root),
        "drive_root": str(drive_root),
        "backup_root": str(backup_root),
        "contains_restricted_data": False,
        "entries": entries,
    }
    (backup_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create lightweight evidence backup.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--drive-root", required=True)
    parser.add_argument("--backup-root", required=True)
    parser.add_argument("--downloads-root")
    args = parser.parse_args()
    manifest = create_lightweight_backup(
        repo_root=args.repo_root,
        drive_root=args.drive_root,
        backup_root=args.backup_root,
        downloads_root=args.downloads_root,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
