import shutil
from pathlib import Path

from vehicle_damage_pipeline.backup.manifest import build_manifest_entries, should_include_backup_file


def test_backup_filter_excludes_restricted_large_artifacts():
    blocked = [
        Path("data_raw/CarDD_release.zip"),
        Path("data_coco/image.jpg"),
        Path("data_yolo/labels/train/a.txt"),
        Path("runs/train/yolo11n_seg/weights/best.pt"),
        Path("llm_adapters/qwen/checkpoint-26/optimizer.pt"),
    ]

    assert all(not should_include_backup_file(path) for path in blocked)


def test_backup_filter_allows_lightweight_evidence_artifacts():
    allowed = [
        Path("notebooks/01_train_cardd_yolo11_seg.ipynb"),
        Path("reports/qwen7b_final_report.md"),
        Path("runs/predict/demo/image0.jpg"),
        Path("exports/best.onnx"),
        Path("README.md"),
    ]

    assert all(should_include_backup_file(path) for path in allowed)


def test_manifest_entries_record_source_destination_size_and_restricted_flag():
    tmp_root = Path(".test_tmp") / "manifest"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    source = tmp_root / "reports" / "metrics.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}", encoding="utf-8")

    entries = build_manifest_entries(
        files=[source],
        source_root=tmp_root,
        backup_root=tmp_root / "backup",
    )

    assert entries == [
        {
            "relative_path": "reports/metrics.json",
            "source": str(source),
            "destination": str(tmp_root / "backup" / "reports" / "metrics.json"),
            "size_bytes": 2,
            "restricted": False,
        }
    ]
    shutil.rmtree(tmp_root)
