from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from vehicle_damage_pipeline.paths import ensure_project_dirs


PER_CLASS_TEST_MAP50 = [
    {"class": "dent", "box_map50": 0.492, "mask_map50": 0.496},
    {"class": "scratch", "box_map50": 0.511, "mask_map50": 0.475},
    {"class": "crack", "box_map50": 0.211, "mask_map50": 0.219},
    {"class": "glass shatter", "box_map50": 0.978, "mask_map50": 0.978},
    {"class": "lamp broken", "box_map50": 0.790, "mask_map50": 0.778},
    {"class": "tire flat", "box_map50": 0.882, "mask_map50": 0.882},
]


def build_report_context(drive_root: str | Path, run_name: str = "yolo11n_seg") -> dict[str, Any]:
    paths = ensure_project_dirs(drive_root)
    train_root = paths["runs"] / "train" / run_name
    reports_root = paths["reports"]
    test_metrics_json = reports_root / f"{run_name}_test_metrics.json"
    run_summary_json = reports_root / f"{run_name}_run_summary.json"
    results_csv = train_root / "results.csv"
    if not test_metrics_json.exists():
        raise FileNotFoundError(f"Missing test metrics: {test_metrics_json}")
    if not run_summary_json.exists():
        raise FileNotFoundError(f"Missing run summary: {run_summary_json}")
    if not results_csv.exists():
        raise FileNotFoundError(f"Missing results CSV: {results_csv}")

    test_metrics = json.loads(test_metrics_json.read_text(encoding="utf-8"))
    run_summary = json.loads(run_summary_json.read_text(encoding="utf-8"))
    results_df = pd.read_csv(results_csv)
    last_row = results_df.iloc[-1].to_dict()
    best_box_row = results_df.iloc[results_df["metrics/mAP50(B)"].idxmax()].to_dict()
    best_mask_row = results_df.iloc[results_df["metrics/mAP50(M)"].idxmax()].to_dict()
    demo_root = paths["runs"] / "predict" / "demo"
    demo_images = sorted([p.name for p in demo_root.glob("*.jpg")]) if demo_root.exists() else []
    demo_labels = sorted([p.name for p in (demo_root / "labels").glob("*.txt")]) if (demo_root / "labels").exists() else []
    context = {
        "project": "AI-Powered Vehicle Damage Assessment Pipeline",
        "dataset": "CarDD car damage detection and instance segmentation dataset",
        "task": "vehicle damage detection, instance segmentation, and report generation",
        "model": "YOLO11n-seg",
        "llm_report_model": "Qwen/Qwen2.5-7B-Instruct with QLoRA adapter when available",
        "training_setup": {
            "epochs": 100,
            "imgsz": 1024,
            "batch": 7,
            "gpu": "Colab L4",
            "training_time_hours": 4.109,
            "optimizer": "AdamW",
            "augmentations": ["multi_scale", "mosaic", "mixup", "copy_paste"],
        },
        "test_metrics": test_metrics,
        "last_epoch_metrics": last_row,
        "best_box_epoch": best_box_row,
        "best_mask_epoch": best_mask_row,
        "per_class_test_map50": PER_CLASS_TEST_MAP50,
        "demo_images": demo_images,
        "demo_labels": demo_labels,
        "artifacts": {
            "best_pt": run_summary.get("best_pt"),
            "last_pt": run_summary.get("last_pt"),
            "onnx": str(paths["exports"] / "best.onnx"),
            "test_metrics_json": str(test_metrics_json),
            "demo_dir": str(demo_root),
        },
    }
    return context


def main() -> None:
    parser = argparse.ArgumentParser(description="Build report context JSON from Drive artifacts.")
    parser.add_argument("--drive-root", default="/content/drive/MyDrive/CarDD_YOLO11")
    parser.add_argument("--run-name", default="yolo11n_seg")
    parser.add_argument("--output")
    args = parser.parse_args()
    context = build_report_context(args.drive_root, run_name=args.run_name)
    output = Path(args.output) if args.output else Path(args.drive_root) / "reports" / "qwen7b_report_context.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(context, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Context saved:", output)
    print(json.dumps(context["test_metrics"], indent=2))


if __name__ == "__main__":
    main()
