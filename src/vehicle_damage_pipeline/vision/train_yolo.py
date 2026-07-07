from __future__ import annotations

import argparse
import json
from pathlib import Path

from vehicle_damage_pipeline.paths import ensure_project_dirs


def train_yolo(
    *,
    drive_root: str | Path,
    model_name: str = "yolo11n-seg.pt",
    run_name: str = "yolo11n_seg",
    epochs: int = 100,
    imgsz: int = 1024,
    batch: int = 7,
    strict_resume: bool = False,
) -> Path:
    from ultralytics import YOLO

    paths = ensure_project_dirs(drive_root)
    data_yaml = paths["data_yolo"] / "cardd.yaml"
    data_ready = paths["data_yolo"] / "data_ready.json"
    if not data_ready.exists() or not data_yaml.exists():
        raise FileNotFoundError("Run prepare_cardd before training.")

    train_project = paths["runs"] / "train"
    last_pt = train_project / run_name / "weights" / "last.pt"
    train_kwargs = dict(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        optimizer="AdamW",
        amp=True,
        overlap_mask=True,
        multi_scale=True,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        patience=30,
        save_period=1,
        workers=2,
        seed=42,
        project=str(train_project),
        name=run_name,
        exist_ok=True,
    )
    if last_pt.exists():
        print("Continuing from:", last_pt)
        model = YOLO(str(last_pt))
        if strict_resume:
            model.train(resume=True)
        else:
            model.train(**train_kwargs)
    else:
        print("Starting from:", model_name)
        model = YOLO(model_name)
        model.train(**train_kwargs)
    return train_project / run_name


def evaluate_and_export(
    *,
    drive_root: str | Path,
    run_name: str = "yolo11n_seg",
    imgsz: int = 1024,
) -> dict[str, object]:
    from ultralytics import YOLO

    paths = ensure_project_dirs(drive_root)
    train_dir = paths["runs"] / "train" / run_name
    best_pt = train_dir / "weights" / "best.pt"
    last_pt = train_dir / "weights" / "last.pt"
    weights = best_pt if best_pt.exists() else last_pt
    if not weights.exists():
        raise FileNotFoundError("No best.pt or last.pt checkpoint found.")
    split = "test" if (paths["data_yolo"] / "images" / "test").exists() else "val"
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(paths["data_yolo"] / "cardd.yaml"),
        split=split,
        imgsz=imgsz,
        project=str(paths["runs"] / "val"),
        name=f"{run_name}_{split}",
        plots=True,
        exist_ok=True,
    )
    metrics_path = paths["reports"] / f"{run_name}_{split}_metrics.json"
    metrics_path.write_text(json.dumps(metrics.results_dict, indent=2), encoding="utf-8")
    exported_path = None
    try:
        exported = Path(model.export(format="onnx", imgsz=imgsz))
        target = paths["exports"] / exported.name
        if exported.exists() and exported.resolve() != target.resolve():
            import shutil

            shutil.copy2(exported, target)
        exported_path = target if target.exists() else exported
    except Exception as exc:  # pragma: no cover - depends on Colab export stack
        print("ONNX export skipped or failed:", exc)
    return {
        "weights": str(weights),
        "split": split,
        "metrics_path": str(metrics_path),
        "exported_path": str(exported_path) if exported_path else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate YOLO segmentation.")
    parser.add_argument("--drive-root", default="/content/drive/MyDrive/CarDD_YOLO11")
    parser.add_argument("--model", default="yolo11n-seg.pt")
    parser.add_argument("--run-name", default="yolo11n_seg")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=7)
    parser.add_argument("--strict-resume", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()
    if not args.skip_train:
        train_yolo(
            drive_root=args.drive_root,
            model_name=args.model,
            run_name=args.run_name,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            strict_resume=args.strict_resume,
        )
    if not args.skip_eval:
        print(json.dumps(evaluate_and_export(drive_root=args.drive_root, run_name=args.run_name, imgsz=args.imgsz), indent=2))


if __name__ == "__main__":
    main()
