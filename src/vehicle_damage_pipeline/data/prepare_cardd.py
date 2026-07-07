from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from vehicle_damage_pipeline.paths import ensure_project_dirs


DEFAULT_CLASS_NAMES = [
    "dent",
    "scratch",
    "crack",
    "glass shatter",
    "lamp broken",
    "tire flat",
]


def infer_split(path: Path) -> str | None:
    name = path.name.lower()
    if "train" in name:
        return "train"
    if "val" in name or "valid" in name:
        return "val"
    if "test" in name:
        return "test"
    return None


def extract_first_zip_if_needed(data_raw: Path, data_coco: Path) -> None:
    if any(data_coco.rglob("*.json")):
        return
    zips = sorted(data_raw.glob("*.zip"))
    if not zips:
        return
    print("Extracting:", zips[0])
    with zipfile.ZipFile(zips[0], "r") as zf:
        zf.extractall(data_coco)


def find_annotation_files(data_raw: Path, data_coco: Path) -> dict[str, Path]:
    files: list[Path] = []
    for root in [data_coco, data_raw]:
        files.extend(root.rglob("*.json"))
    split_to_file: dict[str, Path] = {}
    for file in sorted(files):
        split = infer_split(file)
        if split and split not in split_to_file:
            split_to_file[split] = file
    if "train" not in split_to_file:
        raise FileNotFoundError("No train annotation JSON found under data_raw or data_coco.")
    if "val" not in split_to_file and "test" not in split_to_file:
        raise FileNotFoundError("Need at least a val or test annotation JSON for evaluation.")
    return split_to_file


def build_image_index(data_raw: Path, data_coco: Path) -> dict[str, Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    index: dict[str, Path] = {}
    for root in [data_coco, data_raw]:
        for file in root.rglob("*"):
            if file.suffix.lower() in exts:
                index.setdefault(file.name, file)
                rel = str(file.relative_to(root)).replace("\\", "/")
                index.setdefault(rel, file)
    return index


def normalize_polygon(poly: list[float], width: int, height: int) -> list[float]:
    values = []
    for i, value in enumerate(poly):
        limit = width if i % 2 == 0 else height
        values.append(max(0.0, min(1.0, float(value) / float(limit))))
    return values


def _image_size(image: dict[str, Any], path: Path) -> tuple[int, int]:
    width = image.get("width")
    height = image.get("height")
    if width and height:
        return int(width), int(height)
    import cv2

    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    return int(w), int(h)


def convert_one_split(
    *,
    split: str,
    ann_file: Path,
    image_index: dict[str, Path],
    data_yolo: Path,
    split_ready_dir: Path,
    force: bool,
    class_names: list[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    coco = json.loads(ann_file.read_text(encoding="utf-8"))
    categories = sorted(coco["categories"], key=lambda x: x["id"])
    names = class_names or [category["name"] for category in categories]
    cat_to_cls = {category["id"]: idx for idx, category in enumerate(categories)}
    images = {image["id"]: image for image in coco["images"]}
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        anns_by_image[ann["image_id"]].append(ann)

    ready_file = split_ready_dir / f"{split}.json"
    if ready_file.exists() and not force:
        return json.loads(ready_file.read_text(encoding="utf-8")), names

    out_img_dir = data_yolo / "images" / split
    out_lbl_dir = data_yolo / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    split_ready_dir.mkdir(parents=True, exist_ok=True)

    copied = skipped = labels = 0
    missing: list[str] = []
    for image_id, image in tqdm(images.items(), desc=f"convert {split}"):
        file_name = image["file_name"]
        dst_img = out_img_dir / Path(file_name).name
        label_file = out_lbl_dir / (Path(file_name).stem + ".txt")
        if dst_img.exists() and label_file.exists() and not force:
            skipped += 1
            continue
        src = image_index.get(file_name) or image_index.get(Path(file_name).name)
        if src is None:
            missing.append(file_name)
            continue
        if not dst_img.exists() or force:
            shutil.copy2(src, dst_img)
            copied += 1
        width, height = _image_size(image, dst_img)
        lines = []
        for ann in anns_by_image.get(image_id, []):
            seg = ann.get("segmentation")
            if not isinstance(seg, list) or not seg:
                continue
            poly = max(seg, key=len)
            if len(poly) < 6:
                continue
            cls = cat_to_cls[ann["category_id"]]
            values = normalize_polygon(poly, width, height)
            lines.append(str(cls) + " " + " ".join(f"{value:.6f}" for value in values))
        label_file.write_text("\n".join(lines), encoding="utf-8")
        labels += len(lines)

    summary = {
        "images": len(images),
        "copied": copied,
        "skipped": skipped,
        "labels": labels,
        "missing": len(missing),
        "annotation_file": str(ann_file),
    }
    ready_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if missing:
        print(f"Warning: {len(missing)} images missing in {split}. First:", missing[:3])
    return summary, names


def prepare_cardd_dataset(drive_root: str | Path, force: bool = False) -> dict[str, Any]:
    paths = ensure_project_dirs(drive_root)
    data_ready = paths["data_yolo"] / "data_ready.json"
    data_yaml = paths["data_yolo"] / "cardd.yaml"
    split_ready_dir = paths["data_yolo"] / "_split_ready"
    if data_ready.exists() and data_yaml.exists() and not force:
        return json.loads(data_ready.read_text(encoding="utf-8"))

    extract_first_zip_if_needed(paths["data_raw"], paths["data_coco"])
    split_files = find_annotation_files(paths["data_raw"], paths["data_coco"])
    image_index = build_image_index(paths["data_raw"], paths["data_coco"])
    summary: dict[str, Any] = {}
    class_names = None
    for split, ann_file in split_files.items():
        summary[split], class_names = convert_one_split(
            split=split,
            ann_file=ann_file,
            image_index=image_index,
            data_yolo=paths["data_yolo"],
            split_ready_dir=split_ready_dir,
            force=force,
            class_names=class_names,
        )
    names = class_names or DEFAULT_CLASS_NAMES
    val_split = "val" if "val" in summary else "test"
    yaml_data = {
        "path": str(paths["data_yolo"]),
        "train": "images/train",
        "val": f"images/{val_split}",
        "names": {i: name for i, name in enumerate(names)},
    }
    if "test" in summary:
        yaml_data["test"] = "images/test"
    data_yaml.write_text(yaml.safe_dump(yaml_data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    payload = {"summary": summary, "names": names}
    data_ready.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CarDD YOLO segmentation data.")
    parser.add_argument("--drive-root", default="/content/drive/MyDrive/CarDD_YOLO11")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = prepare_cardd_dataset(args.drive_root, force=args.force)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
