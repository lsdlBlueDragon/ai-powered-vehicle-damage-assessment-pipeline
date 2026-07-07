from __future__ import annotations

import argparse
import json
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from vehicle_damage_pipeline.llm.adapter import adapter_status, report_adapter_dir
from vehicle_damage_pipeline.paths import ensure_project_dirs


TOTTO_URL = "https://storage.googleapis.com/totto-public/totto_data.zip"
TOTTO_TRAIN_MEMBER = "totto_data/totto_train_data.jsonl"
TOTTO_DEV_MEMBER = "totto_data/totto_dev_data.jsonl"
SYSTEM_PROMPT = (
    "You write faithful, concise technical report text from structured evidence. "
    "Use only the provided evidence and do not invent numbers or claims."
)


def format_messages(system: str, user: str, assistant: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def _linearize_totto(example: dict, max_cells: int = 24) -> str:
    table = example.get("table") or []
    parts = []
    if example.get("table_page_title"):
        parts.append(f"page_title: {example['table_page_title']}")
    if example.get("table_section_title"):
        parts.append(f"section_title: {example['table_section_title']}")
    for pair in (example.get("highlighted_cells") or [])[:max_cells]:
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        row, col = int(pair[0]), int(pair[1])
        try:
            value = str(table[row][col].get("value", "")).strip()
        except Exception:
            value = ""
        if value:
            parts.append(f"cell[{row},{col}]: {value}")
    return "\n".join(parts)


def _read_totto_rows(zip_path: Path, member: str, limit: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(member, "r") as handle:
            for raw in handle:
                example = json.loads(raw.decode("utf-8"))
                evidence = _linearize_totto(example)
                for ann in example.get("sentence_annotations") or []:
                    target = ann.get("final_sentence", "").strip()
                    if evidence and target:
                        rows.append({"linearized_input": evidence, "target": target})
                    if len(rows) >= limit:
                        return rows
    return rows


def _ensure_totto_zip(paths: dict[str, Path]) -> Path:
    cache_dir = paths["external_data"] / "totto"
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "totto_data.zip"
    if not zip_path.exists() or zip_path.stat().st_size == 0:
        print("Downloading ToTTo:", TOTTO_URL)
        urllib.request.urlretrieve(TOTTO_URL, zip_path)
    return zip_path


def build_sft_dataset(
    *,
    drive_root: str | Path,
    model_id: str,
    open_train_examples: int,
    open_eval_examples: int,
):
    from datasets import Dataset, concatenate_datasets
    from transformers import AutoTokenizer

    paths = ensure_project_dirs(drive_root)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    zip_path = _ensure_totto_zip(paths)
    train_rows = _read_totto_rows(zip_path, TOTTO_TRAIN_MEMBER, open_train_examples)
    eval_rows = _read_totto_rows(zip_path, TOTTO_DEV_MEMBER, open_eval_examples)
    train_dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(eval_rows)

    def to_messages(example):
        user = "Convert the following structured evidence into a concise factual report sentence.\n\nEvidence:\n" + example["linearized_input"]
        return {"messages": format_messages(SYSTEM_PROMPT, user, example["target"])}

    train_dataset = train_dataset.map(to_messages, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(to_messages, remove_columns=eval_dataset.column_names)

    cardd_examples = []
    metrics_path = paths["reports"] / "yolo11n_seg_test_metrics.json"
    summary_path = paths["reports"] / "yolo11n_seg_run_summary.json"
    results_csv = paths["runs"] / "train" / "yolo11n_seg" / "results.csv"
    if metrics_path.exists() and summary_path.exists() and results_csv.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        final_row = pd.read_csv(results_csv).iloc[-1].to_dict()
        evidence = json.dumps(
            {
                "model": "YOLO11n-seg",
                "dataset": "CarDD",
                "epochs": 100,
                "test_metrics": metrics,
                "artifacts": summary,
                "last_epoch_metrics": final_row,
            },
            indent=2,
        )
        response = (
            "The CarDD reproduction fine-tuned YOLO11n-seg for 100 epochs. "
            f"On the test split, box mAP50 was {metrics['metrics/mAP50(B)']:.3f} "
            f"and mask mAP50 was {metrics['metrics/mAP50(M)']:.3f}."
        )
        cardd_examples.append({"messages": format_messages(SYSTEM_PROMPT, "Write a project result paragraph from this evidence:\n\n" + evidence, response)})
    if cardd_examples:
        train_dataset = concatenate_datasets([train_dataset, Dataset.from_list(cardd_examples)])

    def add_text(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

    train_dataset = train_dataset.map(add_text)
    eval_dataset = eval_dataset.map(add_text)
    sft_data = paths["reports"] / "qwen7b_report_sft_data.jsonl"
    with sft_data.open("w", encoding="utf-8") as handle:
        for row in train_dataset:
            handle.write(json.dumps({"messages": row["messages"]}, ensure_ascii=False) + "\n")
    metadata = {
        "model_id": model_id,
        "open_dataset": "GEM/totto",
        "open_dataset_source_url": TOTTO_URL,
        "open_train_examples": open_train_examples,
        "open_eval_examples": open_eval_examples,
        "cardd_examples": len(cardd_examples),
        "train_rows": len(train_dataset),
        "eval_rows": len(eval_dataset),
        "loader": "direct_totto_jsonl_zip_no_dataset_script",
        "packing": False,
    }
    (paths["reports"] / "qwen7b_report_sft_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return train_dataset, eval_dataset, tokenizer, metadata


def finetune_report_lora(
    *,
    drive_root: str | Path,
    model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    train_examples: int = 1200,
    eval_examples: int = 120,
    epochs: int = 1,
    max_seq_length: int = 1024,
    force_retrain: bool = False,
) -> Path:
    paths = ensure_project_dirs(drive_root)
    adapter_dir = report_adapter_dir(drive_root)
    status = adapter_status(adapter_dir)
    if status["complete"] and not force_retrain:
        metadata = {
            "model_id": model_id,
            "adapter_dir": str(adapter_dir),
            "status": "skipped_existing_adapter",
            "reason": "Complete Qwen report LoRA adapter already exists in Drive.",
            "adapter_status": status,
            "packing": False,
        }
        (paths["reports"] / "qwen7b_report_sft_metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        return adapter_dir

    import torch
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    train_dataset, eval_dataset, tokenizer, metadata = build_sft_dataset(
        drive_root=drive_root,
        model_id=model_id,
        open_train_examples=train_examples,
        open_eval_examples=eval_examples,
    )
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    training_args = SFTConfig(
        output_dir=str(adapter_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        max_length=max_seq_length,
        packing=False,
        dataset_text_field="text",
        report_to="none",
        seed=42,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    metadata["adapter_dir"] = str(adapter_dir)
    metadata["status"] = "complete"
    (paths["reports"] / "qwen7b_report_sft_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return adapter_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen report adapter with QLoRA.")
    parser.add_argument("--drive-root", default="/content/drive/MyDrive/CarDD_YOLO11")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train-examples", type=int, default=1200)
    parser.add_argument("--eval-examples", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--build-dataset-only", action="store_true")
    parser.add_argument("--skip-if-complete", action="store_true", default=True)
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()
    if args.build_dataset_only:
        _, _, _, metadata = build_sft_dataset(
            drive_root=args.drive_root,
            model_id=args.model_id,
            open_train_examples=args.train_examples,
            open_eval_examples=args.eval_examples,
        )
        print(json.dumps(metadata, indent=2))
        return
    adapter_dir = finetune_report_lora(
        drive_root=args.drive_root,
        model_id=args.model_id,
        train_examples=args.train_examples,
        eval_examples=args.eval_examples,
        epochs=args.epochs,
        max_seq_length=args.max_seq_length,
        force_retrain=args.force_retrain,
    )
    print("Adapter saved:", adapter_dir)


if __name__ == "__main__":
    main()
