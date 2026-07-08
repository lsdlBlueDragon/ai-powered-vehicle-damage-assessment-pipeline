from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_QWEN_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
REPORT_SYSTEM_PROMPT = (
    "你是车辆损伤 AI 工程项目的技术报告助手。只允许使用用户提供的结构化证据，"
    "不要编造指标、SOTA、生产部署或保险定损结论。输出要专业、简洁、可面试展示。"
)


def build_project_report_messages(context: dict[str, Any], language: str = "Chinese") -> list[dict[str, str]]:
    language_instruction = "中文" if language.lower().startswith("ch") else "English"
    user = (
        f"请基于下面 JSON 证据生成一份 {language_instruction} Markdown 项目报告。\n"
        "必须包含：项目概览、数据与任务、方法、测试结果、RAG/LLM 评估、局限性与下一步。\n"
        "测试指标必须忠实引用 JSON 中的 box/mask mAP50 与 mAP50-95。\n\n"
        f"JSON evidence:\n{json.dumps(context, ensure_ascii=False, indent=2)}"
    )
    return [
        {"role": "system", "content": REPORT_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_assessment_report_messages(prediction: dict[str, Any], language: str = "Chinese") -> list[dict[str, str]]:
    language_instruction = "中文" if language.lower().startswith("ch") else "English"
    user = (
        f"请基于下面车辆损伤检测 JSON 生成一段 {language_instruction} 辅助评估报告。\n"
        "只允许使用 JSON 中的 image_name、class_name、confidence、bbox_xyxy 和检测数量。"
        "可以给人工复核提示，但不要推断车身部位、车头车尾方向、损伤面积、严重程度、维修方案、费用、责任或理赔结论。"
        "输出应包含自然语言摘要，并列出每处损伤的类别、置信度和 bbox。\n\n"
        f"Prediction JSON:\n{json.dumps(prediction, ensure_ascii=False, indent=2)}"
    )
    return [
        {"role": "system", "content": REPORT_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def generate_qwen_text(
    *,
    messages: list[dict[str, str]],
    model_id: str = DEFAULT_QWEN_MODEL_ID,
    adapter_dir: str | Path | None = None,
    max_new_tokens: int = 900,
    temperature: float = 0.2,
    load_in_4bit: bool = True,
) -> str:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer_source = str(adapter_dir) if adapter_dir else model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    if adapter_dir:
        model = PeftModel.from_pretrained(model, str(adapter_dir))

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    do_sample = temperature > 0
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
        top_p=0.9 if do_sample else 1.0,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = generated[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
