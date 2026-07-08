from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


DEFAULT_REQUIRED_SECTIONS = ["项目概览", "结果", "局限性"]
FORBIDDEN_CLAIMS = [
    "sota",
    "state-of-the-art",
    "production-ready insurance assessment",
    "生产级保险定损",
]


def _metric_value(context: dict[str, Any], key: str) -> float | None:
    metrics = context.get("test_metrics", {})
    value = metrics.get(key)
    return float(value) if value is not None else None


def _metric_is_mentioned(report: str, value: float | None) -> bool:
    if value is None:
        return False
    rounded = f"{value:.3f}"
    percent = f"{value * 100:.1f}"
    return rounded in report or percent in report


def _section_present(report: str, section: str) -> bool:
    pattern = re.compile(rf"^#+\s*.*{re.escape(section)}.*$", re.MULTILINE)
    return bool(pattern.search(report))


def _is_negated_claim(prefix: str) -> bool:
    negation_markers = [
        "not ",
        "does not",
        "is not",
        "not a",
        "不是",
        "不属于",
        "不用于",
        "不作为",
        "不能",
        "不要",
        "不声称",
        "并非",
        "非",
    ]
    return any(marker in prefix for marker in negation_markers)


def _forbidden_claims(report: str) -> list[str]:
    text = report.lower()
    found = []
    for claim in FORBIDDEN_CLAIMS:
        needle = claim.lower()
        if needle not in text:
            continue
        index = text.find(needle)
        prefix = text[max(0, index - 12) : index]
        if _is_negated_claim(prefix):
            continue
        found.append(claim)
    return found


def evaluate_report(
    context: dict[str, Any],
    report: str,
    required_sections: list[str] | None = None,
) -> dict[str, Any]:
    sections = required_sections or DEFAULT_REQUIRED_SECTIONS
    box_map50 = _metric_value(context, "metrics/mAP50(B)")
    mask_map50 = _metric_value(context, "metrics/mAP50(M)")
    section_results = {section: _section_present(report, section) for section in sections}
    metric_mentions = {
        "box_map50": _metric_is_mentioned(report, box_map50),
        "mask_map50": _metric_is_mentioned(report, mask_map50),
    }
    forbidden = _forbidden_claims(report)
    return {
        "grounding": {
            "required_metric_mentions": metric_mentions,
            "grounded": all(metric_mentions.values()),
        },
        "quality": {
            "required_sections_present": section_results,
            "sections_complete": all(section_results.values()),
            "forbidden_claims": forbidden,
            "passes_forbidden_claim_check": not forbidden,
        },
        "passed": all(metric_mentions.values()) and all(section_results.values()) and not forbidden,
    }


def evaluate_report_files(
    context_path: str | Path,
    report_path: str | Path,
    output_json: str | Path | None = None,
    output_markdown: str | Path | None = None,
) -> dict[str, Any]:
    context = json.loads(Path(context_path).read_text(encoding="utf-8"))
    report = Path(report_path).read_text(encoding="utf-8")
    result = evaluate_report(context, report)
    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(output_json).write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if output_markdown:
        Path(output_markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(output_markdown).write_text(render_eval_markdown(result), encoding="utf-8")
    return result


def render_eval_markdown(result: dict[str, Any]) -> str:
    status = "PASS" if result["passed"] else "FAIL"
    lines = [
        "# LLM Report Evaluation",
        "",
        f"Overall: **{status}**",
        "",
        "## Grounding",
    ]
    for name, ok in result["grounding"]["required_metric_mentions"].items():
        lines.append(f"- {name}: {'PASS' if ok else 'FAIL'}")
    lines.extend(["", "## Report Quality"])
    for name, ok in result["quality"]["required_sections_present"].items():
        lines.append(f"- section {name}: {'PASS' if ok else 'FAIL'}")
    forbidden = result["quality"]["forbidden_claims"]
    lines.append(f"- forbidden claims: {', '.join(forbidden) if forbidden else 'none'}")
    return "\n".join(lines) + "\n"
