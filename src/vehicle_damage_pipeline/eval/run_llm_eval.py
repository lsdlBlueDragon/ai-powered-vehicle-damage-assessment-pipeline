from __future__ import annotations

import argparse
import json
from pathlib import Path

from vehicle_damage_pipeline.eval.llm_eval import evaluate_report_files, render_eval_markdown
from vehicle_damage_pipeline.eval.rag import evaluate_retrieval, load_knowledge_documents


def run_eval(
    *,
    context: str | Path,
    report: str | Path,
    knowledge_root: str | Path,
    output_json: str | Path,
    output_markdown: str | Path,
) -> dict[str, object]:
    report_eval = evaluate_report_files(context, report)
    docs = load_knowledge_documents(knowledge_root)
    retrieval_eval = evaluate_retrieval(docs)
    summary = {
        "report_eval": report_eval,
        "retrieval_eval": retrieval_eval,
        "passed": report_eval["passed"] and retrieval_eval["recall_at_5"] >= 0.67,
    }
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md = render_eval_markdown(report_eval)
    md += "\n## Retrieval\n"
    md += f"- recall@5: {retrieval_eval['recall_at_5']:.3f}\n"
    for name, check in retrieval_eval["checks"].items():
        md += f"- {name}: {'PASS' if check['hit'] else 'FAIL'} ({', '.join(check['retrieved'])})\n"
    Path(output_markdown).write_text(md, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG/LLM report evaluation.")
    parser.add_argument("--context", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--knowledge-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    args = parser.parse_args()
    result = run_eval(
        context=args.context,
        report=args.report,
        knowledge_root=args.knowledge_root,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
