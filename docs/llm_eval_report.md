# RAG/LLM Evaluation

The report-generation module is evaluated as an AI application, not only as text generation.

## Checks

- Retrieval coverage over README, docs, metrics JSON, run summary, and demo labels.
- Grounding of numeric metrics against `qwen7b_report_context.json`.
- Required sections: project overview, results, and limitations.
- Forbidden claims: SOTA, state-of-the-art, production-ready insurance assessment.

## Command

```bash
python -m vehicle_damage_pipeline.eval.run_llm_eval ^
  --context reports/qwen7b_report_context.json ^
  --report reports/qwen7b_final_report.md ^
  --knowledge-root . ^
  --output-json reports/llm_eval_summary.json ^
  --output-markdown reports/llm_eval_summary.md
```

## Why This Matters

The goal is to show evaluation-driven AI engineering. The LLM report must be faithful to structured experiment evidence and must avoid overstating project readiness.
