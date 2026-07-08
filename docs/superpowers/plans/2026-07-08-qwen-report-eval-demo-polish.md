# Qwen Report Eval Demo Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Qwen-backed report path pass LLM evaluation with the latest metrics, then make single-image Gradio/PDF demo output concise, grounded, and safe for interview presentation.

**Architecture:** Keep YOLO training and artifact generation untouched. Add deterministic report validation and fallback around Qwen output, update retrieval/docs metrics to the latest context, constrain image-level assessment reports, and hide full mask polygons from the primary demo surface while preserving structured output for debugging.

**Tech Stack:** Python package under `src/vehicle_damage_pipeline`, pytest, Gradio, Qwen2.5-7B LoRA adapter, lightweight TF-IDF retrieval/eval.

---

## File Structure

- Modify `src/vehicle_damage_pipeline/report/generate.py`
  - Add post-generation validation for Qwen project reports.
  - Fall back to the deterministic template when Qwen output misses metrics, required sections, or forbidden-claim checks.
  - Include validation/fallback metadata in the `.metadata.json` output.

- Modify `src/vehicle_damage_pipeline/eval/rag.py`
  - Replace old retrieval checks for `0.644` and `0.638` with latest public display metrics `0.6746` and `0.6712`.

- Modify `src/vehicle_damage_pipeline/llm/qwen_reporter.py`
  - Tighten assessment-report prompt constraints so Qwen cannot infer severity, car-body position, repair advice, insurance conclusions, or "no review needed" claims.

- Modify `src/vehicle_damage_pipeline/service/predictor.py`
  - Replace image-level template report text with a fact-only report.
  - Add Qwen assessment report validation and fallback to the safe template.

- Modify `src/vehicle_damage_pipeline/vision/formatting.py`
  - Add a compact display formatter that replaces full `mask_polygon` arrays with point counts for demo/PDF display.

- Modify `src/vehicle_damage_pipeline/service/gradio_app.py`
  - Show compact JSON and report in the primary view.
  - Put full JSON in a closed accordion for debugging.

- Modify `src/vehicle_damage_pipeline/service/colab_public_demo.py`
  - Mirror the local Gradio display changes for Colab public demo.

- Modify docs:
  - `README.md`
  - `docs/model_card.md`
  - `docs/experiment_card.md`
  - `docs/results_summary.md`
  - `docs/portfolio_pitch_zh.md`
  - `docs/cardd_resume_interview_report_zh.md`
  - Keep `docs/project_handoff_2026-07-08.md` as historical handoff; do not rewrite its old-vs-new comparison section except if adding a note.

- Modify tests:
  - `tests/test_qwen_report_workflow.py`
  - `tests/test_llm_eval.py`
  - `tests/test_assessment_report.py`
  - `tests/test_prediction_format.py`

## Preconditions

- Do not retrain YOLO.
- Do not delete or rewrite Drive artifacts.
- Do not commit CarDD raw data, weights, adapters, ONNX files, generated PDF demos, or private Drive links.
- Baseline before this plan was clean:

```powershell
python -m pytest tests -q -p no:cacheprovider
# Expected: 15 passed

python -m compileall src scripts
# Expected: successful compile listing, no SyntaxError
```

### Task 1: Validate Qwen Project Reports And Fall Back Safely

**Files:**
- Modify: `tests/test_qwen_report_workflow.py`
- Modify: `src/vehicle_damage_pipeline/report/generate.py`

- [ ] **Step 1: Add failing test for invalid Qwen project output**

Append this test to `tests/test_qwen_report_workflow.py`:

```python
def test_project_report_falls_back_when_qwen_output_fails_eval():
    from vehicle_damage_pipeline.report.generate import generate_report_from_context

    root = _fresh_case("report_validation_fallback")
    adapter = _make_complete_adapter(root)
    context = {
        "project": {"name": "AI-Powered Vehicle Damage Assessment Pipeline"},
        "dataset": {"name": "CarDD"},
        "test_metrics": {
            "metrics/mAP50(B)": 0.6745857662514867,
            "metrics/mAP50-95(B)": 0.5111031193401403,
            "metrics/mAP50(M)": 0.6711594915715345,
            "metrics/mAP50-95(M)": 0.49173212749837997,
        },
    }

    result = generate_report_from_context(
        context,
        adapter_dir=adapter,
        qwen_generate_fn=lambda **_: "# Qwen report\n\nOnly project overview.",
    )

    assert result.metadata["backend"] == "template"
    assert result.metadata["requested_backend"] == "qwen"
    assert result.metadata["fallback_reason"] == "qwen_report_validation_failed"
    assert result.metadata["qwen_validation"]["passed"] is False
    assert "box mAP50 为 0.675" in result.text
    assert "mask mAP50 为 0.671" in result.text
    assert "## 测试结果" in result.text
    assert "## 局限性与下一步" in result.text
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```powershell
python -m pytest tests/test_qwen_report_workflow.py::test_project_report_falls_back_when_qwen_output_fails_eval -q -p no:cacheprovider
```

Expected: FAIL because `generate_report_from_context()` currently returns Qwen text without validating required metrics/sections.

- [ ] **Step 3: Implement project report validation fallback**

In `src/vehicle_damage_pipeline/report/generate.py`, add the import:

```python
from vehicle_damage_pipeline.eval.llm_eval import evaluate_report
```

Then replace the Qwen return block in `generate_report_from_context()` with this logic after `text = generator(...)`:

```python
    validation = evaluate_report(context, text)
    if not validation["passed"] and fallback_to_template:
        return GeneratedReport(
            text=generate_template_report(context, language=language),
            metadata={
                "backend": "template",
                "requested_backend": "qwen",
                "fallback_reason": "qwen_report_validation_failed",
                "qwen_backend": "qwen_adapter",
                "adapter_dir": str(resolved_adapter),
                "adapter_complete": is_report_adapter_complete(resolved_adapter),
                "model_id": model_id,
                "language": language,
                "qwen_validation": validation,
            },
        )
    if not validation["passed"] and not fallback_to_template:
        raise ValueError(f"Qwen report failed validation: {validation}")
    return GeneratedReport(
        text=text,
        metadata={
            "backend": "qwen_adapter",
            "requested_backend": "qwen",
            "adapter_dir": str(resolved_adapter),
            "adapter_complete": is_report_adapter_complete(resolved_adapter),
            "model_id": model_id,
            "language": language,
            "qwen_validation": validation,
        },
    )
```

- [ ] **Step 4: Run focused tests**

Run:

```powershell
python -m pytest tests/test_qwen_report_workflow.py -q -p no:cacheprovider
```

Expected: all tests in `tests/test_qwen_report_workflow.py` pass.

- [ ] **Step 5: Commit**

```powershell
git add src/vehicle_damage_pipeline/report/generate.py tests/test_qwen_report_workflow.py
git commit -m "fix: validate qwen project reports before accepting output"
```

### Task 2: Update LLM Eval Retrieval And Public Metrics

**Files:**
- Modify: `tests/test_llm_eval.py`
- Modify: `src/vehicle_damage_pipeline/eval/rag.py`
- Modify: `README.md`
- Modify: `docs/model_card.md`
- Modify: `docs/experiment_card.md`
- Modify: `docs/results_summary.md`
- Modify: `docs/portfolio_pitch_zh.md`
- Modify: `docs/cardd_resume_interview_report_zh.md`

- [ ] **Step 1: Update LLM eval metric tests to latest context**

In `tests/test_llm_eval.py`, replace old metric values and report mentions:

```python
"metrics/mAP50(B)": 0.6745857662514867,
"metrics/mAP50(M)": 0.6711594915715345,
```

Use this report sentence in the passing test:

```python
本项目在测试集上取得 box mAP50 0.675 和 mask mAP50 0.671。
```

- [ ] **Step 2: Run test and verify current behavior**

Run:

```powershell
python -m pytest tests/test_llm_eval.py -q -p no:cacheprovider
```

Expected: PASS after the test data update, because `evaluate_report()` reads metrics from the provided context.

- [ ] **Step 3: Update retrieval checks**

In `src/vehicle_damage_pipeline/eval/rag.py`, replace:

```python
"box_map50": "0.644",
"mask_map50": "0.638",
```

with:

```python
"box_map50": "0.6746",
"mask_map50": "0.6712",
```

- [ ] **Step 4: Update public docs with latest metric block**

Use this canonical block for public display:

```text
Box precision:  0.6717
Box recall:     0.6374
Box mAP50:      0.6746
Box mAP50-95:   0.5111

Mask precision: 0.6795
Mask recall:    0.6242
Mask mAP50:     0.6712
Mask mAP50-95:  0.4917
```

For short resume/pitch wording, use:

```text
box mAP50 0.6746, mask mAP50 0.6712
```

Do not remove historical old-metric comparison lines from `docs/project_handoff_2026-07-08.md`.

- [ ] **Step 5: Search for stale public metrics**

Run:

```powershell
rg -n "0\.644|0\.638|0\.631|0\.626" README.md docs src tests notebooks scripts
```

Expected: remaining hits are only in historical comparison sections or old plan files where the text explicitly says they are old values.

- [ ] **Step 6: Run focused tests**

Run:

```powershell
python -m pytest tests/test_llm_eval.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add README.md docs src/vehicle_damage_pipeline/eval/rag.py tests/test_llm_eval.py
git commit -m "docs: update vehicle damage metrics across public materials"
```

### Task 3: Constrain Single-Image Assessment Reports

**Files:**
- Modify: `tests/test_assessment_report.py`
- Modify: `tests/test_qwen_report_workflow.py`
- Modify: `src/vehicle_damage_pipeline/service/predictor.py`
- Modify: `src/vehicle_damage_pipeline/llm/qwen_reporter.py`

- [ ] **Step 1: Add failing tests for safe template wording**

Replace the core assertions in `test_assessment_report_adds_plain_language_explanation()` with:

```python
    assert "车辆损伤自动检测摘要" in report
    assert "检测到 dent（凹陷）候选区域" in report
    assert "置信度 0.680" in report
    assert "位置框 [282.0, 244.1, 621.1, 529.7]" in report
    assert "建议结合原图、更多角度照片和人工复核判断" in report
    assert "该输出仅用于辅助评估" in report
    assert "维修" not in report
    assert "严重" not in report
    assert "无需复核" not in report
```

Replace the empty-detection assertions with:

```python
    assert "当前阈值下未发现高置信度损伤候选" in report
    assert "建议结合原图、更多角度照片和人工复核判断" in report
    assert "无需复核" not in report
    assert "确认无损伤" not in report
```

- [ ] **Step 2: Add failing test for unsafe Qwen image report fallback**

Append this test to `tests/test_qwen_report_workflow.py`:

```python
def test_assessment_report_falls_back_when_qwen_adds_unsupported_claims():
    from vehicle_damage_pipeline.service.predictor import generate_assessment_report

    root = _fresh_case("assessment_validation_fallback")
    adapter = _make_complete_adapter(root)
    prediction = {
        "detections": [
            {
                "class_name": "dent",
                "confidence": 0.68,
                "bbox_xyxy": [282.0, 244.1, 621.1, 529.7],
            }
        ]
    }

    report = generate_assessment_report(
        prediction,
        backend="qwen",
        adapter_dir=adapter,
        qwen_generate_fn=lambda **_: "位于车身左侧中部，面积较大且严重，建议维修。",
    )

    assert "检测到 dent（凹陷）候选区域" in report
    assert "车身左侧" not in report
    assert "面积较大" not in report
    assert "严重" not in report
    assert "维修" not in report
```

- [ ] **Step 3: Run focused tests and verify failures**

Run:

```powershell
python -m pytest tests/test_assessment_report.py tests/test_qwen_report_workflow.py -q -p no:cacheprovider
```

Expected: FAIL because the current template includes repair-style advice and Qwen image output is accepted without validation.

- [ ] **Step 4: Replace assessment template with fact-only wording**

In `src/vehicle_damage_pipeline/service/predictor.py`, replace `CLASS_REPORT_GUIDANCE` with:

```python
CLASS_LABELS_ZH = {
    "dent": "凹陷",
    "scratch": "划痕",
    "crack": "裂纹",
    "glass shatter": "玻璃破碎",
    "lamp broken": "灯具破损",
    "tire flat": "轮胎亏气",
}

ASSESSMENT_DISCLAIMER = "该输出仅用于辅助评估，不构成生产级保险定损结论。"
ASSESSMENT_REVIEW_NOTE = "建议结合原图、更多角度照片和人工复核判断。"
ASSESSMENT_FORBIDDEN_TERMS = [
    "无需复核",
    "确认无损伤",
    "车辆状态正常",
    "严重",
    "轻微",
    "大面积",
    "面积较大",
    "车身左侧",
    "车身右侧",
    "车头",
    "车尾",
    "维修",
    "报价",
    "理赔",
    "赔付",
]
```

Add:

```python
def validate_assessment_report_text(text: str) -> list[str]:
    return [term for term in ASSESSMENT_FORBIDDEN_TERMS if term in text]
```

Replace `generate_template_assessment_report()` with:

```python
def generate_template_assessment_report(prediction: dict[str, Any]) -> str:
    detections = prediction.get("detections", [])
    if not detections:
        return "\n".join(
            [
                "车辆损伤自动检测摘要：",
                "- 当前阈值下未发现高置信度损伤候选；该结果不等同于确认无损伤。",
                f"- {ASSESSMENT_REVIEW_NOTE}",
                "",
                ASSESSMENT_DISCLAIMER,
            ]
        )

    lines = ["车辆损伤自动检测摘要："]
    for item in detections:
        class_name = str(item.get("class_name", "unknown"))
        label = CLASS_LABELS_ZH.get(class_name, class_name)
        confidence = float(item.get("confidence", 0.0))
        bbox = item.get("bbox_xyxy", [])
        lines.append(
            f"- 检测到 {class_name}（{label}）候选区域，置信度 {confidence:.3f}，位置框 {bbox}。"
        )
    lines.append(f"- 当前输出仅反映模型在该图像上的检测结果；{ASSESSMENT_REVIEW_NOTE}")
    lines.extend(["", ASSESSMENT_DISCLAIMER])
    return "\n".join(lines)
```

In `generate_assessment_report()`, after Qwen text is generated, add:

```python
    violations = validate_assessment_report_text(text)
    if violations and fallback_to_template:
        return generate_template_assessment_report(prediction)
    if violations:
        raise ValueError(f"Qwen assessment report failed validation: {violations}")
```

- [ ] **Step 5: Tighten Qwen image prompt**

In `src/vehicle_damage_pipeline/llm/qwen_reporter.py`, replace the assessment user prompt body with:

```python
    user = (
        f"请基于下面车辆损伤检测 JSON 生成一段 {language_instruction} 辅助评估报告。\n"
        "只能引用 JSON 中的类别、置信度和 bbox。必须包含免责声明：该输出仅用于辅助评估，不构成生产级保险定损结论。\n"
        "禁止推断车身部位、损伤严重程度、损伤面积、维修方案、维修费用、保险理赔、事故责任或是否无需复核。\n"
        "如果没有 detections，只能说明当前阈值下未发现高置信度损伤候选，并建议结合原图、更多角度照片和人工复核判断。\n\n"
        f"Prediction JSON:\n{json.dumps(prediction, ensure_ascii=False, indent=2)}"
    )
```

- [ ] **Step 6: Run focused tests**

Run:

```powershell
python -m pytest tests/test_assessment_report.py tests/test_qwen_report_workflow.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add src/vehicle_damage_pipeline/service/predictor.py src/vehicle_damage_pipeline/llm/qwen_reporter.py tests/test_assessment_report.py tests/test_qwen_report_workflow.py
git commit -m "fix: constrain single image assessment reports"
```

### Task 4: Hide Full Mask Polygons From Primary Demo Output

**Files:**
- Modify: `tests/test_prediction_format.py`
- Modify: `src/vehicle_damage_pipeline/vision/formatting.py`
- Modify: `src/vehicle_damage_pipeline/service/gradio_app.py`
- Modify: `src/vehicle_damage_pipeline/service/colab_public_demo.py`
- Modify: `docs/api_usage.md`

- [ ] **Step 1: Add failing compact display formatter test**

Append this test to `tests/test_prediction_format.py`:

```python
def test_compact_prediction_response_hides_full_mask_polygon():
    from vehicle_damage_pipeline.vision.formatting import compact_prediction_for_display

    response = {
        "image_name": "demo.jpg",
        "detections": [
            {
                "class_name": "dent",
                "confidence": 0.68,
                "bbox_xyxy": [282.0, 244.1, 621.1, 529.7],
                "mask_polygon": [[1, 2], [3, 4], [5, 6], [7, 8]],
            }
        ],
    }

    compact = compact_prediction_for_display(response)

    assert "mask_polygon" not in compact["detections"][0]
    assert compact["detections"][0]["mask_polygon_points"] == 4
    assert compact["detections"][0]["mask_polygon_hidden"] is True
```

- [ ] **Step 2: Run test and verify failure**

Run:

```powershell
python -m pytest tests/test_prediction_format.py::test_compact_prediction_response_hides_full_mask_polygon -q -p no:cacheprovider
```

Expected: FAIL because `compact_prediction_for_display()` does not exist.

- [ ] **Step 3: Implement compact display formatter**

In `src/vehicle_damage_pipeline/vision/formatting.py`, add:

```python
def compact_prediction_for_display(prediction: dict[str, Any]) -> dict[str, Any]:
    compact = dict(prediction)
    compact_detections = []
    for detection in prediction.get("detections", []):
        item = dict(detection)
        polygon = item.pop("mask_polygon", None)
        if polygon is not None:
            item["mask_polygon_points"] = len(polygon)
            item["mask_polygon_hidden"] = True
        compact_detections.append(item)
    compact["detections"] = compact_detections
    return compact
```

- [ ] **Step 4: Update local Gradio UI outputs**

In `src/vehicle_damage_pipeline/service/gradio_app.py`, import:

```python
from vehicle_damage_pipeline.vision.formatting import compact_prediction_for_display
```

Change `run()` to return compact JSON, report, and full JSON:

```python
    def run(image_path: str):
        prediction = predictor.predict_file(Path(image_path))
        report_text = generate_assessment_report(
            prediction,
            backend=report_backend,
            adapter_dir=adapter_dir,
            model_id=model_id,
        )
        compact_json = json.dumps(compact_prediction_for_display(prediction), indent=2, ensure_ascii=False)
        full_json = json.dumps(prediction, indent=2, ensure_ascii=False)
        return compact_json, report_text, full_json
```

Replace the output widgets with:

```python
        compact_json = gr.Code(label="Compact detection summary", language="json")
        report = gr.Textbox(label="Assessment report", lines=8)
        with gr.Accordion("Full prediction JSON for debugging", open=False):
            output_json = gr.Code(label="Full prediction JSON", language="json")
        run_button.click(run, inputs=image, outputs=[compact_json, report, output_json])
```

- [ ] **Step 5: Update Colab public Gradio UI outputs**

Apply the same import, `run()` return shape, and widget layout in `src/vehicle_damage_pipeline/service/colab_public_demo.py`.

- [ ] **Step 6: Update API usage docs**

In `docs/api_usage.md`, add this note under the Gradio section:

```markdown
The primary Gradio view shows a compact detection summary. Full `mask_polygon` arrays are kept inside a closed debugging accordion so browser/PDF demos do not expand into long coordinate dumps.
```

- [ ] **Step 7: Run focused tests**

Run:

```powershell
python -m pytest tests/test_prediction_format.py tests/test_colab_public_demo.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 8: Commit**

```powershell
git add src/vehicle_damage_pipeline/vision/formatting.py src/vehicle_damage_pipeline/service/gradio_app.py src/vehicle_damage_pipeline/service/colab_public_demo.py tests/test_prediction_format.py docs/api_usage.md
git commit -m "fix: compact demo prediction display"
```

### Task 5: Final Verification And Handoff Update

**Files:**
- Modify if needed: `docs/project_handoff_2026-07-08.md`
- Modify if needed: `notebooks/05_colab_qwen_report_adapter_workflow.ipynb`

- [ ] **Step 1: Run full local verification**

Run:

```powershell
python -m pytest tests -q -p no:cacheprovider
python -m compileall src scripts
```

Expected:

```text
15 passed
```

The exact count may increase if new tests were added; all tests must pass.

- [ ] **Step 2: Run stale metric scan**

Run:

```powershell
rg -n "0\.644|0\.638|0\.631|0\.626" README.md docs src tests notebooks scripts
```

Expected: stale values remain only in historical/old-plan context that explicitly labels them as old.

- [ ] **Step 3: Optional Colab verification**

When Drive is mounted and artifacts are present, rerun only the report/eval cells in `notebooks/05_colab_qwen_report_adapter_workflow.ipynb`:

```text
Build report context
Generate Qwen report
Run RAG/LLM evaluation
Display report/eval outputs
Launch Gradio demo if needed
```

Expected:

```json
{
  "passed": true
}
```

for `reports/llm_eval_summary.json`, unless the notebook intentionally tests an invalid Qwen output path.

- [ ] **Step 4: Update handoff note if Colab verification is completed**

If Colab verification is run and passes, append a dated note to `docs/project_handoff_2026-07-08.md`:

```markdown
## 2026-07-08 Follow-up Verification

- Qwen project-report output is now validated before acceptance; invalid Qwen output falls back to the deterministic template.
- `llm_eval_summary.json` passes with latest metrics: box mAP50 0.6746, mask mAP50 0.6712.
- Gradio primary output hides full `mask_polygon` arrays and keeps them in a debugging section.
- Single-image reports avoid severity, car-body-position, repair, insurance, and no-review-needed claims.
```

- [ ] **Step 5: Commit final documentation note if changed**

```powershell
git add docs/project_handoff_2026-07-08.md notebooks/05_colab_qwen_report_adapter_workflow.ipynb
git commit -m "docs: record qwen report eval polish verification"
```

Only run this commit if one of those files actually changed.

## Self-Review

Spec coverage:
- P0 is covered by Task 1 and Task 2: project reports are validated/fallback-safe, eval uses latest context metrics, and local tests prove the behavior.
- P1 is covered by Task 2: public docs and retrieval checks move from old `0.644/0.638` to latest `0.6746/0.6712`.
- P2 is covered by Task 3 and Task 4: single-image report hallucination risk is constrained, no-damage wording is softened, and primary demo output no longer dumps full mask polygons.
- Red lines are preserved: no YOLO retraining, no Drive artifact deletion, no data/weights/adapter/ONNX commits, no SOTA or production insurance claims.

Placeholder scan:
- No TODO/TBD/fill-later steps remain.
- Each code-changing step names exact files, snippets, commands, and expected outcomes.

Type consistency:
- `GeneratedReport.metadata` remains a `dict[str, Any]`.
- `compact_prediction_for_display()` accepts and returns `dict[str, Any]`.
- Assessment report validation returns `list[str]` and is used only for fallback decisions.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-08-qwen-report-eval-demo-polish.md`. Two execution options:

1. Subagent-Driven (recommended) - dispatch a fresh subagent per task, review between tasks, fast iteration.
2. Inline Execution - execute tasks in this session using the plan, batch execution with checkpoints.

Recommended first execution target: Task 1, then immediately Task 2. That gets `llm_eval_summary` onto the path to PASS before polishing the demo layer.
