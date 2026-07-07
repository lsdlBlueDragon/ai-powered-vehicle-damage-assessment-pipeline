from vehicle_damage_pipeline.eval.llm_eval import evaluate_report


def test_evaluate_report_checks_grounded_metrics_and_required_sections():
    context = {
        "test_metrics": {
            "metrics/mAP50(B)": 0.6439247593491015,
            "metrics/mAP50(M)": 0.6380694210039354,
        }
    }
    report = """
# 车辆损伤检测报告

## 项目概览
本项目在测试集上取得 box mAP50 0.644 和 mask mAP50 0.638。

## 结果
测试指标来自结构化 context。

## 局限性
这不是生产级保险定损系统。
"""

    result = evaluate_report(context, report)

    assert result["grounding"]["required_metric_mentions"]["box_map50"] is True
    assert result["grounding"]["required_metric_mentions"]["mask_map50"] is True
    assert result["quality"]["required_sections_present"]["项目概览"] is True
    assert result["quality"]["required_sections_present"]["结果"] is True
    assert result["quality"]["required_sections_present"]["局限性"] is True
    assert result["quality"]["forbidden_claims"] == []


def test_evaluate_report_flags_missing_metrics_and_forbidden_claims():
    context = {
        "test_metrics": {
            "metrics/mAP50(B)": 0.6439247593491015,
            "metrics/mAP50(M)": 0.6380694210039354,
        }
    }
    report = "This production-ready insurance assessment system reaches SOTA performance."

    result = evaluate_report(context, report)

    assert result["grounding"]["required_metric_mentions"]["box_map50"] is False
    assert result["grounding"]["required_metric_mentions"]["mask_map50"] is False
    assert "sota" in result["quality"]["forbidden_claims"]
    assert "production-ready insurance assessment" in result["quality"]["forbidden_claims"]
