from vehicle_damage_pipeline.eval.rag import evaluate_retrieval


def test_retrieval_checks_use_latest_public_metrics():
    docs = [
        {
            "id": "README.md",
            "text": (
                "AI-Powered Vehicle Damage Assessment Pipeline\n"
                "Dataset: CarDD\n"
                "Model: YOLO11n-seg\n"
                "Box mAP50: 0.6746\n"
                "Mask mAP50: 0.6712\n"
                "Demo label: scratch\n"
                "Artifact: best.pt\n"
            ),
        }
    ]

    result = evaluate_retrieval(docs)

    assert result["checks"]["box_map50"]["query"] == "0.6746"
    assert result["checks"]["box_map50"]["hit"] is True
    assert result["checks"]["mask_map50"]["query"] == "0.6712"
    assert result["checks"]["mask_map50"]["hit"] is True


def test_retrieval_hit_uses_full_document_not_truncated_preview():
    long_prefix = "portfolio evidence " * 80
    docs = [
        {
            "id": "docs/long_metrics.md",
            "text": (
                f"{long_prefix}\n"
                "Dataset: CarDD\n"
                "Model: YOLO11n-seg\n"
                "Box mAP50: 0.6746\n"
                "Mask mAP50: 0.6712\n"
                "Demo label: scratch\n"
                "Artifact: best.pt\n"
            ),
        }
    ]

    result = evaluate_retrieval(docs)

    assert result["checks"]["box_map50"]["hit"] is True
    assert result["checks"]["mask_map50"]["hit"] is True
