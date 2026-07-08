from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from vehicle_damage_pipeline.llm.adapter import REPORT_ADAPTER_NAME
from vehicle_damage_pipeline.llm.qwen_reporter import DEFAULT_QWEN_MODEL_ID
from vehicle_damage_pipeline.service.display import (
    build_debug_prediction_json,
    build_detection_table,
    build_public_prediction_summary,
)
from vehicle_damage_pipeline.service.predictor import DamagePredictor, generate_assessment_report


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_COLAB_DRIVE_ROOT = Path("/content/drive/MyDrive/CarDD_YOLO11")
DEFAULT_COLAB_ADAPTER_DIR = DEFAULT_COLAB_DRIVE_ROOT / "llm_adapters" / REPORT_ADAPTER_NAME


def collect_example_images(sample_dir: str | Path | None, limit: int = 6) -> list[str]:
    if not sample_dir:
        return []
    root = Path(sample_dir)
    if not root.exists():
        return []
    files = [
        path
        for path in sorted(root.iterdir(), key=lambda item: item.name.lower())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    return [str(path) for path in files[:limit]]


def build_colab_launch_kwargs(server_port: int = 7860) -> dict[str, Any]:
    return {
        "share": True,
        "server_name": "0.0.0.0",
        "server_port": server_port,
        "prevent_thread_lock": False,
    }


def launch_colab_demo(
    *,
    weights: str | Path,
    output_dir: str | Path,
    sample_dir: str | Path | None = None,
    server_port: int = 7860,
    report_backend: str = "qwen",
    adapter_dir: str | Path | None = DEFAULT_COLAB_ADAPTER_DIR,
    model_id: str = DEFAULT_QWEN_MODEL_ID,
) -> None:
    try:
        import gradio as gr
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Install gradio before running the Colab public demo.") from exc

    predictor = DamagePredictor(weights=weights, output_dir=output_dir)
    examples = collect_example_images(sample_dir)

    def run(image_path: str):
        prediction = predictor.predict_file(Path(image_path))
        return (
            build_public_prediction_summary(prediction),
            build_detection_table(prediction),
            generate_assessment_report(
                prediction,
                backend=report_backend,
                adapter_dir=adapter_dir,
                model_id=model_id,
            ),
            build_debug_prediction_json(prediction),
        )

    with gr.Blocks(title="Vehicle Damage Assessment - Colab Public Demo") as demo:
        gr.Markdown(
            "# AI-Powered Vehicle Damage Assessment Pipeline\n"
            "Upload a vehicle image or choose one of the prepared samples. "
            "This Colab-specific demo uses `share=True` to create an external Gradio URL. "
            "Assessment reports use the Qwen LoRA adapter by default; add `--no-qwen` to use the template fallback."
        )
        image = gr.Image(type="filepath", label="Vehicle image")
        run_button = gr.Button("Run detection")
        output_summary = gr.JSON(label="Prediction summary")
        detection_table = gr.Dataframe(label="Detections", interactive=False)
        report = gr.Textbox(label="Assessment report", lines=8)
        with gr.Accordion("Debug prediction JSON", open=False):
            debug_json = gr.Code(label="Full prediction JSON", language="json")
        if examples:
            gr.Examples(examples=examples, inputs=image, label="Prepared upload samples")
        run_button.click(run, inputs=image, outputs=[output_summary, detection_table, report, debug_json])

    print("Launching Gradio with share=True. Use the public gradio.live URL shown below.")
    demo.launch(**build_colab_launch_kwargs(server_port=server_port))


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a Colab public Gradio demo.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-dir")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--report-backend", choices=["qwen", "template"], default="qwen")
    parser.add_argument("--no-qwen", action="store_true", help="Use template reports instead of the Qwen adapter.")
    parser.add_argument("--adapter-dir", default=str(DEFAULT_COLAB_ADAPTER_DIR))
    parser.add_argument("--model-id", default=DEFAULT_QWEN_MODEL_ID)
    args = parser.parse_args()
    launch_colab_demo(
        weights=args.weights,
        output_dir=args.output_dir,
        sample_dir=args.sample_dir,
        server_port=args.server_port,
        report_backend="template" if args.no_qwen else args.report_backend,
        adapter_dir=args.adapter_dir,
        model_id=args.model_id,
    )


if __name__ == "__main__":
    main()
