from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vehicle_damage_pipeline.service.predictor import DamagePredictor, generate_assessment_report


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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
            json.dumps(prediction, indent=2, ensure_ascii=False),
            generate_assessment_report(prediction),
        )

    with gr.Blocks(title="Vehicle Damage Assessment - Colab Public Demo") as demo:
        gr.Markdown(
            "# AI-Powered Vehicle Damage Assessment Pipeline\n"
            "Upload a vehicle image or choose one of the prepared samples. "
            "This Colab-specific demo uses `share=True` to create an external Gradio URL."
        )
        image = gr.Image(type="filepath", label="Vehicle image")
        run_button = gr.Button("Run detection")
        output_json = gr.Code(label="Prediction JSON", language="json")
        report = gr.Textbox(label="Assessment report", lines=8)
        if examples:
            gr.Examples(examples=examples, inputs=image, label="Prepared upload samples")
        run_button.click(run, inputs=image, outputs=[output_json, report])

    print("Launching Gradio with share=True. Use the public gradio.live URL shown below.")
    demo.launch(**build_colab_launch_kwargs(server_port=server_port))


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a Colab public Gradio demo.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-dir")
    parser.add_argument("--server-port", type=int, default=7860)
    args = parser.parse_args()
    launch_colab_demo(
        weights=args.weights,
        output_dir=args.output_dir,
        sample_dir=args.sample_dir,
        server_port=args.server_port,
    )


if __name__ == "__main__":
    main()
