from __future__ import annotations

import argparse
import json
from pathlib import Path

from vehicle_damage_pipeline.service.predictor import DamagePredictor, generate_assessment_report


def launch(weights: str, output_dir: str = "runs/gradio_predict") -> None:
    try:
        import gradio as gr
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Install gradio to run the visual demo.") from exc

    predictor = DamagePredictor(weights=weights, output_dir=output_dir)

    def run(image_path: str):
        prediction = predictor.predict_file(Path(image_path))
        return json.dumps(prediction, indent=2, ensure_ascii=False), generate_assessment_report(prediction)

    with gr.Blocks(title="Vehicle Damage Assessment") as demo:
        gr.Markdown("# AI-Powered Vehicle Damage Assessment Pipeline")
        image = gr.Image(type="filepath", label="Vehicle image")
        run_button = gr.Button("Run detection")
        output_json = gr.Code(label="Prediction JSON", language="json")
        report = gr.Textbox(label="Assessment report", lines=8)
        run_button.click(run, inputs=image, outputs=[output_json, report])
    demo.launch()


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Gradio vehicle damage demo.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", default="runs/gradio_predict")
    args = parser.parse_args()
    launch(args.weights, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
