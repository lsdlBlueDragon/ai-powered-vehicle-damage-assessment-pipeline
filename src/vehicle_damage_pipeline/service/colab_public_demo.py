from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from vehicle_damage_pipeline.llm.adapter import REPORT_ADAPTER_NAME
from vehicle_damage_pipeline.llm.qwen_reporter import DEFAULT_QWEN_MODEL_ID
from vehicle_damage_pipeline.service.display import (
    DETECTION_TABLE_HEADERS,
    build_debug_prediction_json,
    build_detection_table,
    build_public_prediction_summary,
)
from vehicle_damage_pipeline.service.predictor import DamagePredictor, generate_assessment_report


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CONTENT_TYPE_SUFFIXES = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
}
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


def build_demo_css() -> str:
    return """
.debug-json-panel textarea,
.debug-json-panel pre {
  max-height: 360px;
  overflow: auto;
}
@media print {
  .debug-json-panel {
    display: none !important;
  }
}
"""


def _suffix_from_url_or_content_type(image_url: str, content_type: str | None) -> str:
    suffix = Path(urlparse(image_url).path).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return suffix
    if content_type:
        media_type = content_type.split(";", 1)[0].strip().lower()
        if media_type in CONTENT_TYPE_SUFFIXES:
            return CONTENT_TYPE_SUFFIXES[media_type]
    return ".jpg"


def download_image_url(image_url: str, output_dir: str | Path, *, max_bytes: int = 15_000_000) -> str:
    parsed = urlparse(image_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Image URL must be an http(s) direct image URL.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    request = Request(
        image_url,
        headers={"User-Agent": "Mozilla/5.0 vehicle-damage-demo/1.0"},
    )
    with urlopen(request, timeout=30) as response:  # noqa: S310 - user-provided URL is a demo input.
        content_type = response.headers.get("Content-Type", "")
        if content_type and not content_type.lower().startswith("image/"):
            raise ValueError(f"URL did not return an image content type: {content_type}")
        payload = response.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError(f"Image URL response is larger than {max_bytes} bytes.")

    suffix = _suffix_from_url_or_content_type(image_url, content_type)
    digest = hashlib.sha256(image_url.encode("utf-8")).hexdigest()[:12]
    target = output_dir / f"url_image_{digest}{suffix}"
    target.write_bytes(payload)
    return str(target)


def resolve_image_input(
    *,
    uploaded_image_path: str | Path | None,
    image_url: str | None,
    url_download_dir: str | Path,
    downloader=download_image_url,
) -> str:
    normalized_url = (image_url or "").strip()
    if normalized_url:
        return downloader(normalized_url, Path(url_download_dir))
    if uploaded_image_path:
        return str(uploaded_image_path)
    raise ValueError("Upload an image or provide an image URL.")


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
    url_download_dir = Path(output_dir) / "url_inputs"

    def run(image_path: str | None, image_url: str):
        resolved_image = resolve_image_input(
            uploaded_image_path=image_path,
            image_url=image_url,
            url_download_dir=url_download_dir,
        )
        prediction = predictor.predict_file(Path(resolved_image))
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

    with gr.Blocks(title="Vehicle Damage Assessment - Colab Public Demo", css=build_demo_css()) as demo:
        gr.Markdown(
            "# AI-Powered Vehicle Damage Assessment Pipeline\n"
            "Upload a vehicle image, paste a direct image URL, or choose one of the prepared samples. "
            "This Colab-specific demo uses `share=True` to create an external Gradio URL. "
            "Assessment reports use the Qwen LoRA adapter by default; add `--no-qwen` to use the template fallback."
        )
        image = gr.Image(type="filepath", label="Vehicle image")
        image_url = gr.Textbox(label="Direct image URL", placeholder="https://example.com/damaged-car.jpg")
        run_button = gr.Button("Run detection")
        output_summary = gr.JSON(label="Prediction summary")
        detection_table = gr.Dataframe(
            headers=DETECTION_TABLE_HEADERS,
            label="Detections",
            interactive=False,
        )
        report = gr.Textbox(label="Assessment report", lines=8)
        with gr.Accordion("Debug prediction JSON", open=False, elem_classes=["debug-json-panel"]):
            debug_json = gr.Code(label="Full prediction JSON", language="json")
        if examples:
            gr.Examples(examples=examples, inputs=image, label="Prepared upload samples")
        run_button.click(run, inputs=[image, image_url], outputs=[output_summary, detection_table, report, debug_json])

    print("Launching Gradio with share=True. Use the public gradio.live URL shown below.", flush=True)
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
