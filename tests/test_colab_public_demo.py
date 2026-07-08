import shutil
from pathlib import Path

from vehicle_damage_pipeline.service.colab_public_demo import (
    build_demo_css,
    build_colab_launch_kwargs,
    collect_example_images,
    resolve_image_input,
)


def test_colab_launch_defaults_create_public_share_link():
    kwargs = build_colab_launch_kwargs(server_port=7860)

    assert kwargs["share"] is True
    assert kwargs["server_name"] == "0.0.0.0"
    assert kwargs["server_port"] == 7860
    assert kwargs["prevent_thread_lock"] is False


def test_collect_example_images_finds_supported_files_in_stable_order():
    tmp_path = Path(".test_tmp") / "colab_examples"
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True)
    (tmp_path / "b.txt").write_text("skip", encoding="utf-8")
    (tmp_path / "c.png").write_bytes(b"png")
    (tmp_path / "a.jpg").write_bytes(b"jpg")

    examples = collect_example_images(tmp_path, limit=2)

    assert examples == [str(tmp_path / "a.jpg"), str(tmp_path / "c.png")]
    shutil.rmtree(tmp_path)


def test_resolve_image_input_uses_upload_when_url_is_empty():
    image = resolve_image_input(
        uploaded_image_path="local.jpg",
        image_url=" ",
        url_download_dir=Path(".test_tmp") / "url_inputs",
    )

    assert image == "local.jpg"


def test_resolve_image_input_prefers_url_and_uses_downloader():
    calls = []

    def fake_downloader(url: str, output_dir: Path) -> str:
        calls.append((url, output_dir))
        return str(output_dir / "downloaded.jpg")

    image = resolve_image_input(
        uploaded_image_path="local.jpg",
        image_url="https://example.test/damage.jpg",
        url_download_dir=Path(".test_tmp") / "url_inputs",
        downloader=fake_downloader,
    )

    assert image.endswith("downloaded.jpg")
    assert calls == [("https://example.test/damage.jpg", Path(".test_tmp") / "url_inputs")]


def test_resolve_image_input_requires_upload_or_url():
    try:
        resolve_image_input(
            uploaded_image_path=None,
            image_url="",
            url_download_dir=Path(".test_tmp") / "url_inputs",
        )
    except ValueError as exc:
        assert "Upload an image or provide an image URL" in str(exc)
    else:
        raise AssertionError("Expected missing input to raise ValueError.")


def test_demo_css_hides_debug_json_when_printing():
    css = build_demo_css()

    assert ".debug-json-panel" in css
    assert "@media print" in css
    assert "display: none" in css
