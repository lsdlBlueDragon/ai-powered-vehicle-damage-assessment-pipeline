import shutil
from pathlib import Path

from vehicle_damage_pipeline.service.colab_public_demo import (
    build_colab_launch_kwargs,
    collect_example_images,
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
