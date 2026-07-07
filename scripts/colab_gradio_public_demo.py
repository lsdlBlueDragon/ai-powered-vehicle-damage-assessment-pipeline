"""Colab-only Gradio launcher that always creates an external share link."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vehicle_damage_pipeline.service.colab_public_demo import main


if __name__ == "__main__":
    main()
