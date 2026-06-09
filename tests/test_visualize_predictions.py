import subprocess
import sys
from pathlib import Path

from fishdetect.config import dump_yaml
from fishdetect.pipeline import prepare_dataset_pipeline
from fishdetect.utils.files import ensure_dir, write_text


def test_visual_qc_script_runs_on_tiny_dataset(synthetic_dataset, tmp_path):
    prepared = tmp_path / "prepared"
    config = {
        "dataset": {
            "root": str(synthetic_dataset),
            "prepared_root": str(prepared),
            "split_seed": 1,
            "train": 0.70,
            "val": 0.15,
            "test": 0.15,
            "stratify": False,
            "group_by": "sha256",
            "link_images": True,
        },
        "training": {},
        "experiments": [{"name": "toy", "family": "yolo", "model": "toy.pt"}],
    }
    prepare_dataset_pipeline(config)
    exp = ensure_dir(tmp_path / "outputs" / "experiments" / "toy")
    write_text(exp / "config_used.yaml", dump_yaml(config))
    script = Path(__file__).resolve().parents[1] / "scripts" / "visualize_predictions.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--experiment",
            str(exp),
            "--split",
            "test",
            "--n",
            "2",
            "--mode",
            "mixed",
            "--config",
            str(Path(__file__).resolve().parents[1] / "configs" / "experiments.yaml"),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (exp / "predictions" / "galleries" / "test_mixed" / "index.html").exists()
