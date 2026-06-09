from fishdetect.config import (
    experiment_names_for_group,
    experiments_for_group,
    get_experiment,
    get_experiment_group,
    load_config,
)


def test_config_loads_env_paths(monkeypatch):
    monkeypatch.setenv("FISHDETECT_DATASET_ROOT", "/tmp/input")
    monkeypatch.setenv("FISHDETECT_PREPARED_ROOT", "/tmp/prepared")
    config = load_config("configs/experiments.yaml")
    assert config["dataset"]["root"] == "/tmp/input"
    assert config["dataset"]["prepared_root"] == "/tmp/prepared"


def test_experiment_group_selection(monkeypatch):
    monkeypatch.setenv("FISHDETECT_DATASET_ROOT", "/tmp/input")
    monkeypatch.setenv("FISHDETECT_PREPARED_ROOT", "/tmp/prepared")
    config = load_config("configs/experiments.yaml")
    assert experiment_names_for_group(config, "smoke") == ["yolo8n_det"]
    names = [exp["name"] for exp in experiments_for_group(config, "main_comparison")]
    assert "yolo8s_det" in names
    assert "yolo12s_det" in names


def test_notebook_experiment_helpers(monkeypatch):
    monkeypatch.setenv("FISHDETECT_DATASET_ROOT", "/tmp/input")
    monkeypatch.setenv("FISHDETECT_PREPARED_ROOT", "/tmp/prepared")
    config = load_config("configs/experiments.yaml")
    assert get_experiment_group(config, "smoke") == ["yolo8n_det"]
    assert get_experiment(config, "yolo8n_det")["model"] == "yolov8n.pt"
