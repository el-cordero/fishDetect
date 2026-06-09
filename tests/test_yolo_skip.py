import pytest

from fishdetect.models.yolo import _resolve_batch, train_yolo


def test_resolve_batch_auto_smoke():
    assert _resolve_batch("auto", smoke_test=True) == 2


def test_resolve_batch_auto_full_training():
    assert _resolve_batch("auto", smoke_test=False) == 4


def test_resolve_batch_numeric_strings():
    assert _resolve_batch("8") == 8
    assert _resolve_batch("0.5") == 0.5


def test_resolve_batch_numeric_value():
    assert _resolve_batch(2) == 2


def test_resolve_batch_invalid_string():
    with pytest.raises(ValueError, match="Invalid batch value"):
        _resolve_batch("abc")


def test_yolo_smoke_skips_remote_weight_without_download(synthetic_dataset, tmp_path):
    prepared = tmp_path / "prepared"
    (prepared / "yolo_det").mkdir(parents=True)
    (prepared / "yolo_det" / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\nnames:\n  0: fish\n", encoding="utf-8")
    config = {
        "dataset": {"prepared_root": str(prepared)},
        "training": {"skip_network_weight_downloads_in_smoke": True},
    }
    exp = {"name": "missing_weight_det", "family": "yolo", "model": "definitely_missing_fishdetect_test_model.pt"}
    metrics = train_yolo(config, exp, tmp_path / "out", smoke_test=True, allow_downloads=False)
    assert metrics["status"] == "skipped"
    assert "remote downloads are disabled" in metrics["reason"]
