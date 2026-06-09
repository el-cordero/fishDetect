from fishdetect.models.yolo import train_yolo


def test_yolo_smoke_skips_remote_weight_without_download(synthetic_dataset, tmp_path):
    prepared = tmp_path / "prepared"
    (prepared / "yolo_det").mkdir(parents=True)
    (prepared / "yolo_det" / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\nnames:\n  0: fish\n", encoding="utf-8")
    config = {
        "dataset": {"prepared_root": str(prepared)},
        "training": {"skip_network_weight_downloads_in_smoke": True},
    }
    exp = {"name": "yolo8n_det", "family": "yolo", "model": "yolov8n.pt"}
    metrics = train_yolo(config, exp, tmp_path / "out", smoke_test=True, allow_downloads=False)
    assert metrics["status"] == "skipped"
    assert "remote downloads are disabled" in metrics["reason"]
