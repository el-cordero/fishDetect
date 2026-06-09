from fishdetect.audit import audit_dataset


def test_dataset_audit_outputs_summary(synthetic_dataset, tmp_path):
    summary = audit_dataset(synthetic_dataset, tmp_path / "audit", rare_threshold=2)
    assert summary["box_only"] is True
    assert summary["image_count"] == 4
    assert summary["annotation_count"] == 3
    assert (tmp_path / "audit" / "dataset_audit_summary.json").exists()
    assert (tmp_path / "audit" / "class_frequency.csv").exists()
