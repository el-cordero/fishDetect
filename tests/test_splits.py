from fishdetect.dataset import build_annotation_table
from fishdetect.splits import assert_no_leakage, make_kfold_manifests, make_split_manifest


def test_split_reproducible_and_no_hash_leakage(synthetic_dataset):
    annotations, images, _ = build_annotation_table(synthetic_dataset)
    split_a = make_split_manifest(images, annotations, seed=42, stratify=False)
    split_b = make_split_manifest(images, annotations, seed=42, stratify=False)
    assert split_a == split_b
    assert_no_leakage(split_a)


def test_kfold_manifests(synthetic_dataset):
    annotations, images, _ = build_annotation_table(synthetic_dataset)
    folds = make_kfold_manifests(images, annotations, folds=2, seed=42)
    assert len(folds) == 2
    for fold in folds:
        assert_no_leakage(fold)
