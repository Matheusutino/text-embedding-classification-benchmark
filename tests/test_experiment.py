import csv
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, train_test_split

from embeddings_pipeline.experiment import run_experiment


def test_stratified_splits_do_not_overlap() -> None:
    labels = ["a"] * 20 + ["b"] * 20
    texts = [f"text {i}" for i in range(len(labels))]
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for outer_train_idx, test_idx in splitter.split(texts, labels):
        inner_train_rel, val_rel = train_test_split(
            list(range(len(outer_train_idx))),
            test_size=0.2,
            random_state=42,
            stratify=[labels[i] for i in outer_train_idx],
        )
        train_idx = set(outer_train_idx[inner_train_rel])
        val_idx = set(outer_train_idx[val_rel])
        test_idx = set(test_idx)
        assert train_idx.isdisjoint(val_idx)
        assert train_idx.isdisjoint(test_idx)
        assert val_idx.isdisjoint(test_idx)


def test_functional_run_with_lexical_representation(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    rows = []
    for i in range(15):
        rows.append({"text": f"ham message {i}", "class": "ham"})
    for i in range(15):
        rows.append({"text": f"spam offer {i}", "class": "spam"})
    with dataset_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "class"])
        writer.writeheader()
        writer.writerows(rows)

    result = run_experiment(
        dataset_path=dataset_path,
        representation_name="tfidf_unigram",
        output_dir=tmp_path / "artifacts",
    )
    assert result.status == "ok"
    assert result.summary_path.exists()
    assert result.folds_path.exists()
    assert (
        tmp_path
        / "artifacts"
        / "features"
        / "tfidf_unigram"
        / "dataset"
        / "fold_1"
        / "train"
        / "metadata.json"
    ).exists()
    assert (
        tmp_path
        / "artifacts"
        / "features"
        / "tfidf_unigram"
        / "dataset"
        / "fold_1"
        / "val"
        / "metadata.json"
    ).exists()
    assert (
        tmp_path
        / "artifacts"
        / "features"
        / "tfidf_unigram"
        / "dataset"
        / "fold_1"
        / "train_val"
        / "metadata.json"
    ).exists()
    assert (
        tmp_path
        / "artifacts"
        / "features"
        / "tfidf_unigram"
        / "dataset"
        / "fold_1"
        / "test_final"
        / "metadata.json"
    ).exists()
