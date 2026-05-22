from pathlib import Path

import pytest

from embeddings_pipeline.dataset import load_dataset


def test_load_dataset_requires_expected_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("body,label\nhello,spam\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(path)
