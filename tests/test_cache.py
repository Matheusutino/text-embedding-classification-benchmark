from pathlib import Path

import numpy as np

from embeddings_pipeline.cache import build_metadata, load_if_valid, save_cache


def test_dense_cache_hit_and_force_recompute_logic(tmp_path: Path) -> None:
    cache_dir = tmp_path / "embeddings" / "model" / "dataset"
    data = np.ones((2, 3), dtype=np.float32)
    metadata = build_metadata(
        dataset="sample.csv",
        representation="dense_repr",
        model="model/name",
        parameters={"normalize_embeddings": True},
        normalize_embeddings=True,
        device_used="cpu",
        generation_time_seconds=0.1,
        dtype="float32",
        shape=[2, 3],
        text_prefix="",
    )
    save_cache(cache_dir, data, metadata, sparse_format=False)

    expected = build_metadata(
        dataset="sample.csv",
        representation="dense_repr",
        model="model/name",
        parameters={"normalize_embeddings": True},
        normalize_embeddings=True,
        device_used="cpu",
        generation_time_seconds=0.0,
        dtype="float32",
        shape=[],
        text_prefix="",
    )
    cached = load_if_valid(cache_dir, expected, sparse_format=False)
    assert cached is not None
    assert cached.cache_hit is True
    assert cached.metadata["dataset"] == "sample.csv"

    changed = build_metadata(
        dataset="sample.csv",
        representation="dense_repr",
        model="model/name",
        parameters={"normalize_embeddings": False},
        normalize_embeddings=False,
        device_used="cpu",
        generation_time_seconds=0.0,
        dtype="float32",
        shape=[],
        text_prefix="",
    )
    assert load_if_valid(cache_dir, changed, sparse_format=False) is None


def test_metadata_contains_required_fields() -> None:
    metadata = build_metadata(
        dataset="sample.csv",
        representation="tfidf_unigram",
        model=None,
        parameters={"max_features": 1000},
        normalize_embeddings=False,
        device_used="cpu",
        generation_time_seconds=0.1,
        dtype="float32",
        shape=[2, 3],
        text_prefix="",
        fold=1,
        split="train",
    )
    required = {
        "dataset",
        "representation",
        "model",
        "parameters",
        "normalize_embeddings",
        "pipeline_version",
        "generated_at",
        "dtype",
        "shape",
        "device_used",
        "generation_time_seconds",
        "text_prefix",
        "fold",
        "split",
    }
    assert required.issubset(metadata.keys())
