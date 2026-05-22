from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from embeddings_pipeline import PIPELINE_VERSION
from embeddings_pipeline.utils import read_json, write_json


@dataclass(frozen=True)
class CacheResult:
    data: Any
    metadata: dict[str, Any]
    cache_hit: bool
    data_path: Path
    metadata_path: Path


def build_metadata(
    *,
    dataset: str,
    representation: str,
    model: str | None,
    parameters: dict[str, Any],
    normalize_embeddings: bool,
    device_used: str,
    generation_time_seconds: float,
    dtype: str,
    shape: list[int],
    text_prefix: str,
    fold: int | None = None,
    split: str | None = None,
    pipeline_version: str = PIPELINE_VERSION,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "representation": representation,
        "model": model,
        "parameters": parameters,
        "normalize_embeddings": normalize_embeddings,
        "pipeline_version": pipeline_version,
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "dtype": dtype,
        "shape": shape,
        "device_used": device_used,
        "generation_time_seconds": generation_time_seconds,
        "text_prefix": text_prefix,
        "fold": fold,
        "split": split,
    }


def metadata_matches(current: dict[str, Any], existing: dict[str, Any]) -> bool:
    keys = [
        "dataset",
        "representation",
        "model",
        "parameters",
        "normalize_embeddings",
        "pipeline_version",
        "text_prefix",
        "fold",
        "split",
    ]
    return all(current.get(key) == existing.get(key) for key in keys)


def load_if_valid(
    cache_dir: Path,
    expected_metadata: dict[str, Any],
    *,
    sparse_format: bool,
) -> CacheResult | None:
    metadata_path = cache_dir / "metadata.json"
    data_path = cache_dir / ("features.npz" if sparse_format else "embeddings.npy")
    if not metadata_path.exists() or not data_path.exists():
        return None
    existing = read_json(metadata_path)
    if not metadata_matches(expected_metadata, existing):
        return None
    data = sparse.load_npz(data_path) if sparse_format else np.load(data_path)
    return CacheResult(
        data=data,
        metadata=existing,
        cache_hit=True,
        data_path=data_path,
        metadata_path=metadata_path,
    )


def save_cache(
    cache_dir: Path,
    data: Any,
    metadata: dict[str, Any],
    *,
    sparse_format: bool,
) -> CacheResult:
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / "metadata.json"
    data_path = cache_dir / ("features.npz" if sparse_format else "embeddings.npy")
    if sparse_format:
        sparse.save_npz(data_path, data)
    else:
        np.save(data_path, np.asarray(data, dtype=np.float32))
    write_json(metadata_path, metadata)
    return CacheResult(
        data=data,
        metadata=metadata,
        cache_hit=False,
        data_path=data_path,
        metadata_path=metadata_path,
    )
