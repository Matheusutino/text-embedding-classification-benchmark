from __future__ import annotations

import json
import os
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from embeddings_pipeline.cache import build_metadata, load_if_valid, save_cache
from embeddings_pipeline.config import RepresentationSpec
from embeddings_pipeline.utils import detect_device, slugify


@dataclass(frozen=True)
class FeatureSet:
    train: Any
    val: Any | None
    test: Any
    metadata: dict[str, Any]


def get_text_prefix(model_name: str | None) -> str:
    if model_name == "nomic-ai/nomic-embed-text-v1.5":
        return "classification: "
    if model_name == "intfloat/e5-large-v2":
        return "query: "
    return ""


def apply_text_prefix(texts: list[str], model_name: str | None) -> list[str]:
    prefix = get_text_prefix(model_name)
    if not prefix:
        return texts
    return [f"{prefix}{text}" for text in texts]


def create_vectorizer(spec: RepresentationSpec):
    kwargs = {
        "analyzer": spec.analyzer,
        "ngram_range": spec.ngram_range,
        "max_features": spec.max_features,
        "dtype": np.float32,
    }
    if spec.vectorizer_type == "count":
        return CountVectorizer(**kwargs)
    if spec.vectorizer_type == "tfidf":
        return TfidfVectorizer(**kwargs)
    raise ValueError(f"Unsupported vectorizer type: {spec.vectorizer_type}")


def lexical_cache_dir(
    output_dir: Path,
    spec: RepresentationSpec,
    dataset_name: str,
    fold: int,
    split: str,
) -> Path:
    return (
        output_dir
        / "features"
        / spec.slug
        / slugify(dataset_name)
        / f"fold_{fold}"
        / split
    )


def dense_cache_dir(output_dir: Path, spec: RepresentationSpec, dataset_name: str) -> Path:
    return output_dir / "features" / spec.slug / slugify(dataset_name) / "full_dataset"


def _lexical_expected_metadata(
    *,
    dataset_name: str,
    spec: RepresentationSpec,
    parameters: dict[str, Any],
    fold: int,
    split: str,
) -> dict[str, Any]:
    return build_metadata(
        dataset=dataset_name,
        representation=spec.name,
        model=None,
        parameters=parameters,
        normalize_embeddings=False,
        device_used="cpu",
        generation_time_seconds=0.0,
        dtype="float32",
        shape=[],
        text_prefix="",
        fold=fold,
        split=split,
    )


def _cache_lexical_matrix(
    *,
    cache_dir: Path,
    dataset_name: str,
    spec: RepresentationSpec,
    parameters: dict[str, Any],
    fit_texts: list[str],
    transform_texts: list[str],
    fold: int,
    split: str,
    fit: bool,
    force_recompute: bool,
) -> tuple[Any, dict[str, Any], bool]:
    expected_metadata = _lexical_expected_metadata(
        dataset_name=dataset_name,
        spec=spec,
        parameters=parameters,
        fold=fold,
        split=split,
    )
    if not force_recompute:
        cached = load_if_valid(cache_dir, expected_metadata, sparse_format=True)
        if cached is not None:
            return cached.data, cached.metadata, True

    vectorizer = create_vectorizer(spec)
    start = time.perf_counter()
    if fit:
        matrix = vectorizer.fit_transform(transform_texts).astype(np.float32)
    else:
        vectorizer.fit(fit_texts)
        matrix = vectorizer.transform(transform_texts).astype(np.float32)
    elapsed = time.perf_counter() - start
    metadata = build_metadata(
        dataset=dataset_name,
        representation=spec.name,
        model=None,
        parameters=parameters,
        normalize_embeddings=False,
        device_used="cpu",
        generation_time_seconds=elapsed,
        dtype="float32",
        shape=list(matrix.shape),
        text_prefix="",
        fold=fold,
        split=split,
    )
    save_cache(cache_dir, matrix, metadata, sparse_format=True)
    return matrix, metadata, False


def generate_lexical_features(
    *,
    spec: RepresentationSpec,
    dataset_name: str,
    output_dir: Path,
    train_texts: list[str],
    val_texts: list[str],
    fold: int,
    force_recompute: bool,
) -> FeatureSet:
    parameters = {
        "vectorizer_type": spec.vectorizer_type,
        "analyzer": spec.analyzer,
        "ngram_range": list(spec.ngram_range or []),
        "max_features": spec.max_features,
    }

    legacy_test_dir = lexical_cache_dir(output_dir, spec, dataset_name, fold, "test")
    if legacy_test_dir.exists():
        shutil.rmtree(legacy_test_dir)

    train_matrix, train_metadata, train_cache_hit = _cache_lexical_matrix(
        cache_dir=lexical_cache_dir(output_dir, spec, dataset_name, fold, "train"),
        dataset_name=dataset_name,
        spec=spec,
        parameters=parameters,
        fit_texts=train_texts,
        transform_texts=train_texts,
        fold=fold,
        split="train",
        fit=True,
        force_recompute=force_recompute,
    )
    val_matrix, _, _ = _cache_lexical_matrix(
        cache_dir=lexical_cache_dir(output_dir, spec, dataset_name, fold, "val"),
        dataset_name=dataset_name,
        spec=spec,
        parameters=parameters,
        fit_texts=train_texts,
        transform_texts=val_texts,
        fold=fold,
        split="val",
        fit=False,
        force_recompute=force_recompute,
    )
    train_metadata = dict(train_metadata)
    train_metadata["cache_hit"] = train_cache_hit
    train_metadata["representation_kind"] = "lexical"
    return FeatureSet(
        train=train_matrix,
        val=val_matrix,
        test=None,
        metadata=train_metadata,
    )


def generate_lexical_final_features(
    *,
    spec: RepresentationSpec,
    dataset_name: str,
    output_dir: Path,
    train_val_texts: list[str],
    test_texts: list[str],
    fold: int,
    force_recompute: bool,
) -> FeatureSet:
    parameters = {
        "vectorizer_type": spec.vectorizer_type,
        "analyzer": spec.analyzer,
        "ngram_range": list(spec.ngram_range or []),
        "max_features": spec.max_features,
    }
    train_val_matrix, train_val_metadata, train_val_cache_hit = _cache_lexical_matrix(
        cache_dir=lexical_cache_dir(output_dir, spec, dataset_name, fold, "train_val"),
        dataset_name=dataset_name,
        spec=spec,
        parameters=parameters,
        fit_texts=train_val_texts,
        transform_texts=train_val_texts,
        fold=fold,
        split="train_val",
        fit=True,
        force_recompute=force_recompute,
    )
    test_matrix, _, _ = _cache_lexical_matrix(
        cache_dir=lexical_cache_dir(output_dir, spec, dataset_name, fold, "test_final"),
        dataset_name=dataset_name,
        spec=spec,
        parameters=parameters,
        fit_texts=train_val_texts,
        transform_texts=test_texts,
        fold=fold,
        split="test_final",
        fit=False,
        force_recompute=force_recompute,
    )
    train_val_metadata = dict(train_val_metadata)
    train_val_metadata["cache_hit"] = train_val_cache_hit
    train_val_metadata["representation_kind"] = "lexical"
    return FeatureSet(
        train=train_val_matrix,
        val=None,
        test=test_matrix,
        metadata=train_val_metadata,
    )


def _load_sentence_transformer(model_name: str, device: str):
    from sentence_transformers import SentenceTransformer

    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token:
        return SentenceTransformer(model_name, device=device, token=token)
    return SentenceTransformer(model_name, device=device)


def _get_openrouter_api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    return api_key


def _chunk_texts(texts: list[str], chunk_size: int) -> list[list[str]]:
    return [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]


def _fetch_openrouter_embedding_batch(model_name: str, texts: list[str]) -> np.ndarray:
    api_key = _get_openrouter_api_key()
    payload = json.dumps({"model": model_name, "input": texts}).encode("utf-8")
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/embeddings",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"OpenRouter HTTP {exc.code} for model {model_name}: {details}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenRouter connection error for model {model_name}: {exc}") from exc

    data = body.get("data")
    if not data:
        raise RuntimeError(f"No embedding data received: {body}")

    embeddings = [item.get("embedding") for item in data]
    if any(embedding is None for embedding in embeddings):
        raise RuntimeError(f"Malformed embedding payload received: {body}")
    return np.asarray(embeddings, dtype=np.float32)


def _fetch_openrouter_embeddings(model_name: str, texts: list[str], batch_size: int = 100) -> np.ndarray:
    if not texts:
        return np.asarray([], dtype=np.float32)

    batches = _chunk_texts(texts, batch_size)
    outputs = [_fetch_openrouter_embedding_batch(model_name, batch) for batch in batches]
    return np.vstack(outputs).astype(np.float32)


def generate_dense_embeddings(
    *,
    spec: RepresentationSpec,
    dataset_name: str,
    output_dir: Path,
    texts: list[str],
    force_recompute: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    device = detect_device()
    cache_dir = dense_cache_dir(output_dir, spec, dataset_name)
    text_prefix = get_text_prefix(spec.model_name)
    parameters = {"normalize_embeddings": spec.normalize_embeddings}
    expected_metadata = build_metadata(
        dataset=dataset_name,
        representation=spec.name,
        model=spec.model_name,
        parameters=parameters,
        normalize_embeddings=spec.normalize_embeddings,
        device_used=device,
        generation_time_seconds=0.0,
        dtype="float32",
        shape=[],
        text_prefix=text_prefix,
    )

    if not force_recompute:
        cached = load_if_valid(cache_dir, expected_metadata, sparse_format=False)
        if cached is not None:
            return np.asarray(cached.data, dtype=np.float32), cached.metadata

    prefixed_texts = apply_text_prefix(texts, spec.model_name)
    start = time.perf_counter()
    if spec.family == "dense":
        model = _load_sentence_transformer(spec.model_name, device=device)
        embeddings = model.encode(
            prefixed_texts,
            normalize_embeddings=spec.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    elif spec.family == "remote":
        embeddings = _fetch_openrouter_embeddings(spec.model_name, prefixed_texts)
    else:
        raise ValueError(f"Unsupported dense family: {spec.family}")

    embeddings = np.asarray(embeddings, dtype=np.float32)
    elapsed = time.perf_counter() - start
    metadata = build_metadata(
        dataset=dataset_name,
        representation=spec.name,
        model=spec.model_name,
        parameters=parameters,
        normalize_embeddings=spec.normalize_embeddings,
        device_used=device,
        generation_time_seconds=elapsed,
        dtype="float32",
        shape=list(embeddings.shape),
        text_prefix=text_prefix,
    )
    result = save_cache(cache_dir, embeddings, metadata, sparse_format=False)
    return np.asarray(result.data, dtype=np.float32), metadata
