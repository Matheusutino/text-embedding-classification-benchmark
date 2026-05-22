from __future__ import annotations

import csv
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import LinearSVC

from embeddings_pipeline import PIPELINE_VERSION
from embeddings_pipeline.config import (
    ALL_SPECS,
    LOGISTIC_C_VALUES,
    METRIC_FOR_SELECTION,
    OUTER_FOLDS,
    RepresentationSpec,
    VAL_SIZE,
)
from embeddings_pipeline.dataset import load_dataset
from embeddings_pipeline.representations import (
    FeatureSet,
    generate_dense_embeddings,
    generate_lexical_final_features,
    generate_lexical_features,
)
from embeddings_pipeline.utils import detect_device, set_global_seed, slugify, write_json


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path
    summary_path: Path
    folds_path: Path
    status: str


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


def fit_logistic_regression(X_train: Any, y_train: np.ndarray, c_value: float) -> LogisticRegression:
    classifier = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    )
    classifier.fit(X_train, y_train)
    return classifier


def fit_linear_svc(
    X_train: Any,
    y_train: np.ndarray,
    c_value: float,
    multi_class: str,
) -> LinearSVC:
    classifier = LinearSVC(
        C=c_value,
        class_weight="balanced",
        multi_class=multi_class,
        max_iter=5000,
        random_state=42,
    )
    classifier.fit(X_train, y_train)
    return classifier


def fit_classifier(
    X_train: Any,
    y_train: np.ndarray,
    c_value: float,
    classifier_name: str,
    linear_svc_multi_class: str,
) -> Any:
    if classifier_name == "logistic_regression":
        return fit_logistic_regression(X_train, y_train, c_value)
    if classifier_name == "linear_svc":
        return fit_linear_svc(X_train, y_train, c_value, linear_svc_multi_class)
    raise ValueError(f"Unsupported classifier: {classifier_name}")


def select_best_model(
    X_train: Any,
    y_train: np.ndarray,
    X_val: Any,
    y_val: np.ndarray,
    classifier_name: str,
    linear_svc_multi_class: str,
) -> tuple[float, dict[str, float], float]:
    best_c = LOGISTIC_C_VALUES[0]
    best_metrics: dict[str, float] | None = None
    best_score = float("-inf")

    for c_value in LOGISTIC_C_VALUES:
        clf = fit_classifier(
            X_train,
            y_train,
            c_value,
            classifier_name,
            linear_svc_multi_class,
        )
        val_pred = clf.predict(X_val)
        metrics = compute_metrics(y_val, val_pred)
        score = metrics[METRIC_FOR_SELECTION]
        if score > best_score:
            best_score = score
            best_c = c_value
            best_metrics = metrics

    if best_metrics is None:
        raise RuntimeError("Failed to select hyperparameters")
    return best_c, best_metrics, best_score


def prepare_features(
    *,
    spec: RepresentationSpec,
    dataset_name: str,
    output_dir: Path,
    texts: list[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    fold: int,
    force_recompute: bool,
) -> FeatureSet:
    if spec.family == "lexical":
        return generate_lexical_features(
            spec=spec,
            dataset_name=dataset_name,
            output_dir=output_dir,
            train_texts=[texts[i] for i in train_idx],
            val_texts=[texts[i] for i in val_idx],
            fold=fold,
            force_recompute=force_recompute,
        )

    embeddings, metadata = generate_dense_embeddings(
        spec=spec,
        dataset_name=dataset_name,
        output_dir=output_dir,
        texts=texts,
        force_recompute=force_recompute,
    )
    dense_metadata = dict(metadata)
    dense_metadata["representation_kind"] = "dense"
    return FeatureSet(
        train=embeddings[train_idx],
        val=embeddings[val_idx],
        test=embeddings[test_idx],
        metadata=dense_metadata,
    )


def prepare_final_features(
    *,
    spec: RepresentationSpec,
    dataset_name: str,
    output_dir: Path,
    texts: list[str],
    train_val_idx: np.ndarray,
    test_idx: np.ndarray,
    fold: int,
    force_recompute: bool,
) -> FeatureSet:
    if spec.family == "lexical":
        return generate_lexical_final_features(
            spec=spec,
            dataset_name=dataset_name,
            output_dir=output_dir,
            train_val_texts=[texts[i] for i in train_val_idx],
            test_texts=[texts[i] for i in test_idx],
            fold=fold,
            force_recompute=force_recompute,
        )

    embeddings, metadata = generate_dense_embeddings(
        spec=spec,
        dataset_name=dataset_name,
        output_dir=output_dir,
        texts=texts,
        force_recompute=force_recompute,
    )
    dense_metadata = dict(metadata)
    dense_metadata["representation_kind"] = "dense"
    return FeatureSet(
        train=embeddings[train_val_idx],
        val=None,
        test=embeddings[test_idx],
        metadata=dense_metadata,
    )


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def run_experiment(
    *,
    dataset_path: str | Path,
    representation_name: str,
    output_dir: str | Path,
    force_recompute: bool = False,
    classifier_name: str = "logistic_regression",
    linear_svc_multi_class: str = "ovr",
) -> RunArtifacts:
    set_global_seed()
    output_root = Path(output_dir)
    dataset = load_dataset(dataset_path)
    if representation_name not in ALL_SPECS:
        raise ValueError(f"Unknown representation: {representation_name}")
    spec = ALL_SPECS[representation_name]

    dataset_slug = slugify(dataset.name)
    run_dir = output_root / "runs" / representation_name / dataset_slug
    if classifier_name != "logistic_regression":
        run_dir = output_root / "runs" / classifier_name / representation_name / dataset_slug
    run_dir.mkdir(parents=True, exist_ok=True)
    folds_path = run_dir / "fold_metrics.csv"
    summary_path = run_dir / "summary.json"

    outer_cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=42)
    fold_rows: list[dict[str, Any]] = []
    status = "ok"

    for fold_idx, (outer_train_idx, test_idx) in enumerate(
        outer_cv.split(dataset.texts, dataset.labels),
        start=1,
    ):
        outer_train_labels = dataset.labels[outer_train_idx]
        inner_train_rel, val_rel = train_test_split(
            np.arange(len(outer_train_idx)),
            test_size=VAL_SIZE,
            random_state=42,
            stratify=outer_train_labels,
        )
        train_idx = outer_train_idx[inner_train_rel]
        val_idx = outer_train_idx[val_rel]

        try:
            features = prepare_features(
                spec=spec,
                dataset_name=dataset.name,
                output_dir=output_root,
                texts=dataset.texts,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                fold=fold_idx,
                force_recompute=force_recompute,
            )
        except Exception as exc:
            status = "skipped"
            write_json(
                summary_path,
                {
                    "status": "skipped",
                    "reason": str(exc),
                    "dataset": dataset.name,
                    "representation": representation_name,
                    "model": spec.model_name,
                    "classifier": classifier_name,
                    "class_weight": "balanced",
                    "linear_svc_multi_class": linear_svc_multi_class,
                    "pipeline_version": PIPELINE_VERSION,
                },
            )
            return RunArtifacts(
                run_dir=run_dir,
                summary_path=summary_path,
                folds_path=folds_path,
                status="skipped",
            )

        y_train = dataset.labels[train_idx]
        y_val = dataset.labels[val_idx]
        y_test = dataset.labels[test_idx]

        tuning_start = time.perf_counter()
        best_c, val_metrics, _ = select_best_model(
            features.train,
            y_train,
            features.val,
            y_val,
            classifier_name,
            linear_svc_multi_class,
        )
        tuning_time = time.perf_counter() - tuning_start

        retrain_start = time.perf_counter()
        train_val_idx = np.concatenate([train_idx, val_idx])
        y_train_val = dataset.labels[train_val_idx]
        final_features = prepare_final_features(
            spec=spec,
            dataset_name=dataset.name,
            output_dir=output_root,
            texts=dataset.texts,
            train_val_idx=train_val_idx,
            test_idx=test_idx,
            fold=fold_idx,
            force_recompute=force_recompute,
        )
        final_model = fit_classifier(
            final_features.train,
            y_train_val,
            best_c,
            classifier_name,
            linear_svc_multi_class,
        )
        retrain_time = time.perf_counter() - retrain_start

        eval_start = time.perf_counter()
        test_pred = final_model.predict(final_features.test)
        test_metrics = compute_metrics(y_test, test_pred)
        eval_time = time.perf_counter() - eval_start

        fold_rows.append(
            {
                "fold": fold_idx,
                "best_c": best_c,
                "val_f1_macro": val_metrics["f1_macro"],
                "test_f1_macro": test_metrics["f1_macro"],
                "test_accuracy": test_metrics["accuracy"],
                "test_precision_macro": test_metrics["precision_macro"],
                "test_recall_macro": test_metrics["recall_macro"],
                "feature_generation_time_seconds": features.metadata["generation_time_seconds"],
                "final_feature_generation_time_seconds": final_features.metadata[
                    "generation_time_seconds"
                ],
                "feature_device_used": features.metadata["device_used"],
                "feature_cache_hit": features.metadata.get("cache_hit", False),
                "final_feature_cache_hit": final_features.metadata.get("cache_hit", False),
                "tuning_time_seconds": tuning_time,
                "retrain_time_seconds": retrain_time,
                "evaluation_time_seconds": eval_time,
            }
        )

    with folds_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fold_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fold_rows)

    aggregated = {
        metric: _mean_std([row[metric] for row in fold_rows])
        for metric in [
            "test_f1_macro",
            "test_accuracy",
            "test_precision_macro",
            "test_recall_macro",
        ]
    }
    summary = {
        "status": status,
        "dataset": dataset.name,
        "representation": representation_name,
        "model": spec.model_name,
        "classifier": classifier_name,
        "class_weight": "balanced",
        "linear_svc_multi_class": linear_svc_multi_class,
        "device_detected": detect_device(),
        "pipeline_version": PIPELINE_VERSION,
        "outer_folds": OUTER_FOLDS,
        "val_size": VAL_SIZE,
        "metric_for_selection": METRIC_FOR_SELECTION,
        "logistic_c_values": LOGISTIC_C_VALUES,
        "metrics": aggregated,
        "folds": fold_rows,
    }
    write_json(summary_path, summary)
    return RunArtifacts(
        run_dir=run_dir,
        summary_path=summary_path,
        folds_path=folds_path,
        status=status,
    )
