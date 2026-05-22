from __future__ import annotations

import argparse

from dotenv import load_dotenv

from embeddings_pipeline.experiment import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run text classification experiments with cached representations."
    )
    parser.add_argument("--dataset", required=True, help="Path to the dataset CSV file.")
    parser.add_argument(
        "--representation",
        required=True,
        help="Representation key to use, e.g. tfidf_unigram or all_minilm_l6_v2.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where runs and caches will be stored.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore cache and recompute artifacts.",
    )
    parser.add_argument(
        "--classifier",
        choices=["logistic_regression", "linear_svc"],
        default="logistic_regression",
        help="Classifier to train on top of the selected representation.",
    )
    parser.add_argument(
        "--linear-svc-multi-class",
        choices=["ovr", "crammer_singer"],
        default="ovr",
        help="LinearSVC multiclass strategy. Only used when --classifier linear_svc.",
    )
    return parser


def main() -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    result = run_experiment(
        dataset_path=args.dataset,
        representation_name=args.representation,
        output_dir=args.output_dir,
        force_recompute=args.force_recompute,
        classifier_name=args.classifier,
        linear_svc_multi_class=args.linear_svc_multi_class,
    )
    print(f"status={result.status}")
    print(f"run_dir={result.run_dir}")
    print(f"summary={result.summary_path}")
