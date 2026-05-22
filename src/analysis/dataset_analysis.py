from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate dataset analysis figures."
    )
    parser.add_argument(
        "--runs-dir",
        default="artifacts/runs",
        help="Directory containing experiment run outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/analysis",
        help="Directory where analysis figures will be saved.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional dataset name filter, e.g. CSTR.csv or sms_spam.csv.",
    )
    return parser


def read_run_summaries(runs_dir: Path) -> list[dict]:
    summaries = []
    for summary_path in sorted(runs_dir.glob("*/*/summary.json")):
        summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
    return summaries


def collect_f1_by_representation(
    summaries: list[dict],
    dataset_filter: str | None,
) -> tuple[list[str], list[list[float]]]:
    grouped: dict[str, list[float]] = {}
    for summary in summaries:
        if summary.get("status") != "ok":
            continue
        dataset_name = summary.get("dataset")
        if dataset_filter is not None and dataset_name != dataset_filter:
            continue
        representation = summary.get("representation")
        folds = summary.get("folds", [])
        scores = [fold["test_f1_macro"] for fold in folds if "test_f1_macro" in fold]
        if not scores:
            continue
        grouped.setdefault(representation, []).extend(scores)

    labels = sorted(grouped.keys())
    values = [grouped[label] for label in labels]
    return labels, values


def plot_f1_violin(
    f1_by_representation: list[list[float]],
    labels: list[str],
    output_path: Path,
    title_suffix: str,
) -> None:
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.9), 6))
    violin = ax.violinplot(f1_by_representation, showmeans=True, showmedians=True)
    for body in violin["bodies"]:
        body.set_alpha(0.6)

    ax.set_title(f"F1-macro Distribution by Representation{title_suffix}")
    ax.set_ylabel("Test F1-macro")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def print_summary(labels: list[str], f1_by_representation: list[list[float]]) -> None:
    print("F1-macro summary by representation:")
    for label, scores in zip(labels, f1_by_representation):
        print(
            f"{label}: n={len(scores)} "
            f"mean={statistics.mean(scores):.4f} "
            f"median={statistics.median(scores):.4f}"
        )


def main() -> None:
    args = build_parser().parse_args()
    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir)

    summaries = read_run_summaries(runs_dir)
    if not summaries:
        raise FileNotFoundError(f"No summary.json files found in {runs_dir}")

    labels, f1_by_representation = collect_f1_by_representation(summaries, args.dataset)
    if not labels:
        dataset_msg = f" for dataset {args.dataset}" if args.dataset else ""
        raise ValueError(f"No successful runs with F1 data found{dataset_msg}")

    dataset_suffix = f"_{args.dataset.replace('.csv', '').lower()}" if args.dataset else "_all_datasets"
    title_suffix = f" ({args.dataset})" if args.dataset else " (all datasets)"
    output_path = output_dir / f"f1_violin_by_representation{dataset_suffix}.pdf"
    plot_f1_violin(f1_by_representation, labels, output_path, title_suffix)
    print_summary(labels, f1_by_representation)
    print(f"saved={output_path}")
