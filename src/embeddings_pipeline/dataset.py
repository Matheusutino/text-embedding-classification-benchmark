from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    path: Path
    texts: list[str]
    labels: np.ndarray


def load_dataset(path: str | Path) -> DatasetBundle:
    dataset_path = Path(path)
    rows: list[dict[str, str]] = []
    with dataset_path.open(encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required = {"text", "class"}
        missing = required - fieldnames
        if missing:
            raise ValueError(
                f"Dataset {dataset_path} is missing required columns: {sorted(missing)}"
            )
        for row in reader:
            rows.append(row)

    texts = [(row.get("text") or "").strip() for row in rows]
    labels = np.asarray([(row.get("class") or "").strip() for row in rows], dtype=object)
    if len(texts) == 0:
        raise ValueError(f"Dataset {dataset_path} is empty")
    return DatasetBundle(
        name=dataset_path.name,
        path=dataset_path,
        texts=texts,
        labels=labels,
    )
