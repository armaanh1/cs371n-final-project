from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from datasets import DatasetDict, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HF_CACHE_DIR = PROJECT_ROOT / ".hf_cache"


@dataclass(frozen=True)
class DatasetBundle:
    train_texts: list[str]
    train_labels: np.ndarray
    val_texts: list[str]
    val_labels: np.ndarray
    test_texts: list[str]
    test_labels: np.ndarray
    label_names: list[str]
    metadata: dict[str, Any]


def load_text_classification_dataset(
    dataset_name: str,
    text_column: str,
    label_column: str,
    seed: int,
    max_train_examples: int = 0,
    max_val_examples: int = 0,
    max_test_examples: int = 0,
    validation_fraction: float = 0.1,
) -> DatasetBundle:
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_dataset(dataset_name, cache_dir=str(HF_CACHE_DIR))
    original_splits = {split: len(raw[split]) for split in raw.keys()}
    raw = DatasetDict(
        {
            "train": raw["train"],
            "validation": raw["validation"],
            "test": raw["test"],
        }
    )

    raw["train"] = _limit_split(raw["train"], max_train_examples, seed)
    raw["validation"] = _limit_split(raw["validation"], max_val_examples, seed)
    raw["test"] = _limit_split(raw["test"], max_test_examples, seed)

    label_names = _label_names(raw, label_column)
    metadata = {
        "dataset_name": dataset_name,
        "text_column": text_column,
        "label_column": label_column,
        "label_names": label_names,
        "validation_fraction": validation_fraction,
        "original_splits": original_splits,
        "splits": {split: len(raw[split]) for split in ["train", "validation", "test"]},
    }

    return DatasetBundle(
        train_texts=_as_texts(raw["train"][text_column]),
        train_labels=np.asarray(raw["train"][label_column], dtype=np.int64),
        val_texts=_as_texts(raw["validation"][text_column]),
        val_labels=np.asarray(raw["validation"][label_column], dtype=np.int64),
        test_texts=_as_texts(raw["test"][text_column]),
        test_labels=np.asarray(raw["test"][label_column], dtype=np.int64),
        label_names=label_names,
        metadata=metadata,
    )
def _limit_split(split, max_examples: int, seed: int):
    if max_examples <= 0 or len(split) <= max_examples:
        return split
    return split.shuffle(seed=seed).select(range(max_examples))


def _label_names(raw, label_column: str) -> list[str]:
    feature = raw["train"].features[label_column]
    names = getattr(feature, "names", None)
    if names:
        return list(names)

    labels = set()
    for split in ["train", "validation", "test"]:
        labels.update(raw[split][label_column])
    return [str(label) for label in sorted(labels)]


def _as_texts(values) -> list[str]:
    return [str(value) for value in values]
