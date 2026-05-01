from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from datasets import DatasetDict, load_dataset


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
    raw = load_dataset(dataset_name)
    original_splits = {split: len(raw[split]) for split in raw.keys()}
    raw = _build_train_val_test(raw, seed=seed, label_column=label_column, validation_fraction=validation_fraction)

    raw["train"] = _limit_split(raw["train"], max_train_examples, seed)
    raw["validation"] = _limit_split(raw["validation"], max_val_examples, seed)
    raw["test"] = _limit_split(raw["test"], max_test_examples, seed)

    label_names = _label_names(raw, label_column)
    metadata = {
        "dataset_name": dataset_name,
        "text_column": text_column,
        "label_column": label_column,
        "label_names": label_names,
        "split_protocol": "validation is a stratified holdout from the original train split; test is held out for final evaluation",
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


def _build_train_val_test(raw, seed: int, label_column: str, validation_fraction: float) -> DatasetDict:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if "train" in raw and "test" in raw:
        train_val = raw["train"].train_test_split(
            test_size=validation_fraction,
            seed=seed,
            stratify_by_column=label_column,
        )
        return DatasetDict(
            {
                "train": train_val["train"],
                "validation": train_val["test"],
                "test": raw["test"],
            }
        )
    if "train" not in raw:
        raise ValueError("Dataset must contain a train split or explicit train/validation/test splits.")

    train_test = raw["train"].train_test_split(test_size=0.2, seed=seed, stratify_by_column=label_column)
    train_val = train_test["train"].train_test_split(
        test_size=validation_fraction,
        seed=seed,
        stratify_by_column=label_column,
    )
    return DatasetDict(
        {
            "train": train_val["train"],
            "validation": train_val["test"],
            "test": train_test["test"],
        }
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
