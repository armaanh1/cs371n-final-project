from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


@dataclass(frozen=True)
class EvaluationResult:
    model_name: str
    metrics: dict[str, Any]
    output_dir: Path


def save_evaluation(
    model_name: str,
    texts: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    output_dir: Path,
    probabilities: np.ndarray | None = None,
    write_predictions: bool = True,
) -> EvaluationResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    metrics = {
        "model": model_name,
        "num_examples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "label_names": label_names,
    }
    per_class_report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(label_names))),
        target_names=label_names,
        zero_division=0,
        output_dict=True,
    )
    metrics["per_class"] = {
        label: {
            "precision": float(per_class_report[label]["precision"]),
            "recall": float(per_class_report[label]["recall"]),
            "f1": float(per_class_report[label]["f1-score"]),
            "support": int(per_class_report[label]["support"]),
        }
        for label in label_names
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(label_names))),
        target_names=label_names,
        zero_division=0,
    )
    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")

    pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=list(range(len(label_names)))),
        index=[f"gold_{label}" for label in label_names],
        columns=[f"pred_{label}" for label in label_names],
    ).to_csv(output_dir / "confusion_matrix.csv")

    predictions = _prediction_frame(texts, y_true, y_pred, label_names, probabilities)
    if write_predictions:
        predictions.to_csv(output_dir / "predictions.csv", index=False)
    save_errors(predictions, output_dir)

    return EvaluationResult(model_name=model_name, metrics=metrics, output_dir=output_dir)


def save_errors(predictions: pd.DataFrame, output_dir: Path, max_examples: int = 200) -> Path:
    errors = predictions[predictions["gold_id"] != predictions["pred_id"]].copy()
    if "confidence" in errors.columns:
        errors = errors.sort_values("confidence", ascending=False)
    errors.head(max_examples).to_csv(output_dir / "errors.csv", index=False)
    return output_dir / "errors.csv"


def _prediction_frame(
    texts: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    probabilities: np.ndarray | None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "text": texts,
            "gold_id": y_true,
            "gold_label": [label_names[idx] for idx in y_true],
            "pred_id": y_pred,
            "pred_label": [label_names[idx] for idx in y_pred],
        }
    )
    if probabilities is not None:
        frame["confidence"] = probabilities.max(axis=1)
        for idx, label in enumerate(label_names):
            frame[f"prob_{label}"] = probabilities[:, idx]
    return frame

