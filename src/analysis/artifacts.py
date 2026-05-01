from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.metrics import EvaluationResult


def save_top_tfidf_features(vectorizer, classifier, label_names: list[str], output_dir: Path, top_n: int = 25) -> Path:
    """Save the strongest positive TF-IDF coefficient features per class."""
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_names = np.asarray(vectorizer.get_feature_names_out())
    rows = []

    coefficients = classifier.coef_
    if coefficients.shape[0] == 1 and len(label_names) == 2:
        coefficients = np.vstack([-coefficients[0], coefficients[0]])

    for class_idx, label in enumerate(label_names):
        class_weights = coefficients[class_idx]
        top_indices = np.argsort(class_weights)[-top_n:][::-1]
        for rank, feature_idx in enumerate(top_indices, start=1):
            rows.append(
                {
                    "label": label,
                    "rank": rank,
                    "feature": feature_names[feature_idx],
                    "coefficient": float(class_weights[feature_idx]),
                }
            )

    path = output_dir / "top_tfidf_features.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def save_summary(results: list[EvaluationResult], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in results:
        rows.append(
            {
                "model": result.model_name,
                "evaluation_split": result.metrics.get("split", "test"),
                "accuracy": result.metrics["accuracy"],
                "macro_f1": result.metrics["macro_f1"],
                "weighted_f1": result.metrics["weighted_f1"],
                "num_test_examples": result.metrics["num_examples"],
                "output_dir": str(result.output_dir),
            }
        )

    summary = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    path = output_dir / "results_summary.csv"
    summary.to_csv(path, index=False)
    return path
