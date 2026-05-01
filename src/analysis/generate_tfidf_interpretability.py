#!/usr/bin/env python3
"""Generate coefficient-inspection artifacts for the TF-IDF logistic regression baseline."""

from __future__ import annotations

import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import joblib
import numpy as np
import pandas as pd

from analysis.interpretability_utils import ANALYSIS_DIR, OUTPUTS_DIR, ensure_analysis_dir, load_bundle_from_run_config


def main() -> None:
    ensure_analysis_dir()
    bundle = load_bundle_from_run_config()
    pipeline = joblib.load(OUTPUTS_DIR / "tfidf_logreg" / "model.joblib")
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]

    feature_names = np.asarray(vectorizer.get_feature_names_out())
    coefficients = classifier.coef_
    if coefficients.shape[0] == 1 and len(bundle.label_names) == 2:
        coefficients = np.vstack([-coefficients[0], coefficients[0]])

    rows = []
    for class_idx, label in enumerate(bundle.label_names):
        weights = coefficients[class_idx]
        top_positive = np.argsort(weights)[-20:][::-1]
        top_negative = np.argsort(weights)[:20]

        for rank, feature_idx in enumerate(top_positive, start=1):
            rows.append(
                {
                    "label": label,
                    "direction": "positive",
                    "rank": rank,
                    "feature": feature_names[feature_idx],
                    "coefficient": float(weights[feature_idx]),
                }
            )
        for rank, feature_idx in enumerate(top_negative, start=1):
            rows.append(
                {
                    "label": label,
                    "direction": "negative",
                    "rank": rank,
                    "feature": feature_names[feature_idx],
                    "coefficient": float(weights[feature_idx]),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(ANALYSIS_DIR / "tfidf_positive_negative_features.csv", index=False)

    latex_rows = []
    for label in bundle.label_names:
        subset = df[df["label"] == label]
        positive = ", ".join(subset[subset["direction"] == "positive"].head(5)["feature"].tolist())
        negative = ", ".join(subset[subset["direction"] == "negative"].head(5)["feature"].tolist())
        latex_rows.append({"Label": label, "Top positive": positive, "Top negative": negative})
    pd.DataFrame(latex_rows).to_latex(
        ANALYSIS_DIR / "tfidf_positive_negative_features.tex",
        index=False,
        caption="Representative positive and negative TF-IDF coefficient features per class.",
        label="tab:tfidf-positive-negative",
    )

    print(f"Wrote TF-IDF interpretability artifacts to {ANALYSIS_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
