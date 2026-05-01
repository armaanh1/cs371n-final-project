from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from analysis.artifacts import save_top_tfidf_features
from data import DatasetBundle
from analysis.metrics import EvaluationResult, save_evaluation


def run_tfidf(
    bundle: DatasetBundle,
    output_root: Path,
    seed: int,
    max_features: int,
    ngram_max: int,
    min_df: int,
    c: float,
) -> EvaluationResult:
    output_dir = output_root / "tfidf_logreg"
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, ngram_max),
                    min_df=min_df,
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=c,
                    max_iter=5_000,
                    solver="saga",
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )

    pipeline.fit(bundle.train_texts, bundle.train_labels)
    val_predictions = pipeline.predict(bundle.val_texts)
    val_probabilities = pipeline.predict_proba(bundle.val_texts)
    predictions = pipeline.predict(bundle.test_texts)
    probabilities = pipeline.predict_proba(bundle.test_texts)

    joblib.dump(pipeline, output_dir / "model.joblib")
    save_top_tfidf_features(
        vectorizer=pipeline.named_steps["tfidf"],
        classifier=pipeline.named_steps["classifier"],
        label_names=bundle.label_names,
        output_dir=output_dir,
    )
    save_evaluation(
        model_name="tfidf_logreg_validation",
        texts=bundle.val_texts,
        y_true=bundle.val_labels,
        y_pred=val_predictions,
        label_names=bundle.label_names,
        output_dir=output_dir / "validation",
        probabilities=val_probabilities,
        split="validation",
    )

    return save_evaluation(
        model_name="tfidf_logreg",
        texts=bundle.test_texts,
        y_true=bundle.test_labels,
        y_pred=predictions,
        label_names=bundle.label_names,
        output_dir=output_dir,
        probabilities=probabilities,
    )
