from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from data import DatasetBundle, load_text_classification_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ANALYSIS_DIR = OUTPUTS_DIR / "analysis"
RUN_CONFIG_PATH = OUTPUTS_DIR / "run_config.json"

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "do", "did", "for",
    "from", "had", "has", "have", "he", "her", "his", "i", "if", "im", "in", "is", "it",
    "its", "ive", "me", "my", "of", "on", "or", "our", "she", "so", "that", "the", "their",
    "them", "they", "this", "to", "too", "was", "we", "were", "with", "you", "your",
}


def ensure_analysis_dir() -> Path:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    return ANALYSIS_DIR


def load_run_config() -> dict:
    with RUN_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_bundle_from_run_config() -> DatasetBundle:
    config = load_run_config()
    return load_text_classification_dataset(
        dataset_name=config["dataset_name"],
        text_column=config["text_column"],
        label_column=config["label_column"],
        seed=int(config["seed"]),
        max_train_examples=int(config.get("max_train_examples", 0)),
        max_val_examples=int(config.get("max_val_examples", 0)),
        max_test_examples=int(config.get("max_test_examples", 0)),
        validation_fraction=float(config.get("validation_fraction", 0.1)),
    )


def select_confident_correct_examples(
    predictions_path: Path,
    label_names: list[str],
    per_label: int,
) -> pd.DataFrame:
    predictions = pd.read_csv(predictions_path)
    if "pred_label_id" not in predictions.columns and "pred_id" in predictions.columns:
        predictions["pred_label_id"] = predictions["pred_id"]
    if "gold_label_id" not in predictions.columns and "gold_id" in predictions.columns:
        predictions["gold_label_id"] = predictions["gold_id"]
    correct = predictions[predictions["gold_label"] == predictions["pred_label"]].copy()
    correct = correct.sort_values(["gold_label", "confidence"], ascending=[True, False])
    groups = []
    for label in label_names:
        groups.append(correct[correct["gold_label"] == label].head(per_label))
    return pd.concat(groups, ignore_index=True) if groups else correct.head(0)


def normalize_word(word: str) -> str | None:
    word = word.lower()
    word = re.sub(r"[^a-z]+", "", word)
    if len(word) < 3:
        return None
    if word in STOPWORDS:
        return None
    return word


def summarize_token_scores(
    rows: list[dict[str, object]],
    tokens_key: str,
    scores_key: str,
    label_names: list[str],
    top_k: int = 15,
) -> pd.DataFrame:
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        label = str(row["label"])
        tokens = row[tokens_key]
        scores = row[scores_key]
        for token, score in zip(tokens, scores):
            normalized = normalize_word(str(token))
            if not normalized:
                continue
            totals[label][normalized] += float(score)
            counts[label][normalized] += 1

    summary_rows: list[dict[str, object]] = []
    for label in label_names:
        ranked = sorted(totals[label].items(), key=lambda item: item[1], reverse=True)
        for rank, (token, total_score) in enumerate(ranked[:top_k], start=1):
            summary_rows.append(
                {
                    "label": label,
                    "rank": rank,
                    "token": token,
                    "total_score": total_score,
                    "count": counts[label][token],
                    "avg_score": total_score / counts[label][token],
                }
            )
    return pd.DataFrame(summary_rows)


def write_label_token_latex(
    summary_df: pd.DataFrame,
    output_path: Path,
    caption: str,
    label: str,
    top_k: int = 5,
) -> None:
    rows = []
    for class_name, group in summary_df.groupby("label"):
        top_tokens = ", ".join(group.sort_values("rank").head(top_k)["token"].tolist())
        rows.append({"Label": class_name, "Top tokens": top_tokens})
    pd.DataFrame(rows).to_latex(output_path, index=False, caption=caption, label=label)
