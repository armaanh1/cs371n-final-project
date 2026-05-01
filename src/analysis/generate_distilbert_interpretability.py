#!/usr/bin/env python3
"""Generate attention-based interpretability artifacts for the fine-tuned DistilBERT model."""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ANALYSIS_DIR = OUTPUTS_DIR / "analysis"
DISTILBERT_DIR = OUTPUTS_DIR / "distilbert"
MODEL_DIR = DISTILBERT_DIR / "model"
PREDICTIONS_PATH = DISTILBERT_DIR / "predictions.csv"
METRICS_PATH = DISTILBERT_DIR / "metrics.json"

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "do", "did", "for",
    "from", "had", "has", "have", "he", "her", "his", "i", "if", "im", "in", "is", "it",
    "its", "ive", "me", "my", "of", "on", "or", "our", "she", "so", "that", "the", "their",
    "them", "they", "this", "to", "too", "was", "we", "were", "with", "you", "your",
}


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_inputs()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        attn_implementation="eager",
    )
    model.eval()

    predictions = pd.read_csv(PREDICTIONS_PATH)
    label_names = _load_label_names()
    max_length = _load_max_length(default=128)

    example_rows: list[dict[str, object]] = []
    token_totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    token_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    correct = predictions[predictions["gold_label"] == predictions["pred_label"]].copy()
    correct = correct.sort_values(["gold_label", "confidence"], ascending=[True, False])

    for label in label_names:
        label_rows = correct[correct["gold_label"] == label].head(5)
        for example_rank, row in enumerate(label_rows.itertuples(index=False), start=1):
            words, scores = _word_attention_scores(
                model=model,
                tokenizer=tokenizer,
                text=row.text,
                max_length=max_length,
            )
            if not words:
                continue

            top_pairs = sorted(zip(words, scores), key=lambda item: item[1], reverse=True)[:8]
            example_rows.append(
                {
                    "label": label,
                    "example_rank": example_rank,
                    "confidence": float(row.confidence),
                    "text": row.text,
                    "top_attention_tokens": ", ".join(word for word, _ in top_pairs),
                    "top_attention_scores": ", ".join(f"{score:.4f}" for _, score in top_pairs),
                }
            )

            for word, score in zip(words, scores):
                normalized = _normalize_word(word)
                if not normalized:
                    continue
                token_totals[label][normalized] += float(score)
                token_counts[label][normalized] += 1

    example_df = pd.DataFrame(example_rows)
    example_df.to_csv(ANALYSIS_DIR / "distilbert_attention_examples.csv", index=False)

    summary_rows: list[dict[str, object]] = []
    for label in label_names:
        ranked = sorted(
            token_totals[label].items(),
            key=lambda item: item[1],
            reverse=True,
        )
        for rank, (token, total_attention) in enumerate(ranked[:15], start=1):
            summary_rows.append(
                {
                    "label": label,
                    "rank": rank,
                    "token": token,
                    "total_attention": total_attention,
                    "count": token_counts[label][token],
                    "avg_attention": total_attention / token_counts[label][token],
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(ANALYSIS_DIR / "distilbert_attention_label_summary.csv", index=False)
    _write_attention_latex(summary_df)
    _write_attention_plot(summary_df)

    print(f"Wrote interpretability artifacts to {ANALYSIS_DIR}")


def _ensure_inputs() -> None:
    missing = [path for path in [MODEL_DIR, PREDICTIONS_PATH, METRICS_PATH] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required DistilBERT artifacts: {missing}")


def _load_label_names() -> list[str]:
    with METRICS_PATH.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    return list(metrics["label_names"])


def _load_max_length(default: int) -> int:
    config_path = DISTILBERT_DIR / "model_config.json"
    if not config_path.exists():
        return default
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return int(config.get("max_length", default))


@torch.no_grad()
def _word_attention_scores(model, tokenizer, text: str, max_length: int) -> tuple[list[str], list[float]]:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    outputs = model(**encoded, output_attentions=True)
    if not outputs.attentions:
        raise RuntimeError("Model did not return attention tensors.")
    attention = outputs.attentions[-1].mean(dim=1)[0]  # [seq, seq]
    cls_attention = attention[0].detach().cpu().numpy()  # attention from [CLS] to all tokens
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])

    words: list[str] = []
    scores: list[float] = []
    for token, score in zip(tokens, cls_attention):
        if token in tokenizer.all_special_tokens:
            continue
        if token.startswith("##") and words:
            words[-1] = words[-1] + token[2:]
            scores[-1] += float(score)
        else:
            words.append(token)
            scores.append(float(score))
    return words, scores


def _normalize_word(word: str) -> str | None:
    word = word.lower()
    word = re.sub(r"[^a-z]+", "", word)
    if len(word) < 3:
        return None
    if word in STOPWORDS:
        return None
    return word


def _write_attention_latex(summary_df: pd.DataFrame) -> None:
    rows = []
    for label, group in summary_df.groupby("label"):
        top_tokens = ", ".join(group.sort_values("rank").head(5)["token"].tolist())
        rows.append({"Label": label, "Top attended tokens": top_tokens})
    table = pd.DataFrame(rows)
    table.to_latex(
        ANALYSIS_DIR / "distilbert_attention_label_summary.tex",
        index=False,
        caption="Representative high-attention tokens from the fine-tuned DistilBERT model, aggregated over correct predictions.",
        label="tab:distilbert-attention-summary",
    )


def _write_attention_plot(summary_df: pd.DataFrame) -> None:
    mpl_config_dir = ANALYSIS_DIR / ".mplconfig"
    xdg_cache_dir = ANALYSIS_DIR / ".cache"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_df = summary_df.groupby("label", as_index=False).first()[["label", "token", "avg_attention"]]

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.barh(plot_df["label"], plot_df["avg_attention"])
    ax.set_xlabel("Average attention score")
    ax.set_title("Top DistilBERT Attention Token per Class")
    for idx, row in enumerate(plot_df.itertuples(index=False)):
        ax.text(float(row.avg_attention) + 0.001, idx, str(row.token), va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "distilbert_attention_top_tokens.png", dpi=200)
    fig.savefig(ANALYSIS_DIR / "distilbert_attention_top_tokens.pdf")
    plt.close(fig)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
