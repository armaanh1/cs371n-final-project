#!/usr/bin/env python3
"""Generate Captum Integrated Gradients artifacts for the fine-tuned DistilBERT model."""

from __future__ import annotations

import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import pandas as pd
import torch
from captum.attr import LayerIntegratedGradients
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from analysis.interpretability_utils import (
    ANALYSIS_DIR,
    OUTPUTS_DIR,
    ensure_analysis_dir,
    load_bundle_from_run_config,
    select_confident_correct_examples,
    summarize_token_scores,
    write_label_token_latex,
)
from utils import choose_device


def main() -> None:
    ensure_analysis_dir()
    bundle = load_bundle_from_run_config()
    output_dir = OUTPUTS_DIR / "distilbert"
    model_dir = output_dir / "model"

    with (output_dir / "model_config.json").open("r", encoding="utf-8") as handle:
        model_config = json.load(handle)
    with (OUTPUTS_DIR / "run_config.json").open("r", encoding="utf-8") as handle:
        run_config = json.load(handle)

    device = choose_device(str(run_config.get("device", "auto")))
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    lig = LayerIntegratedGradients(_forward_logits(model), model.distilbert.embeddings.word_embeddings)
    example_rows = []
    selected = select_confident_correct_examples(output_dir / "predictions.csv", bundle.label_names, per_label=3)

    for row in selected.itertuples(index=False):
        tokens, scores = distilbert_integrated_gradients(
            text=row.text,
            target_label_id=int(row.pred_label_id),
            tokenizer=tokenizer,
            lig=lig,
            max_length=int(model_config["max_length"]),
            device=device,
        )
        if not tokens:
            continue
        top_pairs = sorted(zip(tokens, scores), key=lambda item: item[1], reverse=True)[:8]
        example_rows.append(
            {
                "label": row.gold_label,
                "confidence": float(row.confidence),
                "text": row.text,
                "tokens": tokens,
                "scores": scores,
                "top_tokens": ", ".join(token for token, _ in top_pairs),
                "top_scores": ", ".join(f"{score:.4f}" for _, score in top_pairs),
            }
        )

    examples_df = pd.DataFrame(example_rows)
    examples_df.drop(columns=["tokens", "scores"]).to_csv(ANALYSIS_DIR / "distilbert_integrated_gradients_examples.csv", index=False)
    summary_df = summarize_token_scores(example_rows, "tokens", "scores", bundle.label_names)
    summary_df.to_csv(ANALYSIS_DIR / "distilbert_integrated_gradients_summary.csv", index=False)
    write_label_token_latex(
        summary_df,
        ANALYSIS_DIR / "distilbert_integrated_gradients_summary.tex",
        caption="Representative high-attribution tokens from DistilBERT Integrated Gradients.",
        label="tab:distilbert-integrated-gradients",
    )
    print(f"Wrote DistilBERT Integrated Gradients artifacts to {ANALYSIS_DIR}")


def _forward_logits(model):
    def forward(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return model(input_ids=input_ids, attention_mask=attention_mask).logits

    return forward


def distilbert_integrated_gradients(
    text: str,
    target_label_id: int,
    tokenizer,
    lig: LayerIntegratedGradients,
    max_length: int,
    device: torch.device,
) -> tuple[list[str], list[float]]:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    baseline_ids = torch.full_like(input_ids, fill_value=tokenizer.pad_token_id)

    attributions = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        additional_forward_args=(attention_mask,),
        target=target_label_id,
        n_steps=16,
    )
    token_scores = attributions.sum(dim=-1).squeeze(0).abs().detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
    merged_tokens, merged_scores = merge_wordpieces(tokens, token_scores, tokenizer)
    return merged_tokens, merged_scores


def merge_wordpieces(tokens: list[str], scores: list[float], tokenizer) -> tuple[list[str], list[float]]:
    merged_tokens: list[str] = []
    merged_scores: list[float] = []
    for token, score in zip(tokens, scores):
        if token in tokenizer.all_special_tokens:
            continue
        if token.startswith("##") and merged_tokens:
            merged_tokens[-1] = merged_tokens[-1] + token[2:]
            merged_scores[-1] += float(score)
        else:
            merged_tokens.append(token)
            merged_scores.append(float(score))
    return merged_tokens, merged_scores


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
