#!/usr/bin/env python3
"""Generate gradient-based token attributions for the LSTM model."""

from __future__ import annotations

import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import pandas as pd
import torch
from torch.nn.utils.rnn import pack_padded_sequence

from analysis.interpretability_utils import (
    ANALYSIS_DIR,
    OUTPUTS_DIR,
    ensure_analysis_dir,
    load_bundle_from_run_config,
    select_confident_correct_examples,
    summarize_token_scores,
    write_label_token_latex,
)
from approaches.lstm import LSTMClassifier
from tokenization import simple_tokenize
from utils import choose_device


def main() -> None:
    ensure_analysis_dir()
    bundle = load_bundle_from_run_config()
    output_dir = OUTPUTS_DIR / "lstm"

    with (output_dir / "model_config.json").open("r", encoding="utf-8") as handle:
        model_config = json.load(handle)
    with (output_dir / "vocab.json").open("r", encoding="utf-8") as handle:
        vocab = json.load(handle)
    with (OUTPUTS_DIR / "run_config.json").open("r", encoding="utf-8") as handle:
        run_config = json.load(handle)

    device = choose_device(str(run_config.get("device", "auto")))
    model = LSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=int(model_config["embedding_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        num_labels=len(bundle.label_names),
        padding_idx=int(vocab["<pad>"]),
        num_layers=int(model_config["num_layers"]),
    ).to(device)
    model.load_state_dict(torch.load(output_dir / "model.pt", map_location=device))
    model.eval()

    example_rows = []
    selected = select_confident_correct_examples(output_dir / "predictions.csv", bundle.label_names, per_label=5)
    for row in selected.itertuples(index=False):
        tokens, scores = lstm_token_attributions(
            text=row.text,
            target_label_id=int(row.pred_label_id),
            vocab=vocab,
            model=model,
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
    examples_df.drop(columns=["tokens", "scores"]).to_csv(ANALYSIS_DIR / "lstm_token_attribution_examples.csv", index=False)
    summary_df = summarize_token_scores(example_rows, "tokens", "scores", bundle.label_names)
    summary_df.to_csv(ANALYSIS_DIR / "lstm_token_attribution_summary.csv", index=False)
    write_label_token_latex(
        summary_df,
        ANALYSIS_DIR / "lstm_token_attribution_summary.tex",
        caption="Representative high-attribution tokens for the LSTM model.",
        label="tab:lstm-token-attribution",
    )
    print(f"Wrote LSTM attribution artifacts to {ANALYSIS_DIR}")


def lstm_token_attributions(
    text: str,
    target_label_id: int,
    vocab: dict[str, int],
    model: LSTMClassifier,
    max_length: int,
    device: torch.device,
) -> tuple[list[str], list[float]]:
    tokens = simple_tokenize(text)[:max_length]
    if not tokens:
        return [], []

    unk_id = int(vocab["<unk>"])
    token_ids = [int(vocab.get(token, unk_id)) for token in tokens]
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    lengths = torch.tensor([len(token_ids)], dtype=torch.long, device=device)

    embedded = model.embedding(input_ids)
    embedded.retain_grad()
    dropped = model.embedding_dropout(embedded)
    packed = pack_padded_sequence(dropped, lengths.cpu(), batch_first=True, enforce_sorted=False)
    _, (hidden, _) = model.lstm(packed)
    if model.bidirectional:
        representation = torch.cat([hidden[-2], hidden[-1]], dim=1)
    else:
        representation = hidden[-1]
    logits = model.classifier(model.output_dropout(representation))

    model.zero_grad(set_to_none=True)
    logits[0, target_label_id].backward()
    grads = embedded.grad[0]
    scores = (grads * embedded[0]).sum(dim=1).abs().detach().cpu().numpy()
    return tokens, scores.astype(float).tolist()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
