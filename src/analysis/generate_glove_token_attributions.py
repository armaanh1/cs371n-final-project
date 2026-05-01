#!/usr/bin/env python3
"""Generate gradient-based token attributions for the GloVe MLP model."""

from __future__ import annotations

import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import pandas as pd
import torch

from analysis.interpretability_utils import (
    ANALYSIS_DIR,
    OUTPUTS_DIR,
    ensure_analysis_dir,
    load_bundle_from_run_config,
    select_confident_correct_examples,
    summarize_token_scores,
    write_label_token_latex,
)
from approaches.glove_mlp import EmbeddingMLP, global_mean_embedding, load_embeddings
from tokenization import simple_tokenize
from utils import choose_device


def main() -> None:
    ensure_analysis_dir()
    bundle = load_bundle_from_run_config()
    output_dir = OUTPUTS_DIR / "glove_mlp"

    with (output_dir / "model_config.json").open("r", encoding="utf-8") as handle:
        model_config = json.load(handle)
    with (output_dir / "embedding_metadata.json").open("r", encoding="utf-8") as handle:
        embedding_metadata = json.load(handle)
    with (OUTPUTS_DIR / "run_config.json").open("r", encoding="utf-8") as handle:
        run_config = json.load(handle)

    device = choose_device(str(run_config.get("device", "auto")))
    embeddings = load_embeddings(
        embedding_name=str(embedding_metadata["embedding_name"]),
        glove_path=Path(run_config["glove_path"]) if run_config.get("glove_path") else None,
        max_vectors=int(run_config.get("glove_max_vectors", 0)),
    )
    fallback_embedding = global_mean_embedding(embeddings)
    model = EmbeddingMLP(
        input_dim=int(model_config["input_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        num_labels=int(model_config["num_labels"]),
    ).to(device)
    model.load_state_dict(torch.load(output_dir / "model.pt", map_location=device))
    model.eval()

    example_rows = []
    selected = select_confident_correct_examples(output_dir / "predictions.csv", bundle.label_names, per_label=5)
    for row in selected.itertuples(index=False):
        tokens, scores = glove_token_attributions(
            text=row.text,
            target_label_id=int(row.pred_label_id),
            embeddings=embeddings,
            fallback_embedding=fallback_embedding,
            model=model,
            max_tokens=int(embedding_metadata["max_tokens"]),
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
    examples_df.drop(columns=["tokens", "scores"]).to_csv(ANALYSIS_DIR / "glove_token_attribution_examples.csv", index=False)
    summary_df = summarize_token_scores(example_rows, "tokens", "scores", bundle.label_names)
    summary_df.to_csv(ANALYSIS_DIR / "glove_token_attribution_summary.csv", index=False)
    write_label_token_latex(
        summary_df,
        ANALYSIS_DIR / "glove_token_attribution_summary.tex",
        caption="Representative high-attribution tokens for the GloVe MLP model.",
        label="tab:glove-token-attribution",
    )
    print(f"Wrote GloVe attribution artifacts to {ANALYSIS_DIR}")


def glove_token_attributions(
    text: str,
    target_label_id: int,
    embeddings,
    fallback_embedding: np.ndarray,
    model: EmbeddingMLP,
    max_tokens: int,
    device: torch.device,
) -> tuple[list[str], list[float]]:
    del fallback_embedding
    tokens = simple_tokenize(text)[:max_tokens]
    kept_tokens = [token for token in tokens if token in embeddings]
    if not kept_tokens:
        return [], []

    vector_array = np.stack([np.asarray(embeddings[token], dtype=np.float32) for token in kept_tokens])
    token_vectors = torch.tensor(vector_array, dtype=torch.float32, device=device, requires_grad=True)
    mean_part = token_vectors.mean(dim=0)
    max_part = token_vectors.max(dim=0).values
    feature = torch.cat([mean_part, max_part], dim=0)
    norm = torch.linalg.norm(feature)
    if float(norm.item()) > 0.0:
        feature = feature / norm

    model.zero_grad(set_to_none=True)
    logits = model(feature.unsqueeze(0))
    logits[0, target_label_id].backward()
    grads = token_vectors.grad
    scores = (grads * token_vectors).sum(dim=1).abs().detach().cpu().numpy()
    return kept_tokens, scores.astype(float).tolist()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
