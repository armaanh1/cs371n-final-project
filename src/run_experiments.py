#!/usr/bin/env python3
"""Run the CS371N emotion classification experiment stack."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


APPROACH_ORDER = ["tfidf", "glove", "lstm", "distilbert", "zero_shot"]


@dataclass(frozen=True)
class ExperimentConfig:
    dataset_name: str = "dair-ai/emotion"
    text_column: str = "text"
    label_column: str = "label"
    output_dir: Path = CODE_ROOT.parent / "outputs"
    seed: int = 13
    device: str = "auto"

    max_train_examples: int = 0
    max_val_examples: int = 0
    max_test_examples: int = 0
    validation_fraction: float = 0.1

    tfidf_max_features: int = 50_000
    tfidf_ngram_max: int = 2
    tfidf_min_df: int = 1
    tfidf_c: float = 2.0

    glove_name: str = "antokun/glove.6B.50d"
    glove_path: Path | None = None
    glove_max_vectors: int = 0
    glove_max_tokens: int = 128
    glove_hidden_dim: int = 128
    glove_epochs: int = 20
    glove_batch_size: int = 64
    glove_lr: float = 1e-3
    glove_patience: int = 4

    lstm_max_vocab: int = 20_000
    lstm_max_length: int = 80
    lstm_embedding_dim: int = 128
    lstm_hidden_dim: int = 128
    lstm_layers: int = 1
    lstm_epochs: int = 8
    lstm_batch_size: int = 64
    lstm_lr: float = 1e-3
    lstm_patience: int = 3

    distilbert_model: str = "distilbert-base-uncased"
    distilbert_max_length: int = 128
    distilbert_epochs: int = 3
    distilbert_batch_size: int = 16
    distilbert_lr: float = 2e-5
    distilbert_weight_decay: float = 0.01
    distilbert_warmup_ratio: float = 0.06
    distilbert_patience: int = 2

    zero_shot_model: str = "typeform/distilbert-base-uncased-mnli"
    zero_shot_batch_size: int = 8
    zero_shot_template: str = "This text expresses {}."


def main() -> None:
    if len(sys.argv) > 1:
        raise SystemExit("This runner takes no flags. Use: python3 src/run_experiments.py")

    config = ExperimentConfig()
    approaches = APPROACH_ORDER

    from analysis.artifacts import save_summary
    from approaches.distilbert_finetune import run_distilbert_finetune
    from approaches.glove_mlp import run_glove_mlp
    from approaches.lstm import run_lstm
    from approaches.tfidf import run_tfidf
    from approaches.zero_shot import run_zero_shot
    from data import load_text_classification_dataset
    from utils import save_json, set_global_seed

    set_global_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(asdict(config) | {"approaches_resolved": approaches}, config.output_dir / "run_config.json")

    bundle = load_text_classification_dataset(
        dataset_name=config.dataset_name,
        text_column=config.text_column,
        label_column=config.label_column,
        seed=config.seed,
        max_train_examples=config.max_train_examples,
        max_val_examples=config.max_val_examples,
        max_test_examples=config.max_test_examples,
        validation_fraction=config.validation_fraction,
    )
    save_json(bundle.metadata, config.output_dir / "dataset_metadata.json")

    results = []
    for model_name in approaches:
        if model_name == "tfidf":
            results.append(
                run_tfidf(
                    bundle=bundle,
                    output_root=config.output_dir,
                    seed=config.seed,
                    max_features=config.tfidf_max_features,
                    ngram_max=config.tfidf_ngram_max,
                    min_df=config.tfidf_min_df,
                    c=config.tfidf_c,
                )
            )
        elif model_name == "glove":
            results.append(
                run_glove_mlp(
                    bundle=bundle,
                    output_root=config.output_dir,
                    seed=config.seed,
                    device_name=config.device,
                    embedding_name=config.glove_name,
                    glove_path=config.glove_path,
                    max_vectors=config.glove_max_vectors,
                    max_tokens=config.glove_max_tokens,
                    hidden_dim=config.glove_hidden_dim,
                    epochs=config.glove_epochs,
                    batch_size=config.glove_batch_size,
                    lr=config.glove_lr,
                    patience=config.glove_patience,
                )
            )
        elif model_name == "lstm":
            results.append(
                run_lstm(
                    bundle=bundle,
                    output_root=config.output_dir,
                    seed=config.seed,
                    device_name=config.device,
                    max_vocab=config.lstm_max_vocab,
                    max_length=config.lstm_max_length,
                    embedding_dim=config.lstm_embedding_dim,
                    hidden_dim=config.lstm_hidden_dim,
                    num_layers=config.lstm_layers,
                    epochs=config.lstm_epochs,
                    batch_size=config.lstm_batch_size,
                    lr=config.lstm_lr,
                    patience=config.lstm_patience,
                )
            )
        elif model_name == "distilbert":
            results.append(
                run_distilbert_finetune(
                    bundle=bundle,
                    output_root=config.output_dir,
                    seed=config.seed,
                    device_name=config.device,
                    model_name_or_path=config.distilbert_model,
                    max_length=config.distilbert_max_length,
                    epochs=config.distilbert_epochs,
                    batch_size=config.distilbert_batch_size,
                    lr=config.distilbert_lr,
                    weight_decay=config.distilbert_weight_decay,
                    warmup_ratio=config.distilbert_warmup_ratio,
                    patience=config.distilbert_patience,
                )
            )
        elif model_name == "zero_shot":
            results.append(
                run_zero_shot(
                    bundle=bundle,
                    output_root=config.output_dir,
                    device_name=config.device,
                    model_name_or_path=config.zero_shot_model,
                    batch_size=config.zero_shot_batch_size,
                    hypothesis_template=config.zero_shot_template,
                )
            )

    summary_path = save_summary(results, config.output_dir)
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
