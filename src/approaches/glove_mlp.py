from __future__ import annotations

import gzip
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from data import DatasetBundle
from analysis.metrics import EvaluationResult, save_evaluation
from tokenization import simple_tokenize
from utils import choose_device, save_json, set_global_seed, softmax_numpy


class EmbeddingLookup(Protocol):
    vector_size: int

    def __contains__(self, key: str) -> bool: ...

    def __getitem__(self, key: str) -> np.ndarray: ...


class PlainTextEmbeddings:
    def __init__(self, vectors: dict[str, np.ndarray], vector_size: int):
        self.vectors = vectors
        self.vector_size = vector_size

    def __contains__(self, key: str) -> bool:
        return key in self.vectors

    def __getitem__(self, key: str) -> np.ndarray:
        return self.vectors[key]


class EmbeddingMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_labels: int, dropout: float = 0.25):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def run_glove_mlp(
    bundle: DatasetBundle,
    output_root: Path,
    seed: int,
    device_name: str,
    embedding_name: str,
    glove_path: Path | None,
    max_vectors: int,
    max_tokens: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
) -> EvaluationResult:
    set_global_seed(seed)
    output_dir = output_root / "glove_mlp"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(device_name)

    embeddings = load_embeddings(embedding_name=embedding_name, glove_path=glove_path, max_vectors=max_vectors)
    x_train, train_coverage = average_embeddings(bundle.train_texts, embeddings, max_tokens=max_tokens)
    x_val, val_coverage = average_embeddings(bundle.val_texts, embeddings, max_tokens=max_tokens)
    x_test, test_coverage = average_embeddings(bundle.test_texts, embeddings, max_tokens=max_tokens)
    save_json(
        {
            "embedding_name": embedding_name,
            "glove_path": str(glove_path) if glove_path else None,
            "vector_size": embeddings.vector_size,
            "max_tokens": max_tokens,
            "coverage": {"train": train_coverage, "validation": val_coverage, "test": test_coverage},
        },
        output_dir / "embedding_metadata.json",
    )

    model = EmbeddingMLP(embeddings.vector_size, hidden_dim, len(bundle.label_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = _feature_loader(x_train, bundle.train_labels, batch_size=batch_size, shuffle=True)
    val_loader = _feature_loader(x_val, bundle.val_labels, batch_size=batch_size, shuffle=False)
    test_loader = _feature_loader(x_test, bundle.test_labels, batch_size=batch_size, shuffle=False)

    best_macro_f1 = -1.0
    best_state = None
    bad_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"glove_mlp epoch {epoch}/{epochs}", leave=False)
        for features, labels in progress:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        val_logits, val_labels = _predict_logits(model, val_loader, device)
        val_probs = softmax_numpy(val_logits)
        val_result = save_evaluation(
            model_name="glove_mlp_validation",
            texts=bundle.val_texts,
            y_true=val_labels,
            y_pred=val_probs.argmax(axis=1),
            label_names=bundle.label_names,
            output_dir=output_dir / "validation",
            probabilities=val_probs,
            write_predictions=False,
        )
        val_macro_f1 = val_result.metrics["macro_f1"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(1, len(train_loader)),
                "val_macro_f1": val_macro_f1,
                "val_accuracy": val_result.metrics["accuracy"],
            }
        )

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    save_json(history, output_dir / "training_history.json")
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), output_dir / "model.pt")
    joblib.dump({"x_train_shape": x_train.shape, "x_val_shape": x_val.shape, "x_test_shape": x_test.shape}, output_dir / "feature_shapes.joblib")

    test_logits, test_labels = _predict_logits(model, test_loader, device)
    test_probs = softmax_numpy(test_logits)
    return save_evaluation(
        model_name="glove_mlp",
        texts=bundle.test_texts,
        y_true=test_labels,
        y_pred=test_probs.argmax(axis=1),
        label_names=bundle.label_names,
        output_dir=output_dir,
        probabilities=test_probs,
    )


def load_embeddings(embedding_name: str, glove_path: Path | None, max_vectors: int = 0) -> EmbeddingLookup:
    if glove_path is not None:
        return load_plain_text_embeddings(glove_path, max_vectors=max_vectors)

    return load_huggingface_glove(embedding_name=embedding_name, max_vectors=max_vectors)


def load_huggingface_glove(embedding_name: str, max_vectors: int = 0) -> PlainTextEmbeddings:
    """Load 50d GloVe vectors from a Hugging Face dataset without requiring gensim."""
    dataset_name = _resolve_glove_dataset_name(embedding_name)
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The default GloVe loader requires the datasets package. "
            "Install src/requirements.txt or set glove_path in ExperimentConfig."
        ) from exc

    rows = load_dataset(dataset_name, split="train")
    vectors: dict[str, np.ndarray] = {}
    vector_size = None

    for row in tqdm(rows, desc=f"load {dataset_name}"):
        line = row.get("text")
        if line is None:
            line = next(iter(row.values()))
        parsed = _parse_embedding_line(str(line), expected_size=vector_size)
        if parsed is None:
            continue
        word, vector = parsed
        if vector_size is None:
            vector_size = len(vector)
        vectors[word] = vector
        if max_vectors > 0 and len(vectors) >= max_vectors:
            break

    if vector_size is None or not vectors:
        raise ValueError(f"No embeddings were loaded from Hugging Face dataset {dataset_name}")
    return PlainTextEmbeddings(vectors, vector_size)


def _resolve_glove_dataset_name(embedding_name: str) -> str:
    aliases = {
        "glove-wiki-gigaword-50": "antokun/glove.6B.50d",
        "glove.6B.50d": "antokun/glove.6B.50d",
        "antokun/glove.6B.50d": "antokun/glove.6B.50d",
    }
    if embedding_name in aliases:
        return aliases[embedding_name]
    if "/" in embedding_name:
        return embedding_name
    raise ValueError(
        f"Unknown GloVe embedding source '{embedding_name}'. "
        "Use antokun/glove.6B.50d or set glove_path in ExperimentConfig."
    )


def load_plain_text_embeddings(path: Path, max_vectors: int = 0) -> PlainTextEmbeddings:
    opener = gzip.open if path.suffix == ".gz" else open
    vectors: dict[str, np.ndarray] = {}
    vector_size = None

    with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
        for line_number, line in enumerate(handle):
            parts = line.rstrip().split()
            if not parts:
                continue
            if line_number == 0 and len(parts) == 2 and all(part.isdigit() for part in parts):
                continue
            parsed = _parse_embedding_line(line, expected_size=vector_size)
            if parsed is None:
                continue
            word, vector = parsed
            if vector_size is None:
                vector_size = len(vector)
            vectors[word] = vector
            if max_vectors > 0 and len(vectors) >= max_vectors:
                break

    if vector_size is None or not vectors:
        raise ValueError(f"No embeddings were loaded from {path}")
    return PlainTextEmbeddings(vectors, vector_size)


def _parse_embedding_line(line: str, expected_size: int | None) -> tuple[str, np.ndarray] | None:
    parts = line.rstrip().split()
    if len(parts) < 3:
        return None
    word, values = parts[0], parts[1:]
    if expected_size is not None and len(values) != expected_size:
        return None
    try:
        vector = np.asarray(values, dtype=np.float32)
    except ValueError:
        return None
    return word, vector


def average_embeddings(texts: list[str], embeddings: EmbeddingLookup, max_tokens: int) -> tuple[np.ndarray, dict[str, float]]:
    features = np.zeros((len(texts), embeddings.vector_size), dtype=np.float32)
    total_tokens = 0
    found_tokens = 0
    texts_with_embedding = 0

    for row_idx, text in enumerate(texts):
        vectors = []
        tokens = simple_tokenize(text)[:max_tokens]
        total_tokens += len(tokens)
        for token in tokens:
            if token in embeddings:
                vectors.append(np.asarray(embeddings[token], dtype=np.float32))
                found_tokens += 1
        if vectors:
            features[row_idx] = np.mean(vectors, axis=0)
            texts_with_embedding += 1

    coverage = {
        "token_coverage": found_tokens / max(1, total_tokens),
        "texts_with_any_embedding": texts_with_embedding / max(1, len(texts)),
        "total_tokens": float(total_tokens),
        "found_tokens": float(found_tokens),
    }
    return features, coverage


def _feature_loader(features: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def _predict_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits = []
    labels = []
    for features, batch_labels in tqdm(loader, desc="glove_mlp predict", leave=False):
        features = features.to(device)
        logits.append(model(features).detach().cpu().numpy())
        labels.append(batch_labels.numpy())
    return np.concatenate(logits, axis=0), np.concatenate(labels, axis=0)
