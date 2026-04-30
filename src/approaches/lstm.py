from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from data import DatasetBundle
from analysis.metrics import EvaluationResult, save_evaluation
from tokenization import build_vocab, encode_text
from utils import choose_device, save_json, set_global_seed, softmax_numpy


class SequenceDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray, vocab: dict[str, int], max_length: int):
        self.encoded = [encode_text(text, vocab, max_length=max_length) for text in texts]
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[list[int], int]:
        return self.encoded[idx], int(self.labels[idx])


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_labels: int,
        padding_idx: int,
        num_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.embedding_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        classifier_dim = hidden_dim * (2 if bidirectional else 1)
        self.output_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(classifier_dim, num_labels)
        self.bidirectional = bidirectional

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding_dropout(self.embedding(input_ids))
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        if self.bidirectional:
            representation = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            representation = hidden[-1]
        return self.classifier(self.output_dropout(representation))


def run_lstm(
    bundle: DatasetBundle,
    output_root: Path,
    seed: int,
    device_name: str,
    max_vocab: int,
    max_length: int,
    embedding_dim: int,
    hidden_dim: int,
    num_layers: int,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
) -> EvaluationResult:
    set_global_seed(seed)
    output_dir = output_root / "lstm"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(device_name)

    vocab = build_vocab(bundle.train_texts, max_vocab=max_vocab, min_freq=1)
    save_json(vocab, output_dir / "vocab.json")
    save_json(
        {
            "max_vocab": max_vocab,
            "actual_vocab_size": len(vocab),
            "max_length": max_length,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
        },
        output_dir / "model_config.json",
    )

    train_loader = DataLoader(
        SequenceDataset(bundle.train_texts, bundle.train_labels, vocab, max_length=max_length),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_sequences,
    )
    val_loader = DataLoader(
        SequenceDataset(bundle.val_texts, bundle.val_labels, vocab, max_length=max_length),
        batch_size=batch_size,
        collate_fn=collate_sequences,
    )
    test_loader = DataLoader(
        SequenceDataset(bundle.test_texts, bundle.test_labels, vocab, max_length=max_length),
        batch_size=batch_size,
        collate_fn=collate_sequences,
    )

    model = LSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_labels=len(bundle.label_names),
        padding_idx=vocab["<pad>"],
        num_layers=num_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_macro_f1 = -1.0
    best_state = None
    bad_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"lstm epoch {epoch}/{epochs}", leave=False)
        for input_ids, lengths, labels in progress:
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        val_logits, val_labels = _predict_logits(model, val_loader, device)
        val_probs = softmax_numpy(val_logits)
        val_result = save_evaluation(
            model_name="lstm_validation",
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

    test_logits, test_labels = _predict_logits(model, test_loader, device)
    test_probs = softmax_numpy(test_logits)
    return save_evaluation(
        model_name="lstm",
        texts=bundle.test_texts,
        y_true=test_labels,
        y_pred=test_probs.argmax(axis=1),
        label_names=bundle.label_names,
        output_dir=output_dir,
        probabilities=test_probs,
    )


def collate_sequences(batch: list[tuple[list[int], int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    max_length = int(lengths.max().item())
    input_ids = torch.zeros((len(sequences), max_length), dtype=torch.long)
    for row_idx, sequence in enumerate(sequences):
        input_ids[row_idx, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    return input_ids, lengths, torch.tensor(labels, dtype=torch.long)


@torch.no_grad()
def _predict_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits = []
    labels = []
    for input_ids, lengths, batch_labels in tqdm(loader, desc="lstm predict", leave=False):
        input_ids = input_ids.to(device)
        lengths = lengths.to(device)
        logits.append(model(input_ids, lengths).detach().cpu().numpy())
        labels.append(batch_labels.numpy())
    return np.concatenate(logits, axis=0), np.concatenate(labels, axis=0)
