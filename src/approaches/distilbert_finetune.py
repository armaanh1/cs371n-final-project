from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from data import DatasetBundle
from analysis.metrics import EvaluationResult, save_evaluation
from utils import choose_device, save_json, set_global_seed, softmax_numpy


class TransformerTextDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray, tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(values[idx]) for key, values in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def run_distilbert_finetune(
    bundle: DatasetBundle,
    output_root: Path,
    seed: int,
    device_name: str,
    model_name_or_path: str,
    max_length: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    warmup_ratio: float,
    patience: int,
) -> EvaluationResult:
    set_global_seed(seed)
    output_dir = output_root / "distilbert"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(device_name)
    label2id = {label: idx for idx, label in enumerate(bundle.label_names)}
    id2label = {idx: label for idx, label in enumerate(bundle.label_names)}

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(bundle.label_names),
        label2id=label2id,
        id2label=id2label,
    )
    model.to(device)

    train_loader = DataLoader(
        TransformerTextDataset(bundle.train_texts, bundle.train_labels, tokenizer, max_length),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TransformerTextDataset(bundle.val_texts, bundle.val_labels, tokenizer, max_length),
        batch_size=batch_size,
    )
    test_loader = DataLoader(
        TransformerTextDataset(bundle.test_texts, bundle.test_labels, tokenizer, max_length),
        batch_size=batch_size,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max(1, epochs * len(train_loader))
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_macro_f1 = -1.0
    best_state = None
    bad_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"distilbert epoch {epoch}/{epochs}", leave=False)
        for batch in progress:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        val_logits, val_labels = _predict_logits(model, val_loader, device)
        val_preds = val_logits.argmax(axis=1)
        val_result = save_evaluation(
            model_name="distilbert_validation",
            texts=bundle.val_texts,
            y_true=val_labels,
            y_pred=val_preds,
            label_names=bundle.label_names,
            output_dir=output_dir / "validation",
            probabilities=softmax_numpy(val_logits),
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
            best_state = deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    save_json(history, output_dir / "training_history.json")
    if best_state is not None:
        model.load_state_dict(best_state)
    model.save_pretrained(output_dir / "model")
    tokenizer.save_pretrained(output_dir / "model")

    test_logits, test_labels = _predict_logits(model, test_loader, device)
    test_probs = softmax_numpy(test_logits)
    test_preds = test_probs.argmax(axis=1)
    return save_evaluation(
        model_name="distilbert",
        texts=bundle.test_texts,
        y_true=test_labels,
        y_pred=test_preds,
        label_names=bundle.label_names,
        output_dir=output_dir,
        probabilities=test_probs,
    )


@torch.no_grad()
def _predict_logits(model, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits = []
    labels = []
    for batch in tqdm(loader, desc="distilbert predict", leave=False):
        labels.append(batch["labels"].numpy())
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        logits.append(outputs.logits.detach().cpu().numpy())
    return np.concatenate(logits, axis=0), np.concatenate(labels, axis=0)
