from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm.auto import tqdm
from transformers import pipeline

from data import DatasetBundle
from analysis.metrics import EvaluationResult, save_evaluation
from utils import pipeline_device_arg, save_json


def run_zero_shot(
    bundle: DatasetBundle,
    output_root: Path,
    device_name: str,
    model_name_or_path: str,
    batch_size: int,
    hypothesis_template: str,
) -> EvaluationResult:
    output_dir = output_root / "zero_shot"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "model_name_or_path": model_name_or_path,
            "candidate_labels": bundle.label_names,
            "hypothesis_template": hypothesis_template,
        },
        output_dir / "zero_shot_config.json",
    )

    classifier = pipeline(
        "zero-shot-classification",
        model=model_name_or_path,
        device=pipeline_device_arg(device_name),
    )

    probabilities = np.zeros((len(bundle.test_texts), len(bundle.label_names)), dtype=np.float32)
    predictions = np.zeros(len(bundle.test_texts), dtype=np.int64)

    label_to_id = {label: idx for idx, label in enumerate(bundle.label_names)}
    for start in tqdm(range(0, len(bundle.test_texts), batch_size), desc="zero-shot predict"):
        end = min(start + batch_size, len(bundle.test_texts))
        batch_texts = bundle.test_texts[start:end]
        outputs = classifier(
            batch_texts,
            candidate_labels=bundle.label_names,
            hypothesis_template=hypothesis_template,
            multi_label=False,
            batch_size=batch_size,
        )
        if isinstance(outputs, dict):
            outputs = [outputs]

        for offset, output in enumerate(outputs):
            row = start + offset
            for label, score in zip(output["labels"], output["scores"]):
                probabilities[row, label_to_id[label]] = float(score)
            predictions[row] = int(probabilities[row].argmax())

    return save_evaluation(
        model_name="zero_shot",
        texts=bundle.test_texts,
        y_true=bundle.test_labels,
        y_pred=predictions,
        label_names=bundle.label_names,
        output_dir=output_dir,
        probabilities=probabilities,
    )
