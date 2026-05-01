#!/usr/bin/env python3
"""Generate zero-shot hypothesis-template sensitivity artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm
from transformers import pipeline

from analysis.interpretability_utils import ANALYSIS_DIR, OUTPUTS_DIR, ensure_analysis_dir, load_bundle_from_run_config
from utils import pipeline_device_arg


TEMPLATES = [
    "This text expresses {}.",
    "The emotion of this text is {}.",
    "Overall, the speaker feels {}.",
    "This statement conveys {}.",
    "The writer is feeling {}.",
]


def main() -> None:
    ensure_analysis_dir()
    bundle = load_bundle_from_run_config()
    with (OUTPUTS_DIR / "zero_shot" / "zero_shot_config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    with (OUTPUTS_DIR / "run_config.json").open("r", encoding="utf-8") as handle:
        run_config = json.load(handle)

    classifier = pipeline(
        "zero-shot-classification",
        model=config["model_name_or_path"],
        device=pipeline_device_arg(str(run_config.get("device", "auto"))),
    )

    rows = []
    all_predictions: dict[str, np.ndarray] = {}
    label_to_id = {label: idx for idx, label in enumerate(bundle.label_names)}
    batch_size = int(run_config.get("zero_shot_batch_size", 8))

    for template in TEMPLATES:
        predictions = predict_with_template(classifier, bundle.test_texts, bundle.label_names, label_to_id, template, batch_size)
        all_predictions[template] = predictions
        rows.append(
            {
                "template": template,
                "accuracy": accuracy_score(bundle.test_labels, predictions),
                "macro_f1": f1_score(bundle.test_labels, predictions, average="macro"),
                "weighted_f1": f1_score(bundle.test_labels, predictions, average="weighted"),
            }
        )

    default_predictions = all_predictions[TEMPLATES[0]]
    for row in rows:
        template_predictions = all_predictions[row["template"]]
        row["agreement_with_default"] = float(np.mean(template_predictions == default_predictions))

    summary_df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    summary_df.to_csv(ANALYSIS_DIR / "zero_shot_template_sensitivity.csv", index=False)
    summary_df.to_latex(
        ANALYSIS_DIR / "zero_shot_template_sensitivity.tex",
        index=False,
        caption="Zero-shot performance under different hypothesis templates.",
        label="tab:zero-shot-template-sensitivity",
    )
    print(f"Wrote zero-shot template sensitivity artifacts to {ANALYSIS_DIR}")


def predict_with_template(classifier, texts, label_names, label_to_id, template: str, batch_size: int) -> np.ndarray:
    predictions = np.zeros(len(texts), dtype=np.int64)
    for start in tqdm(range(0, len(texts), batch_size), desc=f"template: {template}", leave=False):
        end = min(start + batch_size, len(texts))
        outputs = classifier(
            texts[start:end],
            candidate_labels=label_names,
            hypothesis_template=template,
            multi_label=False,
            batch_size=batch_size,
        )
        if isinstance(outputs, dict):
            outputs = [outputs]
        for offset, output in enumerate(outputs):
            best_label = output["labels"][0]
            predictions[start + offset] = int(label_to_id[best_label])
    return predictions


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
