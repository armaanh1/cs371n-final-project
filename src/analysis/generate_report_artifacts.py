#!/usr/bin/env python3
"""Generate report-ready plots and tables from completed experiment outputs."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ANALYSIS_DIR = OUTPUTS_DIR / "analysis"


MODEL_DISPLAY_NAMES = {
    "tfidf_logreg": "TF-IDF logistic regression",
    "glove_mlp": "GloVe MLP",
    "lstm": "LSTM from scratch",
    "distilbert": "DistilBERT fine-tune",
    "zero_shot": "Zero-shot DistilBERT-MNLI",
}


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    summary = _load_results_summary()

    _write_test_result_tables(summary)
    _write_training_curve_data()
    _write_latex_include_snippets()
    _write_plots(summary)

    print(f"Wrote report artifacts to {ANALYSIS_DIR}")


def _load_results_summary() -> pd.DataFrame:
    summary_path = OUTPUTS_DIR / "results_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}. Run python3 src/run_experiments.py first.")
    summary = pd.read_csv(summary_path)
    summary["display_model"] = summary["model"].map(MODEL_DISPLAY_NAMES).fillna(summary["model"])
    return summary


def _write_test_result_tables(summary: pd.DataFrame) -> None:
    table = summary[
        ["display_model", "accuracy", "macro_f1", "weighted_f1", "num_test_examples"]
    ].rename(
        columns={
            "display_model": "Model",
            "accuracy": "Accuracy",
            "macro_f1": "Macro F1",
            "weighted_f1": "Weighted F1",
            "num_test_examples": "Test examples",
        }
    )
    table = table.sort_values("Macro F1", ascending=False)
    table.to_csv(ANALYSIS_DIR / "test_results_table.csv", index=False)
    table.to_latex(
        ANALYSIS_DIR / "test_results_table.tex",
        index=False,
        float_format="%.4f",
        caption="Test-set performance on the shared held-out split.",
        label="tab:test-results-generated",
    )


def _write_training_curve_data() -> None:
    rows = []
    for model in ["glove_mlp", "lstm", "distilbert"]:
        history_path = OUTPUTS_DIR / model / "training_history.json"
        if not history_path.exists():
            continue
        with history_path.open("r", encoding="utf-8") as handle:
            history = json.load(handle)
        for row in history:
            rows.append(
                {
                    "model": model,
                    "display_model": MODEL_DISPLAY_NAMES.get(model, model),
                    "epoch": row["epoch"],
                    "train_loss": row["train_loss"],
                    "val_accuracy": row["val_accuracy"],
                    "val_macro_f1": row["val_macro_f1"],
                }
            )

    if not rows:
        raise FileNotFoundError("No training_history.json files found under outputs/.")

    curves = pd.DataFrame(rows)
    curves.to_csv(ANALYSIS_DIR / "training_curves.csv", index=False)

    final_epochs = curves.sort_values("epoch").groupby("model", as_index=False).tail(1)
    final_epochs = final_epochs[
        ["display_model", "epoch", "train_loss", "val_accuracy", "val_macro_f1"]
    ].rename(
        columns={
            "display_model": "Model",
            "epoch": "Final epoch",
            "train_loss": "Train loss",
            "val_accuracy": "Val. accuracy",
            "val_macro_f1": "Val. macro F1",
        }
    )
    final_epochs.to_csv(ANALYSIS_DIR / "training_final_epoch_table.csv", index=False)
    final_epochs.to_latex(
        ANALYSIS_DIR / "training_final_epoch_table.tex",
        index=False,
        float_format="%.4f",
        caption="Final logged validation metrics for trained neural models.",
        label="tab:training-final-generated",
    )


def _write_plots(summary: pd.DataFrame) -> None:
    mpl_config_dir = ANALYSIS_DIR / ".mplconfig"
    xdg_cache_dir = ANALYSIS_DIR / ".cache"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = pd.read_csv(ANALYSIS_DIR / "training_curves.csv")
    plt.style.use("default")

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for model, group in curves.groupby("display_model"):
        group = group.sort_values("epoch")
        ax.plot(group["epoch"], group["val_macro_f1"], marker="o", linewidth=2, label=model)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation macro F1")
    ax.set_title("Validation Macro F1 During Training")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "training_curve_val_macro_f1.png", dpi=200)
    fig.savefig(ANALYSIS_DIR / "training_curve_val_macro_f1.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for model, group in curves.groupby("display_model"):
        group = group.sort_values("epoch")
        ax.plot(group["epoch"], group["train_loss"], marker="o", linewidth=2, label=model)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "training_curve_loss.png", dpi=200)
    fig.savefig(ANALYSIS_DIR / "training_curve_loss.pdf")
    plt.close(fig)

    sorted_summary = summary.sort_values("macro_f1", ascending=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.barh(sorted_summary["display_model"], sorted_summary["macro_f1"])
    ax.set_xlabel("Test macro F1")
    ax.set_title("Held-out Test Performance")
    ax.set_xlim(0, max(1.0, sorted_summary["macro_f1"].max() + 0.05))
    for idx, value in enumerate(sorted_summary["macro_f1"]):
        ax.text(value + 0.01, idx, f"{value:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "test_macro_f1_bar.png", dpi=200)
    fig.savefig(ANALYSIS_DIR / "test_macro_f1_bar.pdf")
    plt.close(fig)


def _write_latex_include_snippets() -> None:
    snippet = r"""
% Generated by: python3 src/analysis/generate_report_artifacts.py
% Copy these into acl_latex.tex where appropriate.

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{../outputs/analysis/training_curve_val_macro_f1.pdf}
  \caption{Validation macro F1 over training epochs for trained neural models.}
  \label{fig:training-curve}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{../outputs/analysis/test_macro_f1_bar.pdf}
  \caption{Macro F1 on the shared held-out test split.}
  \label{fig:test-macro-f1}
\end{figure}
"""
    (ANALYSIS_DIR / "latex_figure_snippets.tex").write_text(snippet.strip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
