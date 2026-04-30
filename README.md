# CS371N Final Project Code

This repository contains reproducible code for the default-track emotion classification experiments on `dair-ai/emotion`.

Project layout:

- `src/run_experiments.py`: command-line entry point for the experiment stack.
- `src/approaches/`: the five modeling approaches: TF-IDF, GloVe MLP, LSTM, DistilBERT fine-tuning, and zero-shot transfer.
- `src/analysis/`: metric computation, summaries, error files, confusion matrices, and interpretability artifacts.
- `src/data.py`, `src/tokenization.py`, `src/utils.py`: shared dataset, preprocessing, and utility code.
- `src/requirements.txt`: Python dependencies.
- `documentation/`: assignment/progress PDFs and this README.

Run commands from the repository root.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r src/requirements.txt
```

## Run the experiments

```bash
python3 src/run_experiments.py
```

Outputs are written under `outputs/`, including metrics, predictions, confusion matrices, error samples, and TF-IDF interpretability artifacts.

The GloVe experiment no longer depends on `gensim`. By default it loads 50d GloVe vectors from the Hugging Face dataset `antokun/glove.6B.50d`.
