#!/usr/bin/env python3
"""Run all interpretability artifact generators."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPTS = [
    "generate_tfidf_interpretability.py",
    "generate_glove_token_attributions.py",
    "generate_lstm_token_attributions.py",
    "generate_distilbert_integrated_gradients.py",
    "generate_zero_shot_template_sensitivity.py",
]


def main() -> None:
    root = Path(__file__).resolve().parent
    for script_name in SCRIPTS:
        script_path = root / script_name
        subprocess.run([sys.executable, str(script_path)], check=True)


if __name__ == "__main__":
    main()
