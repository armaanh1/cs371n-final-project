from __future__ import annotations

import re
from collections import Counter


TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|[0-9]+")
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def simple_tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def build_vocab(texts: list[str], max_vocab: int, min_freq: int = 1) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(simple_tokenize(text))

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, count in counter.most_common():
        if count < min_freq:
            continue
        if len(vocab) >= max_vocab:
            break
        vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int], max_length: int) -> list[int]:
    unk_id = vocab[UNK_TOKEN]
    ids = [vocab.get(token, unk_id) for token in simple_tokenize(text)[:max_length]]
    return ids if ids else [unk_id]

