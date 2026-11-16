"""Спільні утиліти: очищення тексту, правило нейтральності, інференс.

Ці функції ті самі, що й у ноутбуці notebooks/00_research_sentiment_pipeline.ipynb:
артефакт моделі навчено з ними, тому в продакшені вони мають лишатися ідентичними.
"""
import re

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

PAD_IDX, UNK_IDX = 0, 1
CLASSES = ["negative", "neutral", "positive"]

# датасет розбив "doesn't" на "doesn" + "t" і втратив "not": повертаємо явне "not"
DEFAULT_NEG_FRAGMENTS = {
    "doesn": "does not", "didn": "did not", "isn": "is not", "wasn": "was not", "aren": "are not",
    "weren": "were not", "couldn": "could not", "wouldn": "would not", "shouldn": "should not",
    "hasn": "has not", "haven": "have not", "hadn": "had not", "don": "do not", "won": "will not",
    "mustn": "must not", "needn": "need not",
}


def restore_negation(text, neg_fragments=None):
    frag = DEFAULT_NEG_FRAGMENTS if neg_fragments is None else neg_fragments
    return " ".join(frag.get(w, w) for w in str(text).split())


def clean_text(s, restore_neg=True, neg_fragments=None):
    """Сирий текст -> формат датасету: нижній регістр, лише літери, апостроф розбиває слово,
    односимвольні токени геть (крім "i"), далі відновлення заперечень."""
    toks = [w for w in re.sub(r"[^a-z]+", " ", str(s).lower()).split() if len(w) > 1 or w == "i"]
    text = " ".join(toks)
    return restore_negation(text, neg_fragments) if restore_neg else text


def binary_entropy(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def entropy_to_p(thr):
    g = np.linspace(1e-7, 0.5, 200001)
    return float(g[min(np.searchsorted(binary_entropy(g), thr), len(g) - 1)])


def assign_classes(p, rule):
    """rule: {"type": "asym", "lo": a, "hi": b} або {"type": "entropy", "thr": t}.
    asym: p < lo негатив (0), p >= hi позитив (2), між ними нейтральний (1)."""
    p = np.asarray(p)
    if rule["type"] == "entropy":
        pred = np.where(p >= 0.5, 2, 0)
        pred[binary_entropy(p) >= rule["thr"]] = 1
        return pred
    pred = np.ones(len(p), dtype=int)
    pred[p < rule["lo"]] = 0
    pred[p >= rule["hi"]] = 2
    return pred


def rule_bounds(rule):
    if rule["type"] == "entropy":
        lo = entropy_to_p(rule["thr"])
        return lo, 1 - lo
    return rule["lo"], rule["hi"]


def describe_rule(rule):
    lo, hi = rule_bounds(rule)
    if rule["type"] == "entropy":
        return f"H(p) >= {rule['thr']:.2f} -> neutral, тобто p in ({lo:.2f}, {hi:.2f})"
    return f"p < {lo:.3f} -> neg, p >= {hi:.3f} -> pos, між ними neutral"


@torch.no_grad()
def predict_texts_generic(model, stoi, texts, max_len, device, batch=256):
    """p(позитив) для вже очищених текстів (див. clean_text)."""
    model.eval()
    ids = [torch.tensor([stoi.get(w, UNK_IDX) for w in t.split()[:max_len]] or [UNK_IDX], dtype=torch.long)
           for t in texts]
    out = []
    for i in range(0, len(ids), batch):
        b = ids[i:i + batch]
        lens = torch.tensor([len(s) for s in b], dtype=torch.long)
        x = pad_sequence(b, batch_first=True, padding_value=PAD_IDX)
        out.append(torch.sigmoid(model(x.to(device), lens)).cpu())
    return torch.cat(out).numpy() if out else np.zeros(0, dtype=np.float32)
