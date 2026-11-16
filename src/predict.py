"""Інференс: завантаження артефакту та вердикт для текстів.

Приклади:
    python src/predict.py "doesn't have any flaws" "never buying this again"
    python src/predict.py --file reviews.txt            # один текст на рядок
    python src/predict.py                               # smoke test на демо-фразах
З коду:
    from predict import SentimentPredictor
    SentimentPredictor().predict(["great sound"])
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch

from src.common import CLASSES, assign_classes, binary_entropy, clean_text, describe_rule, predict_texts_generic
from src.model import build_model

ROOT = Path(__file__).resolve().parents[1]
CKPT_VERSION = 1

try:                                            # .env необов'язковий: MODEL_PATH, NEUTRAL_LO, NEUTRAL_HI
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

DEMO = ["doesn't have any flaws", "i will never come to this bank again",
        "great sound but the battery is bad", "works fine", "the speaker broke after a week"]


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_artifact(path, map_location="cpu"):
    ck = torch.load(path, map_location=map_location, weights_only=True)
    assert ck["format_version"] == CKPT_VERSION, f"несумісна версія артефакту: {ck['format_version']}"
    emb0 = torch.zeros(len(ck["itos"]), ck["emb_dim"])          # справжні ваги прийдуть із state_dict
    model = build_model(ck["arch"], emb0, pad_idx=ck["pad_idx"])
    model.load_state_dict(ck["state_dict"])
    return model.eval(), {w: i for i, w in enumerate(ck["itos"])}, ck


class SentimentPredictor:
    def __init__(self, model_path=None, device="cpu", neutral_lo=None, neutral_hi=None):
        model_path = resolve(model_path or os.getenv("MODEL_PATH", "models/sentiment_rnn.pt"))
        self.device = device
        self.model, self.stoi, self.ck = load_artifact(model_path, map_location=device)
        self.model.to(device)
        self.rule = dict(self.ck["rule"])
        lo = neutral_lo if neutral_lo is not None else os.getenv("NEUTRAL_LO")
        hi = neutral_hi if neutral_hi is not None else os.getenv("NEUTRAL_HI")
        if lo is not None and hi is not None:                     # межі з .env перекривають збережені
            self.rule = {"type": "asym", "lo": float(lo), "hi": float(hi)}
        cl = self.ck["cleaning"]
        self.restore_neg, self.neg_fragments = cl["restore_neg"], cl["neg_fragments"]
        self.max_len = self.ck["max_len"]

    def predict(self, texts):
        cleaned = [clean_text(t, self.restore_neg, self.neg_fragments) for t in texts]
        p = predict_texts_generic(self.model, self.stoi, cleaned, self.max_len, self.device)
        oov = [float(np.mean([w not in self.stoi for w in c.split()])) if c.split() else 1.0 for c in cleaned]
        return pd.DataFrame({
            "text": list(texts), "cleaned": cleaned, "p_positive": p.round(4),
            "entropy": binary_entropy(p).round(4),
            "label": [CLASSES[i] for i in assign_classes(p, self.rule)],
            "oov_share": np.round(oov, 3),
        })


def main():
    ap = argparse.ArgumentParser(description="Вердикт тональності для текстів")
    ap.add_argument("texts", nargs="*", help="тексти (якщо не задано, беруться демо-фрази)")
    ap.add_argument("--file", help="файл з текстами, один на рядок")
    ap.add_argument("--model", default=None, help="шлях до артефакту (за замовчуванням MODEL_PATH або models/sentiment_rnn.pt)")
    ap.add_argument("--out", default="runs/smoke_predict/predictions.csv", help="куди зберегти результат")
    args = ap.parse_args()

    texts = list(args.texts)
    if args.file:
        texts += [l.strip() for l in open(resolve(args.file), encoding="utf8") if l.strip()]
    if not texts:
        texts = DEMO

    pred = SentimentPredictor(args.model)
    print("Правило:", describe_rule(pred.rule))
    res = pred.predict(texts)
    with pd.option_context("display.max_colwidth", 60, "display.width", 200):
        print(res[["text", "p_positive", "entropy", "label", "oov_share"]].to_string(index=False))

    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    print(f"\nSmoke test completed. Number of results: {len(res)}. Saved: {out}")


if __name__ == "__main__":
    main()
