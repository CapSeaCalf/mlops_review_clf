"""Робить малий семпл для швидких тестів: data/samples/sample_reviews.csv (його МОЖНА комітити).
Береться по N рядків датасету на клас + усі наші фрази (hand/syn).
    python scripts/make_sample.py --n 100
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100, help="рядків датасету на клас")
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

df = pd.read_csv(ROOT / "data/processed/labeled.csv")
csv = df[df["src"].eq("csv")]
parts = [csv[csv["y_final"] == c].sample(min(a.n, int((csv["y_final"] == c).sum())), random_state=a.seed) for c in (0, 1, 2)]
out = pd.concat(parts + [df[~df["src"].eq("csv")]], ignore_index=True)
dst = ROOT / "data/samples/sample_reviews.csv"
dst.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(dst, index=False)
print(f"Збережено {len(out)} рядків -> {dst}")
