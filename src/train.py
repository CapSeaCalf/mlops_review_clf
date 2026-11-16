"""Навчання: configs/config.yaml + data/processed/labeled.csv -> артефакт у тому ж форматі, що й у ноутбуці.

Словник і вектори FastText беруться з готового артефакту (models/sentiment_rnn.pt, ключ embedding.weight),
тому FastText на сервері не потрібен.

    python src/train.py --smoke                 # дуже короткий прогін на CPU -> runs/smoke_train/
    python src/train.py                         # повне навчання за config.yaml -> runs/train/
    python src/train.py --save-model            # додатково перезаписати models/sentiment_rnn.pt
"""
import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence

from src.common import PAD_IDX, UNK_IDX, assign_classes, describe_rule, predict_texts_generic
from src.model import build_model
from src.predict import CKPT_VERSION, resolve


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def encode(texts, stoi, max_len):
    return [torch.tensor([stoi.get(w, UNK_IDX) for w in str(t).split()[:max_len]] or [UNK_IDX], dtype=torch.long)
            for t in texts]


def batches(ids, y, bs, shuffle, seed=0):
    idx = np.arange(len(ids))
    if shuffle:
        np.random.RandomState(seed).shuffle(idx)
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        seqs = [ids[j] for j in b]
        lens = torch.tensor([len(s) for s in seqs], dtype=torch.long)
        yield pad_sequence(seqs, batch_first=True, padding_value=PAD_IDX), lens, torch.as_tensor(y[b], dtype=torch.float32)


def macro_f1(p, y3, rule):
    return f1_score(y3, assign_classes(p, rule), average="macro", labels=[0, 1, 2], zero_division=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--data", default="data/processed/labeled.csv")
    ap.add_argument("--base-artifact", default=None, help="звідки взяти словник і вектори (за замовчуванням MODEL_PATH)")
    ap.add_argument("--out", default=None, help="каталог результатів (runs/train або runs/smoke_train)")
    ap.add_argument("--smoke", action="store_true", help="мінімальний прогін: вибірка, 1 епоха")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--save-model", action="store_true", help="перезаписати models/sentiment_rnn.pt новим артефактом")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(resolve(args.config), encoding="utf8"))
    tr_cfg, rule, seed = cfg["training"], dict(cfg["rule"]), cfg["seed"]
    rule = {k: rule[k] for k in rule if k in ("type", "lo", "hi", "thr")}
    out_dir = resolve(args.out or ("runs/smoke_train" if args.smoke else "runs/train"))
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    torch.set_num_threads(max(1, (os.cpu_count() or 2)))

    # ---- базовий артефакт: словник + вектори
    base_path = resolve(args.base_artifact or os.getenv("MODEL_PATH", "models/sentiment_rnn.pt"))
    base = torch.load(base_path, map_location="cpu", weights_only=True)
    itos, stoi = base["itos"], {w: i for i, w in enumerate(base["itos"])}
    emb = base["state_dict"]["embedding.weight"].float()
    max_len = cfg["preprocessing"]["max_len"]
    print(f"Базовий артефакт: {base_path.name}, словник {len(itos)}, dim {emb.size(1)}")

    # ---- дані
    df = pd.read_csv(resolve(args.data)).dropna(subset=["cleaned_review", "y_final"]).reset_index(drop=True)
    df["y_final"] = df["y_final"].astype(int)
    is_csv = df["src"].eq("csv") if "src" in df else pd.Series(True, index=df.index)
    csv_df, extra = df[is_csv], df[~is_csv]
    if args.smoke:
        csv_df = pd.concat([csv_df[csv_df["y_final"] == c].sample(min(int((csv_df["y_final"] == c).sum()), 200), random_state=seed)
                            for c in (0, 1, 2)])
        extra = extra.sample(min(len(extra), 150), random_state=seed)
    tr, tmp = train_test_split(csv_df, test_size=0.3, stratify=csv_df["y_final"], random_state=seed)
    va, te = train_test_split(tmp, test_size=0.5, stratify=tmp["y_final"], random_state=seed)
    tr = pd.concat([tr, extra], ignore_index=True)          # наші фрази йдуть лише в train
    print(f"train {len(tr)} (з них наших {len(extra)}), val {len(va)}, test {len(te)}")

    sc = {int(k): float(v) for k, v in tr_cfg["scalar_target"].items()}
    smooth = tr_cfg["smooth"]

    def targets(d):
        t = d["y_final"].map(sc).to_numpy(dtype=np.float32)
        return t * (1 - smooth) + 0.5 * smooth

    ids = {n: encode(d["cleaned_review"], stoi, max_len) for n, d in (("tr", tr), ("va", va), ("te", te))}
    y_tr = targets(tr)
    y3 = {"tr": tr["y_final"].to_numpy(), "va": va["y_final"].to_numpy(), "te": te["y_final"].to_numpy()}

    # ---- модель
    model = build_model(cfg["model"], emb, pad_idx=PAD_IDX)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tr_cfg["lr"], weight_decay=tr_cfg["weight_decay"])
    cnt = np.bincount(y3["tr"], minlength=3).astype(float)
    cw = (cnt.sum() / (3 * np.maximum(cnt, 1))) ** tr_cfg["weight_power"]
    cw_t = torch.tensor(cw, dtype=torch.float32)
    epochs = args.epochs or (1 if args.smoke else tr_cfg["epochs"])
    bs = 64 if args.smoke else tr_cfg["batch_size"]
    y3_tr = torch.as_tensor(y3["tr"])
    bce = nn.BCEWithLogitsLoss(reduction="none")

    best, best_state, bad, hist = -1.0, None, 0, []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        order = np.random.RandomState(seed + ep).permutation(len(tr))
        tot = 0.0
        for i in range(0, len(order), bs):
            b = order[i:i + bs]
            seqs = [ids["tr"][j] for j in b]
            lens = torch.tensor([len(s) for s in seqs], dtype=torch.long)
            x = pad_sequence(seqs, batch_first=True, padding_value=PAD_IDX)
            yt = torch.as_tensor(y_tr[b])
            w = cw_t[y3_tr[b]]
            loss = (bce(model(x, lens), yt) * w).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, tr_cfg["clip_norm"])
            opt.step()
            tot += loss.item() * len(b)
        p_va = predict_texts_generic(model, stoi, va["cleaned_review"].tolist(), max_len, "cpu")
        f1 = macro_f1(p_va, y3["va"], rule)
        hist.append({"epoch": ep, "train_loss": tot / len(tr), "val_macro_f1": float(f1)})
        print(f"epoch {ep:>3}/{epochs}  loss {tot / len(tr):.4f}  val macro-F1 {f1:.4f}  ({time.time() - t0:.0f}s)")
        if f1 > best + tr_cfg["min_delta"]:
            best, best_state, bad = f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= tr_cfg["patience"]:
                print("early stopping")
                break
    model.load_state_dict(best_state)

    # ---- оцінка
    p_te = predict_texts_generic(model, stoi, te["cleaned_review"].tolist(), max_len, "cpu")
    pred = assign_classes(p_te, rule)
    m = {
        "rule": describe_rule(rule), "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "best_val_macro_f1": float(best), "test_macro_f1": float(f1_score(y3["te"], pred, average="macro", labels=[0, 1, 2], zero_division=0)),
        "test_accuracy": float(accuracy_score(y3["te"], pred)), "smoke": bool(args.smoke),
        "train_seconds": round(time.time() - t0, 1),
    }
    nn_mask = y3["te"] != 1
    if len(set(y3["te"][nn_mask])) == 2:
        m["test_auc_pos_vs_neg"] = float(roc_auc_score(y3["te"][nn_mask] == 2, p_te[nn_mask]))
    ht = resolve("data/samples/hand_test.csv")
    if ht.exists():
        h = pd.read_csv(ht)
        col = "cleaned_review" if "cleaned_review" in h else h.columns[0]
        ycol = "y_final" if "y_final" in h else ("hand_y" if "hand_y" in h else None)
        if ycol:
            ph = predict_texts_generic(model, stoi, h[col].astype(str).tolist(), max_len, "cpu")
            m["hand_test_accuracy"] = float(accuracy_score(h[ycol].astype(int), assign_classes(ph, rule)))
    print(json.dumps(m, indent=2, ensure_ascii=False))

    json.dump(m, open(out_dir / "metrics.json", "w"), indent=2, ensure_ascii=False)
    json.dump(hist, open(out_dir / "history.json", "w"), indent=2)
    ck = {k: base[k] for k in base if k != "state_dict"}
    ck.update({"format_version": CKPT_VERSION, "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
               "arch": {**cfg["model"], "head_dims": list(cfg["model"]["head_dims"])}, "rule": rule})
    torch.save(ck, out_dir / "sentiment_rnn.pt")
    print(f"Артефакт: {out_dir / 'sentiment_rnn.pt'}")
    if args.save_model:
        torch.save(ck, resolve("models/sentiment_rnn.pt"))
        print("models/sentiment_rnn.pt оновлено")
    print("Train completed." + (" (smoke)" if args.smoke else ""))


if __name__ == "__main__":
    main()
