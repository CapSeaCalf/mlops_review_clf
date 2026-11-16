"""Модель: Embedding -> LayerNorm -> BiGRU/BiLSTM -> LayerNorm -> self-attention ->
mean + max + attention-пулінг -> LayerNorm -> MLP-голова -> один логіт.

Клас ідентичний тому, що в ноутбуці; параметри беруться зі словника arch (configs/config.yaml, розділ model).
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.common import PAD_IDX


def make_norm(kind, dim):
    if kind == "batch":
        return nn.BatchNorm1d(dim)
    if kind == "layer":
        return nn.LayerNorm(dim)
    return nn.Identity()


class BiRNNClassifier(nn.Module):
    def __init__(self, emb_matrix, hidden_size=256, num_layers=2, head_dims=(256, 128, 64),
                 emb_dropout=0.2, rnn_dropout=0.2, head_dropout=0.3, norm="batch",
                 freeze_emb=True, pad_idx=PAD_IDX, rnn_type="lstm",
                 emb_norm=True, out_norm=True, pool_norm=True,
                 self_attn=True, attn_pool=True, attn_heads=4, attn_dropout=0.1):
        super().__init__()
        emb_dim, d = emb_matrix.size(1), 2 * hidden_size
        self.embedding = nn.Embedding.from_pretrained(emb_matrix, freeze=freeze_emb, padding_idx=pad_idx)
        self.emb_norm = nn.LayerNorm(emb_dim) if emb_norm else nn.Identity()
        self.emb_dropout = nn.Dropout(emb_dropout)

        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM}[rnn_type]
        self.rnn = rnn_cls(emb_dim, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True,
                           dropout=rnn_dropout if num_layers > 1 else 0.0)
        self.out_norm = nn.LayerNorm(d) if out_norm else nn.Identity()

        self.self_attn = nn.MultiheadAttention(d, attn_heads, dropout=attn_dropout, batch_first=True) if self_attn else None
        self.attn_norm = nn.LayerNorm(d) if self_attn else None
        self.attn_drop = nn.Dropout(attn_dropout)

        self.pool_score = nn.Sequential(nn.Linear(d, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)) if attn_pool else None
        self.last_attn = None                      # ваги пулінгу [B, T] з останнього forward

        pooled_dim = d * (3 if attn_pool else 2)
        self.pool_norm = nn.LayerNorm(pooled_dim) if pool_norm else nn.Identity()

        layers, in_dim = [nn.Dropout(head_dropout)], pooled_dim
        for hd in head_dims:
            layers += [nn.Linear(in_dim, hd), make_norm(norm, hd), nn.ReLU(), nn.Dropout(head_dropout)]
            in_dim = hd
        layers.append(nn.Linear(in_dim, 1))
        self.fc = nn.Sequential(*layers)

        for name, p in self.rnn.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
                if rnn_type == "lstm" and "bias_ih" in name:
                    p.data[hidden_size:2 * hidden_size] = 1.0     # forget-gate bias = 1

    def forward(self, x, lengths):
        emb = self.emb_dropout(self.emb_norm(self.embedding(x)))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        out = self.out_norm(out)

        lens = lengths.to(x.device)
        valid = torch.arange(x.size(1), device=x.device)[None, :] < lens[:, None]
        mask = valid.unsqueeze(-1)

        if self.self_attn is not None:
            a, _ = self.self_attn(out, out, out, key_padding_mask=~valid, need_weights=False)
            out = self.attn_norm(out + self.attn_drop(a))

        mean_pool = (out * mask).sum(1) / lens.unsqueeze(1).clamp(min=1)
        max_pool = out.masked_fill(~mask, float("-inf")).max(1).values
        parts = [mean_pool, max_pool]
        if self.pool_score is not None:
            s = self.pool_score(out).squeeze(-1).masked_fill(~valid, float("-inf"))
            w = torch.softmax(s, dim=1)
            self.last_attn = w.detach()
            parts.append((w.unsqueeze(-1) * out).sum(1))

        h = self.pool_norm(torch.cat(parts, dim=1))
        return self.fc(h).squeeze(-1)                                          # [B] логіти


def build_model(arch, emb_matrix, pad_idx=PAD_IDX):
    """arch: словник model з config.yaml або з артефакту (head_dims може бути списком)."""
    arch = dict(arch)
    arch["head_dims"] = tuple(arch["head_dims"])
    return BiRNNClassifier(emb_matrix, pad_idx=pad_idx, **arch)
