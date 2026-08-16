"""
Cascade FiLM Fusion experiment for LG-AI competition.

Hierarchy: Macro -> Sector -> Target/Filing (cascade FiLM modulation)
Residual connection: output = (1-sigmoid(alpha))*base + sigmoid(alpha)*film_out
alpha initialized at -2.0 -> sigmoid ~ 0.12 (88% base preserved at start)

Usage:
    python forecasting_task/run_cascade_film.py \
        --tickers AAPL,NVDA,MSFT \
        --model_type PatchTST \
        --num_epoch 30 --batch_size 64 --lr 1e-3

Grid (10 tickers):
    python forecasting_task/run_cascade_film.py \
        --run_grid --num_tickers 10 \
        --model_types PatchTST,FiLM \
        --seeds 7,13,42
"""

import argparse
import copy
import csv
import importlib
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
FORECASTING_DIR = Path(__file__).resolve().parent
REPO_ROOT = FORECASTING_DIR.parent
for _p in (str(FORECASTING_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MODEL_REGISTRY = {
    "patchtst": "forecasting_task.models.PatchTST",
    "film":     "forecasting_task.models.FiLM",
}

# Text level column prefixes (in cascade order)
LEVEL_PREFIXES = [
    ("macro",         ["macro_category"]),
    ("sector",        ["sector_category"]),
    ("target",        ["targetCompany_category", "filing_", "lseg_news"]),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_str_list(v):
    if not v:
        return []
    if isinstance(v, list):
        return v
    return [x.strip() for x in v.split(",") if x.strip()]

def _parse_int_list(v):
    return [int(x) for x in _parse_str_list(v)]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Embedding store (lazy singleton)
# ---------------------------------------------------------------------------
class EmbeddingStore:
    """
    Loads linq_textemb.parquet once; provides fast numpy lookup.
    Stores embeddings as float16 to save RAM (~7 GB for 857K x 4096).
    """
    _instance = None

    def __init__(self, emb_path: str):
        print(f"[EmbeddingStore] Loading {emb_path} ...")
        df = pd.read_parquet(emb_path)
        emb_cols = [c for c in df.columns if c != "text_id"]
        self.emb_dim = len(emb_cols)
        # Vectorized: build numpy matrix + index dict (fast, ~7 GB float16)
        ids = df["text_id"].values
        mat = df[emb_cols].values.astype(np.float16)   # [N, emb_dim]
        self._idx  = {tid: i for i, tid in enumerate(ids)}
        self._mat  = mat
        print(f"[EmbeddingStore] Loaded {len(ids):,} embeddings, dim={self.emb_dim}")

    @classmethod
    def get(cls, emb_path: str) -> "EmbeddingStore":
        if cls._instance is None:
            cls._instance = cls(emb_path)
        return cls._instance

    def lookup(self, text_id) -> np.ndarray:
        """Return float32 embedding; zeros for missing/NaN ids."""
        if text_id is None or (isinstance(text_id, float) and np.isnan(text_id)):
            return np.zeros(self.emb_dim, dtype=np.float32)
        idx = self._idx.get(text_id, None)
        if idx is None:
            return np.zeros(self.emb_dim, dtype=np.float32)
        return self._mat[idx].astype(np.float32)

    def lookup_mean(self, ids) -> np.ndarray:
        """Average embeddings for a list of text_ids (skip empty)."""
        valid_rows = []
        for tid in ids:
            if tid is None or (isinstance(tid, float) and np.isnan(tid)):
                continue
            idx = self._idx.get(tid, None)
            if idx is not None:
                valid_rows.append(idx)
        if not valid_rows:
            return np.zeros(self.emb_dim, dtype=np.float32)
        return self._mat[valid_rows].mean(axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def _get_level_id_cols(df_cols):
    """
    Return list of (level_name, [col_names]) matching LEVEL_PREFIXES.
    Dynamically finds all matching columns in the dataframe.
    """
    result = []
    for level_name, prefixes in LEVEL_PREFIXES:
        matched = []
        for pfx in prefixes:
            matched += [c for c in df_cols if c.startswith(pfx)]
        result.append((level_name, matched))
    return result


class LGAIDataset(Dataset):
    """
    Multi-ticker dataset for train/test.parquet + linq_textemb.parquet.

    - flag='train': uses train_path rows with date < 2022-01-01
    - flag='val':   uses train_path rows with 2022-01-01 <= date < 2023-01-01
    - flag='test':  uses test_path  rows with date >= 2023-01-01
      (test.parquet has data up to Dec 2023 including 2022 context for seq)

    Returns per sample:
        price_x    : [seq_len, 4]
        price_y    : [label+pred, 4]
        level_embs : [num_levels, emb_dim]
        x_mark     : [seq_len, 3]
        y_mark     : [label+pred, 3]
    """

    def __init__(
        self,
        train_path: str,
        emb_path: str,
        flag: str,
        tickers: list,
        seq_len: int = 64,
        label_len: int = 16,
        pred_len: int = 20,
        test_path: str = None,
        forecast_gap: int = 0,
        ema_span: int = 0,
    ):
        assert flag in ("train", "val", "test")
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.forecast_gap = forecast_gap
        self.ema_span = ema_span
        self.flag = flag

        store = EmbeddingStore.get(emb_path)
        self.emb_dim = store.emb_dim
        self.num_levels = len(LEVEL_PREFIXES)

        # For test, use test.parquet if provided (has 2019-2023 context)
        if flag == "test" and test_path is not None:
            src_path = test_path
        else:
            src_path = train_path

        print(f"[Dataset:{flag}] Loading from {src_path} ...")
        df_all = pd.read_parquet(src_path)
        df_all["date"] = pd.to_datetime(df_all["date"])
        df_all = df_all[df_all["ticker"].isin(tickers)].sort_values(["ticker", "date"])

        level_cols = _get_level_id_cols(list(df_all.columns))

        all_price_x, all_price_y = [], []
        all_level_embs = []
        all_x_mark, all_y_mark = [], []

        for ticker in tickers:
            df_t = df_all[df_all["ticker"] == ticker].reset_index(drop=True)
            total_horizon = forecast_gap + pred_len
            if len(df_t) < seq_len + total_horizon + 10:
                continue
            self._build_ticker(
                df_t, flag, seq_len, label_len, pred_len,
                level_cols, store,
                all_price_x, all_price_y, all_level_embs,
                all_x_mark, all_y_mark,
                forecast_gap=forecast_gap,
                ema_span=ema_span,
            )

        if not all_price_x:
            raise RuntimeError(
                f"[Dataset:{flag}] No samples found. Check date ranges and tickers."
            )
        self.price_x     = np.stack(all_price_x,     axis=0)
        self.price_y     = np.stack(all_price_y,     axis=0)
        self.level_embs  = np.stack(all_level_embs,  axis=0)
        self.x_mark      = np.stack(all_x_mark,      axis=0)
        self.y_mark      = np.stack(all_y_mark,      axis=0)
        print(f"[Dataset:{flag}] {len(self.price_x):,} samples, "
              f"price_x={self.price_x.shape}, level_embs={self.level_embs.shape}")

    @staticmethod
    def _build_ticker(
        df, flag, seq_len, label_len, pred_len,
        level_cols, store,
        out_px, out_py, out_le, out_xm, out_ym,
        forecast_gap: int = 0,
        ema_span: int = 0,
    ):
        dates = df["date"]
        n = len(df)
        total_horizon = forecast_gap + pred_len

        if flag == "test":
            # test.parquet has full history (2019-2023);
            # we predict Jan-Dec 2023 (need seq_len context before 2023-01-01)
            test_start_i = int((dates < "2023-01-01").sum())
            b1 = max(0, test_start_i - seq_len)
            b2 = n
        else:
            n_train = int((dates < "2022-01-01").sum())
            n_val   = int(((dates >= "2022-01-01") & (dates < "2023-01-01")).sum())
            if flag == "train":
                b1, b2 = 0, n_train
            else:  # val
                b1 = max(0, n_train - seq_len)
                b2 = n_train + n_val

        # Fit scaler on train portion only
        n_train_fit = int((dates < "2022-01-01").sum())

        price_raw = df[["open", "high", "low", "close"]].values.astype(np.float32)
        scaler = StandardScaler()
        scaler.fit(price_raw[:n_train_fit])
        price_norm = scaler.transform(price_raw).astype(np.float32)

        stamp = np.stack([
            dates.dt.month.values,
            dates.dt.day.values,
            dates.dt.weekday.values,
        ], axis=1).astype(np.float32)

        # Pre-compute level embeddings per row (expensive, done once)
        level_emb_arr = np.zeros((n, len(level_cols), store.emb_dim), dtype=np.float32)
        for li, (lname, cols) in enumerate(level_cols):
            if not cols:
                continue
            for row_i in range(n):
                ids = df.iloc[row_i][cols].tolist() if cols else []
                level_emb_arr[row_i, li] = store.lookup_mean(ids)

        for idx in range(b2 - b1 - seq_len - total_horizon + 1):
            s  = b1 + idx
            se = s + seq_len

            if forecast_gap > 0:
                # Direct forecasting: skip 'gap' days, predict 'pred_len' days
                target_start = se + forecast_gap
                target_end   = target_start + pred_len
                # price_y = [label context from input end] + [target days]
                if label_len > 0:
                    label_part  = price_norm[se - label_len:se]  # [label_len, 4]
                    target_part = price_norm[target_start:target_end]  # [pred_len, 4]
                    py = np.concatenate([label_part, target_part], axis=0)
                    # stamps
                    stamp_label  = stamp[se - label_len:se]
                    stamp_target = stamp[target_start:target_end]
                    ym = np.concatenate([stamp_label, stamp_target], axis=0)
                else:
                    py = price_norm[target_start:target_end]  # [pred_len, 4]
                    ym = stamp[target_start:target_end]
            else:
                # Original: contiguous multi-step prediction
                re = se - label_len
                ree = re + label_len + pred_len
                py = price_norm[re:ree]
                ym = stamp[re:ree]

            out_px.append(price_norm[s:se])
            out_py.append(py)
            # Text embedding: EMA over window or last day only
            if ema_span > 0:
                alpha_ema = 2.0 / (ema_span + 1)
                window_embs = level_emb_arr[s:se]  # [seq_len, num_levels, emb_dim]
                ema = window_embs[0].copy()
                for t in range(1, seq_len):
                    ema = alpha_ema * window_embs[t] + (1.0 - alpha_ema) * ema
                out_le.append(ema)
            else:
                out_le.append(level_emb_arr[se - 1])   # last timestep in window
            out_xm.append(stamp[s:se])
            out_ym.append(ym)

    def __len__(self):
        return len(self.price_x)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.price_x[idx]),
            torch.from_numpy(self.price_y[idx]),
            torch.from_numpy(self.level_embs[idx]),
            torch.from_numpy(self.x_mark[idx]),
            torch.from_numpy(self.y_mark[idx]),
        )


# ---------------------------------------------------------------------------
# Cascade FiLM Fusion Module
# ---------------------------------------------------------------------------
class FiLMLayer(nn.Module):
    """
    Single FiLM modulation: gamma*x + beta, both conditioned on text embedding.
    When bottleneck_dim is set, text_dim is first compressed to bottleneck_dim
    via a shared projection before computing gamma/beta. This prevents high-dim
    text noise (e.g. 4096-d) from directly corrupting time-series features.
    """
    def __init__(self, text_dim: int, feat_size: int, hidden: int = 256,
                 bottleneck_dim: int = 0):
        super().__init__()
        # Optional bottleneck: compress text_dim → bottleneck_dim first
        in_dim = text_dim
        if bottleneck_dim > 0:
            self.bottleneck = nn.Sequential(
                nn.Linear(text_dim, bottleneck_dim),
                nn.LayerNorm(bottleneck_dim),
                nn.GELU(),
            )
            in_dim = bottleneck_dim
        else:
            self.bottleneck = None

        # gamma branch (initialized so output ≈ 0 → residual = 1)
        self.gamma_net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, feat_size),
        )
        # beta branch
        self.beta_net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, feat_size),
        )
        # initialize last layers to near-zero so gamma≈0, beta≈0 at start
        nn.init.zeros_(self.gamma_net[-1].weight)
        nn.init.zeros_(self.gamma_net[-1].bias)
        nn.init.zeros_(self.beta_net[-1].weight)
        nn.init.zeros_(self.beta_net[-1].bias)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """
        x: [B, feat_size]
        e: [B, text_dim]
        returns: [B, feat_size]   (gamma+1)*x + beta  (residual form)
        """
        if self.bottleneck is not None:
            e = self.bottleneck(e)  # [B, bottleneck_dim]
        gamma = self.gamma_net(e)   # [B, feat_size]
        beta  = self.beta_net(e)    # [B, feat_size]
        return (1.0 + gamma) * x + beta


class CascadeFilmFusion(nn.Module):
    """
    Cascade FiLM: text levels modulate the forecast in hierarchy order.

    Architecture:
        flat(base_output) → FiLM_macro → FiLM_sector → FiLM_target → reshape
    Then mixed with base_output via learnable sigmoid gate (alpha):
        out = (1 - sigmoid(alpha)) * base_output + sigmoid(alpha) * film_output

    alpha initialized at -2.0 → sigmoid ≈ 0.12 (88% base preserved initially)
    """

    def __init__(
        self,
        c_out: int,
        pred_len: int,
        num_levels: int,
        text_dim: int,
        film_hidden: int = 256,
        dropout: float = 0.1,
        bottleneck_dim: int = 0,
    ):
        super().__init__()
        self.pred_len   = pred_len
        self.c_out      = c_out
        self.num_levels = num_levels
        feat_size = pred_len * c_out

        self.film_layers = nn.ModuleList([
            FiLMLayer(text_dim, feat_size, film_hidden, bottleneck_dim=bottleneck_dim)
            for _ in range(num_levels)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(feat_size) for _ in range(num_levels)
        ])
        self.dropout = nn.Dropout(dropout)

        # Learnable residual gate (per-level or global)
        self.alpha = nn.Parameter(torch.full((num_levels,), -2.0))  # sigmoid(-2)≈0.12

    def forward(self, base_output: torch.Tensor, level_embs: torch.Tensor) -> torch.Tensor:
        """
        base_output : [B, pred_len, c_out]
        level_embs  : [B, num_levels, text_dim]
        returns     : [B, pred_len, c_out]
        """
        B, L, C = base_output.shape
        feat_size = L * C

        x = base_output.reshape(B, feat_size)  # flatten

        for i in range(self.num_levels):
            e_i = level_embs[:, i, :]           # [B, text_dim]
            valid = (e_i.abs().sum(-1) > 1e-6).float().unsqueeze(-1)  # [B, 1]

            film_out = self.film_layers[i](x, e_i)
            film_out = self.norms[i](film_out)
            film_out = self.dropout(film_out)

            alpha_i = torch.sigmoid(self.alpha[i])
            # only blend where text is valid; skip empty embeddings
            x = (1 - alpha_i * valid) * x + alpha_i * valid * film_out

        film_reshaped = x.reshape(B, L, C)
        return film_reshaped


# ---------------------------------------------------------------------------
# Late Fusion (baseline comparison)
# ---------------------------------------------------------------------------
class LateFusion(nn.Module):
    """
    Late fusion: pool text levels, project to pred_len*c_out,
    then blend with base output via a fixed text_weight.

    output = (1 - text_weight) * base_output + text_weight * text_output

    When learnable_level_weights=True, uses softmax-normalized learnable weights
    instead of simple mean pooling across Macro/Sector/Target levels.
    """
    def __init__(self, c_out: int, pred_len: int, text_dim: int,
                 num_levels: int = 3, hidden: int = 256,
                 text_weight: float = 0.1,
                 learnable_level_weights: bool = False):
        super().__init__()
        self.text_weight = text_weight
        self.pred_len = pred_len
        self.c_out = c_out
        self.learnable_level_weights = learnable_level_weights
        if learnable_level_weights:
            self.level_weights = nn.Parameter(torch.zeros(num_levels))
        feat = pred_len * c_out
        self.net = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, feat),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, base_output: torch.Tensor,
                level_embs: torch.Tensor) -> torch.Tensor:
        """
        base_output : [B, pred_len, c_out]
        level_embs  : [B, num_levels, text_dim]
        """
        B, L, C = base_output.shape
        if self.learnable_level_weights:
            w_levels = F.softmax(self.level_weights, dim=0)  # [num_levels]
            pooled = (level_embs * w_levels.view(1, -1, 1)).sum(dim=1)  # [B, text_dim]
        else:
            pooled = level_embs.mean(dim=1)              # [B, text_dim]
        valid   = (pooled.abs().sum(-1) > 1e-6).float().view(B, 1, 1)
        text_out = self.net(pooled).view(B, L, C)     # [B, pred_len, c_out]
        w = self.text_weight
        return (1 - w * valid) * base_output + w * valid * text_out


# ---------------------------------------------------------------------------
# V5 Fusion Methods
# ---------------------------------------------------------------------------

class EncoderTokenFusion(nn.Module):
    """
    Inject pooled text embedding as an extra token into PatchTST encoder.
    The self-attention naturally learns text-price interactions.
    Only works with PatchTST backbone.
    """
    needs_raw_input = True  # signals run_epoch to pass raw inputs

    def __init__(self, configs, text_dim: int, num_levels: int):
        super().__init__()
        module = importlib.import_module(MODEL_REGISTRY["patchtst"])
        self.base = module.Model(configs, patch_len=configs.patch_len,
                                 stride=configs.patch_len)
        self.pred_len = configs.pred_len
        self.text_proj = nn.Linear(text_dim, configs.d_model)
        self.level_weights = nn.Parameter(torch.zeros(num_levels))

    def forward(self, price_x, x_mark, dec_inp, y_mark, level_embs):
        # Pool text levels with learnable weights
        w = F.softmax(self.level_weights, dim=0)
        pooled = (level_embs * w.view(1, -1, 1)).sum(dim=1)   # [B, text_dim]
        text_token = self.text_proj(pooled).unsqueeze(1)       # [B, 1, d_model]
        has_text = (pooled.abs().sum(-1) > 1e-6).float()       # [B]

        # PatchTST normalize
        means = price_x.mean(1, keepdim=True).detach()
        x_norm = price_x - means
        stdev = torch.sqrt(
            torch.var(x_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_norm = x_norm / stdev

        # Patch embedding
        enc_input = x_norm.permute(0, 2, 1)
        enc_out, n_vars = self.base.patch_embedding(enc_input)  # [B*nv, P, d]

        # Insert text token as extra patch (repeat for each variable)
        B = price_x.shape[0]
        text_exp = text_token.repeat_interleave(n_vars, dim=0)  # [B*nv, 1, d]
        # Zero out text token where no text available
        mask = has_text.repeat_interleave(n_vars).view(-1, 1, 1)
        text_exp = text_exp * mask
        enc_out = torch.cat([enc_out, text_exp], dim=1)         # [B*nv, P+1, d]

        # Encoder (self-attention sees text + price patches together)
        enc_out, _ = self.base.encoder(enc_out)
        enc_out = enc_out[:, :-1, :]  # remove text token before head

        # Head
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)
        dec_out = self.base.head(enc_out).permute(0, 2, 1)

        # De-normalize
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(
            1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(
            1, self.pred_len, 1)
        return dec_out[:, -self.pred_len:, :]


class GMUFusion(nn.Module):
    """
    Gated Multimodal Unit: element-wise gating decides which modality
    to trust per hidden dimension. More expressive than fixed-weight blending.
    """
    def __init__(self, c_out: int, pred_len: int, text_dim: int,
                 num_levels: int = 3, hidden: int = 256):
        super().__init__()
        feat = pred_len * c_out
        self.pred_len = pred_len
        self.c_out = c_out
        self.level_weights = nn.Parameter(torch.zeros(num_levels))
        self.ts_proj   = nn.Linear(feat, hidden)
        self.text_proj = nn.Linear(text_dim, hidden)
        self.gate_net  = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.Sigmoid(),
        )
        self.output_proj = nn.Linear(hidden, feat)
        # Init to near-zero residual (default to base_output)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, base_output: torch.Tensor,
                level_embs: torch.Tensor) -> torch.Tensor:
        B, L, C = base_output.shape
        feat = L * C
        w = F.softmax(self.level_weights, dim=0)
        pooled = (level_embs * w.view(1, -1, 1)).sum(dim=1)
        valid = (pooled.abs().sum(-1) > 1e-6).float().view(B, 1)

        h_ts   = self.ts_proj(base_output.reshape(B, feat))   # [B, hidden]
        h_text = self.text_proj(pooled)                        # [B, hidden]

        gate   = self.gate_net(torch.cat([h_ts, h_text], dim=-1))  # [B, hidden]
        h_fused = gate * h_ts + (1 - gate) * h_text                # [B, hidden]

        residual = self.output_proj(h_fused).reshape(B, L, C)
        return base_output + valid.unsqueeze(-1) * residual


class DirectionMagnitudeFusion(nn.Module):
    """
    Text predicts direction (tanh → [-1,1]), time-series supplies magnitude.
    Text's strength: sentiment/direction. Time-series strength: scale/patterns.
    """
    def __init__(self, c_out: int, pred_len: int, text_dim: int,
                 num_levels: int = 3, hidden: int = 256):
        super().__init__()
        feat = pred_len * c_out
        self.pred_len = pred_len
        self.c_out = c_out
        self.level_weights = nn.Parameter(torch.zeros(num_levels))
        self.dir_net = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, feat),
            nn.Tanh(),  # output in [-1, 1]
        )
        # Learnable scale initialized small so text correction starts gentle
        self.scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, base_output: torch.Tensor,
                level_embs: torch.Tensor) -> torch.Tensor:
        B, L, C = base_output.shape
        w = F.softmax(self.level_weights, dim=0)
        pooled = (level_embs * w.view(1, -1, 1)).sum(dim=1)
        valid = (pooled.abs().sum(-1) > 1e-6).float().view(B, 1, 1)

        direction = self.dir_net(pooled).view(B, L, C)  # [-1, 1]
        # adjustment = scale * direction * |base_prediction|
        adjustment = self.scale * direction * base_output.abs()
        return base_output + valid * adjustment


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------
def _fill_compat_args(args):
    args.enc_in  = 4
    args.dec_in  = 4
    args.c_out   = 4
    args.d_layers = getattr(args, "e_layers", 2)
    args.p_hidden_dims   = [128, 128]
    args.p_hidden_layers = 2
    args.output_attention = False
    args.ratio   = 0.5


def build_base_model(args):
    key = args.model_type.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model_type: {args.model_type}. "
                         f"Choose from {list(MODEL_REGISTRY)}")
    module = importlib.import_module(MODEL_REGISTRY[key])
    return module.Model(args)


def build_fusion(args, emb_dim: int):
    ftype = getattr(args, "fusion_type", "cascade_film")
    num_levels = len(LEVEL_PREFIXES)
    if ftype == "late":
        return LateFusion(
            c_out=args.c_out, pred_len=args.pred_len,
            text_dim=emb_dim, num_levels=num_levels,
            hidden=args.film_hidden,
            text_weight=getattr(args, "text_weight", 0.1),
            learnable_level_weights=getattr(args, "learnable_level_weights", False),
        )
    if ftype == "encoder_token":
        return EncoderTokenFusion(
            configs=args, text_dim=emb_dim, num_levels=num_levels,
        )
    if ftype == "gmu":
        return GMUFusion(
            c_out=args.c_out, pred_len=args.pred_len,
            text_dim=emb_dim, num_levels=num_levels,
            hidden=args.film_hidden,
        )
    if ftype == "dir_mag":
        return DirectionMagnitudeFusion(
            c_out=args.c_out, pred_len=args.pred_len,
            text_dim=emb_dim, num_levels=num_levels,
            hidden=args.film_hidden,
        )
    # default: cascade_film
    return CascadeFilmFusion(
        c_out          = args.c_out,
        pred_len       = args.pred_len,
        num_levels     = num_levels,
        text_dim       = emb_dim,
        film_hidden    = args.film_hidden,
        dropout        = args.fusion_dropout,
        bottleneck_dim = getattr(args, "bottleneck_dim", 0),
    )


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def _prepare_batch(batch, device, pred_len, label_len):
    price_x, price_y, level_embs, x_mark, y_mark = batch
    price_x    = price_x.float().to(device)
    price_y    = price_y.float().to(device)
    level_embs = level_embs.float().to(device)
    x_mark     = x_mark.float().to(device)
    y_mark     = y_mark.float().to(device)

    dec_inp = torch.zeros_like(price_y[:, -pred_len:, :])
    dec_inp = torch.cat([price_y[:, :label_len, :], dec_inp], dim=1)
    ground_truth = price_y[:, -pred_len:, :]
    return price_x, level_embs, x_mark, y_mark, dec_inp, ground_truth


def run_epoch(loader, model, fusion, args, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    fusion.train(is_train)

    mse_sum = mae_sum = close_mse_sum = close_mae_sum = cnt = 0.0
    uses_raw = getattr(fusion, "needs_raw_input", False)

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            price_x, level_embs, x_mark, y_mark, dec_inp, gt = _prepare_batch(
                batch, device, args.pred_len, args.label_len
            )
            if is_train:
                optimizer.zero_grad()

            if uses_raw:
                # EncoderTokenFusion: text is injected into encoder directly
                out = fusion(price_x, x_mark, dec_inp, y_mark, level_embs)
            else:
                base_out = model(price_x, x_mark, dec_inp, y_mark)
                out      = fusion(base_out, level_embs)

            mse  = torch.mean((out - gt) ** 2)
            mae  = torch.mean(torch.abs(out - gt))
            # close is column index 3 — evaluate only the LAST predicted day (4-week target)
            close_mse = torch.mean((out[:, -1, 3] - gt[:, -1, 3]) ** 2)
            close_mae = torch.mean(torch.abs(out[:, -1, 3] - gt[:, -1, 3]))

            loss = mse
            if is_train and hasattr(fusion, "alpha") and getattr(args, "alpha_penalty", 0.0) > 0:
                alpha_val = torch.sigmoid(fusion.alpha)
                loss = loss + args.alpha_penalty * torch.mean(alpha_val ** 2)

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(fusion.parameters()), 1.0
                )
                optimizer.step()

            bs = out.shape[0]
            mse_sum       += mse.item()       * bs
            mae_sum       += mae.item()       * bs
            close_mse_sum += close_mse.item() * bs
            close_mae_sum += close_mae.item() * bs
            cnt           += bs

    n = max(cnt, 1)
    return mse_sum/n, mae_sum/n, close_mse_sum/n, close_mae_sum/n


# ---------------------------------------------------------------------------
# Result logging
# ---------------------------------------------------------------------------
RESULT_FIELDS = [
    "model", "fusion", "tickers", "seed",
    "stop_epoch", "val_mse", "test_mse", "close_mse", "close_mae",
]

def append_result(args, stop_epoch, val_mse, test_mse, close_mse, close_mae):
    path = Path(args.results_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "model":      args.model_type,
        "fusion":     getattr(args, "fusion_type", "cascade_film"),
        "tickers":    args.num_tickers,
        "seed":       args.seed,
        "stop_epoch": stop_epoch,
        "val_mse":    round(val_mse, 6),
        "test_mse":   round(test_mse, 6),
        "close_mse":  round(close_mse, 6),
        "close_mae":  round(close_mae, 6),
    }
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"[Result] saved to {path}")


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------
def run_single(args, tickers: list):
    _fill_compat_args(args)
    set_seed(args.seed)
    os.makedirs(args.logdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Job] model={args.model_type} seed={args.seed} "
          f"tickers={len(tickers)} device={device}")

    # Datasets
    make_ds = lambda flag: LGAIDataset(
        train_path=args.train_path,
        emb_path=args.emb_path,
        flag=flag,
        tickers=tickers,
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        test_path=args.test_path,
        forecast_gap=getattr(args, "forecast_gap", 0),
        ema_span=getattr(args, "ema_span", 0),
    )
    train_ds = make_ds("train")
    val_ds   = make_ds("val")
    test_ds  = make_ds("test")

    make_dl = lambda ds, shuffle: DataLoader(
        ds, batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.num_workers, drop_last=shuffle, pin_memory=True,
    )
    train_dl = make_dl(train_ds, True)
    val_dl   = make_dl(val_ds,   False)
    test_dl  = make_dl(test_ds,  False)

    emb_dim = train_ds.emb_dim

    # Models
    model  = build_base_model(args).to(device)
    fusion = build_fusion(args, emb_dim).to(device)

    params = list(model.parameters()) + list(fusion.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epoch, eta_min=1e-5
    )

    best_val = float("inf")
    best_test = (float("inf"), float("inf"), float("inf"))
    patience_cnt = 0
    stop_epoch = args.num_epoch

    for epoch in range(1, args.num_epoch + 1):
        tr_mse, _, _, _ = run_epoch(train_dl, model, fusion, args, device, optimizer)
        scheduler.step()
        val_mse, _, _, _            = run_epoch(val_dl,  model, fusion, args, device)
        te_mse, _, te_cmse, te_cmae = run_epoch(test_dl, model, fusion, args, device)

        alpha_str = ""
        if hasattr(fusion, "alpha"):
            alpha_str = f"  alpha={torch.sigmoid(fusion.alpha).detach().cpu().numpy().round(3)}"
        print(f"  [Ep{epoch:02d}] train_mse={tr_mse:.5f}  "
              f"val={val_mse:.5f}  test={te_mse:.5f}  "
              f"close_mse={te_cmse:.5f}  close_mae={te_cmae:.5f}{alpha_str}")

        if val_mse < best_val:
            best_val   = val_mse
            best_test  = (te_mse, te_cmse, te_cmae)
            patience_cnt = 0
            stop_epoch = epoch
            torch.save({
                "model": model.state_dict(),
                "fusion": fusion.state_dict(),
            }, os.path.join(args.logdir, "best_model.pt"))
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"  [EarlyStop] epoch={epoch}")
                break

    print(f"\n[Best] val_mse={best_val:.5f}  "
          f"test_mse={best_test[0]:.5f}  "
          f"close_mse={best_test[1]:.5f}  close_mae={best_test[2]:.5f}")

    append_result(args, stop_epoch, best_val, *best_test)
    return best_val, best_test


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Cascade FiLM Fusion experiment")

    # Data
    p.add_argument("--train_path", default="/workspace/Dataset/train.parquet")
    p.add_argument("--test_path",  default="/workspace/Dataset/test.parquet")
    p.add_argument("--emb_path",   default="/workspace/Dataset/linq_textemb.parquet")
    p.add_argument("--tickers",    type=_parse_str_list, default=None,
                   help="Comma-separated ticker list. If None, uses top --num_tickers.")
    p.add_argument("--num_tickers", type=int, default=10,
                   help="Number of tickers to use when --tickers not set.")
    p.add_argument("--ticker_offset", type=int, default=0,
                   help="Skip first N tickers (for staged 10->100 runs).")

    # Sequence
    p.add_argument("--seq_len",   type=int, default=64)
    p.add_argument("--label_len", type=int, default=16)
    p.add_argument("--pred_len",  type=int, default=20)

    # Model
    p.add_argument("--model_type", default="PatchTST",
                   choices=["PatchTST", "FiLM", "patchtst", "film"])
    p.add_argument("--task_name",  default="short_term_forecast")
    p.add_argument("--d_model",    type=int, default=64)
    p.add_argument("--n_heads",    type=int, default=4)
    p.add_argument("--e_layers",   type=int, default=2)
    p.add_argument("--d_ff",       type=int, default=64)
    p.add_argument("--patch_len",  type=int, default=8)
    p.add_argument("--dropout",    type=float, default=0.1)
    p.add_argument("--activation", default="relu")
    p.add_argument("--factor",     type=int, default=1)
    p.add_argument("--embed",      default="timeF")
    p.add_argument("--freq",       default="d")
    p.add_argument("--distil",     action="store_true", default=True)
    p.add_argument("--moving_avg", type=int, default=25)
    p.add_argument("--decomp_method",        default="moving_avg")
    p.add_argument("--channel_independence", type=int, default=1)
    p.add_argument("--use_norm",             type=int, default=1)
    p.add_argument("--down_sampling_layers", type=int, default=0)
    p.add_argument("--down_sampling_window", type=int, default=1)
    p.add_argument("--down_sampling_method", default=None)

    # Fusion
    p.add_argument("--fusion_type",  default="cascade_film",
                   choices=["cascade_film", "late", "encoder_token", "gmu", "dir_mag"],
                   help="Fusion method")
    p.add_argument("--fusion_types", type=_parse_str_list, default=None,
                   help="Grid-only: comma-separated list, e.g. cascade_film,late")
    p.add_argument("--text_weight",  type=float, default=0.1,
                   help="Text blend weight for late fusion")
    p.add_argument("--film_hidden",    type=int,   default=256,
                   help="Hidden size in FiLM gamma/beta MLP")
    p.add_argument("--fusion_dropout", type=float, default=0.1)
    p.add_argument("--alpha_penalty",  type=float, default=0.0,
                   help="L2 penalty on alpha to prevent overfitting")
    p.add_argument("--bottleneck_dim", type=int,   default=0,
                   help="Bottleneck dim to compress text emb before FiLM (0=disabled)")
    p.add_argument("--forecast_gap",  type=int,   default=0,
                   help="Gap days between input end and prediction target (0=contiguous)")
    p.add_argument("--ema_span",      type=int,   default=0,
                   help="EMA span for text window aggregation (0=last day only)")
    p.add_argument("--learnable_level_weights", action="store_true", default=False,
                   help="Use learnable softmax weights for Macro/Sector/Target levels")

    # Training
    p.add_argument("--num_epoch",   type=int,   default=30)
    p.add_argument("--patience",    type=int,   default=5)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--seed",        type=int,   default=7)

    # Logging
    p.add_argument("--logdir",      default="logs/cascade_film")
    p.add_argument("--results_csv", default="logs/cascade_film_results.csv")

    # Grid
    p.add_argument("--run_grid",    action="store_true")
    p.add_argument("--model_types", type=_parse_str_list, default=None)
    p.add_argument("--seeds",       type=_parse_int_list, default=None)
    p.add_argument("--ticker_counts", type=_parse_int_list, default=None,
                   help="e.g. 10,50,100 to run progressively more tickers")

    args = p.parse_args()
    return args


# ---------------------------------------------------------------------------
# Ticker selection
# ---------------------------------------------------------------------------
def _select_tickers(args) -> list:
    if args.tickers:
        return args.tickers
    print("[Tickers] Loading ticker list from train_path ...")
    # Read only ticker column (fast)
    df = pd.read_parquet(args.train_path, columns=["ticker"])
    all_tickers = sorted(df["ticker"].unique().tolist())
    end = args.ticker_offset + args.num_tickers
    selected = all_tickers[args.ticker_offset:end]
    print(f"[Tickers] Using {len(selected)} tickers: {selected[:5]}...")
    return selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    _fill_compat_args(args)

    if args.run_grid:
        model_types   = args.model_types   or [args.model_type]
        fusion_types  = args.fusion_types  or [args.fusion_type]
        seeds         = args.seeds         or [args.seed]
        ticker_counts = args.ticker_counts or [args.num_tickers]

        # Load all tickers once
        print("[Grid] Loading full ticker list ...")
        df_t = pd.read_parquet(args.train_path, columns=["ticker"])
        all_tickers = sorted(df_t["ticker"].unique().tolist())
        print(f"[Grid] Total unique tickers: {len(all_tickers)}")

        jobs = []
        for n_t in ticker_counts:
            tickers_for_n = all_tickers if n_t <= 0 else all_tickers[:n_t]
            for ft in fusion_types:
                for m in model_types:
                    for s in seeds:
                        job = copy.deepcopy(args)
                        job.model_type  = m
                        job.fusion_type = ft
                        job.seed        = s
                        job.num_tickers = len(tickers_for_n)
                        job.logdir = os.path.join(
                            args.logdir, f"{ft}_{m}_n{len(tickers_for_n)}_seed{s}"
                        )
                        jobs.append((job, tickers_for_n))

        print(f"[Grid] {len(jobs)} total jobs")
        for idx, (job, tickers) in enumerate(jobs, 1):
            print(f"\n{'='*60}")
            print(f"[Grid {idx}/{len(jobs)}] fusion={job.fusion_type} "
                  f"model={job.model_type} tickers={len(tickers)} seed={job.seed}")
            print(f"{'='*60}")
            run_single(job, tickers)
    else:
        tickers = _select_tickers(args)
        run_single(args, tickers)

