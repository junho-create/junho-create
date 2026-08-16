"""
High-Performance Text-to-Signal Pipeline with Consensus Clustering
"""
import argparse
import csv
import gc
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Path Setup
# ---------------------------------------------------------------------------
FORECASTING_DIR = Path(__file__).resolve().parent
REPO_ROOT = FORECASTING_DIR.parent
if str(FORECASTING_DIR) not in sys.path: sys.path.insert(0, str(FORECASTING_DIR))
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Utils & Constants
# ---------------------------------------------------------------------------
LEVEL_PREFIXES = [
    ("macro", ["macro_category"]),
    ("sector", ["sector_category"]),
    ("related", ["relatedCompany_category"]),
    ("target", ["targetCompany_category", "filing_", "lseg_news"]),
]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _get_level_id_cols(df_cols):
    result = []
    for level_name, prefixes in LEVEL_PREFIXES:
        matched = []
        for pfx in prefixes:
            matched += [c for c in df_cols if c.startswith(pfx)]
        result.append((level_name, matched))
    return result

# ---------------------------------------------------------------------------
# Core: Embedding Store
# ---------------------------------------------------------------------------
class EmbeddingStore:
    def __init__(self, emb_path: str, valid_ids=None):
        print(f"[EmbeddingStore] Loading {emb_path} ...", flush=True)
        if emb_path.endswith('.csv'):
            df = pd.read_csv(emb_path)
        else:
            df = pd.read_parquet(emb_path)
        if valid_ids is not None:
            valid_set = set(valid_ids)
            df = df[df["text_id"].isin(valid_set)].reset_index(drop=True)
            import gc; gc.collect()
        
        emb_cols = [c for c in df.columns if c != "text_id"]
        self.emb_dim = len(emb_cols)
        
        ids = df["text_id"].values.copy()
        mat = df[emb_cols].values.astype(np.float32)
            
        self._idx = {tid: i for i, tid in enumerate(ids)}
        self._mat = mat
        del df
        import gc; gc.collect()
        print(f"[EmbeddingStore] Loaded {len(ids):,} embeddings, dim={self.emb_dim}")

    def lookup_mean(self, ids) -> np.ndarray:
        valid_rows = [self._idx[tid] for tid in ids if tid in self._idx and pd.notna(tid)]
        if not valid_rows:
            return np.zeros(self.emb_dim, dtype=np.float32)
        return self._mat[valid_rows].mean(axis=0)

# ---------------------------------------------------------------------------
# Consensus Clustering
# ---------------------------------------------------------------------------
def compute_consensus_clusters(df: pd.DataFrame, emb_dir: str, train_end: str, n_clusters: int):
    """
    Computes a consensus clustering from return correlations and multiple text embeddings.
    Returns: ticker_to_cluster (dict)
    """
    print(f"\n=== Phase 0: Consensus Clustering (n_clusters={n_clusters}) ===")
    t0 = time.time()
    
    df_train = df[df["date"] < train_end]
    tickers = sorted(df["ticker"].unique())
    n_tickers = len(tickers)
    tidx = {t: i for i, t in enumerate(tickers)}
    
    matrices = []
    
    # 1. Price Return Correlation
    print("  [1/N] Computing price return correlation matrix...")
    ret_df = df_train.pivot(index="date", columns="ticker", values="close").pct_change().fillna(0)
    # Ensure all tickers are present
    for t in tickers:
        if t not in ret_df.columns:
            ret_df[t] = 0.0
    ret_corr = ret_df[tickers].corr().fillna(0).values
    matrices.append(ret_corr)
    
    # 2. Text Embedding Similarities
    parquet_files = list(Path(emb_dir).glob("*_textemb.parquet"))
    target_cols = [c for c in df_train.columns if c.startswith("targetCompany_category") or c.startswith("lseg_news")]
    
    valid_ids_train = set(df_train[target_cols].values.flatten().tolist())
    
    for i, p_file in enumerate(parquet_files, start=2):
        print(f"  [{i}/N] Computing similarity for {p_file.name}...")
        store = EmbeddingStore(str(p_file), valid_ids=valid_ids_train)
        
        # Calculate mean train embedding per ticker
        ticker_embs = np.zeros((n_tickers, store.emb_dim), dtype=np.float32)
        for t in tickers:
            t_df = df_train[df_train["ticker"] == t]
            if t_df.empty or not target_cols:
                continue
            all_ids = t_df[target_cols].values.flatten().tolist()
            ticker_embs[tidx[t]] = store.lookup_mean(all_ids)
            
        sim = cosine_similarity(ticker_embs)
        matrices.append(sim)
        del store
        del ticker_embs
        gc.collect()
        
    # 3. Apply clustering on each matrix and build consensus
    consensus_matrix = np.zeros((n_tickers, n_tickers), dtype=np.float32)
    
    for mat in matrices:
        # Distance = 1 - sim
        # Clip to avoid negative distances due to numerical errors
        dist = np.clip(1.0 - mat, 0, 2)
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage="average")
        labels = clusterer.fit_predict(dist)
        
        # Add to consensus: +1 if same cluster
        same_cluster = (labels[:, None] == labels[None, :]).astype(np.float32)
        consensus_matrix += same_cluster
        
    consensus_matrix /= len(matrices)
    
    # 4. Final clustering
    print("  [Final] Performing final clustering on Consensus Matrix...")
    final_dist = np.clip(1.0 - consensus_matrix, 0, 1)
    final_clusterer = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage="average")
    final_labels = final_clusterer.fit_predict(final_dist)
    
    ticker_to_cluster = {t: int(final_labels[tidx[t]]) for t in tickers}
    cluster_sizes = np.bincount(final_labels, minlength=n_clusters)
    
    print(f"[Consensus Clusters] Done in {time.time()-t0:.1f}s")
    print(f"  Cluster Sizes: {cluster_sizes.tolist()}")
    
    return ticker_to_cluster

def precompute_cluster_embeddings(df: pd.DataFrame, store: EmbeddingStore, ticker_to_cluster: dict):
    """
    Computes a daily mean target/news embedding for each consensus cluster.
    Returns: cluster_emb_dict[(date, cluster_id)] = embedding
    """
    print("[Precompute] Building dynamic cluster sector embeddings...", flush=True)
    t0 = time.time()
    
    target_cols = [c for c in df.columns if c.startswith("targetCompany_category") or c.startswith("lseg_news")]
    
    cluster_emb_dict = {}
    df_c = df.copy()
    df_c["cluster_id"] = df_c["ticker"].map(ticker_to_cluster)
    
    for date, day_g in df_c.groupby("date"):
        for cid, cluster_g in day_g.groupby("cluster_id"):
            all_ids = cluster_g[target_cols].values.flatten().tolist()
            emb = store.lookup_mean(all_ids)
            cluster_emb_dict[(date, cid)] = emb
            
    print(f"[Precompute] Done in {time.time()-t0:.1f}s. Computed {len(cluster_emb_dict)} day-cluster pairs.")
    return cluster_emb_dict

# ---------------------------------------------------------------------------
# High-Performance Dataset
# ---------------------------------------------------------------------------
class ConsensusDataset(Dataset):
    def __init__(self, df: pd.DataFrame, store: EmbeddingStore, ticker_to_cluster: dict, cluster_emb_dict: dict, 
                 tickers: list, seq_len: int, pred_len: int, label_len: int, ema_span: int, is_train: bool):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        
        level_cols = _get_level_id_cols(list(df.columns))
        sector_idx = next(i for i, (lname, _) in enumerate(level_cols) if lname == "sector")
        
        df_train = df[df["date"] < "2022-01-01"]
        self.scaler = StandardScaler()
        self.scaler.fit(df_train[["open", "high", "low", "close"]].values)
        
        all_px, all_py, all_le, all_xm, all_ym = [], [], [], [], []
        
        for ticker in tickers:
            df_t = df[df["ticker"] == ticker].reset_index(drop=True)
            if len(df_t) < seq_len + pred_len + 10:
                continue
                
            dates = df_t["date"]
            n = len(df_t)
            cluster_id = ticker_to_cluster[ticker]
            
            if is_train:
                b1, b2 = 0, int((dates < "2022-01-01").sum())
            else:
                b1 = max(0, int((dates < "2023-01-01").sum()) - seq_len)
                b2 = n
                
            if b2 - b1 <= seq_len + pred_len:
                continue
                
            price_norm = self.scaler.transform(df_t[["open", "high", "low", "close"]].values).astype(np.float32)
            stamp = np.stack([dates.dt.month, dates.dt.day, dates.dt.weekday], axis=1).astype(np.float32)
            
            le_arr = np.zeros((n, len(level_cols), store.emb_dim), dtype=np.float32)
            for row_i in range(n):
                d = dates.iloc[row_i]
                for li, (lname, cols) in enumerate(level_cols):
                    if lname == "sector":
                        le_arr[row_i, li] = cluster_emb_dict.get((d, cluster_id), np.zeros(store.emb_dim))
                    else:
                        ids = df_t.iloc[row_i][cols].tolist() if cols else []
                        le_arr[row_i, li] = store.lookup_mean(ids)
                        
            if ema_span > 0:
                alpha = 2.0 / (ema_span + 1)
                for t in range(1, n):
                    le_arr[t] = alpha * le_arr[t] + (1 - alpha) * le_arr[t-1]
            
            for idx in range(b1, b2 - seq_len - pred_len + 1):
                s = idx
                se = s + seq_len
                re = se - label_len
                ree = se + pred_len
                
                all_px.append(price_norm[s:se].copy())
                all_py.append(price_norm[re:ree].copy())
                all_le.append(le_arr[se - 1].copy())
                all_xm.append(stamp[s:se].copy())
                all_ym.append(stamp[re:ree].copy())
                
        self.px = np.stack(all_px)
        self.py = np.stack(all_py)
        self.le = np.stack(all_le)
        self.xm = np.stack(all_xm)
        self.ym = np.stack(all_ym)
        print(f"[Dataset] Train={is_train} | Samples: {len(self.px):,} | Shape: {self.px.shape}")

    def __len__(self):
        return len(self.px)

    def __getitem__(self, i):
        return (i, torch.from_numpy(self.px[i]), torch.from_numpy(self.py[i]), torch.from_numpy(self.le[i]), 
                torch.from_numpy(self.xm[i]), torch.from_numpy(self.ym[i]))

# ---------------------------------------------------------------------------
# Models (TextClassifier + V7Model)
# ---------------------------------------------------------------------------
class TextClassifier(nn.Module):
    def __init__(self, text_dim, level_indices, hidden=512, seq_len=64, num_layers=4, nhead=8):
        super().__init__()
        self.level_indices = level_indices
        n = len(level_indices)
        self.proj = nn.Linear(text_dim, hidden)
        self.level_embed = nn.Embedding(n, hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=nhead, dim_feedforward=hidden * 2, dropout=0.1, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dir_head = nn.Linear(hidden, 1)
        self.temporal_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, seq_len)
        )

    def forward(self, level_embs):
        active = level_embs[:, self.level_indices, :]
        B, n, _ = active.shape
        x = self.proj(active)
        x = x + self.level_embed(torch.arange(n, device=x.device)).unsqueeze(0)
        x = self.encoder(x)
        pooled = x.mean(dim=1)
        return self.dir_head(pooled), torch.sigmoid(self.temporal_head(pooled))

class V7Model(nn.Module):
    def __init__(self, configs, text_clf, joint=False):
        super().__init__()
        self.text_clf = text_clf
        self.joint = joint
        if not joint:
            for p in self.text_clf.parameters(): p.requires_grad = False
            self.text_clf.eval()

        configs.enc_in = 5
        configs.c_out = 5
        configs.task_name = 'short_term_forecast'
        configs.output_attention = False
        configs.factor = 1
        configs.activation = "relu"
        from forecasting_task.models import PatchTST
        self.backbone = PatchTST.Model(configs, patch_len=configs.patch_len, stride=configs.stride)

    def forward(self, price_x, x_mark, dec_inp, y_mark, level_embs):
        B, L, _ = price_x.shape
        with torch.enable_grad() if self.joint else torch.no_grad():
            dir_logit, p_temporal = self.text_clf(level_embs)
            
        active = level_embs[:, self.text_clf.level_indices, :]
        has_text = (active.abs().sum(dim=(1, 2)) > 1e-6).float().view(B, 1, 1)
        p_seq = p_temporal.unsqueeze(-1) * has_text + 0.5 * (1 - has_text)
        
        x_enc = torch.cat([price_x, p_seq], dim=-1)
        dec_inp_5 = torch.cat([dec_inp, torch.zeros(B, dec_inp.shape[1], 1, device=dec_inp.device)], dim=-1)
        out = self.backbone(x_enc, x_mark, dec_inp_5, y_mark)
        return out[:, :, :4], out[:, :, 4:5], torch.sigmoid(dir_logit)

# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------
def _prep_batch(batch, device, pred_len, label_len):
    idx = batch[0]
    px, py, le, xm, ym = [b.to(device) for b in batch[1:]]
    dec_inp = torch.zeros_like(py[:, -pred_len:, :]).float()
    dec_inp = torch.cat([py[:, :label_len, :], dec_inp], dim=1).float().to(device)
    gt = py[:, -pred_len:, :]
    return idx, px, le, xm, ym, dec_inp, gt

def train_phase1(loader, text_dim, level_indices, args, device):
    print("\n=== Phase 1: Pretraining Text Classifier ===")
    model = TextClassifier(
        text_dim, level_indices, seq_len=args.seq_len, 
        hidden=args.text_hidden, num_layers=args.text_layers, nhead=args.text_heads
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    for ep in range(1, args.text_epochs + 1):
        model.train()
        t_acc, cnt = 0.0, 0
        for batch in loader:
            idx, px, le, xm, ym, dec_inp, gt = _prep_batch(batch, device, args.pred_len, args.label_len)
            target = (gt[:, -1, 3] > px[:, -1, 3]).float().unsqueeze(1)
            active = le[:, model.level_indices, :]
            has_text = (active.abs().sum(dim=(1, 2)) > 1e-6).float().unsqueeze(1)
            
            opt.zero_grad()
            logits, _ = model(le)
            loss = (criterion(logits, target) * has_text).sum() / has_text.sum().clamp_min(1.0)
            loss.backward()
            opt.step()
            
            preds = (torch.sigmoid(logits) > 0.5).float()
            t_acc += ((preds == target).float() * has_text).sum().item()
            cnt += has_text.sum().item()
            
        print(f"  [Ep{ep:02d}] Train Acc: {t_acc/max(cnt, 1):.4f}")
    return model

def run_epoch(loader, model, args, device, optimizer=None, global_weights=None, global_errors=None):
    is_train = optimizer is not None
    model.train(is_train)
    mse_sum, cnt = 0.0, 0
    all_pred, all_true = [], []
    
    with torch.enable_grad() if is_train else torch.no_grad():
        for batch in loader:
            idx, px, le, xm, ym, dec_inp, gt = _prep_batch(batch, device, args.pred_len, args.label_len)
            if is_train: optimizer.zero_grad()
            
            out, log_var, p_dir = model(px, xm, dec_inp, ym, le)
            
            # Clamp log_var more strictly to prevent explosion
            log_var = torch.clamp(log_var, min=-5.0, max=5.0)
            
            mse = torch.mean((out - gt)**2)
            
            pred_diff = out[:, -1, 3] - px[:, -1, 3]
            true_diff = gt[:, -1, 3] - px[:, -1, 3]
            
            if is_train and global_weights is not None:
                # Per-sample loss calculation
                sample_dir_penalty = torch.clamp(torch.relu(-1.0 * pred_diff * true_diff), max=1.0)
                
                # Standard MSE Loss per sample
                sample_mse_loss = torch.mean((out - gt)**2, dim=(1, 2))
                
                sample_loss = sample_mse_loss + args.dir_weight * sample_dir_penalty
                
                if args.joint:
                    target = (gt[:, -1, 3] > px[:, -1, 3]).float().unsqueeze(1)
                    has_text = (le[:, model.text_clf.level_indices, :].abs().sum(dim=(1, 2)) > 1e-6).float().view(-1)
                    sample_dir_loss = F.binary_cross_entropy(p_dir, target, reduction='none').view(-1)
                    sample_loss = sample_loss + args.aux_weight * sample_dir_loss * has_text
                
                batch_weights = global_weights[idx].to(device)
                loss = (sample_loss * batch_weights).mean()
                
                # Record error in global tracker (EMA of Directional Error)
                if global_errors is not None:
                    pred_diff_np = pred_diff.detach().cpu().numpy()
                    true_diff_np = true_diff.detach().cpu().numpy()
                    # 1.0 if direction is wrong, 0.0 if direction is correct
                    dir_wrong = (pred_diff_np * true_diff_np <= 0).astype(np.float32)
                    
                    idx_np = idx.cpu().numpy() if isinstance(idx, torch.Tensor) else idx.numpy()
                    global_errors[idx_np] = 0.8 * global_errors[idx_np] + 0.2 * dir_wrong
                
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            else:
                dir_penalty = torch.clamp(torch.relu(-1.0 * pred_diff * true_diff), max=1.0).mean()
                mse_loss = torch.mean((out - gt)**2)
                loss = mse_loss + args.dir_weight * dir_penalty
                
                if is_train:
                    if args.joint:
                        target = (gt[:, -1, 3] > px[:, -1, 3]).float().unsqueeze(1)
                        has_text = (le[:, model.text_clf.level_indices, :].abs().sum(dim=(1, 2)) > 1e-6).float().unsqueeze(1)
                        dir_loss = F.binary_cross_entropy(p_dir, target, reduction='none')
                        loss += args.aux_weight * (dir_loss * has_text).sum() / has_text.sum().clamp_min(1.0)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
            bs = out.shape[0]
            mse_sum += mse.item() * bs
            cnt += bs
            
            if not is_train:
                all_pred.extend((out[:, -1, 3] - px[:, -1, 3]).detach().cpu().numpy())
                all_true.extend((gt[:, -1, 3] - px[:, -1, 3]).detach().cpu().numpy())
                
    res = {"mse": mse_sum / max(cnt, 1)}
    if not is_train and all_pred:
        p, t = np.array(all_pred), np.array(all_true)
        res["da"] = np.mean((p > 0) == (t > 0))
        res["rank_ic"], _ = spearmanr(p, t)
    return res

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="/workspace/Dataset/test.parquet")
    p.add_argument("--emb_dir", default="/root/LG-AI/data/multimodal_dataset_dsl")
    p.add_argument("--primary_emb", default="gemini_textemb.parquet")
    p.add_argument("--n_clusters", type=int, default=11)
    
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--label_len", type=int, default=16)
    p.add_argument("--pred_len", type=int, default=20)
    p.add_argument("--ema_span", type=int, default=5)
    
    # PatchTST config
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--e_layers", type=int, default=2)
    p.add_argument("--d_ff", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--patch_len", type=int, default=8)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--moving_avg", type=int, default=25)
    
    # Text Classifier config
    p.add_argument("--text_hidden", type=int, default=256)
    p.add_argument("--text_layers", type=int, default=2)
    p.add_argument("--text_heads", type=int, default=4)
    
    p.add_argument("--text_epochs", type=int, default=10)
    p.add_argument("--num_epoch", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-4) # Lowered LR for stability
    p.add_argument("--batch_size", type=int, default=64)
    
    p.add_argument("--joint", action="store_true", default=True) # Joint training ON by default
    p.add_argument("--aux_weight", type=float, default=0.5)      # Increased aux weight
    p.add_argument("--dir_weight", type=float, default=0.1)      # Lowered directional penalty weight
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading raw data...", flush=True)
    if args.data_path.endswith('.csv'):
        df = pd.read_csv(args.data_path)
    else:
        df = pd.read_parquet(args.data_path)
    df["date"] = pd.to_datetime(df["date"])
    tickers = sorted(df["ticker"].unique().tolist())
    
    ticker_to_cluster = compute_consensus_clusters(df, args.emb_dir, "2022-01-01", args.n_clusters)
    
    target_cols = [c for c in df.columns if c.startswith("targetCompany_category") or c.startswith("lseg_news")]
    valid_ids_all = set(df[target_cols].values.flatten().tolist())
    primary_store = EmbeddingStore(os.path.join(args.emb_dir, args.primary_emb), valid_ids=valid_ids_all)
    cluster_emb_dict = precompute_cluster_embeddings(df, primary_store, ticker_to_cluster)
    
    train_ds = ConsensusDataset(df, primary_store, ticker_to_cluster, cluster_emb_dict, tickers, args.seq_len, args.pred_len, args.label_len, args.ema_span, is_train=True)
    test_ds = ConsensusDataset(df, primary_store, ticker_to_cluster, cluster_emb_dict, tickers, args.seq_len, args.pred_len, args.label_len, args.ema_span, is_train=False)
    
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    
    level_names = [lv[0] for lv in LEVEL_PREFIXES]
    level_indices = [level_names.index(l) for l in ["macro", "sector", "related", "target"]]
    
    if args.joint:
        text_clf = TextClassifier(
            primary_store.emb_dim, level_indices, seq_len=args.seq_len,
            hidden=args.text_hidden, num_layers=args.text_layers, nhead=args.text_heads
        ).to(device)
        model = V7Model(args, text_clf, joint=True).to(device)
    else:
        text_clf = train_phase1(train_dl, primary_store.emb_dim, level_indices, args, device)
        model = V7Model(args, text_clf, joint=False).to(device)
        
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    print(f"\n=== Phase 2: Training Final Model ===")
    
    n_train = len(train_ds)
    global_weights = torch.ones(n_train, dtype=torch.float32)
    global_errors = np.zeros(n_train, dtype=np.float32)
    
    # Cartography down-weight scale parameter
    # With directional error rate (0.0 ~ 1.0), scale 20.0 means:
    # Error 1.0 (always wrong) -> weight 1/21 (approx 0.04)
    # Error 0.5 (random noise) -> weight 1/11 (approx 0.09)
    # Error 0.0 (always right) -> weight 1.0
    cartography_scale = 20.0
    
    for ep in range(1, args.num_epoch + 1):
        tr = run_epoch(train_dl, model, args, device, opt, global_weights=global_weights, global_errors=global_errors)
        te = run_epoch(test_dl, model, args, device)
        
        # Soft Down-weighting: weights become inversely proportional to tracked errors
        new_weights = 1.0 / (1.0 + cartography_scale * global_errors)
        # Normalize so mean weight is 1.0 to preserve learning rate scale
        new_weights = new_weights / (np.mean(new_weights) + 1e-8)
        global_weights = torch.from_numpy(new_weights).float()
        
        print(f"  [Ep{ep:02d}] Train MSE: {tr['mse']:.5f} | Test MSE: {te['mse']:.5f} | Test DA: {te.get('da', 0):.4f} | Test RankIC: {te.get('rank_ic', 0):.4f}")
        
    print("\n[Done] Pipeline executed successfully.")

if __name__ == "__main__":
    main()
