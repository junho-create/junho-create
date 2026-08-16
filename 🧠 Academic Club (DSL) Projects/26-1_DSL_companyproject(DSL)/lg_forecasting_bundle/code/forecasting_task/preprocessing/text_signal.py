"""
Step2-4 alternative: a supervised pretrained text-direction signal, porting the
`TextClassifier` idea from `run_maxDAnewText2Signal.py`'s V7Model. The existing
Step1 text factors (`text_pooling.build_text_pca_factors`) and the Step3 sector
pooled-embedding factor compress each level's embedding via *unsupervised* PCA --
that keeps the axes of maximum variance, which need not correlate with returns
at all. Here a small transformer is trained (BCE, train-period rows only) to
predict horizon-ahead return direction directly from the four level embeddings
(macro/sector/related/target), then scores every row -- a task-aware, supervised
compression instead of a variance-based one.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .embedding_store import EmbeddingStore
from .levels import get_level_id_cols
from .text_pooling import pool_level


class _TextDirClassifier(nn.Module):
    def __init__(self, emb_dim, n_levels, hidden=128, num_layers=2, nhead=4, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(emb_dim, hidden)
        self.level_embed = nn.Embedding(n_levels, hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=nhead, dim_feedforward=hidden * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(hidden, 1)

    def forward(self, level_embs):
        x = self.proj(level_embs)
        x = x + self.level_embed(torch.arange(level_embs.shape[1], device=x.device)).unsqueeze(0)
        x = self.encoder(x)
        pooled = x.mean(dim=1)
        return self.head(pooled).squeeze(-1)


def build_text_signal_factors(df: pd.DataFrame, emb_path: str, horizon: int,
                               train_mask: np.ndarray, hidden: int = 128,
                               num_layers: int = 2, nhead: int = 4,
                               epochs: int = 15, lr: float = 1e-3,
                               batch_size: int = 512, seed: int = 0) -> pd.DataFrame:
    """
    Returns a single-column DataFrame `textsignal_dir_prob`: the sigmoid
    probability that the horizon-ahead return is positive, predicted for every
    row of `df` by a classifier trained only on train_mask rows with a known
    (non-NaN) forward return -- so test/valid rows never leak into training,
    they're only ever scored by a fixed, already-trained model.
    """
    torch.manual_seed(seed)
    store = EmbeddingStore.get(emb_path)
    level_cols = get_level_id_cols(list(df.columns))
    n_levels = len(level_cols)

    pooled_per_level = [pool_level(df, cols, store) for _, cols in level_cols]
    level_embs = np.stack(pooled_per_level, axis=1)  # [N, n_levels, emb_dim]
    has_text = np.stack(
        [(np.abs(p).sum(axis=1) > 1e-6) for p in pooled_per_level], axis=1
    )
    row_has_any_text = has_text.any(axis=1)

    fwd_logret = df.groupby("ticker", sort=False)["close"].transform(
        lambda s: np.log(s.shift(-horizon) / s)
    ).to_numpy()
    label = (fwd_logret > 0).astype(np.float32)
    valid_label = ~np.isnan(fwd_logret)

    train_idx = np.where(train_mask & valid_label & row_has_any_text)[0]
    if len(train_idx) < 100:
        print(f"[TextSignal] too few labeled train rows ({len(train_idx)}) -- skipping, neutral 0.5")
        return pd.DataFrame({"textsignal_dir_prob": np.full(len(df), 0.5, dtype=np.float32)}, index=df.index)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _TextDirClassifier(level_embs.shape[2], n_levels, hidden=hidden,
                                num_layers=num_layers, nhead=nhead).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    x_train = torch.from_numpy(level_embs[train_idx]).to(device)
    y_train = torch.from_numpy(label[train_idx]).to(device)
    n = len(train_idx)

    model.train()
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n, device=device)
        tot_loss, tot_correct, cnt = 0.0, 0, 0
        for s in range(0, n, batch_size):
            bidx = perm[s:s + batch_size]
            xb, yb = x_train[bidx], y_train[bidx]
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(bidx)
            tot_correct += ((logits > 0).float() == yb).sum().item()
            cnt += len(bidx)
        print(f"[TextSignal] epoch {ep:02d} train_loss={tot_loss / cnt:.4f} train_acc={tot_correct / cnt:.4f}")

    model.eval()
    probs = np.full(len(df), 0.5, dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(df), 4096):
            e = min(s + 4096, len(df))
            xb = torch.from_numpy(level_embs[s:e]).to(device)
            p = torch.sigmoid(model(xb)).cpu().numpy()
            mask = row_has_any_text[s:e]
            probs[s:e][mask] = p[mask]
    return pd.DataFrame({"textsignal_dir_prob": probs}, index=df.index)
