"""
TabPFN direction-classifier baseline on the existing tabular factor panel.

TabPFN is one of the four pretrained models explicitly allowed by the
competition rules. Unlike Chronos/Moirai/TimesFM (built for raw numeric
time series), TabPFN is a tabular foundation model -- and this repo's
panel (`forecasting_task/preprocessing/panel.py`) already reduces each
(ticker, date) row to a flat feature vector (price factors + text-PCA +
novelty/drift signals), i.e. exactly TabPFN's native input shape. This
script reuses build_panel() (same factors as the DoubleAdapt runs), takes
the *current* (most recent) value of each factor per row -- no DoubleAdapt
meta-learning, no gradient loop -- and fits TabPFNClassifier to predict
the sign of the horizon-ahead price move directly.

Usage
-----
    python -m forecasting_task.run_tabpfn --train_sample 8000 --eval_sample 4000
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from forecasting_task.preprocessing.panel import PanelConfig, build_panel  # noqa: E402


def load_reference_config(logdir: str) -> argparse.Namespace:
    with open(os.path.join(logdir, "args.yaml")) as f:
        return argparse.Namespace(**yaml.safe_load(f))


def extract_current_features(panel_df: pd.DataFrame, factor_cols, seq_len: int, window_k: int = 1) -> pd.DataFrame:
    """panel_df['feature'] is Alpha360-style: factor_num blocks of seq_len timesteps each,
    oldest->newest within a block. Column (fi+1)*seq_len-1 is factor fi's "as of now" value.
    window_k>1 also includes the k-1 preceding days per factor (as separate tabular columns)
    so TabPFN sees short-term trend/momentum instead of a single snapshot."""
    feat = panel_df["feature"]
    cols, names = [], []
    for fi, name in enumerate(factor_cols):
        block_end = (fi + 1) * seq_len  # exclusive
        for lag in range(window_k):
            cols.append(block_end - 1 - lag)
            names.append(name if lag == 0 else f"{name}_lag{lag}")
    X = feat.iloc[:, cols].to_numpy(dtype=np.float32)
    return pd.DataFrame(X, index=panel_df.index, columns=names)


def compute_direction_labels(raw_close: pd.Series, horizon: int):
    """direction=1 if close `horizon` trading days ahead (same ticker) is higher, else 0.
    -1 sentinel where the forward window falls past the end of available price history."""
    rc = raw_close.reset_index().sort_values(["instrument", "datetime"])
    rc["target_close"] = rc.groupby("instrument")["close"].shift(-horizon)
    rc = rc.set_index(["datetime", "instrument"]).sort_index()
    labeled = rc["target_close"].notna()
    direction = pd.Series(-1, index=rc.index, dtype=int)
    direction[labeled] = (rc.loc[labeled, "target_close"] > rc.loc[labeled, "close"]).astype(int)
    return direction, rc["target_close"], rc["close"]


def main():
    p = argparse.ArgumentParser(description="TabPFN direction-classifier baseline")
    p.add_argument("--reference_logdir", default=str(REPO_ROOT / "logs" / "dlinear_final_compliant"),
                   help="Reuses this run's PanelConfig (factor set, dates, panel cache) for apples-to-apples comparison.")
    p.add_argument("--train_sample", type=int, default=8000,
                   help="TabPFN's pretraining context is capped for speed/quality; subsample the ~75k-row "
                        "2019-2021 training panel down to this many rows (0 = use all, slow on CPU).")
    p.add_argument("--eval_sample", type=int, default=4000,
                   help="Subsample valid/test for a quick score estimate (0 = full set, slow on CPU).")
    p.add_argument("--n_estimators", type=int, default=8)
    p.add_argument("--window_k", type=int, default=1,
                   help="How many trailing days per factor to include as separate tabular columns "
                        "(1 = current value only, matching the DoubleAdapt factor_cols set).")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out_dir", default=str(REPO_ROOT / "logs" / "tabpfn_direction"))
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ref = load_reference_config(args.reference_logdir)

    df = pd.read_parquet(ref.data_csv) if str(ref.data_csv).endswith(".parquet") else pd.read_csv(ref.data_csv)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    cfg = PanelConfig(
        seq_len=ref.seq_len, horizon=ref.horizon, n_text_components=ref.n_text_components,
        train_start=ref.train_start, train_end=ref.train_end, primary_emb_model=ref.primary_emb_model,
        use_macro=ref.use_macro, use_sector=ref.use_sector,
        use_target_novelty=ref.use_target_novelty, use_related_novelty=ref.use_related_novelty,
        novelty_window=ref.novelty_window, novelty_quantile=ref.novelty_quantile,
        novelty_consensus_frac=ref.novelty_consensus_frac,
        sector_agreement_threshold=ref.sector_agreement_threshold,
        macro_similarity_threshold=ref.macro_similarity_threshold, macro_drift_window=ref.macro_drift_window,
        use_text_signal=getattr(ref, "use_text_signal", False),
        text_signal_hidden=getattr(ref, "text_signal_hidden", 128),
        text_signal_layers=getattr(ref, "text_signal_layers", 2),
        text_signal_heads=getattr(ref, "text_signal_heads", 4),
        text_signal_epochs=getattr(ref, "text_signal_epochs", 15),
    )
    t0 = time.time()
    panel_df, raw_close, label_mean, label_std, factor_num, seq_len, factor_cols = build_panel(
        df, ref.data_dir, cfg, cache_dir=getattr(ref, "panel_cache_dir", None)
    )
    print(f"[Panel] shape={panel_df.shape} factor_num={factor_num} seq_len={seq_len} built in {time.time()-t0:.1f}s")

    X_all = extract_current_features(panel_df, factor_cols, seq_len, window_k=args.window_k)
    direction, target_close, anchor_close = compute_direction_labels(raw_close, ref.horizon)

    dt = X_all.index.get_level_values("datetime")
    has_label = direction != -1
    train_mask = (dt >= pd.Timestamp(ref.train_start)) & (dt <= pd.Timestamp(ref.train_end)) & has_label
    valid_mask = (dt >= pd.Timestamp(ref.valid_start)) & (dt <= pd.Timestamp(ref.valid_end)) & has_label
    test_mask = (dt >= pd.Timestamp(ref.test_start)) & (dt <= pd.Timestamp(ref.test_end)) & has_label
    print(f"[Split] train={train_mask.sum()} valid={valid_mask.sum()} test={test_mask.sum()} "
          f"(rows with a real horizon-ahead label)")

    rng = np.random.default_rng(args.seed)

    def subsample(mask, n):
        idx = np.flatnonzero(mask.to_numpy())
        if n and len(idx) > n:
            idx = rng.choice(idx, size=n, replace=False)
        return idx

    train_idx = subsample(train_mask, args.train_sample)
    X_train = X_all.iloc[train_idx].to_numpy()
    y_train = direction.iloc[train_idx].to_numpy()
    print(f"[Train] using {len(train_idx)} rows (class balance: {y_train.mean():.3f} up)")

    from tabpfn import TabPFNClassifier
    clf = TabPFNClassifier(
        n_estimators=args.n_estimators, device="cpu", ignore_pretraining_limits=True,
        random_state=args.seed, show_progress_bar=True,
    )
    t0 = time.time()
    clf.fit(X_train, y_train)
    print(f"[Fit] done in {time.time()-t0:.1f}s")

    def eval_split(mask, n, name):
        idx = subsample(mask, n)
        Xs = X_all.iloc[idx].to_numpy()
        ys = direction.iloc[idx].to_numpy()
        t1 = time.time()
        preds = clf.predict(Xs)
        hit_rate = float((preds == ys).mean())
        print(f"[Eval:{name}] n={len(idx)} hit_rate={hit_rate:.4f} "
              f"(base rate up={ys.mean():.3f}, pred rate up={preds.mean():.3f}) inference {time.time()-t1:.1f}s")
        return hit_rate, len(idx)

    valid_hit, n_valid = eval_split(valid_mask, args.eval_sample, "valid_2022")
    test_hit, n_test = eval_split(test_mask, args.eval_sample, "test_2023")

    with open(os.path.join(args.out_dir, "results.txt"), "w") as f:
        f.write(f"valid_2022_hit_rate={valid_hit:.6f} (n={n_valid})\n")
        f.write(f"test_2023_hit_rate={test_hit:.6f} (n={n_test})\n")
        f.write(f"train_rows={len(train_idx)}\n")
    with open(os.path.join(args.out_dir, "factor_cols.json"), "w") as f:
        json.dump(factor_cols, f)
    print(f"[Done] wrote {args.out_dir}/results.txt")


if __name__ == "__main__":
    main()
