"""
DoubleAdapt (KDD'23) meta-learning incremental-learning runner for the FinTexTS
4-week-ahead close-price forecasting challenge.

Pipeline
--------
1. Build a per-(date, ticker) factor panel from `--data_csv` (a strict
   superset of data/train.csv covering 2019-01-01..2023-12-01) and whatever
   `<model>_textemb.parquet` files are found under `--data_dir`
   (`forecasting_task/preprocessing/`):
     - Step1 (`preprocessing/price_factors.py`): 5 price factors
       (log-return / range / momentum), standardized per ticker on the
       training period only, plus 4 mean-pooled + PCA-reduced text levels
       (macro / sector / related-company / target-company) as the Step1
       baseline text signal.
     - Step2 (`preprocessing/macro_factors.py`, `--use_macro`): cross-day
       repetition streak, 20d embedding-drift regime-change indicator, and a
       train-fit Ridge probe (macro embedding -> market-avg forward return)
       gated against each ticker's own momentum.
     - Step3 (`preprocessing/sector_factors.py`, `--use_sector`): sector
       membership recovered directly from which tickers share the same
       `sector_category1` text_id on a date (a stable ~11-cluster partition,
       not statistical clustering -- see module docstring), then peer
       return / relative strength / pooled cluster text embedding.
     - Step4 (`preprocessing/target_factors.py`, `--use_target_novelty` /
       `--use_related_novelty`): per-embedding-model novelty score (cosine
       distance to each ticker's own trailing rolling-mean embedding),
       aggregated into a cross-model consensus vote, crossed with recent
       price movement into "already priced in" vs "drift" features.
   Each row's "feature" is a `factor_num * seq_len` flattened lookback window
   (factor-major order, i.e. Alpha360-style) ending at that date; "label" is
   the standardized `horizon`-trading-day-ahead log-return of close.
2. Feed this panel into `DoubleAdapt/src`'s qlib-independent
   `DoubleAdaptManager`/`IncrementalManager` (offline meta-training on
   2019-2021/valid on 2022, then rolling online adaptation through 2023),
   wrapping a `--backbone` picked from `forecasting_task/backbones.py`
   (Step1's swappable time-series backbone; GRU today, or any
   Time-Series-Library architecture already in forecasting_task/models/).
3. Reconstruct predicted close prices and report the challenge's directional
   Weighted Hit Rate (anchored on the close price `horizon` trading days
   before the target date) plus MSE/MAE on price.

Usage
-----
    python -m forecasting_task.run_DoubleAdapt \
        --tickers AAPL,MSFT,NVDA --logdir logs/doubleadapt_smoke

    # with Step2/3/4 preprocessing and a non-GRU backbone:
    python -m forecasting_task.run_DoubleAdapt \
        --tickers AAPL,MSFT,NVDA --use_macro --use_sector --use_target_novelty \
        --use_related_novelty --backbone patchtst --logdir logs/doubleadapt_full
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
FORECASTING_DIR = Path(__file__).resolve().parent
REPO_ROOT = FORECASTING_DIR.parent
DOUBLEADAPT_SRC = REPO_ROOT / "DoubleAdapt" / "src"

for _p in (str(REPO_ROOT), str(DOUBLEADAPT_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# DoubleAdapt/src/model.py does unqualified `from utils import ...` / `import
# higher_optim`, so DoubleAdapt/src itself (not just DoubleAdapt/) must be on
# sys.path. We therefore import these as flat top-level modules, not via
# `from src.model import ...` (which is what DoubleAdapt/main.py does and which
# would break these bare imports if DoubleAdapt/src isn't also on the path).
import model as da_model  # noqa: E402  (DoubleAdapt/src/model.py)
import utils as da_utils  # noqa: E402  (DoubleAdapt/src/utils.py)

from forecasting_task.backbones import build_backbone  # noqa: E402
from forecasting_task.preprocessing.panel import PanelConfig, build_panel  # noqa: E402
from forecasting_task.utils.metrics import MAE, MSE  # noqa: E402


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Weighted Hit Rate (challenge metric)
# ---------------------------------------------------------------------------
def weighted_hit_rate(pred_close: pd.Series, true_close: pd.Series, anchor_close: pd.Series,
                       weights: pd.Series = None) -> float:
    """sign(pred_close - anchor_close) == sign(true_close - anchor_close), weighted mean."""
    valid = true_close.notna()
    pred_close, true_close, anchor_close = pred_close[valid], true_close[valid], anchor_close[valid]
    hit = (np.sign(pred_close - anchor_close) == np.sign(true_close - anchor_close)).astype(float)
    if weights is None:
        w = pd.Series(1.0, index=hit.index)
    else:
        w = weights.reindex(hit.index).fillna(1.0)
    return float((hit * w).sum() / w.sum())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="DoubleAdapt runner for FinTexTS 4-week close-price forecasting")
    p.add_argument("--data_csv", default=str(REPO_ROOT / "data" / "test.parquet"),
                   help="Superset file (.csv or .parquet). train.parquet's 2019-2022 rows are a byte-identical "
                        "subset of this -- verified via DataFrame.equals(). Only used for causal feature "
                        "computation during the frozen online/eval phase (see --freeze_online); offline "
                        "training rolling tasks only ever touch rows dated <= valid_end anyway.")
    p.add_argument("--freeze_online", type=int, default=0,
                   help="1 = the online/eval phase (valid+test rolling tasks) does a plain frozen forward "
                        "pass -- no gradient step, no local few-shot adaptation. Required for competition "
                        "compliance when eval-period rows come from a file (test.parquet) that must not be "
                        "used for training: model weights are then only ever updated using data dated "
                        "<= train_end/valid_end. 0 (default) = original DoubleAdapt online-adaptation "
                        "behavior, unchanged (used for all prior single-CSV internal experiments).")
    p.add_argument("--data_dir", default=str(REPO_ROOT / "data"),
                   help="Directory scanned for *_textemb.parquet embedding models.")
    p.add_argument("--tickers", default=None, help="Comma-separated tickers; default = all in data_csv.")
    p.add_argument("--panel_cache_dir", default=str(REPO_ROOT / "data" / ".panel_cache"),
                   help="Memoizes build_panel() output (keyed on tickers/dates/PanelConfig/embedding mtimes) "
                        "since panel construction is backbone-independent -- runs that only vary --backbone "
                        "skip the full rebuild. Pass '' to disable.")

    p.add_argument("--seq_len", type=int, default=60)
    p.add_argument("--horizon", type=int, default=20, help="True forward distance in trading days (~4 weeks).")
    p.add_argument("--step", type=int, default=25, help="Rolling task interval; must be > horizon.")
    p.add_argument("--n_text_components", type=int, default=6)
    p.add_argument("--primary_emb_model", default=None,
                   help="Embedding model used for single-model features (Step1 text PCA, Step2 macro, "
                        "Step3 sector pooled embedding). Default = first model EmbeddingRegistry finds.")

    # Step2/3/4 preprocessing toggles (all off = original Step1-only baseline).
    p.add_argument("--use_macro", action="store_true", help="Step2: macro streak/regime-shift/direction-align.")
    p.add_argument("--use_sector", action="store_true", help="Step3: sector peer/relative-strength/pooled-embed.")
    p.add_argument("--use_target_novelty", action="store_true", help="Step4: target-level novelty consensus.")
    p.add_argument("--use_related_novelty", action="store_true", help="Step4: related-company novelty consensus.")
    p.add_argument("--use_sector_novelty", action="store_true",
                   help="Step4 variant: sector-level novelty consensus. Weaker signal expected -- "
                        "sector_category text_ids are shared by every ticker in the same cluster "
                        "(~9-10 tickers), so novelty is far less per-sample-differentiated than "
                        "target/related (each ticker's own, mostly-unique text).")
    p.add_argument("--novelty_window", type=int, default=20)
    p.add_argument("--novelty_quantile", type=float, default=0.9)
    p.add_argument("--novelty_consensus_frac", type=float, default=2.0 / 3.0)
    p.add_argument("--sector_agreement_threshold", type=float, default=0.8)
    p.add_argument("--macro_similarity_threshold", type=float, default=0.9)
    p.add_argument("--macro_drift_window", type=int, default=20)
    p.add_argument("--use_text_signal", action="store_true",
                    help="Supervised pretrained text-direction signal (ported from V7Model's "
                         "TextClassifier): a small transformer trained on train-period rows only "
                         "to predict horizon-ahead direction from the 4 level embeddings, scored "
                         "on every row -- a task-aware alternative to Step1/3's unsupervised PCA.")
    p.add_argument("--text_signal_hidden", type=int, default=128)
    p.add_argument("--text_signal_layers", type=int, default=2)
    p.add_argument("--text_signal_heads", type=int, default=4)
    p.add_argument("--text_signal_epochs", type=int, default=15)

    # Step1 backbone.
    p.add_argument("--backbone", default="gru", help="gru, dlinear, patchtst, itransformer, timesnet.")
    p.add_argument("--hidden_size", type=int, default=64, help="GRU backbone only.")
    p.add_argument("--num_layers", type=int, default=2, help="GRU backbone only.")
    p.add_argument("--d_model", type=int, default=64, help="TSLib backbones only.")
    p.add_argument("--n_heads", type=int, default=4, help="TSLib backbones only.")
    p.add_argument("--e_layers", type=int, default=2, help="TSLib backbones only.")
    p.add_argument("--d_ff", type=int, default=128, help="TSLib backbones only.")
    p.add_argument("--backbone_dropout", type=float, default=0.1, help="TSLib backbones only.")

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_da", type=float, default=1e-2)
    p.add_argument("--lr_ma", type=float, default=1e-3)
    p.add_argument("--reg", type=float, default=0.5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--num_head", type=int, default=8)
    p.add_argument("--tau", type=float, default=10.0)
    p.add_argument("--adapt_x", type=int, default=1)
    p.add_argument("--adapt_y", type=int, default=1)
    p.add_argument("--naive", type=int, default=0, help="1 = plain IncrementalManager baseline (no meta-learning).")
    p.add_argument("--aux_weight", type=float, default=0.0,
                   help="0.0 (default) = plain MSE loss, unchanged. >0 switches to JointMSEDirLoss "
                        "(MSE + aux_weight * direction BCE), which supplies a directional gradient on "
                        "every training batch -- unlike the (already tried, reverted) hit_rate-based "
                        "checkpoint selection, which failed because a small validation set made the "
                        "signal too noisy to select on.")
    p.add_argument("--dir_scale", type=float, default=5.0,
                   help="Scales the regression output into a direction logit for the BCE term.")
    p.add_argument("--news_weight_mult", type=float, default=1.0,
                   help="1.0 (default) = every sample weighted equally in the loss, unchanged. "
                        ">1 upweights samples where target_novelty_consensus>0 or related_drift_flag>0 "
                        "by this multiplier, so the model is penalized more for getting those directions "
                        "wrong -- these are our proxy for the officially-undisclosed 'news matters more "
                        "here' weighting the real Weighted Hit Rate metric uses. Requires "
                        "--use_target_novelty and/or --use_related_novelty so those columns exist.")
    p.add_argument("--graded_news_weight", action="store_true",
                   help="Replaces the binary is_news_important gate with a continuous one: weight = "
                        "1 + (news_weight_mult - 1) * max(target_novelty_consensus, related_novelty_consensus), "
                        "where that max term is exactly the k-of-N embedding-model agreement fraction "
                        "(k=1..N-1 out of N=6 models each get a distinct weight, instead of everything "
                        "below the --novelty_consensus_frac threshold being 1x and everything at/above "
                        "being a flat news_weight_mult). novelty_consensus_frac itself is unused in this "
                        "mode (it only gates the binary already_priced/drift_flag columns, not the raw "
                        "consensus fraction this reads).")

    p.add_argument("--train_start", default="2019-01-01")
    p.add_argument("--train_end", default="2021-12-31")
    p.add_argument("--valid_start", default="2022-01-01")
    p.add_argument("--valid_end", default="2022-12-31")
    p.add_argument("--test_start", default="2023-01-01")
    p.add_argument("--test_end", default="2023-12-01")

    p.add_argument("--weights_csv", default=None,
                   help="Optional CSV with columns date,ticker,weight for the Weighted Hit Rate.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--logdir", default="logs/doubleadapt_default")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "args.yaml"), "w") as f:
        yaml.safe_dump(vars(args), f, sort_keys=False)

    df = pd.read_parquet(args.data_csv) if str(args.data_csv).endswith(".parquet") else pd.read_csv(args.data_csv)
    df["date"] = pd.to_datetime(df["date"])
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
        df = df[df["ticker"].isin(tickers)]
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    cfg = PanelConfig(
        seq_len=args.seq_len, horizon=args.horizon, n_text_components=args.n_text_components,
        train_start=args.train_start, train_end=args.train_end, primary_emb_model=args.primary_emb_model,
        use_macro=args.use_macro, use_sector=args.use_sector,
        use_target_novelty=args.use_target_novelty, use_related_novelty=args.use_related_novelty,
        use_sector_novelty=args.use_sector_novelty,
        novelty_window=args.novelty_window, novelty_quantile=args.novelty_quantile,
        novelty_consensus_frac=args.novelty_consensus_frac,
        sector_agreement_threshold=args.sector_agreement_threshold,
        macro_similarity_threshold=args.macro_similarity_threshold, macro_drift_window=args.macro_drift_window,
        use_text_signal=args.use_text_signal, text_signal_hidden=args.text_signal_hidden,
        text_signal_layers=args.text_signal_layers, text_signal_heads=args.text_signal_heads,
        text_signal_epochs=args.text_signal_epochs,
    )

    t0 = time.time()
    panel_df, raw_close, label_mean, label_std, factor_num, seq_len, factor_cols = build_panel(
        df, args.data_dir, cfg, cache_dir=args.panel_cache_dir or None
    )
    if args.news_weight_mult != 1.0 and args.graded_news_weight:
        consensus_cols = [c for c in ("target_novelty_consensus", "related_novelty_consensus") if c in factor_cols]
        if not consensus_cols:
            print("[NewsWeight] --graded_news_weight set but no target_novelty_consensus/"
                  "related_novelty_consensus column found -- skipping, weight=1 everywhere.")
        else:
            feat = panel_df["feature"]
            consensus_score = np.zeros(len(panel_df), dtype=np.float32)
            for name in consensus_cols:
                fi = factor_cols.index(name)
                last_step_col = (fi + 1) * seq_len - 1  # most recent day in the window == "as of now" value
                consensus_score = np.maximum(consensus_score, feat.iloc[:, last_step_col].to_numpy())
            weight_vec = (1.0 + (args.news_weight_mult - 1.0) * consensus_score).astype(np.float32)
            panel_df[("weight", 0)] = weight_vec
            print(f"[NewsWeight] graded by k-of-N consensus fraction: mean_consensus={consensus_score.mean():.4f} "
                  f"mean_weight={weight_vec.mean():.4f} max_weight={weight_vec.max():.4f}")
    elif args.news_weight_mult != 1.0:
        news_cols = [c for c in ("target_novelty_consensus", "related_drift_flag", "sector_drift_flag")
                     if c in factor_cols]
        if not news_cols:
            print("[NewsWeight] --news_weight_mult set but no target_novelty_consensus/related_drift_flag/"
                  "sector_drift_flag column found (need --use_target_novelty/--use_related_novelty/"
                  "--use_sector_novelty) -- skipping, weight=1 everywhere.")
        else:
            feat = panel_df["feature"]
            is_news = np.zeros(len(panel_df), dtype=bool)
            for name in news_cols:
                fi = factor_cols.index(name)
                last_step_col = (fi + 1) * seq_len - 1  # most recent day in the window == "as of now" value
                is_news |= (feat.iloc[:, last_step_col].to_numpy() > 0)
            weight_vec = np.where(is_news, args.news_weight_mult, 1.0).astype(np.float32)
            panel_df[("weight", 0)] = weight_vec
            print(f"[NewsWeight] {is_news.mean():.4f} of samples flagged news-important, "
                  f"weighted {args.news_weight_mult}x in the training loss.")
    print(f"[Panel] shape={panel_df.shape} factor_num={factor_num} seq_len={seq_len} "
          f"built in {time.time() - t0:.1f}s")
    with open(os.path.join(args.logdir, "factor_cols.json"), "w") as f:
        json.dump(factor_cols, f)

    calendar = pd.Series(sorted(panel_df.index.get_level_values("datetime").unique()))
    ta = da_utils.TimeAdjuster(calendar)

    segments = {
        "train": (args.train_start, args.train_end),
        "valid": (args.valid_start, args.valid_end),
        "test": (args.test_start, args.test_end),
    }
    # Our label already looks exactly `horizon` trading days ahead (no off-by-one),
    # whereas DoubleAdapt's utils.preprocess() internally uses H = 1 + horizon_param
    # to size the "meta_end" boundary. Pass horizon_param = horizon - 1 so H == horizon.
    horizon_param = args.horizon - 1
    rolling_tasks = da_utils.organize_all_tasks(
        segments, ta, step=args.step, trunc_days=args.horizon, rtype=da_utils.TimeAdjuster.SHIFT_SD,
    )
    rolling_tasks_data = {
        k: da_utils.get_rolling_data(
            rolling_tasks[k], data=panel_df, factor_num=factor_num, horizon=horizon_param,
            not_sequence=False, sequence_last_dim=True, to_tensor=True,
        )
        for k in ["train", "valid", "test"]
    }
    print(f"[Tasks] train={len(rolling_tasks_data['train'])} valid={len(rolling_tasks_data['valid'])} "
          f"test={len(rolling_tasks_data['test'])}")

    net_model = build_backbone(
        args.backbone, factor_num, seq_len,
        hidden_size=args.hidden_size, num_layers=args.num_layers,
        d_model=args.d_model, n_heads=args.n_heads, e_layers=args.e_layers, d_ff=args.d_ff,
        dropout=args.backbone_dropout,
    )

    if args.naive:
        framework = da_model.IncrementalManager(
            net_model, x_dim=factor_num * seq_len, lr_model=args.lr, weight_decay=args.weight_decay,
            need_permute=False, begin_valid_epoch=0,
            aux_weight=args.aux_weight, dir_scale=args.dir_scale, label_mean=label_mean, label_std=label_std,
            freeze_online=bool(args.freeze_online),
        )
    else:
        framework = da_model.DoubleAdaptManager(
            net_model, x_dim=factor_num * seq_len, lr_model=args.lr, weight_decay=args.weight_decay,
            first_order=True, begin_valid_epoch=0, factor_num=factor_num,
            lr_da=args.lr_da, lr_ma=args.lr_ma, adapt_x=bool(args.adapt_x), adapt_y=bool(args.adapt_y),
            reg=args.reg, num_head=args.num_head, temperature=args.tau, need_permute=False,
            aux_weight=args.aux_weight, dir_scale=args.dir_scale, label_mean=label_mean, label_std=label_std,
            freeze_online=bool(args.freeze_online),
        )

    ckpt_path = os.path.join(args.logdir, "checkpoint.pt")
    t0 = time.time()
    framework.fit(rolling_tasks_data["train"], rolling_tasks_data["valid"], checkpoint_path=ckpt_path)
    print(f"[Offline] fit done in {time.time() - t0:.1f}s")

    test_slice = slice(ta.align_time(args.test_start, tp_type="start"), ta.align_time(args.test_end, tp_type="end"))
    t0 = time.time()
    pred_y_all = framework.inference(rolling_tasks_data["test"], date_slice=test_slice)
    print(f"[Online] inference done in {time.time() - t0:.1f}s, {len(pred_y_all)} predictions")

    # De-standardize and reconstruct absolute close prices.
    raw_pred = pred_y_all["pred"] * label_std + label_mean
    raw_true = pred_y_all["label"] * label_std + label_mean
    anchor_close = raw_close.reindex(pred_y_all.index)
    pred_close = anchor_close * np.exp(raw_pred)
    true_close = anchor_close * np.exp(raw_true)  # NaN for the unlabeled forward tail (real submission targets)

    weights = None
    if args.weights_csv:
        wdf = pd.read_csv(args.weights_csv, parse_dates=["date"])
        weights = wdf.set_index([wdf["date"], wdf["ticker"]])["weight"]
        weights.index.names = ["datetime", "instrument"]

    hit_rate = weighted_hit_rate(pred_close, true_close, anchor_close, weights=weights)
    valid_mask = true_close.notna()
    mse = MSE(pred_close[valid_mask].to_numpy(), true_close[valid_mask].to_numpy())
    mae = MAE(pred_close[valid_mask].to_numpy(), true_close[valid_mask].to_numpy())

    print(f"[Result] Weighted Hit Rate={hit_rate:.4f} MSE={mse:.4f} MAE={mae:.4f} "
          f"(n_labeled={int(valid_mask.sum())}, n_total={len(pred_y_all)})")

    out = pd.DataFrame({
        "anchor_close": anchor_close,
        "pred_close": pred_close,
        "true_close": true_close,
    })
    out.index.names = ["date", "ticker"]
    out.to_csv(os.path.join(args.logdir, "predictions.csv"))
    with open(os.path.join(args.logdir, "results.txt"), "w") as f:
        f.write(f"weighted_hit_rate={hit_rate:.6f}\nmse={mse:.6f}\nmae={mae:.6f}\n"
                f"n_labeled={int(valid_mask.sum())}\nn_total={len(pred_y_all)}\n")


if __name__ == "__main__":
    main()
