"""
Build a Kaggle-format submission.csv from a completed run_DoubleAdapt.py checkpoint.

Reloads <logdir>/args.yaml + factor_cols.json + checkpoint.pt (no retraining),
rebuilds the panel, and runs online inference over BOTH the valid (2022, public
leaderboard) and test (2023, private leaderboard) rolling-task segments -- the
training script only ever ran inference on "test" for scoring, so this adds the
2022 pass needed for a full submission.

Each prediction is anchored at date t and represents the model's predicted
close price `horizon` trading days later (t_target). The submission format
requires rows keyed by t_target, not t, so target dates are derived by
shifting each ticker's own trading-day sequence forward by `horizon` positions
(falling back to a business-day approximation past the edge of our price data).

Every required weekday ID (100 tickers x all weekdays in 2022-01-01..2023-12-31)
gets a value: real prediction where we have one, otherwise forward/back-filled
from the nearest available prediction or actual close (holidays / early-2022
dates before our first anchor's target date included).

Usage
-----
    python -m forecasting_task.generate_kaggle_submission --logdir logs/dlinear_mult10_reg10 \
        --out submission_dlinear_newsweight10_reg10_lr002.csv
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

FORECASTING_DIR = Path(__file__).resolve().parent
REPO_ROOT = FORECASTING_DIR.parent
DOUBLEADAPT_SRC = REPO_ROOT / "DoubleAdapt" / "src"
for _p in (str(REPO_ROOT), str(DOUBLEADAPT_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import model as da_model  # noqa: E402
import utils as da_utils  # noqa: E402

from forecasting_task.backbones import build_backbone  # noqa: E402
from forecasting_task.preprocessing.panel import PanelConfig, build_panel  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Build a Kaggle submission.csv from a trained checkpoint")
    p.add_argument("--logdir", required=True)
    p.add_argument("--out", default=None, help="Default: <repo_root>/submission_<logdir basename>.csv")
    p.add_argument("--sub_start", default="2022-01-01")
    p.add_argument("--sub_end", default="2023-12-31")
    cli_args = p.parse_args()

    with open(os.path.join(cli_args.logdir, "args.yaml")) as f:
        args = argparse.Namespace(**yaml.safe_load(f))

    df = pd.read_parquet(args.data_csv) if str(args.data_csv).endswith(".parquet") else pd.read_csv(args.data_csv)
    df["date"] = pd.to_datetime(df["date"])
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
        df = df[df["ticker"].isin(tickers)]
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    all_tickers = sorted(df["ticker"].unique().tolist())

    cfg = PanelConfig(
        seq_len=args.seq_len, horizon=args.horizon, n_text_components=args.n_text_components,
        train_start=args.train_start, train_end=args.train_end, primary_emb_model=args.primary_emb_model,
        use_macro=args.use_macro, use_sector=args.use_sector,
        use_target_novelty=args.use_target_novelty, use_related_novelty=args.use_related_novelty,
        use_sector_novelty=getattr(args, "use_sector_novelty", False),
        novelty_window=args.novelty_window, novelty_quantile=args.novelty_quantile,
        novelty_consensus_frac=args.novelty_consensus_frac,
        sector_agreement_threshold=args.sector_agreement_threshold,
        macro_similarity_threshold=args.macro_similarity_threshold, macro_drift_window=args.macro_drift_window,
        use_text_signal=getattr(args, "use_text_signal", False),
        text_signal_hidden=getattr(args, "text_signal_hidden", 128),
        text_signal_layers=getattr(args, "text_signal_layers", 2),
        text_signal_heads=getattr(args, "text_signal_heads", 4),
        text_signal_epochs=getattr(args, "text_signal_epochs", 15),
    )
    panel_cache_dir = getattr(args, "panel_cache_dir", str(REPO_ROOT / "data" / ".panel_cache")) or None
    panel_df, raw_close, label_mean, label_std, factor_num, seq_len, factor_cols = build_panel(
        df, args.data_dir, cfg, cache_dir=panel_cache_dir
    )

    news_weight_mult = getattr(args, "news_weight_mult", 1.0)
    if news_weight_mult != 1.0:
        news_cols = [c for c in ("target_novelty_consensus", "related_drift_flag", "sector_drift_flag")
                     if c in factor_cols]
        if news_cols:
            feat = panel_df["feature"]
            is_news = np.zeros(len(panel_df), dtype=bool)
            for name in news_cols:
                fi = factor_cols.index(name)
                last_step_col = (fi + 1) * seq_len - 1
                is_news |= (feat.iloc[:, last_step_col].to_numpy() > 0)
            panel_df[("weight", 0)] = np.where(is_news, news_weight_mult, 1.0).astype(np.float32)

    calendar = pd.Series(sorted(panel_df.index.get_level_values("datetime").unique()))
    ta = da_utils.TimeAdjuster(calendar)
    segments = {
        "train": (args.train_start, args.train_end),
        "valid": (args.valid_start, args.valid_end),
        "test": (args.test_start, args.test_end),
    }
    horizon_param = args.horizon - 1
    rolling_tasks = da_utils.organize_all_tasks(
        segments, ta, step=args.step, trunc_days=args.horizon, rtype=da_utils.TimeAdjuster.SHIFT_SD,
    )
    rolling_tasks_data = {
        k: da_utils.get_rolling_data(
            rolling_tasks[k], data=panel_df, factor_num=factor_num, horizon=horizon_param,
            not_sequence=False, sequence_last_dim=True, to_tensor=True,
        )
        for k in ["valid", "test"]
    }

    net_model = build_backbone(
        args.backbone, factor_num, seq_len,
        hidden_size=args.hidden_size, num_layers=args.num_layers,
        d_model=args.d_model, n_heads=args.n_heads, e_layers=args.e_layers, d_ff=args.d_ff,
        dropout=args.backbone_dropout,
    )
    aux_weight = getattr(args, "aux_weight", 0.0)
    dir_scale = getattr(args, "dir_scale", 5.0)
    freeze_online = bool(getattr(args, "freeze_online", 0))
    if args.naive:
        framework = da_model.IncrementalManager(
            net_model, x_dim=factor_num * seq_len, lr_model=args.lr, weight_decay=args.weight_decay,
            need_permute=False, begin_valid_epoch=0,
            aux_weight=aux_weight, dir_scale=dir_scale, label_mean=label_mean, label_std=label_std,
            freeze_online=freeze_online,
        )
    else:
        framework = da_model.DoubleAdaptManager(
            net_model, x_dim=factor_num * seq_len, lr_model=args.lr, weight_decay=args.weight_decay,
            first_order=True, begin_valid_epoch=0, factor_num=factor_num,
            lr_da=args.lr_da, lr_ma=args.lr_ma, adapt_x=bool(args.adapt_x), adapt_y=bool(args.adapt_y),
            reg=args.reg, num_head=args.num_head, temperature=args.tau, need_permute=False,
            freeze_online=freeze_online,
            aux_weight=aux_weight, dir_scale=dir_scale, label_mean=label_mean, label_std=label_std,
        )
    framework.load_state_dict(torch.load(os.path.join(cli_args.logdir, "checkpoint.pt")))

    def run_inference(seg_name, start, end):
        date_slice = slice(ta.align_time(start, tp_type="start"), ta.align_time(end, tp_type="end"))
        pred_y_all = framework.inference(rolling_tasks_data[seg_name], date_slice=date_slice)
        raw_pred = pred_y_all["pred"] * label_std + label_mean
        anchor_close = raw_close.reindex(pred_y_all.index)
        pred_close = anchor_close * np.exp(raw_pred)
        out = pred_close.rename("pred_close").to_frame()
        out.index.names = ["anchor_date", "ticker"]
        return out

    print("[Submission] running online inference on valid (2022) segment...")
    preds_2022 = run_inference("valid", args.valid_start, args.valid_end)
    print(f"[Submission] {len(preds_2022)} anchor predictions from valid segment")
    print("[Submission] running online inference on test (2023) segment...")
    preds_2023 = run_inference("test", args.test_start, args.test_end)
    print(f"[Submission] {len(preds_2023)} anchor predictions from test segment")

    anchor_preds = pd.concat([preds_2022, preds_2023]).sort_index()

    # --- Map each anchor-date prediction to its actual target date (anchor + horizon trading days) ---
    horizon = args.horizon
    target_rows = []
    for ticker, g in df.groupby("ticker", sort=False):
        dates = g["date"].sort_values().reset_index(drop=True)
        pos_of_date = pd.Series(dates.index.values, index=dates.values)
        sub = anchor_preds.xs(ticker, level="ticker", drop_level=False) if ticker in anchor_preds.index.get_level_values("ticker") else None
        if sub is None or len(sub) == 0:
            continue
        for (anchor_date, _), row in sub.iterrows():
            pos = pos_of_date.get(anchor_date)
            if pos is not None and pos + horizon < len(dates):
                target_date = dates.iloc[pos + horizon]
            else:
                # past the edge of our price history -- approximate with a business-day offset
                target_date = anchor_date + pd.tseries.offsets.BDay(horizon)
            target_rows.append((target_date, ticker, row["pred_close"]))

    pred_df = pd.DataFrame(target_rows, columns=["date", "ticker", "pred_close"])
    pred_df = pred_df.groupby(["date", "ticker"], as_index=False)["pred_close"].mean()  # de-dup overlapping rolling tasks
    pred_lookup = pred_df.set_index(["date", "ticker"])["pred_close"]

    # --- Build every required weekday ID and fill ---
    sub_dates = pd.date_range(cli_args.sub_start, cli_args.sub_end, freq="B")
    print(f"[Submission] {len(sub_dates)} weekdays x {len(all_tickers)} tickers = "
          f"{len(sub_dates) * len(all_tickers)} rows required")

    rows = []
    for ticker in all_tickers:
        actual = df[df["ticker"] == ticker].set_index("date")["close"].sort_index()
        last_val = actual[actual.index < cli_args.sub_start]
        last_val = float(last_val.iloc[-1]) if len(last_val) else float(actual.iloc[0]) if len(actual) else 100.0
        for d in sub_dates:
            if (d, ticker) in pred_lookup.index:
                val = float(pred_lookup.loc[(d, ticker)])
                last_val = val
            elif d in actual.index:
                val = float(actual.loc[d])
                last_val = val
            else:
                val = last_val
            rows.append({"ID": f"{ticker}_{d.strftime('%Y-%m-%d')}", "Close": val})

    sub_df = pd.DataFrame(rows)
    out_path = cli_args.out or str(REPO_ROOT / f"submission_{Path(cli_args.logdir).name}.csv")
    sub_df.to_csv(out_path, index=False)
    print(f"[Submission] wrote {len(sub_df)} rows to {out_path}")
    print(sub_df.head())
    print(sub_df.tail())


if __name__ == "__main__":
    main()
