"""
Chronos-2 rolling-forecast baseline, log-return variant.

Frozen sibling of `run_chronos.py` (not modified) -- that script feeds
Chronos-2 the raw close-price *level* as the univariate target ("simple
normalization": the model's own internal scaling handles the level series).
This variant instead transforms each ticker's close series into daily log
returns (log(close_t / close_{t-1})) before handing it to Chronos, forecasts
the log-return path `horizon` steps ahead, and reconstructs the predicted
close by compounding the cumulative predicted log return onto the anchor's
known close: pred_close = anchor_close * exp(sum(predicted daily log rets)).

Rationale (per review notes): 2023 was a sustained uptrend, not a range-bound
market, so a raw close-price series handed to a foundation model is
non-stationary -- forecasting its log-return series instead is much closer to
stationary and should improve on simple-normalization forecasting. This is
reported as orthogonal to DoubleAdapt's novelty-consensus text signal, so we
keep that signal wired in exactly as `run_chronos.py` does (same covariate
columns, same panel/build_panel machinery) and only change the forecasting
target's transform -- isolating the log-return effect for a clean
apples-to-apples comparison against `logs/chronos_direction` / `chronos_news2`.

Usage
-----
    python -m forecasting_task.run_chronos_logret --step 25 --horizon 20
    python -m forecasting_task.run_chronos_logret --tickers AAPL,MSFT,NVDA --out_dir logs/chronos_logret_smoke
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from forecasting_task.preprocessing.panel import PanelConfig, build_panel  # noqa: E402


def build_long_frame_logret(panel_df, raw_close, factor_cols, seq_len, cov_cols=None):
    """One row per (date, ticker): close_level (kept for reconstruction/scoring) +
    target = daily log return of close (the series Chronos actually forecasts) +
    text/novelty covariates (last-value-in-window). Same covariate default and cost
    profile as `run_chronos.build_long_frame` -- only the forecast target differs."""
    if cov_cols is None:
        cov_cols = [c for c in ("target_novelty_consensus", "related_drift_flag") if c in factor_cols]
    feat = panel_df["feature"]
    last_idx = {c: (factor_cols.index(c) + 1) * seq_len - 1 for c in cov_cols}
    cov_df = pd.DataFrame({c: feat.iloc[:, last_idx[c]].to_numpy() for c in cov_cols}, index=panel_df.index)
    wide = raw_close.rename("close_level").to_frame().join(cov_df)
    wide = wide.reset_index().rename(columns={"datetime": "timestamp", "instrument": "id"})
    wide = wide.sort_values(["id", "timestamp"]).reset_index(drop=True)
    wide["target"] = wide.groupby("id", sort=False)["close_level"].transform(
        lambda s: np.log(s / s.shift(1))
    )
    # first row per ticker has no prior close -> log return undefined; drop rather than
    # impute, since Chronos should never see a fabricated "0 return" as real history.
    wide = wide.dropna(subset=["target"]).reset_index(drop=True)
    return wide, cov_cols


def main():
    p = argparse.ArgumentParser(description="Chronos-2 rolling direction-forecast baseline (log-return target)")
    p.add_argument("--reference_logdir", default=str(REPO_ROOT / "logs" / "dlinear_final_compliant"))
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--step", type=int, default=25, help="Trading days between rolling anchor dates.")
    p.add_argument("--valid_start", default="2022-01-01")
    p.add_argument("--valid_end", default="2022-12-31")
    p.add_argument("--test_start", default="2023-01-01")
    p.add_argument("--test_end", default="2023-12-01")
    p.add_argument("--device", default="cpu")
    p.add_argument("--tickers", default=None, help="Comma-separated tickers; default = all in data_csv.")
    p.add_argument("--covariates", default=None,
                   help="Comma-separated covariate column names (default: target_novelty_consensus,"
                        "related_drift_flag -- see build_long_frame_logret docstring re: runtime cost).")
    p.add_argument("--out_dir", default=str(REPO_ROOT / "logs" / "chronos_logret_direction"))
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    import yaml
    with open(os.path.join(args.reference_logdir, "args.yaml")) as f:
        ref = argparse.Namespace(**yaml.safe_load(f))

    df = pd.read_parquet(ref.data_csv)
    df["date"] = pd.to_datetime(df["date"])
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
        df = df[df["ticker"].isin(tickers)]
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    cfg = PanelConfig(
        seq_len=ref.seq_len, horizon=args.horizon, n_text_components=ref.n_text_components,
        train_start=ref.train_start, train_end=ref.train_end, primary_emb_model=ref.primary_emb_model,
        use_target_novelty=ref.use_target_novelty, use_related_novelty=ref.use_related_novelty,
        novelty_window=ref.novelty_window, novelty_quantile=ref.novelty_quantile,
        novelty_consensus_frac=ref.novelty_consensus_frac,
    )
    panel_df, raw_close, _, _, factor_num, seq_len, factor_cols = build_panel(
        df, ref.data_dir, cfg, cache_dir=ref.panel_cache_dir
    )
    cov_override = [c.strip() for c in args.covariates.split(",")] if args.covariates else None
    wide, cov_cols = build_long_frame_logret(panel_df, raw_close, factor_cols, seq_len, cov_cols=cov_override)
    print(f"[Data] {len(wide)} rows, {wide['id'].nunique()} tickers, covariates={cov_cols}, target=log_return")

    all_dates = sorted(wide["timestamp"].unique())

    def anchors_in(start, end):
        rng = [d for d in all_dates if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
        return rng[::args.step]

    valid_anchors = anchors_in(args.valid_start, args.valid_end)
    test_anchors = anchors_in(args.test_start, args.test_end)
    print(f"[Anchors] valid={len(valid_anchors)} test={len(test_anchors)}")

    from chronos import Chronos2Pipeline
    t0 = time.time()
    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=args.device)
    print(f"[Model] loaded in {time.time()-t0:.1f}s")

    close_lookup = wide.set_index(["id", "timestamp"])["close_level"]

    def target_date_for(ticker, anchor_date):
        dates = wide.loc[wide["id"] == ticker, "timestamp"]
        pos = dates.searchsorted(anchor_date)
        idx = pos + args.horizon
        return dates.iloc[idx] if idx < len(dates) else None

    def run_anchor(anchor_date):
        context_df = wide[wide["timestamp"] <= anchor_date].drop(columns=["close_level"])
        pred = pipeline.predict_df(
            context_df, prediction_length=args.horizon, quantile_levels=[0.5],
            id_column="id", timestamp_column="timestamp", target="target", freq="B",
        )
        # cumulative predicted log return over the full horizon == sum of the
        # predicted daily log returns for all `horizon` forecasted steps.
        cum_logret = pred.groupby("id")["predictions"].sum()
        rows = []
        for ticker, cum_ret in cum_logret.items():
            tgt_date = target_date_for(ticker, anchor_date)
            if tgt_date is None:
                continue
            anchor_close = close_lookup.get((ticker, anchor_date))
            true_close = close_lookup.get((ticker, tgt_date))
            if anchor_close is None or true_close is None:
                continue
            pred_close = anchor_close * np.exp(cum_ret)
            rows.append({
                "ticker": ticker, "anchor_date": anchor_date, "target_date": tgt_date,
                "anchor_close": anchor_close, "pred_close": pred_close, "true_close": true_close,
                "cum_logret": cum_ret,
            })
        return rows

    def score(anchors, name):
        all_rows = []
        for i, a in enumerate(anchors):
            t1 = time.time()
            rows = run_anchor(a)
            all_rows.extend(rows)
            print(f"[{name}] anchor {i+1}/{len(anchors)} ({a.date()}) -> {len(rows)} preds in {time.time()-t1:.1f}s")
        out = pd.DataFrame(all_rows)
        if len(out) == 0:
            print(f"[{name}] no scoreable rows")
            return out, float("nan")
        hit = (np.sign(out["pred_close"] - out["anchor_close"]) == np.sign(out["true_close"] - out["anchor_close"]))
        hit_rate = float(hit.mean())
        print(f"[{name}] n={len(out)} hit_rate={hit_rate:.4f}")
        return out, hit_rate

    valid_df, valid_hit = score(valid_anchors, "valid_2022")
    test_df, test_hit = score(test_anchors, "test_2023")

    valid_df.to_csv(os.path.join(args.out_dir, "valid_preds.csv"), index=False)
    test_df.to_csv(os.path.join(args.out_dir, "test_preds.csv"), index=False)
    with open(os.path.join(args.out_dir, "results.txt"), "w") as f:
        f.write(f"valid_2022_hit_rate={valid_hit:.6f} (n={len(valid_df)})\n")
        f.write(f"test_2023_hit_rate={test_hit:.6f} (n={len(test_df)})\n")
    print(f"[Done] wrote {args.out_dir}/results.txt")


if __name__ == "__main__":
    main()
