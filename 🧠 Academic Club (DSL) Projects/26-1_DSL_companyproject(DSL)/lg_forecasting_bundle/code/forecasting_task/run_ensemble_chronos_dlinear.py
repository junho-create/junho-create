"""
Ensemble experiment: DLinear+DoubleAdapt (current best, `logs/dlinear_final_compliant`,
weighted_hit_rate=0.5900) combined with the log-return Chronos-2 univariate forecast
(`run_chronos_logret.py`, dense daily-anchor rerun in `logs/chronos_logret_dense`).

Does NOT modify either upstream pipeline -- reads their saved prediction CSVs only.
Both already run on the same 2023 test grid (240 trading days x 100 tickers), so this
is a genuine same-sample-set comparison, not the sparse 900/24000-row overlap the
25-day-anchor Chronos run would have given.

Reported per PROGRESS.md's own lesson ("ensembling with a diverged/weak backbone
drags the average down") -- Chronos-logret alone is 0.5267, well below DLinear's
0.5900, so a blind 50/50 blend is expected to hurt. Tests two more targeted
combinations instead:
  1. equal-weight average of the two models' predicted log returns, then sign
     (a real blend, not a no-op)
  2. "news-important subset override": swap in Chronos's direction only where
     target_novelty_consensus>0 or related_drift_flag>0 (the same subset
     PROGRESS.md already found DLinear itself does meaningfully better on),
     keep DLinear's direction everywhere else

Usage
-----
    python -m forecasting_task.run_ensemble_chronos_dlinear
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from forecasting_task.preprocessing.panel import PanelConfig, build_panel  # noqa: E402

DLINEAR_LOGDIR = REPO_ROOT / "logs" / "dlinear_final_compliant"
CHRONOS_LOGDIR = REPO_ROOT / "logs" / "chronos_logret_dense"


def hit_rate(pred_close, anchor_close, true_close):
    valid = true_close.notna()
    hit = np.sign(pred_close[valid] - anchor_close[valid]) == np.sign(true_close[valid] - anchor_close[valid])
    return float(hit.mean()), int(valid.sum())


def main():
    import argparse
    import yaml
    with open(DLINEAR_LOGDIR / "args.yaml") as f:
        ref = argparse.Namespace(**yaml.safe_load(f))

    dlinear = pd.read_csv(DLINEAR_LOGDIR / "predictions.csv", parse_dates=["date"])
    dlinear = dlinear.rename(columns={"pred_close": "pred_close_dlinear"})

    chronos = pd.read_csv(CHRONOS_LOGDIR / "test_preds.csv", parse_dates=["anchor_date"])
    chronos = chronos.rename(columns={"anchor_date": "date", "pred_close": "pred_close_chronos"})
    chronos = chronos[["date", "ticker", "pred_close_chronos"]]

    merged = dlinear.merge(chronos, on=["date", "ticker"], how="inner")
    print(f"[Merge] dlinear rows={len(dlinear)} chronos rows={len(chronos)} overlap={len(merged)}")

    dlinear_hit, n = hit_rate(merged["pred_close_dlinear"], merged["anchor_close"], merged["true_close"])
    chronos_hit, _ = hit_rate(merged["pred_close_chronos"], merged["anchor_close"], merged["true_close"])
    print(f"[Baseline on overlap] DLinear alone hit_rate={dlinear_hit:.4f} (n={n})")
    print(f"[Baseline on overlap] Chronos-logret alone hit_rate={chronos_hit:.4f} (n={n})")

    # --- Strategy 1: equal-weight average of predicted log returns, then sign ---
    logret_dlinear = np.log(merged["pred_close_dlinear"] / merged["anchor_close"])
    logret_chronos = np.log(merged["pred_close_chronos"] / merged["anchor_close"])
    avg_logret = (logret_dlinear + logret_chronos) / 2
    pred_close_avg = merged["anchor_close"] * np.exp(avg_logret)
    avg_hit, _ = hit_rate(pred_close_avg, merged["anchor_close"], merged["true_close"])
    print(f"[Strategy 1] equal-weight log-return average hit_rate={avg_hit:.4f}")

    # --- Strategy 2: news-important subset override ---
    cfg = PanelConfig(
        seq_len=ref.seq_len, horizon=ref.horizon, n_text_components=ref.n_text_components,
        train_start=ref.train_start, train_end=ref.train_end, primary_emb_model=ref.primary_emb_model,
        use_target_novelty=ref.use_target_novelty, use_related_novelty=ref.use_related_novelty,
        novelty_window=ref.novelty_window, novelty_quantile=ref.novelty_quantile,
        novelty_consensus_frac=ref.novelty_consensus_frac,
    )
    df = pd.read_parquet(ref.data_csv)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel_df, raw_close, _, _, factor_num, seq_len, factor_cols = build_panel(
        df, ref.data_dir, cfg, cache_dir=ref.panel_cache_dir
    )
    feat = panel_df["feature"]
    news_cols = [c for c in ("target_novelty_consensus", "related_drift_flag") if c in factor_cols]
    last_idx = {c: (factor_cols.index(c) + 1) * seq_len - 1 for c in news_cols}
    news_df = pd.DataFrame({c: feat.iloc[:, last_idx[c]].to_numpy() for c in news_cols}, index=panel_df.index)
    news_df = news_df.reset_index().rename(columns={"datetime": "date", "instrument": "ticker"})
    is_news_important = (news_df[news_cols] > 0).any(axis=1)
    news_df["is_news_important"] = is_news_important
    merged2 = merged.merge(news_df[["date", "ticker", "is_news_important"]], on=["date", "ticker"], how="left")
    merged2["is_news_important"] = merged2["is_news_important"].fillna(False)
    print(f"[NewsSubset] {merged2['is_news_important'].mean():.4f} of overlap rows flagged news-important")

    pred_close_override = np.where(
        merged2["is_news_important"], merged2["pred_close_chronos"], merged2["pred_close_dlinear"]
    )
    override_hit, _ = hit_rate(pd.Series(pred_close_override), merged2["anchor_close"], merged2["true_close"])
    print(f"[Strategy 2] news-important-subset Chronos override hit_rate={override_hit:.4f}")

    # subset-only breakdown, for interpretability
    sub = merged2[merged2["is_news_important"]]
    if len(sub) > 0:
        dlinear_sub_hit, n_sub = hit_rate(sub["pred_close_dlinear"], sub["anchor_close"], sub["true_close"])
        chronos_sub_hit, _ = hit_rate(sub["pred_close_chronos"], sub["anchor_close"], sub["true_close"])
        print(f"[NewsSubset only, n={n_sub}] DLinear={dlinear_sub_hit:.4f} Chronos-logret={chronos_sub_hit:.4f}")

    out_dir = REPO_ROOT / "logs" / "ensemble_chronos_logret_dlinear"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.txt", "w") as f:
        f.write(f"overlap_n={n}\n")
        f.write(f"dlinear_alone_hit_rate={dlinear_hit:.6f}\n")
        f.write(f"chronos_logret_alone_hit_rate={chronos_hit:.6f}\n")
        f.write(f"strategy1_logret_average_hit_rate={avg_hit:.6f}\n")
        f.write(f"strategy2_news_subset_override_hit_rate={override_hit:.6f}\n")
    print(f"[Done] wrote {out_dir}/results.txt")


if __name__ == "__main__":
    main()
