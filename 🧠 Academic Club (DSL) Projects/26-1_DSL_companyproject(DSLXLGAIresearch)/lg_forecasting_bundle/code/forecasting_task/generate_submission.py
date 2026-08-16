import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Add root to sys path
sys.path.insert(0, '/root/LG-AI')

# Import components from our training script
from forecasting_task.run_kaggle_weighted import (
    parse_args, set_seed, compute_consensus_clusters, precompute_cluster_embeddings,
    EmbeddingStore, TextClassifier, V7Model, LEVEL_PREFIXES, _get_level_id_cols
)

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading raw data...")
    if args.data_path.endswith('.csv'):
        df = pd.read_csv(args.data_path)
    else:
        df = pd.read_parquet(args.data_path)
    df["date"] = pd.to_datetime(df["date"])
    tickers = sorted(df["ticker"].unique().tolist())
    
    # 1. Fit Global Scaler
    print("Fitting global scaler...")
    df_train = df[df["date"] < "2022-01-01"]
    scaler = StandardScaler()
    scaler.fit(df_train[["open", "high", "low", "close"]].values)
    
    # 2. Re-create Embeddings & Clusters
    ticker_to_cluster = compute_consensus_clusters(df, args.emb_dir, "2022-01-01", args.n_clusters)
    target_cols = [c for c in df.columns if c.startswith("targetCompany_category") or c.startswith("lseg_news")]
    valid_ids_all = set(df[target_cols].values.flatten().tolist())
    primary_store = EmbeddingStore(os.path.join(args.emb_dir, args.primary_emb), valid_ids=valid_ids_all)
    cluster_emb_dict = precompute_cluster_embeddings(df, primary_store, ticker_to_cluster)
    
    # 3. Load Model
    print("Loading trained model...")
    level_names = [lv[0] for lv in LEVEL_PREFIXES]
    level_indices = [level_names.index(l) for l in ["macro", "sector", "related", "target"]]
    
    text_clf = TextClassifier(
        primary_store.emb_dim, level_indices, seq_len=args.seq_len,
        hidden=args.text_hidden, num_layers=args.text_layers, nhead=args.text_heads
    ).to(device)
    model = V7Model(args, text_clf, joint=args.joint).to(device)
    
    model_path = '/root/LG-AI/best_kaggle_model.pth'
    if not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}. Using untrained weights.")
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 4. Define Target Dates (2022-01-01 to 2023-12-31, Weekdays only)
    target_dates = pd.date_range("2022-01-01", "2023-12-31", freq="B").strftime("%Y-%m-%d").tolist()
    print(f"Total target dates (weekdays): {len(target_dates)}")
    
    # 5. Generate Predictions per Ticker
    results = []
    
    for ticker in tqdm(tickers, desc="Generating Predictions"):
        df_t = df[df["ticker"] == ticker].reset_index(drop=True)
        dates = df_t["date"]
        n = len(df_t)
        
        price_norm = scaler.transform(df_t[["open", "high", "low", "close"]].values).astype(np.float32)
        stamp = np.stack([dates.dt.month, dates.dt.day, dates.dt.weekday], axis=1).astype(np.float32)
        
        cluster_id = ticker_to_cluster[ticker]
        le_arr = np.zeros((n, len(level_names), primary_store.emb_dim), dtype=np.float32)
        
        level_cols = _get_level_id_cols(list(df.columns))
        
        for row_i in range(n):
            d = dates.iloc[row_i]
            for li, (lname, cols) in enumerate(level_cols):
                if lname == "sector":
                    le_arr[row_i, li] = cluster_emb_dict.get((d, cluster_id), np.zeros(primary_store.emb_dim))
                else:
                    ids = df_t.iloc[row_i][cols].tolist() if cols else []
                    le_arr[row_i, li] = primary_store.lookup_mean(ids)
                    
        if args.ema_span > 0:
            alpha = 2.0 / (args.ema_span + 1)
            for t in range(1, n):
                le_arr[t] = alpha * le_arr[t] + (1 - alpha) * le_arr[t-1]
                
        ticker_preds = {}
        for idx in range(n - args.seq_len - args.pred_len + 1):
            s = idx
            se = s + args.seq_len
            re = se - args.label_len
            ree = se + args.pred_len
            
            target_date_str = dates.iloc[ree - 1].strftime("%Y-%m-%d")
            if target_date_str < "2022-01-01" or target_date_str > "2023-12-31":
                continue
                
            px = torch.from_numpy(price_norm[s:se]).unsqueeze(0).to(device)
            py = torch.from_numpy(price_norm[re:ree]).unsqueeze(0).to(device)
            le = torch.from_numpy(le_arr[se - 1]).unsqueeze(0).to(device)
            xm = torch.from_numpy(stamp[s:se]).unsqueeze(0).to(device)
            ym = torch.from_numpy(stamp[re:ree]).unsqueeze(0).to(device)
            
            dec_inp = torch.zeros_like(py[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([py[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
            
            with torch.no_grad():
                out, _, _ = model(px, xm, dec_inp, ym, le)
            
            pred_norm = out[0, -1, 3].item()
            dummy = np.zeros((1, 4))
            dummy[0, 3] = pred_norm
            pred_close = scaler.inverse_transform(dummy)[0, 3]
            
            ticker_preds[target_date_str] = pred_close
            
        # Fallback logic: fill missing dates
        last_val = 100.0
        if not df_t.empty:
            last_val = df_t.iloc[-1]["close"]
            
        for t_date in target_dates:
            if t_date in ticker_preds:
                val = ticker_preds[t_date]
                last_val = val
            else:
                val = last_val
            
            results.append({
                "ID": f"{ticker}_{t_date}",
                "Close": val
            })
            
    # 6. Save Submission
    sub_df = pd.DataFrame(results)
    out_path = '/root/LG-AI/submission.csv'
    sub_df.to_csv(out_path, index=False)
    print(f"\n[Success] Generated {len(sub_df)} rows in {out_path}.")
    print(sub_df.head())

if __name__ == "__main__":
    main()
