"""
V7: Text-to-Signal 5-Channel Model (Support PatchTST and DLinear)
Phase 1: Pretrain a Text Classifier to predict target direction.
Phase 2: Use the Text Classifier to generate a 1D 'Signal' and append to OHLC, feeding 5 channels to backbone.
"""

import argparse
import contextlib
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef

FORECASTING_DIR = Path(__file__).resolve().parent
REPO_ROOT = FORECASTING_DIR.parent
if str(FORECASTING_DIR) not in sys.path: sys.path.insert(0, str(FORECASTING_DIR))
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from forecasting_task.run_cascade_film import EmbeddingStore, LGAIDataset, LEVEL_PREFIXES, set_seed, _prepare_batch
from forecasting_task.run_cascade_film import _parse_str_list, _parse_int_list

# EmbeddingStore.get이 싱글턴이라 같은 프로세스에서 다른 emb_path를 무시하는 버그 수정
EmbeddingStore._instances = {}

@classmethod
def _get_by_path(cls, emb_path, valid_ids=None):
    if emb_path not in cls._instances:
        cls._instances[emb_path] = cls(emb_path, valid_ids)
    return cls._instances[emb_path]

EmbeddingStore.get = _get_by_path

PRICE_COLS = ["open", "high", "low", "close"]


class TextClassifier(nn.Module):
    """
    Attention-based text classifier.
    - Projects each level embedding to hidden dim
    - Adds learnable level-type embeddings
    - Runs Transformer encoder across levels (cross-level attention)
    - Mean-pools to get text summary vector
    - dir_head:      scalar direction logit  (for Phase 1 / aux loss)
    - temporal_head: [seq_len] per-timestep signal (replaces constant expand)
    """
    def __init__(self, text_dim, level_indices, hidden=256, seq_len=64):
        super().__init__()
        self.level_indices = level_indices
        n = len(level_indices)
        self.proj = nn.Linear(text_dim, hidden)
        self.level_embed = nn.Embedding(n, hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=4, dim_feedforward=hidden * 2,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.dir_head = nn.Linear(hidden, 1)
        self.temporal_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, seq_len),
        )

    def forward(self, level_embs):
        active = level_embs[:, self.level_indices, :]  # [B, n_levels, emb_dim]
        B, n, _ = active.shape
        x = self.proj(active)                                          # [B, n_levels, hidden]
        ids = torch.arange(n, device=x.device)
        x = x + self.level_embed(ids).unsqueeze(0)                    # add level-type embed
        x = self.encoder(x)                                            # [B, n_levels, hidden]
        pooled = x.mean(dim=1)                                         # [B, hidden]
        dir_logit = self.dir_head(pooled)                              # [B, 1]
        p_temporal = torch.sigmoid(self.temporal_head(pooled))        # [B, seq_len]
        return dir_logit, p_temporal


class V7Model(nn.Module):
    """
    5-channel model with improved text signal injection.
    joint=True  → end-to-end training (no Phase 1, text_clf is NOT frozen)
    joint=False → Phase 1 pretrained text_clf is frozen (original behaviour)
    Signal is now per-timestep [B, L, 1] instead of a constant scalar.
    """
    def __init__(self, configs, text_clf, model_type="PatchTST", joint=False):
        super().__init__()
        self.text_clf = text_clf
        self.joint = joint
        if not joint:
            for p in self.text_clf.parameters():
                p.requires_grad = False
            self.text_clf.eval()

        self.pred_len = configs.pred_len
        self.model_type = model_type

        configs.enc_in = 5
        configs.c_out = 5
        configs.task_name = 'short_term_forecast'
        configs.output_attention = False
        configs.factor = 1
        configs.activation = "relu"
        if not hasattr(configs, 'moving_avg'):
            configs.moving_avg = 25

        if model_type.lower() == "dlinear":
            from forecasting_task.models import DLinear
            self.backbone = DLinear.Model(configs)
        else:
            from forecasting_task.models import PatchTST
            self.backbone = PatchTST.Model(configs, patch_len=configs.patch_len, stride=configs.stride)

    def forward(self, price_x, x_mark, dec_inp, y_mark, level_embs):
        B, L, _ = price_x.shape
        ctx = contextlib.nullcontext() if self.joint else torch.no_grad()
        with ctx:
            dir_logit, p_temporal = self.text_clf(level_embs)  # [B,1], [B,L]

        active = level_embs[:, self.text_clf.level_indices, :]
        has_text = (active.abs().sum(dim=(1, 2)) > 1e-6).float().view(B, 1, 1)

        # Per-timestep signal; fallback to 0.5 when no text available
        p_seq = p_temporal.unsqueeze(-1)                              # [B, L, 1]
        p_seq = p_seq * has_text + 0.5 * (1 - has_text)

        x_enc_5 = torch.cat([price_x, p_seq], dim=-1)
        dec_inp_5 = torch.cat([dec_inp, torch.zeros(B, dec_inp.shape[1], 1, device=dec_inp.device)], dim=-1)
        out_5 = self.backbone(x_enc_5, x_mark, dec_inp_5, y_mark)
        return out_5[:, :, :4], torch.sigmoid(dir_logit)  # OHLC + direction prob


class NoTextModel(nn.Module):
    def __init__(self, configs, model_type="PatchTST"):
        super().__init__()
        import copy
        cfg = copy.deepcopy(configs)
        cfg.enc_in = 4
        cfg.c_out = 4
        cfg.task_name = 'short_term_forecast'
        cfg.output_attention = False
        cfg.factor = 1
        cfg.activation = "relu"
        if not hasattr(cfg, 'moving_avg'):
            cfg.moving_avg = 25
        if model_type.lower() == "dlinear":
            from forecasting_task.models import DLinear
            self.backbone = DLinear.Model(cfg)
        else:
            from forecasting_task.models import PatchTST
            self.backbone = PatchTST.Model(cfg, patch_len=cfg.patch_len, stride=cfg.stride)

    def forward(self, price_x, x_mark, dec_inp, y_mark, level_embs=None):
        return self.backbone(price_x, x_mark, dec_inp, y_mark)


def train_text_classifier(train_dl, val_dl, text_dim, level_indices, args, device):
    print("\n=== Phase 1: Training Text-to-Signal Classifier ===")
    model = TextClassifier(text_dim, level_indices, seq_len=args.seq_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    best_acc = 0.0
    best_state = None

    for ep in range(1, args.text_epochs + 1):
        model.train()
        train_loss, train_acc, cnt = 0.0, 0.0, 0
        for batch in train_dl:
            price_x, level_embs, _, _, _, gt = _prepare_batch(batch, device, args.pred_len, args.label_len)
            target = (gt[:, -1, 3] > price_x[:, -1, 3]).float().unsqueeze(1)
            active_embs = level_embs[:, model.level_indices, :]
            has_text = (active_embs.abs().sum(dim=(1, 2)) > 1e-6).float().unsqueeze(1)

            opt.zero_grad()
            logits, _ = model(level_embs)   # dir_logit, temporal (ignore temporal in Phase 1)
            loss_all = criterion(logits, target)
            loss = (loss_all * has_text).sum() / has_text.sum().clamp_min(1.0)
            loss.backward()
            opt.step()

            preds = (torch.sigmoid(logits) > 0.5).float()
            acc = ((preds == target).float() * has_text).sum().item()
            train_acc += acc
            cnt += has_text.sum().item()

        train_acc /= max(cnt, 1)

        model.eval()
        val_acc, v_cnt = 0.0, 0
        with torch.no_grad():
            for batch in val_dl:
                price_x, level_embs, _, _, _, gt = _prepare_batch(batch, device, args.pred_len, args.label_len)
                target = (gt[:, -1, 3] > price_x[:, -1, 3]).float().unsqueeze(1)
                active_embs = level_embs[:, model.level_indices, :]
                has_text = (active_embs.abs().sum(dim=(1, 2)) > 1e-6).float().unsqueeze(1)

                logits, _ = model(level_embs)
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_acc += ((preds == target).float() * has_text).sum().item()
                v_cnt += has_text.sum().item()

        val_acc /= max(v_cnt, 1)
        print(f"  [Text Clf Ep{ep:02d}] Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}", flush=True)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"[Text Clf] Best Val Acc: {best_acc:.4f}", flush=True)
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model


def run_epoch_v7(loader, model, args, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    # Keep text_clf in eval mode unless joint training (Phase 2 frozen)
    if not is_train and isinstance(model, V7Model) and not model.joint:
        model.text_clf.eval()
    mse_sum = mae_sum = close_mse_sum = close_mae_sum = cnt = 0.0
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        all_pred_ret = []
        all_true_ret = []
        all_price_x_last = []
        all_out = []
        all_gt = []

        for batch in loader:
            price_x, level_embs, x_mark, y_mark, dec_inp, gt = _prepare_batch(
                batch, device, args.pred_len, args.label_len
            )
            if is_train:
                optimizer.zero_grad()

            raw = model(price_x, x_mark, dec_inp, y_mark, level_embs)
            out, p_dir = raw if isinstance(raw, tuple) else (raw, None)

            close_idx = 3
            mse = torch.mean((out - gt) ** 2)
            mae = torch.mean(torch.abs(out - gt))
            close_mse = torch.mean((out[:, :, close_idx] - gt[:, :, close_idx]) ** 2)
            close_mae = torch.mean(torch.abs(out[:, :, close_idx] - gt[:, :, close_idx]))

            if is_train:
                loss = mse
                # Joint mode: add auxiliary direction BCE loss
                if p_dir is not None and getattr(args, 'joint', False):
                    active = level_embs[:, model.text_clf.level_indices, :]
                    has_text = (active.abs().sum(dim=(1, 2)) > 1e-6).float().unsqueeze(1)
                    target_dir = (gt[:, -1, 3] > price_x[:, -1, 3]).float().unsqueeze(1)
                    dir_loss = F.binary_cross_entropy(p_dir, target_dir, reduction='none')
                    dir_loss = (dir_loss * has_text).sum() / has_text.sum().clamp_min(1.0)
                    loss = mse + args.aux_weight * dir_loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            bs = out.shape[0]
            mse_sum += mse.item() * bs
            mae_sum += mae.item() * bs
            close_mse_sum += close_mse.item() * bs
            close_mae_sum += close_mae.item() * bs
            cnt += bs
            
            # For new metrics
            if not is_train:
                px_last = price_x[:, -1, 3].detach().cpu().numpy()
                out_last = out[:, -1, 3].detach().cpu().numpy()
                gt_last = gt[:, -1, 3].detach().cpu().numpy()
                
                # Returns = (Future - Current) / Current
                # Since price_norm can have 0 or negative after StandardScaler, we should be careful.
                # However, predicting direction is just out_last > px_last.
                pred_ret = out_last - px_last
                true_ret = gt_last - px_last
                
                all_pred_ret.extend(pred_ret)
                all_true_ret.extend(true_ret)
                
                if len(all_out) < 16: # Keep a few samples for plotting
                    all_out.append(out[0, :, 3].detach().cpu().numpy())
                    all_gt.append(gt[0, :, 3].detach().cpu().numpy())
                    all_price_x_last.append(px_last[0])

    n = max(cnt, 1)
    results = {"mse": mse_sum / n, "mae": mae_sum / n, "close_mse": close_mse_sum / n, "close_mae": close_mae_sum / n}
    
    if not is_train and len(all_pred_ret) > 0:
        pred_ret = np.array(all_pred_ret)
        true_ret = np.array(all_true_ret)
        
        pred_dir = (pred_ret > 0).astype(int)
        true_dir = (true_ret > 0).astype(int)
        
        da = np.mean(pred_dir == true_dir)
        f1 = f1_score(true_dir, pred_dir, zero_division=0)
        mcc = matthews_corrcoef(true_dir, pred_dir)
        rank_ic, _ = spearmanr(pred_ret, true_ret)
        if np.isnan(rank_ic):
            rank_ic = 0.0
            
        # Virtual Portfolio Return (Long/Short)
        # Position is +1 if pred_dir==1 else -1. Return is true_ret.
        # But wait, true_ret is based on scaled prices. It's a proxy for actual return.
        port_return = np.mean((pred_dir * 2 - 1) * true_ret)
        
        results.update({"da": da, "f1": f1, "mcc": mcc, "rank_ic": rank_ic, "port_return": port_return})
        
        # Save plot
        plot_dir = Path("plots")
        plot_dir.mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(4, 4, figsize=(16, 12))
        axes = axes.flatten()
        for i, (out_arr, gt_arr, px0) in enumerate(zip(all_out[:16], all_gt[:16], all_price_x_last[:16])):
            axes[i].plot(np.arange(1, len(gt_arr)+1), gt_arr, label="True", color="black")
            axes[i].plot(np.arange(1, len(out_arr)+1), out_arr, label="Pred", color="blue", linestyle="--")
            axes[i].scatter([0], [px0], color="red", label="Current")
            axes[i].set_title(f"Sample {i}")
            if i == 0:
                axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(plot_dir / f"eval_plot_{args.model_type}.png")
        plt.close()

    return results


_DATASET_CACHE = {}

def make_datasets_v7(args, tickers):
    emb_name = args.emb_path.split('/')[-1]
    def make_ds(flag):
        # int() ensures ema_span=0 and ema_span=5 are distinct keys (cache bug fix)
        key = (args.emb_path, int(args.ema_span), flag)
        if key not in _DATASET_CACHE:
            print(f"[DatasetCache] Building ({emb_name}, ema={args.ema_span}, {flag}) ...", flush=True)
            _DATASET_CACHE[key] = LGAIDataset(
                train_path=args.train_path, emb_path=args.emb_path,
                flag=flag, tickers=tickers,
                seq_len=args.seq_len, label_len=args.label_len,
                pred_len=args.pred_len, test_path=args.test_path,
                forecast_gap=args.forecast_gap, ema_span=int(args.ema_span),
            )
        else:
            print(f"[DatasetCache] Reusing ({emb_name}, ema={args.ema_span}, {flag})", flush=True)
        return _DATASET_CACHE[key]
    return make_ds("train"), make_ds("val"), make_ds("test")


def run_single(args, tickers, model_type_override=None, emb_path_override=None, temporal_override=None):
    import copy
    args = copy.deepcopy(args)

    if model_type_override:
        args.model_type = model_type_override

    # 시간 집계 방식: ema(최근 가중 EMA) 또는 last(최신 하루만)
    temporal = temporal_override or args.temporal
    if temporal == "last":
        args.ema_span = 0
    # "ema"는 args.ema_span 그대로 사용

    use_text = (emb_path_override != "notext")
    if use_text and emb_path_override:
        args.emb_path = emb_path_override
    elif not use_text:
        args.emb_path = getattr(args, "bert_path", args.emb_path)

    # 사용할 레벨 인덱스 계산 (예: ["macro","sector"] → [0, 1])
    level_names = [lv[0] for lv in LEVEL_PREFIXES]
    level_indices = [level_names.index(lv) for lv in args.use_levels if lv in level_names]

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.logdir, exist_ok=True)

    if use_text:
        import pandas as pd
        from forecasting_task.run_cascade_film import _get_level_id_cols
        print("Extracting valid text_ids for memory efficiency...")
        df_train = pd.read_parquet(args.train_path)
        df_test = pd.read_parquet(args.test_path) if getattr(args, 'test_path', None) and Path(args.test_path).exists() else pd.DataFrame()
        df_all = pd.concat([df_train, df_test]) if not df_test.empty else df_train
        df_t = df_all[df_all["ticker"].isin(tickers)]
        level_cols = [c for lname, cols in _get_level_id_cols(list(df_t.columns)) for c in cols]
        valid_ids = set()
        for c in level_cols:
            valid_ids.update(df_t[c].dropna().unique().tolist())
        
        print(f"Extracted {len(valid_ids)} unique text IDs.")
        EmbeddingStore.get(args.emb_path, valid_ids=list(valid_ids))

    train_ds, val_ds, test_ds = make_datasets_v7(args, tickers)
    emb_dim = train_ds.emb_dim

    if not use_text:
        emb_type = "NoText"
    elif "linq" in args.emb_path.lower():
        emb_type = "Linq"
    else:
        emb_type = "Bert"

    temporal_label = f"last" if args.ema_span == 0 else f"ema{args.ema_span}"
    levels_label = "+".join(args.use_levels)
    print(f"\n[Run Config] Model: {args.model_type} | Emb: {emb_type} ({emb_dim}d) | Levels: {levels_label} | Temporal: {temporal_label} | Seed: {args.seed}")

    mk = lambda ds, shuf: DataLoader(ds, batch_size=args.batch_size,
                                     shuffle=shuf, num_workers=args.num_workers,
                                     drop_last=shuf, pin_memory=True)
    train_dl = mk(train_ds, True)
    val_dl = mk(val_ds, False)
    test_dl = mk(test_ds, False)

    if use_text:
        if getattr(args, 'joint', False):
            print(f"\n=== Joint Training: 5-Channel V7 Model ({args.model_type}) ===")
            text_clf = TextClassifier(emb_dim, level_indices, seq_len=args.seq_len).to(device)
            model = V7Model(args, text_clf, args.model_type, joint=True).to(device)
        else:
            text_clf = train_text_classifier(train_dl, val_dl, emb_dim, level_indices, args, device)
            print(f"\n=== Phase 2: Training 5-Channel V7 Model ({args.model_type}) ===")
            model = V7Model(args, text_clf, args.model_type, joint=False).to(device)
    else:
        print(f"\n=== Training NoText Model ({args.model_type}) ===")
        model = NoTextModel(args, args.model_type).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_mse = float("inf")
    patience_cnt = 0
    best_state = None

    for ep in range(1, args.num_epoch + 1):
        tr = run_epoch_v7(train_dl, model, args, device, opt)
        with torch.no_grad():
            va = run_epoch_v7(val_dl, model, args, device)

        print(f"  [V7 Ep{ep:02d}] train_mse={tr['mse']:.5f}  val_mse={va['mse']:.5f}  val_close_mse={va['close_mse']:.5f}")

        if va["mse"] < best_val_mse:
            best_val_mse = va["mse"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"  [EarlyStop] ep={ep}")
                break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    test_avg = run_epoch_v7(test_dl, model, args, device)
    
    test_da = test_avg.get("da", 0.0)
    test_f1 = test_avg.get("f1", 0.0)
    test_mcc = test_avg.get("mcc", 0.0)
    test_rank_ic = test_avg.get("rank_ic", 0.0)
    test_port_return = test_avg.get("port_return", 0.0)
    
    print(f"[Result] V7 ({args.model_type}_{emb_type}): test_mse={test_avg['mse']:.5f} close_mse={test_avg['close_mse']:.5f}")
    print(f"         Directional Metrics: DA={test_da:.4f}, F1={test_f1:.4f}, MCC={test_mcc:.4f}, Rank_IC={test_rank_ic:.4f}, Port_Ret={test_port_return:.4f}")

    row = {
        "fusion": "v7_notext" if not use_text else "v7_signal",
        "model_type": args.model_type, "emb_type": emb_type,
        "levels": "+".join(args.use_levels),
        "temporal": temporal_label,
        "seed": args.seed, "tickers": len(tickers), "stop_epoch": ep,
        "val_mse": round(best_val_mse, 6),
        "test_mse": round(test_avg["mse"], 6),
        "test_mae": round(test_avg["mae"], 6),
        "test_close_mse": round(test_avg["close_mse"], 6),
        "test_close_mae": round(test_avg["close_mae"], 6),
        "test_da": round(test_da, 4),
        "test_f1": round(test_f1, 4),
        "test_mcc": round(test_mcc, 4),
        "test_rank_ic": round(test_rank_ic, 4),
        "test_port_return": round(test_port_return, 4),
    }
    p = Path(args.results_csv)
    p.parent.mkdir(parents=True, exist_ok=True)
    hdr = not p.exists()
    with p.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if hdr:
            w.writeheader()
        w.writerow(row)
    print(f"[Result] saved to {args.results_csv}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_path", default="/workspace/Dataset/train.parquet")
    p.add_argument("--test_path", default="/workspace/Dataset/test.parquet")
    p.add_argument("--emb_path", default="/workspace/Dataset/linq_textemb.parquet")
    p.add_argument("--model_type", default="PatchTST", choices=["PatchTST", "DLinear"])
    p.add_argument("--num_tickers", type=int, default=0)
    p.add_argument("--ticker_offset", type=int, default=0)

    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--label_len", type=int, default=0)
    p.add_argument("--pred_len", type=int, default=1)
    p.add_argument("--forecast_gap", type=int, default=19)
    p.add_argument("--ema_span", type=int, default=5,
                   help="EMA span for text aggregation (0=last day only, >0=recent-weighted EMA)")
    p.add_argument("--use_levels", type=_parse_str_list, default=["macro", "sector"],
                   help="Which text levels to use, e.g. macro,sector or macro,sector,target")
    p.add_argument("--temporal", default="ema", choices=["ema", "last", "both"],
                   help="ema: recent-weighted EMA (uses ema_span), last: most recent day only, both: run both")

    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--e_layers", type=int, default=2)
    p.add_argument("--d_ff", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--patch_len", type=int, default=8)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--moving_avg", type=int, default=25)

    p.add_argument("--text_epochs", type=int, default=15)
    p.add_argument("--num_epoch", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--seeds", type=_parse_int_list, default=None)

    p.add_argument("--bert_path", default="/workspace/Dataset/bert_textemb.parquet")
    p.add_argument("--linq_path", default="/workspace/Dataset/linq_textemb.parquet")

    p.add_argument("--logdir", default="logs/v7_combinations")
    p.add_argument("--results_csv", default="logs/v7_combinations_report.csv")
    p.add_argument("--run_grid", action="store_true")
    p.add_argument("--run_all", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="Skip combos already present in results_csv")
    p.add_argument("--joint", action="store_true",
                   help="End-to-end joint training (no Phase 1, text_clf trained with MSE+BCE)")
    p.add_argument("--aux_weight", type=float, default=0.1,
                   help="Weight for direction BCE aux loss in joint mode")
    return p.parse_args()


def run_grid(args):
    import pandas as pd
    df = pd.read_parquet(args.train_path)
    all_tk = sorted(df["ticker"].unique().tolist())
    n = args.num_tickers if args.num_tickers > 0 else len(all_tk)
    tickers = all_tk[args.ticker_offset:args.ticker_offset + n]

    for seed in args.seeds:
        import copy
        a = copy.deepcopy(args)
        a.seed = seed
        run_single(a, tickers)


def run_all_combinations(args, tickers):
    combos = [
        ("PatchTST", args.bert_path),
        ("PatchTST", args.linq_path),
        ("PatchTST", "notext"),
        ("DLinear",  args.bert_path),
        ("DLinear",  args.linq_path),
        ("DLinear",  "notext"),
    ]
    temporals = ["ema", "last"] if args.temporal == "both" else [args.temporal]

    # 이미 완료된 조합 로드 (--resume 플래그 시)
    done_keys = set()
    if args.resume and Path(args.results_csv).exists():
        import pandas as pd
        df_done = pd.read_csv(args.results_csv)
        for _, r in df_done.iterrows():
            done_keys.add((r["model_type"], r["emb_type"], r["temporal"]))
        print(f"[Resume] Already done: {len(done_keys)} combos → {done_keys}")

    for temporal in temporals:
        for model_type, emb_path in combos:
            emb_label = "NoText" if emb_path == "notext" else ("Linq" if "linq" in emb_path.lower() else "Bert")
            t_label = "last" if temporal == "last" else f"ema{args.ema_span}"
            if (model_type, emb_label, t_label) in done_keys:
                print(f"[Resume] Skip: {model_type} + {emb_label} | temporal={t_label}")
                continue
            print(f"\n{'='*60}")
            print(f"[Combo] {model_type} + {emb_label} | temporal={t_label}")
            print(f"{'='*60}")
            run_single(args, tickers, model_type_override=model_type,
                       emb_path_override=emb_path, temporal_override=temporal)


def main():
    args = parse_args()
    os.makedirs(args.logdir, exist_ok=True)

    import pandas as pd
    df = pd.read_parquet(args.train_path)
    all_tk = sorted(df["ticker"].unique().tolist())
    n = args.num_tickers if args.num_tickers > 0 else len(all_tk)
    tickers = all_tk[args.ticker_offset:args.ticker_offset + n]

    if args.run_all:
        run_all_combinations(args, tickers)
    elif args.run_grid:
        if args.seeds is None:
            args.seeds = [args.seed]
        run_grid(args)
    else:
        run_single(args, tickers)


if __name__ == "__main__":
    main()
