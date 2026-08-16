"""
Step1: pluggable time-series backbones for the DoubleAdapt panel.

Every backbone exposes the interface DoubleAdapt's `ForecastModel` expects:
    forward(x: [batch, factor_num*seq_len]) -> [batch]        (need_permute=False)

`GRURegressor` is the original hand-rolled backbone (moved here unchanged from
run_DoubleAdapt.py). `TSLibBackbone` adapts forecasting_task/models/*.py
(Time-Series-Library-style architectures already in this repo, used by
run_cascade_film.py) behind that same interface, so DoubleAdapt's
meta-learning wrapper never has to change when the backbone does. Pick the
backbone based on which one tracks actual price fluctuation/regime-change in
a pred-vs-actual plot instead of going flat -- that's the fund-manager
"Step1" validation criterion, not just lowest val loss.

Add a new architecture by adding one line to TSLIB_MODEL_REGISTRY, as long as
its `Model.forward(x_enc, x_mark_enc, x_dec, x_mark_dec)` follows the usual
Time-Series-Library convention.
"""
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

FORECASTING_DIR = Path(__file__).resolve().parent
REPO_ROOT = FORECASTING_DIR.parent
# forecasting_task/models/*.py do bare `from layers... import ...`, so
# forecasting_task/ itself (not just REPO_ROOT) must be on sys.path -- same
# reasoning as DoubleAdapt/src in run_DoubleAdapt.py. Matches run_cascade_film.py.
for _p in (str(FORECASTING_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TSLIB_MODEL_REGISTRY = {
    "dlinear": "forecasting_task.models.DLinear",
    "patchtst": "forecasting_task.models.PatchTST",
    "itransformer": "forecasting_task.models.iTransformer",
    "timesnet": "forecasting_task.models.TimesNet",
}


def _import_tslib_model(dotted_path: str):
    """`importlib.import_module` with the bare `utils` name-collision worked around.

    run_DoubleAdapt.py's `import model as da_model` (DoubleAdapt/src/model.py) binds
    sys.modules['utils'] to the flat DoubleAdapt/src/utils.py *module*. Some
    forecasting_task/models/*.py (PatchTST, iTransformer, via
    layers/SelfAttention_Family.py) bare-import `from utils.masking import ...`,
    expecting the forecasting_task/utils/ *package* instead -- same name, two
    different things on sys.path. Evict the cached entry so Python re-resolves
    `utils` fresh (finding the real package, since FORECASTING_DIR precedes
    DOUBLEADAPT_SRC on sys.path), then restore it so `da_utils`/`da_model` (which
    already hold direct references to their imported symbols) keep working.
    """
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "utils" or k.startswith("utils.")}
    try:
        return importlib.import_module(dotted_path)
    finally:
        for k in list(sys.modules):
            if k == "utils" or k.startswith("utils."):
                del sys.modules[k]
        sys.modules.update(saved)


class GRURegressor(nn.Module):
    """[batch, factor_num*seq_len] (time-major flatten, need_permute=False) -> [batch] scalar."""

    def __init__(self, factor_num, seq_len, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()
        self.factor_num = factor_num
        self.seq_len = seq_len
        self.gru = nn.GRU(
            input_size=factor_num,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.reshape(x.shape[0], self.seq_len, self.factor_num)
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class TSLibBackbone(nn.Module):
    """Wraps a Time-Series-Library `Model` for the DoubleAdapt scalar-forecast interface.

    `x_mark_enc`/`x_mark_dec` are passed as None -- the panel carries no explicit
    calendar features today, and for iTransformer in particular a zero-valued
    x_mark would be embedded as spurious extra "variate" tokens rather than
    being ignored, so None (skip the branch) is the correct no-op, not zeros.
    """

    def __init__(self, model_key, factor_num, seq_len, d_model=64, n_heads=4, e_layers=2,
                 d_ff=128, dropout=0.1):
        super().__init__()
        module = _import_tslib_model(TSLIB_MODEL_REGISTRY[model_key])
        pred_len = 1
        configs = SimpleNamespace(
            task_name="long_term_forecast", seq_len=seq_len, label_len=0, pred_len=pred_len,
            enc_in=factor_num, dec_in=factor_num, c_out=factor_num,
            d_model=d_model, n_heads=n_heads, e_layers=e_layers, d_layers=1, d_ff=d_ff,
            dropout=dropout, factor=1, activation="gelu", output_attention=False,
            moving_avg=25, embed="timeF", freq="d", num_class=1,
            top_k=5, num_kernels=6, distil=True,
        )
        self.backbone = module.Model(configs)
        self.factor_num = factor_num
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.head = nn.Linear(factor_num, 1)

    def forward(self, x):
        x_enc = x.reshape(x.shape[0], self.seq_len, self.factor_num)
        B = x_enc.shape[0]
        x_dec = torch.zeros(B, self.pred_len, self.factor_num, device=x.device, dtype=x.dtype)
        out = self.backbone(x_enc, None, x_dec, None)  # [B, pred_len, factor_num]
        return self.head(out[:, -1, :]).squeeze(-1)


def build_backbone(name: str, factor_num: int, seq_len: int, **kwargs) -> nn.Module:
    name = name.lower()
    if name == "gru":
        return GRURegressor(
            factor_num, seq_len,
            hidden_size=kwargs.get("hidden_size", 64),
            num_layers=kwargs.get("num_layers", 2),
            dropout=kwargs.get("dropout", 0.0),
        )
    if name in TSLIB_MODEL_REGISTRY:
        return TSLibBackbone(
            name, factor_num, seq_len,
            d_model=kwargs.get("d_model", 64), n_heads=kwargs.get("n_heads", 4),
            e_layers=kwargs.get("e_layers", 2), d_ff=kwargs.get("d_ff", 128),
            dropout=kwargs.get("dropout", 0.1),
        )
    raise ValueError(f"Unknown backbone {name!r}. Choose from: gru, {list(TSLIB_MODEL_REGISTRY)}")
