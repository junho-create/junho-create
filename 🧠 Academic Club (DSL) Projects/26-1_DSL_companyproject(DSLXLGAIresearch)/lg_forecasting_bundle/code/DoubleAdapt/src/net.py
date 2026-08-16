import collections
import math

import torch

from torch import nn
from torch.nn import functional as F, init


def cosine(x1, x2, eps=1e-8):
    x1 = x1 / (torch.norm(x1, p=2, dim=-1, keepdim=True) + eps)
    x2 = x2 / (torch.norm(x2, p=2, dim=-1, keepdim=True) + eps)
    return x1 @ x2.transpose(0, 1)


# class LabelAdaptHead(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.weight = nn.Parameter(torch.empty(1))
#         self.bias = nn.Parameter(torch.ones(1) / 8)
#         init.uniform_(self.weight, 0.75, 1.25)
#
#     def forward(self, y, inverse=False):
#         if inverse:
#             return (y - self.bias) / (self.weight + 1e-9)
#         else:
#             return (self.weight + 1e-9) * y + self.bias

class LabelAdaptHeads(nn.Module):
    def __init__(self, num_head):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1, num_head))
        self.bias = nn.Parameter(torch.ones(1, num_head) / 8)
        init.uniform_(self.weight, 0.75, 1.25)

    def forward(self, y, inverse=False):
        if inverse:
            return (y.view(-1, 1) - self.bias) / (self.weight + 1e-9)
        else:
            return (self.weight + 1e-9) * y.view(-1, 1) + self.bias

class LabelAdapter(nn.Module):
    def __init__(self, x_dim, num_head=4, temperature=4, hid_dim=32):
        super().__init__()
        self.num_head = num_head
        self.linear = nn.Linear(x_dim, hid_dim, bias=False)
        self.P = nn.Parameter(torch.empty(num_head, hid_dim))
        init.kaiming_uniform_(self.P, a=math.sqrt(5))
        # self.heads = nn.ModuleList([LabelAdaptHead() for _ in range(num_head)])
        self.heads = LabelAdaptHeads(num_head)
        self.temperature = temperature

    def forward(self, x, y, inverse=False):
        v = self.linear(x.reshape(len(x), -1))
        gate = cosine(v, self.P)
        gate = torch.softmax(gate / self.temperature, -1)
        # return sum([gate[:, i] * self.heads[i](y, inverse=inverse) for i in range(self.num_head)])
        return (gate * self.heads(y, inverse=inverse)).sum(-1)


class FiLM(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.scale = nn.Parameter(torch.empty(in_dim))
        nn.init.uniform_(self.scale, 0.75, 1.25)

    def forward(self, x):
        return x * self.scale


class FeatureAdapter(nn.Module):
    def __init__(self, in_dim, num_head=4, temperature=4):
        super().__init__()
        self.num_head = num_head
        self.P = nn.Parameter(torch.empty(num_head, in_dim))
        init.kaiming_uniform_(self.P, a=math.sqrt(5))
        self.heads = nn.ModuleList([nn.Linear(in_dim, in_dim, bias=True) for _ in range(num_head)])
        self.temperature = temperature

    def forward(self, x):
        s_hat = torch.cat(
            [torch.cosine_similarity(x, self.P[i], dim=-1).unsqueeze(-1) for i in range(self.num_head)], -1,
        )
        # s_hat = cosine(x, self.P)
        s = torch.softmax(s_hat / self.temperature, -1).unsqueeze(-1)
        return x + sum([s[..., i, :] * self.heads[i](x) for i in range(self.num_head)])


class JointMSEDirLoss(nn.Module):
    """MSE (magnitude) + a BCE-on-sign auxiliary term (direction), added together.

    Reuses the single scalar regression output as its own direction logit
    rather than requiring a second output head -- the label's sign is the
    binary target. This gives a directional gradient signal on every training
    batch, which is far less noisy than trying to select checkpoints by a
    validation-set hit rate (that failed: too few validation samples ->
    high-variance estimate -> early stopping chased noise instead of signal).

    `pred`/`label` here are in STANDARDIZED label space (mean/std computed
    over the training period), not raw log-return space. The standardized
    label's zero point is shifted by `label_mean`, so sign(label) != sign(raw
    label) in general -- both the BCE target and logit are re-centered by
    `shift = label_mean / label_std` so the decision boundary matches the
    actual raw_return > 0 boundary the Weighted Hit Rate eval metric uses.
    """

    def __init__(self, aux_weight: float = 0.1, dir_scale: float = 5.0,
                 label_mean: float = 0.0, label_std: float = 1.0):
        super().__init__()
        self.aux_weight = aux_weight
        self.dir_scale = dir_scale
        self.shift = label_mean / label_std if label_std else 0.0
        self.mse = nn.MSELoss()

    def forward(self, pred, label, weight=None):
        """`weight`: optional per-sample tensor (same shape as pred/label). When given,
        both terms use a weighted mean instead of a plain mean -- e.g. to upweight
        samples flagged as "news-important" so the model is penalized more for
        getting those directions wrong. None (default) reproduces the original
        unweighted behavior exactly."""
        if weight is None:
            mse_loss = self.mse(pred, label)
        else:
            se = (pred - label) ** 2
            mse_loss = (se * weight).sum() / weight.sum().clamp_min(1e-8)
        if self.aux_weight <= 0:
            return mse_loss
        target_dir = (label + self.shift > 0).float()
        dir_logit = (pred + self.shift) * self.dir_scale
        if weight is None:
            dir_loss = F.binary_cross_entropy_with_logits(dir_logit, target_dir)
        else:
            dir_loss_per = F.binary_cross_entropy_with_logits(dir_logit, target_dir, reduction="none")
            dir_loss = (dir_loss_per * weight).sum() / weight.sum().clamp_min(1e-8)
        return mse_loss + self.aux_weight * dir_loss


class ForecastModel(nn.Module):
    def __init__(self, model: nn.Module, x_dim: int = None, lr: float = 0.001, weight_decay: float = 0,
                 need_permute: bool = False, aux_weight: float = 0.0, dir_scale: float = 5.0,
                 label_mean: float = 0.0, label_std: float = 1.0):
        """

        Args:
            model (nn.Module): the forecast model
            x_dim (int): the dimension of stock features (e.g., factor_num * time_series_length)
            lr (float): learning rate of forecast model
            weight_decay (float): L2 regularization of the (Adam) optimizer
            need_permute (bool): True when it requires time-series inputs to be shaped in [batch_size, factor_num * time_series_length] (e.g., in Qlib Alpha360)
            aux_weight (float): if > 0, uses JointMSEDirLoss (MSE + aux_weight * direction BCE)
                instead of plain MSE. 0.0 (default) = original behavior, unchanged.
            dir_scale (float): scales `pred` into a direction logit for the BCE term.
            label_mean/label_std: de-standardization constants so the BCE term's decision
                boundary lines up with the real raw_return > 0 boundary (only used if aux_weight > 0).
        """
        super().__init__()
        self.lr = lr
        # self.lr = task_config["model"]['kwargs']['lr']
        # Always route through JointMSEDirLoss: with aux_weight=0 and weight=None it reduces
        # to exactly nn.MSELoss()'s behavior, but it also accepts an optional per-sample
        # `weight` (see model.py's news_weight_mult), which nn.MSELoss() cannot.
        self.criterion = JointMSEDirLoss(
            aux_weight=aux_weight, dir_scale=dir_scale, label_mean=label_mean, label_std=label_std
        )
        self.model = model
        self.device = torch.device("cuda")
        self.need_permute = need_permute
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        if self.device is not None:
            self.to(self.device)

    def forward(self, X, model=None):
        """

        Args:
            X: [batch_size, x_dim]
            model: 

        Returns:
            predictions
        """
        if model is None:
            model = self.model
        if X.dim() == 3:
            X = X.permute(0, 2, 1).reshape(len(X), -1) if self.need_permute else X.reshape(len(X), -1)
        y_hat = model(X)
        y_hat = y_hat.view(-1)
        return y_hat


class DoubleAdapt(ForecastModel):
    def __init__(
        self, model, factor_num, x_dim=None, lr=0.001, weight_decay=0,
            need_permute=False, num_head=8, temperature=10,
            aux_weight=0.0, dir_scale=5.0, label_mean=0.0, label_std=1.0,
    ):
        super().__init__(
            model, x_dim=x_dim, lr=lr, need_permute=need_permute, weight_decay=weight_decay,
            aux_weight=aux_weight, dir_scale=dir_scale, label_mean=label_mean, label_std=label_std,
        )
        self.teacher_x = FeatureAdapter(factor_num, num_head, temperature)
        self.teacher_y = LabelAdapter(factor_num if x_dim is None else x_dim, num_head, temperature)
        self.meta_params = list(self.teacher_x.parameters()) + list(self.teacher_y.parameters())
        if self.device is not None:
            self.to(self.device)

    def forward(self, X, model=None, transform=False):
        """

        Args:
            X: [batch_size, x_dim]
            model: a forecast model generated by MAML

        Returns:
            immediate predictions. If adapt_y is True, still need to transform y_hat in the outer space.
        """
        if transform:
            """ For a L-length time-series, X should be shaped in [batch_size, L, factor_num] """
            X = self.teacher_x(X)
        return super().forward(X, model), X

