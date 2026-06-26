"""
models/gan.py ─ 回帰 GAN (Generative Adversarial Network) 回帰。

- Generator G    : スペクトル → 含水率 ŷ (回帰器本体)
- Discriminator D: (スペクトル, 含水率) のペアが「本物 (x, y_true)」か
                   「偽物 (x, G(x))」かを識別

G は教師あり回帰損失 (MSE) で正確さを学びつつ、D を騙すことで予測の
分布を実データに近づける。予測は G(x) を使う。

2 つのオプティマイザを使うため TorchRegressorBase の学習ループは使わず、
自己完結で fit / predict を実装する (seed / device は base から流用)。
"""

import copy
import numpy as np
from .base import TORCH_AVAILABLE, set_seed, get_device

TYPE = "gan"
REQUIRES_TORCH = True


if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    def _mlp(dims, last_activation=False, dropout=0.0):
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            elif last_activation:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    class _Generator(nn.Module):
        def __init__(self, n_features, hidden_dims):
            super().__init__()
            self.net = _mlp([n_features] + list(hidden_dims) + [1],
                            dropout=0.2)

        def forward(self, x):
            return self.net(x)

    class _Discriminator(nn.Module):
        def __init__(self, n_features, hidden_dims):
            super().__init__()
            # 入力は (スペクトル, 含水率) を連結
            self.net = _mlp([n_features + 1] + list(hidden_dims) + [1],
                            dropout=0.3)

        def forward(self, x, y):
            return self.net(torch.cat([x, y], dim=1))

    class GANRegressor:
        """sklearn 互換の回帰 GAN。"""

        def __init__(self, epochs=150, batch_size=64, lr=2e-4,
                     weight_decay=0.0, hidden_dims=(256, 128),
                     adv_weight=0.1, reg_weight=1.0,
                     early_stopping=True, val_fraction=0.1,
                     es_patience=20, min_epochs=30,
                     random_state=42, verbose=False):
            self.epochs = epochs
            self.batch_size = batch_size
            self.lr = lr
            self.weight_decay = weight_decay
            self.hidden_dims = list(hidden_dims)
            self.adv_weight = adv_weight
            self.reg_weight = reg_weight
            self.early_stopping = early_stopping
            self.val_fraction = val_fraction
            self.es_patience = es_patience
            self.min_epochs = min_epochs
            self.random_state = random_state
            self.verbose = verbose
            self.device = get_device()
            self.G_ = None
            self.x_mean_ = None
            self.x_std_ = None
            self.y_mean_ = 0.0
            self.y_std_ = 1.0

        def fit(self, X, y):
            set_seed(self.random_state)
            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32)
            n = X.shape[0]

            # 早期終了用の train / val 分割
            use_es = (bool(self.early_stopping)
                      and self.val_fraction and n >= 20)
            rng = np.random.RandomState(self.random_state)
            if use_es:
                perm = rng.permutation(n)
                n_val = max(1, int(round(n * self.val_fraction)))
                val_idx = perm[:n_val]
                tr_idx = perm[n_val:]
            else:
                tr_idx = np.arange(n)
                val_idx = np.array([], dtype=int)

            # 入力 X を標準化 (train 部分の統計のみで fit)
            self.x_mean_ = X[tr_idx].mean(axis=0)
            self.x_std_ = X[tr_idx].std(axis=0) + 1e-8
            Xs = (X - self.x_mean_) / self.x_std_

            # ターゲット y を正規化 (train 部分の統計のみで fit)
            self.y_mean_ = float(y[tr_idx].mean())
            self.y_std_ = float(y[tr_idx].std()) + 1e-8
            y_norm = (y - self.y_mean_) / self.y_std_

            n_features = X.shape[1]
            self.G_ = _Generator(n_features, self.hidden_dims).to(self.device)
            D = _Discriminator(n_features, self.hidden_dims).to(self.device)

            opt_g = torch.optim.Adam(self.G_.parameters(), lr=self.lr,
                                     betas=(0.5, 0.999),
                                     weight_decay=self.weight_decay)
            opt_d = torch.optim.Adam(D.parameters(), lr=self.lr,
                                     betas=(0.5, 0.999),
                                     weight_decay=self.weight_decay)
            bce = nn.BCEWithLogitsLoss()
            mse = nn.MSELoss()

            Xtr_t = torch.FloatTensor(Xs[tr_idx]).to(self.device)
            ytr_t = torch.FloatTensor(y_norm[tr_idx]).unsqueeze(1).to(self.device)
            loader = DataLoader(
                TensorDataset(Xtr_t, ytr_t),
                batch_size=self.batch_size, shuffle=True,
            )
            if use_es:
                Xval_t = torch.FloatTensor(Xs[val_idx]).to(self.device)
                yval_t = torch.FloatTensor(
                    y_norm[val_idx]).unsqueeze(1).to(self.device)

            best_val = float("inf")
            best_state = None
            patience = 0
            for epoch in range(self.epochs):
                self.G_.train(); D.train()
                for xb, yb in loader:
                    bs = xb.size(0)
                    real = torch.ones(bs, 1, device=self.device)
                    fake = torch.zeros(bs, 1, device=self.device)

                    # ---- Discriminator ----
                    opt_d.zero_grad()
                    y_pred = self.G_(xb).detach()
                    d_real = bce(D(xb, yb), real)
                    d_fake = bce(D(xb, y_pred), fake)
                    d_loss = d_real + d_fake
                    d_loss.backward()
                    opt_d.step()

                    # ---- Generator ----
                    opt_g.zero_grad()
                    y_gen = self.G_(xb)
                    g_adv = bce(D(xb, y_gen), real)   # D を騙す
                    g_reg = mse(y_gen, yb)            # 教師あり回帰
                    g_loss = self.adv_weight * g_adv + self.reg_weight * g_reg
                    g_loss.backward()
                    opt_g.step()

                # 早期終了: 検証集合の回帰 MSE で best 生成器を保持
                if use_es:
                    self.G_.eval()
                    with torch.no_grad():
                        val_mse = float(mse(self.G_(Xval_t), yval_t).item())
                    if val_mse < best_val - 1e-6:
                        best_val = val_mse
                        best_state = copy.deepcopy(self.G_.state_dict())
                        patience = 0
                    else:
                        patience += 1
                    if (epoch + 1) >= self.min_epochs and patience >= self.es_patience:
                        break

                if self.verbose and (epoch + 1) % 30 == 0:
                    msg = f"G-reg={g_reg.item():.6f}"
                    if use_es:
                        msg += f"  ValMSE={best_val:.6f}"
                    print(f"      GAN Epoch {epoch+1}/{self.epochs}  {msg}")

            if use_es and best_state is not None:
                self.G_.load_state_dict(best_state)
            return self

        def predict(self, X):
            X = np.asarray(X, dtype=np.float32)
            Xs = (X - self.x_mean_) / self.x_std_
            self.G_.eval()
            X_t = torch.FloatTensor(Xs).to(self.device)
            preds = []
            with torch.no_grad():
                for i in range(0, len(X_t), self.batch_size):
                    preds.append(self.G_(X_t[i:i + self.batch_size]))
            out = torch.cat(preds).cpu().numpy().flatten()
            return out * self.y_std_ + self.y_mean_


def build(params: dict, config: dict):
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch が必要です: pip install torch")
    return GANRegressor(**params, random_state=config["random_state"])
