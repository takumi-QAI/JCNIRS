"""
models/base.py
==============
PyTorch ベースの回帰モデルで共通する処理をまとめた基底クラス。

- 入力 (X) の標準化 → 前処理に依らず学習を安定化 (精度向上)
- ターゲット (y) の正規化 → 学習安定化
- Adam + ReduceLROnPlateau
- 早期終了 (検証分割の損失で best 重みを保持し過学習を抑制 → 精度向上)
- ミニバッチ学習ループ / device 自動選択 / 乱数シード固定
- sklearn 互換 API (fit / predict)

サブクラスは ``build_module(n_features) -> nn.Module`` を実装するだけでよい。
再構成損失など複合損失を持つモデル (AE / SAE / VAE) は ``loss_fn`` を
オーバーライドする。予測は既定で ``module(x)`` を回帰出力とみなす
(AE 系の nn.Module は forward が予測のみを返し、損失計算用の
中間出力は別メソッド forward_all で提供する設計)。
"""

import copy
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:                                    # pragma: no cover
    TORCH_AVAILABLE = False


def set_seed(seed: int):
    """乱数シードを固定する (numpy + torch)。"""
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def get_device():
    """利用可能なら CUDA、なければ CPU を返す。"""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


if TORCH_AVAILABLE:

    class TorchRegressorBase:
        """sklearn 互換の PyTorch 回帰モデル基底クラス。

        Parameters
        ----------
        epochs : int          学習エポック数
        batch_size : int      ミニバッチサイズ
        lr : float            学習率
        weight_decay : float  L2 正則化
        random_state : int    乱数シード
        verbose : bool        学習経過を表示するか
        """

        def __init__(self, epochs=100, batch_size=64, lr=1e-3,
                     weight_decay=1e-4, random_state=42, verbose=False,
                     early_stopping=True, val_fraction=0.1,
                     es_patience=15, min_epochs=20, **kwargs):
            self.epochs = epochs
            self.batch_size = batch_size
            self.lr = lr
            self.weight_decay = weight_decay
            self.random_state = random_state
            self.verbose = verbose
            # 早期終了の設定
            self.early_stopping = early_stopping
            self.val_fraction = val_fraction
            self.es_patience = es_patience
            self.min_epochs = min_epochs
            # サブクラス固有のハイパーパラメータを属性として保持
            for k, v in kwargs.items():
                setattr(self, k, v)
            self.device = get_device()
            self.net_ = None
            self.x_mean_ = None
            self.x_std_ = None
            self.y_mean_ = 0.0
            self.y_std_ = 1.0

        # ---- サブクラスが実装 ----
        def build_module(self, n_features: int) -> "nn.Module":
            raise NotImplementedError

        # ---- 損失 (既定: 予測の MSE)。複合損失モデルはオーバーライド ----
        def loss_fn(self, module, xb, yb):
            criterion = nn.MSELoss()
            return criterion(self._forward_pred(module, xb), yb)

        # ---- 予測 (既定: module(x))。必要ならオーバーライド ----
        @staticmethod
        def _forward_pred(module, xb):
            out = module(xb)
            if out.dim() == 1:
                out = out.unsqueeze(1)
            return out

        # ---- 学習 ----
        def fit(self, X, y):
            set_seed(self.random_state)
            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32)
            n = X.shape[0]

            # 早期終了用に train / val を分割 (サンプルが少なすぎる場合は無効)
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

            # 入力 X を標準化 (train 部分の統計のみで fit → リーク回避)
            self.x_mean_ = X[tr_idx].mean(axis=0)
            self.x_std_ = X[tr_idx].std(axis=0) + 1e-8
            Xs = (X - self.x_mean_) / self.x_std_

            # ターゲット y を正規化 (train 部分の統計のみで fit)
            self.y_mean_ = float(y[tr_idx].mean())
            self.y_std_ = float(y[tr_idx].std()) + 1e-8
            y_norm = (y - self.y_mean_) / self.y_std_

            self.net_ = self.build_module(X.shape[1]).to(self.device)
            optimizer = torch.optim.Adam(
                self.net_.parameters(),
                lr=self.lr, weight_decay=self.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=10, factor=0.5, min_lr=1e-6,
            )

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
                self.net_.train()
                total_loss = 0.0
                for xb, yb in loader:
                    optimizer.zero_grad()
                    loss = self.loss_fn(self.net_, xb, yb)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * xb.size(0)
                train_loss = total_loss / len(tr_idx)

                if use_es:
                    self.net_.eval()
                    with torch.no_grad():
                        val_loss = float(
                            self.loss_fn(self.net_, Xval_t, yval_t).item())
                    scheduler.step(val_loss)
                    if val_loss < best_val - 1e-6:
                        best_val = val_loss
                        best_state = copy.deepcopy(self.net_.state_dict())
                        patience = 0
                    else:
                        patience += 1
                    if (epoch + 1) >= self.min_epochs and patience >= self.es_patience:
                        break
                else:
                    scheduler.step(train_loss)

                if self.verbose and (epoch + 1) % 20 == 0:
                    msg = f"Loss={train_loss:.6f}"
                    if use_es:
                        msg += f"  ValLoss={best_val:.6f}"
                    print(f"      {type(self).__name__} "
                          f"Epoch {epoch+1}/{self.epochs}  {msg}")

            if use_es and best_state is not None:
                self.net_.load_state_dict(best_state)
            return self

        # ---- 予測 ----
        def predict(self, X):
            X = np.asarray(X, dtype=np.float32)
            Xs = (X - self.x_mean_) / self.x_std_
            self.net_.eval()
            X_t = torch.FloatTensor(Xs).to(self.device)
            preds = []
            with torch.no_grad():
                for i in range(0, len(X_t), self.batch_size):
                    out = self._forward_pred(self.net_, X_t[i:i + self.batch_size])
                    preds.append(out)
            out = torch.cat(preds).cpu().numpy().flatten()
            return out * self.y_std_ + self.y_mean_
