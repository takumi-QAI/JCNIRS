"""
models/sae.py ─ スパース自己符号化器 (Sparse AutoEncoder, SAE) 回帰。

AE と同構造だが、潜在表現の活性に L1 スパース penalty を加えることで
少数の潜在ユニットのみが発火する「スパースな」表現を学習する。
    loss = recon_weight · MSE(recon, x)
         + reg_weight   · MSE(ŷ, y)
         + sparsity_weight · mean(|z|)
"""

from .base import TORCH_AVAILABLE, TorchRegressorBase

TYPE = "sae"
REQUIRES_TORCH = True


if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn

    def _mlp(dims, last_activation=True):
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if last_activation or i < len(dims) - 2:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    class _SAENet(nn.Module):
        """エンコーダ + デコーダ + 回帰ヘッド (潜在は ReLU でスパース化)。"""

        def __init__(self, n_features, latent_dim, hidden_dims):
            super().__init__()
            enc_dims = [n_features] + list(hidden_dims) + [latent_dim]
            dec_dims = [latent_dim] + list(reversed(hidden_dims)) + [n_features]
            self.encoder = _mlp(enc_dims, last_activation=True)
            self.decoder = _mlp(dec_dims, last_activation=False)
            self.head = nn.Sequential(
                nn.Linear(latent_dim, 64), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, 1),
            )

        def encode(self, x):
            return self.encoder(x)

        def forward_all(self, x):
            z = self.encode(x)
            recon = self.decoder(z)
            pred = self.head(z)
            return pred, recon, z

        def forward(self, x):
            z = self.encode(x)
            return self.head(z)

    class SAERegressor(TorchRegressorBase):
        """Sparse AE + 回帰ヘッド同時学習モデル。"""

        latent_dim = 64
        hidden_dims = [256, 128]
        recon_weight = 1.0
        reg_weight = 1.0
        sparsity_weight = 1e-3

        def build_module(self, n_features: int):
            return _SAENet(n_features, self.latent_dim, self.hidden_dims)

        def loss_fn(self, module, xb, yb):
            mse = nn.MSELoss()
            pred, recon, z = module.forward_all(xb)
            sparsity = torch.mean(torch.abs(z))
            return (self.reg_weight * mse(pred, yb)
                    + self.recon_weight * mse(recon, xb)
                    + self.sparsity_weight * sparsity)


def build(params: dict, config: dict):
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch が必要です: pip install torch")
    return SAERegressor(**params, random_state=config["random_state"])
