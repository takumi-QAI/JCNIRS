"""
models/autoencoder.py ─ 自己符号化器 (AutoEncoder, AE) 回帰。

エンコーダ + デコーダ (再構成) と、潜在表現からの回帰ヘッドを
**同時学習** する教師あり回帰モデル。
    loss = recon_weight · MSE(recon, x) + reg_weight · MSE(ŷ, y)
"""

from .base import TORCH_AVAILABLE, TorchRegressorBase

TYPE = "ae"
REQUIRES_TORCH = True


if TORCH_AVAILABLE:
    import torch.nn as nn

    def _mlp(dims, last_activation=True):
        """全結合層を積み重ねる (各層 ReLU)。"""
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if last_activation or i < len(dims) - 2:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    class _AENet(nn.Module):
        """エンコーダ + デコーダ + 回帰ヘッド。"""

        def __init__(self, n_features, latent_dim, hidden_dims):
            super().__init__()
            enc_dims = [n_features] + list(hidden_dims) + [latent_dim]
            dec_dims = [latent_dim] + list(reversed(hidden_dims)) + [n_features]
            self.encoder = _mlp(enc_dims, last_activation=True)
            # デコーダ最終層は活性化なし (再構成は実数値)
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
            return pred, recon

        def forward(self, x):
            z = self.encode(x)
            return self.head(z)

    class AERegressor(TorchRegressorBase):
        """AE + 回帰ヘッド同時学習モデル。"""

        latent_dim = 32
        hidden_dims = [256, 128]
        recon_weight = 1.0
        reg_weight = 1.0

        def build_module(self, n_features: int):
            return _AENet(n_features, self.latent_dim, self.hidden_dims)

        def loss_fn(self, module, xb, yb):
            mse = nn.MSELoss()
            pred, recon = module.forward_all(xb)
            return (self.reg_weight * mse(pred, yb)
                    + self.recon_weight * mse(recon, xb))


def build(params: dict, config: dict):
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch が必要です: pip install torch")
    return AERegressor(**params, random_state=config["random_state"])
