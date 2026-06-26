"""
models/vae.py ─ 変分自己符号化器 (Variational AutoEncoder, VAE) 回帰。

エンコーダが潜在分布のパラメータ (μ, logσ²) を出力し、再パラメータ化
トリックでサンプリングした潜在変数からデコーダで再構成する。
回帰ヘッドは μ (分布の平均) から含水率を予測する。
    loss = recon_weight · MSE(recon, x)
         + kl_weight    · KL(q(z|x) ‖ N(0, I))
         + reg_weight   · MSE(ŷ, y)
"""

from .base import TORCH_AVAILABLE, TorchRegressorBase

TYPE = "vae"
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

    class _VAENet(nn.Module):
        """変分エンコーダ + デコーダ + 回帰ヘッド。"""

        def __init__(self, n_features, latent_dim, hidden_dims):
            super().__init__()
            enc_dims = [n_features] + list(hidden_dims)
            self.encoder = _mlp(enc_dims, last_activation=True)
            self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
            self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
            dec_dims = [latent_dim] + list(reversed(hidden_dims)) + [n_features]
            self.decoder = _mlp(dec_dims, last_activation=False)
            self.head = nn.Sequential(
                nn.Linear(latent_dim, 64), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, 1),
            )

        def encode(self, x):
            h = self.encoder(x)
            return self.fc_mu(h), self.fc_logvar(h)

        @staticmethod
        def reparameterize(mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward_all(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            recon = self.decoder(z)
            pred = self.head(mu)          # 回帰は安定した μ から
            return pred, recon, mu, logvar

        def forward(self, x):
            mu, _ = self.encode(x)
            return self.head(mu)

    class VAERegressor(TorchRegressorBase):
        """VAE + 回帰ヘッド同時学習モデル。"""

        latent_dim = 32
        hidden_dims = [256, 128]
        recon_weight = 1.0
        reg_weight = 1.0
        kl_weight = 1e-3

        def build_module(self, n_features: int):
            return _VAENet(n_features, self.latent_dim, self.hidden_dims)

        def loss_fn(self, module, xb, yb):
            mse = nn.MSELoss()
            pred, recon, mu, logvar = module.forward_all(xb)
            kl = -0.5 * torch.mean(
                1 + logvar - mu.pow(2) - logvar.exp())
            return (self.reg_weight * mse(pred, yb)
                    + self.recon_weight * mse(recon, xb)
                    + self.kl_weight * kl)


def build(params: dict, config: dict):
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch が必要です: pip install torch")
    return VAERegressor(**params, random_state=config["random_state"])
