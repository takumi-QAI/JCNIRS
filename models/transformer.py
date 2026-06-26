"""
models/transformer.py ─ スペクトル Transformer 回帰。

スペクトルを長さ patch_size のパッチに分割し、各パッチを線形埋め込み
(トークン化) → 位置エンコーディング → Transformer Encoder → 平均プーリング
→ 全結合 回帰ヘッド、という ViT 風の構成。
"""

from .base import TORCH_AVAILABLE, TorchRegressorBase

TYPE = "transformer"
REQUIRES_TORCH = True


if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn

    class _SpectralTransformer(nn.Module):
        def __init__(self, n_features, patch_size, d_model,
                     n_heads, n_layers, dropout):
            super().__init__()
            self.n_features = n_features
            self.patch_size = patch_size
            self.n_patches = (n_features + patch_size - 1) // patch_size
            self.pad = self.n_patches * patch_size - n_features

            self.embed = nn.Linear(patch_size, d_model)
            self.pos = nn.Parameter(
                torch.zeros(1, self.n_patches, d_model))
            nn.init.trunc_normal_(self.pos, std=0.02)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model * 4, dropout=dropout,
                activation="gelu", batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Sequential(
                nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            if self.pad > 0:
                x = nn.functional.pad(x, (0, self.pad))
            # (batch, n_patches, patch_size)
            x = x.view(x.size(0), self.n_patches, self.patch_size)
            x = self.embed(x) + self.pos
            x = self.encoder(x)
            x = self.norm(x.mean(dim=1))      # パッチ方向に平均プーリング
            return self.head(x)

    class TransformerRegressor(TorchRegressorBase):
        """sklearn 互換のスペクトル Transformer 回帰モデル。"""

        patch_size = 32
        d_model = 64
        n_heads = 4
        n_layers = 2
        dropout = 0.1

        def build_module(self, n_features: int):
            return _SpectralTransformer(
                n_features, self.patch_size, self.d_model,
                self.n_heads, self.n_layers, self.dropout,
            )


def build(params: dict, config: dict):
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch が必要です: pip install torch")
    return TransformerRegressor(**params, random_state=config["random_state"])
