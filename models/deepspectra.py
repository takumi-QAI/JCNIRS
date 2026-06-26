"""
models/deepspectra.py ─ DeepSpectra (1D Inception CNN) 回帰。

Zhang et al. (2019) "DeepSpectra: An end-to-end deep learning approach for
quantitative spectral analysis" にならった構成:
    初段 Conv1d (ストライドでダウンサンプル)
      → Inception ブロック (1×1 / 1×1→k3 / 1×1→k5 / pool→1×1 を channel 結合)
      → AdaptiveAvgPool → 全結合 回帰ヘッド
"""

from .base import TORCH_AVAILABLE, TorchRegressorBase

TYPE = "deepspectra"
REQUIRES_TORCH = True


if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn

    class _Inception1D(nn.Module):
        """1D Inception ブロック (4 分岐を channel 方向に結合)。"""

        def __init__(self, in_ch, branch_ch):
            super().__init__()
            self.b1 = nn.Sequential(
                nn.Conv1d(in_ch, branch_ch, kernel_size=1),
                nn.ReLU(),
            )
            self.b2 = nn.Sequential(
                nn.Conv1d(in_ch, branch_ch, kernel_size=1), nn.ReLU(),
                nn.Conv1d(branch_ch, branch_ch, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.b3 = nn.Sequential(
                nn.Conv1d(in_ch, branch_ch, kernel_size=1), nn.ReLU(),
                nn.Conv1d(branch_ch, branch_ch, kernel_size=5, padding=2),
                nn.ReLU(),
            )
            self.b4 = nn.Sequential(
                nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
                nn.Conv1d(in_ch, branch_ch, kernel_size=1), nn.ReLU(),
            )

        def forward(self, x):
            return torch.cat([self.b1(x), self.b2(x),
                              self.b3(x), self.b4(x)], dim=1)

    class _DeepSpectraNet(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=7, stride=3, padding=3),
                nn.BatchNorm1d(16), nn.ReLU(),
            )
            self.inception1 = _Inception1D(16, 16)   # 出力 64 ch
            self.inception2 = _Inception1D(64, 32)   # 出力 128 ch
            self.pool = nn.AdaptiveAvgPool1d(8)
            self.fc = nn.Sequential(
                nn.Linear(128 * 8, 128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, 1),
            )

        def forward(self, x):
            x = x.unsqueeze(1)               # (batch, 1, n_features)
            x = self.stem(x)
            x = self.inception1(x)
            x = self.inception2(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    class DeepSpectraRegressor(TorchRegressorBase):
        """sklearn 互換の DeepSpectra 回帰モデル。"""

        def build_module(self, n_features: int):
            return _DeepSpectraNet(n_features)


def build(params: dict, config: dict):
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch が必要です: pip install torch")
    return DeepSpectraRegressor(**params, random_state=config["random_state"])
