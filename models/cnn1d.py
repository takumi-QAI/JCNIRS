"""
models/cnn1d.py ─ 1D 畳み込みニューラルネット回帰 (1D-CNN)。

3 層 1D 畳み込み + 全結合層。スペクトル系列から局所特徴を自動抽出する。
"""

from .base import TORCH_AVAILABLE, TorchRegressorBase

TYPE = "cnn1d"
REQUIRES_TORCH = True


if TORCH_AVAILABLE:
    import torch.nn as nn

    class _CNN1DNet(nn.Module):
        """3 層 1D 畳み込み + 全結合層。"""

        def __init__(self, n_features: int):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=11, padding=5),
                nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),

                nn.Conv1d(32, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),

                nn.Conv1d(64, 128, kernel_size=5, padding=2),
                nn.BatchNorm1d(128), nn.ReLU(),
                nn.AdaptiveAvgPool1d(8),
            )
            self.fc = nn.Sequential(
                nn.Linear(128 * 8, 256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 64),      nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            x = x.unsqueeze(1)            # (batch, 1, n_features)
            x = self.conv(x)
            x = x.view(x.size(0), -1)     # flatten
            return self.fc(x)

    class CNN1DRegressor(TorchRegressorBase):
        """sklearn 互換の 1D-CNN 回帰モデル。"""

        def build_module(self, n_features: int):
            return _CNN1DNet(n_features)


def build(params: dict, config: dict):
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch が必要です: pip install torch")
    return CNN1DRegressor(**params, random_state=config["random_state"])
