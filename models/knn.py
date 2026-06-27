"""
models/knn.py ─ k 近傍回帰 (k-Nearest Neighbors)。

スペクトルの「局所的な類似性」を使う回帰。NIR では似たスペクトル＝似た含水率に
なりやすいため、前処理 (SNV/微分/標準化) 後の距離で近傍を引くと有効なことがある。
"""

from sklearn.neighbors import KNeighborsRegressor

TYPE = "knn"
REQUIRES_TORCH = False


def build(params: dict, config: dict):
    return KNeighborsRegressor(**params)
