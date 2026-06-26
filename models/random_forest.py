"""models/random_forest.py ─ ランダムフォレスト回帰。"""

from sklearn.ensemble import RandomForestRegressor

TYPE = "rf"
REQUIRES_TORCH = False


def build(params: dict, config: dict):
    return RandomForestRegressor(**params)
