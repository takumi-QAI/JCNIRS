"""models/lasso.py ─ Lasso 回帰。"""

from sklearn.linear_model import Lasso

TYPE = "lasso"
REQUIRES_TORCH = False


def build(params: dict, config: dict):
    return Lasso(**params)
