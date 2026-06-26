"""models/ridge.py ─ Ridge 回帰。"""

from sklearn.linear_model import Ridge

TYPE = "ridge"
REQUIRES_TORCH = False


def build(params: dict, config: dict):
    return Ridge(**params)
