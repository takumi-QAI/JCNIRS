"""models/svr.py ─ サポートベクター回帰 (SVR)。"""

from sklearn.svm import SVR

TYPE = "svr"
REQUIRES_TORCH = False


def build(params: dict, config: dict):
    return SVR(**params)
