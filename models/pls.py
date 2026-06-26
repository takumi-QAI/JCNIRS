"""models/pls.py ─ 偏最小二乗回帰 (PLS Regression)。"""

from sklearn.cross_decomposition import PLSRegression

TYPE = "pls"
REQUIRES_TORCH = False


def build(params: dict, config: dict):
    return PLSRegression(**params)
