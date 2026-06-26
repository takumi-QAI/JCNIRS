"""models/elasticnet.py ─ ElasticNet 回帰。"""

from sklearn.linear_model import ElasticNet

TYPE = "elasticnet"
REQUIRES_TORCH = False


def build(params: dict, config: dict):
    return ElasticNet(**params)
