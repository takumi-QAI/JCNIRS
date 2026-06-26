"""models/pcr.py ─ 主成分回帰 (PCA → 線形回帰)。"""

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

TYPE = "pcr"
REQUIRES_TORCH = False


def build(params: dict, config: dict):
    return Pipeline([
        ("pca", PCA(n_components=params.get("n_components", 10))),
        ("reg", LinearRegression()),
    ])
