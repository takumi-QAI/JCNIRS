"""
models パッケージ
=================
各モデルは 1 ファイルにまとめられ、それぞれ以下を公開する:

- ``TYPE``           : config の ``models[*]["type"]`` と対応する識別子
- ``REQUIRES_TORCH`` : PyTorch を要するか
- ``build(params, config)`` : sklearn 互換の推定器を返すファクトリ

``build_all_models(config)`` で CONFIG["models"] から全モデルを生成する。
PyTorch 未インストール時は torch を要するモデル(深層)を自動スキップする。
"""

from .base import TORCH_AVAILABLE

# ---- 古典的な機械学習モデル ----
from . import (
    pls, pcr, ridge, lasso, elasticnet, svr, knn,
    random_forest, extratrees, xgboost_model, lightgbm_model,
)
# ---- 深層学習モデル (PyTorch) ----
from . import (
    cnn1d, autoencoder, sae, vae, gan, deepspectra, transformer,
)

_MODULES = [
    pls, pcr, ridge, lasso, elasticnet, svr, knn,
    random_forest, extratrees, xgboost_model, lightgbm_model,
    cnn1d, autoencoder, sae, vae, gan, deepspectra, transformer,
]

# type → build 関数 / PyTorch を要するか
MODEL_BUILDERS = {m.TYPE: m.build for m in _MODULES}
MODEL_REQUIRES_TORCH = {m.TYPE: m.REQUIRES_TORCH for m in _MODULES}


def build_all_models(config: dict) -> dict:
    """CONFIG["models"] から全モデル辞書 {名前: 推定器} を生成する。

    PyTorch 未インストール時、torch を要するモデルはスキップする。
    """
    models = {}
    for name, spec in config["models"].items():
        mtype = spec["type"]
        if mtype not in MODEL_BUILDERS:
            raise ValueError(f"Unknown model type: {mtype}")
        if MODEL_REQUIRES_TORCH.get(mtype) and not TORCH_AVAILABLE:
            print(f"  ⚠ {name} をスキップ (PyTorch 未インストール)")
            continue
        models[name] = MODEL_BUILDERS[mtype](spec["params"], config)
    return models
