"""
models/xgboost_multiseed.py ─ マルチシード + 早期終了 XGBoost 回帰。

LightGBM_MS の XGBoost 版。複数 seed の早期終了 XGBoost を平均して分散を下げる。
pipeline は log1p 空間の y を渡すため、内部 val も log1p 空間で評価される (整合)。
"""

import numpy as np
import xgboost as xgb

TYPE = "xgb_ms"
REQUIRES_TORCH = False


class MultiSeedXGB:
    """複数 seed の早期終了 XGBoost を平均するアンサンブル回帰器。"""

    def __init__(self, n_seeds=4, val_fraction=0.12, es_rounds=50,
                 random_state=42, **xgb_params):
        self.n_seeds = n_seeds
        self.val_fraction = val_fraction
        self.es_rounds = es_rounds
        self.random_state = random_state
        self.xgb_params = xgb_params
        self.models_ = []

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n = len(y)
        self.models_ = []
        for s in range(self.n_seeds):
            rng = np.random.RandomState(self.random_state + 1000 * s)
            idx = rng.permutation(n)
            nv = max(20, int(round(n * self.val_fraction)))
            nv = min(nv, n // 5)
            va, tr = idx[:nv], idx[nv:]
            m = xgb.XGBRegressor(
                random_state=s,
                early_stopping_rounds=self.es_rounds,
                **self.xgb_params,
            )
            m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
            self.models_.append(m)
        return self

    def predict(self, X):
        X = np.asarray(X)
        return np.mean([m.predict(X) for m in self.models_], axis=0)


def build(params: dict, config: dict):
    p = dict(params)
    p.setdefault("random_state", config.get("random_state", 42))
    return MultiSeedXGB(**p)
