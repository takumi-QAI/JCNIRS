"""
models/lightgbm_multiseed.py ─ マルチシード + 早期終了 LightGBM 回帰。

実績ベース (友人の EXPERIMENTS.md, v1/v11 = LB 10.6) の主力モデル。
- num_leaves=31・2000 本・learning_rate 小さめ + **早期終了**(内部 val 分割)
- 複数 seed で学習し予測を平均 → 分散を下げて安定化
- 単発の固定本数 LightGBM より honest に汎化する (v16 の固定 500 本は LB 22 で失敗)

sklearn 互換 (fit / predict)。pipeline は log1p 空間の y を渡すため、早期終了の
内部 val も log1p 空間で評価される (整合)。
"""

import numpy as np
import lightgbm as lgb

TYPE = "lgbm_ms"
REQUIRES_TORCH = False


class MultiSeedLGBM:
    """複数 seed の早期終了 LightGBM を平均するアンサンブル回帰器。"""

    def __init__(self, n_seeds=4, val_fraction=0.12, es_rounds=100,
                 random_state=42, **lgb_params):
        self.n_seeds = n_seeds
        self.val_fraction = val_fraction
        self.es_rounds = es_rounds
        self.random_state = random_state
        self.lgb_params = lgb_params
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
            nv = min(nv, n // 5)            # val は最大でも 1/5
            va, tr = idx[:nv], idx[nv:]
            m = lgb.LGBMRegressor(random_state=s, **self.lgb_params)
            m.fit(
                X[tr], y[tr],
                eval_set=[(X[va], y[va])],
                callbacks=[
                    lgb.early_stopping(self.es_rounds, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
            self.models_.append(m)
        return self

    def predict(self, X):
        X = np.asarray(X)
        return np.mean([m.predict(X) for m in self.models_], axis=0)


def build(params: dict, config: dict):
    p = dict(params)
    p.setdefault("random_state", config.get("random_state", 42))
    return MultiSeedLGBM(**p)
