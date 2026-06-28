"""models/extratrees.py ─ Extremely Randomized Trees 回帰。

RandomForest より分割をランダム化した木のアンサンブル。本データの honest
species-LOSO で emsc+sg_d1 と組むと強く (CV≈18.8)、SVR/GBM と毛色が違うため
アンサンブルの多様性源として有用。
"""

from sklearn.ensemble import ExtraTreesRegressor

TYPE = "et"
REQUIRES_TORCH = False


def build(params: dict, config: dict):
    p = dict(params)
    p.setdefault("random_state", config.get("random_state", 42))
    p.setdefault("n_jobs", -1)
    return ExtraTreesRegressor(**p)
