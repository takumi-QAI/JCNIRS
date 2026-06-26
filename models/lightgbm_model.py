"""models/lightgbm_model.py ─ LightGBM 勾配ブースティング回帰 (高速)。"""

import lightgbm as lgb

TYPE = "lgbm"
REQUIRES_TORCH = False


def build(params: dict, config: dict):
    return lgb.LGBMRegressor(**params)
