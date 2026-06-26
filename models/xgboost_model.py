"""models/xgboost_model.py ─ XGBoost 勾配ブースティング回帰。"""

import xgboost as xgb

TYPE = "xgb"
REQUIRES_TORCH = False


def build(params: dict, config: dict):
    return xgb.XGBRegressor(**params)
