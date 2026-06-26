"""
JCNIRS 古典的特徴量選択 (feature_selection.py)
===============================================
量子アニーリング以外の特徴量選択手法をこのファイルに集約します。

① none      特徴量選択なし (全波長 = ベースライン)
② filter    フィルター法   (相互情報量でランキング → 上位 k 個)
③ wrapper   ラッパー法     (RFE: Recursive Feature Elimination)
④ embedded  埋め込み法     (Lasso の係数絶対値でランキング → 上位 k 個)

⑤ Amplify QUBO (量子アニーリング) は feature_selection_quantum.py に分離。

各関数は ``(X, y, config_fs) -> bool マスク (shape: n_features,)`` を返す。
``SELECTORS`` 辞書で戦略名 → 関数を引ける。
"""

import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso as LassoFS


def select_none(X, y, config):
    """① 全特徴量を使用 (ベースライン)。"""
    return np.ones(X.shape[1], dtype=bool)


def select_filter(X, y, config):
    """② フィルター法: 相互情報量で上位 k 個を選択。"""
    cfg = config["filter"]
    k = cfg["n_features"]
    print(f"    相互情報量を計算中 ({X.shape[1]} 特徴量) ...")
    mi = mutual_info_regression(X, y, random_state=42)
    top_k = np.argsort(mi)[-k:]
    mask = np.zeros(X.shape[1], dtype=bool)
    mask[top_k] = True
    print(f"    MI 上位 {k} 個を選択")
    return mask


def select_wrapper(X, y, config):
    """③ ラッパー法: RFE (LightGBM) で上位 k 個を選択。"""
    import lightgbm as lgb
    from sklearn.feature_selection import RFE

    cfg = config["wrapper"]
    estimator = lgb.LGBMRegressor(
        n_estimators=cfg["n_estimators"],
        verbose=-1, random_state=42,
    )
    print(f"    RFE 実行中 (step={cfg['step']}) ...")
    selector = RFE(
        estimator,
        n_features_to_select=cfg["n_features"],
        step=cfg["step"],
    )
    selector.fit(X, y)
    print(f"    {selector.support_.sum()} 特徴量を選択")
    return selector.support_


def select_embedded(X, y, config):
    """④ 埋め込み法: Lasso の係数絶対値で上位 k 個を選択。"""
    cfg = config["embedded"]
    k = cfg["n_features"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"    Lasso 学習中 (alpha={cfg['alpha']}) ...")
    lasso = LassoFS(alpha=cfg["alpha"], max_iter=10000, random_state=42)
    lasso.fit(X_scaled, y)

    importance = np.abs(lasso.coef_)
    n_nonzero = np.sum(importance > 0)
    print(f"    非ゼロ係数: {n_nonzero} → 上位 {k} 個を選択")

    top_k = np.argsort(importance)[-k:]
    mask = np.zeros(X.shape[1], dtype=bool)
    mask[top_k] = True
    return mask


# 戦略名 → 関数 (古典的手法のみ。amplify は run_all.py で統合)
SELECTORS = {
    "none":     select_none,
    "filter":   select_filter,
    "wrapper":  select_wrapper,
    "embedded": select_embedded,
}
