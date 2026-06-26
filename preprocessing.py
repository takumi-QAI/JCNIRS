"""
JCNIRS 前処理 (preprocessing.py)
=================================
NIR スペクトルデータ向けの前処理をこのファイルに集約します。

method を「+」で連結すると順に適用される。
例: ``"snv+sg_d1+standard"`` → SNV → SG 1 次微分 → StandardScaler

対応メソッド
------------
スペクトル変換 (行方向):
    snv             Standard Normal Variate
    msc             Multiplicative Scatter Correction
    sg_d1           Savitzky-Golay 1 次微分
    sg_d2           Savitzky-Golay 2 次微分
    genlog          log1p 変換
スケーリング (列方向):
    standard        StandardScaler
    minmax          MinMaxScaler
    powertransformer  Yeo-Johnson
    rankgauss       QuantileTransformer (正規分布)
"""

import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, PowerTransformer, QuantileTransformer,
)


class SpectralPreprocessor:
    """NIR スペクトルデータ向けの前処理パイプライン。"""

    _SCALER_FACTORY = {
        "standard":         lambda: StandardScaler(),
        "minmax":           lambda: MinMaxScaler(),
        "powertransformer": lambda: PowerTransformer(method="yeo-johnson"),
        "rankgauss":        lambda: QuantileTransformer(output_distribution="normal"),
    }

    def __init__(self, method: str = "standard", config: dict | None = None):
        self.method = method
        self.steps = method.split("+")
        self.config = config or {}
        self._fitted: list = []

    # ---- public API ----
    def fit(self, X, y=None):
        self._fitted = []
        X_cur = X.copy()
        for step in self.steps:
            fitted_obj, X_cur = self._fit_one(step, X_cur)
            self._fitted.append((step, fitted_obj))
        return self

    def transform(self, X):
        X_cur = X.copy()
        for step, fitted_obj in self._fitted:
            X_cur = self._transform_one(step, X_cur, fitted_obj)
        return X_cur

    # ---- internal ----
    def _fit_one(self, step, X):
        if step in ("snv", "genlog", "sg_d1", "sg_d2"):
            return None, self._transform_one(step, X, None)
        if step == "msc":
            ref = np.mean(X, axis=0)
            return ref, self._transform_one(step, X, ref)
        if step in self._SCALER_FACTORY:
            scaler = self._SCALER_FACTORY[step]()
            scaler.fit(X)
            return scaler, scaler.transform(X)
        raise ValueError(f"Unknown preprocessing step: '{step}'")

    def _transform_one(self, step, X, fitted_obj):
        if step == "snv":
            return self._snv(X)
        if step == "msc":
            return self._msc(X, fitted_obj)
        if step == "sg_d1":
            return self._sg_deriv(X, deriv=1)
        if step == "sg_d2":
            return self._sg_deriv(X, deriv=2)
        if step == "genlog":
            return np.log1p(np.maximum(X, 0))
        return fitted_obj.transform(X)

    # ---- スペクトル変換の実装 ----
    @staticmethod
    def _snv(X):
        """Standard Normal Variate: 各スペクトルを自身の平均 / 標準偏差で正規化。"""
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True) + 1e-10
        return (X - mean) / std

    @staticmethod
    def _msc(X, reference):
        """Multiplicative Scatter Correction。"""
        out = np.zeros_like(X)
        for i in range(X.shape[0]):
            coef = np.polyfit(reference, X[i], 1)
            out[i] = (X[i] - coef[1]) / (coef[0] + 1e-10)
        return out

    def _sg_deriv(self, X, deriv=1):
        """Savitzky-Golay 微分フィルタ。"""
        wl = self.config.get("sg_window_length", 15)
        po = self.config.get("sg_polyorder", 2)
        return savgol_filter(X, window_length=wl, polyorder=po,
                             deriv=deriv, axis=1)
