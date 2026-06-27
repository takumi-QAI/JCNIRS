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
    msc_t           MSC (参照= train+test 合算平均, トランスダクティブ)
    emsc            Extended MSC (多項式ベースライン + 乗法補正)
    emsc_t          EMSC (参照= train+test 合算平均, トランスダクティブ)
    sg_d1           Savitzky-Golay 1 次微分
    sg_d2           Savitzky-Golay 2 次微分
    detrend         波長方向の 1 次トレンド除去 (A_detrend 系)
    l2norm          各スペクトルを L2 正規化 (A_norm 系)
    dwt             多重解像度 Haar ウェーブレット係数
    wband           水吸収バンド特徴を末尾に付加 (要 _wavelengths 注入)
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
        if step in ("snv", "genlog", "sg_d1", "sg_d2", "detrend", "l2norm",
                    "dwt", "wband"):
            return None, self._transform_one(step, X, None)
        if step in ("msc", "msc_t", "emsc", "emsc_t"):
            #   _t 系は train+test 合算平均(トランスダクティブ)を参照に使う
            ref = self._scatter_ref(step, X)
            return ref, self._transform_one(step, X, ref)
        if step in self._SCALER_FACTORY:
            scaler = self._SCALER_FACTORY[step]()
            scaler.fit(X)
            return scaler, scaler.transform(X)
        raise ValueError(f"Unknown preprocessing step: '{step}'")

    def _scatter_ref(self, step, X):
        """散乱補正(MSC/EMSC)の参照スペクトル。

        ``_t`` で終わるステップは config["_scatter_ref"] (= train+test 合算の平均、
        run_all で注入) を使う = トランスダクティブ。無ければ fit データ平均に戻す。
        """
        if step.endswith("_t"):
            ref = self.config.get("_scatter_ref")
            if ref is not None:
                return np.asarray(ref, dtype=float)
        return np.mean(X, axis=0)

    def _transform_one(self, step, X, fitted_obj):
        if step == "snv":
            return self._snv(X)
        if step in ("msc", "msc_t"):
            return self._msc(X, fitted_obj)
        if step in ("emsc", "emsc_t"):
            return self._emsc(X, fitted_obj)
        if step == "sg_d1":
            return self._sg_deriv(X, deriv=1)
        if step == "sg_d2":
            return self._sg_deriv(X, deriv=2)
        if step == "genlog":
            return np.log1p(np.maximum(X, 0))
        if step == "detrend":
            return self._detrend(X)
        if step == "l2norm":
            return self._l2norm(X)
        if step == "dwt":
            return self._dwt(X)
        if step == "wband":
            return self._wband(X, self.config.get("_wavelengths"))
        return fitted_obj.transform(X)

    # ---- スペクトル変換の実装 ----
    @staticmethod
    def _snv(X):
        """Standard Normal Variate: 各スペクトルを自身の平均 / 標準偏差で正規化。"""
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True) + 1e-10
        return (X - mean) / std

    @staticmethod
    def _detrend(X):
        """各スペクトルから波長方向の 1 次トレンド(直線)を除去する。

        散乱・ベースラインの傾きを取り除く NIR 前処理 (A_detrend 系)。
        各行を index に対する最小二乗直線でフィットし差し引く (ベクトル化)。
        """
        p = X.shape[1]
        x = np.arange(p, dtype=float)
        xc = x - x.mean()
        denom = (xc ** 2).sum() + 1e-12
        slope = (X * xc).sum(axis=1, keepdims=True) / denom   # sum(xc)=0 なので OK
        pred = slope * xc[None, :] + X.mean(axis=1, keepdims=True)
        return X - pred

    @staticmethod
    def _l2norm(X):
        """各スペクトルを L2 ノルムで正規化する (A_norm 系)。"""
        norm = np.sqrt((X ** 2).sum(axis=1, keepdims=True)) + 1e-12
        return X / norm

    @staticmethod
    def _msc(X, reference):
        """Multiplicative Scatter Correction。"""
        out = np.zeros_like(X)
        for i in range(X.shape[0]):
            coef = np.polyfit(reference, X[i], 1)
            out[i] = (X[i] - coef[1]) / (coef[0] + 1e-10)
        return out

    @staticmethod
    def _emsc(X, reference, poly_order=2):
        """Extended Multiplicative Scatter Correction (EMSC)。

        各スペクトルを  x ≈ a0 + a1·λ + a2·λ² + ... + b·ref  と最小二乗近似し、
        加法ベースライン(波長の多項式)を除去・乗法効果(b)で割り戻す:
            corrected = (x - Σ a_k·λ^k) / b
        MSC より柔軟に「板ごとの散乱・ベースライン差(ドメインシフト)」を除去する。
        全スペクトル共通の計画行列なので pinv 一発でベクトル化できる。
        """
        X = np.asarray(X, dtype=float)
        p = X.shape[1]
        lam = np.linspace(-1.0, 1.0, p)
        cols = [np.ones(p)] + [lam ** k for k in range(1, poly_order + 1)]
        cols.append(np.asarray(reference, dtype=float))   # 最終列 = 参照 (乗法)
        M = np.column_stack(cols)                          # (p, poly_order+2)
        P = np.linalg.pinv(M)                              # (k, p)
        C = X @ P.T                                        # (n, k) 係数
        b = C[:, -1:]                                      # 乗法係数 (n,1)
        baseline = C[:, :-1] @ M[:, :-1].T                 # 加法多項式 (n,p)
        return (X - baseline) / (np.where(np.abs(b) < 1e-9, 1e-9, b))

    @staticmethod
    def _dwt(X, levels=4):
        """多重解像度 Haar 離散ウェーブレット変換の係数を特徴量にする。

        各スペクトルを Haar で levels 段分解し、各段の detail と最終 approx を
        連結して返す (依存ライブラリ不要)。ノイズ抑制と多解像度表現を兼ねる。
        """
        X = np.asarray(X, dtype=float)
        a = X
        details = []
        for _ in range(levels):
            if a.shape[1] < 2:
                break
            if a.shape[1] % 2:
                a = a[:, :-1]
            even = a[:, 0::2]
            odd = a[:, 1::2]
            details.append((even - odd) / np.sqrt(2.0))    # detail
            a = (even + odd) / np.sqrt(2.0)                # approx
        return np.hstack([a] + details[::-1])             # approx + 粗→細

    @staticmethod
    def _wband(X, wavelengths):
        """水吸収バンド由来の物理特徴を末尾に付加する。

        NIR の O-H バンド (~5150 cm⁻¹ 結合音, ~6900 cm⁻¹ 第一倍音) 周辺の平均吸光、
        その比・差を特徴として連結する。wavelengths 未注入なら無変換。
        """
        if wavelengths is None:
            return X
        X = np.asarray(X, dtype=float)
        wl = np.asarray(wavelengths, dtype=float)

        def band(lo, hi):
            m = (wl >= lo) & (wl <= hi)
            if m.sum() == 0:
                return np.zeros((X.shape[0], 1))
            return X[:, m].mean(axis=1, keepdims=True)

        w1 = band(5100, 5300)      # O-H combination
        w2 = band(6800, 7000)      # O-H first overtone
        ref = band(4000, 10000)    # overall level
        feats = [w1, w2,
                 w1 / (w2 + 1e-9),
                 w1 / (ref + 1e-9),
                 w2 / (ref + 1e-9),
                 w1 - w2]
        return np.hstack([X] + feats)

    def _sg_deriv(self, X, deriv=1):
        """Savitzky-Golay 微分フィルタ。"""
        wl = self.config.get("sg_window_length", 15)
        po = self.config.get("sg_polyorder", 2)
        return savgol_filter(X, window_length=wl, polyorder=po,
                             deriv=deriv, axis=1)
