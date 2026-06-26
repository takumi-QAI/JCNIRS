"""
JCNIRS 評価指標 (metrics.py)
=============================
回帰モデルの精度を多面的に評価するための指標群を計算します。
各指標の詳細な意味・解釈は docs/REPORT_GUIDE.md を参照してください。

提供する指標
------------
RMSE     二乗平均平方根誤差   (低いほど良い、目的変数と同じ単位)
MAE      平均絶対誤差         (低いほど良い、外れ値に頑健)
R2       決定係数             (高いほど良い、1 が完全)
RPD      残差予測偏差比        (高いほど良い、NIR 分野の定番。std(y)/RMSE)
RPIQ     四分位範囲性能比      (高いほど良い、RPD の頑健版。IQR(y)/RMSE)
Bias     平均誤差 (ME)        (0 に近いほど良い、系統的な偏り)
MAPE     平均絶対パーセント誤差 (低いほど良い、相対誤差。小さい実測値に注意)
MaxError 最大絶対誤差          (低いほど良い、最悪ケース)

``compute_metrics`` は上記をまとめた dict を返します。``METRIC_INFO`` には
各指標の説明（日本語名・式・範囲・良し悪しの向き・解釈）を保持します。
"""

import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, max_error,
)


# ============================================================
# 指標の説明 (レポート / README / 図のキャプション用)
# ============================================================
METRIC_INFO = {
    "RMSE": {
        "name_ja": "二乗平均平方根誤差",
        "formula": "sqrt(mean((y - ŷ)^2))",
        "range":   "0 〜 ∞ (目的変数と同じ単位)",
        "better":  "low",
        "desc":    "誤差を二乗して平均し平方根を取った値。大きな誤差を強く罰する。"
                   "本コンペの主要指標であり、モデル選択の基準にも用いる。",
    },
    "MAE": {
        "name_ja": "平均絶対誤差",
        "formula": "mean(|y - ŷ|)",
        "range":   "0 〜 ∞ (目的変数と同じ単位)",
        "better":  "low",
        "desc":    "誤差の絶対値の平均。外れ値の影響が RMSE より小さく、"
                   "典型的な誤差の大きさを表す。RMSE との差が大きいほど"
                   "予測に大外しが含まれることを示唆する。",
    },
    "R2": {
        "name_ja": "決定係数",
        "formula": "1 - SS_res / SS_tot",
        "range":   "-∞ 〜 1 (1 が完全予測、0 は平均予測と同等)",
        "better":  "high",
        "desc":    "目的変数の分散のうちモデルが説明できた割合。1 に近いほど良い。"
                   "負値は平均値で予測するより悪いことを意味する。",
    },
    "RPD": {
        "name_ja": "残差予測偏差比 (Ratio of Performance to Deviation)",
        "formula": "std(y) / RMSE",
        "range":   "0 〜 ∞ (高いほど良い)",
        "better":  "high",
        "desc":    "実測値の標準偏差を RMSE で割った比。NIR 分光分析で広く使われる。"
                   "目安: <1.5 不十分, 1.5–2.0 粗い判別, 2.0–2.5 良好, "
                   ">2.5 非常に良好, >3.0 優秀。",
    },
    "RPIQ": {
        "name_ja": "四分位範囲性能比 (Ratio of Performance to InterQuartile distance)",
        "formula": "IQR(y) / RMSE  (IQR = Q3 - Q1)",
        "range":   "0 〜 ∞ (高いほど良い)",
        "better":  "high",
        "desc":    "RPD の頑健版で、分布の歪みや外れ値に強い。"
                   "本データのように右に裾を引く分布では RPD より信頼できる。"
                   "目安: >2.0 良好, >2.5 非常に良好。",
    },
    "Bias": {
        "name_ja": "平均誤差 (バイアス, Mean Error)",
        "formula": "mean(ŷ - y)",
        "range":   "-∞ 〜 ∞ (0 が理想)",
        "better":  "zero",
        "desc":    "予測の系統的な偏り。正なら過大予測、負なら過小予測の傾向。"
                   "0 から離れるほどキャリブレーションのずれが大きい。",
    },
    "MAPE": {
        "name_ja": "平均絶対パーセント誤差",
        "formula": "mean(|(y - ŷ) / y|) × 100 [%]",
        "range":   "0 〜 ∞ [%] (低いほど良い)",
        "better":  "low",
        "desc":    "相対誤差の平均。スケール非依存だが、実測値が小さいサンプル"
                   "(本データの含水率 <5% など) で値が極端に大きくなりうるため"
                   "補助指標として参照する。",
    },
    "MaxError": {
        "name_ja": "最大絶対誤差",
        "formula": "max(|y - ŷ|)",
        "range":   "0 〜 ∞ (目的変数と同じ単位)",
        "better":  "low",
        "desc":    "最も外した 1 サンプルの誤差。最悪ケースの大きさを表す。",
    },
}

# CV 結果テーブルでの列名 (例: RMSE → "CV-RMSE")
def cv_column(metric_key: str) -> str:
    return f"CV-{metric_key}"


# 既定で計算・表示する指標の順序
DEFAULT_METRICS = ["RMSE", "MAE", "R2", "RPD", "RPIQ", "Bias", "MAPE", "MaxError"]


# ============================================================
# 指標計算
# ============================================================
def compute_metrics(y_true, y_pred, metrics=None) -> dict:
    """回帰指標をまとめて計算して dict で返す。

    Parameters
    ----------
    y_true, y_pred : array-like
    metrics : list[str] | None
        計算する指標キー。None なら DEFAULT_METRICS。
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    keys = metrics or DEFAULT_METRICS

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    std_y = float(np.std(y_true))
    iqr_y = float(np.percentile(y_true, 75) - np.percentile(y_true, 25))
    eps = 1e-12

    out = {}
    for k in keys:
        if k == "RMSE":
            out[k] = rmse
        elif k == "MAE":
            out[k] = float(mean_absolute_error(y_true, y_pred))
        elif k == "R2":
            out[k] = float(r2_score(y_true, y_pred))
        elif k == "RPD":
            out[k] = float(std_y / (rmse + eps))
        elif k == "RPIQ":
            out[k] = float(iqr_y / (rmse + eps))
        elif k == "Bias":
            out[k] = float(np.mean(y_pred - y_true))
        elif k == "MAPE":
            # ゼロ割回避: 実測値の絶対値に微小値を加える
            denom = np.maximum(np.abs(y_true), eps)
            out[k] = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
        elif k == "MaxError":
            out[k] = float(max_error(y_true, y_pred))
        else:
            raise ValueError(f"Unknown metric: {k}")
    return out
