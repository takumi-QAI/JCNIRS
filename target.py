"""
JCNIRS 目的変数変換 (target.py)
================================
目的変数 (含水率) の変換と逆変換をまとめます。

含水率は右に裾を引く分布 (min 0.84 / 中央値 29 / 最大 299 / std 49.5) のため、
log1p 変換で分布を圧縮すると多くのモデルで学習が安定し精度が向上しやすい。
学習は変換後の空間で行い、予測値は逆変換して元スケールに戻してから評価する
(指標は必ず元スケールで計算する)。

対応する変換 (kind)
-------------------
none   変換なし (恒等)
log1p  y -> log(1 + y)、逆変換は expm1
"""

import numpy as np

# log1p 空間の予測値の上限 (expm1 のオーバーフロー & 非現実的な値を防ぐ)。
# log1p(1e4) ≈ 9.21 → 元スケールで最大 1e4 [%]。学習データ上限 (約 299) より
# 十分大きく、かつ float64 で有限に収まる安全な上限。
_LOG1P_SAFE_MAX = np.log1p(1e4)


def forward_transform(y, kind: str = "none"):
    """目的変数を変換する (学習前)。"""
    y = np.asarray(y, dtype=float)
    if kind in (None, "none"):
        return y
    if kind == "log1p":
        # 含水率は基本的に非負だが念のため下限を -1 より大きくクリップ
        return np.log1p(np.maximum(y, -1 + 1e-6))
    raise ValueError(f"Unknown target_transform: {kind}")


def inverse_transform(y_hat, kind: str = "none"):
    """変換後空間の予測値を元スケールへ戻す (予測後)。"""
    y_hat = np.asarray(y_hat, dtype=float)
    if kind in (None, "none"):
        return y_hat
    if kind == "log1p":
        # expm1 のオーバーフロー回避のため log 空間で上限クリップしてから逆変換
        return np.expm1(np.clip(y_hat, None, _LOG1P_SAFE_MAX))
    raise ValueError(f"Unknown target_transform: {kind}")


def clip_predictions(y_pred, clip):
    """予測値を [low, high] にクリップし、非有限値 (inf/NaN) を除去する。

    clip = [low, high] (None は無制限)。含水率は非負なので既定で下限 0。
    学習が不安定なモデルが inf/NaN を返してもアンサンブルが壊れないよう、
    非有限値は安全な値に置換する。
    """
    y_pred = np.asarray(y_pred, dtype=float)
    if clip:
        low, high = clip
        y_pred = np.clip(y_pred, low, high)
    else:
        low, high = None, None
    # 残った非有限値を除去 (nan→下限 or 0, +inf→上限 or 大きな有限値, -inf→下限 or 0)
    nan_fill = low if low is not None else 0.0
    pos_fill = high if high is not None else 1e4
    neg_fill = low if low is not None else 0.0
    return np.nan_to_num(y_pred, nan=nan_fill, posinf=pos_fill, neginf=neg_fill)
