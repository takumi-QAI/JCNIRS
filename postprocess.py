"""
JCNIRS 後処理 (postprocess.py)
================================
乾燥時系列の構造を使った予測の後処理。

本データは「同一ボード(= species number)を乾燥させながら sample number 順に
連続スキャンした時系列」であり、ボード内では含水率が **sample number に対して
単調非増加** の滑らかな乾燥曲線になる (実測では 3 次多項式で R²≈0.999)。

スペクトルからの per-scan 予測はボードあたり数十〜200 点あるため、その系列に
「単調非増加」という物理制約を課す (= 保序回帰 / isotonic regression) と、
per-scan のスペクトルノイズが系列方向に平均化され誤差が下がる。

honest leave-one-board-out では overall RMSE 54→34 に改善することを確認済み。

※ この後処理は「ボード単位の乾燥時系列」という本データ固有の構造に依存する。
  時系列でないデータに流用する場合は CONFIG["postprocess_board_smooth"]=False に。
"""

import numpy as np


def _pava_non_increasing(y):
    """最小二乗の意味で最良の「単調非増加」近似を返す (PAVA)。

    Pool Adjacent Violators Algorithm。隣接で増加 (violation) していたら
    重み付き平均にプールする。
    """
    y = np.asarray(y, dtype=float)
    # 各ブロック [値, 重み(=点数)]
    blocks = [[float(v), 1.0] for v in y]
    j = 0
    while j < len(blocks) - 1:
        # 単調非増加の違反: 左 < 右
        if blocks[j][0] < blocks[j + 1][0] - 1e-12:
            tot = blocks[j][1] + blocks[j + 1][1]
            mv = (blocks[j][0] * blocks[j][1]
                  + blocks[j + 1][0] * blocks[j + 1][1]) / tot
            blocks[j] = [mv, tot]
            del blocks[j + 1]
            if j > 0:
                j -= 1
        else:
            j += 1
    out = np.empty(len(y), dtype=float)
    pos = 0
    for mv, wt in blocks:
        c = int(round(wt))
        out[pos:pos + c] = mv
        pos += c
    return out


def smooth_by_board(pred, sample_numbers, board_ids, method="isotonic",
                    clip=None):
    """ボード(= board_ids)ごとに sample number 順へ並べ、乾燥曲線を平滑化する。

    Parameters
    ----------
    pred : array  per-scan 予測 (元スケール)
    sample_numbers : array  各行の sample number (時間順)
    board_ids : array  各行のボード ID (= species number)
    method : "isotonic" | "poly1".."poly3"
        isotonic = 単調非増加制約 (推奨, パラメータ無し・物理的)
        polyN    = N 次多項式フィット (滑らかだが端で外れることがある)
    clip : [low, high] or None  平滑化後のクリップ

    Returns
    -------
    out : array  平滑化後の予測 (元の行順)
    """
    pred = np.asarray(pred, dtype=float)
    sn = np.asarray(sample_numbers)
    bid = np.asarray(board_ids)
    out = pred.copy()

    for b in np.unique(bid):
        idx = np.where(bid == b)[0]
        if len(idx) < 3:
            continue
        order = idx[np.argsort(sn[idx], kind="mergesort")]
        v = pred[order]
        if method == "isotonic":
            sv = _pava_non_increasing(v)
        elif method.startswith("poly"):
            deg = int(method[4:]) if len(method) > 4 else 2
            x = sn[order].astype(float)
            x = (x - x.mean()) / (x.std() + 1e-9)
            sv = np.polyval(np.polyfit(x, v, deg), x)
        else:
            sv = v
        out[order] = sv

    if clip:
        low, high = clip
        out = np.clip(out, low, high)
    return out
