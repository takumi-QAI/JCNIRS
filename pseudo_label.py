"""
pseudo_label.py ─ 擬似ラベル / transductive 自己学習 (honest 検証つき)
================================================================================
本データの test は「6 枚の未知ボードを丸ごと当てる」問題で、各ボードは
sample number 順に含水率が単調に減る乾燥曲線。この **構造** を使う半教師あり学習:

  ① base モデルを train で学習 → test を予測 (擬似ラベル, ノイズ大)
  ② 各 test ボードで sample number 順に **単調平滑化** (isotonic) → 擬似ラベル精製
  ③ test スペクトル + 擬似ラベルを学習データに足して再学習 → test を再予測
  ④ ②③を数回反復

擬似ラベルは諸刃の剣 (友人 EXPERIMENTS.md でも「悪化しがち」)。そこで
**species 単位 leave-one-board-out (≒LB) で「本当に効くか」を必ず検証**してから
提出ファイルを作る。LOSO で改善しなければ採用しない。

使い方:  python pseudo_label.py
出力 (submissions_manual/):
  - pseudo_loso.csv               各手法の species-LOSO RMSE 比較表
  - sub_pseudo_<model>_<variant>.csv  本番 test 予測 (LOSO で最良だった手法)
  ログ末尾の「>>> 推奨提出」を提出する (ただし LOSO が base より良い時のみ)。
"""
import os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from preprocessing import SpectralPreprocessor
from target import forward_transform as ft, inverse_transform as it, clip_predictions as clip
from postprocess import smooth_by_board
import lightgbm as lgb
from sklearn.svm import SVR

# ============================================================
#  設定 (編集可)
# ============================================================
PREP        = "emsc+sg_d1"   # 探索で単体最良だった前処理
ITERS       = 2              # 自己学習の反復回数
SMOOTH      = "isotonic"     # 擬似ラベルの平滑化 (isotonic=単調非増加)
PSEUDO_FRAC = 1.0            # 擬似ラベル行の重み (LGB のみ。1.0=本物と同等)
TK = CONFIG.get("target_transform", "log1p")
CL = CONFIG.get("clip_predictions")

def svr10(): return SVR(kernel="rbf", C=10, epsilon=0.05, gamma="scale")
def lgbm():  return lgb.LGBMRegressor(
    n_estimators=900, learning_rate=0.03, num_leaves=31, subsample=0.8,
    subsample_freq=1, colsample_bytree=0.4, min_child_samples=20,
    reg_lambda=2.0, random_state=0, verbose=-1, n_jobs=-1)
BASES = {"SVR10": svr10, "LGB": lgbm}

# ============================================================
#  データ
# ============================================================
tr = pd.read_csv("train.csv", encoding=CONFIG["encoding"])
te = pd.read_csv("test.csv",  encoding=CONFIG["encoding"])
spec = list(tr.columns[4:])
Xall = tr[spec].values.astype(float)
yall = tr[CONFIG["target_col"]].values.astype(float)
sn_all = tr[CONFIG["id_col"]].values
board_all = tr["species number"].values
Xte = te[te.columns[3:]].values.astype(float)
sn_te = te[CONFIG["id_col"]].values
board_te = te["species number"].values.astype(int)
ids = te[CONFIG["id_col"]].values


def fit_predict(make, Xtr, ytr, Xpred, sw=None):
    m = make()
    if sw is not None:
        try:
            m.fit(Xtr, ft(ytr, TK), sample_weight=sw)
        except TypeError:
            m.fit(Xtr, ft(ytr, TK))         # SVR は sample_weight 非対応
    else:
        m.fit(Xtr, ft(ytr, TK))
    return clip(it(np.asarray(m.predict(Xpred)).ravel(), TK), CL)


def self_train(make, A_tr, y_tr, B_te, sn_b, board_b, iters):
    """自己学習。base→平滑化擬似ラベル→追加再学習を iters 回。

    Returns: dict {"base","smooth","st1","st2",...} の予測 (B_te に対する)
    """
    out = {}
    p = fit_predict(make, A_tr, y_tr, B_te)
    out["base"]   = p
    out["smooth"] = smooth_by_board(p, sn_b, board_b, SMOOTH, clip=CL)
    pseudo = out["smooth"]
    for k in range(1, iters + 1):
        augX = np.vstack([A_tr, B_te])
        augy = np.concatenate([y_tr, pseudo])
        sw = None
        # LGB は擬似ラベル行を PSEUDO_FRAC で重み付け (SVR は無視される)
        sw = np.concatenate([np.ones(len(y_tr)),
                             np.full(len(pseudo), PSEUDO_FRAC)])
        p = fit_predict(make, augX, augy, B_te, sw=sw)
        p = smooth_by_board(p, sn_b, board_b, SMOOTH, clip=CL)
        out[f"st{k}"] = p
        pseudo = p
    return out


# ============================================================
#  honest 検証 (species leave-one-board-out)
# ============================================================
boards = sorted(np.unique(board_all))
methods = ["base", "smooth"] + [f"st{k}" for k in range(1, ITERS + 1)]
print(f"PREP={PREP}  ITERS={ITERS}  bases={list(BASES)}  "
      f"LOSO over {len(boards)} boards", flush=True)

t0 = time.time()
loso = {m: {mt: np.zeros(len(yall)) for mt in methods} for m in BASES}
for b in boards:
    te_idx = np.where(board_all == b)[0]
    tr_idx = np.where(board_all != b)[0]
    pp = SpectralPreprocessor(PREP, CONFIG)
    A_tr = pp.fit(Xall[tr_idx]).transform(Xall[tr_idx])
    B_te = pp.transform(Xall[te_idx])
    sn_b = sn_all[te_idx]; bd_b = board_all[te_idx]
    for mname, make in BASES.items():
        res = self_train(make, A_tr, yall[tr_idx], B_te, sn_b, bd_b, ITERS)
        for mt in methods:
            loso[mname][mt][te_idx] = res[mt]
    print(f"  board {b:2d} done ({time.time()-t0:.0f}s)", flush=True)

def rmse(p): return float(np.sqrt(np.mean((p - yall) ** 2)))
print("\n=== species-LOSO RMSE (≒ LB) ===")
rows = []
for mname in BASES:
    for mt in methods:
        r = rmse(loso[mname][mt])
        rows.append({"base": mname, "method": mt, "loso_rmse": round(r, 3)})
        print(f"  {mname:6s} {mt:7s}  {r:6.2f}")
dfl = pd.DataFrame(rows)
os.makedirs("submissions_manual", exist_ok=True)
dfl.to_csv("submissions_manual/pseudo_loso.csv", index=False)

# 各 base で「自己学習が base/smooth を改善したか」を判定
print("\n=== 判定 (自己学習が base を改善したか) ===")
verdict = {}
for mname in BASES:
    base_r = rmse(loso[mname]["base"])
    best_mt = min(methods, key=lambda mt: rmse(loso[mname][mt]))
    best_r = rmse(loso[mname][best_mt])
    verdict[mname] = (best_mt, best_r, base_r)
    tag = "改善" if best_r < base_r - 1e-6 else "改善せず(自己学習は不採用)"
    print(f"  {mname:6s}: base {base_r:.2f} → best={best_mt} {best_r:.2f}  [{tag}]")

# ============================================================
#  本番 test 予測の生成 (LOSO で最良だった手法で)
# ============================================================
print("\n=== 本番 test 予測を生成 ===")
pp = SpectralPreprocessor(PREP, CONFIG)
A_all = pp.fit(Xall).transform(Xall)
B_real = pp.transform(Xte)

def save(name, pred):
    pd.DataFrame({0: ids, 1: pred}).to_csv(
        os.path.join("submissions_manual", name), index=False, header=False)

global_best = None
for mname, make in BASES.items():
    res = self_train(make, A_all, yall, B_real, sn_te, board_te, ITERS)
    best_mt, best_r, base_r = verdict[mname]
    save(f"sub_pseudo_{mname}_{best_mt}.csv", res[best_mt])
    print(f"  saved sub_pseudo_{mname}_{best_mt}.csv  (LOSO {best_r:.2f})")
    if global_best is None or best_r < global_best[1]:
        global_best = (f"sub_pseudo_{mname}_{best_mt}.csv", best_r, mname, best_mt)

# 2 base の最良手法を等重み blend (多様性)
b1 = self_train(BASES["SVR10"], A_all, yall, B_real, sn_te, board_te, ITERS)[verdict["SVR10"][0]]
b2 = self_train(BASES["LGB"],   A_all, yall, B_real, sn_te, board_te, ITERS)[verdict["LGB"][0]]
save("sub_pseudo_blend.csv", smooth_by_board((b1 + b2) / 2, sn_te, board_te, SMOOTH, clip=CL))
print("  saved sub_pseudo_blend.csv (SVR/LGB 最良手法の平均→平滑化)")

print(f"\n  >>> 推奨提出: {global_best[0]}  (species-LOSO {global_best[1]:.2f} ≒ 期待LB)")
print("  ※ ただし上の判定が全 base 『改善せず』なら、擬似ラベルは効いていない。")
print("     その場合は search_best.py の単体/blend (~17) を提出すること。")
print(f"DONE {time.time()-t0:.0f}s", flush=True)
