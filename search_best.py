"""
search_best.py ─ honest 総当たりサーチ (あなたが実行 → ログから最良を提出)
================================================================================
species 単位 leave-one-board-out CV は実測で LB に一致する
  (例: SVR(C=10)×emsc+sg_d1  species-CV 17.95 / LGB×emsc+sg_d1 19.42 ≒ 実LB 19.56)。
そこで「多数の前処理 × 多数のモデル(パラメータ違い) × 連結特徴 × 貪欲アンサンブル」を
総当たりし、CV(≒LB)で最良のものを選ぶ。**CV が honest なので「CV 最良 ≒ LB 最良」**。

使い方
------
    python search_best.py            # フル探索 (時間がかかる。下の QUICK=False)
    # まず動作確認したいときは QUICK=True にして実行 (数分)

出力 (submissions_manual/)
  - search_ranking.csv              全候補(単体+アンサンブル)の species-CV ランキング
  - sub_search_best_single.csv      単体 CV 最良
  - sub_search_top2.csv / top3.csv  単体 2,3 位
  - sub_search_greedy_ensemble.csv  貪欲アンサンブル(Caruana, 等重み積み上げ) ← 多くは最良
  ログ末尾に「>>> 推奨提出」を表示する。その CSV を提出する。

★ EXPERIMENTS.md は参考程度: 実際にこのサーチの honest CV(≒LB)で決めること
  (例: 友人は「SVR はダメ」と書いたが C=10+EMSC では本サーチで単体最良だった)。
※ 特徴量選択(filter/QUBO 等)の比較は run_all.py 側で行う。本サーチは精度重視で
  全波長(+連結)を対象にする (このデータでは特徴選択は精度を悪化させたため)。
"""
import os, sys, re, time, warnings
import numpy as np, pandas as pd
from collections import Counter
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from preprocessing import SpectralPreprocessor
from target import forward_transform as ft, inverse_transform as it, clip_predictions as clip

from sklearn.model_selection import GroupKFold
import lightgbm as lgb, xgboost as xgb
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import (
    ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import mutual_info_regression

# ============================================================
#  ユーザ設定 (ここを編集して探索範囲を変えられる)
# ============================================================
QUICK          = False   # True: 少数で高速確認(数分) / False: フル探索(数時間)
#   環境変数でも切替可: SEARCH_QUICK=1 python search_best.py
if os.environ.get("SEARCH_QUICK", "").lower() in ("1", "true", "yes"):
    QUICK = True
N_SPLITS       = 5       # species GroupKFold の分割数
GBM_SEEDS      = 3       # GBM の full-data テスト予測のシード平均数 (安定化)
GREEDY_POOL    = 40      # 貪欲アンサンブルの候補プール (CV 上位 N からのみ選ぶ)
GREEDY_ITERS   = 30      # 貪欲アンサンブルの最大反復 (等重みで積み上げ)
TOPN_SUBMIT    = 3       # 単体上位 N も提出 CSV にする

# ---- 前処理 (単体): 「+」で連結適用。preprocessing.py の全ステップが使える ----
SINGLES_FULL = [
    "snv", "msc", "emsc",
    "snv+sg_d1", "snv+sg_d2", "msc+sg_d1", "msc+sg_d2",
    "emsc+sg_d1", "emsc+sg_d2",
    "snv+sg_d1+standard", "sg_d1+standard", "sg_d2+standard",
    "snv+detrend+sg_d1", "snv+sg_d1+l2norm", "emsc+detrend+sg_d1",
    "snv+sg_d1+dwt", "emsc+sg_d1+dwt", "emsc+sg_d1+standard",
]
SINGLES_QUICK = ["emsc+sg_d1", "snv+sg_d1", "msc+sg_d2", "snv+sg_d1+dwt"]

# ---- 連結特徴 (複数前処理を横連結 = 友人の A/B/C 流)。GBM のみ適用(高次元) ----
CONCAT_FULL = {
    "C2":  "concat:emsc+sg_d1|msc+sg_d2",
    "C2d": "concat:emsc+sg_d1|snv+sg_d1+dwt",
    "C3":  "concat:emsc+sg_d1|snv+sg_d1+dwt|msc+sg_d2",
    "C3b": "concat:emsc+sg_d1|snv+detrend+sg_d1|msc+sg_d2",
    "C4":  "concat:emsc+sg_d1|snv+detrend+sg_d1|snv+sg_d1+l2norm|msc+sg_d2",
    "C5":  "concat:emsc+sg_d1|snv+sg_d1+dwt|msc+sg_d2|emsc|snv+detrend+sg_d1",
}
CONCAT_QUICK = {"C3": "concat:emsc+sg_d1|snv+sg_d1+dwt|msc+sg_d2"}

# ---- モデル (パラメータ違いを多数)。is_gbm=True は連結特徴にも適用 ----
def lgbf(nl, lr, cs, n):
    return lambda s=0: lgb.LGBMRegressor(
        n_estimators=n, learning_rate=lr, num_leaves=nl, subsample=0.8,
        subsample_freq=1, colsample_bytree=cs, min_child_samples=20,
        reg_lambda=2.0, random_state=s, verbose=-1, n_jobs=-1)
def xgbf(d, lr, cs, n):
    return lambda s=0: xgb.XGBRegressor(
        n_estimators=n, learning_rate=lr, max_depth=d, subsample=0.8,
        colsample_bytree=cs, reg_lambda=2.0, min_child_weight=1,
        random_state=s, verbosity=0, n_jobs=-1)
def histf(lr, leaves, l2):
    return lambda s=0: HistGradientBoostingRegressor(
        learning_rate=lr, max_leaf_nodes=leaves, l2_regularization=l2,
        max_iter=600, random_state=s)
def svrf(C, eps=0.05):
    return lambda s=0: SVR(kernel="rbf", C=C, epsilon=eps, gamma="scale")
def krrf(a, g):
    return lambda s=0: KernelRidge(kernel="rbf", alpha=a, gamma=g)
def plsf(nc):
    return lambda s=0: PLSRegression(n_components=nc)
def ridgef(a):
    return lambda s=0: Ridge(alpha=a)
def enetf(a, l1):
    return lambda s=0: ElasticNet(alpha=a, l1_ratio=l1, max_iter=10000)
def etf(n):
    return lambda s=0: ExtraTreesRegressor(n_estimators=n, random_state=s, n_jobs=-1)
def rff(n):
    return lambda s=0: RandomForestRegressor(n_estimators=n, random_state=s, n_jobs=-1)
def knnf(k):
    return lambda s=0: KNeighborsRegressor(n_neighbors=k, weights="distance")

#                name           factory                 is_gbm
MODELS_FULL = [
    ("LGB_nl7",   lgbf(7,  0.05, 0.6, 1500), True),
    ("LGB_nl15",  lgbf(15, 0.03, 0.5, 800),  True),
    ("LGB_nl31",  lgbf(31, 0.03, 0.4, 900),  True),
    ("LGB_nl31c3",lgbf(31, 0.03, 0.3, 1100), True),
    ("LGB_nl63",  lgbf(63, 0.02, 0.3, 1200), True),
    ("XGB_d2",    xgbf(2,  0.05, 0.5, 1000), True),
    ("XGB_d3",    xgbf(3,  0.05, 0.5, 800),  True),
    ("XGB_d3c3",  xgbf(3,  0.03, 0.3, 1100), True),
    ("XGB_d4",    xgbf(4,  0.03, 0.4, 700),  True),
    ("Hist_l31",  histf(0.05, 31, 0.0),      True),
    ("Hist_l15",  histf(0.05, 15, 1.0),      True),
    ("SVR_C5",    svrf(5),   False),
    ("SVR_C10",   svrf(10),  False),
    ("SVR_C20",   svrf(20),  False),
    ("SVR_C50",   svrf(50),  False),
    ("KRR",       krrf(1.0, 1e-3), False),
    ("PLS_nc8",   plsf(8),   False),
    ("PLS_nc12",  plsf(12),  False),
    ("PLS_nc16",  plsf(16),  False),
    ("PLS_nc24",  plsf(24),  False),
    ("Ridge_100", ridgef(100),  False),
    ("Ridge_1000",ridgef(1000), False),
    ("ENet",      enetf(0.05, 0.5), False),
    ("ExtraT600", etf(600),  False),
    ("ExtraT300", etf(300),  False),
    ("RF400",     rff(400),  False),
    ("kNN3",      knnf(3),   False),
    ("kNN7",      knnf(7),   False),
]
MODELS_QUICK = [
    ("LGB_nl31c3", lgbf(31, 0.03, 0.3, 1000), True),
    ("XGB_d3",     xgbf(3, 0.05, 0.5, 800),   True),
    ("SVR_C10",    svrf(10),  False),
    ("ExtraT600",  etf(600),  False),
    ("PLS_nc16",   plsf(16),  False),
]

SINGLES = SINGLES_QUICK if QUICK else SINGLES_FULL
CONCAT  = CONCAT_QUICK  if QUICK else CONCAT_FULL
MODELS  = MODELS_QUICK  if QUICK else MODELS_FULL

# ---- 特徴量選択の軸 (None=全波長 / MI上位k を fold内honest で選択) ----
#   ※ このデータでは選択は精度を悪化させがち(全波長が有利)だが、比較のため軸を持つ。
#     選択(k!=None)は計算が重いので STRONG_FOR_SELECT のモデル × 単体前処理のみに適用。
SELECTIONS = ({"all": None, "mi300": 300}
              if QUICK else {"all": None, "mi400": 400, "mi200": 200, "mi100": 100})
STRONG_FOR_SELECT = {"SVR_C10", "LGB_nl31c3", "LGB_nl15", "XGB_d3",
                     "ExtraT600", "PLS_nc16"}

# ============================================================
#  データ
# ============================================================
tr = pd.read_csv("train.csv", encoding=CONFIG["encoding"])
te = pd.read_csv("test.csv",  encoding=CONFIG["encoding"])
spec = list(tr.columns[4:])
X   = tr[spec].values.astype(float)
y   = tr[CONFIG["target_col"]].values.astype(float)
Xte = te[te.columns[3:]].values.astype(float)
ids = te[CONFIG["id_col"]].values
board = tr["species number"].values
sp = list(GroupKFold(N_SPLITS).split(X, y, board))   # species-LOSO ≒ LB
TK = CONFIG.get("target_transform", "log1p")
CL = CONFIG.get("clip_predictions")

REPS = {r: r for r in SINGLES}
REPS.update(CONCAT)
CONCAT_KEYS = set(CONCAT)

def rmse(p):
    return float(np.sqrt(np.mean((p - y) ** 2)))

def get_fold_data(rep):
    """rep の前処理を fold(train fit)+full に適用してキャッシュ (モデル間で再利用)。"""
    folds = []
    for trI, vaI in sp:
        pp = SpectralPreprocessor(rep, CONFIG)
        folds.append((pp.fit(X[trI]).transform(X[trI]), pp.transform(X[vaI])))
    pf = SpectralPreprocessor(rep, CONFIG)
    return folds, pf.fit(X).transform(X), pf.transform(Xte)

# ---- 特徴量選択 (MI 上位 k) を fold 内で honest に行うためのマスク (k 毎にキャッシュ) ----
_mi_fold, _mi_full = {}, {}
def _fold_masks(k):
    if k not in _mi_fold:
        ms = []
        for trI, _ in sp:
            mi = mutual_info_regression(X[trI], y[trI], random_state=42)
            m = np.zeros(X.shape[1], bool); m[np.argsort(mi)[-k:]] = True; ms.append(m)
        _mi_fold[k] = ms
    return _mi_fold[k]
def _full_mask(k):
    if k not in _mi_full:
        mi = mutual_info_regression(X, y, random_state=42)
        m = np.zeros(X.shape[1], bool); m[np.argsort(mi)[-k:]] = True; _mi_full[k] = m
    return _mi_full[k]

def rep_data(rep, k):
    """k=None なら全波長 (キャッシュ)。k!=None なら fold ごとに MI 上位 k を選択して前処理。"""
    if k is None:
        return get_fold_data(rep)
    folds = []
    for (trI, vaI), mf in zip(sp, _fold_masks(k)):
        pp = SpectralPreprocessor(rep, CONFIG)
        folds.append((pp.fit(X[trI][:, mf]).transform(X[trI][:, mf]),
                      pp.transform(X[vaI][:, mf])))
    fm = _full_mask(k)
    pf = SpectralPreprocessor(rep, CONFIG)
    return folds, pf.fit(X[:, fm]).transform(X[:, fm]), pf.transform(Xte[:, fm])

# ============================================================
#  探索
# ============================================================
# 連結特徴は GBM のみ、かつ XGB は高次元で激遅なので除外 (LGB/Hist は高速・高精度)
def _concat_ok(nm, gbm):
    return gbm and not nm.startswith("XGB")

def _applies(k, rk, mname, is_gbm):
    """(selection, rep, model) の組合せを実行するか。"""
    is_concat = rk in CONCAT_KEYS
    if k is not None and is_concat:
        return False                      # 選択は単体前処理のみ
    if k is None:
        return not (is_concat and not _concat_ok(mname, is_gbm))
    return mname in STRONG_FOR_SELECT     # 選択は重いので主力モデルのみ

n_combo = sum(_applies(k, rk, nm, g)
              for k in SELECTIONS.values() for rk in REPS for nm, _, g in MODELS)
print(f"MODE={'QUICK' if QUICK else 'FULL'}  reps={len(REPS)} "
      f"(singles {len(SINGLES)} + concat {len(CONCAT)})  models={len(MODELS)}  "
      f"selections={list(SELECTIONS)}  →  {n_combo} 組合せ × {N_SPLITS}-fold "
      f"(species-LOSO ≒ LB)", flush=True)

t0 = time.time(); done = 0
oof, tst, cv = {}, {}, {}
for selname, k in SELECTIONS.items():
    for rk, rspec in REPS.items():
        ms = [(nm, fc, g) for nm, fc, g in MODELS if _applies(k, rk, nm, g)]
        if not ms:
            continue
        try:
            folds, Af, Bf = rep_data(rspec, k)   # (sel,rep) ごとに前処理は 1 回
        except Exception as e:
            print(f"  [prep SKIP] {rk}/{selname}: {type(e).__name__} {e}", flush=True)
            continue
        for mname, fac, is_gbm in ms:
            key = f"{mname}__{rk}" + ("" if k is None else f"__{selname}")
            done += 1
            try:
                o = np.zeros(len(y))
                for (A, B), (trI, vaI) in zip(folds, sp):
                    m = fac(); m.fit(A, ft(y[trI], TK))
                    o[vaI] = clip(it(np.asarray(m.predict(B)).ravel(), TK), CL)
                seeds = range(GBM_SEEDS) if is_gbm else [0]
                ps = []
                for s in seeds:
                    m = fac(s); m.fit(Af, ft(y, TK))
                    ps.append(clip(it(np.asarray(m.predict(Bf)).ravel(), TK), CL))
                oof[key] = o; tst[key] = np.mean(ps, axis=0); cv[key] = rmse(o)
                print(f"  [{done}/{n_combo}] {key:30s} CV={cv[key]:6.2f}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
            except Exception as e:
                print(f"  {key:30s} SKIP ({type(e).__name__}: {e})", flush=True)

if not cv:
    print("候補がありません。設定を確認してください。"); sys.exit(1)

order = sorted(cv, key=cv.get)
print(f"\n=== 単体 TOP {min(25,len(order))} (species-CV ≒ LB) ===", flush=True)
for k in order[:25]:
    print(f"  {cv[k]:6.2f}  {k}", flush=True)

# ============================================================
#  貪欲アンサンブル (Caruana, 上位プールから等重み積み上げ; CV 過適合に頑健)
# ============================================================
pool = order[:min(GREEDY_POOL, len(order))]
cur, ssum, hist = [], np.zeros(len(y)), []
for _ in range(GREEDY_ITERS):
    bk = min(pool, key=lambda k: rmse((ssum + oof[k]) / (len(cur) + 1)))
    cur.append(bk); ssum = ssum + oof[bk]
    hist.append((tuple(cur), rmse(ssum / len(cur))))
mem = Counter(min(hist, key=lambda x: x[1])[0])
ens_cv = min(h[1] for h in hist)
tot = sum(mem.values())
ens_test = sum(c * tst[k] for k, c in mem.items()) / tot
print(f"\n=== 貪欲アンサンブル species-CV = {ens_cv:.2f} ===", flush=True)
for k, c in mem.most_common():
    print(f"  {c}x {k}", flush=True)

# ============================================================
#  出力
# ============================================================
out = os.path.join(CONFIG["data_dir"], "submissions_manual")
os.makedirs(out, exist_ok=True)

rows = [{"rank": i + 1, "candidate": k, "species_cv": round(cv[k], 4)}
        for i, k in enumerate(order)]
rows.append({"rank": "ENS", "candidate": f"greedy {dict(mem)}",
             "species_cv": round(ens_cv, 4)})
pd.DataFrame(rows).to_csv(os.path.join(out, "search_ranking.csv"), index=False)

def save(name, pred):
    pd.DataFrame({0: ids, 1: pred}).to_csv(
        os.path.join(out, name), index=False, header=False)

save("sub_search_greedy_ensemble.csv", ens_test)
for i in range(min(TOPN_SUBMIT, len(order))):
    nm = "sub_search_best_single.csv" if i == 0 else f"sub_search_top{i+1}.csv"
    save(nm, tst[order[i]])

best_overall = ("greedy_ensemble", ens_cv) if ens_cv <= cv[order[0]] else (order[0], cv[order[0]])
print(f"\n出力 → {out}/")
print(f"  search_ranking.csv  (全 {len(order)} 候補 + 貪欲アンサンブル)")
print(f"  sub_search_greedy_ensemble.csv  (CV {ens_cv:.2f})")
print(f"  sub_search_best_single.csv      ({order[0]}, CV {cv[order[0]]:.2f})")
print(f"\n  >>> 推奨提出: "
      + ("sub_search_greedy_ensemble.csv" if best_overall[0] == "greedy_ensemble"
         else "sub_search_best_single.csv")
      + f"  (species-CV {best_overall[1]:.2f} ≒ 期待LB)")
print(f"  ※ CV は honest(species-LOSO)なので LB もこの近辺になる見込み。最終判断は LB で。")
print(f"DONE  {time.time()-t0:.0f}s", flush=True)
