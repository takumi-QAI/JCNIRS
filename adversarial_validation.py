"""
adversarial_validation.py ─ 学習データとテストデータの分布ずれを定量診断する。
=====================================================================
kagglebook ch07-03 の adversarial validation を、この NIR コンペ
(test=6 ボード丸ごと held-out) に合わせて実装したもの。

何のため？
  「species-LOSO CV が ~17 なのに 過去の実 LB は ~14」というズレの正体を見るため。
  test が train とどれだけ違うか、どの train ボードが test に似ているかを測る。

やること:
  1. train/test スペクトルを結合し「test らしさ」を当てる 2 値分類器を CV で学習。
     - OOF AUC ≈ 0.5  → train と test は見分けがつかない = CV は LB に一致しやすい
     - OOF AUC 高い    → 強い分布ずれ = 一様な LOSO は LB を当てない。test に似た
                          ボードでの成績を重視すべき。
  2. test と train を分ける波長 (重要度) を表示。
  3. 各 train サンプル/ボードの「test らしさ」を算出し CSV 保存。
     → test に似たボードほど、その held-out 成績が実 LB をよく予測する。

実行: python adversarial_validation.py
"""

import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(errors="backslashreplace")
except Exception:
    pass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

from config import CONFIG
from preprocessing import SpectralPreprocessor

PREP = os.environ.get("ADV_PREP", "emsc+sg_d1")   # 詳細解析に使う前処理
N_SPLITS = 5
# train/test のズレをどの前処理が消せるか比較する候補 (_t = 参照を train+test 合算
# にするトランスダクティブ版。理屈上ズレを減らせる)
SCAN_PREPS = ["raw", "snv", "msc", "emsc", "msc_t", "emsc_t",
              "snv+sg_d1", "emsc+sg_d1", "emsc_t+sg_d1", "snv+sg_d2",
              "snv+detrend+sg_d1", "snv+sg_d1+l2norm"]


def _adv_auc(Xtr, Xte, return_oof=False):
    """train(0)/test(1) を当てる分類器を CV し OOF AUC を返す。"""
    X = np.vstack([Xtr, Xte])
    yadv = np.r_[np.zeros(len(Xtr)), np.ones(len(Xte))]
    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=42)
    oof = np.zeros(len(X)); imp = np.zeros(X.shape[1])
    for trI, vaI in skf.split(X, yadv):
        clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                                 num_leaves=31, subsample=0.8, colsample_bytree=0.5,
                                 reg_lambda=2.0, random_state=42, verbose=-1, n_jobs=-1)
        clf.fit(X[trI], yadv[trI])
        oof[vaI] = clf.predict_proba(X[vaI])[:, 1]
        imp += clf.feature_importances_
    auc = roc_auc_score(yadv, oof)
    return (auc, oof, imp) if return_oof else auc


def main():
    enc = CONFIG["encoding"]
    tr = pd.read_csv("train.csv", encoding=enc)
    te = pd.read_csv("test.csv",  encoding=enc)
    spec = list(tr.columns[4:])
    wl = np.array([float(c) for c in spec]) if all(
        c.replace(".", "").replace("-", "").isdigit() for c in spec) else np.arange(len(spec))
    Xtr_raw = tr[spec].values.astype(float)
    Xte_raw = te[te.columns[3:]].values.astype(float)
    board = tr["species number"].values

    print(f"=== Adversarial Validation ===")
    print(f"  train {len(Xtr_raw)} 件 / test {len(Xte_raw)} 件 / 波長 {len(spec)}")

    def prep_xy(name):
        if name == "raw":
            return Xtr_raw.copy(), Xte_raw.copy()
        pp = SpectralPreprocessor(name, CONFIG)
        return pp.fit(Xtr_raw).transform(Xtr_raw), pp.transform(Xte_raw)

    # ---- 前処理スキャン: どの前処理が train/test のズレを最も消すか (AUC 低=良) ----
    print("\n  ▼ 前処理別 train/test 分離 AUC (1.0=完全に別物, 0.5=見分け不能):")
    scan = []
    for name in SCAN_PREPS:
        try:
            scan.append((name, _adv_auc(*prep_xy(name))))
        except Exception as e:
            print(f"      {name:22s} SKIP ({type(e).__name__})")
    for name, a in sorted(scan, key=lambda t: t[1]):
        mark = " ← 最もズレ小" if (scan and name == min(scan, key=lambda t: t[1])[0]) else ""
        print(f"      {name:22s} AUC={a:.3f}{mark}")

    # ---- 詳細解析 (PREP) ----
    Xtr, Xte = prep_xy(PREP)
    auc, oof, imp = _adv_auc(Xtr, Xte, return_oof=True)
    print(f"\n  ▼ 詳細解析 (前処理={PREP}):  OOF AUC = {auc:.3f}")
    if auc < 0.6:
        print("    → train/test はほぼ見分けがつかない。species-LOSO CV は LB に一致しやすい。")
    elif auc < 0.8:
        print("    → 中程度の分布ずれ。一様 LOSO は実 LB をやや外す。test に似たボード重視が有効。")
    else:
        print("    → 強い分布ずれ! 一様 LOSO は LB を当てない。test らしいボードの成績/重みで選ぶべき。")

    # 分布ずれを生む波長 (top)
    top = np.argsort(imp)[::-1][:15]
    print("\n  ▼ train/test を分ける波長 top15 (重要度):")
    print("    " + ", ".join(f"{wl[i]:.0f}" for i in top))

    # 各 train サンプルの test らしさ → ボード単位に集計
    tl = oof[:len(Xtr)]
    dfb = (pd.DataFrame({"board": board, "test_likeness": tl})
           .groupby("board")["test_likeness"].mean().sort_values(ascending=False))
    print("\n  ▼ test に似た train ボード (= その held-out 成績が実 LB をよく予測):")
    for b, v in dfb.head(6).items():
        print(f"      board {int(b):>3d} : {v:.3f}")
    print("  ▼ test に似ていない train ボード (LOSO で過度に悲観的になりがち):")
    for b, v in dfb.tail(6).items():
        print(f"      board {int(b):>3d} : {v:.3f}")

    # 保存 (search_best/run_research から重み付き CV に使える)
    out = pd.DataFrame({"id": tr[CONFIG["id_col"]].values, "board": board,
                        "test_likeness": tl})
    os.makedirs("submissions_manual", exist_ok=True)
    path = os.path.join("submissions_manual", "adversarial_test_likeness.csv")
    out.to_csv(path, index=False)
    print(f"\n  test らしさ重み → {path} に保存 (重み付き CV/選定に利用可)")
    print("\n  使い方: AUC が高いなら、提出選定は『test に似たボードでの成績』や"
          "\n          test らしさで重み付けした RMSE を基準にすると実 LB に近づく。")


if __name__ == "__main__":
    main()
