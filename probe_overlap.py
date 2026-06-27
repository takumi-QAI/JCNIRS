"""
probe_overlap.py ─ 「test が学習データとボード重複しているか」を1提出で判定する探針。

全学習データで k-NN(スペクトル類似) を作り test を予測する。
- もし test が train と同一/隣接ボードを多く含むなら、最近傍がほぼ正解になり
  Public スコアが極端に小さくなる (重複あり → kNN/ルックアップが最強)。
- 重複が無ければ Public は honest CV と同程度 (~18) になる (天井確定)。

出力: submissions_probe/sub_knn1.csv, sub_knn5.csv
使い方: python probe_overlap.py  → どちらかを提出して Public を確認。
"""
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from config import CONFIG
from data import load_data
from preprocessing import SpectralPreprocessor

(dftr, dfte, Xtr, Xte, Xtrc, Xtec, y, wl, groups) = load_data(CONFIG)

# SNV+SG1 空間 (kNN に効きやすい) で標準化
prep = SpectralPreprocessor("snv+sg_d1+standard", CONFIG)
A = np.hstack([prep.fit(Xtr).transform(Xtr), Xtrc])
B = np.hstack([prep.transform(Xte), Xtec])

out_dir = os.path.join(CONFIG["data_dir"], "submissions_probe")
os.makedirs(out_dir, exist_ok=True)
ids = dfte[CONFIG["id_col"]].values

for k in (1, 5):
    m = KNeighborsRegressor(n_neighbors=k, weights="distance")
    m.fit(A, np.log1p(y))
    pred = np.clip(np.expm1(m.predict(B)), 0, 320)
    pd.DataFrame({0: ids, 1: pred}).to_csv(
        os.path.join(out_dir, f"sub_knn{k}.csv"), index=False, header=False)
    print(f"  sub_knn{k}.csv: mean={pred.mean():.2f} min={pred.min():.2f} "
          f"max={pred.max():.2f}")

# 参考: test 各行の最近傍train距離 (小さいほど重複の疑い)
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=1).fit(A)
dist, _ = nn.kneighbors(B)
print(f"\n  test→train 最近傍距離: median={np.median(dist):.3f} "
      f"min={dist.min():.3f} max={dist.max():.3f}")
print("  → どちらかを提出。Public が極小なら重複あり(kNN路線へ)、~18 なら天井。")
