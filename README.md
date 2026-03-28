# JCNIRS — NIR 分光データによる木材含水率予測パイプライン

近赤外 (NIR) 分光データから木材の含水率を機械学習で予測するプロジェクトです。

---

## ファイル構造

```
JCNIRS/
├── JCNIRS.py                     # コアパイプライン（モデル学習・予測・提出ファイル生成）
├── JCNIRS_feature_selection.py   # 特徴量選択比較パイプライン（メイン実行ファイル）
├── Dockerfile                    # Docker環境定義
├── train.csv                     # 学習データ
├── test.csv                      # テストデータ
├── sample_submit.csv             # 提出フォーマットのサンプル
└── README.md                     # このファイル
```

### 実行後に生成されるファイル

```
JCNIRS/
├── submissions_none/             # ① 特徴量選択なし の提出ファイル群
├── submissions_filter/           # ② フィルター法 (MI) の提出ファイル群
├── submissions_wrapper/          # ③ ラッパー法 (RFE) の提出ファイル群
├── submissions_embedded/         # ④ 埋め込み法 (Lasso) の提出ファイル群
├── submissions_amplify/          # ⑤ Amplify QUBO の提出ファイル群（要SDKインストール）
├── heatmap_cv_rmse.png           # モデル×前処理のCV-RMSEヒートマップ
├── barplot_best_cv_rmse.png      # 各モデルのベストCV-RMSE棒グラフ
└── feature_selection_comparison.png  # 特徴量選択手法の比較グラフ
```

---

## 実行方法

### Docker を使う（推奨）

#### 1. イメージのビルド（初回のみ）
```powershell
docker build -t jcnirs .
```

#### 2. 実行
```powershell
# JCNIRS_feature_selection.py を実行（デフォルト）
docker run --rm --name jcnirs_run -v ${PWD}:/app jcnirs

# JCNIRS.py を実行
docker run --rm --name jcnirs_run -v ${PWD}:/app jcnirs python JCNIRS.py
```

#### 3. シェルに入って手動操作する場合
```powershell
docker run --rm -it --name jcnirs_shell -v ${PWD}:/app jcnirs bash
```
コンテナ内で：
```bash
python JCNIRS_feature_selection.py
python JCNIRS.py
```

---

### Python に直接インストールして使う

```powershell
pip install numpy pandas matplotlib seaborn scikit-learn scipy xgboost lightgbm torch
```

インストール後：
```powershell
python JCNIRS_feature_selection.py
python JCNIRS.py
```

---

## 特徴量選択手法（JCNIRS_feature_selection.py）

| 手法 | 説明 |
|------|------|
| ① 選択なし | 全波長をそのまま使用（ベースライン） |
| ② フィルター法 | 相互情報量 (MI) で上位 200 特徴量を選択 |
| ③ ラッパー法 | RFE + LightGBM で上位 200 特徴量を選択 |
| ④ 埋め込み法 | Lasso の係数絶対値で上位 200 特徴量を選択 |
| ⑤ Amplify QUBO | mRMR を QUBO 定式化し Amplify AE で求解（要 `pip install amplify`） |

---

## モデル一覧（JCNIRS.py）

| モデル名 | 種類 |
|----------|------|
| PLS | 偏最小二乗回帰 |
| PCR | 主成分回帰 |
| Ridge | Ridge 回帰 |
| Lasso | Lasso 回帰 |
| ElasticNet | ElasticNet 回帰 |
| SVR | サポートベクター回帰 |
| RandomForest | ランダムフォレスト |
| XGBoost | 勾配ブースティング |
| LightGBM | 勾配ブースティング（高速） |
| 1D-CNN | 1次元畳み込みニューラルネット（要 PyTorch） |

---

## 前処理一覧

| 前処理名 | 説明 |
|----------|------|
| standard | StandardScaler |
| snv | Standard Normal Variate |
| msc | Multiplicative Scatter Correction |
| sg_d1 | Savitzky-Golay 1次微分 |
| sg_d2 | Savitzky-Golay 2次微分 |
| sg_d1+standard | 1次微分 → StandardScaler |
| sg_d2+standard | 2次微分 → StandardScaler |
| snv+sg_d1 | SNV → 1次微分 |

---

## アンサンブル手法

各モデル×前処理のCV結果をもとに以下の3手法でアンサンブルを実施：

| 手法 | 説明 |
|------|------|
| Stacking Ridge | Ridge をメタ学習器としたスタッキング |
| Stacking Lasso | Lasso をメタ学習器としたスタッキング |
| Weighted Average | SLSQP 最適化による加重平均 |

---

## 提出ファイルについて

各戦略のフォルダ内に以下が生成されます：

- `sub_[モデル名]__[前処理名].csv` — 個別モデル×前処理の予測
- `sub_ensemble_stacking_ridge.csv` — スタッキング Ridge
- `sub_ensemble_stacking_lasso.csv` — スタッキング Lasso
- `sub_ensemble_weighted_avg.csv` — 加重平均
- `sub_BEST.csv` — CV-RMSE が最も低い手法の予測（これを提出）
