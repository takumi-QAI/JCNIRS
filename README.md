# JCNIRS — NIR 分光データによる木材含水率予測パイプライン

近赤外 (NIR) 分光データから木材の含水率を機械学習で予測するプロジェクトです。
SIGNATE「近赤外研究会 スペクトル分析チャレンジ」用のコードで、研究テーマは
**「量子アニーリング (Amplify QUBO) による特徴量選択の有用性」** です。

最終目標:
1. **木材 NIR スペクトルからの含水率予測精度の向上**
2. **特徴量選択における量子アニーリングの有用性の提示**

> 📑 **指標と図の詳しい意味は [docs/REPORT_GUIDE.md](docs/REPORT_GUIDE.md) にまとめています。**
> 研究レポートを書くときはこちらを参照してください。

---

## 目次
- [クイックスタート](#クイックスタート)
- [ファイル構造](#ファイル構造)
- [出力されるもの](#出力されるもの)
- [処理パイプライン](#処理パイプライン)
- [モデル一覧](#モデル一覧models)
- [前処理一覧](#前処理一覧)
- [特徴量選択手法](#特徴量選択手法)
- [評価指標](#評価指標)
- [可視化（出力図カタログ）](#可視化出力図カタログ)
- [精度向上のための工夫](#精度向上のための工夫)
- [アンサンブル手法](#アンサンブル手法)
- [設定（config.py の主要ノブ）](#設定configpy-の主要ノブ)
- [提出ファイル](#提出ファイル)
- [結果の解釈ガイド](#結果の解釈ガイド)

---

## クイックスタート

これ 1 本で全戦略・全モデル・全前処理 + 指標算出 + 図の生成まで走ります:

```powershell
python run_all.py
```

### Docker（推奨）

```powershell
docker build -t jcnirs .                                          # 初回のみ
docker run --rm --gpus all --name jcnirs_run -v ${PWD}:/app jcnirs  # GPU 実行
docker run --rm -it --gpus all -v ${PWD}:/app jcnirs bash          # 手動操作
```

> 🎮 **GPU 利用**: Dockerfile は CUDA 版 PyTorch を導入済みなので、NVIDIA GPU が
> あれば `docker run` に **`--gpus all`** を付けるだけで深層モデルが GPU で動きます
> （コードは `models/base.py` の `get_device()` で CUDA を自動選択）。
> `--gpus all` を付けなければ自動的に CPU で実行されます。
> GPU が無い環境では Dockerfile の torch 行を CPU ビルド
> (`--index-url https://download.pytorch.org/whl/cpu`) に戻してください。

### Python に直接インストール

```powershell
pip install numpy pandas matplotlib seaborn scikit-learn scipy xgboost lightgbm torch python-dotenv
pip install amplify   # 量子アニーリング特徴量選択を使う場合
python run_all.py
```

> ⚠ **実行時間**: 既定は **全16モデル × 全9前処理 × 全5特徴量選択 × 5-fold CV** で、
> 深層モデルを含むため CPU では数時間規模（GPU 推奨）。速度優先なら `config.py` の
> `CONFIG["models"]` / `CONFIG["preprocessors"]` / `CONFIG_FS["strategies"]` を
> コメントアウトして絞り込めます。

> 🔑 **量子アニーリング (amplify) の利用**: `pip install amplify` に加え、Fixstars Amplify の
> トークンを `.env` に `AMPLIFY_TOKEN=...` として設定してください。未設定だと amplify 戦略は
> 自動でスキップされます。

---

## ファイル構造

```
JCNIRS/
├── run_all.py                   # ★ 唯一の実行入口（全戦略×全モデル×全前処理＋図）
├── config.py                    # 全パラメータを一元管理（CONFIG / CONFIG_FS）
├── data.py                      # データ読み込み + species one-hot + 波長配列
├── preprocessing.py             # 前処理を集約（SpectralPreprocessor）
├── pipeline.py                  # CV / 指標算出 / スタッキング / 加重平均 / 提出生成
├── metrics.py                   # 評価指標の計算 + 各指標の説明（METRIC_INFO）
├── target.py                    # 目的変数の変換（log1p）と予測クリップ
├── visualization.py             # 全プロット関数（EDA / 戦略別 / 戦略横断）
├── feature_selection.py         # 古典的特徴量選択（none / filter / wrapper / embedded）
├── feature_selection_quantum.py # 量子アニーリング特徴量選択（Amplify QUBO のみ）
├── models/                      # 機械学習モデル（1モデル = 1ファイル）
│   ├── __init__.py              # モデルレジストリ + build_all_models()
│   ├── base.py                  # PyTorch 共通基底（入力標準化＋早期終了）
│   ├── pls.py pcr.py ridge.py lasso.py elasticnet.py svr.py
│   ├── random_forest.py xgboost_model.py lightgbm_model.py
│   ├── cnn1d.py autoencoder.py sae.py vae.py gan.py deepspectra.py transformer.py
├── docs/REPORT_GUIDE.md         # 📑 指標・図の詳細解説（レポート用）
├── Dockerfile
├── train.csv / test.csv / sample_submit.csv
└── README.md
```

---

## 出力されるもの

```
submissions_none/  submissions_filter/  submissions_wrapper/
submissions_embedded/  submissions_amplify/        # 各戦略の提出ファイル群
figures/
├── eda/                                           # データ概観
├── strategies/<strategy>/                         # 戦略ごと（モデル比較など）
└── comparison/                                     # 戦略横断（量子アニーリング比較）
```

図はカテゴリ別のサブフォルダに整理されます（ファイル名は変わりません）。
中身は [可視化（出力図カタログ）](#可視化出力図カタログ) を参照。

---

## 処理パイプライン

```
data.load_data            CSV 読み込み・one-hot・波長取得
        │
visualization (EDA)       含水率分布・スペクトル概観
        │
各特徴量選択戦略について:
  feature_selection(_quantum)   スペクトルから特徴量を選択
        │
  pipeline.run_full_evaluation  全モデル×前処理を 5-fold CV（log1p 学習→逆変換→指標算出）
        │
  pipeline.run_stacking / run_weighted_average   アンサンブル
        │
  pipeline.create_all_submissions                提出 CSV 生成
        │
  visualization (戦略別)   ヒートマップ・予測 vs 実測・残差
        │
visualization (戦略横断)   戦略×指標比較・選択波長・Jaccard・関連度-冗長度 …
```

---

## モデル一覧（models/）

| モデル名 | 種類 | ファイル |
|----------|------|----------|
| PLS | 偏最小二乗回帰 | `models/pls.py` |
| PCR | 主成分回帰 | `models/pcr.py` |
| Ridge | Ridge 回帰 | `models/ridge.py` |
| Lasso | Lasso 回帰 | `models/lasso.py` |
| ElasticNet | ElasticNet 回帰 | `models/elasticnet.py` |
| SVR | サポートベクター回帰 | `models/svr.py` |
| RandomForest | ランダムフォレスト | `models/random_forest.py` |
| XGBoost | 勾配ブースティング | `models/xgboost_model.py` |
| LightGBM | 勾配ブースティング（高速） | `models/lightgbm_model.py` |
| 1D-CNN | 1次元畳み込み NN | `models/cnn1d.py` |
| AE | 自己符号化器（再構成＋回帰ヘッド同時学習） | `models/autoencoder.py` |
| SAE | スパース自己符号化器 | `models/sae.py` |
| VAE | 変分自己符号化器 | `models/vae.py` |
| GAN | 回帰 GAN（G が含水率を予測、D が真偽を識別） | `models/gan.py` |
| DeepSpectra | 1D Inception CNN | `models/deepspectra.py` |
| Transformer | スペクトル Transformer（パッチ埋め込み） | `models/transformer.py` |

深層モデル（1D-CNN 以降）は PyTorch を要し、未インストール時は自動スキップ。

### モデルの追加方法
`models/` に新ファイルを作り `TYPE` / `REQUIRES_TORCH` / `build(params, config)` を定義 →
`models/__init__.py` の import に追加 → `config.py` の `CONFIG["models"]` にエントリ追加。
PyTorch モデルは `models/base.py` の `TorchRegressorBase` を継承し `build_module()` を実装。

---

## 前処理一覧

| 前処理名 | 説明 |
|----------|------|
| standard | StandardScaler |
| snv | Standard Normal Variate（散乱補正） |
| msc | Multiplicative Scatter Correction |
| sg_d1 | Savitzky-Golay 1次微分 |
| sg_d2 | Savitzky-Golay 2次微分 |
| sg_d1+standard | 1次微分 → StandardScaler |
| sg_d2+standard | 2次微分 → StandardScaler |
| snv+sg_d1 | SNV → 1次微分 |
| snv+sg_d1+standard | SNV → 1次微分 → StandardScaler（NIR 定番） |

「+」で連結すると順に適用されます（`preprocessing.py`）。

---

## 特徴量選択手法

| 手法 | 説明 | ファイル |
|------|------|----------|
| ① 選択なし (none) | 全波長をそのまま使用（ベースライン） | `feature_selection.py` |
| ② フィルター法 (filter) | 相互情報量 (MI) で上位 200 特徴量 | `feature_selection.py` |
| ③ ラッパー法 (wrapper) | RFE + LightGBM で上位 200 特徴量 | `feature_selection.py` |
| ④ 埋め込み法 (embedded) | Lasso の係数絶対値で上位 200 特徴量 | `feature_selection.py` |
| ⑤ Amplify QUBO (amplify) | mRMR を QUBO 定式化し Amplify AE で求解（D-Wave 量子アニーリング移行を見据えた L0 選択） | `feature_selection_quantum.py` |

⑤ は量子アニーリング (Amplify) 依存のため別ファイルに分離。`client` を差し替えるだけで
D-Wave へ移行可能な QUBO 形式です。mRMR = **最大関連・最小冗長** の定式化:

```
minimize  -Σ relevance_i · q_i  +  λ·Σ redundancy_ij · q_i · q_j
subject to  Σ q_i = k,  q_i ∈ {0,1}
```

---

## 評価指標

OOF（5-fold 交差検証）予測を元スケール（含水率 [%]）で評価します。
**詳細・式・解釈は [docs/REPORT_GUIDE.md](docs/REPORT_GUIDE.md) を参照。**

| 指標 | 向き | 一言で |
|------|:----:|--------|
| **RMSE** | 低い | 二乗平均平方根誤差。主要指標・モデル選択基準 |
| **MAE** | 低い | 平均絶対誤差。外れ値に頑健。RMSE との差で大外しを検知 |
| **R²** | 高い | 決定係数。説明できた分散の割合（1 が完全） |
| **RPD** | 高い | std(y)/RMSE。NIR 定番。>2.0 良好, >2.5 非常に良好 |
| **RPIQ** | 高い | IQR(y)/RMSE。RPD の頑健版（歪んだ分布向き） |
| **Bias** | 0 | 平均誤差。系統的な過大/過小予測 |
| **MAPE** | 低い | 平均絶対％誤差。低含水率で過大になるため補助的に |
| **MaxError** | 低い | 最大絶対誤差。最悪ケース |

計算する指標は `CONFIG["metrics"]` で変更できます。

---

## 可視化（出力図カタログ）

`figures/` 以下にカテゴリ別サブフォルダで PNG 出力（ファイル名は固定）。図中テキストは
日本語フォント非依存のため英語表記（意味は [docs/REPORT_GUIDE.md](docs/REPORT_GUIDE.md)）。

| サブフォルダ | 内容 |
|--------------|------|
| `figures/eda/` | データ概観（EDA） |
| `figures/strategies/<strategy>/` | 戦略ごと（モデル×前処理の比較・予測診断） |
| `figures/comparison/` | 戦略横断（量子アニーリング vs 古典 FS） |

### EDA（データ概観・1回） → `figures/eda/`
| ファイル | 内容 |
|----------|------|
| `eda_target_distribution.png` | 含水率の分布（生 / 箱ひげ / log1p 後）。log1p 変換の動機 |
| `eda_spectra_overview.png` | 平均スペクトル±std、サンプル波形、波長-含水率の相関 |

### 戦略ごと（`<strategy>_` 接頭辞） → `figures/strategies/<strategy>/`
| ファイル | 内容 |
|----------|------|
| `<strategy>_heatmap_metrics.png` | モデル×前処理の指標ヒートマップ（RMSE/R²/RPD） |
| `<strategy>_barplot_model_metrics.png` | モデル別ベストの指標バー |
| `<strategy>_pred_vs_actual.png` | 予測 vs 実測（ベスト単体＋ベストアンサンブル、1:1 線） |
| `<strategy>_residuals.png` | 残差プロット（残差 vs 予測・残差ヒスト） |

### 戦略横断（量子アニーリング比較＝レポートの核） → `figures/comparison/`
| ファイル | 内容 |
|----------|------|
| `compare_strategy_metrics.png` | 戦略 × 指標（RMSE/R²/RPD/RPIQ）のグループ棒 |
| `compare_features_vs_accuracy.png` | 特徴量数 vs 精度（QUBO の効率性） |
| `compare_runtime.png` | 戦略別の実行時間（量子求解コストの位置づけ） |
| `compare_feature_count.png` | 戦略別の選択特徴量数 |
| `compare_selected_wavelengths.png` | 平均スペクトル上の選択波長（★QUBO の分散性） |
| `compare_feature_overlap.png` | 戦略間の Jaccard 類似度（★QUBO の独自性） |
| `compare_qubo_diagnostics.png` | 関連度 vs 冗長度（★★mRMR 目的の達成度） |

---

## 精度向上のための工夫

本パイプラインに組み込んだ精度向上策（`config.py` で切替可）:

| 工夫 | 内容 | 設定 |
|------|------|------|
| **★ ボード単位 GroupKFold** | 本データは同一ボードの繰り返し測定を含むため、ランダム KFold はリークし CV が極端に楽観的(R²≈0.997)になる。sample number 順の隣接スペクトル相関でボードを推定し、同一ボードを同じ fold に固める honest CV にすることで、汎化するモデルだけが選ばれる | `CONFIG["cv_grouped"]=True`, `group_corr_threshold` |
| **頑健モデルを既定化** | 1322行・実質~150ボードしか無く深層モデルは過学習。既定は PLS/Ridge/SVR/kNN/RF/XGB/LightGBM 等の頑健モデル（正則化強め）。深層モデルはレポート用に `DEEP_MODELS` として温存 | `CONFIG["models"]=dict(ROBUST_MODELS)` |
| **浅いXGBoost** | honest CV 調整で `max_depth=3`＋多本数(800)が最良。浅い木は未知ボードへの汎化に強い（深い木はボードを丸暗記）。例: filter で 15.18→13.59 | `ROBUST_MODELS["XGBoost"]` |
| **目的変数の log1p 変換** | 右裾の重い含水率分布を圧縮し学習を安定化。学習は log1p 空間、評価は逆変換して元スケール | `CONFIG["target_transform"]="log1p"` |
| **予測の物理クリップ** | 含水率は非負・実用上~300%上限。暴走予測を範囲内に抑える安全網（リーク修正前は予測が1900に暴走しPublic 1884になった） | `CONFIG["clip_predictions"]=[0.0, 320.0]` |
| **深層モデルの入力標準化＋早期終了** | 前処理に依らず標準化し、検証損失でベスト重みを復元して過学習抑制（train部分の統計で fit、リーク防止） | `models/base.py` / `models/gan.py` |
| **NIR 定番前処理の追加** | `snv+sg_d1+standard` を前処理候補に追加 | `CONFIG["preprocessors"]` |

> 🔬 **なぜ GroupKFold が決定的か**: ランダム KFold では同一ボードの繰り返しスキャンが
> train/val 両方に入り「丸暗記」が起きる。その結果、リーク CV が選ぶ「ベスト」は
> テストで暴走（例: filter 戦略の予測が全件~1900、Public RMSE 1884）。GroupKFold に
> すると CV-RMSE がリーダーボード実測(~14)と一致し、選ばれる submission が安定する。

---

## アンサンブル手法

各モデル×前処理の CV 結果（CV-RMSE）をもとに 3 手法でアンサンブル:

| 手法 | 説明 |
|------|------|
| Stacking Ridge | Ridge をメタ学習器としたスタッキング |
| Stacking Lasso | Lasso をメタ学習器としたスタッキング |
| Weighted Average | SLSQP 最適化による加重平均（重み ≥0・総和 1） |

---

## 設定（config.py の主要ノブ）

| キー | 役割 | 既定 |
|------|------|------|
| `CONFIG["models"]` | 使用するモデルとハイパラ | 全16モデル |
| `CONFIG["preprocessors"]` | 使用する前処理 | 9種 |
| `CONFIG["n_splits"]` | 交差検証の分割数 | 5 |
| `CONFIG["cv_grouped"]` | ボード単位 GroupKFold（リーク防止） | `True` |
| `CONFIG["group_corr_threshold"]` | ボード境界の隣接相関しきい値 | `0.9999` |
| `CONFIG["target_transform"]` | 目的変数の変換 | `"log1p"` |
| `CONFIG["clip_predictions"]` | 予測のクリップ範囲 | `[0.0, 320.0]` |
| `CONFIG["metrics"]` | 計算・表示する指標 | 8指標 |
| `CONFIG["figure_dir"]` | 図の出力フォルダ | `"figures"` |
| `CONFIG["eda_figures"]` | EDA 図を出すか | `True` |
| `CONFIG["log_dir"]` | 実行ログの出力フォルダ（画面と同時に保存） | `"logs"` |
| `CONFIG_FS["strategies"]` | 実行する特徴量選択戦略 | 5戦略 |
| `CONFIG_FS["amplify"]["token"]` | Amplify トークン | `.env` の `AMPLIFY_TOKEN` |

---

## 提出ファイル

各戦略のフォルダ内に生成:

- `sub_[モデル名]__[前処理名].csv` — 個別モデル×前処理の予測
- `sub_ensemble_stacking_ridge.csv` — スタッキング Ridge
- `sub_ensemble_stacking_lasso.csv` — スタッキング Lasso
- `sub_ensemble_weighted_avg.csv` — 加重平均
- **`sub_BEST_1.csv` 〜 `sub_BEST_10.csv`** — 全候補（個別＋アンサンブル）を性能順に並べた
  **上位10件**。`sub_BEST_1.csv` が最良（**これを提出**）、以降 2位・3位…
- `sub_BEST_ranking.csv` — 上位10件の手法名と各指標（RMSE/R²/RPD/RPIQ…）の一覧表

> 📌 **ランキング基準について**: 候補は **CV-RMSE 昇順**で並べます。**R²・RPD・RPIQ は
> いずれも RMSE の単調関数**（RPD=std(y)/RMSE, RPIQ=IQR(y)/RMSE, R²=1−RMSE²/var(y)）
> なので、**CV-RMSE 順は R²・RPD・RPIQ の順と完全に一致**します。つまり `sub_BEST_1.csv` は
> これら3指標すべてで同時に最良です。`sub_BEST_ranking.csv` でその一致を確認できます。
> （RMSE と順位が食い違いうるのは MAE / Bias / MAPE / MaxError のみ。）

---

## 結果の解釈ガイド

**精度を見るとき**: まず `compare_strategy_metrics.png` と各戦略の
`<strategy>_pred_vs_actual.png` を見て、RPD を NIR 基準（>2.0 良好）と照らす。

**量子アニーリングの有用性を論じるとき**:
1. `compare_strategy_metrics` / `compare_features_vs_accuracy` … 同等以上の精度を少特徴で達成か
2. `compare_qubo_diagnostics` … 高関連・低冗長（mRMR 目的）を達成しているか
3. `compare_selected_wavelengths` / `compare_feature_overlap` … 選択波長が独自で分散的か
4. `compare_runtime` … 計算コストとのトレードオフ

詳しい読み方・主張の組み立ては **[docs/REPORT_GUIDE.md](docs/REPORT_GUIDE.md)** を参照してください。
