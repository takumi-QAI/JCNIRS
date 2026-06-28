# JCNIRS — NIR 分光データによる木材含水率予測

近赤外 (NIR) スペクトルから木材の含水率を予測する SIGNATE「近赤外研究会 スペクトル分析
チャレンジ」用コードです。目的は **(1) 予測精度の向上** と **(2) 量子アニーリング
(Amplify QUBO) による特徴量選択の有用性の研究** の 2 つ。

> **すべて Docker で実行できます。** 深層モデルは CPU でも動きます（GPU 任意）。

---

## 🎯 2 つのプログラム（用途で使い分け）

| プログラム | 目的 | 実行コマンド（Docker） |
|---|---|---|
| **`search_best.py`** | **精度の最善探索**。前処理×モデル(古典＋深層, パラメータ違い)×特徴量選択×連結×**擬似ラベル**×アンサンブルを総当たりし、honest CV(≒LB) で最良の構成と提出ファイルを出す | `docker run --rm -v ${PWD}:/app jcnirs` |
| **`run_research.py`** | **研究用の対照実験**。同一の(モデル×前処理)グリッドを固定し、特徴量選択の手法（none/filter/wrapper/embedded/**amplify=QUBO**）だけを変えて**公平に**比較・可視化 | `docker run --rm -v ${PWD}:/app jcnirs python run_research.py` |

提出は **`search_best.py`** の出力（`submissions_manual/` ＋ ログ末尾の「`>>> 推奨提出`」）が基本。
擬似ラベル/transductive も search_best の探索候補に統合済み（honest CV で自動比較される）。

---

## クイックスタート（Docker）

```powershell
docker build -t jcnirs .                                              # 初回のみ

docker run --rm -v ${PWD}:/app -e SEARCH_QUICK=1 jcnirs               # ① まず動作確認（数分）
docker run --rm -v ${PWD}:/app jcnirs                                 # ② 本番フル探索（数時間〜）
docker run --rm -v ${PWD}:/app jcnirs python run_research.py          # 研究（QUBO 比較）
```
`-v ${PWD}:/app` でカレントを共有するので、結果 (`submissions_manual/`, `figures/`,
`logs/`) はホスト側に出ます。深層モデルを GPU で回す場合は Dockerfile の torch を
cu124 ビルドにし、`docker run` に `--gpus all` を付けます（既定は CPU）。

### Docker を使わない場合
```powershell
pip install numpy pandas matplotlib seaborn scikit-learn scipy xgboost lightgbm python-dotenv torch
pip install amplify          # run_research.py の amplify(QUBO) 戦略を使う場合
python search_best.py
```

> 🔑 **量子アニーリング (amplify)**: `pip install amplify` ＋ `.env` に `AMPLIFY_TOKEN=...`。
> 未設定なら run_research.py の amplify 戦略は自動スキップされます。

---

## ⭐ このデータの正体と「honest な精度の天井」（最重要）

数十バージョンの実験で判明した事実。**ここを理解しないと CV に騙されます。**

- データは **19 枚の板（= `species number`）をそれぞれ乾燥させながら `sample number` 順に
  連続スキャンした乾燥時系列**。板内では含水率が単調に減少する滑らかな曲線。
- **test = 6 枚の板を丸ごと held-out**（train に無い未知の板）。= 「未知の6本の乾燥曲線を
  当てる」問題。
- **正しい CV は species 単位 GroupKFold**（`group_by_species=True`）。実測で
  **CV ≒ LB**（LGB×emsc+sg_d1: species-CV 19.42 ≒ 実LB 19.56）。
  ランダム KFold や相関ヒューリスティックは板内リークで楽観的になり当てになりません。
- **honest な精度の天井は ~17**（総当たり・パラメータ調整・アンサンブル・擬似ラベルを
  尽くしても ~16〜18）。最良は **SVR(C=10)×emsc+sg_d1 ≈ 17.9**、貪欲アンサンブルで ~16台。
- リーダーボードの 1 桁スコアは、乾燥曲線が低自由度なことを突いた **LB プロービング**や
  **リーク**（public 過適合）の可能性が高く、honest な汎化とは別物です。

> 📑 図・指標の詳細と研究(QUBO)の考察の組み立ては **[docs/REPORT_GUIDE.md](docs/REPORT_GUIDE.md)**。

---

## 各プログラムの詳細

### `search_best.py` — 精度の最善探索
- **何を総当たりするか**: 前処理（単体18＋連結6）× モデル（**古典 40 種のパラメータ違い**
  ＝LGB/XGB/HistGBM/SVR/KernelRidge/PLS/Ridge/ENet/ExtraTrees/RF/kNN、さらに `SEARCH_DEEP=1`
  で**深層 7 種**＝CNN/AE/SAE/VAE/GAN/DeepSpectra/Transformer）× 特徴量選択（all/MI上位 k）×
  **擬似ラベル/transductive 変種**（強モデルのみ）× 貪欲アンサンブル。
  **honest species-LOSO CV(≒LB)** で評価。
- **1 回実行すれば**: ログ末尾に **★完璧な構成（最良の前処理・モデル・パラメータ）** と
  **`>>> 推奨提出`** が出る。`submissions_manual/search_ranking.csv` に全候補＋パラメータ。
- **編集/トグル**: 冒頭の `SINGLES_FULL / CONCAT_FULL / MODELS_FULL / SELECTIONS / GBM_SEEDS`
  で探索範囲を増減。環境変数 `SEARCH_QUICK=1`（高速確認）/ `SEARCH_DEEP=0`（深層off）/
  `SEARCH_PSEUDO=0`（擬似ラベルoff）。
- ⚠ 注: 実測では擬似ラベルも深層も honest CV を改善しない（ランキングで確認できる）。
  精度の現実的な天井は ~16〜17（SVR(C=10)×emsc+sg_d1 ≈ 17.9、貪欲アンサンブルで ~16台）。

### `run_research.py` — 特徴量選択の対照実験（QUBO 研究、公平版）
- **同一の (モデル×前処理) グリッドを固定**（冒頭 `RESEARCH_MODELS / RESEARCH_PREPS`）し、
  **選択手法だけ**を none/filter/wrapper/embedded/**amplify(QUBO)** と変えて比較。
- ★ **公平性**: 単一の組を恣意的に固定すると相性で結果が偏るため、選択に敏感な
  線形/カーネル（PLS/Ridge/SVR）と内部選択を持つ木（XGB/LGB）を含む同一グリッドで
  評価し、**平均(周辺化) CV-RMSE** で比較する（ログの「公平比較」表）。
- 選択は各 fold 内で実行（リーク防止）。`figures/comparison/` が QUBO 考察の核：
  `compare_strategy_metrics` / `compare_selected_wavelengths` / `compare_feature_overlap` /
  `compare_qubo_diagnostics`（関連度 vs 冗長度）。

---

## ファイル構造

```
JCNIRS/
├── search_best.py        ★ 精度の最善探索（提出はここから。深層・擬似ラベルも探索）
├── run_research.py       ★ 特徴量選択の対照実験（QUBO 研究、公平版）
├── config.py             全パラメータ一元管理（CONFIG / CONFIG_FS / ROBUST/DEEP_MODELS）
├── data.py               データ読み込み + species one-hot + 波長配列
├── preprocessing.py      前処理（SNV/MSC/EMSC/SG微分/detrend/L2/DWT/連結 …）
├── feature_selection.py          古典的選択（none/filter/wrapper/embedded）
├── feature_selection_quantum.py  量子アニーリング選択（Amplify QUBO）
├── pipeline.py           CV / 指標 / アンサンブル / 提出生成（run_research が使用）
├── postprocess.py        乾燥曲線の単調平滑化
├── target.py             目的変数 log1p 変換 + 予測クリップ
├── metrics.py            評価指標（RMSE/MAE/R²/RPD/RPIQ/…）
├── visualization.py      図（EDA / 戦略別 / 戦略横断）
├── models/               機械学習モデル（1モデル=1ファイル）
│   ├── 古典: pls pcr ridge lasso elasticnet svr knn random_forest extratrees
│   │         xgboost_model lightgbm_model
│   └── 深層: base cnn1d autoencoder sae vae gan deepspectra transformer (PyTorch)
├── docs/REPORT_GUIDE.md  📑 指標・図の詳細解説（レポート用）
├── Dockerfile / train.csv / test.csv / sample_submit.csv / README.md
```

---

## 前処理一覧（`preprocessing.py`）

| 名前 | 説明 |
|------|------|
| snv | Standard Normal Variate（散乱補正） |
| msc / emsc | (Extended) Multiplicative Scatter Correction。**emsc が板間差に強い** |
| msc_t / emsc_t | 参照を train+test 合算平均に（トランスダクティブ） |
| sg_d1 / sg_d2 | Savitzky-Golay 1次/2次微分 |
| detrend / l2norm | 波長トレンド除去 / L2 正規化（A系バリアント） |
| dwt | 多重解像度 Haar ウェーブレット係数 |
| wband | 水吸収バンド（~5150/6900 cm⁻¹）特徴 |
| standard / minmax / powertransformer / rankgauss | スケーリング |
| `concat:A\|B\|C` | 複数前処理を**横連結**（友人の A/B/C 流。GBM と相性◎） |

「+」で連結適用（例 `snv+sg_d1+standard`）。**実績上の最有力は `emsc+sg_d1` と連結特徴。**

## モデル一覧（`models/`）
- **古典**: PLS / PCR / Ridge / Lasso / ElasticNet / **SVR** / kNN / RandomForest /
  **ExtraTrees** / **XGBoost** / **LightGBM**
- **深層 (PyTorch)**: 1D-CNN / AE / SAE / VAE / GAN / DeepSpectra / Transformer
  （`search_best.py` の `SEARCH_DEEP=1` で探索に追加。本データでは精度は古典に及ばないが
  「多くの組合せ」を見るため保持。PyTorch 未導入なら自動スキップ）

## 評価指標
RMSE（主指標）/ MAE / R² / RPD / RPIQ / Bias / MAPE / MaxError。
R²・RPD・RPIQ は RMSE の単調関数なので順位は RMSE と一致。詳細は REPORT_GUIDE。

---

## 設定（`config.py` の主要ノブ）

| キー | 役割 | 既定 |
|------|------|------|
| `CONFIG["group_by_species"]` | **species 単位 CV（CV≒LB）** | `True` ★ |
| `CONFIG["target_transform"]` | 目的変数の変換 | `"log1p"` |
| `CONFIG["clip_predictions"]` | 予測クリップ範囲 | `[0.0, 320.0]` |
| `CONFIG["postprocess_board_smooth"]` | 乾燥曲線の単調平滑化 | `False` |
| `CONFIG_FS["strategies"]` | 比較する選択手法 | 5 種 |
| `CONFIG_FS["amplify"]` | QUBO 設定（token / n_features / `use_count_constraint` 等） | — |

- `run_research.py` の固定グリッドは同ファイル冒頭の **`RESEARCH_MODELS / RESEARCH_PREPS`**。
- `search_best.py` の探索範囲は同ファイル冒頭で編集（config とは独立）。

---

## 出力

```
submissions_manual/   search_best の提出 CSV + search_ranking.csv（全候補+パラメータ）
submissions_<戦略>/   run_research の各選択手法の提出ファイル
figures/eda, figures/strategies/<戦略>, figures/comparison/   研究用の図
logs/run_<日時>.log   run_research の実行ログ
```

提出は基本 `submissions_manual/sub_search_greedy_ensemble.csv`（または
ログの「推奨提出」が示すファイル）。期待 LB はログの species-CV とほぼ一致します。
