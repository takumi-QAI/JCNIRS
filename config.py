"""
JCNIRS 設定 (config.py)
=======================
全パラメータをこのファイルで一元管理します。

- CONFIG       : データパス / 交差検証 / 前処理 / モデル / メタ学習器 / 可視化
- CONFIG_FS    : 特徴量選択 (none / filter / wrapper / embedded / amplify)
- STRATEGY_LABELS : 特徴量選択戦略の日本語ラベル

★ 重要: 交差検証は「ボード単位の GroupKFold」が既定です
--------------------------------------------------------
本データは同一ボード(木材)を乾燥させながら複数回スキャンした「繰り返し測定」を
含みます。ランダム KFold では同じボードが train/val 両方に入りリークし、CV が
極端に楽観的(R²≈0.997)になります。そこで sample number 順の隣接スペクトル相関で
ボードを推定し、GroupKFold で同一ボードを同じ fold に固める honest CV を行います
(CONFIG["cv_grouped"] = True)。

速度を優先したい場合
--------------------
- CONFIG["models"]        : 不要なモデルをコメントアウト
- CONFIG["preprocessors"] : 不要な前処理をコメントアウト
- CONFIG_FS["strategies"] : 不要な特徴量選択をコメントアウト
"""

import os

# 環境変数 (.env の AMPLIFY_TOKEN など) を読み込み
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ============================================================
# モデル定義 (頑健モデル / 深層モデルを分けて管理)
# ============================================================
# 既定 = leaderboard 向けの頑健・高速モデル。
# honest(GroupKFold) CV で汎化するよう正則化を強めにしてある。
ROBUST_MODELS = {
    "PLS":        {"type": "pls",  "params": {"n_components": 12}},
    "PCR":        {"type": "pcr",  "params": {"n_components": 20}},
    "Ridge":      {"type": "ridge", "params": {"alpha": 100.0}},
    "Lasso":      {"type": "lasso", "params": {"alpha": 0.05, "max_iter": 10000}},
    "ElasticNet": {"type": "elasticnet",
                   "params": {"alpha": 0.05, "l1_ratio": 0.5, "max_iter": 10000}},
    "SVR":        {"type": "svr",
                   "params": {"kernel": "rbf", "C": 10.0, "epsilon": 0.05}},
    "kNN":        {"type": "knn",
                   "params": {"n_neighbors": 3, "weights": "distance"}},
    "RandomForest": {"type": "rf",
                     "params": {"n_estimators": 300, "random_state": 42,
                                "n_jobs": -1}},
    "XGBoost":    {"type": "xgb",
                   "params": {"n_estimators": 400, "learning_rate": 0.03,
                              "max_depth": 4, "subsample": 0.8,
                              "colsample_bytree": 0.5, "random_state": 42}},
    "LightGBM":   {"type": "lgbm",
                   "params": {"n_estimators": 400, "learning_rate": 0.03,
                              "num_leaves": 31, "subsample": 0.8,
                              "colsample_bytree": 0.5, "random_state": 42,
                              "verbose": -1}},
}

# 深層モデル (研究レポートのモデル比較用)。
#   ※ 1322 行・実質 150 ボードしか無いため honest CV では過学習しやすく、
#     学習も低速。leaderboard 目的では既定で無効にしている。
#   モデル比較レポートを作るときは CONFIG["models"] を
#     {**ROBUST_MODELS, **DEEP_MODELS} に変更して有効化する。
DEEP_MODELS = {
    "1D-CNN": {"type": "cnn1d",
               "params": {"epochs": 80, "batch_size": 64, "lr": 1e-3,
                          "weight_decay": 1e-4}},
    "AE": {"type": "ae",
           "params": {"epochs": 120, "batch_size": 64, "lr": 1e-3,
                      "weight_decay": 1e-5, "latent_dim": 32,
                      "hidden_dims": [256, 128], "recon_weight": 1.0,
                      "reg_weight": 1.0}},
    "SAE": {"type": "sae",
            "params": {"epochs": 120, "batch_size": 64, "lr": 1e-3,
                       "weight_decay": 1e-5, "latent_dim": 64,
                       "hidden_dims": [256, 128], "recon_weight": 1.0,
                       "reg_weight": 1.0, "sparsity_weight": 1e-3}},
    "VAE": {"type": "vae",
            "params": {"epochs": 120, "batch_size": 64, "lr": 1e-3,
                       "weight_decay": 1e-5, "latent_dim": 32,
                       "hidden_dims": [256, 128], "recon_weight": 1.0,
                       "reg_weight": 1.0, "kl_weight": 1e-3}},
    "GAN": {"type": "gan",
            "params": {"epochs": 150, "batch_size": 64, "lr": 2e-4,
                       "weight_decay": 0.0, "hidden_dims": [256, 128],
                       "adv_weight": 0.1, "reg_weight": 1.0}},
    "DeepSpectra": {"type": "deepspectra",
                    "params": {"epochs": 100, "batch_size": 64, "lr": 1e-3,
                               "weight_decay": 1e-4}},
    "Transformer": {"type": "transformer",
                    "params": {"epochs": 100, "batch_size": 64, "lr": 5e-4,
                               "weight_decay": 1e-4, "patch_size": 32,
                               "d_model": 64, "n_heads": 4, "n_layers": 2,
                               "dropout": 0.1}},
}


# ============================================================
# CONFIG ─ コアパイプラインの設定
# ============================================================
CONFIG = {
    # ── パス ──────────────────────────────────────────────
    "data_dir":       os.path.dirname(os.path.abspath(__file__)),
    "train_file":     "train.csv",
    "test_file":      "test.csv",
    "sample_submit":  "sample_submit.csv",
    "encoding":       "shift_jis",
    "submission_dir": "submissions",        # 提出ファイルの出力フォルダ

    # ── カラム ────────────────────────────────────────────
    "target_col":  "含水率",
    "id_col":      "sample number",
    "drop_cols":   ["sample number", "樹種", "含水率"],  # species number は除外しない
    "onehot_cols": ["species number"],                    # ワンホットにする列

    # ── 交差検証 ──────────────────────────────────────────
    "n_splits":     5,
    "random_state": 42,
    #   ボード単位 GroupKFold (リーク防止)。False で従来のランダム KFold。
    "cv_grouped":            True,
    "group_corr_threshold":  0.9999,  # 隣接スペクトル相関がこれ未満で別ボード

    # ── 目的変数の変換 / 予測の後処理 ─────────────────────
    #   含水率は右に裾を引く分布のため log1p 変換で学習を安定化する
    #   (学習は変換後空間、予測は逆変換 → クリップ → 元スケールで評価)
    "target_transform": "log1p",       # "log1p" or "none"
    #   含水率は非負・実用上 ~300% が上限 → 暴走予測を物理範囲にクリップ
    #   (リーク無しでも不安定モデルが out-of-range を出すのを防ぐ安全網)
    "clip_predictions": [0.0, 320.0],

    # ── 評価指標 ──────────────────────────────────────────
    #   計算・表示する指標 (詳細は metrics.py / docs/REPORT_GUIDE.md)
    "metrics": ["RMSE", "MAE", "R2", "RPD", "RPIQ", "Bias", "MAPE", "MaxError"],

    # ── 前処理 ────────────────────────────────────────────
    #   「+」で連結すると順に適用される
    #   スペクトル変換 : snv, msc, sg_d1, sg_d2
    #   スケーリング   : standard, minmax, powertransformer, rankgauss, genlog
    "preprocessors": [
        "standard",            # StandardScaler のみ
        "snv",                 # Standard Normal Variate
        "msc",                 # Multiplicative Scatter Correction
        "sg_d1",               # Savitzky-Golay 1 次微分
        "sg_d2",               # Savitzky-Golay 2 次微分
        "sg_d1+standard",      # 1 次微分 → StandardScaler
        "sg_d2+standard",      # 2 次微分 → StandardScaler
        "snv+sg_d1",           # SNV → 1 次微分
        "snv+sg_d1+standard",  # SNV → 1 次微分 → StandardScaler (NIR 定番)
    ],

    # Savitzky-Golay パラメータ
    "sg_window_length": 15,    # 窓幅 (奇数)
    "sg_polyorder":     2,     # 多項式の次数

    # ── モデル ────────────────────────────────────────────
    #   既定は頑健モデルのみ。深層モデルも使う(レポート比較)場合は:
    #     "models": {**ROBUST_MODELS, **DEEP_MODELS},
    "models": dict(ROBUST_MODELS),

    # ── メタ学習器 (スタッキング) ─────────────────────────
    "meta_ridge_alpha":    1.0,
    "meta_lasso_alpha":    0.1,
    "meta_lasso_max_iter": 10000,

    # ── 可視化 ────────────────────────────────────────────
    "save_figures": True,
    "figure_dpi":   150,
    "figure_dir":   "figures",   # 図の出力フォルダ (PNG)
    "eda_figures":  True,        # 開始時に EDA 図 (含水率分布・スペクトル概観) を出力
}

# ── 深層モデルの早期終了について ───────────────────────────
#   torch 系モデルは models/base.py の TorchRegressorBase により
#   既定で「入力標準化 + 早期終了」が有効 (early_stopping=True,
#   val_fraction=0.1, es_patience=15, min_epochs=20)。
#   個別に変えたい場合は DEEP_MODELS の params に
#   early_stopping / val_fraction / es_patience / min_epochs を追加する。


# ============================================================
# CONFIG_FS ─ 特徴量選択のパラメータ
# ============================================================
CONFIG_FS = {
    # ── 実行する戦略の一覧 ─────────────────────────────────
    #   不要な戦略はコメントアウトするだけで除外可能
    "strategies": [
        "none",        # ① 特徴量選択なし
        "filter",      # ② フィルター法
        "wrapper",     # ③ ラッパー法
        "embedded",    # ④ 埋め込み法
        "amplify",     # ⑤ Amplify QUBO (量子アニーリング)
    ],

    # ── 共通 ───────────────────────────────────────────────
    "n_features_select": 200,  # 各手法で選択する特徴量数 (統一)

    # ── ② フィルター法 ────────────────────────────────────
    "filter": {
        "n_features": 200,     # 選択する特徴量数
    },

    # ── ③ ラッパー法 (RFE) ────────────────────────────────
    "wrapper": {
        "n_features":   200,   # 選択する特徴量数
        "step":         50,    # 1 反復で除去する特徴量数 (大きいほど高速)
        "n_estimators": 50,    # RFE 内部の LightGBM の木の数
    },

    # ── ④ 埋め込み法 (Lasso) ──────────────────────────────
    "embedded": {
        "n_features": 200,     # 選択する特徴量数
        "alpha":      0.01,    # Lasso の正則化強度
    },

    # ── ⑤ Amplify QUBO ───────────────────────────────────
    #   Amplify AE のトークンとソルバー設定
    #   将来 D-Wave に切り替える場合は client 部分を差し替えるだけ
    "amplify": {
        "token":             os.environ.get("AMPLIFY_TOKEN", ""),
        "n_candidates":      300,   # 事前フィルタ: MI 上位 N 個を QUBO の候補に
        "n_features":        200,   # QUBO で最終的に選択する特徴量数
        "lambda_redundancy": 0.5,   # 冗長性ペナルティの重み (大きい=多様性重視)
        "time_limit_ms":     5000,  # ソルバー実行時間 [ms]
        "corr_threshold":    0.1,   # QUBO 項のスパース化閾値
    },
}


# ============================================================
# 戦略ごとの日本語ラベル
# ============================================================
STRATEGY_LABELS = {
    "none":     "① 選択なし (全波長)",
    "filter":   "② フィルター法 (MI)",
    "wrapper":  "③ ラッパー法 (RFE)",
    "embedded": "④ 埋め込み法 (Lasso)",
    "amplify":  "⑤ Amplify QUBO",
}
