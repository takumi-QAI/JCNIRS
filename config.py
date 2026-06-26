"""
JCNIRS 設定 (config.py)
=======================
全パラメータをこのファイルで一元管理します。

- CONFIG       : データパス / 交差検証 / 前処理 / モデル / メタ学習器 / 可視化
- CONFIG_FS    : 特徴量選択 (none / filter / wrapper / embedded / amplify)
- STRATEGY_LABELS : 特徴量選択戦略の日本語ラベル

速度を優先したい場合
--------------------
- CONFIG["models"]        : 不要なモデルをコメントアウト
- CONFIG["preprocessors"] : 不要な前処理をコメントアウト
- CONFIG_FS["strategies"] : 不要な特徴量選択をコメントアウト
既定では「全16モデル × 全8前処理 × 全5特徴量選択 × 5-fold CV」を実行するため、
CPU 環境では数時間規模になります (深層モデルは GPU 推奨)。
"""

import os

# 環境変数 (.env の AMPLIFY_TOKEN など) を読み込み
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


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

    # ── 目的変数の変換 / 予測の後処理 ─────────────────────
    #   含水率は右に裾を引く分布のため log1p 変換で学習を安定化する
    #   (学習は変換後空間、予測は逆変換 → クリップ → 元スケールで評価)
    "target_transform": "log1p",       # "log1p" or "none"
    "clip_predictions": [0.0, None],   # 含水率は非負 → 下限 0 にクリップ

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
    #   type は models/ 以下の各モジュールの TYPE と対応
    "models": {
        # ---- 古典的な機械学習モデル ----
        "PLS": {
            "type":   "pls",
            "params": {"n_components": 20},
        },
        "PCR": {
            "type":   "pcr",
            "params": {"n_components": 10},
        },
        "Ridge": {
            "type":   "ridge",
            "params": {"alpha": 1.0},
        },
        "Lasso": {
            "type":   "lasso",
            "params": {"alpha": 0.1, "max_iter": 10000},
        },
        "ElasticNet": {
            "type":   "elasticnet",
            "params": {"alpha": 0.1, "l1_ratio": 0.5, "max_iter": 10000},
        },
        "SVR": {
            "type":   "svr",
            "params": {"kernel": "rbf", "C": 1.0, "epsilon": 0.1},
        },
        "RandomForest": {
            "type":   "rf",
            "params": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
        },
        "XGBoost": {
            "type":   "xgb",
            "params": {
                "n_estimators": 100, "learning_rate": 0.1,
                "max_depth": 5, "random_state": 42,
            },
        },
        "LightGBM": {
            "type":   "lgbm",
            "params": {
                "n_estimators": 100, "learning_rate": 0.1,
                "random_state": 42, "verbose": -1,
            },
        },

        # ---- 深層学習モデル (要 PyTorch) ----
        "1D-CNN": {
            "type":   "cnn1d",
            "params": {
                "epochs":       80,
                "batch_size":   64,
                "lr":           1e-3,
                "weight_decay": 1e-4,
            },
        },
        "AE": {
            "type":   "ae",
            "params": {
                "epochs":        120,
                "batch_size":    64,
                "lr":            1e-3,
                "weight_decay":  1e-5,
                "latent_dim":    32,
                "hidden_dims":   [256, 128],
                "recon_weight":  1.0,    # 再構成損失の重み
                "reg_weight":    1.0,    # 回帰損失の重み
            },
        },
        "SAE": {
            "type":   "sae",
            "params": {
                "epochs":          120,
                "batch_size":      64,
                "lr":              1e-3,
                "weight_decay":    1e-5,
                "latent_dim":      64,
                "hidden_dims":     [256, 128],
                "recon_weight":    1.0,
                "reg_weight":      1.0,
                "sparsity_weight": 1e-3,  # latent への L1 スパース penalty
            },
        },
        "VAE": {
            "type":   "vae",
            "params": {
                "epochs":       120,
                "batch_size":   64,
                "lr":           1e-3,
                "weight_decay": 1e-5,
                "latent_dim":   32,
                "hidden_dims":  [256, 128],
                "recon_weight": 1.0,
                "reg_weight":   1.0,
                "kl_weight":    1e-3,    # KL ダイバージェンスの重み (β)
            },
        },
        "GAN": {
            "type":   "gan",
            "params": {
                "epochs":       150,
                "batch_size":   64,
                "lr":           2e-4,
                "weight_decay": 0.0,
                "hidden_dims":  [256, 128],
                "adv_weight":   0.1,     # 敵対的損失の重み
                "reg_weight":   1.0,     # 教師あり回帰損失の重み
            },
        },
        "DeepSpectra": {
            "type":   "deepspectra",
            "params": {
                "epochs":       100,
                "batch_size":   64,
                "lr":           1e-3,
                "weight_decay": 1e-4,
            },
        },
        "Transformer": {
            "type":   "transformer",
            "params": {
                "epochs":       100,
                "batch_size":   64,
                "lr":           5e-4,
                "weight_decay": 1e-4,
                "patch_size":   32,      # スペクトルを分割するパッチ長
                "d_model":      64,
                "n_heads":      4,
                "n_layers":     2,
                "dropout":      0.1,
            },
        },
    },

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
#   個別に変えたい場合は上記 CONFIG["models"][name]["params"] に
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
