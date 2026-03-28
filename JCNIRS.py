"""
JCNIRS ─ NIR 分光データによる木材含水率予測パイプライン
======================================================
近赤外 (NIR) 分光データから木材の含水率を予測する
機械学習パイプラインです。

特徴
----
- species number をワンホットエンコーディングで特徴量化
- スペクトル特化の前処理 (SNV / MSC / Savitzky-Golay 微分)
- 1D-CNN を含む多様なモデル (10 種)
- 全 モデル×前処理 + アンサンブル 3 手法の提出ファイルを自動生成
- CONFIG 辞書で全パラメータを一元管理

処理フロー
----------
Step 1 : データ読み込み (ワンホットエンコーディング含む)
Step 2 : 前処理の確認
Step 3 : 全組み合わせの交差検証 + テスト予測
Step 4 : 結果の可視化
Step 5 : スタッキング (Ridge / Lasso メタ学習器)
Step 6 : 加重平均アンサンブル
Step 7 : 全提出ファイルの作成
"""

# ============================================================
# インポート
# ============================================================
import os
import sys
import copy
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # GUI 不要 (画像はファイル保存のみ)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, PowerTransformer, QuantileTransformer,
)
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import xgboost as xgb
import lightgbm as lgb
from scipy.optimize import minimize
from scipy.signal import savgol_filter

# --- PyTorch (1D-CNN 用、未インストール時はスキップ) ---
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

warnings.filterwarnings("ignore")


# ============================================================
# 設定 (CONFIG) ─ 調整したいパラメータはここを変更
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
    ],

    # Savitzky-Golay パラメータ
    "sg_window_length": 15,    # 窓幅 (奇数)
    "sg_polyorder":     2,     # 多項式の次数

    # ── モデル ────────────────────────────────────────────
    "models": {
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
        "1D-CNN": {
            "type":   "cnn1d",
            "params": {
                "epochs":       80,      # エポック数
                "batch_size":   64,      # バッチサイズ
                "lr":           1e-3,    # 学習率
                "weight_decay": 1e-4,    # L2 正則化
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
}


# ============================================================
# ユーティリティ
# ============================================================
def _header(title: str):
    """ステップ見出しを表示する。"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ============================================================
# 前処理クラス (スペクトル変換 + スケーリング)
# ============================================================
class SpectralPreprocessor:
    """NIR スペクトルデータ向けの前処理パイプライン。

    method を「+」で連結すると順に適用される。
    例: ``"snv+sg_d1+standard"`` → SNV → SG 1 次微分 → StandardScaler

    対応メソッド
    ------------
    スペクトル変換 (行方向):
        snv             Standard Normal Variate
        msc             Multiplicative Scatter Correction
        sg_d1           Savitzky-Golay 1 次微分
        sg_d2           Savitzky-Golay 2 次微分
    スケーリング (列方向):
        standard        StandardScaler
        minmax          MinMaxScaler
        powertransformer  Yeo-Johnson
        rankgauss       QuantileTransformer (正規分布)
        genlog          log1p 変換
    """

    _SCALER_FACTORY = {
        "standard":         lambda: StandardScaler(),
        "minmax":           lambda: MinMaxScaler(),
        "powertransformer": lambda: PowerTransformer(method="yeo-johnson"),
        "rankgauss":        lambda: QuantileTransformer(output_distribution="normal"),
    }

    def __init__(self, method: str = "standard", config: dict | None = None):
        self.method = method
        self.steps = method.split("+")
        self.config = config or {}
        self._fitted: list = []

    # ---- public API ----
    def fit(self, X, y=None):
        self._fitted = []
        X_cur = X.copy()
        for step in self.steps:
            fitted_obj, X_cur = self._fit_one(step, X_cur)
            self._fitted.append((step, fitted_obj))
        return self

    def transform(self, X):
        X_cur = X.copy()
        for step, fitted_obj in self._fitted:
            X_cur = self._transform_one(step, X_cur, fitted_obj)
        return X_cur

    # ---- internal ----
    def _fit_one(self, step, X):
        if step in ("snv", "genlog", "sg_d1", "sg_d2"):
            return None, self._transform_one(step, X, None)
        if step == "msc":
            ref = np.mean(X, axis=0)
            return ref, self._transform_one(step, X, ref)
        if step in self._SCALER_FACTORY:
            scaler = self._SCALER_FACTORY[step]()
            scaler.fit(X)
            return scaler, scaler.transform(X)
        raise ValueError(f"Unknown preprocessing step: '{step}'")

    def _transform_one(self, step, X, fitted_obj):
        if step == "snv":
            return self._snv(X)
        if step == "msc":
            return self._msc(X, fitted_obj)
        if step == "sg_d1":
            return self._sg_deriv(X, deriv=1)
        if step == "sg_d2":
            return self._sg_deriv(X, deriv=2)
        if step == "genlog":
            return np.log1p(np.maximum(X, 0))
        return fitted_obj.transform(X)

    # ---- スペクトル変換の実装 ----
    @staticmethod
    def _snv(X):
        """Standard Normal Variate: 各スペクトルを自身の平均 / 標準偏差で正規化。"""
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True) + 1e-10
        return (X - mean) / std

    @staticmethod
    def _msc(X, reference):
        """Multiplicative Scatter Correction。"""
        out = np.zeros_like(X)
        for i in range(X.shape[0]):
            coef = np.polyfit(reference, X[i], 1)
            out[i] = (X[i] - coef[1]) / (coef[0] + 1e-10)
        return out

    def _sg_deriv(self, X, deriv=1):
        """Savitzky-Golay 微分フィルタ。"""
        wl = self.config.get("sg_window_length", 15)
        po = self.config.get("sg_polyorder", 2)
        return savgol_filter(X, window_length=wl, polyorder=po,
                             deriv=deriv, axis=1)


# ============================================================
# 1D-CNN モデル (PyTorch)
# ============================================================
if TORCH_AVAILABLE:

    class _CNN1DNet(nn.Module):
        """3 層 1D 畳み込み + 全結合層。"""

        def __init__(self, n_features: int):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=11, padding=5),
                nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),

                nn.Conv1d(32, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),

                nn.Conv1d(64, 128, kernel_size=5, padding=2),
                nn.BatchNorm1d(128), nn.ReLU(),
                nn.AdaptiveAvgPool1d(8),
            )
            self.fc = nn.Sequential(
                nn.Linear(128 * 8, 256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 64),      nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            x = x.unsqueeze(1)            # (batch, 1, n_features)
            x = self.conv(x)
            x = x.view(x.size(0), -1)     # flatten
            return self.fc(x)

    class CNN1DRegressor:
        """sklearn 互換の 1D-CNN 回帰モデル。

        Parameters
        ----------
        epochs : int          学習エポック数
        batch_size : int      ミニバッチサイズ
        lr : float            学習率
        weight_decay : float  L2 正則化
        random_state : int    乱数シード
        verbose : bool        学習経過を表示するか
        """

        def __init__(self, epochs=80, batch_size=64, lr=1e-3,
                     weight_decay=1e-4, random_state=42, verbose=False):
            self.epochs = epochs
            self.batch_size = batch_size
            self.lr = lr
            self.weight_decay = weight_decay
            self.random_state = random_state
            self.verbose = verbose
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self.net_ = None
            self.y_mean_ = 0.0
            self.y_std_ = 1.0

        def fit(self, X, y):
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

            # ターゲットを正規化して学習安定化
            self.y_mean_ = float(y.mean())
            self.y_std_ = float(y.std()) + 1e-8
            y_norm = (y - self.y_mean_) / self.y_std_

            self.net_ = _CNN1DNet(X.shape[1]).to(self.device)
            optimizer = torch.optim.Adam(
                self.net_.parameters(),
                lr=self.lr, weight_decay=self.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=10, factor=0.5, min_lr=1e-6,
            )
            criterion = nn.MSELoss()

            X_t = torch.FloatTensor(X).to(self.device)
            y_t = torch.FloatTensor(y_norm).unsqueeze(1).to(self.device)
            loader = DataLoader(
                TensorDataset(X_t, y_t),
                batch_size=self.batch_size, shuffle=True,
            )

            self.net_.train()
            for epoch in range(self.epochs):
                total_loss = 0.0
                for xb, yb in loader:
                    optimizer.zero_grad()
                    loss = criterion(self.net_(xb), yb)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * xb.size(0)
                scheduler.step(total_loss / len(X_t))
                if self.verbose and (epoch + 1) % 20 == 0:
                    print(f"      CNN Epoch {epoch+1}/{self.epochs}  "
                          f"Loss={total_loss / len(X_t):.6f}")
            return self

        def predict(self, X):
            self.net_.eval()
            X_t = torch.FloatTensor(X).to(self.device)
            preds = []
            with torch.no_grad():
                for i in range(0, len(X_t), self.batch_size):
                    preds.append(self.net_(X_t[i:i + self.batch_size]))
            out = torch.cat(preds).cpu().numpy().flatten()
            return out * self.y_std_ + self.y_mean_


# ============================================================
# モデル構築ヘルパー
# ============================================================
def _build_model(model_type: str, params: dict, config: dict):
    """モデルタイプに応じてインスタンスを生成する。"""
    _BUILDERS = {
        "pcr":        lambda p: Pipeline([
                          ("pca", PCA(n_components=p.get("n_components", 10))),
                          ("reg", LinearRegression()),
                      ]),
        "pls":        lambda p: PLSRegression(**p),
        "ridge":      lambda p: Ridge(**p),
        "lasso":      lambda p: Lasso(**p),
        "elasticnet": lambda p: ElasticNet(**p),
        "svr":        lambda p: SVR(**p),
        "rf":         lambda p: RandomForestRegressor(**p),
        "xgb":        lambda p: xgb.XGBRegressor(**p),
        "lgbm":       lambda p: lgb.LGBMRegressor(**p),
    }
    if model_type == "cnn1d":
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch が必要です: pip install torch")
        return CNN1DRegressor(**params, random_state=config["random_state"])
    if model_type not in _BUILDERS:
        raise ValueError(f"Unknown model type: {model_type}")
    return _BUILDERS[model_type](params)


def build_all_models(config: dict) -> dict:
    """CONFIG["models"] から全モデル辞書を生成する。"""
    models = {}
    for name, spec in config["models"].items():
        if spec["type"] == "cnn1d" and not TORCH_AVAILABLE:
            print(f"  ⚠ {name} をスキップ (PyTorch 未インストール)")
            continue
        models[name] = _build_model(spec["type"], spec["params"], config)
    return models


# ============================================================
# Step 1: データ読み込み (ワンホットエンコーディング含む)
# ============================================================
def load_data(config: dict):
    """CSV を読み込み、スペクトル特徴量 / カテゴリ特徴量 / ターゲットに分割する。

    Returns
    -------
    df_train, df_test : DataFrame
    X_train_spec, X_test_spec : ndarray  スペクトル特徴量
    X_train_cat,  X_test_cat  : ndarray  ワンホットエンコード済みカテゴリ特徴量
    y_train : ndarray
    """
    _header("Step 1: データ読み込み")
    d = config["data_dir"]
    enc = config["encoding"]

    df_train = pd.read_csv(os.path.join(d, config["train_file"]),  encoding=enc)
    df_test  = pd.read_csv(os.path.join(d, config["test_file"]),   encoding=enc)

    # ---- スペクトル特徴量 ----
    exclude = set(config["drop_cols"] + config["onehot_cols"])
    spec_cols = [c for c in df_train.columns if c not in exclude]
    X_train_spec = df_train[spec_cols].values.astype(np.float64)
    X_test_spec  = df_test[spec_cols].values.astype(np.float64)

    # ---- ワンホットエンコーディング (train + test を統合して生成) ----
    combined = pd.concat([
        df_train[config["onehot_cols"]],
        df_test[config["onehot_cols"]],
    ], ignore_index=True)
    onehot = pd.get_dummies(combined, columns=config["onehot_cols"])
    X_train_cat = onehot.iloc[:len(df_train)].values.astype(np.float64)
    X_test_cat  = onehot.iloc[len(df_train):].values.astype(np.float64)

    # ---- ターゲット ----
    y_train = df_train[config["target_col"]].values.astype(np.float64)

    # ---- 表示 ----
    print(f"  Train           : {df_train.shape[0]} samples")
    print(f"  Test            : {df_test.shape[0]} samples")
    print(f"  スペクトル特徴量 : {X_train_spec.shape[1]}")
    print(f"  カテゴリ特徴量   : {X_train_cat.shape[1]}  "
          f"(one-hot: {config['onehot_cols']})")
    print(f"  合計特徴量       : {X_train_spec.shape[1] + X_train_cat.shape[1]}")

    return (df_train, df_test,
            X_train_spec, X_test_spec,
            X_train_cat, X_test_cat,
            y_train)


# ============================================================
# Step 2: 前処理の確認
# ============================================================
def verify_preprocessors(X_train_spec: np.ndarray, config: dict):
    """各前処理メソッドの変換結果サマリを表示する。"""
    _header("Step 2: 前処理の確認")
    for m in config["preprocessors"]:
        prep = SpectralPreprocessor(m, config)
        X_t = prep.fit(X_train_spec).transform(X_train_spec)
        print(f"  {m:22s} | Shape: {str(X_t.shape):16s} | "
              f"Mean: {np.mean(X_t):+10.4f} | Std: {np.std(X_t):.4f}")


# ============================================================
# Step 3: 全組み合わせの CV + テスト予測
# ============================================================
def run_full_evaluation(X_train_spec, X_test_spec,
                        X_train_cat, X_test_cat,
                        y_train, models, config):
    """全 前処理×モデル の 5-Fold CV を実行し、OOF 予測とテスト予測を収集する。

    Returns
    -------
    df_results     : DataFrame  (Preprocessor, Model, CV-RMSE, combo_key)
    all_oof_train  : dict  {combo_key: ndarray(n_train,)}
    all_test_preds : dict  {combo_key: ndarray(n_test,)}
    """
    _header("Step 3: 交差検証 (全組み合わせ)")

    kf = KFold(
        n_splits=config["n_splits"], shuffle=True,
        random_state=config["random_state"],
    )
    preprocessors = config["preprocessors"]
    n_train = X_train_spec.shape[0]
    n_test  = X_test_spec.shape[0]

    results        = []
    all_oof_train  = {}
    all_test_preds = {}

    total = len(preprocessors) * len(models)
    count = 0

    for prep_name in preprocessors:
        for model_name, model_template in models.items():
            count += 1
            combo_key = f"{model_name}__{prep_name}"
            print(f"  [{count:3d}/{total}] {model_name:12s} × {prep_name:22s} ",
                  end="", flush=True)

            oof = np.zeros(n_train)
            test_folds = np.zeros((n_test, config["n_splits"]))

            for fi, (tr_idx, val_idx) in enumerate(kf.split(X_train_spec)):
                # 前処理 (スペクトルのみに適用)
                prep = SpectralPreprocessor(prep_name, config)
                Xtr_s = prep.fit(X_train_spec[tr_idx]).transform(
                    X_train_spec[tr_idx])
                Xva_s = prep.transform(X_train_spec[val_idx])
                Xte_s = prep.transform(X_test_spec)

                # カテゴリ特徴量と結合
                Xtr = np.hstack([Xtr_s, X_train_cat[tr_idx]])
                Xva = np.hstack([Xva_s, X_train_cat[val_idx]])
                Xte = np.hstack([Xte_s, X_test_cat])

                # 学習 & 予測
                m = copy.deepcopy(model_template)
                m.fit(Xtr, y_train[tr_idx])
                oof[val_idx] = m.predict(Xva).flatten()
                test_folds[:, fi] = m.predict(Xte).flatten()

            cv_rmse = np.sqrt(mean_squared_error(y_train, oof))
            all_oof_train[combo_key]  = oof
            all_test_preds[combo_key] = test_folds.mean(axis=1)

            results.append({
                "Preprocessor": prep_name,
                "Model":        model_name,
                "CV-RMSE":      cv_rmse,
                "combo_key":    combo_key,
            })
            print(f"CV-RMSE = {cv_rmse:.4f}")

    df_results = pd.DataFrame(results)

    print("\n  --- CV 結果サマリ (上位 15) ---")
    print(df_results.sort_values("CV-RMSE").head(15)
          .to_string(index=False))

    return df_results, all_oof_train, all_test_preds


# ============================================================
# Step 4: 結果の可視化
# ============================================================
def visualize_results(df_results: pd.DataFrame, config: dict):
    """ヒートマップ + 棒グラフで CV-RMSE を可視化する。"""
    _header("Step 4: 結果の可視化")
    save = config["save_figures"]
    dpi  = config["figure_dpi"]
    out  = config["data_dir"]

    # --- ヒートマップ ---
    pivot = df_results.pivot(
        index="Model", columns="Preprocessor", values="CV-RMSE")
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd_r",
                linewidths=0.5, ax=ax)
    ax.set_title("CV-RMSE : Model × Preprocessor")
    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(out, "heatmap_cv_rmse.png"), dpi=dpi)
        print(f"    保存: heatmap_cv_rmse.png")
    plt.close(fig)

    # --- 各モデルの最良前処理 (棒グラフ) ---
    best_per_model = (
        df_results
        .loc[df_results.groupby("Model")["CV-RMSE"].idxmin()]
        .sort_values("CV-RMSE")
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    labels = (best_per_model["Model"] + "\n("
              + best_per_model["Preprocessor"] + ")")
    bars = ax.barh(
        labels, best_per_model["CV-RMSE"],
        color=sns.color_palette("viridis", len(best_per_model)),
    )
    ax.set_xlabel("CV-RMSE")
    ax.set_title("Best CV-RMSE per Model")
    for bar, val in zip(bars, best_per_model["CV-RMSE"]):
        ax.text(bar.get_width() + 0.1,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)
    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(out, "barplot_best_cv_rmse.png"), dpi=dpi)
        print(f"    保存: barplot_best_cv_rmse.png")
    plt.close(fig)

    # --- 全体ベスト ---
    overall_best = df_results.loc[df_results["CV-RMSE"].idxmin()]
    print("\n  各モデルの最良前処理:")
    for _, row in best_per_model.iterrows():
        print(f"    {row['Model']:12s}  {row['Preprocessor']:22s}  "
              f"CV-RMSE = {row['CV-RMSE']:.4f}")
    print(f"\n  >>> 全体ベスト: {overall_best['Model']} × "
          f"{overall_best['Preprocessor']}  "
          f"CV-RMSE = {overall_best['CV-RMSE']:.4f}")

    return best_per_model, overall_best


# ============================================================
# Step 5: スタッキング (Ridge / Lasso メタ学習器)
# ============================================================
def run_stacking(df_results, all_oof_train, all_test_preds,
                 y_train, config):
    """各モデルの最良前処理を使ってスタッキングを行う。

    Returns
    -------
    meta_ridge, meta_lasso : 学習済みメタ学習器
    test_matrix : ndarray  ベースモデルのテスト予測行列
    ridge_cv, lasso_cv : float  CV-RMSE
    base_keys : list[str]  ベースモデルの combo_key リスト
    """
    _header("Step 5: スタッキング (Ridge / Lasso)")

    kf = KFold(
        n_splits=config["n_splits"], shuffle=True,
        random_state=config["random_state"],
    )

    # --- ベースモデル選定 (各モデルの最良前処理) ---
    best_per_model = (
        df_results
        .loc[df_results.groupby("Model")["CV-RMSE"].idxmin()]
        .sort_values("CV-RMSE")
    )
    base_keys  = best_per_model["combo_key"].tolist()
    base_names = best_per_model["Model"].tolist()

    oof_matrix  = np.column_stack([all_oof_train[k]  for k in base_keys])
    test_matrix = np.column_stack([all_test_preds[k] for k in base_keys])
    n_train = oof_matrix.shape[0]

    print(f"\n  ベースモデル数: {len(base_keys)}")
    for name, key in zip(base_names, base_keys):
        print(f"    - {key}")

    # --- Ridge メタ学習器 ---
    meta_ridge = Ridge(alpha=config["meta_ridge_alpha"])
    meta_ridge.fit(oof_matrix, y_train)

    ridge_oof = np.zeros(n_train)
    for tr_idx, val_idx in kf.split(oof_matrix):
        m = Ridge(alpha=config["meta_ridge_alpha"])
        m.fit(oof_matrix[tr_idx], y_train[tr_idx])
        ridge_oof[val_idx] = m.predict(oof_matrix[val_idx])
    ridge_cv = np.sqrt(mean_squared_error(y_train, ridge_oof))

    print(f"\n  [Stacking Ridge]  CV-RMSE = {ridge_cv:.4f}")
    print(f"    weights: "
          f"{dict(zip(base_names, np.round(meta_ridge.coef_, 4)))}")

    # --- Lasso メタ学習器 ---
    meta_lasso = Lasso(
        alpha=config["meta_lasso_alpha"],
        max_iter=config["meta_lasso_max_iter"],
    )
    meta_lasso.fit(oof_matrix, y_train)

    lasso_oof = np.zeros(n_train)
    for tr_idx, val_idx in kf.split(oof_matrix):
        m = Lasso(
            alpha=config["meta_lasso_alpha"],
            max_iter=config["meta_lasso_max_iter"],
        )
        m.fit(oof_matrix[tr_idx], y_train[tr_idx])
        lasso_oof[val_idx] = m.predict(oof_matrix[val_idx])
    lasso_cv = np.sqrt(mean_squared_error(y_train, lasso_oof))

    print(f"\n  [Stacking Lasso]  CV-RMSE = {lasso_cv:.4f}")
    print(f"    weights: "
          f"{dict(zip(base_names, np.round(meta_lasso.coef_, 4)))}")

    return meta_ridge, meta_lasso, test_matrix, ridge_cv, lasso_cv, base_keys


# ============================================================
# Step 6: 加重平均アンサンブル
# ============================================================
def run_weighted_average(df_results, all_oof_train, all_test_preds,
                         y_train, config):
    """最適重みによる加重平均アンサンブル。

    Returns
    -------
    opt_w : ndarray        最適重み
    wa_cv : float          CV-RMSE
    wa_test_pred : ndarray テスト予測
    """
    _header("Step 6: 加重平均アンサンブル")

    kf = KFold(
        n_splits=config["n_splits"], shuffle=True,
        random_state=config["random_state"],
    )

    best_per_model = (
        df_results
        .loc[df_results.groupby("Model")["CV-RMSE"].idxmin()]
        .sort_values("CV-RMSE")
    )
    base_keys  = best_per_model["combo_key"].tolist()
    base_names = best_per_model["Model"].tolist()

    oof_matrix  = np.column_stack([all_oof_train[k]  for k in base_keys])
    test_matrix = np.column_stack([all_test_preds[k] for k in base_keys])
    n_base  = len(base_keys)
    n_train = oof_matrix.shape[0]

    def _wa_rmse(w, oof, y):
        return np.sqrt(mean_squared_error(y, oof @ w))

    init_w = np.ones(n_base) / n_base
    bounds = [(0, 1)] * n_base
    cons   = {"type": "eq", "fun": lambda w: w.sum() - 1.0}

    res = minimize(
        _wa_rmse, init_w, args=(oof_matrix, y_train),
        method="SLSQP", bounds=bounds, constraints=cons,
    )
    opt_w = res.x

    # CV 評価
    wa_oof = np.zeros(n_train)
    for tr_idx, val_idx in kf.split(oof_matrix):
        cv_res = minimize(
            _wa_rmse, init_w,
            args=(oof_matrix[tr_idx], y_train[tr_idx]),
            method="SLSQP", bounds=bounds, constraints=cons,
        )
        wa_oof[val_idx] = oof_matrix[val_idx] @ cv_res.x
    wa_cv = np.sqrt(mean_squared_error(y_train, wa_oof))

    print("\n  最適重み:")
    for name, w in zip(base_names, opt_w):
        print(f"    {name:12s}: {w:.4f}")
    print(f"\n  [Weighted Avg]  CV-RMSE = {wa_cv:.4f}")

    wa_test_pred = test_matrix @ opt_w
    return opt_w, wa_cv, wa_test_pred


# ============================================================
# Step 7: 全提出ファイルの作成
# ============================================================
def create_all_submissions(
    df_test, df_results, all_test_preds,
    meta_ridge, meta_lasso, test_matrix,
    wa_test_pred, overall_best,
    ridge_cv, lasso_cv, wa_cv, config,
):
    """全モデル×前処理の個別予測 + アンサンブル予測を CSV に出力する。"""
    _header("Step 7: 全提出ファイルの作成")

    sub_dir = os.path.join(config["data_dir"], config["submission_dir"])
    os.makedirs(sub_dir, exist_ok=True)
    test_ids = df_test[config["id_col"]].values

    def _save(fname, pred):
        path = os.path.join(sub_dir, fname)
        pd.DataFrame({0: test_ids, 1: pred}).to_csv(
            path, index=False, header=False)

    # ---- 個別モデル × 前処理 ----
    print("\n  --- 個別モデル × 前処理 ---")
    for _, row in df_results.iterrows():
        key  = row["combo_key"]
        safe = (key.replace(" ", "_").replace("+", "_")
                .replace("/", "_").replace(":", "_"))
        _save(f"sub_{safe}.csv", all_test_preds[key])
    print(f"    {len(all_test_preds)} ファイル作成")

    # ---- アンサンブル ----
    print("\n  --- アンサンブル ---")
    ridge_pred = meta_ridge.predict(test_matrix)
    lasso_pred = meta_lasso.predict(test_matrix)

    _save("sub_ensemble_stacking_ridge.csv", ridge_pred)
    _save("sub_ensemble_stacking_lasso.csv", lasso_pred)
    _save("sub_ensemble_weighted_avg.csv",   wa_test_pred)

    # ベスト手法を判定
    ens_summary = pd.DataFrame({
        "Method": [
            f"Single Best ({overall_best['Model']} × "
            f"{overall_best['Preprocessor']})",
            "Stacking Ridge",
            "Stacking Lasso",
            "Weighted Average",
        ],
        "CV-RMSE": [
            overall_best["CV-RMSE"], ridge_cv, lasso_cv, wa_cv,
        ],
    }).sort_values("CV-RMSE")

    best_method = ens_summary.iloc[0]["Method"]
    if "Stacking Ridge" in best_method:
        best_pred = ridge_pred
    elif "Stacking Lasso" in best_method:
        best_pred = lasso_pred
    elif "Weighted Average" in best_method:
        best_pred = wa_test_pred
    else:
        best_pred = all_test_preds[overall_best["combo_key"]]

    _save("sub_BEST.csv", best_pred)
    print("    4 ファイル作成 (Ridge / Lasso / WA / BEST)")

    # ---- サマリ ----
    total = len(all_test_preds) + 4
    print(f"\n  合計 {total} 個の提出ファイルを "
          f"{config['submission_dir']}/ に保存しました")

    print("\n  --- アンサンブル比較 ---")
    print(ens_summary.to_string(index=False))
    print(f"\n  >>> 推奨: {best_method}  "
          f"(CV-RMSE = {ens_summary.iloc[0]['CV-RMSE']:.4f})")

    print("\n  BEST submission プレビュー (先頭 5 行):")
    print(pd.DataFrame({"id": test_ids, "pred": best_pred})
          .head().to_string(index=False))
    print(f"\n  テスト予測統計:")
    print(f"    Mean = {best_pred.mean():.2f}  "
          f"Std = {best_pred.std():.2f}  "
          f"Min = {best_pred.min():.2f}  "
          f"Max = {best_pred.max():.2f}")


# ============================================================
# メイン実行
# ============================================================
def main():
    """パイプライン全体を順に実行する。"""

    # 1. データ読み込み
    (df_train, df_test,
     X_train_spec, X_test_spec,
     X_train_cat, X_test_cat,
     y_train) = load_data(CONFIG)

    # 2. 前処理の確認
    verify_preprocessors(X_train_spec, CONFIG)

    # 3. 全組み合わせ CV + テスト予測
    models = build_all_models(CONFIG)
    df_results, all_oof_train, all_test_preds = run_full_evaluation(
        X_train_spec, X_test_spec,
        X_train_cat, X_test_cat,
        y_train, models, CONFIG,
    )

    # 4. 可視化
    best_per_model, overall_best = visualize_results(df_results, CONFIG)

    # 5. スタッキング
    (meta_ridge, meta_lasso,
     test_matrix, ridge_cv, lasso_cv, _) = run_stacking(
        df_results, all_oof_train, all_test_preds, y_train, CONFIG,
    )

    # 6. 加重平均
    opt_w, wa_cv, wa_test_pred = run_weighted_average(
        df_results, all_oof_train, all_test_preds, y_train, CONFIG,
    )

    # 7. 全提出ファイル作成
    create_all_submissions(
        df_test, df_results, all_test_preds,
        meta_ridge, meta_lasso, test_matrix,
        wa_test_pred, overall_best,
        ridge_cv, lasso_cv, wa_cv, CONFIG,
    )

    _header("完了！")
    print(f"  提出ファイル: {CONFIG['submission_dir']}/")


if __name__ == "__main__":
    main()


