"""
JCNIRS Feature Selection Comparison
====================================
5 つの特徴量選択手法を比較し、各手法ごとに全モデル×前処理の
提出ファイルを生成するパイプラインです。

手法
----
① 特徴量選択なし (全波長 = ベースライン)
② フィルター法   (相互情報量でランキング → 上位 k 個を選択)
③ ラッパー法     (RFE: Recursive Feature Elimination)
④ 埋め込み法     (Lasso の係数絶対値でランキング → 上位 k 個を選択)
⑤ Amplify QUBO   (mRMR を QUBO 定式化 → Amplify AE で求解)
                   ※ D-Wave 量子アニーリングへの移行を見据えた L0 特徴量選択

出力
----
submissions_none/       ① のファイル群 (84 ファイル)
submissions_filter/     ② のファイル群
submissions_wrapper/    ③ のファイル群
submissions_embedded/   ④ のファイル群
submissions_amplify/    ⑤ のファイル群
feature_selection_comparison.png   精度比較グラフ

使い方
------
    python JCNIRS_feature_selection.py
"""

# ============================================================
# インポート
# ============================================================
import os
import sys
import copy
import time
import warnings
from dotenv import load_dotenv

load_dotenv()  # .env ファイルから環境変数を読み込み

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso as LassoFS

# コアパイプラインを JCNIRS.py からインポート
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from JCNIRS import (                           # noqa: E402
    CONFIG as BASE_CONFIG,
    build_all_models,
    load_data,
    run_full_evaluation,
    run_stacking,
    run_weighted_average,
    create_all_submissions,
    _header,
)

# Amplify SDK (未インストール時はスキップ)
try:
    from amplify import (
        VariableGenerator, Model,
        AmplifyAEClient, solve, equal_to,
    )
    AMPLIFY_AVAILABLE = True
except ImportError:
    AMPLIFY_AVAILABLE = False

warnings.filterwarnings("ignore")


# ============================================================
# 設定 (CONFIG_FS) ─ 特徴量選択のパラメータ
# ============================================================
CONFIG_FS = {
    # ── 実行する戦略の一覧 ─────────────────────────────────
    #   不要な戦略はコメントアウトするだけで除外可能
    "strategies": [
        "none",        # ① 特徴量選択なし
        "filter",      # ② フィルター法
        "wrapper",     # ③ ラッパー法
        "embedded",    # ④ 埋め込み法
        "amplify",     # ⑤ Amplify QUBO
    ],

    # ── 共通 ───────────────────────────────────────────────
    "n_features_select": 200,  # 各手法で選択する特徴量数 (統一)

    # ── ② フィルター法 ────────────────────────────────────
    "filter": {
        "n_features": 200,     # 選択する特徴量数
    },

    # ── ③ ラッパー法 (RFE) ────────────────────────────────
    "wrapper": {
        "n_features": 200,     # 選択する特徴量数
        "step":       50,      # 1 反復で除去する特徴量数 (大きいほど高速)
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
        "token":            os.environ.get("AMPLIFY_TOKEN", ""),
        "n_candidates":     300,    # 事前フィルタ: MI 上位 N 個を QUBO の候補に
        "n_features":       200,    # QUBO で最終的に選択する特徴量数
        "lambda_redundancy": 0.5,   # 冗長性ペナルティの重み (大きい=多様性重視)
        "time_limit_ms":    5000,   # ソルバー実行時間 [ms]
        "corr_threshold":   0.1,    # QUBO 項のスパース化閾値
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


# ============================================================
# 特徴量選択関数
# ============================================================

def select_none(X, y, config):
    """① 全特徴量を使用 (ベースライン)。"""
    return np.ones(X.shape[1], dtype=bool)


def select_filter(X, y, config):
    """② フィルター法: 相互情報量で上位 k 個を選択。"""
    cfg = config["filter"]
    k = cfg["n_features"]
    print(f"    相互情報量を計算中 ({X.shape[1]} 特徴量) ...")
    mi = mutual_info_regression(X, y, random_state=42)
    top_k = np.argsort(mi)[-k:]
    mask = np.zeros(X.shape[1], dtype=bool)
    mask[top_k] = True
    print(f"    MI 上位 {k} 個を選択")
    return mask


def select_wrapper(X, y, config):
    """③ ラッパー法: RFE (LightGBM) で上位 k 個を選択。"""
    import lightgbm as lgb
    from sklearn.feature_selection import RFE

    cfg = config["wrapper"]
    estimator = lgb.LGBMRegressor(
        n_estimators=cfg["n_estimators"],
        verbose=-1, random_state=42,
    )
    print(f"    RFE 実行中 (step={cfg['step']}) ...")
    selector = RFE(
        estimator,
        n_features_to_select=cfg["n_features"],
        step=cfg["step"],
    )
    selector.fit(X, y)
    print(f"    {selector.support_.sum()} 特徴量を選択")
    return selector.support_


def select_embedded(X, y, config):
    """④ 埋め込み法: Lasso の係数絶対値で上位 k 個を選択。"""
    cfg = config["embedded"]
    k = cfg["n_features"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"    Lasso 学習中 (alpha={cfg['alpha']}) ...")
    lasso = LassoFS(alpha=cfg["alpha"], max_iter=10000, random_state=42)
    lasso.fit(X_scaled, y)

    importance = np.abs(lasso.coef_)
    n_nonzero = np.sum(importance > 0)
    print(f"    非ゼロ係数: {n_nonzero} → 上位 {k} 個を選択")

    top_k = np.argsort(importance)[-k:]
    mask = np.zeros(X.shape[1], dtype=bool)
    mask[top_k] = True
    return mask


def select_amplify(X, y, config):
    """⑤ Amplify QUBO: mRMR 定式化を Amplify AE で求解。

    定式化 (mRMR - minimum Redundancy Maximum Relevance):
        minimize  - Σ_i  relevance_i · q_i
                  + λ · Σ_{i<j} redundancy_{ij} · q_i · q_j
        subject to  Σ_i q_i = k       (ちょうど k 個を選択)
        where       q_i ∈ {0, 1}      (L0 特徴量選択)

    この QUBO は D-Wave 量子アニーリングマシンでも
    そのまま求解可能な形式です。
    """
    if not AMPLIFY_AVAILABLE:
        raise ImportError(
            "Amplify SDK が必要です: pip install amplify"
        )

    cfg = config["amplify"]
    n_total = X.shape[1]
    n_cand = min(cfg["n_candidates"], n_total)
    n_select = cfg["n_features"]
    lambda_val = cfg["lambda_redundancy"]
    threshold = cfg["corr_threshold"]

    # ---- Step 1: 事前フィルタ (MI 上位 n_cand 個) ----
    print(f"    事前フィルタ: MI 上位 {n_cand} 候補を計算中 ...")
    mi = mutual_info_regression(X, y, random_state=42)
    top_cand = np.argsort(mi)[-n_cand:]
    X_cand = X[:, top_cand]

    # ---- Step 2: 関連度 / 冗長度の計算 ----
    relevance = mi[top_cand].copy()
    relevance /= (relevance.max() + 1e-10)    # [0, 1] に正規化

    corr = np.abs(np.corrcoef(X_cand.T))
    np.fill_diagonal(corr, 0)
    corr /= (corr.max() + 1e-10)              # [0, 1] に正規化

    # ---- Step 3: QUBO 定式化 ----
    print(f"    QUBO 構築中 (候補: {n_cand}, 選択: {n_select}) ...")
    gen = VariableGenerator()
    q = gen.array("Binary", n_cand)

    # 関連度項: maximize relevance → minimize -relevance
    obj = -sum(float(relevance[i]) * q[i] for i in range(n_cand))

    # 冗長度項: 相関が高いペアの同時選択にペナルティ
    rows, cols = np.where(np.triu(corr, k=1) > threshold)
    for i, j in zip(rows, cols):
        obj += float(lambda_val * corr[i, j]) * q[i] * q[j]
    print(f"    QUBO 項数: {len(rows)} (閾値={threshold})")

    # 制約: ちょうど n_select 個を選択
    constraint = equal_to(sum(q[i] for i in range(n_cand)), n_select)

    model = Model(obj, constraint)

    # ---- Step 4: Amplify AE で求解 ----
    # ※ D-Wave に切り替える場合はここを DWaveSamplerClient に差し替え
    client = AmplifyAEClient()
    client.token = cfg["token"]
    client.parameters.time_limit_ms = cfg["time_limit_ms"]

    print(f"    Amplify AE 求解中 ({cfg['time_limit_ms']} ms) ...")
    result = solve(model, client)

    if len(result) == 0:
        print("    ⚠ 解が見つかりませんでした。MI 上位を使用します。")
        top_k = np.argsort(mi)[-n_select:]
        mask = np.zeros(n_total, dtype=bool)
        mask[top_k] = True
        return mask

    # ---- Step 5: 結果の抽出 ----
    selected_vals = q.evaluate(result.best.values)
    selected_cand = np.where(np.array(selected_vals) == 1)[0]
    selected_global = top_cand[selected_cand]

    mask = np.zeros(n_total, dtype=bool)
    mask[selected_global] = True

    print(f"    選択数: {mask.sum()} (目的関数値: {result.best.objective:.6f})")
    return mask


# 戦略名 → 関数のマッピング (拡張時はここに追加するだけ)
STRATEGY_FUNCS = {
    "none":     select_none,
    "filter":   select_filter,
    "wrapper":  select_wrapper,
    "embedded": select_embedded,
    "amplify":  select_amplify,
}


# ============================================================
# 結果抽出ヘルパー (プロットなしで best_per_model を取得)
# ============================================================
def _get_best_models(df_results):
    """CV 結果から各モデルの最良前処理と全体ベストを取得する。"""
    best_per_model = (
        df_results
        .loc[df_results.groupby("Model")["CV-RMSE"].idxmin()]
        .sort_values("CV-RMSE")
    )
    overall_best = df_results.loc[df_results["CV-RMSE"].idxmin()]
    return best_per_model, overall_best


# ============================================================
# パイプライン実行 (1 戦略分)
# ============================================================
def run_strategy(strategy_name,
                 X_train_spec, X_test_spec,
                 X_train_cat, X_test_cat,
                 y_train, df_test, config_fs):
    """1 つの特徴量選択戦略でフルパイプラインを実行する。

    Returns
    -------
    dict  戦略名、特徴量数、各手法の CV-RMSE などのサマリ
    """
    label = STRATEGY_LABELS[strategy_name]
    _header(label)
    t0 = time.time()

    # ---- 特徴量選択 ----
    print("  特徴量選択:")
    mask = STRATEGY_FUNCS[strategy_name](X_train_spec, y_train, config_fs)
    n_sel = int(mask.sum())
    print(f"  → {n_sel} / {X_train_spec.shape[1]} 特徴量を使用")

    X_sel_train = X_train_spec[:, mask]
    X_sel_test  = X_test_spec[:, mask]

    # ---- パイプライン設定 (提出先フォルダを戦略ごとに分離) ----
    config = copy.deepcopy(BASE_CONFIG)
    config["submission_dir"] = f"submissions_{strategy_name}"
    config["save_figures"] = False   # 個別の図は省略

    # ---- モデル構築 → CV → スタッキング → 加重平均 → 提出 ----
    models = build_all_models(config)

    df_results, all_oof_train, all_test_preds = run_full_evaluation(
        X_sel_train, X_sel_test,
        X_train_cat, X_test_cat,
        y_train, models, config,
    )

    best_per_model, overall_best = _get_best_models(df_results)

    meta_ridge, meta_lasso, test_matrix, ridge_cv, lasso_cv, _ = (
        run_stacking(
            df_results, all_oof_train, all_test_preds,
            y_train, config,
        )
    )

    opt_w, wa_cv, wa_test_pred = run_weighted_average(
        df_results, all_oof_train, all_test_preds,
        y_train, config,
    )

    create_all_submissions(
        df_test, df_results, all_test_preds,
        meta_ridge, meta_lasso, test_matrix,
        wa_test_pred, overall_best,
        ridge_cv, lasso_cv, wa_cv, config,
    )

    elapsed = time.time() - t0
    best_cv = min(overall_best["CV-RMSE"], ridge_cv, lasso_cv, wa_cv)

    summary = {
        "strategy":    strategy_name,
        "label":       label,
        "n_features":  n_sel,
        "best_single": float(overall_best["CV-RMSE"]),
        "best_combo":  (f"{overall_best['Model']} × "
                        f"{overall_best['Preprocessor']}"),
        "ridge_cv":    ridge_cv,
        "lasso_cv":    lasso_cv,
        "wa_cv":       wa_cv,
        "best_cv":     best_cv,
        "time_sec":    elapsed,
    }

    print(f"\n  >>> ベスト CV-RMSE: {best_cv:.4f}  ({elapsed:.1f} 秒)")
    return summary


# ============================================================
# 全戦略の比較と可視化
# ============================================================
def compare_strategies(results: list):
    """全戦略の精度を比較し、グラフと表を出力する。"""
    _header("全戦略の比較")

    df = pd.DataFrame(results)
    out_dir = BASE_CONFIG["data_dir"]

    # ---- 比較表 ----
    print("\n  ┌─ 比較表 ─────────────────────────────────────────"
          "──────────────────────┐")
    for _, r in df.iterrows():
        print(f"  │ {r['label']:22s} │ "
              f"特徴量={r['n_features']:5d} │ "
              f"BestSingle={r['best_single']:.4f} │ "
              f"Ridge={r['ridge_cv']:.4f} │ "
              f"Lasso={r['lasso_cv']:.4f} │ "
              f"WA={r['wa_cv']:.4f} │ "
              f"BEST={r['best_cv']:.4f} │ "
              f"{r['time_sec']:.0f}s")
    print("  └──────────────────────────────────────────────────"
          "──────────────────────┘")

    # ---- グラフ 1: 全体ベスト CV-RMSE の比較 ----
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax = axes[0]
    df_s = df.sort_values("best_cv")
    colors = sns.color_palette("viridis", len(df_s))
    bars = ax.barh(df_s["label"], df_s["best_cv"], color=colors)
    ax.set_xlabel("Best CV-RMSE (低いほど良い)")
    ax.set_title("特徴量選択手法の精度比較")
    for bar, val in zip(bars, df_s["best_cv"]):
        ax.text(bar.get_width() + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=10, fontweight="bold")

    # ---- グラフ 2: 手法別アンサンブルスコア ----
    ax = axes[1]
    x = np.arange(len(df))
    w = 0.22
    ax.bar(x - w, df["best_single"], w, label="Best Single Model", alpha=0.85)
    ax.bar(x,     df["ridge_cv"],    w, label="Stacking Ridge",    alpha=0.85)
    ax.bar(x + w, df["wa_cv"],       w, label="Weighted Average",  alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [r["label"].split("(")[0].strip() for _, r in df.iterrows()],
        rotation=30, ha="right",
    )
    ax.set_ylabel("CV-RMSE")
    ax.set_title("手法別 × アンサンブル比較")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(
        os.path.join(out_dir, "feature_selection_comparison.png"), dpi=150)
    plt.close(fig)
    print("\n  保存: feature_selection_comparison.png")

    # ---- 最良戦略 ----
    best = df.loc[df["best_cv"].idxmin()]
    print("\n  ╔══════════════════════════════════════╗")
    print(f"  ║  最良戦略: {best['label']:24s}  ║")
    print(f"  ║  Best CV-RMSE: {best['best_cv']:.4f}              ║")
    print(f"  ║  特徴量数    : {best['n_features']:<24d}║")
    print("  ╚══════════════════════════════════════╝")


# ============================================================
# メイン実行
# ============================================================
def main():
    """全戦略を順に実行し、比較結果を出力する。"""

    # ---- データ読み込み (全戦略で共通) ----
    (df_train, df_test,
     X_train_spec, X_test_spec,
     X_train_cat, X_test_cat,
     y_train) = load_data(BASE_CONFIG)

    # ---- 各戦略を実行 ----
    results = []
    strategies = CONFIG_FS["strategies"]
    total = len(strategies)

    for idx, strategy in enumerate(strategies, 1):
        print(f"\n{'#' * 60}")
        print(f"#  戦略 {idx}/{total}: {STRATEGY_LABELS[strategy]}")
        print(f"{'#' * 60}")

        # Amplify 未インストール時はスキップ
        if strategy == "amplify" and not AMPLIFY_AVAILABLE:
            print("\n  ⚠ Amplify SDK 未インストール → スキップ")
            print("    インストール: pip install amplify")
            continue

        summary = run_strategy(
            strategy,
            X_train_spec, X_test_spec,
            X_train_cat, X_test_cat,
            y_train, df_test, CONFIG_FS,
        )
        results.append(summary)

    # ---- 全戦略の比較 ----
    if len(results) >= 2:
        compare_strategies(results)

    # ---- 完了 ----
    n_files = len(results) * 84
    _header("完了！")
    print(f"  {len(results)} 戦略 × 84 ファイル = {n_files} 個の提出ファイル")
    for r in results:
        print(f"    {r['label']:22s}  →  submissions_{r['strategy']}/")


if __name__ == "__main__":
    main()
