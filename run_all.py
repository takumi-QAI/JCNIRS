"""
JCNIRS 全実行スクリプト (run_all.py)
=====================================
これ 1 本を実行すれば、全特徴量選択戦略 × 全モデル × 全前処理 +
アンサンブルの提出ファイル生成、評価指標の算出、研究レポート用の
可視化までを通しで行います。

    python run_all.py

処理フロー
----------
1. データ読み込み (data.load_data)
2. EDA 図 (含水率分布・スペクトル概観) を出力
3. 各特徴量選択戦略でスペクトル特徴量を選択
   ① none ② filter ③ wrapper ④ embedded ⑤ amplify(量子アニーリング)
4. 戦略ごとに pipeline で CV → スタッキング → 加重平均 → 提出生成 → 戦略別の図
5. 全戦略の精度・特徴量選択品質を比較しグラフ化 (figures/ に出力)

出力
----
submissions_<strategy>/    各戦略の提出ファイル群
figures/                   EDA・戦略別・戦略横断の図 (PNG)

各図・各指標の詳しい意味は docs/REPORT_GUIDE.md を参照。

⚠ 実行時間について
------------------
既定では「全16モデル × 全9前処理 × 全5戦略 × 5-fold CV」を実行します。
深層モデルを含むため CPU 環境では数時間規模です。速度を優先する場合は
config.py の CONFIG["models"] / CONFIG["preprocessors"] / CONFIG_FS["strategies"]
を絞り込んでください (該当行をコメントアウトするだけ)。
"""

import os
import sys
import copy
import time
import datetime
import warnings

import numpy as np
import pandas as pd

from config import CONFIG, CONFIG_FS, STRATEGY_LABELS
from data import load_data, _header
from models import build_all_models
from metrics import compute_metrics, METRIC_INFO
from pipeline import (
    run_full_evaluation, run_stacking, run_weighted_average,
    create_all_submissions, get_best_models,
)
from feature_selection import SELECTORS as CLASSICAL_SELECTORS
from feature_selection_quantum import select_amplify, AMPLIFY_AVAILABLE
import visualization as viz

warnings.filterwarnings("ignore")


# 古典的手法 + 量子アニーリング手法を統合した戦略ディスパッチ
SELECTORS = {**CLASSICAL_SELECTORS, "amplify": select_amplify}


# ============================================================
# ログのファイル出力 (コンソール + logs/run_<日時>.log)
# ============================================================
class _Tee:
    """複数ストリームへ同時書き込みする (画面とログファイル)。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except UnicodeEncodeError:
                # コンソールが cp932 等で一部文字を出せない場合の保険
                enc = getattr(s, "encoding", None) or "ascii"
                s.write(data.encode(enc, errors="replace").decode(enc))
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def setup_logging(config):
    """標準出力/標準エラーを logs/run_<日時>.log にも複製する。"""
    log_dir = os.path.join(config["data_dir"], config.get("log_dir", "logs"))
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(log_dir, f"run_{ts}.log")
    f = open(path, "a", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, f)
    sys.stderr = _Tee(sys.__stderr__, f)
    return path


# ============================================================
# パイプライン実行 (1 戦略分)
# ============================================================
def run_strategy(strategy_name,
                 X_train_spec, X_test_spec,
                 X_train_cat, X_test_cat,
                 y_train, df_test, config_fs, groups=None,
                 train_sn=None, test_sn=None, test_groups=None,
                 train_board=None):
    """1 つの特徴量選択戦略でフルパイプラインを実行する。

    Returns
    -------
    dict  戦略名・特徴量数・各指標・選択マスクなどのサマリ
    """
    label = STRATEGY_LABELS[strategy_name]
    _header(label)
    t0 = time.time()

    # ---- 特徴量選択 ----
    #   ★ CV の中(各 fold の train だけ)で選択する = リーク防止 (run_full_evaluation)。
    #     ここで作る mask は「全データ基準」で、可視化(選択波長図)とサマリ表示にのみ使う。
    print("  特徴量選択 (CV は fold 内で再選択; 下記は全データ基準の表示用):")
    mask = SELECTORS[strategy_name](X_train_spec, y_train, config_fs)
    n_sel = int(mask.sum())
    print(f"  → 全データ基準 {n_sel} / {X_train_spec.shape[1]} 特徴 "
          f"(実際の CV は fold ごとに再選択)")

    # ---- パイプライン設定 (提出先フォルダを戦略ごとに分離) ----
    config = copy.deepcopy(CONFIG)
    config["submission_dir"] = f"submissions_{strategy_name}"
    config["save_figures"] = False   # pipeline 内のヒートマップは省略 (figures は別途)
    #   _scatter_ref / _wavelengths は全(生)スペクトル基準のまま渡す。
    #   fold ごとの選択次元への整合は run_full_evaluation 内で行う。

    # ---- モデル構築 → CV(fold内選択) → スタッキング → 加重平均 → 提出 ----
    models = build_all_models(config)

    df_results, all_oof_train, all_test_preds = run_full_evaluation(
        X_train_spec, X_test_spec,
        X_train_cat, X_test_cat,
        y_train, models, config, groups=groups,
        train_sn=train_sn, test_sn=test_sn, test_groups=test_groups,
        train_board=train_board,
        selector=SELECTORS[strategy_name], config_fs=config_fs,
    )

    best_per_model, overall_best = get_best_models(df_results)

    (meta_ridge, meta_lasso, test_matrix, ridge_cv, lasso_cv,
     _, ridge_oof, lasso_oof) = run_stacking(
        df_results, all_oof_train, all_test_preds, y_train, config,
        groups=groups, train_sn=train_sn, train_board=train_board,
    )

    opt_w, wa_cv, wa_test_pred, wa_oof = run_weighted_average(
        df_results, all_oof_train, all_test_preds, y_train, config,
        groups=groups, train_sn=train_sn, test_sn=test_sn,
        test_groups=test_groups, train_board=train_board,
    )

    create_all_submissions(
        df_test, df_results, all_test_preds,
        meta_ridge, meta_lasso, test_matrix,
        wa_test_pred, overall_best,
        ridge_cv, lasso_cv, wa_cv, config,
        y_train=y_train,
        ridge_oof=ridge_oof, lasso_oof=lasso_oof, wa_oof=wa_oof,
        top_k=10,
        test_sn=test_sn, test_groups=test_groups,
        all_oof_train=all_oof_train, blend_top_k=5,
    )

    # ---- ベスト手法の OOF と全指標 ----
    single_oof = all_oof_train[overall_best["combo_key"]]
    method_oof = {
        "single": single_oof,
        "stacking_ridge": ridge_oof,
        "stacking_lasso": lasso_oof,
        "weighted_avg": wa_oof,
    }
    method_rmse = {
        "single": float(overall_best["CV-RMSE"]),
        "stacking_ridge": ridge_cv,
        "stacking_lasso": lasso_cv,
        "weighted_avg": wa_cv,
    }
    best_method = min(method_rmse, key=method_rmse.get)
    best_oof = method_oof[best_method]
    metrics_best = compute_metrics(y_train, best_oof, config["metrics"])
    # 最良アンサンブル (single を除く) を pred-vs-actual に併記
    ens_rmse = {k: v for k, v in method_rmse.items() if k != "single"}
    best_ens = min(ens_rmse, key=ens_rmse.get)

    # ---- 戦略別の図 (figures/strategies/<strategy>/ に出力) ----
    fig_subdir = f"strategies/{strategy_name}"
    try:
        viz.plot_metric_heatmaps(df_results, config, prefix=f"{strategy_name}_",
                                 subdir=fig_subdir)
        viz.plot_model_metric_bars(df_results, config, prefix=f"{strategy_name}_",
                                   subdir=fig_subdir)
        viz.plot_pred_vs_actual(
            [(f"single: {overall_best['Model']}×{overall_best['Preprocessor']}",
              y_train, single_oof),
             (f"ensemble: {best_ens}", y_train, method_oof[best_ens])],
            config, fname=f"{strategy_name}_pred_vs_actual.png",
            suptitle=f"{label}: OOF prediction vs actual", subdir=fig_subdir,
        )
        viz.plot_residuals(
            f"{strategy_name} / {best_method}", y_train, best_oof,
            config, fname=f"{strategy_name}_residuals.png", subdir=fig_subdir,
        )
    except Exception as e:    # 図の失敗で全体を止めない
        print(f"  ⚠ 戦略別の図生成に失敗: {e}")

    elapsed = time.time() - t0
    best_cv = method_rmse[best_method]

    summary = {
        "strategy":     strategy_name,
        "label":        label,
        "n_features":   n_sel,
        "best_single":  float(overall_best["CV-RMSE"]),
        "best_combo":   (f"{overall_best['Model']} × "
                         f"{overall_best['Preprocessor']}"),
        "ridge_cv":     ridge_cv,
        "lasso_cv":     lasso_cv,
        "wa_cv":        wa_cv,
        "best_cv":      best_cv,
        "best_method":  best_method,
        "metrics_best": metrics_best,
        "mask":         mask,
        "time_sec":     elapsed,
    }

    print(f"\n  >>> ベスト ({best_method}) CV-RMSE: {best_cv:.4f}  "
          f"R2: {metrics_best['R2']:.3f}  RPD: {metrics_best['RPD']:.2f}  "
          f"({elapsed:.1f} 秒)")
    return summary


# ============================================================
# 全戦略の比較と可視化
# ============================================================
def compare_strategies(results: list):
    """全戦略の精度・特徴量選択品質を比較し、表と図を出力する。"""
    _header("全戦略の比較")

    # ---- 比較表 (各戦略のベスト pipeline の全指標) ----
    metric_keys = list(results[0]["metrics_best"].keys())
    header = (f"  {'strategy':10s} {'#feat':>6s} {'method':>14s}  "
              + "  ".join(f"{k:>8s}" for k in metric_keys))
    print("\n" + header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        vals = "  ".join(f"{r['metrics_best'][k]:8.3f}" for k in metric_keys)
        print(f"  {r['strategy']:10s} {r['n_features']:6d} "
              f"{r['best_method']:>14s}  {vals}")

    print("\n  (指標の意味は docs/REPORT_GUIDE.md を参照)")

    # ---- 戦略横断の図 (結果ベース) ----
    try:
        viz.plot_strategy_comparison(results, CONFIG)
        viz.plot_features_vs_accuracy(results, CONFIG)
        viz.plot_runtime(results, CONFIG)
    except Exception as e:
        print(f"  ⚠ 戦略横断図 (結果) の生成に失敗: {e}")

    # ---- 最良戦略 ----
    best = min(results, key=lambda r: r["best_cv"])
    print("\n  ╔══════════════════════════════════════╗")
    print(f"  ║  最良戦略: {best['label']:24s}  ║")
    print(f"  ║  Best CV-RMSE: {best['best_cv']:.4f}              ║")
    print(f"  ║  特徴量数    : {best['n_features']:<24d}║")
    print("  ╚══════════════════════════════════════╝")


def visualize_feature_selection(results, wavelengths, X_train_spec, y_train):
    """特徴量選択そのものの品質を可視化する (マスクベース)。"""
    masks = {r["strategy"]: r["mask"] for r in results}
    try:
        viz.plot_feature_count(masks, CONFIG)
        viz.plot_selected_wavelengths(
            masks, wavelengths, X_train_spec.mean(axis=0), CONFIG)
        viz.plot_feature_overlap(masks, CONFIG)
        viz.plot_qubo_diagnostics(masks, X_train_spec, y_train, CONFIG)
    except Exception as e:
        print(f"  ⚠ 戦略横断図 (特徴量選択) の生成に失敗: {e}")


# ============================================================
# メイン実行
# ============================================================
def main():
    """全戦略を順に実行し、比較結果を出力する。"""

    # ---- ログのファイル出力を開始 ----
    log_path = setup_logging(CONFIG)
    print(f"  ログ出力: {os.path.relpath(log_path, CONFIG['data_dir'])}")

    # ---- データ読み込み (全戦略で共通) ----
    (df_train, df_test,
     X_train_spec, X_test_spec,
     X_train_cat, X_test_cat,
     y_train, wavelengths, groups) = load_data(CONFIG)

    # 乾燥曲線の後処理 (ボード単位の単調平滑化) 用の情報
    #   train_sn / test_sn  : スキャン順 (sample number)
    #   test_groups         : テスト各行のボード ID (= species number)
    train_sn = df_train[CONFIG["id_col"]].values
    test_sn  = df_test[CONFIG["id_col"]].values
    #   平滑化のボード ID は真のボード = species number (CV の groups とは別)
    train_board = (df_train["species number"].values.astype(int)
                   if "species number" in df_train.columns else None)
    test_groups = (df_test["species number"].values.astype(int)
                   if "species number" in df_test.columns else None)

    # トランスダクティブ散乱補正 (msc_t / emsc_t) の参照 = train+test 合算の平均
    #   (test の *スペクトル* のみ使用。ラベルは使わない = 合法)。
    # 水バンド特徴 (wband) 用に波長軸も注入する。
    CONFIG["_scatter_ref"] = np.vstack([X_train_spec, X_test_spec]).mean(axis=0)
    CONFIG["_wavelengths"] = wavelengths

    # ---- EDA 図 ----
    if CONFIG.get("eda_figures", True):
        _header("EDA 図の生成")
        try:
            viz.plot_target_distribution(y_train, CONFIG)
            viz.plot_spectra_overview(X_train_spec, wavelengths, y_train, CONFIG)
        except Exception as e:
            print(f"  ⚠ EDA 図の生成に失敗: {e}")

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
            y_train, df_test, CONFIG_FS, groups=groups,
            train_sn=train_sn, test_sn=test_sn, test_groups=test_groups,
            train_board=train_board,
        )
        results.append(summary)

    # ---- 全戦略の比較 ----
    if len(results) >= 2:
        compare_strategies(results)
        visualize_feature_selection(results, wavelengths, X_train_spec, y_train)

    # ---- 完了 ----
    _header("完了！")
    print(f"  図は {CONFIG.get('figure_dir', 'figures')}/ に出力 "
          f"(意味は docs/REPORT_GUIDE.md)")
    for r in results:
        m = r["metrics_best"]
        print(f"    {r['label']:22s}  →  submissions_{r['strategy']}/  "
              f"(RMSE={m['RMSE']:.3f}  R2={m['R2']:.3f}  RPD={m['RPD']:.2f})")


if __name__ == "__main__":
    main()
