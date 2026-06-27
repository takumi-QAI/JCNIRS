"""
JCNIRS パイプライン (pipeline.py)
==================================
特徴量選択後のデータを受け取り、以下を実行するコア処理:

Step 3 : 全 前処理×モデル の 5-Fold CV + テスト予測
Step 4 : 結果の可視化 (ヒートマップ / 棒グラフ)
Step 5 : スタッキング (Ridge / Lasso メタ学習器)
Step 6 : 加重平均アンサンブル
Step 7 : 全提出ファイルの作成

前処理は preprocessing.SpectralPreprocessor、モデルは models.build_all_models
を利用する。
"""

import os
import copy

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # GUI 不要 (画像はファイル保存のみ)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

from data import _header
from preprocessing import SpectralPreprocessor
from metrics import compute_metrics, cv_column, DEFAULT_METRICS
from target import forward_transform, inverse_transform, clip_predictions


# ============================================================
# 交差検証の分割 (ボード単位 GroupKFold or ランダム KFold)
# ============================================================
def make_splits(config, n_samples, groups=None):
    """fold の (train_idx, val_idx) リストを返す。

    config["cv_grouped"] が True かつ groups があればボード単位 GroupKFold。
    同一ボードが train/val に跨らないため honest な CV になる。
    """
    n_splits = config["n_splits"]
    if config.get("cv_grouped", True) and groups is not None:
        gkf = GroupKFold(n_splits=n_splits)
        dummy = np.zeros(n_samples)
        return list(gkf.split(dummy, dummy, groups))
    kf = KFold(n_splits=n_splits, shuffle=True,
               random_state=config["random_state"])
    return list(kf.split(np.zeros(n_samples)))


# ============================================================
# 結果抽出ヘルパー
# ============================================================
def get_best_models(df_results: pd.DataFrame):
    """CV 結果から各モデルの最良前処理と全体ベストを取得する。"""
    best_per_model = (
        df_results
        .loc[df_results.groupby("Model")["CV-RMSE"].idxmin()]
        .sort_values("CV-RMSE")
    )
    overall_best = df_results.loc[df_results["CV-RMSE"].idxmin()]
    return best_per_model, overall_best


# ============================================================
# Step 2 (任意): 前処理の確認
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
                        y_train, models, config, groups=None):
    """全 前処理×モデル の 5-Fold CV を実行し、OOF 予測とテスト予測を収集する。

    groups を渡すとボード単位 GroupKFold (リーク防止) になる。

    Returns
    -------
    df_results     : DataFrame  (Preprocessor, Model, combo_key,
                                 CV-RMSE, CV-MAE, CV-R2, CV-RPD, ... 各指標)
    all_oof_train  : dict  {combo_key: ndarray(n_train,)}  OOF 予測 (元スケール)
    all_test_preds : dict  {combo_key: ndarray(n_test,)}   テスト予測 (元スケール)
    """
    _header("Step 3: 交差検証 (全組み合わせ)")

    splits = make_splits(config, X_train_spec.shape[0], groups)
    if config.get("cv_grouped", True) and groups is not None:
        print(f"  CV: ボード単位 GroupKFold ({config['n_splits']} fold, "
              f"{int(groups.max()) + 1} ボード)")
    preprocessors = config["preprocessors"]
    metric_keys = config.get("metrics", DEFAULT_METRICS)
    tkind = config.get("target_transform", "none")
    clip = config.get("clip_predictions")
    n_train = X_train_spec.shape[0]
    n_test  = X_test_spec.shape[0]

    if tkind != "none":
        print(f"  目的変数の変換: {tkind} (予測は逆変換して元スケールで評価)")

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

            for fi, (tr_idx, val_idx) in enumerate(splits):
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

                # 目的変数を変換した空間で学習 → 予測は逆変換 → クリップ
                y_tr = forward_transform(y_train[tr_idx], tkind)
                m = copy.deepcopy(model_template)
                m.fit(Xtr, y_tr)
                pred_va = clip_predictions(
                    inverse_transform(np.asarray(m.predict(Xva)).flatten(), tkind),
                    clip)
                pred_te = clip_predictions(
                    inverse_transform(np.asarray(m.predict(Xte)).flatten(), tkind),
                    clip)
                oof[val_idx] = pred_va
                test_folds[:, fi] = pred_te

            all_oof_train[combo_key]  = oof
            all_test_preds[combo_key] = test_folds.mean(axis=1)

            # 全指標を元スケールで計算
            scores = compute_metrics(y_train, oof, metric_keys)
            row = {
                "Preprocessor": prep_name,
                "Model":        model_name,
                "combo_key":    combo_key,
            }
            for k, v in scores.items():
                row[cv_column(k)] = v
            results.append(row)
            print(f"CV-RMSE = {scores['RMSE']:.4f}  R2 = {scores['R2']:.3f}  "
                  f"RPD = {scores['RPD']:.2f}")

    df_results = pd.DataFrame(results)
    # 選択基準は CV-RMSE 列 (= cv_column("RMSE"))

    print("\n  --- CV 結果サマリ (CV-RMSE 上位 15) ---")
    show_cols = (["Model", "Preprocessor", "CV-RMSE"]
                 + [cv_column(k) for k in metric_keys if k != "RMSE"])
    print(df_results.sort_values("CV-RMSE").head(15)[show_cols]
          .to_string(index=False))

    return df_results, all_oof_train, all_test_preds


# ============================================================
# Step 4: 結果の可視化
# ============================================================
def visualize_results(df_results: pd.DataFrame, config: dict,
                      fig_prefix: str = ""):
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
        fig.savefig(os.path.join(out, f"{fig_prefix}heatmap_cv_rmse.png"),
                    dpi=dpi)
        print(f"    保存: {fig_prefix}heatmap_cv_rmse.png")
    plt.close(fig)

    # --- 各モデルの最良前処理 (棒グラフ) ---
    best_per_model, overall_best = get_best_models(df_results)

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
        fig.savefig(os.path.join(out, f"{fig_prefix}barplot_best_cv_rmse.png"),
                    dpi=dpi)
        print(f"    保存: {fig_prefix}barplot_best_cv_rmse.png")
    plt.close(fig)

    # --- 全体ベスト ---
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
                 y_train, config, groups=None):
    """各モデルの最良前処理を使ってスタッキングを行う。

    Returns
    -------
    meta_ridge, meta_lasso : 学習済みメタ学習器
    test_matrix : ndarray  ベースモデルのテスト予測行列
    ridge_cv, lasso_cv : float  CV-RMSE
    base_keys : list[str]  ベースモデルの combo_key リスト
    ridge_oof, lasso_oof : ndarray  メタ学習器の OOF 予測 (可視化用、元スケール)
    """
    _header("Step 5: スタッキング (Ridge / Lasso)")

    splits = make_splits(config, len(y_train), groups)
    clip = config.get("clip_predictions")

    best_per_model, _ = get_best_models(df_results)
    base_keys  = best_per_model["combo_key"].tolist()
    base_names = best_per_model["Model"].tolist()

    oof_matrix  = np.column_stack([all_oof_train[k]  for k in base_keys])
    test_matrix = np.column_stack([all_test_preds[k] for k in base_keys])
    n_train = oof_matrix.shape[0]

    print(f"\n  ベースモデル数: {len(base_keys)}")
    for key in base_keys:
        print(f"    - {key}")

    # --- Ridge メタ学習器 ---
    meta_ridge = Ridge(alpha=config["meta_ridge_alpha"])
    meta_ridge.fit(oof_matrix, y_train)

    ridge_oof = np.zeros(n_train)
    for tr_idx, val_idx in splits:
        m = Ridge(alpha=config["meta_ridge_alpha"])
        m.fit(oof_matrix[tr_idx], y_train[tr_idx])
        ridge_oof[val_idx] = m.predict(oof_matrix[val_idx])
    ridge_oof = clip_predictions(ridge_oof, clip)
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
    for tr_idx, val_idx in splits:
        m = Lasso(
            alpha=config["meta_lasso_alpha"],
            max_iter=config["meta_lasso_max_iter"],
        )
        m.fit(oof_matrix[tr_idx], y_train[tr_idx])
        lasso_oof[val_idx] = m.predict(oof_matrix[val_idx])
    lasso_oof = clip_predictions(lasso_oof, clip)
    lasso_cv = np.sqrt(mean_squared_error(y_train, lasso_oof))

    print(f"\n  [Stacking Lasso]  CV-RMSE = {lasso_cv:.4f}")
    print(f"    weights: "
          f"{dict(zip(base_names, np.round(meta_lasso.coef_, 4)))}")

    return (meta_ridge, meta_lasso, test_matrix, ridge_cv, lasso_cv,
            base_keys, ridge_oof, lasso_oof)


# ============================================================
# Step 6: 加重平均アンサンブル
# ============================================================
def run_weighted_average(df_results, all_oof_train, all_test_preds,
                         y_train, config, groups=None):
    """最適重みによる加重平均アンサンブル。

    Returns
    -------
    opt_w : ndarray        最適重み
    wa_cv : float          CV-RMSE
    wa_test_pred : ndarray テスト予測 (元スケール)
    wa_oof : ndarray       OOF 予測 (可視化用、元スケール)
    """
    _header("Step 6: 加重平均アンサンブル")

    splits = make_splits(config, len(y_train), groups)
    clip = config.get("clip_predictions")

    best_per_model, _ = get_best_models(df_results)
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
    for tr_idx, val_idx in splits:
        cv_res = minimize(
            _wa_rmse, init_w,
            args=(oof_matrix[tr_idx], y_train[tr_idx]),
            method="SLSQP", bounds=bounds, constraints=cons,
        )
        wa_oof[val_idx] = oof_matrix[val_idx] @ cv_res.x
    wa_oof = clip_predictions(wa_oof, clip)
    wa_cv = np.sqrt(mean_squared_error(y_train, wa_oof))

    print("\n  最適重み:")
    for name, w in zip(base_names, opt_w):
        print(f"    {name:12s}: {w:.4f}")
    print(f"\n  [Weighted Avg]  CV-RMSE = {wa_cv:.4f}")

    wa_test_pred = clip_predictions(test_matrix @ opt_w, clip)
    return opt_w, wa_cv, wa_test_pred, wa_oof


# ============================================================
# Step 7: 全提出ファイルの作成
# ============================================================
def create_all_submissions(
    df_test, df_results, all_test_preds,
    meta_ridge, meta_lasso, test_matrix,
    wa_test_pred, overall_best,
    ridge_cv, lasso_cv, wa_cv, config,
    y_train=None, ridge_oof=None, lasso_oof=None, wa_oof=None, top_k=10,
):
    """全モデル×前処理の個別予測 + アンサンブル予測を CSV に出力する。

    さらに、全候補 (個別 combo + アンサンブル 3 種) を CV-RMSE 昇順
    (= R2 / RPD / RPIQ でも最良順。これらは RMSE の単調関数) でランキングし、
    上位 top_k 件を sub_BEST_1.csv 〜 sub_BEST_{k}.csv として出力する。
    ランキング表は sub_BEST_ranking.csv に保存。
    """
    _header("Step 7: 全提出ファイルの作成")

    sub_dir = os.path.join(config["data_dir"], config["submission_dir"])
    os.makedirs(sub_dir, exist_ok=True)
    test_ids = df_test[config["id_col"]].values

    # 前回実行の sub_BEST*.csv を掃除 (古いランクの残骸が混ざらないように)
    for f in os.listdir(sub_dir):
        if f.startswith("sub_BEST"):
            try:
                os.remove(os.path.join(sub_dir, f))
            except OSError:
                pass

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
    clip = config.get("clip_predictions")
    ridge_pred = clip_predictions(meta_ridge.predict(test_matrix), clip)
    lasso_pred = clip_predictions(meta_lasso.predict(test_matrix), clip)

    _save("sub_ensemble_stacking_ridge.csv", ridge_pred)
    _save("sub_ensemble_stacking_lasso.csv", lasso_pred)
    _save("sub_ensemble_weighted_avg.csv",   wa_test_pred)
    print("    3 ファイル作成 (Ridge / Lasso / WA)")

    # ---- 全候補のランキング → 上位 top_k を sub_BEST_n.csv に出力 ----
    metric_keys = config.get("metrics", DEFAULT_METRICS)

    candidates = []
    # 個別 combo (指標は df_results に算出済み)
    for _, row in df_results.iterrows():
        scores = {m: float(row[cv_column(m)])
                  for m in metric_keys if cv_column(m) in row}
        candidates.append({
            "name":    row["combo_key"],
            "kind":    "single",
            "rmse":    float(row["CV-RMSE"]),
            "metrics": scores,
            "test":    all_test_preds[row["combo_key"]],
        })
    # アンサンブル 3 種 (OOF があれば全指標を算出、無ければ RMSE のみ)
    ens_items = [
        ("ensemble_stacking_ridge", ridge_pred, ridge_oof, ridge_cv),
        ("ensemble_stacking_lasso", lasso_pred, lasso_oof, lasso_cv),
        ("ensemble_weighted_avg",   wa_test_pred, wa_oof, wa_cv),
    ]
    for name, tpred, oof, rmse in ens_items:
        if oof is not None and y_train is not None:
            scores = compute_metrics(y_train, oof, metric_keys)
            rmse = scores["RMSE"]
        else:
            scores = {"RMSE": rmse}
        candidates.append({
            "name": name, "kind": "ensemble",
            "rmse": float(rmse), "metrics": scores, "test": tpred,
        })

    # CV-RMSE 昇順 = R2 / RPD / RPIQ でも最良順 (RMSE の単調変換のため)
    candidates.sort(key=lambda c: c["rmse"])
    k = min(top_k, len(candidates))

    print(f"\n  --- 上位 {k} を sub_BEST_1..{k}.csv に出力 ---")
    rank_rows = []
    for i, c in enumerate(candidates[:k], 1):
        _save(f"sub_BEST_{i}.csv", c["test"])
        rr = {"rank": i, "method": c["name"], "kind": c["kind"]}
        for m in metric_keys:
            rr[m] = c["metrics"].get(m, float("nan"))
        rank_rows.append(rr)

    rank_df = pd.DataFrame(rank_rows)
    rank_df.to_csv(os.path.join(sub_dir, "sub_BEST_ranking.csv"), index=False)
    print(f"    ランキング表: sub_BEST_ranking.csv")

    # ---- サマリ ----
    total = len(all_test_preds) + 3 + k
    print(f"\n  合計 {total} 個の提出ファイルを "
          f"{config['submission_dir']}/ に保存しました")

    print("\n  --- 提出候補ランキング (上位) ---")
    print(rank_df.to_string(index=False))
    print(f"\n  >>> 推奨提出: sub_BEST_1.csv  "
          f"({candidates[0]['name']}, CV-RMSE = {candidates[0]['rmse']:.4f})")
    print("      ※ CV-RMSE 順 = R2 / RPD / RPIQ 順 (これらは RMSE の単調関数)")

    best_pred = candidates[0]["test"]
    print("\n  sub_BEST_1 プレビュー (先頭 5 行):")
    print(pd.DataFrame({"id": test_ids, "pred": best_pred})
          .head().to_string(index=False))
    print(f"\n  テスト予測統計 (sub_BEST_1):")
    print(f"    Mean = {best_pred.mean():.2f}  "
          f"Std = {best_pred.std():.2f}  "
          f"Min = {best_pred.min():.2f}  "
          f"Max = {best_pred.max():.2f}")
