"""
JCNIRS 可視化 (visualization.py)
=================================
研究レポート向けの図を生成する関数群。各図の詳しい読み方・意味は
docs/REPORT_GUIDE.md を参照してください。

図は大きく 3 種類:

EDA (データ概観, 開始時に1回)
    plot_target_distribution   含水率の分布 (log1p 変換の動機)
    plot_spectra_overview      平均スペクトルと波長-含水率相関

戦略ごと (各特徴量選択戦略について)
    plot_metric_heatmaps       Model × 前処理 の指標ヒートマップ
    plot_pred_vs_actual        OOF 予測 vs 実測 (1:1 線)
    plot_residuals             残差プロット
    plot_model_metric_bars     モデル別ベストの指標バー

戦略横断 (量子アニーリング vs 古典 FS の比較=レポートの核)
    plot_strategy_comparison   戦略 × 指標 のグループ棒
    plot_features_vs_accuracy  特徴量数 vs 精度 (効率性)
    plot_runtime               戦略別の実行時間
    plot_feature_count         戦略別の選択特徴量数
    plot_selected_wavelengths  平均スペクトル上の選択波長
    plot_feature_overlap       戦略間の選択集合 Jaccard 類似度
    plot_qubo_diagnostics      選択特徴の関連度 vs 冗長度

注: 図中のテキストは日本語フォント非依存にするため英語表記。
意味の和文解説は docs/REPORT_GUIDE.md にまとめている。
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from metrics import METRIC_INFO, cv_column

sns.set_theme(style="whitegrid")


# ============================================================
# 共通ヘルパー
# ============================================================
def _fig_dir(config):
    d = os.path.join(config["data_dir"], config.get("figure_dir", "figures"))
    os.makedirs(d, exist_ok=True)
    return d


def _save(fig, config, fname):
    path = os.path.join(_fig_dir(config), fname)
    fig.tight_layout()
    fig.savefig(path, dpi=config.get("figure_dpi", 150), bbox_inches="tight")
    plt.close(fig)
    print(f"    保存: {os.path.join(config.get('figure_dir', 'figures'), fname)}")
    return path


def _annotate(ax, text):
    ax.text(0.05, 0.95, text, transform=ax.transAxes, va="top", ha="left",
            fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.8))


# ============================================================
# EDA 図
# ============================================================
def plot_target_distribution(y, config):
    """含水率の分布 (生 / 箱ひげ / log1p 後) を可視化する。"""
    y = np.asarray(y, dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].hist(y, bins=50, color="steelblue", edgecolor="white")
    axes[0].set_title("Moisture content (raw)")
    axes[0].set_xlabel("moisture [%]"); axes[0].set_ylabel("count")
    _annotate(axes[0],
              f"n={len(y)}\nmean={y.mean():.1f}\nmedian={np.median(y):.1f}\n"
              f"std={y.std():.1f}\nmin={y.min():.1f} max={y.max():.1f}")

    axes[1].boxplot(y, vert=True, widths=0.5)
    axes[1].set_title("Moisture content (boxplot)")
    axes[1].set_ylabel("moisture [%]")

    axes[2].hist(np.log1p(y), bins=50, color="darkorange", edgecolor="white")
    axes[2].set_title("log1p(moisture)  (training space)")
    axes[2].set_xlabel("log1p(moisture)"); axes[2].set_ylabel("count")

    fig.suptitle("Target distribution — motivation for log1p transform",
                 fontsize=13)
    return _save(fig, config, "eda_target_distribution.png")


def plot_spectra_overview(X_spec, wavelengths, y, config, n_samples=40):
    """平均スペクトル±std、サンプル例、波長-含水率の |相関| を可視化する。"""
    X_spec = np.asarray(X_spec, dtype=float)
    y = np.asarray(y, dtype=float)
    wl = np.asarray(wavelengths, dtype=float)

    mean_spec = X_spec.mean(axis=0)
    std_spec = X_spec.std(axis=0)

    # 各波長と含水率の相関 (絶対値)
    Xc = X_spec - X_spec.mean(axis=0)
    yc = y - y.mean()
    denom = (np.sqrt((Xc ** 2).sum(axis=0)) * np.sqrt((yc ** 2).sum()) + 1e-12)
    corr = np.abs((Xc * yc[:, None]).sum(axis=0) / denom)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    ax = axes[0]
    rng = np.random.RandomState(0)
    idx = rng.choice(len(X_spec), size=min(n_samples, len(X_spec)),
                     replace=False)
    for i in idx:
        ax.plot(wl, X_spec[i], color="gray", alpha=0.15, lw=0.6)
    ax.plot(wl, mean_spec, color="crimson", lw=2, label="mean")
    ax.fill_between(wl, mean_spec - std_spec, mean_spec + std_spec,
                    color="crimson", alpha=0.2, label="±1 std")
    ax.set_ylabel("absorbance")
    ax.set_title("NIR spectra: mean ± std and sample spectra")
    ax.legend(loc="upper right")

    ax = axes[1]
    ax.plot(wl, corr, color="navy", lw=1)
    ax.set_ylabel("|corr(wavelength, moisture)|")
    ax.set_xlabel("wavenumber [cm$^{-1}$]")
    ax.set_title("Absolute correlation of each wavelength with moisture")
    if wl[0] > wl[-1]:
        ax.invert_xaxis()

    return _save(fig, config, "eda_spectra_overview.png")


# ============================================================
# 戦略ごとの図
# ============================================================
def _metric_cmap(metric_key):
    """良し悪しの向きに応じた colormap を返す (緑=良い)。"""
    better = METRIC_INFO.get(metric_key, {}).get("better", "low")
    # 低いほど良い → 値が小さいほど緑 (reversed)、高いほど良い → そのまま
    return "RdYlGn_r" if better == "low" else "RdYlGn"


def plot_metric_heatmaps(df_results, config, prefix="",
                         metrics=("RMSE", "R2", "RPD")):
    """Model × 前処理 の指標ヒートマップ (指標ごとに 1 枚)。"""
    metrics = [m for m in metrics
               if cv_column(m) in df_results.columns or m == "RMSE"]
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6), squeeze=False)
    for ax, mkey in zip(axes[0], metrics):
        col = "CV-RMSE" if mkey == "RMSE" else cv_column(mkey)
        pivot = df_results.pivot(index="Model", columns="Preprocessor",
                                 values=col)
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap=_metric_cmap(mkey),
                    linewidths=0.5, ax=ax, cbar_kws={"label": mkey})
        ax.set_title(f"{mkey}  (Model × Preprocessor)")
    return _save(fig, config, f"{prefix}heatmap_metrics.png")


def plot_model_metric_bars(df_results, config, prefix="",
                           metrics=("RMSE", "R2", "RPD")):
    """各モデルのベスト前処理での指標を棒グラフで比較。"""
    metrics = [m for m in metrics
               if cv_column(m) in df_results.columns or m == "RMSE"]
    # CV-RMSE 最小の前処理を各モデルのベストとする
    best = df_results.loc[df_results.groupby("Model")["CV-RMSE"].idxmin()]
    best = best.sort_values("CV-RMSE")
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), squeeze=False)
    for ax, mkey in zip(axes[0], metrics):
        col = "CV-RMSE" if mkey == "RMSE" else cv_column(mkey)
        order = best.sort_values(
            col, ascending=(METRIC_INFO[mkey]["better"] == "low"))
        ax.barh(order["Model"], order[col],
                color=sns.color_palette("viridis", len(order)))
        ax.set_xlabel(mkey)
        ax.set_title(f"Best-per-model: {mkey}")
        ax.invert_yaxis()
    return _save(fig, config, f"{prefix}barplot_model_metrics.png")


def plot_pred_vs_actual(series, config, fname, suptitle=""):
    """OOF 予測 vs 実測の散布図 (1:1 線つき)。

    series : list[(name, y_true, y_pred)]
    """
    n = len(series)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5), squeeze=False)
    for ax, (name, y_true, y_pred) in zip(axes[0], series):
        y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
        ax.scatter(y_true, y_pred, s=10, alpha=0.4, color="steelblue")
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, label="1:1")
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) + 1e-12
        r2 = 1 - ss_res / ss_tot
        rpd = float(np.std(y_true) / (rmse + 1e-12))
        _annotate(ax, f"RMSE={rmse:.2f}\nR2={r2:.3f}\nRPD={rpd:.2f}")
        ax.set_xlabel("actual moisture [%]")
        ax.set_ylabel("predicted moisture [%]")
        ax.set_title(name)
        ax.legend(loc="lower right")
    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    return _save(fig, config, fname)


def plot_residuals(name, y_true, y_pred, config, fname):
    """残差 vs 予測 と 残差ヒストグラム。"""
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    resid = y_pred - y_true
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_pred, resid, s=10, alpha=0.4, color="teal")
    axes[0].axhline(0, color="red", ls="--", lw=1)
    axes[0].set_xlabel("predicted moisture [%]")
    axes[0].set_ylabel("residual (pred - actual)")
    axes[0].set_title(f"Residuals vs prediction — {name}")
    _annotate(axes[0], f"bias={resid.mean():+.2f}\nstd={resid.std():.2f}")

    axes[1].hist(resid, bins=40, color="teal", edgecolor="white")
    axes[1].axvline(0, color="red", ls="--", lw=1)
    axes[1].set_xlabel("residual (pred - actual)")
    axes[1].set_ylabel("count")
    axes[1].set_title("Residual distribution")

    return _save(fig, config, fname)


# ============================================================
# 戦略横断の図 (レポートの核)
# ============================================================
def plot_strategy_comparison(results, config,
                             metrics=("RMSE", "R2", "RPD", "RPIQ")):
    """戦略 × 指標 のグループ棒。各戦略のベスト pipeline の指標を比較。

    results : list[dict]  各 dict に "strategy" と "metrics_best"(dict) を含む
    """
    metrics = [m for m in metrics
               if all(m in r.get("metrics_best", {}) for r in results)]
    strategies = [r["strategy"] for r in results]
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    for ax, mkey in zip(axes[0], metrics):
        vals = [r["metrics_best"][mkey] for r in results]
        colors = sns.color_palette("viridis", len(strategies))
        bars = ax.bar(strategies, vals, color=colors)
        ax.set_title(f"{mkey} by FS strategy")
        ax.set_ylabel(mkey)
        ax.tick_params(axis="x", rotation=30)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Feature-selection strategies × accuracy metrics", fontsize=13)
    return _save(fig, config, "compare_strategy_metrics.png")


def plot_features_vs_accuracy(results, config):
    """特徴量数 vs 精度 (RMSE と R2)。QUBO の効率性 (少ない特徴で高精度) を示す。"""
    nfeat = [r["n_features"] for r in results]
    rmse = [r["metrics_best"]["RMSE"] for r in results]
    r2 = [r["metrics_best"]["R2"] for r in results]
    strategies = [r["strategy"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, vals, name, better in (
        (axes[0], rmse, "Best CV-RMSE", "low"),
        (axes[1], r2, "Best R2", "high"),
    ):
        ax.scatter(nfeat, vals, s=80, color="crimson", zorder=3)
        for x, yv, s in zip(nfeat, vals, strategies):
            ax.annotate(s, (x, yv), textcoords="offset points",
                        xytext=(6, 4), fontsize=9)
        ax.set_xlabel("number of selected features")
        ax.set_ylabel(name)
        ax.set_title(f"{name} vs #features ({'lower' if better=='low' else 'higher'} = better)")
    fig.suptitle("Accuracy vs feature-set size (efficiency of selection)",
                 fontsize=13)
    return _save(fig, config, "compare_features_vs_accuracy.png")


def plot_runtime(results, config):
    """戦略別の実行時間 (特徴量選択 + 学習)。量子求解のコスト位置づけ。"""
    strategies = [r["strategy"] for r in results]
    times = [r["time_sec"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(strategies, times,
                  color=sns.color_palette("mako", len(strategies)))
    ax.set_ylabel("time [sec]")
    ax.set_title("Runtime per FS strategy (selection + CV training)")
    ax.tick_params(axis="x", rotation=30)
    for b, v in zip(bars, times):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{v:.0f}s", ha="center", va="bottom", fontsize=9)
    return _save(fig, config, "compare_runtime.png")


def plot_feature_count(masks, config):
    """戦略別の選択特徴量数。"""
    strategies = list(masks.keys())
    counts = [int(np.sum(masks[s])) for s in strategies]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(strategies, counts,
                  color=sns.color_palette("crest", len(strategies)))
    ax.set_ylabel("number of selected features")
    ax.set_title("Number of selected wavelengths per strategy")
    ax.tick_params(axis="x", rotation=30)
    for b, v in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                str(v), ha="center", va="bottom", fontsize=9)
    return _save(fig, config, "compare_feature_count.png")


def plot_selected_wavelengths(masks, wavelengths, mean_spectrum, config):
    """平均スペクトル上に各戦略が選択した波長を帯で表示する。"""
    wl = np.asarray(wavelengths, float)
    mean_spec = np.asarray(mean_spectrum, float)
    strategies = list(masks.keys())
    n = len(strategies)
    fig, axes = plt.subplots(n, 1, figsize=(13, 2.1 * n), squeeze=False,
                             sharex=True)
    for ax, s in zip(axes[:, 0], strategies):
        ax.plot(wl, mean_spec, color="gray", lw=1)
        sel = np.asarray(masks[s], bool)
        ymin, ymax = mean_spec.min(), mean_spec.max()
        ax.vlines(wl[sel], ymin, ymax, color="crimson", alpha=0.25, lw=0.5)
        ax.set_ylabel("absorb.")
        ax.set_title(f"{s}  ({int(sel.sum())} wavelengths selected)",
                     fontsize=10, loc="left")
        if wl[0] > wl[-1]:
            ax.invert_xaxis()
    axes[-1, 0].set_xlabel("wavenumber [cm$^{-1}$]")
    fig.suptitle("Selected wavelengths overlaid on the mean spectrum",
                 fontsize=13)
    return _save(fig, config, "compare_selected_wavelengths.png")


def plot_feature_overlap(masks, config):
    """戦略間で選択された特徴量集合の Jaccard 類似度ヒートマップ。"""
    strategies = list(masks.keys())
    n = len(strategies)
    J = np.zeros((n, n))
    for i, a in enumerate(strategies):
        for j, b in enumerate(strategies):
            sa = np.asarray(masks[a], bool)
            sb = np.asarray(masks[b], bool)
            inter = np.sum(sa & sb)
            union = np.sum(sa | sb)
            J[i, j] = inter / union if union > 0 else 0.0
    fig, ax = plt.subplots(figsize=(1.3 * n + 3, 1.0 * n + 3))
    sns.heatmap(J, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=strategies, yticklabels=strategies,
                vmin=0, vmax=1, ax=ax, cbar_kws={"label": "Jaccard similarity"})
    ax.set_title("Overlap of selected feature sets (Jaccard)")
    return _save(fig, config, "compare_feature_overlap.png")


def plot_qubo_diagnostics(masks, X_spec, y, config):
    """選択特徴の「関連度」と「冗長度」を戦略間で比較する。

    relevance  = mean |corr(selected feature, target)|     (高いほど良い)
    redundancy = mean |corr| among selected features pairs (低いほど良い)

    mRMR/QUBO (Amplify) は高関連・低冗長を狙う定式化のため、
    フィルタ法 (隣接波長の冗長性が高くなりがち) との差が現れることを示す。
    """
    X_spec = np.asarray(X_spec, float)
    y = np.asarray(y, float)

    # 各波長と目的変数の相関 (関連度)
    Xc = X_spec - X_spec.mean(axis=0)
    yc = y - y.mean()
    denom = (np.sqrt((Xc ** 2).sum(axis=0)) * np.sqrt((yc ** 2).sum()) + 1e-12)
    corr_y = np.abs((Xc * yc[:, None]).sum(axis=0) / denom)

    strategies = list(masks.keys())
    relevance, redundancy = [], []
    for s in strategies:
        sel = np.where(np.asarray(masks[s], bool))[0]
        relevance.append(float(np.mean(corr_y[sel])) if len(sel) else 0.0)
        if len(sel) > 1:
            # 冗長度: 選択特徴間の平均 |相関| (上三角)
            sub = X_spec[:, sel]
            cm = np.abs(np.corrcoef(sub.T))
            iu = np.triu_indices(len(sel), k=1)
            redundancy.append(float(np.mean(cm[iu])))
        else:
            redundancy.append(0.0)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    colors = sns.color_palette("viridis", len(strategies))

    axes[0].bar(strategies, relevance, color=colors)
    axes[0].set_title("Mean relevance  |corr(feature, target)|  (higher=better)")
    axes[0].set_ylabel("mean |corr| with target")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(strategies, redundancy, color=colors)
    axes[1].set_title("Mean redundancy  |corr| among selected  (lower=better)")
    axes[1].set_ylabel("mean pairwise |corr|")
    axes[1].tick_params(axis="x", rotation=30)

    # 関連度-冗長度 平面 (右上=高関連, 下=低冗長が理想)
    axes[2].scatter(redundancy, relevance, s=90, color="crimson", zorder=3)
    for x, yv, s in zip(redundancy, relevance, strategies):
        axes[2].annotate(s, (x, yv), textcoords="offset points",
                         xytext=(6, 4), fontsize=9)
    axes[2].set_xlabel("redundancy (lower better →)")
    axes[2].set_ylabel("relevance (higher better ↑)")
    axes[2].set_title("Relevance vs redundancy (mRMR view)")

    fig.suptitle("Feature-selection quality: relevance vs redundancy",
                 fontsize=13)
    return _save(fig, config, "compare_qubo_diagnostics.png")
