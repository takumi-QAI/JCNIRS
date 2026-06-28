"""
JCNIRS 量子アニーリング特徴量選択 (feature_selection_quantum.py)
=================================================================
⑤ Amplify QUBO ─ mRMR を QUBO 定式化し Amplify AE で求解する特徴量選択。

定式化 (mRMR - minimum Redundancy Maximum Relevance):
    minimize  - Σ_i  relevance_i · q_i
              + λ · Σ_{i<j} redundancy_{ij} · q_i · q_j
    subject to  Σ_i q_i = k       (ちょうど k 個を選択)
    where       q_i ∈ {0, 1}      (L0 特徴量選択)

この QUBO は D-Wave 量子アニーリングマシンでもそのまま求解可能な形式です。
将来 D-Wave に切り替える場合は ``client`` 部分を差し替えるだけで済みます。

このファイルは量子アニーリング (Amplify) に依存する特徴量選択のみを扱い、
古典的手法は feature_selection.py に分離されています。
"""

import numpy as np
from sklearn.feature_selection import mutual_info_regression

# Amplify SDK (未インストール時はスキップ)
try:
    from amplify import (
        VariableGenerator, Model,
        AmplifyAEClient, solve, equal_to,
    )
    AMPLIFY_AVAILABLE = True
except ImportError:
    AMPLIFY_AVAILABLE = False


def select_amplify(X, y, config):
    """⑤ Amplify QUBO: mRMR 定式化を Amplify AE で求解。

    ``(X, y, config_fs) -> bool マスク (shape: n_features,)`` を返す。
    """
    if not AMPLIFY_AVAILABLE:
        raise ImportError("Amplify SDK が必要です: pip install amplify")

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
    #   use_count_constraint=False で「特徴量数のペナルティ項」を外す → relevance と
    #   redundancy のバランスだけで自然に個数が決まる (研究比較用)。
    if cfg.get("use_count_constraint", True):
        constraint = equal_to(sum(q[i] for i in range(n_cand)), n_select)
        model = Model(obj, constraint)
        print(f"    制約: ちょうど {n_select} 個")
    else:
        model = Model(obj)
        print("    制約なし (個数ペナルティ項を除去 → 自然に個数が決まる)")

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
