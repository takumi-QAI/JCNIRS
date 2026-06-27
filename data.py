"""
JCNIRS データ読み込み (data.py)
================================
CSV を読み込み、スペクトル特徴量 / カテゴリ特徴量 / ターゲットに分割します。
`species number` は train + test を統合してワンホットエンコーディングします。
"""

import os
import numpy as np
import pandas as pd


def _header(title: str):
    """ステップ見出しを表示する。"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def compute_groups(sample_numbers, X_spec, threshold=0.9999):
    """ボード(同一試料)単位のグループ ID を推定する。

    本データは同一ボードを乾燥させながら複数回スキャンした繰り返しを含む。
    sample number 順に並べ、隣接スペクトルの相関が threshold 未満になったら
    別ボードとみなしてグループ ID を増やす。これを GroupKFold に渡すことで
    同一ボードの行が train/val に跨らない honest CV を実現する。

    Returns
    -------
    groups : ndarray(int)  各行のボード ID (元の行順)
    """
    sn = np.asarray(sample_numbers)
    X = np.asarray(X_spec, dtype=np.float64)
    order = np.argsort(sn, kind="mergesort")
    Xo = X[order]
    # 行ごとに平均0・分散1へ正規化してドット積=相関に
    Xn = (Xo - Xo.mean(axis=1, keepdims=True)) / (Xo.std(axis=1, keepdims=True) + 1e-9)
    d = Xn.shape[1]
    gid_sorted = np.zeros(len(Xo), dtype=int)
    cur = 0
    for i in range(1, len(Xo)):
        corr = float(np.dot(Xn[i], Xn[i - 1]) / d)
        if corr < threshold:
            cur += 1
        gid_sorted[i] = cur
    groups = np.empty(len(Xo), dtype=int)
    groups[order] = gid_sorted
    return groups


def load_data(config: dict):
    """CSV を読み込み、スペクトル特徴量 / カテゴリ特徴量 / ターゲットに分割する。

    Returns
    -------
    df_train, df_test : DataFrame
    X_train_spec, X_test_spec : ndarray  スペクトル特徴量
    X_train_cat,  X_test_cat  : ndarray  ワンホットエンコード済みカテゴリ特徴量
    y_train : ndarray
    wavelengths : ndarray  スペクトル各列の波数 [cm^-1] (可視化用)
    """
    _header("Step 1: データ読み込み")
    d = config["data_dir"]
    enc = config["encoding"]

    df_train = pd.read_csv(os.path.join(d, config["train_file"]), encoding=enc)
    df_test  = pd.read_csv(os.path.join(d, config["test_file"]),  encoding=enc)

    # ---- スペクトル特徴量 ----
    exclude = set(config["drop_cols"] + config["onehot_cols"])
    spec_cols = [c for c in df_train.columns if c not in exclude]
    X_train_spec = df_train[spec_cols].values.astype(np.float64)
    X_test_spec  = df_test[spec_cols].values.astype(np.float64)

    # スペクトル列名 (波数) を float 配列へ。数値化できない場合は連番にフォールバック
    try:
        wavelengths = np.array([float(c) for c in spec_cols], dtype=np.float64)
    except (TypeError, ValueError):
        wavelengths = np.arange(len(spec_cols), dtype=np.float64)

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

    # ---- ボード単位グループ (GroupKFold 用、リーク防止) ----
    #   本データは 1 species = 1 ボード(乾燥曲線) であり、test は 6 ボードを
    #   丸ごと held-out している。よって species number をそのままボード ID に
    #   使うのが test 条件と一致する honest CV になる (既定)。
    #   group_by_species=False のときだけ隣接スペクトル相関で推定する。
    if not config.get("cv_grouped", True):
        groups = None
    elif config.get("group_by_species", True) and "species number" in df_train.columns:
        groups = df_train["species number"].values.astype(int)
    else:
        groups = compute_groups(
            df_train[config["id_col"]].values, X_train_spec,
            threshold=config.get("group_corr_threshold", 0.9999),
        )

    # ---- 表示 ----
    print(f"  Train           : {df_train.shape[0]} samples")
    print(f"  Test            : {df_test.shape[0]} samples")
    print(f"  スペクトル特徴量 : {X_train_spec.shape[1]}")
    print(f"  カテゴリ特徴量   : {X_train_cat.shape[1]}  "
          f"(one-hot: {config['onehot_cols']})")
    print(f"  合計特徴量       : {X_train_spec.shape[1] + X_train_cat.shape[1]}")
    if groups is not None:
        print(f"  推定ボード数     : {int(groups.max()) + 1}  "
              f"(GroupKFold でリーク防止)")

    return (df_train, df_test,
            X_train_spec, X_test_spec,
            X_train_cat, X_test_cat,
            y_train, wavelengths, groups)
