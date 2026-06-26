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
            y_train, wavelengths)
