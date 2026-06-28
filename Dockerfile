FROM python:3.11-slim

WORKDIR /app

# gcc/g++ はビルド用、libgomp1 は LightGBM/XGBoost の OpenMP 実行に必要
RUN apt-get update && apt-get install -y gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 依存パッケージ
RUN pip install --no-cache-dir \
    numpy pandas matplotlib seaborn scikit-learn scipy \
    xgboost lightgbm python-dotenv

# PyTorch (深層モデル用)。既定は CPU ビルド (軽量・GPU 不要)。
#   GPU を使うなら下記を cu124 ビルドに変え、docker run に --gpus all を付ける:
#     torch --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# 量子アニーリング特徴量選択 (run_research の amplify 戦略) 用
RUN pip install --no-cache-dir amplify

COPY . /app

# 既定は「精度探索」。同じイメージで他のプログラムも実行できる:
#   精度の最善探索 :  docker run --rm -v ${PWD}:/app jcnirs                       (= search_best.py)
#   擬似ラベル検証 :  docker run --rm -v ${PWD}:/app jcnirs python pseudo_label.py
#   研究(選択比較) :  docker run --rm -v ${PWD}:/app jcnirs python run_research.py
#   動作確認(高速) :  docker run --rm -v ${PWD}:/app -e SEARCH_QUICK=1 jcnirs
CMD ["python", "search_best.py"]
