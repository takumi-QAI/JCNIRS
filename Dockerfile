FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    numpy pandas matplotlib seaborn scikit-learn scipy \
    xgboost lightgbm python-dotenv

# PyTorch (CUDA 12.4 ビルド)。GPU を使うには docker run 時に --gpus all を付ける。
# ※ NVIDIA GPU が無い / GPU 不要なら下記を CPU ビルドに差し替え可:
#     torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir amplify

COPY . /app

CMD ["python", "run_all.py"]
