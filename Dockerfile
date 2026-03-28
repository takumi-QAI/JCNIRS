FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    numpy pandas matplotlib seaborn scikit-learn scipy \
    xgboost lightgbm python-dotenv

RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

COPY . /app

CMD ["python", "JCNIRS_feature_selection.py"]
