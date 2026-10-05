FROM python:3.12-slim

WORKDIR /app

COPY training/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ ./shared/
COPY training/ ./training/

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

CMD ["python", "training/train.py", "--csv", "/data/btcusdt_1d.csv", "--saida", "/model"]
