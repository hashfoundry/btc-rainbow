FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=UTC

WORKDIR /app

# Dependencies first so source edits do not invalidate the wheel install layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
# Baked in so `docker run btc-rainbow` works with no mounts. Compose bind-mounts
# ./data over this, which is what lets the updated CSV persist on the host.
COPY data/ ./data/

RUN mkdir -p /app/output

ENV DATA_FILE=/app/data/bitcoin_data.csv \
    OUTPUT_DIR=/app/output \
    MODELS=all \
    EXTEND_MONTHS=9 \
    EXCHANGES=binance,coinbase,kraken,bitstamp

ENTRYPOINT ["python", "src/main.py"]
