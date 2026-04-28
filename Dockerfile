FROM python:3.11-slim

ENV CHESS_BOT_CONFIG=configs/restible.yaml \
    CHESS_BOT_CHECKPOINT=data/models/full/best.pt \
    CHESS_BOT_SELECTION_POLICY=sample_probability_power \
    CHESS_BOT_PROBABILITY_EXPONENT=2.0 \
    PORT=8001 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY configs ./configs
COPY src ./src
COPY third_party ./third_party
COPY data/models/full/best.pt ./data/models/full/best.pt

RUN pip install --no-cache-dir ./third_party/maia2
RUN pip install --no-cache-dir .
RUN python -c "import torch; from restible_bot.maia2_model import load_pretrained_model; load_pretrained_model(torch.device('cpu'))"

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=3)"

CMD ["sh", "-c", "uvicorn restible_bot.api:app --host 0.0.0.0 --port ${PORT:-8001}"]
