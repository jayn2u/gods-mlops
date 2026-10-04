FROM ghcr.io/astral-sh/uv:0.12.20@sha256:100047e74f30778ab704942321a09750d6158739573ff58bf3924085cc6cd2d8 AS uv

FROM docker.io/library/python:3.12-slim-bookworm@sha256:54c85f3c47607a77f32adec749d3c81d1348bf25833671f512b26a9b6d778cb3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    GODS_MODEL_CACHE=/mnt/model-cache \
    GODS_MLOPS_ROOT=/app \
    HOME=/home/gods \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

COPY --from=uv /uv /uvx /usr/local/bin/
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY models/lock.json ./models/lock.json
COPY infra/versions.lock.yaml ./infra/versions.lock.yaml
RUN uv sync --frozen --no-dev --no-editable \
    && mkdir -p /home/gods \
    && mkdir -p /mnt/model-cache \
    && chown -R 10001:10001 /home/gods /mnt/model-cache

ENV PATH="/opt/venv/bin:${PATH}"
USER 10001:10001

ENTRYPOINT ["gods-mlops"]
CMD ["check-training-image"]
