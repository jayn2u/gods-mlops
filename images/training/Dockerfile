FROM ghcr.io/astral-sh/uv:0.12.20@sha256:100047e74f30778ab704942321a09750d6158739573ff58bf3924085cc6cd2d8 AS uv

FROM docker.io/library/python:3.12-slim-bookworm@sha256:54c85f3c47607a77f32adec749d3c81d1348bf25833671f512b26a9b6d778cb3

ARG GODS_MLOPS_SOURCE_REVISION
LABEL org.opencontainers.image.revision=${GODS_MLOPS_SOURCE_REVISION}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_CACHE_DIR=/tmp/uv-cache \
    GODS_MODEL_CACHE=/mnt/model-cache \
    GODS_MLOPS_ROOT=/app \
    HOME=/home/gods \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

COPY --from=uv /uv /uvx /usr/local/bin/
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends openssh-client=1:9.2p1-2+deb12u10 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/tmp/uv-cache,sharing=locked \
    uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY models/lock.json ./models/lock.json
COPY infra/versions.lock.yaml ./infra/versions.lock.yaml
RUN --mount=type=cache,target=/tmp/uv-cache,sharing=locked \
    uv sync --frozen --no-dev --no-editable \
    && printf 'gods:x:10001:10001:Gods MLOps:/home/gods:/usr/sbin/nologin\n' >> /etc/passwd \
    && printf 'gods:x:10001:\n' >> /etc/group \
    && mkdir -p /home/gods \
    && mkdir -p /mnt/model-cache \
    && chown 10001:10001 /home/gods /mnt/model-cache

ENV PATH="/opt/venv/bin:${PATH}"
USER 10001:10001

ENTRYPOINT ["gods-mlops"]
CMD ["check-training-image"]
