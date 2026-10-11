FROM ghcr.io/astral-sh/uv:0.12.20@sha256:100047e74f30778ab704942321a09750d6158739573ff58bf3924085cc6cd2d8 AS uv-tools

FROM docker.io/library/python:3.12-slim-bookworm@sha256:54c85f3c47607a77f32adec749d3c81d1348bf25833671f512b26a9b6d778cb3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /opt/gods-mlops
COPY --from=uv-tools /uv /uvx /usr/local/bin/
COPY pyproject.toml uv.lock ./
RUN apt-get update \
    && apt-get install --no-install-recommends -y openssh-client=1:9.2p1-2+deb12u10 \
    && rm -rf /var/lib/apt/lists/*
RUN groupadd --gid 10001 gods-mlops \
    && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /nonexistent gods-mlops
RUN uv export --locked --no-dev --no-emit-project --no-hashes --output-file /tmp/uv-constraints.txt
COPY infra/kubeflow/ingestion/requirements.txt /tmp/ingestion-requirements.txt
RUN pip install --no-cache-dir --constraint /tmp/uv-constraints.txt --requirement /tmp/ingestion-requirements.txt \
    && rm -f /tmp/uv-constraints.txt /tmp/ingestion-requirements.txt

COPY src/gods_mlops/__init__.py /opt/gods-mlops/gods_mlops/__init__.py
COPY src/gods_mlops/ingestion /opt/gods-mlops/gods_mlops/ingestion
COPY src/gods_mlops/annotations /opt/gods-mlops/gods_mlops/annotations
COPY src/gods_mlops/datasets /opt/gods-mlops/gods_mlops/datasets
COPY src/gods_mlops/migrations /opt/gods-mlops/gods_mlops/migrations
COPY src/gods_mlops/retention /opt/gods-mlops/gods_mlops/retention
COPY src/gods_mlops/jobs /opt/gods-mlops/gods_mlops/jobs

ENV PYTHONPATH=/opt/gods-mlops
USER 10001:10001
EXPOSE 8080
ENTRYPOINT ["uvicorn", "gods_mlops.ingestion.app:create_application", "--factory", "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]
