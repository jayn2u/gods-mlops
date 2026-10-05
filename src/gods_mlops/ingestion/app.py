"""Run the independently deployable candidate-ingestion API."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from gods_mlops.jobs.queue import PostgresJobQueueRepository
from .collection_gate import CollectionStorageGate, MIN_UBUNTU_FILESYSTEM_FREE_BYTES
from .routes import build_ingestion_router
from .service import IngestionService
from .storage import PostgresIngestionRepository, S3SampleStore



@dataclass(frozen=True, slots=True)
class IngestionSettings:
    """Receiver connection targets loaded from environment-backed secrets."""

    database_url: str
    s3_endpoint_url: str
    s3_access_key: str
    s3_secret_key: str
    s3_bucket: str
    s3_region: str
    bearer_token: str
    ubuntu_host_identity: str | None = None
    ubuntu_gpu_uuid: str | None = None
    ubuntu_filesystem_identity: str | None = None
    ubuntu_storage_path: str | None = None
    ubuntu_storage_min_free_bytes: int = MIN_UBUNTU_FILESYSTEM_FREE_BYTES
    bind_host: str = "0.0.0.0"
    port: int = 8080


def build_app(
    settings: IngestionSettings,
    *,
    repository: PostgresIngestionRepository | None = None,
    objects: S3SampleStore | None = None,
) -> FastAPI:
    """Compose the production receiver while allowing isolated dependency tests."""
    if len(settings.bearer_token) < 32:
        raise ValueError("ingestion bearer token must contain at least 32 characters")
    database = repository or PostgresIngestionRepository(database_url=settings.database_url)
    object_store = objects or S3SampleStore(
        endpoint_url=settings.s3_endpoint_url,
        access_key=settings.s3_access_key,
        secret_key=settings.s3_secret_key,
        bucket=settings.s3_bucket,
        region=settings.s3_region,
    )
    resource_observations = PostgresJobQueueRepository(database_url=settings.database_url)
    collection_gate = CollectionStorageGate(
        repository=resource_observations,
        expected_host_identity=settings.ubuntu_host_identity,
        expected_gpu_uuid=settings.ubuntu_gpu_uuid,
        expected_filesystem_identity=settings.ubuntu_filesystem_identity,
        expected_storage_path=settings.ubuntu_storage_path,
        min_free_bytes=settings.ubuntu_storage_min_free_bytes,
    )
    ingestion = IngestionService(
        repository=database,
        objects=object_store,
        collection_gate=collection_gate,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await database.ensure_schema()
        await resource_observations.ensure_schema()
        try:
            yield
        finally:
            await database.close()
            await resource_observations.close()

    app = FastAPI(title="Gods candidate ingestion", lifespan=lifespan)
    app.include_router(build_ingestion_router(service=ingestion, bearer_token=settings.bearer_token))

    @app.get("/readyz", response_class=JSONResponse)
    async def ready() -> dict[str, str]:
        try:
            await ingestion.ready()
        except Exception as error:  # noqa: BLE001 - readiness must not leak connection details
            raise HTTPException(status_code=503, detail={"status": "not_ready"}) from error
        return {"status": "ready"}

    return app


def _settings_from_environment() -> IngestionSettings:
    """Read required process configuration without printing any secret values."""
    return IngestionSettings(
        database_url=_required("GODS_MLOPS_DATABASE_URL"),
        s3_endpoint_url=_required("GODS_MLOPS_S3_ENDPOINT_URL"),
        s3_access_key=_required("GODS_MLOPS_S3_ACCESS_KEY"),
        s3_secret_key=_required("GODS_MLOPS_S3_SECRET_KEY"),
        s3_bucket=_required("GODS_MLOPS_S3_BUCKET"),
        s3_region=os.environ.get("GODS_MLOPS_S3_REGION", "us-east-1"),
        bearer_token=_required("GODS_MLOPS_INGESTION_TOKEN"),
        ubuntu_host_identity=os.environ.get("GODS_MLOPS_UBUNTU_HOST_IDENTITY"),
        ubuntu_gpu_uuid=os.environ.get("GODS_MLOPS_UBUNTU_GPU_UUID"),
        ubuntu_filesystem_identity=os.environ.get("GODS_MLOPS_UBUNTU_FILESYSTEM_IDENTITY"),
        ubuntu_storage_path=os.environ.get("GODS_MLOPS_UBUNTU_STORAGE_PATH"),
        ubuntu_storage_min_free_bytes=int(
            os.environ.get("GODS_MLOPS_UBUNTU_STORAGE_MIN_FREE_BYTES", str(MIN_UBUNTU_FILESYSTEM_FREE_BYTES))
        ),
        bind_host=os.environ.get("GODS_MLOPS_BIND_HOST", "0.0.0.0"),
        port=int(os.environ.get("GODS_MLOPS_PORT", "8080")),
    )


def _required(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise ValueError(f"required setting is missing: {name}")
    return value


def create_application() -> FastAPI:
    """Create the process app from Kubernetes-injected environment configuration."""
    return build_app(_settings_from_environment())


def main() -> None:
    """Start Uvicorn with the service's constrained bind and port settings."""
    import uvicorn

    settings = _settings_from_environment()
    uvicorn.run(build_app(settings), host=settings.bind_host, port=settings.port, access_log=False)


__all__ = ["IngestionSettings", "build_app", "create_application", "main"]
