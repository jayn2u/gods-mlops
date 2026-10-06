"""Compose the loopback-only operator UI from existing MLOps services."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import urlsplit

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from gods_mlops.annotations.label_studio import LabelStudioApiClient, LabelStudioMediaCleanupClient
from gods_mlops.annotations.service import AnnotationService
from gods_mlops.annotations.storage import PostgresAnnotationRepository
from gods_mlops.annotations.workflow import LabelStudioReviewWorkflow
from gods_mlops.datasets.publish import DatasetObjectStore, DatasetPublisher
from gods_mlops.ingestion.service import IngestionService
from gods_mlops.ingestion.storage import PostgresIngestionRepository, S3SampleStore
from gods_mlops.jobs.queue import JobQueue, PostgresJobQueueRepository
from gods_mlops.jobs.sources import DatasetSourceRegistry
from gods_mlops.training.artifacts import S3ResultArtifactStore

from .auth import OperatorAuth, OperatorAuthSettings
from .routes import build_operator_router

_WEB_ROOT = Path(__file__).parent


@dataclass(frozen=True, slots=True)
class OperatorUIIntegrations:
    label_studio_url: str | None = None
    label_studio_bbox_project_id: int | None = None
    kubeflow_url: str | None = None

    def __post_init__(self) -> None:
        for name, value in (("Label Studio", self.label_studio_url), ("Kubeflow", self.kubeflow_url)):
            if value is None:
                continue
            parsed = urlsplit(value)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(f"{name} URL must be HTTP(S) without embedded credentials or query data")
        if self.label_studio_bbox_project_id is not None and self.label_studio_bbox_project_id <= 0:
            raise ValueError("Label Studio bbox project ID must be positive")

    @classmethod
    def from_environment(cls) -> OperatorUIIntegrations:
        project_value = os.environ.get("GODS_MLOPS_LABEL_STUDIO_BBOX_PROJECT_ID", "").strip()
        try:
            project_id = int(project_value) if project_value else None
        except ValueError as error:
            raise ValueError("GODS_MLOPS_LABEL_STUDIO_BBOX_PROJECT_ID must be a positive integer") from error
        return cls(
            label_studio_url=_optional_environment("GODS_MLOPS_LABEL_STUDIO_URL"),
            label_studio_bbox_project_id=project_id,
            kubeflow_url=_optional_environment("GODS_MLOPS_KUBEFLOW_URL"),
        )


@dataclass(slots=True)
class OperatorServices:
    ingestion: Any
    annotations: Any
    review_workflow: Any | None
    publisher: Any
    queue: Any
    result_store: Any | None
    integrations: OperatorUIIntegrations = field(default_factory=OperatorUIIntegrations)
    startup: Callable[[], Awaitable[None]] | None = field(default=None, repr=False)
    shutdown: Callable[[], Awaitable[None]] | None = field(default=None, repr=False)

    async def start(self) -> None:
        if self.startup is not None:
            await self.startup()

    async def close(self) -> None:
        if self.shutdown is not None:
            await self.shutdown()


def create_app(
    services: OperatorServices,
    *,
    auth_settings: OperatorAuthSettings,
    integrations: OperatorUIIntegrations | None = None,
) -> FastAPI:
    """Build a testable SSR app; its owner controls service startup and shutdown."""
    selected_integrations = integrations or services.integrations
    auth = OperatorAuth(auth_settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await services.start()
        try:
            yield
        finally:
            await services.close()

    app = FastAPI(
        title="Gods MLOps operator",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.operator_services = services
    app.state.operator_integrations = selected_integrations
    app.state.operator_auth = auth
    app.include_router(auth.router)
    app.include_router(build_operator_router(auth=auth))
    app.mount("/static", StaticFiles(directory=_WEB_ROOT / "static"), name="static")
    return app


def build_default_operator_services(
    integrations: OperatorUIIntegrations,
) -> OperatorServices:
    """Build local services from protected runtime environment values."""
    database_url = _required_environment("GODS_MLOPS_DATABASE_URL")
    endpoint_url = _required_environment("GODS_MLOPS_S3_ENDPOINT_URL")
    access_key = _required_environment("GODS_MLOPS_S3_ACCESS_KEY")
    secret_key = _required_environment("GODS_MLOPS_S3_SECRET_KEY")
    bucket = _required_environment("GODS_MLOPS_S3_BUCKET")
    region = os.environ.get("GODS_MLOPS_S3_REGION", "us-east-1")

    sample_repository = PostgresIngestionRepository(database_url=database_url)
    sample_store = S3SampleStore(
        endpoint_url=endpoint_url,
        access_key=access_key,
        secret_key=secret_key,
        bucket=bucket,
        region=region,
    )
    ingestion = IngestionService(
        repository=sample_repository,
        objects=sample_store,
        collection_gate=None,
    )

    annotation_repository = PostgresAnnotationRepository(database_url=database_url)
    label_studio_url = integrations.label_studio_url
    label_studio_token = _optional_environment("GODS_MLOPS_LABEL_STUDIO_API_TOKEN")
    label_media_cleanup_url = _optional_environment("GODS_MLOPS_LABEL_MEDIA_CLEANUP_URL")
    label_media_cleanup_token = _optional_environment("GODS_MLOPS_LABEL_MEDIA_CLEANUP_TOKEN")
    label_studio = (
        LabelStudioApiClient(base_url=label_studio_url, api_token=label_studio_token)
        if label_studio_url and label_studio_token
        else None
    )
    annotations = AnnotationService(repository=annotation_repository, label_studio=label_studio)
    review_workflow = None
    if (
        label_studio is not None
        and integrations.label_studio_bbox_project_id is not None
        and label_media_cleanup_url
        and label_media_cleanup_token
    ):
        review_workflow = LabelStudioReviewWorkflow(
            repository=annotation_repository,
            objects=sample_store,
            label_studio=label_studio,
            media_cleanup=LabelStudioMediaCleanupClient(
                base_url=label_media_cleanup_url,
                token=label_media_cleanup_token,
            ),
        )

    dataset_objects = DatasetObjectStore(
        endpoint_url=endpoint_url,
        access_key=access_key,
        secret_key=secret_key,
        bucket=bucket,
        region=region,
    )
    publisher = DatasetPublisher(database_url=database_url, objects=dataset_objects)
    queue_repository = PostgresJobQueueRepository(database_url=database_url)
    source_registry = DatasetSourceRegistry(database_url=database_url)
    queue = JobQueue(repository=queue_repository, sources=source_registry)
    result_store = S3ResultArtifactStore(objects=dataset_objects, bucket=bucket)

    async def startup() -> None:
        await sample_repository.ensure_schema()
        await annotation_repository.ensure_schema()
        await publisher.ensure_schema()
        await queue_repository.ensure_schema()

    async def shutdown() -> None:
        await queue_repository.close()
        await source_registry.close()
        await publisher.close()
        await annotation_repository.close()
        await sample_repository.close()

    return OperatorServices(
        ingestion=ingestion,
        annotations=annotations,
        review_workflow=review_workflow,
        publisher=publisher,
        queue=queue,
        result_store=result_store,
        integrations=integrations,
        startup=startup,
        shutdown=shutdown,
    )


def main() -> None:
    import uvicorn

    auth_settings = OperatorAuthSettings.from_environment()
    integrations = OperatorUIIntegrations.from_environment()
    services = build_default_operator_services(integrations)
    port = _ui_port()
    uvicorn.run(
        create_app(services, auth_settings=auth_settings, integrations=integrations),
        host="127.0.0.1",
        port=port,
        access_log=False,
    )


def _ui_port() -> int:
    value = os.environ.get("GODS_MLOPS_OPERATOR_UI_PORT", "8091")
    try:
        port = int(value)
    except ValueError as error:
        raise ValueError("GODS_MLOPS_OPERATOR_UI_PORT must be an integer") from error
    if not 1 <= port <= 65535:
        raise ValueError("GODS_MLOPS_OPERATOR_UI_PORT must be between 1 and 65535")
    return port


def _optional_environment(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required to start the local MLOps operator UI")
    return value
