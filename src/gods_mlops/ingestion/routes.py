"""Authenticated FastAPI routes for durable candidate receipt."""

from __future__ import annotations

import secrets
from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile, status
from pydantic import ValidationError

from .collection_gate import CollectionPausedError
from .schemas import (
    CandidateMetadata,
    CandidateReason,
    DailySampleLimitError,
    GlobalObjectLimitError,
    SampleConflictError,
    SampleExpiredError,
    SampleReceipt,
    SampleStorageError,
)
from .service import IngestionService

_MAX_SAMPLE_BYTES = 20 * 1024 * 1024


def build_ingestion_router(*, service: IngestionService, bearer_token: str) -> APIRouter:
    """Build the receiver router with mandatory token authentication."""
    if len(bearer_token) < 32:
        raise ValueError("ingestion bearer token must contain at least 32 characters")

    def require_token(authorization: Annotated[str | None, Header()] = None) -> None:
        expected = f"Bearer {bearer_token}"
        if authorization is None or not secrets.compare_digest(authorization, expected):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "invalid_ingestion_token"},
                headers={"WWW-Authenticate": "Bearer"},
            )

    router = APIRouter(prefix="/api/samples", tags=["samples"], dependencies=[Depends(require_token)])

    @router.post("", response_model=SampleReceipt, status_code=status.HTTP_201_CREATED)
    async def receive_sample(
        sample_id: Annotated[UUID, Form()],
        camera_id: Annotated[UUID, Form()],
        captured_at_utc: Annotated[datetime, Form()],
        reason: Annotated[CandidateReason, Form()],
        sha256: Annotated[str, Form()],
        model_revision: Annotated[str, Form()],
        processor_revision: Annotated[str, Form()],
        image: Annotated[UploadFile, File()],
    ) -> SampleReceipt:
        if image.content_type not in {"image/jpeg", "application/octet-stream"}:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail={"code": "sample_must_be_jpeg"},
            )
        image_bytes = await image.read(_MAX_SAMPLE_BYTES + 1)
        if len(image_bytes) > _MAX_SAMPLE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail={"code": "sample_too_large"},
            )
        try:
            metadata = CandidateMetadata(
                sample_id=sample_id,
                camera_id=camera_id,
                captured_at_utc=captured_at_utc,
                reason=reason,
                sha256=sha256,
                model_revision=model_revision,
                processor_revision=processor_revision,
            )
        except ValidationError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"code": "invalid_candidate_metadata"},
            ) from error
        try:
            return await service.receive(metadata, image_bytes)
        except CollectionPausedError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=error.as_detail(),
                headers={"Retry-After": "5"},
            ) from error
        except SampleConflictError as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"code": "sample_id_conflict"},
            ) from error
        except DailySampleLimitError as error:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={"code": "camera_daily_sample_limit"},
            ) from error
        except GlobalObjectLimitError as error:
            raise HTTPException(
                status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
                detail={"code": "global_object_limit"},
            ) from error
        except SampleExpiredError as error:
            raise HTTPException(
                status_code=status.HTTP_410_GONE,
                detail={"code": "sample_expired"},
            ) from error
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"code": "invalid_sample"},
            ) from error
        except SampleStorageError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"code": "sample_persistence_pending"},
            ) from error
        finally:
            await image.close()

    return router


__all__ = ["build_ingestion_router"]
