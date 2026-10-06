"""Authenticated server-rendered operator pages and state-changing actions."""

from __future__ import annotations

from html import escape
import json
from pathlib import Path
import secrets
from string import Template
from typing import Annotated, Any
from urllib.parse import quote

from fastapi import APIRouter, Depends, Form, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse, RedirectResponse

from gods_mlops.datasets.deletion import invalidate_sample, preview_invalidate_sample
from gods_mlops.evaluation.report import load_operator_evaluation_report

from .auth import OperatorAuth

_BASE_TEMPLATE = Path(__file__).with_name("templates") / "base.html"
_RETRY_INTENT_COOKIE = "gods_mlops_operator_retry_intent"
_WAITING_STATES = {"queued", "waiting_profile", "waiting_gpu", "waiting_storage", "waiting_capacity"}
_TERMINAL_STATES = {"completed", "failed", "cancelled"}


def build_operator_router(*, auth: OperatorAuth) -> APIRouter:
    router = APIRouter()

    @router.get("/samples", response_class=HTMLResponse)
    async def samples(
        request: Request,
        state: str | None = None,
        operator_id: str = Depends(auth.require_operator),
    ) -> HTMLResponse:
        try:
            items = await request.app.state.operator_services.ingestion.list_samples(limit=100, state=state)
        except Exception as error:  # noqa: BLE001 - page renders a bounded service error
            return _error_page(request, "Samples", error, status_code=503)
        integrations = request.app.state.operator_integrations
        workflow = request.app.state.operator_services.review_workflow
        rows = []
        for item in items:
            sample_id = _e(item["sample_id"])
            status_label = _e(item.get("state", "unknown"))
            actions = []
            if item.get("state") == "received" and not item.get("selected"):
                if workflow is not None and integrations.label_studio_bbox_project_id is not None:
                    actions.append(
                        _form(
                            f"/reviews/start",
                            f'<input type="hidden" name="sample_id" value="{sample_id}">'
                            '<button type="submit">Send to bbox review</button>',
                        )
                    )
                else:
                    actions.append('<span class="muted">Label Studio review setup is not configured.</span>')
            if item.get("state") == "received":
                actions.append(f'<a href="/samples/{sample_id}/invalidation-preview">Review impact</a>')
            rows.append(
                "<tr>"
                f"<td><code>{sample_id}</code></td>"
                f"<td>{_e(item.get('camera_id'))}</td>"
                f"<td>{_e(item.get('captured_at_utc'))}</td>"
                f"<td>{_e(item.get('reason'))}</td>"
                f"<td>{status_label}</td>"
                f"<td>{_e(item.get('object_size_bytes'))}</td>"
                f"<td>{_e(item.get('last_failure_code') or '—')}</td>"
                f"<td>{' '.join(actions)}</td>"
                "</tr>"
            )
        state_filter = (
            '<form method="get" action="/samples">'
            '<label>State <select name="state"><option value="">All</option>'
            + "".join(
                f'<option value="{name}"{" selected" if state == name else ""}>{name}</option>'
                for name in ("received", "pending", "storage_limited", "purge_pending", "expired")
            )
            + '</select></label><button type="submit">Filter</button></form>'
        )
        body = (
            '<section><h2>Candidate collection status</h2>'
            '<p>Rows show stored candidates and their retention state. Starting a bbox review creates the existing sample assignment.</p>'
            f"{state_filter}<table><thead><tr><th>Sample</th><th>Camera</th><th>Captured</th><th>Reason</th>"
            f"<th>Status</th><th>Bytes</th><th>Last failure</th><th>Actions</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
            "</section>"
        )
        return _page(request, "Samples", body)

    @router.post("/reviews/start", response_class=HTMLResponse)
    async def start_review(
        request: Request,
        sample_id: Annotated[str, Form()],
        operator_id: str = Depends(auth.require_operator),
        _csrf: None = Depends(auth.require_csrf),
    ) -> Response:
        services = request.app.state.operator_services
        project_id = request.app.state.operator_integrations.label_studio_bbox_project_id
        if services.review_workflow is None or project_id is None:
            return _error_page(request, "Reviews", RuntimeError("Label Studio review setup is not configured"), 409)
        try:
            await services.annotations.start_bbox_review(
                sample_id,
                project_id=project_id,
                workflow=services.review_workflow,
            )
        except Exception as error:  # noqa: BLE001 - assignment authority owns the state transition
            return _error_page(request, "Reviews", error, status_code=409)
        return _redirect("/reviews")

    @router.get("/reviews", response_class=HTMLResponse)
    async def reviews(
        request: Request,
        operator_id: str = Depends(auth.require_operator),
    ) -> HTMLResponse:
        try:
            items = await request.app.state.operator_services.annotations.list_reviews(limit=100)
        except Exception as error:  # noqa: BLE001 - keep database details out of HTML
            return _error_page(request, "Reviews", error, status_code=503)
        base_url = request.app.state.operator_integrations.label_studio_url
        rows = []
        for item in items:
            task_id = item.get("label_studio_task_id")
            project_id = item.get("project_id") or request.app.state.operator_integrations.label_studio_bbox_project_id
            if base_url and project_id:
                project_url = base_url.rstrip("/") + f"/projects/{project_id}/data"
                project_link = f'<a href="{_e(project_url)}" rel="noreferrer">Open Label Studio project</a>'
            else:
                project_link = '<span class="muted">Label Studio integration is not configured.</span>'
            human = "Label Studio annotation recorded" if item.get("human_review_recorded") else "No submitted human annotation recorded"
            finalize = ""
            if item.get("state") == "active" and getattr(
                request.app.state.operator_services.annotations,
                "label_studio_configured",
                False,
            ):
                finalize = _form(
                    f"/reviews/{_e(item['sample_id'])}/{_e(item['revision'])}/finalize",
                    '<button type="submit">Finalize submitted annotation</button>',
                )
            rows.append(
                "<tr>"
                f"<td><code>{_e(item.get('sample_id'))}</code></td>"
                f"<td>{_e(item.get('stage'))}</td><td>{_e(item.get('state'))}</td>"
                f"<td>{_e(task_id if task_id is not None else 'Not provisioned')}</td>"
                f"<td>{_e(human)}</td><td>{project_link} {finalize}</td></tr>"
            )
        label_api_ready = getattr(
            request.app.state.operator_services.annotations,
            "label_studio_configured",
            False,
        )
        if base_url and not label_api_ready:
            integration_note = (
                '<p class="waiting">The Label Studio project link is configured, but review synchronization '
                "credentials are not configured for this UI process.</p>"
            )
        elif not base_url:
            integration_note = '<p class="muted">Label Studio integration is not configured.</p>'
        else:
            integration_note = ""
        return _page(
            request,
            "Reviews",
            '<section><h2>Review assignments</h2>'
            '<p>Submitted annotation provenance is shown separately from execution and deployment eligibility.</p>'
            f"{integration_note}"
            f"<table><thead><tr><th>Sample</th><th>Stage</th><th>Status</th><th>Label Studio task</th><th>Human review</th><th>Actions</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
            "</section>",
        )

    @router.post("/reviews/{sample_id}/{revision}/finalize", response_class=HTMLResponse)
    async def finalize_review(
        request: Request,
        sample_id: str,
        revision: str,
        operator_id: str = Depends(auth.require_operator),
        _csrf: None = Depends(auth.require_csrf),
    ) -> Response:
        if not getattr(request.app.state.operator_services.annotations, "label_studio_configured", False):
            return _error_page(request, "Reviews", RuntimeError("Label Studio integration is not configured"), 409)
        try:
            await request.app.state.operator_services.annotations.finalize_annotation(sample_id, revision)
        except Exception as error:  # noqa: BLE001 - annotation service owns finalization authority
            return _error_page(request, "Reviews", error, status_code=409)
        return _redirect("/reviews")

    @router.get("/datasets", response_class=HTMLResponse)
    async def datasets(
        request: Request,
        operator_id: str = Depends(auth.require_operator),
    ) -> HTMLResponse:
        try:
            publisher = request.app.state.operator_services.publisher
            publications = await publisher.list_publications(limit=100)
            candidates = await publisher.list_publication_candidates(limit=100)
        except Exception as error:  # noqa: BLE001 - do not expose database/S3 configuration
            return _error_page(request, "Datasets", error, status_code=503)
        publication_rows = []
        for item in publications:
            publication_rows.append(
                "<tr>"
                f"<td><code>{_e(item.get('dataset_version'))}</code></td><td>{_e(item.get('state'))}</td>"
                f"<td>{_eligibility_label(item.get('training_ready'), item.get('training_reasons'))}</td>"
                f"<td>{_eligibility_label(item.get('evaluation_eligible'), item.get('evaluation_reasons'))}</td></tr>"
            )
        sample_options = []
        for item in candidates["samples"]:
            reviewed = "bbox review finalized" if item.get("bbox_reviewed") else "bbox review required"
            sample_options.append(
                f'<label><input type="checkbox" name="sample_ids" value="{_e(item["sample_id"])}">'
                f'{_e(item["sample_id"])} · {_e(reviewed)}</label>'
            )
        crop_options = []
        for item in candidates["crops"]:
            state = item.get("caption_state")
            crop_options.append(
                f'<label><input type="checkbox" name="crop_ids" value="{_e(item["crop_id"])}">'
                f'{_e(item["crop_id"])} · caption {_e(state)}</label>'
            )
        body = (
            '<section><h2>Immutable dataset publications</h2>'
            f"<table><thead><tr><th>Version</th><th>Status</th><th>Training</th><th>Evaluation</th></tr></thead><tbody>{''.join(publication_rows)}</tbody></table></section>"
            '<section><h2>Publish reviewed candidates</h2>'
            '<p>The publisher rechecks revisions, split readiness, and human truth before committing immutable objects.</p>'
            '<form method="post" action="/datasets/publish" data-csrf-form>'
            '<label>Target <select name="target" required><option value="detr">DETR</option><option value="clip">CLIP</option><option value="both">Both</option></select></label>'
            '<label>Dataset config version <input name="config_version" required maxlength="255"></label>'
            f"<fieldset><legend>Frames</legend>{''.join(sample_options) or '<span>No received frames.</span>'}</fieldset>"
            f"<fieldset><legend>Crops</legend>{''.join(crop_options) or '<span>No stored crops.</span>'}</fieldset>"
            '<button type="submit">Publish selected dataset</button></form></section>'
            '<section><h2>Queue model training</h2><p>Training submission uses the current immutable dataset and measured resource profile checks.</p>'
            '<form method="post" action="/jobs/queue" data-csrf-form>'
            '<label>Published dataset version <input name="dataset_version" required></label>'
            '<label>Model <select name="model_kind"><option value="detr">DETR</option><option value="clip">CLIP</option></select></label>'
            '<label>Measured config version <input name="config_version" required></label>'
            '<button type="submit">Queue training</button></form></section>'
        )
        return _page(request, "Datasets", body)

    @router.post("/datasets/publish", response_class=HTMLResponse)
    async def publish_dataset(
        request: Request,
        target: Annotated[str, Form()],
        config_version: Annotated[str, Form()],
        sample_ids: Annotated[list[str] | None, Form()] = None,
        crop_ids: Annotated[list[str] | None, Form()] = None,
        operator_id: str = Depends(auth.require_operator),
        _csrf: None = Depends(auth.require_csrf),
    ) -> Response:
        try:
            publisher = request.app.state.operator_services.publisher
            selection = await publisher.publication_selection(
                target=target,
                sample_ids=sample_ids or [],
                crop_ids=crop_ids or [],
            )
            await publisher.publish_dataset(selection, config_version)
        except Exception as error:  # noqa: BLE001 - publisher owns eligibility and storage authority
            return _error_page(request, "Datasets", error, status_code=409)
        return _redirect("/datasets")

    @router.get("/jobs", response_class=HTMLResponse)
    async def jobs(
        request: Request,
        operator_id: str = Depends(auth.require_operator),
    ) -> HTMLResponse:
        try:
            items = await request.app.state.operator_services.queue.list_jobs(limit=100)
        except Exception as error:  # noqa: BLE001 - bound database error details
            return _error_page(request, "Jobs", error, status_code=503)
        body = _jobs_table(items)
        response = _page(request, "Jobs", body)
        _set_retry_intent_cookie(response, auth, secrets.token_urlsafe(32))
        return response

    @router.get("/jobs/{job_id}", response_class=HTMLResponse)
    async def job_detail(
        request: Request,
        job_id: str,
        operator_id: str = Depends(auth.require_operator),
    ) -> HTMLResponse:
        services = request.app.state.operator_services
        try:
            job = await services.queue.get_job_detail(job_id)
        except KeyError as error:
            return _error_page(request, "Job", error, status_code=404)
        except Exception as error:  # noqa: BLE001 - do not leak connection details
            return _error_page(request, "Job", error, status_code=503)
        report_result = None
        if job.get("phase") == "evaluation":
            if services.result_store is None:
                report_result = {
                    "availability": "not_available",
                    "reason_code": "evaluation_result_store_not_configured",
                    "report": None,
                    "deployment_eligibility": None,
                }
            else:
                try:
                    report_result = await load_operator_evaluation_report(
                        job_id,
                        queue=services.queue,
                        result_store=services.result_store,
                    )
                except Exception as error:  # noqa: BLE001 - report absence remains explicit
                    report_result = {
                        "availability": "not_available",
                        "reason_code": _safe_failure(error),
                        "report": None,
                        "deployment_eligibility": None,
                    }
        body = _job_detail_body(job, report_result, request.app.state.operator_integrations)
        response = _page(request, "Job detail", body)
        _set_retry_intent_cookie(response, auth, secrets.token_urlsafe(32))
        return response

    @router.post("/jobs/queue", response_class=HTMLResponse)
    async def queue_training(
        request: Request,
        dataset_version: Annotated[str, Form()],
        model_kind: Annotated[str, Form()],
        config_version: Annotated[str, Form()],
        operator_id: str = Depends(auth.require_operator),
        _csrf: None = Depends(auth.require_csrf),
    ) -> Response:
        try:
            job_id = await request.app.state.operator_services.queue.submit(
                dataset_version,
                model_kind,
                config_version,
            )
        except Exception as error:  # noqa: BLE001 - queue service owns source/profile validation
            return _error_page(request, "Jobs", error, status_code=409)
        return _redirect(f"/jobs/{quote(job_id, safe='')}")

    @router.post("/jobs/{job_id}/retry", response_class=HTMLResponse)
    async def retry_job(
        request: Request,
        job_id: str,
        operator_id: str = Depends(auth.require_operator),
        _csrf: None = Depends(auth.require_csrf),
    ) -> Response:
        intent_token = request.cookies.get(_RETRY_INTENT_COOKIE)
        if intent_token is None:
            raise HTTPException(status_code=403, detail="Retry intent cookie is required")
        try:
            new_job_id = await request.app.state.operator_services.queue.retry_job(
                job_id,
                operator_id=operator_id,
                intent_token=intent_token,
            )
        except Exception as error:  # noqa: BLE001 - queue service revalidates immutable source eligibility
            return _error_page(request, "Jobs", error, status_code=409)
        return _redirect(f"/jobs/{quote(new_job_id, safe='')}")

    @router.post("/jobs/{job_id}/cancel", response_class=HTMLResponse)
    async def cancel_job(
        request: Request,
        job_id: str,
        operator_id: str = Depends(auth.require_operator),
        _csrf: None = Depends(auth.require_csrf),
    ) -> Response:
        try:
            result = await request.app.state.operator_services.queue.cancel_queued(
                job_id,
                operator_id=operator_id,
            )
        except Exception as error:  # noqa: BLE001 - service enforces waiting-only cancellation
            return _error_page(request, "Jobs", error, status_code=409)
        if result.get("cleanup_pending"):
            return _page(
                request,
                "Jobs",
                f'<p class="waiting">Job {_e(job_id)} is cancelled. Artifact quota cleanup is pending; review this action.</p>',
                status_code=202,
            )
        return _redirect("/jobs")

    @router.post("/jobs/reorder", response_class=HTMLResponse)
    async def reorder_job(
        request: Request,
        job_id: Annotated[str, Form()],
        before_job_id: Annotated[str | None, Form()] = None,
        operator_id: str = Depends(auth.require_operator),
        _csrf: None = Depends(auth.require_csrf),
    ) -> Response:
        try:
            await request.app.state.operator_services.queue.reorder_queued(
                job_id,
                before_job_id=before_job_id or None,
                operator_id=operator_id,
            )
        except Exception as error:  # noqa: BLE001 - service enforces queue ownership fences
            return _error_page(request, "Jobs", error, status_code=409)
        return _redirect("/jobs")

    @router.get("/actions", response_class=HTMLResponse)
    async def actions(
        request: Request,
        operator_id: str = Depends(auth.require_operator),
    ) -> HTMLResponse:
        try:
            items = await request.app.state.operator_services.queue.list_jobs(limit=100)
        except Exception as error:  # noqa: BLE001 - do not leak backend connection strings
            return _error_page(request, "Actions", error, status_code=503)
        needing_action = [
            item
            for item in items
            if item.get("state") == "failed"
            or item.get("state") in _WAITING_STATES
            or item.get("reason_code")
        ]
        return _page(request, "Actions", _jobs_table(needing_action, empty="No jobs currently need operator action."))

    @router.get("/samples/{sample_id}/invalidation-preview", response_class=HTMLResponse)
    async def invalidation_preview(
        request: Request,
        sample_id: str,
        operator_id: str = Depends(auth.require_operator),
    ) -> HTMLResponse:
        try:
            impact = await preview_invalidate_sample(
                sample_id,
                publisher=request.app.state.operator_services.publisher,
            )
        except Exception as error:  # noqa: BLE001 - preview remains read-only
            return _error_page(request, "Sample impact", error, status_code=409)
        if not impact.get("sample_exists"):
            return _error_page(request, "Sample impact", KeyError("sample does not exist"), status_code=404)
        body = (
            '<section><h2>Invalidate sample eligibility</h2>'
            '<p>This action records an invalidation and blocks affected training/evaluation. It does not physically delete source bytes or published history.</p>'
            f"<p>Physical deletion: {'yes' if impact.get('physical_deletion') else 'no'}</p>"
            f"<p>Sample state: {_e(impact.get('sample_state'))}</p>"
            f"<p>Retained for review: {'yes' if impact.get('retained_for_review') else 'no'}</p>"
            f"<p>Active reviews: {_e(impact.get('active_review_count'))}</p>"
            f"<p>Active jobs: {_e(impact.get('active_job_count'))}</p>"
            f"<p>Affected datasets: {_e(', '.join(impact.get('datasets', [])) or 'none')}</p>"
            f"<p>Affected models: {_e(', '.join(impact.get('models', [])) or 'none')}</p>"
            f"{_form(f'/samples/{_e(sample_id)}/invalidate', '<button type="submit">Invalidate sample eligibility</button>')}"
            "</section>"
        )
        return _page(request, "Sample impact", body)

    @router.post("/samples/{sample_id}/invalidate", response_class=HTMLResponse)
    async def invalidate_sample_route(
        request: Request,
        sample_id: str,
        operator_id: str = Depends(auth.require_operator),
        _csrf: None = Depends(auth.require_csrf),
    ) -> Response:
        try:
            await invalidate_sample(sample_id, publisher=request.app.state.operator_services.publisher)
        except Exception as error:  # noqa: BLE001 - invalidation authority rechecks exact current impact
            return _error_page(request, "Sample invalidation", error, status_code=409)
        return _redirect("/datasets")

    return router


def _jobs_table(items: list[dict[str, Any]], *, empty: str = "No jobs recorded.") -> str:
    if not items:
        return f'<section><p class="muted">{_e(empty)}</p></section>'
    rows = []
    active = []
    for item in items:
        if item.get("state") in _WAITING_STATES:
            active.append(item)
        job_id = _e(item.get("job_id"))
        state = item.get("state")
        label, css = _status_label(item)
        checkpoint_sha = item.get("checkpoint_sha256")
        checkpoint_text = f"Checkpoint {_e(checkpoint_sha)}" if checkpoint_sha else "No committed checkpoint"
        failure = _reason_text(item)
        actions = [f'<a href="/jobs/{job_id}">Details</a>']
        if state in _WAITING_STATES:
            actions.append(_form(f"/jobs/{job_id}/cancel", '<button type="submit">Cancel</button>'))
        if state in _TERMINAL_STATES and item.get("phase") != "probe":
            actions.append(_form(f"/jobs/{job_id}/retry", '<button type="submit">Rerun</button>'))
        rows.append(
            "<tr>"
            f"<td><code>{job_id}</code></td><td>{_e(item.get('phase'))}</td>"
            f"<td>{_e(item.get('queue_position') if item.get('queue_position') is not None else '—')}</td>"
            f'<td class="{css}">{_e(label)}</td><td>{_e(item.get("reason_code") or "—")}</td>'
            f"<td>{_e(failure)}</td><td>{checkpoint_text}</td><td>{' '.join(actions)}</td></tr>"
        )
    reorder_form = ""
    if active:
        options = ''.join(
            f'<option value="{_e(item["job_id"])}">{_e(item["job_id"])} · {_e(item.get("state"))}</option>'
            for item in active
        )
        reorder_form = (
            '<section><h2>Reorder waiting jobs</h2><form method="post" action="/jobs/reorder" data-csrf-form>'
            f'<label>Move job <select name="job_id">{options}</select></label>'
            f'<label>Before job <select name="before_job_id"><option value="">End of queue</option>{options}</select></label>'
            '<button type="submit">Save queue order</button></form></section>'
        )
    return (
        '<section><h2>Execution status</h2>'
        '<p>Resource waiting, data insufficiency, execution failure, pipeline success, human review, and deployment eligibility are separate states.</p>'
        f"<table><thead><tr><th>Job ID</th><th>Phase</th><th>Queue position</th><th>Status</th><th>Cause code</th><th>Details</th><th>Checkpoint</th><th>Actions</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>"
        + reorder_form
    )


def _job_detail_body(job: dict[str, Any], report_result: dict[str, Any] | None, integrations) -> str:
    job_id = _e(job.get("job_id"))
    reason_detail = _e(json.dumps(job.get("reason_detail") or {}, sort_keys=True, ensure_ascii=False))
    checkpoint = job.get("checkpoint")
    checkpoint_sha = checkpoint.get("sha256") if isinstance(checkpoint, dict) else job.get("checkpoint_sha256")
    checkpoint_text = f"Latest committed checkpoint SHA-256: {_e(checkpoint_sha)}" if checkpoint_sha else "No committed checkpoint is available."
    if integrations.kubeflow_url:
        kubeflow_url = _e(integrations.kubeflow_url.rstrip("/") + f"/pipeline/#/runs/details/{quote(str(job.get('job_id')), safe='')}")
        kubeflow_link = f'<a href="{kubeflow_url}" rel="noreferrer">Open Kubeflow run details, logs, and artifacts</a>'
    else:
        kubeflow_link = '<span class="muted">Kubeflow run, log, and artifact links are not configured.</span>'
    report_html = ""
    if report_result is not None:
        if report_result.get("availability") != "available":
            report_html = (
                '<section><h2>Evaluation report</h2><p class="waiting">Report not available: '
                f"{_e(report_result.get('reason_code') or 'unknown')}</p></section>"
            )
        else:
            report = report_result["report"]
            metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
            gate = report_result.get("deployment_eligibility") or {}
            reasons = gate.get("reasons", [])
            gate_status = "Deployment eligible" if gate.get("eligible") else "Deployment blocked"
            metric_rows = "".join(
                f"<tr><th>{_e(name)}</th><td>{_e(value)}</td></tr>" for name, value in sorted(metrics.items())
            )
            report_html = (
                '<section><h2>Committed evaluation report</h2>'
                f"<p>Execution: {_e(report.get('execution_status'))} · Metrics: {_e(report.get('status'))}</p>"
                f"<table><tbody>{metric_rows or '<tr><td>No metric values in report.</td></tr>'}</tbody></table>"
                f'<p class="{"success" if gate.get("eligible") else "blocked"}">{_e(gate_status)}</p>'
                f"<p>Release gate reasons: {_e(', '.join(reasons) or 'none')}</p></section>"
            )
    actions = []
    if job.get("state") in _WAITING_STATES:
        actions.append(_form(f"/jobs/{job_id}/cancel", '<button type="submit">Cancel waiting job</button>'))
    if job.get("state") in _TERMINAL_STATES and job.get("phase") != "probe":
        actions.append(_form(f"/jobs/{job_id}/retry", '<button type="submit">Rerun job</button>'))
    return (
        '<section><h2>Job state</h2>'
        f"<p>Job ID: <code>{job_id}</code></p><p>Phase: {_e(job.get('phase'))}</p>"
        f"<p>Status: {_e(_status_label(job)[0])}</p><p>Cause code: {_e(job.get('reason_code') or '—')}</p>"
        f"<p>Cause details: <code>{reason_detail}</code></p><p>{checkpoint_text}</p>"
        f"<p>{kubeflow_link}</p><p>{' '.join(actions)}</p></section>{report_html}"
    )


def _status_label(job: dict[str, Any]) -> tuple[str, str]:
    state = job.get("state")
    reason = str(job.get("reason_code") or "")
    if state in {"waiting_gpu", "waiting_storage", "waiting_capacity", "waiting_profile"}:
        if "dataset" in reason or "source" in reason or "insufficient" in reason:
            return "Data insufficient", "blocked"
        return "Resource waiting", "waiting"
    if state == "failed":
        return "Execution failure", "blocked"
    if state == "completed":
        return "Pipeline success", "success"
    if state == "running":
        return "Running", "waiting"
    if state == "cancelled":
        return "Cancelled", "muted"
    return str(state or "Unknown"), "muted"


def _reason_text(job: dict[str, Any]) -> str:
    detail = job.get("reason_detail")
    if isinstance(detail, dict):
        return "; ".join(f"{key}={value}" for key, value in sorted(detail.items())) or "—"
    return str(detail) if detail else "—"


def _eligibility_label(eligible: Any, reasons: Any) -> str:
    if eligible is True:
        return "Eligible"
    if eligible is False:
        reason_list = reasons if isinstance(reasons, list) else []
        return "Blocked: " + (", ".join(map(str, reason_list)) or "eligibility incomplete")
    return "Not measured"


def _form(action: str, controls: str) -> str:
    return f'<form method="post" action="{_e(action)}" data-csrf-form>{controls}</form>'


def _page(request: Request, title: str, content: str, *, status_code: int = 200) -> HTMLResponse:
    template = Template(_BASE_TEMPLATE.read_text(encoding="utf-8"))
    body = template.safe_substitute(title=_e(title), content=content)
    return HTMLResponse(body, status_code=status_code, headers={"Cache-Control": "no-store"})


def _error_page(request: Request, title: str, error: Exception, *, status_code: int) -> HTMLResponse:
    message = _safe_failure(error)
    return _page(request, title, f'<section><h2>Action needs attention</h2><p class="error">{_e(message)}</p></section>', status_code=status_code)


def _safe_failure(error: Exception) -> str:
    reasons = getattr(error, "reasons", None)
    if isinstance(reasons, (list, tuple)) and reasons:
        return ", ".join(map(str, reasons))
    if type(error).__module__.startswith("gods_mlops."):
        return str(error)
    return type(error).__name__


def _redirect(location: str) -> RedirectResponse:
    return RedirectResponse(location, status_code=status.HTTP_303_SEE_OTHER, headers={"Cache-Control": "no-store"})


def _set_retry_intent_cookie(response, auth: OperatorAuth, token: str) -> None:
    response.set_cookie(
        _RETRY_INTENT_COOKIE,
        token,
        max_age=auth.settings.session_seconds,
        path="/jobs",
        secure=auth.settings.secure_cookie,
        httponly=True,
        samesite="strict",
    )


def _e(value: Any) -> str:
    return escape(str(value) if value is not None else "", quote=True)
