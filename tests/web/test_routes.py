from __future__ import annotations

import http.cookiejar
import json
import socket
import threading
import time
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request
from uuid import uuid4

import uvicorn

from gods_mlops.evaluation.report import load_operator_evaluation_report
from gods_mlops.jobs.queue import _operator_retry_dedupe_key
from gods_mlops.web.auth import AUTH_COOKIE, CSRF_COOKIE
from gods_mlops.web.app import OperatorServices, OperatorUIIntegrations, create_app
from gods_mlops.web.auth import OperatorAuthSettings


PASSWORD = "test-only-operator-password-42"
SECRET = "test-only-session-secret-with-more-than-32-bytes"
RETRY_COOKIE = "gods_mlops_operator_retry_intent"


class _NoRedirect(urllib_request.HTTPRedirectHandler):
    def redirect_request(self, request, file, code, message, headers, new_url):
        return None


class _Response:
    def __init__(self, response):
        self._response = response
        self.status_code = response.status if hasattr(response, "status") else response.code
        self.headers = response.headers

    def read(self):
        return self._response.read()

    def geturl(self):
        return self._response.geturl()


class _Browser:
    def __init__(self, app):
        self.cookies = http.cookiejar.CookieJar()
        self.listen_socket = socket.socket()
        self.listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listen_socket.bind(("127.0.0.1", 0))
        self.listen_socket.listen(128)
        self.listen_socket.setblocking(False)
        self.port = self.listen_socket.getsockname()[1]
        self.server = uvicorn.Server(
            uvicorn.Config(app, log_level="critical", access_log=False, lifespan="on")
        )
        self.opener = urllib_request.build_opener(
            urllib_request.HTTPCookieProcessor(self.cookies),
            _NoRedirect(),
        )
        self.thread = threading.Thread(
            target=self.server.run,
            kwargs={"sockets": [self.listen_socket]},
            daemon=True,
        )
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.server.started and time.monotonic() < deadline:
            time.sleep(0.025)
        assert self.server.started

    def close(self):
        self.server.should_exit = True
        self.thread.join(timeout=10)
        self.listen_socket.close()

    def request(self, path, *, method="GET", data=None, headers=None):
        payload = urllib_parse.urlencode(data).encode("utf-8") if data is not None else None
        request = urllib_request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=payload,
            headers=headers or {},
            method=method,
        )
        try:
            response = self.opener.open(request, timeout=10)
        except urllib_error.HTTPError as error:
            response = error
        return _Response(response)

    def cookie(self, name):
        return next((item.value for item in self.cookies if item.name == name), None)

    def login(self):
        page = self.request("/login")
        csrf = self.cookie(CSRF_COOKIE)
        assert csrf
        response = self.request(
            "/login",
            method="POST",
            data={"username": "operator", "password": PASSWORD},
            headers={"X-CSRF-Token": csrf},
        )
        assert response.status_code == 303
        return response


class _Ingestion:
    async def list_samples(self, *, limit=100, state=None):
        return [
            {
                "sample_id": "10000000-0000-4000-8000-000000000001",
                "camera_id": "20000000-0000-4000-8000-000000000002",
                "captured_at_utc": "2026-10-07T00:00:00+00:00",
                "reason": "operator",
                "sha256": "a" * 64,
                "state": "received",
                "selected": False,
                "object_size_bytes": 128,
                "last_failure_code": None,
                "received_at": "2026-10-07T00:00:00+00:00",
                "retention_until": None,
            }
        ]


class _Annotations:
    def __init__(self):
        self.start_calls = []
        self.finalize_calls = []
        self.label_studio_configured = True

    async def list_reviews(self, *, limit=100):
        return [
            {
                "sample_id": "10000000-0000-4000-8000-000000000001",
                "revision": "30000000-0000-4000-8000-000000000003",
                "stage": "bbox",
                "state": "active",
                "label_studio_task_id": 33,
                "project_id": 7,
                "created_at": "2026-10-07T00:00:00+00:00",
                "finished_at": None,
                "annotation_revision_id": None,
                "provenance": None,
                "human_review_recorded": False,
            }
        ]

    async def start_bbox_review(self, sample_id, *, project_id, workflow):
        self.start_calls.append((sample_id, project_id, workflow))
        return {"sample_id": sample_id, "revision": "30000000-0000-4000-8000-000000000003"}

    async def finalize_annotation(self, sample_id, revision):
        self.finalize_calls.append((sample_id, revision))
        return {"sample_id": sample_id, "revision": revision}


class _Publisher:
    def __init__(self):
        self.selection_calls = []
        self.publish_calls = []
        self.preview_calls = []
        self.invalidate_calls = []

    async def list_publications(self, *, limit=100):
        return [
            {
                "dataset_version": "dataset-123",
                "manifest_hash": "b" * 64,
                "state": "published",
                "split_counts": {"train": {"frames": 20}},
                "training_ready": True,
                "training_reasons": [],
                "evaluation_eligible": False,
                "evaluation_reasons": ["human_relevance_truth_missing"],
            }
        ]

    async def list_publication_candidates(self, *, limit=100):
        return {
            "samples": [
                {
                    "sample_id": "10000000-0000-4000-8000-000000000001",
                    "camera_id": "20000000-0000-4000-8000-000000000002",
                    "captured_at_utc": "2026-10-07T00:00:00+00:00",
                    "reason": "operator",
                    "state": "received",
                    "selected": False,
                    "object_size_bytes": 128,
                    "latest_bbox_revision": "30000000-0000-4000-8000-000000000003",
                    "bbox_reviewed": True,
                }
            ],
            "crops": [],
        }

    async def publication_selection(self, *, target, sample_ids=(), crop_ids=()):
        selection = {"target": target, "sample_ids": list(sample_ids), "crop_ids": list(crop_ids)}
        self.selection_calls.append(selection)
        return selection

    async def publish_dataset(self, selection, config_version):
        self.publish_calls.append((selection, config_version))
        return {"dataset_version": "dataset-456", "state": "blocked", "training_ready": False}

    async def preview_sample_invalidation(self, sample_id):
        self.preview_calls.append(sample_id)
        return {
            "sample_id": sample_id,
            "sample_exists": True,
            "sample_state": "received",
            "physical_deletion": False,
            "datasets": ["dataset-123"],
            "dataset_states": {"dataset-123": "published"},
            "models": ["model-current"],
            "active_review_count": 1,
            "active_job_count": 2,
            "retained_for_review": True,
            "block_training": True,
            "block_evaluation": True,
        }

    async def invalidate_sample(self, sample_id):
        self.invalidate_calls.append(sample_id)
        return {
            "sample_id": sample_id,
            "datasets": ["dataset-123"],
            "models": ["model-current"],
            "block_training": True,
            "block_evaluation": True,
        }


class _JobRepository:
    def __init__(self, job, artifacts=(), identity=None):
        self.job = job
        self.artifacts = list(artifacts)
        self.identity = identity

    async def result_artifacts_for(self, job_id):
        return self.artifacts

    async def checkpoint_identity(self, job_id):
        return self.identity


class _Identity:
    def __init__(self, job_id):
        self.job_id = job_id

    def as_dict(self):
        return {"job_id": self.job_id, "phase": "evaluation"}


class _ResultStore:
    def __init__(self, payload):
        self.payload = payload

    def read_committed(self, details, *, expected_identity):
        return object(), self.payload


class _Queue:
    def __init__(self, parent_job_id):
        self.retry_calls = []
        self.queued_jobs = {}
        self.cancel_calls = []
        self.reorder_calls = []
        self.submit_calls = []
        self.cancel_pending = False
        self.parent_job_id = parent_job_id
        self.jobs = [
            {"job_id": parent_job_id, "phase": "training", "state": "failed", "reason_code": "probe_execution_failed",
             "reason_detail": {"cause": "worker exited"}, "checkpoint_sha256": None, "lease_token": None,
             "owner_pid": None, "dataset_version": "dataset-123", "model_kind": "detr", "config_version": "train-v1"}
        ]
        self.evaluation_job = None
        self.repository = _JobRepository({}, artifacts=[])

    async def list_jobs(self, *, limit=100):
        return self.jobs[:limit]

    async def get_job_detail(self, job_id):
        return next((job for job in self.jobs if job["job_id"] == job_id), {"job_id": job_id, "state": "unknown"})

    async def get(self, job_id):
        return self.evaluation_job or await self.get_job_detail(job_id)

    async def evaluation_source_block_reasons(self, job_id):
        return ()

    async def retry_job(self, job_id, *, operator_id, intent_token):
        self.retry_calls.append((job_id, operator_id, intent_token))
        key = _operator_retry_dedupe_key(
            parent_job_id=job_id,
            operator_id=operator_id,
            intent_token=intent_token,
        )
        if key not in self.queued_jobs:
            self.queued_jobs[key] = str(uuid4())
        return self.queued_jobs[key]

    async def cancel_queued(self, job_id, *, operator_id):
        self.cancel_calls.append((job_id, operator_id))
        return {
            "job": {"job_id": job_id, "state": "cancelled"},
            "reservation_cleanup": "pending" if self.cancel_pending else "settled",
            "cleanup_pending": self.cancel_pending,
        }

    async def reorder_queued(self, job_id, *, before_job_id, operator_id):
        self.reorder_calls.append((job_id, before_job_id, operator_id))
        return [job_id, before_job_id]

    async def submit(self, dataset_version, model_kind, config_version):
        self.submit_calls.append((dataset_version, model_kind, config_version))
        return "50000000-0000-4000-8000-000000000005"


def _services():
    parent_job_id = str(uuid4())
    services = OperatorServices(
        ingestion=_Ingestion(),
        annotations=_Annotations(),
        review_workflow=object(),
        publisher=_Publisher(),
        queue=_Queue(parent_job_id),
        result_store=None,
        integrations=OperatorUIIntegrations(),
    )
    return services


def _client(services, integrations=None):
    settings = OperatorAuthSettings(
        username="operator",
        password=PASSWORD,
        session_secret=SECRET,
        secure_cookie=False,
    )
    app = create_app(services, auth_settings=settings, integrations=integrations)
    return app, _Browser(app)


def test_unauthenticated_mutations_are_rejected_before_service_calls() -> None:
    services = _services()
    app, browser = _client(services)
    try:
        response = browser.request(
            "/datasets/publish",
            method="POST",
            data={"target": "detr", "config_version": "train-v1"},
            headers={"X-CSRF-Token": "forged"},
        )

        assert response.status_code == 401, response.read()
        assert services.publisher.selection_calls == []
        assert services.publisher.publish_calls == []
    finally:
        browser.close()


def test_authenticated_mutation_requires_csrf_before_publication_side_effects() -> None:
    services = _services()
    app, browser = _client(services)
    try:
        browser.login()

        missing = browser.request(
            "/datasets/publish",
            method="POST",
            data={"target": "detr", "config_version": "train-v1"},
        )
        invalid = browser.request(
            "/datasets/publish",
            method="POST",
            data={"target": "detr", "config_version": "train-v1"},
            headers={"X-CSRF-Token": "wrong"},
        )

        assert missing.status_code == 403, missing.read()
        assert invalid.status_code == 403
        assert services.publisher.selection_calls == []
        assert services.publisher.publish_calls == []

        valid = browser.request(
            "/datasets/publish",
            method="POST",
            data={"target": "detr", "config_version": "train-v1", "sample_ids": "10000000-0000-4000-8000-000000000001"},
            headers={"X-CSRF-Token": browser.cookie(CSRF_COOKIE)},
        )

        assert valid.status_code == 303
        assert services.publisher.selection_calls == [
            {"target": "detr", "sample_ids": ["10000000-0000-4000-8000-000000000001"], "crop_ids": []}
        ]
        assert len(services.publisher.publish_calls) == 1
    finally:
        browser.close()


def test_duplicate_retry_clicks_share_one_httponly_intent_and_one_job() -> None:
    services = _services()
    app, browser = _client(services)
    try:
        browser.login()
        page = browser.request("/jobs")
        page_html = page.read().decode("utf-8")
        intent = browser.cookie(RETRY_COOKIE)
        csrf = browser.cookie(CSRF_COOKIE)
        parent = services.queue.parent_job_id
        assert intent and csrf
        assert intent not in page_html
        retry_cookie_headers = [value for key, value in page.headers.items() if key.lower() == "set-cookie"]
        assert not retry_cookie_headers or "HttpOnly" in " ".join(retry_cookie_headers)

        first = browser.request(
            f"/jobs/{parent}/retry",
            method="POST",
            headers={"X-CSRF-Token": csrf},
        )
        second = browser.request(
            f"/jobs/{parent}/retry",
            method="POST",
            headers={"X-CSRF-Token": csrf},
        )

        assert first.status_code == 303
        assert second.status_code == 303
        assert first.headers["Location"] == second.headers["Location"]
        assert len(services.queue.retry_calls) == 2
        assert services.queue.retry_calls[0][2] == services.queue.retry_calls[1][2] == intent
        assert len(services.queue.queued_jobs) == 1
        assert intent not in first.headers["Location"]
    finally:
        browser.close()


def test_invalidation_preview_labels_effect_and_confirm_calls_invalidation_service() -> None:
    services = _services()
    app, browser = _client(services)
    sample_id = "10000000-0000-4000-8000-000000000001"
    try:
        browser.login()
        page = browser.request(f"/samples/{sample_id}/invalidation-preview")
        content = page.read().decode("utf-8")
        assert page.status_code == 200
        assert "Physical deletion: no" in content
        assert "dataset-123" in content
        assert "model-current" in content
        assert "Active reviews: 1" in content
        assert "Active jobs: 2" in content
        assert services.publisher.invalidate_calls == []

        response = browser.request(
            f"/samples/{sample_id}/invalidate",
            method="POST",
            headers={"X-CSRF-Token": browser.cookie(CSRF_COOKIE)},
        )

        assert response.status_code == 303
        assert services.publisher.invalidate_calls == [sample_id]
    finally:
        browser.close()


def test_job_report_page_displays_committed_report_and_existing_blocked_gate() -> None:
    services = _services()
    job_id = "60000000-0000-4000-8000-000000000006"
    job = {
        "job_id": job_id,
        "phase": "evaluation",
        "state": "completed",
        "dataset_version": "dataset-123",
        "input_sha256": "a" * 64,
        "model_kind": "clip",
        "config_version": "eval-v1",
        "config_sha256": "b" * 64,
        "source_refs": {"checkpoint": {"checkpoint_sha256": "c" * 64}},
        "reason_code": None,
        "reason_detail": {},
        "checkpoint_sha256": None,
        "lease_token": None,
        "owner_pid": None,
    }
    report = {
        "schema_version": 1,
        "execution_status": "succeeded",
        "status": "complete",
        "model_kind": "clip",
        "candidate": {"checkpoint_sha256": "c" * 64},
        "baseline": {"verified": True},
        "source": {"dataset_version": "dataset-123", "manifest_sha256": "a" * 64},
        "evaluation_config": {"version": "eval-v1", "sha256": "b" * 64},
        "evaluation_job": {"job_id": job_id, "config_version": "eval-v1", "config_sha256": "b" * 64},
        "metrics": {"recall_at_5": 0.72},
    }
    services.queue.evaluation_job = job
    services.queue.jobs = [job]
    services.queue.repository = _JobRepository(
        job,
        artifacts=[{"kind": "evaluation", "sha256": "d" * 64, "size_bytes": 16}],
        identity=_Identity(job_id),
    )
    services.result_store = _ResultStore(json.dumps(report).encode("utf-8"))
    app, browser = _client(services)
    try:
        browser.login()

        page = browser.request(f"/jobs/{job_id}")
        content = page.read().decode("utf-8")

        assert page.status_code == 200
        assert "0.72" in content
        assert "Deployment blocked" in content
        assert "cuhk_report_missing" in content
        assert "trusted_quality_policy_missing" in content
    finally:
        browser.close()


def test_all_five_operator_pages_render_and_keep_optional_integrations_explicit() -> None:
    services = _services()
    app, browser = _client(services)
    try:
        browser.login()
        pages = {
            "/samples": "Candidate collection status",
            "/reviews": "Review assignments",
            "/datasets": "Immutable dataset publications",
            "/jobs": "Execution status",
            "/actions": "Execution status",
        }
        for path, heading in pages.items():
            response = browser.request(path)
            body = response.read().decode("utf-8")
            assert response.status_code == 200
            assert heading in body
        assert "Label Studio review setup is not configured" in browser.request("/samples").read().decode("utf-8")
        assert "Label Studio integration is not configured" in browser.request("/reviews").read().decode("utf-8")
        detail = browser.request(f"/jobs/{services.queue.parent_job_id}").read().decode("utf-8")
        assert "Kubeflow run, log, and artifact links are not configured" in detail
    finally:
        browser.close()


def test_configured_external_links_contain_no_embedded_credentials() -> None:
    services = _services()
    integrations = OperatorUIIntegrations(
        label_studio_url="http://label-studio.local",
        label_studio_bbox_project_id=7,
        kubeflow_url="http://127.0.0.1:18080",
    )
    app, browser = _client(services, integrations=integrations)
    try:
        browser.login()
        review_body = browser.request("/reviews").read().decode("utf-8")
        job_body = browser.request(f"/jobs/{services.queue.parent_job_id}").read().decode("utf-8")
        assert "http://label-studio.local/projects/7/data" in review_body
        assert "http://127.0.0.1:18080/pipeline/#/runs/details/" in job_body
        for body in (review_body, job_body):
            assert PASSWORD not in body
            assert SECRET not in body
    finally:
        browser.close()


def test_review_and_job_routes_call_the_existing_operator_services() -> None:
    services = _services()
    integrations = OperatorUIIntegrations(
        label_studio_url="http://label-studio.local",
        label_studio_bbox_project_id=7,
    )
    app, browser = _client(services, integrations=integrations)
    try:
        browser.login()
        csrf = browser.cookie(CSRF_COOKIE)
        sample_id = "10000000-0000-4000-8000-000000000001"

        review = browser.request(
            "/reviews/start",
            method="POST",
            data={"sample_id": sample_id},
            headers={"X-CSRF-Token": csrf},
        )
        queued = browser.request(
            "/jobs/queue",
            method="POST",
            data={"dataset_version": "dataset-123", "model_kind": "detr", "config_version": "train-v1"},
            headers={"X-CSRF-Token": csrf},
        )
        cancelled = browser.request(
            f"/jobs/{services.queue.parent_job_id}/cancel",
            method="POST",
            headers={"X-CSRF-Token": csrf},
        )
        reordered = browser.request(
            "/jobs/reorder",
            method="POST",
            data={"job_id": services.queue.parent_job_id, "before_job_id": ""},
            headers={"X-CSRF-Token": csrf},
        )

        assert review.status_code == 303
        assert queued.status_code == 303
        assert cancelled.status_code == 303
        assert reordered.status_code == 303
        assert services.annotations.start_calls == [(sample_id, 7, services.review_workflow)]
        assert services.queue.submit_calls == [("dataset-123", "detr", "train-v1")]
        assert services.queue.cancel_calls[0][0] == services.queue.parent_job_id
        assert services.queue.reorder_calls[0][0] == services.queue.parent_job_id
    finally:
        browser.close()


def test_cancel_cleanup_pending_is_rendered_as_pending_not_success() -> None:
    services = _services()
    services.queue.cancel_pending = True
    app, browser = _client(services)
    try:
        browser.login()
        response = browser.request(
            f"/jobs/{services.queue.parent_job_id}/cancel",
            method="POST",
            headers={"X-CSRF-Token": browser.cookie(CSRF_COOKIE)},
        )
        body = response.read().decode("utf-8")

        assert response.status_code == 202
        assert "Artifact quota cleanup is pending" in body
        assert services.queue.cancel_calls == [(services.queue.parent_job_id, "operator")]
    finally:
        browser.close()
