from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import http.cookiejar
import json
import socket
import threading
import time
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request
from uuid import uuid4

import pytest
import uvicorn

from gods_mlops.evaluation.report import load_operator_evaluation_report
from gods_mlops.jobs.queue import _operator_retry_dedupe_key
from gods_mlops.web.auth import AUTH_COOKIE, CSRF_COOKIE, OperatorAuth
from gods_mlops.web.app import OperatorServices, OperatorUIIntegrations, create_app
from gods_mlops.web.auth import OperatorAuthSettings
from gods_mlops.web.routes import _retry_intent_scope, _retry_intent_token


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
    def __init__(self):
        self.state_filters = []

    async def list_samples(self, *, limit=100, state=None):
        self.state_filters.append(state)
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
        self.reviews = [
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

    async def list_reviews(self, *, limit=100):
        return self.reviews[:limit]

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
        self.retry_intent_generations = {}
        self.retry_intent_generation_calls = []
        self.cancel_pending = False
        self.cancel_cleanup_status = None
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

    async def retry_intent_generation(self, scope_hash, *, session_expires_at):
        self.retry_intent_generation_calls.append(("read", scope_hash, session_expires_at))
        return self.retry_intent_generations.get(scope_hash, 0)

    async def advance_retry_intent_generation(self, scope_hash, *, session_expires_at):
        self.retry_intent_generation_calls.append(("advance", scope_hash, session_expires_at))
        generation = self.retry_intent_generations.get(scope_hash, 0) + 1
        self.retry_intent_generations[scope_hash] = generation
        return generation

    async def cancel_queued(self, job_id, *, operator_id):
        self.cancel_calls.append((job_id, operator_id))
        cleanup = self.cancel_cleanup_status or ("pending" if self.cancel_pending else "settled")
        return {
            "job": {"job_id": job_id, "state": "cancelled"},
            "reservation_cleanup": cleanup,
            "cleanup_pending": cleanup in {"pending", "unknown"},
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


def test_retry_intent_hmac_is_canonical_and_scoped_to_verified_session_operator_and_parent() -> None:
    auth = OperatorAuth(
        OperatorAuthSettings(
            username="operator",
            password=PASSWORD,
            session_secret=SECRET,
            secure_cookie=False,
        )
    )
    session_nonce, _expires_at = auth.session_identity(auth._issue_session())
    parent = str(uuid4())
    canonical, scope = _retry_intent_scope(
        auth,
        session_nonce=session_nonce,
        operator_id="operator",
        parent_job_id=parent.upper(),
    )
    default_token = _retry_intent_token(
        auth,
        session_nonce=session_nonce,
        operator_id="operator",
        parent_job_id=canonical,
        generation=0,
    )

    assert canonical == parent
    assert scope == _retry_intent_scope(
        auth,
        session_nonce=session_nonce,
        operator_id="operator",
        parent_job_id=parent,
    )[1]
    assert default_token == _retry_intent_token(
        auth,
        session_nonce=session_nonce,
        operator_id="operator",
        parent_job_id=parent,
        generation=0,
    )
    assert default_token != _retry_intent_token(
        auth,
        session_nonce=session_nonce,
        operator_id="operator",
        parent_job_id=parent,
        generation=1,
    )
    assert scope != _retry_intent_scope(
        auth,
        session_nonce="a-different-session-nonce",
        operator_id="operator",
        parent_job_id=parent,
    )[1]
    assert scope != _retry_intent_scope(
        auth,
        session_nonce=session_nonce,
        operator_id="different-operator",
        parent_job_id=parent,
    )[1]
    assert scope != _retry_intent_scope(
        auth,
        session_nonce=session_nonce,
        operator_id="operator",
        parent_job_id=str(uuid4()),
    )[1]


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


def test_duplicate_retry_posts_share_session_derived_intent_without_retry_cookies() -> None:
    services = _services()
    app, browser = _client(services)
    try:
        browser.login()
        page = browser.request("/jobs")
        page_html = page.read().decode("utf-8")
        csrf = browser.cookie(CSRF_COOKIE)
        parent = services.queue.parent_job_id
        assert csrf
        assert "Rerun" in page_html
        retry_cookie_headers = [value for key, value in page.headers.items() if key.lower() == "set-cookie"]
        assert not any(RETRY_COOKIE in value for value in retry_cookie_headers)

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
        assert services.queue.retry_calls[0][2] == services.queue.retry_calls[1][2]
        assert len(services.queue.queued_jobs) == 1
        assert {cookie.name for cookie in browser.cookies} == {AUTH_COOKIE, CSRF_COOKIE}
    finally:
        browser.close()


def test_retry_intent_survives_child_navigation_and_can_be_rotated_deliberately() -> None:
    services = _services()
    app, browser = _client(services)
    parent = services.queue.parent_job_id
    try:
        browser.login()
        browser.request("/jobs").read()
        csrf = browser.cookie(CSRF_COOKIE)

        first = browser.request(f"/jobs/{parent}/retry", method="POST", headers={"X-CSRF-Token": csrf})
        first_child = first.headers["Location"]
        browser.request(first_child).read()

        duplicate = browser.request(f"/jobs/{parent}/retry", method="POST", headers={"X-CSRF-Token": csrf})
        assert duplicate.headers["Location"] == first_child
        assert len(services.queue.queued_jobs) == 1
        first_intent = services.queue.retry_calls[0][2]
        assert services.queue.retry_calls[1][2] == first_intent

        prepared = browser.request(
            f"/jobs/{parent}/retry/new-intent",
            method="POST",
            headers={"X-CSRF-Token": csrf},
        )
        assert prepared.status_code == 303
        assert prepared.headers["Location"] == f"/jobs/{parent}"

        deliberate = browser.request(f"/jobs/{parent}/retry", method="POST", headers={"X-CSRF-Token": csrf})
        assert deliberate.status_code == 303
        assert deliberate.headers["Location"] != first_child
        assert len(services.queue.queued_jobs) == 2
        assert services.queue.retry_calls[2][2] != first_intent
        assert services.queue.retry_intent_generation_calls[-1][0] == "read"
    finally:
        browser.close()


def test_explicit_retry_generation_survives_new_auth_app_instance() -> None:
    services = _services()
    parent = services.queue.parent_job_id
    app, browser = _client(services)
    restarted_browser = None
    try:
        browser.login()
        csrf = browser.cookie(CSRF_COOKIE)
        session = browser.cookie(AUTH_COOKIE)
        first = browser.request(f"/jobs/{parent}/retry", method="POST", headers={"X-CSRF-Token": csrf})
        first_child = first.headers["Location"]
        first_intent = services.queue.retry_calls[-1][2]

        prepared = browser.request(
            f"/jobs/{parent}/retry/new-intent",
            method="POST",
            headers={"X-CSRF-Token": csrf},
        )
        assert prepared.status_code == 303
        deliberate = browser.request(f"/jobs/{parent}/retry", method="POST", headers={"X-CSRF-Token": csrf})
        second_child = deliberate.headers["Location"]
        second_intent = services.queue.retry_calls[-1][2]
        assert second_child != first_child
        assert second_intent != first_intent

        restarted_services = _services()
        restarted_services.queue.parent_job_id = parent
        restarted_services.queue.jobs = services.queue.jobs
        restarted_services.queue.retry_intent_generations = services.queue.retry_intent_generations
        restarted_services.queue.queued_jobs = services.queue.queued_jobs
        _restarted_app, restarted_browser = _client(restarted_services)
        headers = {
            "Cookie": f"{AUTH_COOKIE}={session}; {CSRF_COOKIE}={csrf}",
            "X-CSRF-Token": csrf,
        }
        request = urllib_request.Request(
            f"http://127.0.0.1:{restarted_browser.port}/jobs/{parent}/retry",
            data=b"",
            headers=headers,
            method="POST",
        )
        try:
            response = urllib_request.build_opener(_NoRedirect()).open(request, timeout=10)
        except urllib_error.HTTPError as error:
            response = error

        assert response.status == 303
        assert response.headers["Location"] == second_child
        assert restarted_services.queue.retry_calls[0][2] == second_intent
    finally:
        if restarted_browser is not None:
            restarted_browser.close()
        browser.close()


def test_retry_revalidates_the_same_session_after_generation_lookup_expires(monkeypatch) -> None:
    from types import SimpleNamespace

    import gods_mlops.web.auth as auth_module

    services = _services()
    parent = services.queue.parent_job_id
    app, browser = _client(services)
    try:
        browser.login()
        auth = app.state.operator_auth
        session_cookie = browser.cookie(AUTH_COOKIE)
        session_identity = auth.session_identity(session_cookie)
        assert session_identity is not None
        _session_nonce, session_expiry = session_identity
        fake_now = [session_expiry - 0.5]
        monkeypatch.setattr(auth_module, "time", SimpleNamespace(time=lambda: fake_now[0]))
        original_generation_read = services.queue.retry_intent_generation

        async def generation_read_that_crosses_expiry(scope_sha256, *, session_expires_at):
            generation = await original_generation_read(
                scope_sha256,
                session_expires_at=session_expires_at,
            )
            assert fake_now[0] < session_expiry
            fake_now[0] = session_expiry + 0.5
            return generation

        services.queue.retry_intent_generation = generation_read_that_crosses_expiry
        response = browser.request(
            f"/jobs/{parent}/retry",
            method="POST",
            headers={"X-CSRF-Token": browser.cookie(CSRF_COOKIE)},
        )

        assert response.status_code == 401
        assert services.queue.retry_calls == []
        assert services.queue.queued_jobs == {}
    finally:
        browser.close()


def test_actions_page_renders_rerun_without_issuing_per_job_cookies() -> None:
    services = _services()
    app, browser = _client(services)
    parent = services.queue.parent_job_id
    try:
        browser.login()
        actions = browser.request("/actions")
        body = actions.read().decode("utf-8")

        assert "Rerun" in body
        assert not any(
            RETRY_COOKIE in value
            for key, value in actions.headers.items()
            if key.lower() == "set-cookie"
        )
        browser.request("/actions").read()
        assert {cookie.name for cookie in browser.cookies} == {AUTH_COOKIE, CSRF_COOKIE}
        response = browser.request(
            f"/jobs/{parent}/retry",
            method="POST",
            headers={"X-CSRF-Token": browser.cookie(CSRF_COOKIE)},
        )
        assert response.status_code == 303
    finally:
        browser.close()


def test_many_jobs_and_concurrent_first_gets_do_not_grow_retry_cookies_or_split_intent() -> None:
    services = _services()
    parent = services.queue.parent_job_id
    services.queue.jobs = [
        {
            "job_id": parent if index == 0 else str(uuid4()),
            "phase": "training",
            "state": "failed",
            "reason_code": "probe_execution_failed",
            "reason_detail": {},
            "checkpoint_sha256": None,
        }
        for index in range(300)
    ]
    app, browser = _client(services)

    def raw_request(path, *, method="GET"):
        csrf = browser.cookie(CSRF_COOKIE)
        headers = {
            "Cookie": f"{AUTH_COOKIE}={browser.cookie(AUTH_COOKIE)}; {CSRF_COOKIE}={csrf}",
        }
        payload = None
        if method == "POST":
            headers["X-CSRF-Token"] = csrf
            payload = b""
        request = urllib_request.Request(
            f"http://127.0.0.1:{browser.port}{path}",
            data=payload,
            headers=headers,
            method=method,
        )
        try:
            response = urllib_request.build_opener(_NoRedirect()).open(request, timeout=10)
        except urllib_error.HTTPError as error:
            response = error
        return response.status, response.headers.get("Location"), response.headers.get_all("Set-Cookie", [])

    try:
        browser.login()
        assert len(list(browser.cookies)) == 2

        with ThreadPoolExecutor(max_workers=4) as executor:
            page_results = list(executor.map(lambda path: raw_request(path), ["/jobs", "/actions"] * 2))

        assert all(status == 200 for status, _, _ in page_results)
        assert all(not any(RETRY_COOKIE in value for value in headers) for _, _, headers in page_results)
        assert {cookie.name for cookie in browser.cookies} == {AUTH_COOKIE, CSRF_COOKIE}

        historical_jobs = services.queue.jobs
        for start in range(0, len(historical_jobs), 100):
            services.queue.jobs = historical_jobs[start : start + 100]
            page = browser.request("/jobs")
            page.read()
            assert not any(
                RETRY_COOKIE in value
                for key, value in page.headers.items()
                if key.lower() == "set-cookie"
            )
            assert {cookie.name for cookie in browser.cookies} == {AUTH_COOKIE, CSRF_COOKIE}
        services.queue.jobs = historical_jobs[:100]

        with ThreadPoolExecutor(max_workers=4) as executor:
            retry_results = list(
                executor.map(
                    lambda _: raw_request(f"/jobs/{parent}/retry", method="POST"),
                    range(4),
                )
            )

        assert all(status == 303 for status, _, _ in retry_results)
        assert len({location for _, location, _ in retry_results}) == 1
        assert len({call[2] for call in services.queue.retry_calls}) == 1
        assert len(services.queue.queued_jobs) == 1
        assert len(list(browser.cookies)) == 2
    finally:
        browser.close()


def test_empty_samples_filter_means_all_samples() -> None:
    services = _services()
    app, browser = _client(services)
    try:
        browser.login()
        response = browser.request("/samples?state=")

        assert response.status_code == 200
        assert services.ingestion.state_filters == [None]
    finally:
        browser.close()


def test_missing_review_integrations_return_conflict_after_auth_and_csrf() -> None:
    services = _services()
    services.review_workflow = None
    services.annotations.label_studio_configured = False
    app, browser = _client(services)
    sample_id = "10000000-0000-4000-8000-000000000001"
    revision = "30000000-0000-4000-8000-000000000003"
    try:
        browser.login()
        headers = {"X-CSRF-Token": browser.cookie(CSRF_COOKIE)}
        start = browser.request(
            "/reviews/start",
            method="POST",
            data={"sample_id": sample_id},
            headers=headers,
        )
        finalize = browser.request(
            f"/reviews/{sample_id}/{revision}/finalize",
            method="POST",
            headers=headers,
        )

        assert start.status_code == 409
        assert finalize.status_code == 409
    finally:
        browser.close()


def test_provisioning_review_can_retry_the_exact_existing_assignment() -> None:
    services = _services()
    sample_id = "10000000-0000-4000-8000-000000000001"
    revision = "30000000-0000-4000-8000-000000000003"
    services.annotations.reviews[0].update(
        state="provisioning",
        label_studio_task_id=None,
        project_id=None,
    )

    class _RecoverableWorkflow:
        def __init__(self):
            self.provision_calls = []

        async def provision_task(self, *, revision, project_id):
            self.provision_calls.append((revision, project_id))
            if len(self.provision_calls) == 1:
                raise RuntimeError("temporary Label Studio provisioning failure")
            return {"revision": revision, "project_id": project_id, "state": "active"}

    workflow = _RecoverableWorkflow()
    services.review_workflow = workflow
    app, browser = _client(
        services,
        integrations=OperatorUIIntegrations(label_studio_bbox_project_id=7),
    )
    try:
        browser.login()
        reviews = browser.request("/reviews").read().decode("utf-8")
        assert "Retry Label Studio provisioning" in reviews

        headers = {"X-CSRF-Token": browser.cookie(CSRF_COOKIE)}
        first = browser.request(
            f"/reviews/{sample_id}/{revision}/provision",
            method="POST",
            headers=headers,
        )
        second = browser.request(
            f"/reviews/{sample_id}/{revision}/provision",
            method="POST",
            headers=headers,
        )

        assert first.status_code == 409
        assert second.status_code == 303
        assert second.headers["Location"] == "/reviews"
        assert workflow.provision_calls == [(revision, 7), (revision, 7)]
        assert services.annotations.start_calls == []
    finally:
        browser.close()


def test_provisioning_caption_recovery_prefers_its_recorded_project() -> None:
    services = _services()
    sample_id = "10000000-0000-4000-8000-000000000001"
    revision = "30000000-0000-4000-8000-000000000003"
    services.annotations.reviews[0].update(
        stage="caption",
        state="provisioning",
        label_studio_task_id=None,
        project_id=9,
    )

    class _Workflow:
        def __init__(self):
            self.provision_calls = []

        async def provision_task(self, *, revision, project_id):
            self.provision_calls.append((revision, project_id))
            return {"revision": revision, "project_id": project_id, "state": "active"}

    workflow = _Workflow()
    services.review_workflow = workflow
    app, browser = _client(
        services,
        integrations=OperatorUIIntegrations(
            label_studio_url="http://label-studio.local",
            label_studio_bbox_project_id=7,
        ),
    )
    try:
        browser.login()
        body = browser.request("/reviews").read().decode("utf-8")
        assert "http://label-studio.local/projects/9/data" in body
        assert "Retry Label Studio provisioning" in body

        response = browser.request(
            f"/reviews/{sample_id}/{revision}/provision",
            method="POST",
            headers={"X-CSRF-Token": browser.cookie(CSRF_COOKIE)},
        )

        assert response.status_code == 303
        assert workflow.provision_calls == [(revision, 9)]
    finally:
        browser.close()


@pytest.mark.parametrize("stage", ["caption", "relevance"])
def test_provisioning_without_stage_project_does_not_fall_back_to_bbox(stage) -> None:
    services = _services()
    sample_id = "10000000-0000-4000-8000-000000000001"
    revision = "30000000-0000-4000-8000-000000000003"
    services.annotations.reviews[0].update(
        stage=stage,
        state="provisioning",
        label_studio_task_id=None,
        project_id=None,
    )

    class _Workflow:
        def __init__(self):
            self.provision_calls = []

        async def provision_task(self, *, revision, project_id):
            self.provision_calls.append((revision, project_id))
            return {"revision": revision, "project_id": project_id, "state": "active"}

    workflow = _Workflow()
    services.review_workflow = workflow
    app, browser = _client(
        services,
        integrations=OperatorUIIntegrations(label_studio_bbox_project_id=7),
    )
    try:
        browser.login()
        body = browser.request("/reviews").read().decode("utf-8")
        assert "Retry Label Studio provisioning" not in body
        assert "stage-matched Label Studio project is not configured" in body

        response = browser.request(
            f"/reviews/{sample_id}/{revision}/provision",
            method="POST",
            headers={"X-CSRF-Token": browser.cookie(CSRF_COOKIE)},
        )

        assert response.status_code == 409
        assert workflow.provision_calls == []
    finally:
        browser.close()


@pytest.mark.parametrize(
    ("cleanup_state", "expected_notice"),
    [
        ("pending", "Artifact quota cleanup is pending"),
        ("unknown", "cleanup status could not be verified"),
    ],
)
def test_cancelled_cleanup_state_survives_reload_and_offers_idempotent_retry(cleanup_state, expected_notice) -> None:
    services = _services()
    parent = services.queue.parent_job_id
    services.queue.jobs[0].update(
        state="cancelled",
        reason_code=None,
        reservation_cleanup=cleanup_state,
    )
    services.queue.cancel_pending = cleanup_state == "pending"
    services.queue.cancel_cleanup_status = cleanup_state
    app, browser = _client(services)
    try:
        browser.login()
        for path in ("/jobs", "/actions", f"/jobs/{parent}"):
            body = browser.request(path).read().decode("utf-8")
            assert cleanup_state in body
            assert "Retry artifact quota cleanup" in body

        headers = {"X-CSRF-Token": browser.cookie(CSRF_COOKIE)}
        pending = browser.request(f"/jobs/{parent}/cancel", method="POST", headers=headers)
        assert pending.status_code == 202
        assert expected_notice in pending.read().decode("utf-8")

        services.queue.cancel_cleanup_status = "settled"
        settled = browser.request(f"/jobs/{parent}/cancel", method="POST", headers=headers)
        assert settled.status_code == 303
        assert services.queue.cancel_calls == [(parent, "operator"), (parent, "operator")]
    finally:
        browser.close()


def test_kubeflow_navigation_uses_stored_run_identity_and_never_guesses_job_uuid() -> None:
    services = _services()
    job_id = services.queue.parent_job_id
    kfp_run_id = str(uuid4())
    app, browser = _client(
        services,
        integrations=OperatorUIIntegrations(kubeflow_url="http://127.0.0.1:18080"),
    )
    try:
        browser.login()
        missing = browser.request(f"/jobs/{job_id}").read().decode("utf-8")
        assert "Run details are not available for this job" in missing
        assert f"/runs/details/{job_id}" not in missing
        assert "Open Kubeflow Pipelines" in missing

        services.queue.jobs[0]["kubeflow_run_id"] = kfp_run_id
        mapped = browser.request(f"/jobs/{job_id}").read().decode("utf-8")
        assert f"/runs/details/{kfp_run_id}" in mapped
        assert f"/runs/details/{job_id}" not in mapped
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
        assert "Kubeflow Pipelines navigation is not configured" in detail
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
        assert "Open Kubeflow Pipelines" in job_body
        assert "Run details are not available for this job" in job_body
        assert f"/runs/details/{services.queue.parent_job_id}" not in job_body
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
