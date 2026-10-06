"""Opt-in real-browser flow against a separately acknowledged owned fixture.

This test never creates or deletes a database, S3 bucket/prefix, Kubernetes resource,
GPU job, or Label Studio project. Root must prepare and acknowledge the fixture first.
"""

from __future__ import annotations

import ipaddress
import json
import os
from pathlib import Path
import re
import stat
import time
from urllib.parse import urlsplit
import uuid

import pytest


def _private_json_file(path_value: str, *, label: str) -> dict:
    path = Path(path_value).expanduser().resolve(strict=True)
    info = path.stat()
    if stat.S_IMODE(info.st_mode) != 0o600 or info.st_uid != os.geteuid():
        pytest.fail(f"{label} must be owned by the current user with mode 0600")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pytest.fail(f"{label} is not readable JSON")
    if not isinstance(value, dict):
        pytest.fail(f"{label} must contain a JSON object")
    return value


def _loopback_url(value: str, expected_port: int) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        pytest.fail("browser fixture URL must be credential-free HTTP(S) without query or fragment")
    host = parsed.hostname or ""
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host.lower() == "localhost"
    if not loopback or parsed.port != expected_port:
        pytest.fail("browser fixture must use its explicitly owned loopback port")
    return value.rstrip("/")


def _delay_request(route) -> None:
    time.sleep(0.75)
    route.continue_()


def test_operator_flow_in_root_acknowledged_browser_fixture() -> None:
    config_path = os.environ.get("GODS_MLOPS_OPERATOR_E2E_CONFIG")
    if not config_path:
        pytest.skip("set GODS_MLOPS_OPERATOR_E2E_CONFIG for a root-acknowledged browser fixture")
    config = _private_json_file(config_path, label="operator browser fixture config")
    fixture_id = config.get("fixture_id")
    try:
        fixture_id = str(uuid.UUID(str(fixture_id)))
    except (ValueError, TypeError):
        pytest.fail("browser fixture ID must be a UUID")
    if os.environ.get("GODS_MLOPS_OPERATOR_E2E_ROOT_ACK") != fixture_id:
        pytest.skip("root has not acknowledged this browser fixture ID")

    required = {
        "base_url",
        "loopback_port",
        "database_name",
        "s3_bucket",
        "s3_key_prefixes",
        "label_studio_url",
        "label_studio_port",
        "label_media_cleanup_url",
        "label_media_cleanup_port",
        "label_studio_project_id",
        "kubeflow_url",
        "kubeflow_port",
        "credentials_file",
        "preserve_task8_task9_evidence",
        "cleanup_plan",
        "review_sample_id",
        "publication_sample_ids",
        "publication_crop_ids",
        "dataset_target",
        "dataset_config_version",
        "model_kind",
        "training_config_version",
        "retry_parent_job_id",
        "evaluation_job_id",
        "expected_deployment_block_reasons",
        "artifact_output_dir",
    }
    if required - set(config):
        pytest.fail("browser fixture config is missing required ownership or flow fields")
    if not str(config["database_name"]).startswith("gods_mlops_task10_"):
        pytest.fail("browser fixture must use a dedicated task10 database")
    if str(config["s3_bucket"]) != f"gods-mlops-task10-{fixture_id}":
        pytest.fail("browser fixture must use its dedicated task10 S3 bucket")
    if config["s3_key_prefixes"] != {
        "samples": "samples/",
        "datasets": "datasets/",
        "jobs": "jobs/",
    }:
        pytest.fail("browser fixture must account for the service-owned S3 key prefixes")
    if config["preserve_task8_task9_evidence"] is not True or not str(config["cleanup_plan"]).strip():
        pytest.fail("browser fixture must preserve prior task evidence and provide a cleanup plan")
    if config["dataset_target"] in {"detr", "both"} and len(config["publication_sample_ids"]) < 20:
        pytest.fail("DETR browser publication fixture requires at least 20 reviewed samples")
    if config["dataset_target"] in {"clip", "both"} and len(config["publication_crop_ids"]) < 20:
        pytest.fail("CLIP browser publication fixture requires at least 20 reviewed crops")
    if config["review_sample_id"] in config["publication_sample_ids"]:
        pytest.fail("review sample must be outside the immutable publication selection")
    if type(config["label_studio_project_id"]) is not int or config["label_studio_project_id"] <= 0:
        pytest.fail("browser fixture must use a dedicated positive Label Studio project ID")

    credentials = _private_json_file(str(config["credentials_file"]), label="operator browser credentials")
    if not isinstance(credentials.get("username"), str) or not isinstance(credentials.get("password"), str):
        pytest.fail("operator credential file must contain username and password fields")
    base_url = _loopback_url(str(config["base_url"]), int(config["loopback_port"]))
    label_studio_url = _loopback_url(str(config["label_studio_url"]), int(config["label_studio_port"]))
    _loopback_url(str(config["label_media_cleanup_url"]), int(config["label_media_cleanup_port"]))
    kubeflow_url = _loopback_url(str(config["kubeflow_url"]), int(config["kubeflow_port"]))

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        pytest.fail("install the optional browser dependencies with `uv sync --group e2e`", pytrace=False)

    artifact_root = (Path(__file__).resolve().parents[2] / "output").resolve()
    screenshot_dir = Path(str(config["artifact_output_dir"])).expanduser().resolve()
    if not screenshot_dir.is_relative_to(artifact_root):
        pytest.fail("browser screenshots must remain under the repository output directory")
    if fixture_id not in screenshot_dir.parts:
        pytest.fail("browser screenshots must be stored under the acknowledged fixture ID")
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 1000})
        page = context.new_page()
        page.goto(f"{base_url}/login", wait_until="domcontentloaded")
        page.get_by_label("Operator").fill(credentials["username"])
        page.get_by_label("Password").fill(credentials["password"])
        page.get_by_role("button", name="Sign in").click()
        page.wait_for_url(f"{base_url}/samples")
        page.screenshot(path=str(screenshot_dir / "samples.png"), full_page=True)

        review_sample = str(config["review_sample_id"])
        review_row = page.locator("tr").filter(has_text=review_sample)
        review_row.get_by_role("button", name="Send to bbox review").click()
        page.wait_for_url(f"{base_url}/reviews")
        page.locator("tr").filter(has_text=review_sample).wait_for()
        review_link = page.get_by_role("link", name="Open Label Studio project").first
        review_link.wait_for()
        project_url = review_link.get_attribute("href", timeout=5000)
        assert project_url.startswith(label_studio_url)
        assert f"/projects/{config['label_studio_project_id']}/" in project_url
        page.screenshot(path=str(screenshot_dir / "reviews.png"), full_page=True)
        label_studio_page = context.new_page()
        label_studio_page.goto(project_url, wait_until="domcontentloaded")
        assert label_studio_page.url.startswith(label_studio_url)
        label_studio_page.screenshot(path=str(screenshot_dir / "label-studio-project.png"), full_page=True)
        label_studio_page.close()

        page.goto(f"{base_url}/datasets", wait_until="domcontentloaded")
        versions_before = set(page.locator("section").first.locator("tbody code").all_text_contents())
        page.get_by_label("Target").select_option(str(config["dataset_target"]))
        page.get_by_label("Dataset config version").fill(str(config["dataset_config_version"]))
        for sample_id in config["publication_sample_ids"]:
            page.locator(f'input[name="sample_ids"][value="{sample_id}"]').check()
        for crop_id in config["publication_crop_ids"]:
            page.locator(f'input[name="crop_ids"][value="{crop_id}"]').check()
        page.route("**/datasets/publish", _delay_request)
        page.get_by_role("button", name="Publish selected dataset").click()
        page.wait_for_url(f"{base_url}/datasets")
        page.wait_for_function(
            """before => Array.from(document.querySelectorAll('section:first-of-type tbody code'))
                .map(element => element.textContent.trim())
                .some(version => version && !before.includes(version))""",
            list(versions_before),
            timeout=60_000,
        )
        page.unroute("**/datasets/publish", _delay_request)
        page.screenshot(path=str(screenshot_dir / "datasets.png"), full_page=True)
        versions_after = set(page.locator("section").first.locator("tbody code").all_text_contents())
        new_versions = sorted(versions_after - versions_before)
        assert new_versions, "browser publication did not create a visible immutable dataset version"
        dataset_version = new_versions[-1]
        page.get_by_label("Published dataset version").fill(dataset_version)
        page.get_by_label("Model").select_option(str(config["model_kind"]))
        page.get_by_label("Measured config version").fill(str(config["training_config_version"]))
        page.get_by_role("button", name="Queue training").click()
        page.wait_for_url(re.compile(rf"{re.escape(base_url)}/jobs/[0-9a-f-]+$"))
        queued_job_id = page.url.rsplit("/", 1)[-1]
        assert re.fullmatch(r"[0-9a-f-]{36}", queued_job_id)
        page.screenshot(path=str(screenshot_dir / "queued-job.png"), full_page=True)

        retry_parent_id = str(config["retry_parent_job_id"])
        page.goto(f"{base_url}/jobs/{retry_parent_id}", wait_until="domcontentloaded")
        retry_posts = []

        def count_retry_post(request):
            if request.method == "POST" and request.url.endswith(f"/jobs/{retry_parent_id}/retry"):
                retry_posts.append(request.url)

        page.on("request", count_retry_post)
        page.route(f"**/jobs/{retry_parent_id}/retry", _delay_request)
        page.get_by_role("button", name="Rerun job").dblclick()
        page.wait_for_function(
            """parentId => {
                const match = window.location.pathname.match(/\\/jobs\\/([0-9a-f-]+)$/);
                const visible = Array.from(document.querySelectorAll('p'))
                    .find(element => element.textContent.startsWith('Job ID:'))
                    ?.querySelector('code')?.textContent.trim();
                return match && match[1] !== parentId && visible === match[1];
            }""",
            retry_parent_id,
            timeout=60_000,
        )
        page.unroute(f"**/jobs/{retry_parent_id}/retry", _delay_request)
        retry_job_id = page.url.rsplit("/", 1)[-1]
        assert retry_job_id != retry_parent_id
        assert len(retry_posts) == 1
        assert page.get_by_text(f"Job ID: {retry_job_id}").is_visible()
        page.screenshot(path=str(screenshot_dir / "rerun-job.png"), full_page=True)

        evaluation_job_id = str(config["evaluation_job_id"])
        page.goto(f"{base_url}/jobs/{evaluation_job_id}", wait_until="domcontentloaded")
        page.get_by_text("Committed evaluation report").wait_for()
        page.get_by_text("Deployment blocked").wait_for()
        page.get_by_text("Run details are not available for this job").wait_for()
        kubeflow_link = page.get_by_role("link", name="Open Kubeflow Pipelines")
        assert kubeflow_link.get_attribute("href") == f"{kubeflow_url}/pipeline/#/runs"
        report_text = page.locator("main").inner_text()
        for reason in config["expected_deployment_block_reasons"]:
            assert reason in report_text
        page.screenshot(path=str(screenshot_dir / "blocked-deployment.png"), full_page=True)

        context.close()
        browser.close()


def test_acknowledged_browser_gate_fails_instead_of_skipping_when_playwright_is_missing(
    monkeypatch,
    tmp_path,
) -> None:
    import builtins

    fixture_id = str(uuid.uuid4())
    credentials_path = tmp_path / "credentials.json"
    credentials_path.write_text(json.dumps({"username": "operator", "password": "secret"}), encoding="utf-8")
    credentials_path.chmod(0o600)
    config_path = tmp_path / "config.json"
    sample_ids = [f"sample-{index}" for index in range(20)]
    config = {
        "fixture_id": fixture_id,
        "base_url": "http://127.0.0.1:18000",
        "loopback_port": 18000,
        "database_name": f"gods_mlops_task10_{fixture_id.replace('-', '')}",
        "s3_bucket": f"gods-mlops-task10-{fixture_id}",
        "s3_key_prefixes": {"samples": "samples/", "datasets": "datasets/", "jobs": "jobs/"},
        "label_studio_url": "http://127.0.0.1:18001",
        "label_studio_port": 18001,
        "label_media_cleanup_url": "http://127.0.0.1:18002",
        "label_media_cleanup_port": 18002,
        "label_studio_project_id": 7,
        "kubeflow_url": "http://127.0.0.1:18003",
        "kubeflow_port": 18003,
        "credentials_file": str(credentials_path),
        "preserve_task8_task9_evidence": True,
        "cleanup_plan": "no runtime actions in this dependency-gate test",
        "review_sample_id": "review-sample",
        "publication_sample_ids": sample_ids,
        "publication_crop_ids": [],
        "dataset_target": "detr",
        "dataset_config_version": "dataset-v1",
        "model_kind": "detr",
        "training_config_version": "train-v1",
        "retry_parent_job_id": str(uuid.uuid4()),
        "evaluation_job_id": str(uuid.uuid4()),
        "expected_deployment_block_reasons": [],
        "artifact_output_dir": str(tmp_path / "output" / fixture_id),
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    config_path.chmod(0o600)
    monkeypatch.setenv("GODS_MLOPS_OPERATOR_E2E_CONFIG", str(config_path))
    monkeypatch.setenv("GODS_MLOPS_OPERATOR_E2E_ROOT_ACK", fixture_id)
    real_import = builtins.__import__

    def import_without_playwright(name, *args, **kwargs):
        if name.startswith("playwright"):
            raise ImportError("Playwright intentionally absent in dependency-gate test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_playwright)

    try:
        test_operator_flow_in_root_acknowledged_browser_fixture()
    except pytest.skip.Exception as skipped:
        pytest.fail(f"an acknowledged browser flow skipped instead of failing: {skipped}")
    except pytest.fail.Exception as failure:
        assert "install" in str(failure).lower()
        assert "e2e" in str(failure).lower()
    else:
        pytest.fail("an acknowledged browser flow proceeded without Playwright")
