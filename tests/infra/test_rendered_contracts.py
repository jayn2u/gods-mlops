from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest
import yaml

from gods_mlops.infra_locks import load_image_lock


REPO_ROOT = Path(__file__).resolve().parents[2]
KUBEFLOW_ROOT = REPO_ROOT / "infra" / "kubeflow"
LOCK_PATH = REPO_ROOT / "infra" / "versions.lock.yaml"
EXPECTED_KUBEFLOW_COMMIT = "f09f3eeaa25cc852665f460497a42b7fc68639ac"
PRESERVED_PREFIXES = (
    "/data/jayn2u/minio",
    "/data/jayn2u/labclip-cache",
    "/data/jayn2u/labclip-k3s",
    "/data/jayn2u/gods-mlops-model-preparation",
    "/mnt/data/gods-mlops",
    "/mnt/data/minio-code",
    "/mnt/data/labclip-cache",
    "/mnt/data/labclip-k3s",
)


@pytest.fixture(scope="module")
def rendered_objects() -> list[dict]:
    kubectl = shutil.which("kubectl")
    assert kubectl, "kubectl is required to render the pinned Kubeflow distribution"
    if not (KUBEFLOW_ROOT / "kustomization.yaml").is_file():
        return []
    result = subprocess.run(
        [kubectl, "kustomize", str(KUBEFLOW_ROOT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return [item for item in yaml.safe_load_all(result.stdout) if item]


def _find_one(objects: list[dict], *, kind: str, name: str) -> dict:
    matches = [
        item
        for item in objects
        if item.get("kind") == kind
        and item.get("metadata", {}).get("name") == name
    ]
    assert len(matches) == 1, f"expected one {kind}/{name}, found {len(matches)}"
    return matches[0]


def test_kubeflow_distribution_matches_immutable_version_lock() -> None:
    lock = load_image_lock(LOCK_PATH)
    lock_doc = yaml.safe_load(LOCK_PATH.read_text(encoding="utf-8"))
    assert lock.images
    version_commit = lock_doc["platform"]["kubeflow"]["commit"]
    assert version_commit == EXPECTED_KUBEFLOW_COMMIT

    kustomization_path = KUBEFLOW_ROOT / "kustomization.yaml"
    assert kustomization_path.is_file(), "the Gods Kubeflow kustomization must be defined"
    kustomization = yaml.safe_load(
        kustomization_path.read_text(encoding="utf-8")
    )
    upstream_resources = [
        resource
        for resource in kustomization.get("resources", [])
        if "kubeflow/community-distribution" in resource
    ]
    assert upstream_resources == [
        f"github.com/kubeflow/community-distribution/example?ref={version_commit}"
    ]


def test_profile_and_gpu_admission_are_scoped_to_gods_training_namespace(
    rendered_objects: list[dict],
) -> None:
    profile = _find_one(rendered_objects, kind="Profile", name="gods-mlops")
    dex = _find_one(rendered_objects, kind="ConfigMap", name="dex")
    dex_config = yaml.safe_load(dex["data"]["config.yaml"])
    static_passwords = dex_config["staticPasswords"]
    assert len(static_passwords) == 1, "the local Kubeflow overlay must keep one operator identity"
    assert profile["spec"]["owner"]["name"] == static_passwords[0]["email"]

    mutation = _find_one(
        rendered_objects,
        kind="MutatingAdmissionPolicy",
        name="gods-mlops-gpu-placement",
    )
    mutation_spec = mutation["spec"]
    assert any(
        "request.namespace == 'gods-mlops'" in condition.get("expression", "")
        for condition in mutation_spec.get("matchConditions", [])
    )
    mutation_expression = mutation_spec["mutations"][0]["applyConfiguration"][
        "expression"
    ]
    assert mutation_expression.lstrip().startswith("Object{")
    assert "kubernetes.io/hostname" in mutation_expression
    assert '"ubuntu"' in mutation_expression
    assert '"nvidia"' in mutation_expression
    mutation_binding = _find_one(
        rendered_objects,
        kind="MutatingAdmissionPolicyBinding",
        name="gods-mlops-gpu-placement-binding",
    )
    assert mutation_binding["spec"]["policyName"] == "gods-mlops-gpu-placement"
    assert mutation_binding["spec"]["matchResources"]["namespaceSelector"][
        "matchLabels"
    ]["kubernetes.io/metadata.name"] == "gods-mlops"

    validation = _find_one(
        rendered_objects,
        kind="ValidatingAdmissionPolicy",
        name="gods-mlops-gpu-placement-validation",
    )
    assert any(
        "request.namespace == 'gods-mlops'" in condition.get("expression", "")
        for condition in validation["spec"].get("matchConditions", [])
    )
    validation_expressions = [
        item.get("expression", "") for item in validation["spec"].get("validations", [])
    ]
    assert any(
        "kubernetes.io/hostname" in expression and "ubuntu" in expression
        for expression in validation_expressions
    )
    assert any(
        "runtimeClassName" in expression and "nvidia" in expression
        for expression in validation_expressions
    )
    assert any(
        "size()" in expression and "== 1" in expression
        for expression in validation_expressions
    )
    assert any(
        'quantity("1")' in expression for expression in validation_expressions
    )

    binding = _find_one(
        rendered_objects,
        kind="ValidatingAdmissionPolicyBinding",
        name="gods-mlops-gpu-placement-validation-binding",
    )
    assert binding["spec"]["validationActions"] == ["Deny"]
    assert binding["spec"]["matchResources"]["namespaceSelector"][
        "matchLabels"
    ]["kubernetes.io/metadata.name"] == "gods-mlops"


def test_gods_local_volumes_are_retained_and_use_only_new_paths(
    rendered_objects: list[dict],
) -> None:
    storage_class = _find_one(
        rendered_objects,
        kind="StorageClass",
        name="gods-mlops-local-retain",
    )
    assert storage_class["provisioner"] == "kubernetes.io/no-provisioner"
    assert storage_class["reclaimPolicy"] == "Retain"

    volumes = [
        item
        for item in rendered_objects
        if item.get("kind") == "PersistentVolume"
        and item.get("metadata", {}).get("labels", {}).get("gods.io/managed-storage")
        == "true"
    ]
    expected_claims = {
        ("gods-mlops", "gods-mlops-objects"): ("gods-mlops-objects", "1Ti", "/data/jayn2u/gods-mlops/objects", "object-store"),
        ("gods-mlops", "gods-mlops-cache"): ("gods-mlops-cache", "200Gi", "/data/jayn2u/gods-mlops/cache", "cache"),
        ("gods-mlops", "gods-mlops-metadata"): ("gods-mlops-metadata", "50Gi", "/data/jayn2u/gods-mlops/metadata/platform", "metadata"),
        ("gods-mlops", "gods-mlops-spool"): ("gods-mlops-spool", "20Gi", "/mnt/data/gods-mlops-runtime/spool", "spool"),
        ("gods-mlops", "gods-mlops-ingestion-postgres"): ("gods-mlops-ingestion-postgres", "10Gi", "/data/jayn2u/gods-mlops/metadata/ingestion-postgres", "database"),
        ("gods-mlops", "gods-mlops-label-studio-media"): ("gods-mlops-label-studio-media", "100Gi", "/data/jayn2u/gods-mlops/metadata/label-studio-media", "media"),
        ("gods-mlops", "metadata-postgres"): ("gods-mlops-kfp-metadata-postgres", "10Gi", "/data/jayn2u/gods-mlops/metadata/kubeflow-user/gods-mlops/metadata-postgres", "database"),
        ("kubeflow", "katib-mysql"): ("gods-mlops-katib-mysql", "10Gi", "/data/jayn2u/gods-mlops/metadata/kubeflow/katib-mysql", "database"),
        ("kubeflow", "model-catalog-postgres"): ("gods-mlops-model-catalog-postgres", "5Gi", "/data/jayn2u/gods-mlops/metadata/kubeflow/model-catalog-postgres", "database"),
        ("kubeflow", "mysql-pv-claim"): ("gods-mlops-kfp-mysql", "20Gi", "/data/jayn2u/gods-mlops/metadata/kubeflow/mysql-pv-claim", "database"),
        ("kubeflow", "seaweedfs-pvc"): ("gods-mlops-seaweedfs", "20Gi", "/data/jayn2u/gods-mlops/metadata/kubeflow/seaweedfs-pvc", "object-store"),
    }
    volumes_by_name = {item["metadata"]["name"]: item for item in volumes}
    assert set(volumes_by_name) == {value[0] for value in expected_claims.values()}
    for volume in volumes:
        spec = volume["spec"]
        assert spec["persistentVolumeReclaimPolicy"] == "Retain"
        assert spec["storageClassName"] == "gods-mlops-local-retain"
        local_path = spec["local"]["path"]
        assert local_path == next(value[2] for value in expected_claims.values() if value[0] == volume["metadata"]["name"])
        assert volume["metadata"]["labels"]["gods.io/recovery-kind"] == next(value[3] for value in expected_claims.values() if value[0] == volume["metadata"]["name"])
        local_path_parts = Path(local_path)
        assert any(
            local_path_parts.is_relative_to(Path(root))
            for root in ("/data/jayn2u/gods-mlops", "/mnt/data/gods-mlops-runtime")
        )
        assert not any(
            local_path_parts == Path(path) or Path(path) in local_path_parts.parents
            for path in PRESERVED_PREFIXES
        )

        affinity = spec["nodeAffinity"]["required"]["nodeSelectorTerms"]
        hostnames = {
            value
            for term in affinity
            for expression in term.get("matchExpressions", [])
            if expression.get("key") == "kubernetes.io/hostname"
            for value in expression.get("values", [])
        }
        assert len(hostnames) == 1, f"{volume['metadata']['name']} must have one node pin"
        assert hostnames.issubset({"ubuntu", "vis-lab"})
        claim_ref = spec["claimRef"]
        assert {"namespace": claim_ref["namespace"], "name": claim_ref["name"]} == next(
            {"namespace": namespace, "name": claim_name}
            for (namespace, claim_name), value in expected_claims.items()
            if value[0] == volume["metadata"]["name"]
        )

    claims = [
        item
        for item in rendered_objects
        if item.get("kind") == "PersistentVolumeClaim"
    ]
    claims_by_name = {
        (item.get("metadata", {}).get("namespace", "default"), item["metadata"]["name"]): item
        for item in claims
    }
    assert set(claims_by_name) == set(expected_claims), "every rendered PVC must have an explicit retained backing volume"
    for (namespace, name), item in claims_by_name.items():
        volume_name, expected_size, _, _ = expected_claims[(namespace, name)]
        spec = item["spec"]
        assert item["metadata"]["labels"]["gods.io/managed-storage"] == "true"
        assert spec["storageClassName"] == "gods-mlops-local-retain"
        assert spec["volumeName"] == volume_name
        assert spec["resources"]["requests"]["storage"] == expected_size
        assert "ReadWriteOnce" in spec["accessModes"]
        volume_spec = volumes_by_name[volume_name]["spec"]
        assert volume_spec["capacity"]["storage"] == expected_size


def test_label_studio_uses_private_postgres_backed_single_operator_service_and_shared_media(
    rendered_objects: list[dict],
) -> None:
    service = _find_one(rendered_objects, kind="Service", name="gods-mlops-label-studio")
    assert service["spec"]["type"] == "ClusterIP"
    assert {port["port"] for port in service["spec"]["ports"]} == {8080, 8090}

    deployment = _find_one(rendered_objects, kind="Deployment", name="gods-mlops-label-studio")
    pod = deployment["spec"]["template"]["spec"]
    assert pod["nodeSelector"]["kubernetes.io/hostname"] == "ubuntu"
    assert pod["automountServiceAccountToken"] is False
    assert pod["securityContext"]["runAsNonRoot"] is True
    assert pod["securityContext"]["runAsUser"] == 10001
    assert pod["securityContext"]["runAsGroup"] == 10001
    containers = {container["name"]: container for container in pod["containers"]}
    assert set(containers) == {"label-studio", "media-cleanup"}
    app = containers["label-studio"]
    assert app["image"] == (
        "docker.io/heartexlabs/label-studio:1.23.2"
        "@sha256:afcc516a22775a39d0d66f4c2ddc95d01be3e3d3862b21fcf4f9de34b4ad4e12"
    )
    app_env = {entry["name"]: entry for entry in app["env"]}
    assert app_env["POSTGRE_PASSWORD"]["valueFrom"]["secretKeyRef"]["name"] == "gods-mlops-ingestion-credentials"
    assert app_env["LABEL_STUDIO_USER_TOKEN"]["valueFrom"]["secretKeyRef"]["name"] == "gods-label-studio-credentials"
    cleanup = containers["media-cleanup"]
    cleanup_env = {entry["name"]: entry for entry in cleanup["env"]}
    assert cleanup_env["GODS_MLOPS_LABEL_STUDIO_UPLOAD_ROOT"]["value"] == "/label-studio/data/media/upload"
    assert cleanup_env["GODS_MLOPS_LABEL_MEDIA_CLEANUP_TOKEN"]["valueFrom"]["secretKeyRef"]["name"] == "gods-label-studio-credentials"
    app_data_mount = next(item for item in app["volumeMounts"] if item["name"] == "media")
    cleanup_data_mount = next(item for item in cleanup["volumeMounts"] if item["name"] == "media")
    assert app_data_mount["mountPath"] == cleanup_data_mount["mountPath"] == "/label-studio/data"
    media_volume = next(item for item in pod["volumes"] if item["name"] == "media")
    assert media_volume["persistentVolumeClaim"]["claimName"] == "gods-mlops-label-studio-media"

    config = _find_one(rendered_objects, kind="ConfigMap", name="gods-mlops-label-studio-config")["data"]
    assert config["DJANGO_DB"] == "default"
    assert config["POSTGRE_NAME"] == "gods_ingestion"
    assert config["POSTGRE_HOST"] == "gods-mlops-ingestion-postgres"
    assert config["LABEL_STUDIO_BASE_DATA_DIR"] == "/label-studio/data"
    assert config["DISABLE_SIGNUP_WITHOUT_LINK"] == "true"
    assert config["SSRF_PROTECTION_ENABLED"] == "true"
    assert config["DEBUG"] == "false"
    assert config["COLLECT_ANALYTICS"] == "false"


def test_retention_cronjob_is_bounded_serial_and_uses_the_ingestion_image(
    rendered_objects: list[dict],
) -> None:
    cron = _find_one(rendered_objects, kind="CronJob", name="gods-mlops-retention")
    spec = cron["spec"]
    assert spec["schedule"] == "17 * * * *"
    assert spec["timeZone"] == "Asia/Seoul"
    assert spec["concurrencyPolicy"] == "Forbid"
    assert spec["suspend"] is True
    job = spec["jobTemplate"]["spec"]
    assert job["backoffLimit"] == 2
    assert job["activeDeadlineSeconds"] <= 900
    pod = job["template"]["spec"]
    assert pod["restartPolicy"] == "Never"
    assert pod["automountServiceAccountToken"] is False
    worker = pod["containers"][0]
    assert worker["image"] == "gods-mlops-ingestion:0.2.0"
    assert worker["command"] == ["python", "-m", "gods_mlops.retention.runner"]
    names = {entry["name"] for entry in worker["env"]}
    assert {
        "GODS_MLOPS_DATABASE_URL",
        "GODS_MLOPS_S3_ENDPOINT_URL",
        "GODS_MLOPS_S3_ACCESS_KEY",
        "GODS_MLOPS_S3_SECRET_KEY",
        "GODS_MLOPS_LABEL_STUDIO_API_TOKEN",
        "GODS_MLOPS_LABEL_MEDIA_CLEANUP_TOKEN",
    }.issubset(names)
    api_token_ref = next(entry for entry in worker["env"] if entry["name"] == "GODS_MLOPS_LABEL_STUDIO_API_TOKEN")
    assert api_token_ref["valueFrom"]["secretKeyRef"]["key"] == "LABEL_STUDIO_API_TOKEN"


def test_render_has_no_external_service_or_ingress_exposure(
    rendered_objects: list[dict],
) -> None:
    assert rendered_objects, "the Gods Kubeflow distribution must render at least one object"
    for item in rendered_objects:
        kind = item.get("kind")
        spec = item.get("spec", {})
        if kind == "Service":
            assert spec.get("type", "ClusterIP") not in {"NodePort", "LoadBalancer"}
            assert all("nodePort" not in port for port in spec.get("ports", []))
        assert kind != "Ingress", f"public ingress is outside the approved deployment"
        if kind in {"Pod", "Deployment", "StatefulSet", "DaemonSet", "Job", "CronJob"}:
            pod_spec = (
                spec.get("jobTemplate", {}).get("spec", {}).get("template", {}).get("spec", {})
                if kind == "CronJob"
                else spec.get("template", {}).get("spec", {})
                if kind in {"Deployment", "StatefulSet", "DaemonSet", "Job"}
                else spec
            )
            for container in pod_spec.get("containers", []) + pod_spec.get(
                "initContainers", []
            ):
                assert all("hostPort" not in port for port in container.get("ports", []))


def test_host_preflight_is_read_only_and_writes_the_installation_inventory() -> None:
    playbook_path = REPO_ROOT / "infra" / "ansible" / "preflight.yml"
    assert playbook_path.is_file(), "a read-only host preflight playbook must be defined"
    documents = [item for item in yaml.safe_load_all(playbook_path.read_text(encoding="utf-8")) if item]
    assert len(documents) == 1
    plays = documents[0]
    assert len(plays) == 2, "preflight should inspect remote hosts then write a local report"
    remote_play, report_play = plays
    assert remote_play["hosts"] == "gods_cluster"
    assert remote_play.get("become", False) == (
        "{{ gods_preflight_use_become | default(false) | bool }}"
    )

    allowed_remote_modules = {
        "ansible.builtin.assert",
        "ansible.builtin.command",
        "ansible.builtin.set_fact",
        "ansible.builtin.slurp",
        "ansible.builtin.stat",
    }
    for task in remote_play.get("tasks", []):
        module_name = next(
            (name for name in allowed_remote_modules if name in task),
            None,
        )
        assert module_name is not None, f"remote preflight task can mutate a host: {task}"
        if module_name == "ansible.builtin.command":
            assert task.get("changed_when") is False
            if task.get("name") == (
                "Record the effective UID used for authenticated preflight checks"
            ):
                assert "failed_when" not in task

    local_copy_tasks = [
        task
        for task in report_play.get("tasks", [])
        if "ansible.builtin.copy" in task
    ]
    assert len(local_copy_tasks) == 1
    copy_module = local_copy_tasks[0]["ansible.builtin.copy"]
    assert copy_module.get("mode") == "0600"
    assert copy_module.get("dest") == "{{ gods_preflight_report_path }}"
    assert "preflight.json" in report_play.get("vars", {}).get(
        "gods_preflight_report_path", ""
    )
