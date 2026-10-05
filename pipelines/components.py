"""CPU-only Kubeflow components that submit work to the Task 7 controller."""

import re

from kfp import dsl, kubernetes

_IMAGE_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:@-]*@sha256:[0-9a-f]{64}$")


def validate_training_image(image: str) -> str:
    if not isinstance(image, str) or not _IMAGE_REF.fullmatch(image):
        raise ValueError("KFP components require an immutable training image digest")
    return image


def controller_component(image: str):
    """Create one CPU component baked with the source-matched image reference."""
    image = validate_training_image(image)

    @dsl.component(base_image=image)
    def run_training_job(dataset_version: str, model_kind: str, config_version: str):
        import os
        import subprocess

        subprocess.run(
            [
                "gods-mlops-training",
                "run-training",
                "--dataset-version",
                dataset_version,
                "--model-kind",
                model_kind,
                "--config-version",
                config_version,
                "--worker-image",
                os.environ["GODS_MLOPS_TRAINING_IMAGE"],
            ],
            check=True,
        )

    @dsl.component(base_image=image)
    def run_preparation_job(source_selections_json: str, model_kind: str, config_version: str):
        import os
        import subprocess

        subprocess.run(
            [
                "gods-mlops-training",
                "run-preparation",
                "--source-selections-json",
                source_selections_json,
                "--model-kind",
                model_kind,
                "--config-version",
                config_version,
                "--worker-image",
                os.environ["GODS_MLOPS_TRAINING_IMAGE"],
            ],
            check=True,
        )

    return run_training_job, run_preparation_job


def configure_controller_task(task, *, training_image: str, preparation: bool = False):
    """Attach only Secret/ConfigMap references; parameters remain semantic identities."""
    validate_training_image(training_image)
    task.set_cpu_request("1").set_memory_request("1Gi")
    task.set_caching_options(enable_caching=False)
    task.set_retry(
        num_retries=2,
        backoff_duration="30s",
        backoff_factor=2.0,
        backoff_max_duration="5m",
    )
    task.set_env_variable("GODS_MLOPS_TRAINING_IMAGE", training_image)
    task.set_env_variable("GODS_MLOPS_WORKER_NAMESPACE", "gods-mlops")
    kubernetes.use_secret_as_env(
        task,
        secret_name="gods-mlops-ingestion-credentials",
        secret_key_to_env={"DATABASE_URL": "GODS_MLOPS_DATABASE_URL"},
    )
    kubernetes.use_secret_as_env(
        task,
        secret_name="gods-mlops-storage-s3-secret",
        secret_key_to_env={
            "admin_access_key_id": "GODS_MLOPS_S3_ACCESS_KEY",
            "admin_secret_access_key": "GODS_MLOPS_S3_SECRET_KEY",
        },
    )
    kubernetes.use_config_map_as_env(
        task,
        config_map_name="gods-mlops-ingestion-config",
        config_map_key_to_env={
            name: name
            for name in (
                "GODS_MLOPS_S3_ENDPOINT_URL",
                "GODS_MLOPS_S3_BUCKET",
                "GODS_MLOPS_S3_REGION",
                "GODS_MLOPS_UBUNTU_NODE_ID",
                "GODS_MLOPS_UBUNTU_HOST_IDENTITY",
                "GODS_MLOPS_UBUNTU_GPU_UUID",
                "GODS_MLOPS_UBUNTU_FILESYSTEM_IDENTITY",
                "GODS_MLOPS_UBUNTU_STORAGE_PATH",
                "GODS_MLOPS_UBUNTU_SSH_TARGET",
                "GODS_MLOPS_UBUNTU_SSH_PORT",
                "GODS_MLOPS_UBUNTU_SSH_TIMEOUT_SECONDS",
            )
        },
    )
    kubernetes.use_secret_as_volume(
        task,
        secret_name="gods-mlops-ubuntu-observer-ssh",
        mount_path="/ssh-source",
    )
    task.set_env_variable("GODS_MLOPS_UBUNTU_SSH_IDENTITY_FILE", "/ssh-source/id_ed25519")
    task.set_env_variable("GODS_MLOPS_UBUNTU_SSH_KNOWN_HOSTS", "/ssh-source/known_hosts")
    if preparation:
        kubernetes.use_secret_as_env(
            task,
            secret_name="gods-label-studio-credentials",
            secret_key_to_env={
                "LABEL_STUDIO_API_TOKEN": "GODS_MLOPS_LABEL_STUDIO_API_TOKEN",
                "MEDIA_CLEANUP_TOKEN": "GODS_MLOPS_LABEL_MEDIA_CLEANUP_TOKEN",
            },
        )
        kubernetes.use_config_map_as_env(
            task,
            config_map_name="gods-mlops-label-studio-config",
            config_map_key_to_env={
                "LABEL_STUDIO_HOST": "GODS_MLOPS_LABEL_STUDIO_URL",
                "GODS_MLOPS_LABEL_STUDIO_BBOX_PROJECT_ID": "GODS_MLOPS_LABEL_STUDIO_BBOX_PROJECT_ID",
                "GODS_MLOPS_LABEL_STUDIO_CAPTION_PROJECT_ID": "GODS_MLOPS_LABEL_STUDIO_CAPTION_PROJECT_ID",
            },
            optional=True,
        )
        task.set_env_variable(
            "GODS_MLOPS_LABEL_MEDIA_CLEANUP_URL", "http://gods-mlops-label-studio:8090"
        )
    return task
