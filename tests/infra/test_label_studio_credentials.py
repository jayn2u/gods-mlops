from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = REPO_ROOT / "infra" / "kubeflow" / "label-studio" / "prepare-credentials.sh"


def test_label_studio_credential_generator_writes_private_bounded_tokens(tmp_path: Path) -> None:
    if shutil.which("openssl") is None:
        pytest.skip("openssl is required by the credential generator")
    destination = tmp_path / "label-studio.env"
    result = subprocess.run(
        ["bash", str(GENERATOR), str(destination)],
        input="operator@example.invalid\nStrong local password 123!\n",
        capture_output=True,
        check=False,
        text=True,
        timeout=5,
    )

    assert result.returncode == 0, result.stderr
    assert destination.stat().st_mode & 0o777 == 0o600
    values = dict(line.split("=", 1) for line in destination.read_text(encoding="utf-8").splitlines())
    assert values["LABEL_STUDIO_USERNAME"] == "operator@example.invalid"
    assert len(values["LABEL_STUDIO_USER_TOKEN"]) == 40
    assert len(values["MEDIA_CLEANUP_TOKEN"]) >= 32
    assert "Strong local password 123!" not in result.stdout + result.stderr
    assert values["LABEL_STUDIO_PASSWORD"] not in result.stdout + result.stderr


def test_label_studio_credential_generator_does_not_overwrite_existing_files(tmp_path: Path) -> None:
    destination = tmp_path / "label-studio.env"
    destination.write_text("preserve", encoding="utf-8")

    result = subprocess.run(
        ["bash", str(GENERATOR), str(destination)],
        input="",
        capture_output=True,
        check=False,
        text=True,
        timeout=5,
    )

    assert result.returncode != 0
    assert destination.read_text(encoding="utf-8") == "preserve"
