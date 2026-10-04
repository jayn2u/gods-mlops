import hashlib
import json
from pathlib import Path

import pytest

from gods_mlops import model_locks
from gods_mlops import infra_locks
from gods_mlops.model_locks import LockValidationError, load_model_lock, validate_model_cache


VALID_REVISION = "5650961749fa93567c0d46fc7f43ea4f9e914107"


def _write_lock(path: Path, revision: str = VALID_REVISION, expected_sha256: str | None = None) -> None:
    content = b"locked model file"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": [
                    {
                        "model_id": "example/detector",
                        "revision": revision,
                        "files": [
                            {"path": "config.json", "sha256": hashlib.sha256(content).hexdigest()},
                            {"path": "model.safetensors", "sha256": expected_sha256 or "0" * 64},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_lock_rejects_mutable_revision(tmp_path: Path) -> None:
    lock_path = tmp_path / "lock.json"
    _write_lock(lock_path, revision="main")

    with pytest.raises(LockValidationError, match="immutable 40-character revision"):
        load_model_lock(lock_path)


def test_incomplete_model_is_not_ready(tmp_path: Path) -> None:
    lock_path = tmp_path / "lock.json"
    _write_lock(lock_path)
    model = load_model_lock(lock_path).models[0]
    model_dir = tmp_path / "cache" / model.model_id / model.revision
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_bytes(b"locked model file")

    with pytest.raises(LockValidationError, match="missing required file: model.safetensors"):
        validate_model_cache(model, model_dir)


def test_model_cache_rejects_hash_mismatch(tmp_path: Path) -> None:
    lock_path = tmp_path / "lock.json"
    _write_lock(lock_path)
    model = load_model_lock(lock_path).models[0]
    model_dir = tmp_path / "cache" / model.model_id / model.revision
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_bytes(b"wrong config")
    (model_dir / "model.safetensors").write_bytes(b"wrong weights")

    with pytest.raises(LockValidationError, match="SHA-256 mismatch"):
        validate_model_cache(model, model_dir)


def test_prepare_keeps_unverified_model_out_of_ready_cache(tmp_path: Path) -> None:
    content = b"locked model file"
    lock_path = tmp_path / "lock.json"
    _write_lock(lock_path, expected_sha256=hashlib.sha256(content).hexdigest())
    model = load_model_lock(lock_path).models[0]
    cache_root = tmp_path / "cache"

    def download_incomplete_model(_model: object, staging_dir: Path) -> None:
        (staging_dir / "config.json").write_bytes(content)
        (staging_dir / "model.safetensors").write_bytes(b"wrong weights")

    prepare_model = getattr(model_locks, "prepare_model", None)
    assert callable(prepare_model), "model preparation must validate before publishing"
    with pytest.raises(LockValidationError, match="SHA-256 mismatch"):
        prepare_model(model, cache_root, downloader=download_incomplete_model)

    target = cache_root / "snapshots" / model.model_id / model.revision
    assert not target.exists()
    assert not any((cache_root / ".staging").iterdir())


def test_model_lock_rejects_path_traversal(tmp_path: Path) -> None:
    lock_path = tmp_path / "lock.json"
    _write_lock(lock_path)
    document = json.loads(lock_path.read_text(encoding="utf-8"))
    document["models"][0]["files"][0]["path"] = "../outside.json"
    lock_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(LockValidationError, match="unsafe file path"):
        load_model_lock(lock_path)


def test_model_cache_rejects_symlink_outside_cache(tmp_path: Path) -> None:
    content = b"locked model file"
    lock_path = tmp_path / "lock.json"
    _write_lock(lock_path, expected_sha256=hashlib.sha256(content).hexdigest())
    model = load_model_lock(lock_path).models[0]
    model_dir = tmp_path / "cache" / model.model_id / model.revision
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_bytes(content)
    outside_file = tmp_path / "outside.safetensors"
    outside_file.write_bytes(content)
    (model_dir / "model.safetensors").symlink_to(outside_file)

    with pytest.raises(LockValidationError, match="regular local file|escapes model cache"):
        validate_model_cache(model, model_dir)


def test_image_lock_requires_immutable_digest(tmp_path: Path) -> None:
    lock_path = tmp_path / "versions.lock.yaml"
    lock_path.write_text(
        "schema_version: 1\nimages:\n  - name: storage\n    source: ghcr.io/example/storage\n    tag: 1.0.0\n",
        encoding="utf-8",
    )
    load_image_lock = getattr(infra_locks, "load_image_lock", None)
    assert callable(load_image_lock), "image lock validation must enforce immutable digests"

    with pytest.raises(LockValidationError, match="sha256 digest"):
        load_image_lock(lock_path)


def test_image_lock_accepts_digest_pinned_source(tmp_path: Path) -> None:
    lock_path = tmp_path / "versions.lock.yaml"
    lock_path.write_text(
        "schema_version: 1\nimages:\n  - name: storage\n    source: ghcr.io/example/storage\n    tag: 1.0.0\n"
        "    digest: sha256:" + "a" * 64 + "\n",
        encoding="utf-8",
    )
    load_image_lock = getattr(infra_locks, "load_image_lock", None)
    assert callable(load_image_lock), "valid immutable image lock must load"
    assert load_image_lock(lock_path).images[0].digest == "sha256:" + "a" * 64
