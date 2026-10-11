from __future__ import annotations

import hashlib

import pytest

from gods_mlops.annotations.media_cleanup import delete_review_upload


def test_media_cleanup_is_project_scoped_idempotent_and_hash_checked(tmp_path) -> None:
    upload_root = tmp_path / "media" / "upload"
    upload = upload_root / "17" / "review.jpg"
    upload.parent.mkdir(parents=True)
    data = b"recorded Label Studio image bytes"
    upload.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()

    assert delete_review_upload(
        upload_root=upload_root,
        project_id=17,
        upload_path="/data/upload/17/review.jpg",
        expected_sha256=digest,
        expected_size_bytes=len(data),
    ) is True
    assert upload.exists() is False
    assert delete_review_upload(
        upload_root=upload_root,
        project_id=17,
        upload_path="/data/upload/17/review.jpg",
        expected_sha256=digest,
        expected_size_bytes=len(data),
    ) is False

    upload.write_bytes(data)
    with pytest.raises(ValueError, match="size and SHA-256"):
        delete_review_upload(
            upload_root=upload_root,
            project_id=17,
            upload_path="/data/upload/17/review.jpg",
            expected_sha256="f" * 64,
            expected_size_bytes=len(data),
        )
    with pytest.raises(ValueError, match="project file"):
        delete_review_upload(
            upload_root=upload_root,
            project_id=17,
            upload_path="/data/upload/17/../outside.jpg",
            expected_sha256=digest,
            expected_size_bytes=len(data),
        )
    assert upload.is_file()
