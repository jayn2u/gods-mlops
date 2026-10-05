from __future__ import annotations

from importlib.resources import files


def test_all_schema_migrations_are_packaged_as_stable_resources() -> None:
    migrations = files("gods_mlops.migrations")

    names = sorted(item.name for item in migrations.iterdir() if item.name.endswith(".sql"))

    assert names == [
        "0001_label_review.sql",
        "0002_bbox_revision_heads.sql",
        "0003_retention_events.sql",
        "0004_annotation_crops.sql",
        "0005_crop_batch_completion.sql",
        "0006_dataset_adoptions.sql",
        "0007_revisioned_relevance.sql",
        "0008_label_studio_media_state.sql",
        "0009_crop_expiry_reconciliation.sql",
        "0010_relevance_review_dependencies.sql",
    ]
    assert all(migrations.joinpath(name).read_text(encoding="utf-8").strip() for name in names)
