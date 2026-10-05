# Task 6 review fix report

This round addresses the six Task 6 review findings around split authority, leakage, source invalidation, and model lineage. No training-job, UI, or cluster behavior changed.

## Repairs

1. Split authority now expands through persisted camera/day membership and durable event/clip edges, independent of the current selection. A historical group keeps its assigned split and component identity when a follow-up selection omits old IDs or adds a new member. After the first successful authority epoch, new groups—including groups from a new camera—can join train or validation only.
2. Initial splits are assigned chronologically per camera. Cross-camera components must permit non-empty train, validation, and test partitions for every camera; otherwise publication blocks with `cross_camera_component_prevents_independent_partitions`.
3. A late link crossing fixed splits reports the offending component without looking up an absent prior member. Fixed assignment rows remain unchanged, new members stay unassigned, and the leakage impact persists.
4. `invalidate_sample` writes a source-level tombstone even before the sample belongs to a dataset. New publication checks that tombstone under source locks; an in-flight publication rechecks invalidation before becoming training-ready.
5. A late cross-split link overlays dataset versions that are still publishing. Finalization preserves the existing manifest hash and split rows and leaves evaluation disabled.
6. Late model registration backfills prior leakage and source-invalidation impacts idempotently. Leakage leaves training eligible while blocking evaluation; explicit deletion blocks both.

Migration `0012_dataset_authority_and_invalidations.sql` adds durable link members and source tombstones, and expands the model-impact key to retain multiple reasons for a sample. The runtime migration list and the stable-resource allowlist include this migration.

## Focused verification

The pure and migration-resource selection passed: `15 passed, 11 deselected` across `tests/datasets/test_split.py`, the migration resource test, and focused publisher unit tests. This includes `test_initial_partitions_are_allocated_per_camera`, `test_cross_camera_component_that_breaks_a_camera_partition_is_blocked`, and `test_late_cross_split_component_with_new_member_keeps_fixed_rows_without_stopiteration`.

The following persistence/race selectors each passed against the local isolated PostgreSQL and SeaweedFS test services, using fresh databases so each test starts before any global split authority exists:

- `test_new_camera_after_authority_gets_no_new_test_assignments`
- `test_disjoint_samples_on_historical_camera_day_inherit_without_old_ids_in_selection`
- `test_historical_event_edge_expands_when_old_member_and_edge_are_omitted`
- `test_explicit_source_invalidation_before_first_publication_is_a_durable_tombstone`
- `test_inflight_publication_rechecks_source_tombstone_before_ready_state`
- `test_late_cross_split_link_during_upload_overlay_survives_finalize`
- `test_publish_is_retryable_without_double_count_and_deletion_keeps_manifest_history`

The omitted-edge selector verifies the added sample's component ID and split still match the historical member, and that the omitted member remains stored in `dataset_group_link_members`. The retry/deletion selector verifies late leakage registration twice creates impacts idempotently, then late deletion registration carries the deletion impact and reports both eligibility flags as false.

Migration 0012 also passed an upgrade smoke from schema versions 1–11 with legacy fixtures: two manifest link members were backfilled, a prior `dataset_invalidations` row became a source tombstone, and the same sample retained two distinct model-impact reasons after the key change. The migration resource selector confirms 0012 is discoverable in the packaged resource list.

The review was verified with focused selectors rather than the whole integration file: its shared test database would carry the intentionally global split authority between tests, so each persistence scenario was run separately on a fresh schema. The committed source SHA `1c0447b4fed71ca605a1802d6187d57e67585edb` built as `gods-mlops-ingestion:task6-1c0447b`. Running that image against a fresh PostgreSQL database applied migration 0012 and reported schema version 12, both authority tables, and `PRIMARY KEY (model_id, dataset_version, sample_id, reason)`. SHA-256 values for the image's `publish.py`, `split.py`, `deletion.py`, and migration 0012 matched the committed source files exactly. The local PostgreSQL and SeaweedFS test containers and their private database volume were removed after verification; the unique image tag is retained.
