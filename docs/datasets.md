# Immutable datasets

Task 6 publishes versioned DETR and CLIP inputs from retained ingestion and annotation records. `gods_mlops.datasets.publish.publish_dataset(selection, config_version)` is the Python entry point; this task does not add a network route or training job.

The selection shape is explicit:

```json
{
  "target": "clip",
  "crop_ids": ["<crop UUID>", "<crop UUID>"],
  "caption_revisions": {
    "<crop UUID>": "<reviewed caption revision UUID>",
    "<another crop UUID>": "<reviewed caption revision UUID>"
  },
  "evaluation_gallery_crop_ids": ["<test crop UUID>", "<another test crop UUID>"],
  "event_links": [{"link_id": "event-17", "sample_ids": ["<sample UUID>", "<sample UUID>"]}],
  "clip_links": [],
  "relevance_reviews": [
    {
      "review_id": "<Task 5 review UUID>",
      "judgments": [
        {"crop_id": "<test crop UUID>", "judgment": "relevant"},
        {"crop_id": "<another test crop UUID>", "judgment": "not_relevant"}
      ],
      "provenance": {
        "source": "label_studio",
        "label_studio_annotation_id": 17,
        "reviewer_id": "<reviewer identity>"
      }
    }
  ]
}
```

For DETR, provide `target: "detr"`, finalized `sample_ids`, and a `bbox_revisions` map with the expected current bbox revision for every sample. For both model inputs, use `target: "both"` with frame IDs, crop IDs, and both revision maps. Event and clip links connect camera/date groups before partitioning. The publisher checks each expected revision against Task 5's current heads while holding the source locks. A later edit therefore yields a new version only with an updated explicit selection; retrying the old exact selection returns its original immutable manifest. Strings or identity IDs do not stand in for human relevance judgments.

Frame, crop, bbox-result, caption-result, query, gallery, and judgment hashes are read from their immutable ingestion or Task 5 records and recomputed or checked during publication. The manifest includes these hashes with the revision IDs. Callers provide source identities and expected annotation revisions; they do not provide S3 paths as pinned inputs.

The publisher validates `capture_day` against `captured_at_utc` in `Asia/Seoul`. On the first successful authority epoch, it assigns chronological camera/date groups per camera toward 60:20:20, keeping event and clip components whole and assigning shared cross-camera components consistently. If a camera cannot retain train, validation, and test partitions under those link constraints, the publication is blocked. A frame within five minutes of midnight is excluded only when that midnight meets groups assigned to different splits.

Split authority is durable across selections. The publisher stores camera/day membership and every observed event/clip edge, then expands later requests through those historical components before assigning them. A new sample on an assigned camera/day inherits that day's split even when the old sample IDs are absent from the request. Once the first authority epoch exists, genuinely new groups are assigned only to train or validation, including groups from a new camera; they never create new test assignments. A later selection can therefore have insufficient evaluation coverage and publish with `evaluation_eligible: false` while keeping training eligibility.

Existing sample assignments stay fixed. A later link across fixed splits records a leakage impact and blocks that requested publication. The impact also overlays any matching dataset version that is still publishing, so completing an upload cannot restore evaluation eligibility or change its manifest hash and split assignments.

Training readiness is separate from evaluation readiness. Missing finalized training labels, unavailable required bytes, fewer than three independent day groups per camera, an empty partition, and training-side minimums block publication. Sparse validation or test samples and missing, invalidated, or unresolved CLIP relevance truth publish the training inputs with `evaluation_eligible: false` and reason codes. A completed relevance matrix freezes through Task 5's caller-owned transaction with the publication intent. Its exact query revision, gallery crop IDs and hashes, judgments, and provenance are copied into the manifest. An uncertain matrix remains visible as unresolved and is not eligible for Task 9 evaluation.

Dataset version IDs derive from the normalized selection, config version, and dataset code hash. The same request resumes the same `publishing` version after an upload or acknowledgement failure. Source adoptions commit with the publication intent; manifest, config, and media copies reserve bytes in the shared `ingestion_storage_usage` ledger. Retries reuse those reservations. The final state changes to `published` only after source and copied object bytes match their pinned SHA-256 values and sizes. Dataset files use version-scoped object keys and include content hashes; source object paths are provenance only.

`invalidate_sample(sample_id)` first records a source-level tombstone, even before the sample belongs to a dataset. Every later publication checks its selected sources under the source locks; a tombstoned source blocks it. An invalidation during upload also prevents the pending version from finishing training-ready. Existing dataset versions and registered model lineage are invalidated while the published manifest and adoption history remain intact. Task 7/8 can register model-to-dataset lineage through `DatasetPublisher.register_model_lineage`; this seam does not create model artifacts. Late registration backfills earlier leakage and source-invalidation impacts idempotently: split leakage leaves training eligible but removes evaluation eligibility, while explicit source invalidation blocks both.

A later event/clip link that joins fixed groups across splits blocks the requested version and writes `dataset_split_leakage_impacts`. Any already-published version containing both sides keeps its manifest hash, gains a `late_cross_boundary_link` evaluation reason, and is linked through `dataset_version_leakage_impacts`; registered model lineage receives `evaluation_split_leakage` impacts. Task 9 can query the version overlay table to exclude those evaluations without changing the historical split or manifest.

The ingestion image includes the dataset package and migration resources so the same private runtime can load this contract. Configure the Python entry point with `GODS_MLOPS_DATABASE_URL`, `GODS_MLOPS_S3_ENDPOINT_URL`, `GODS_MLOPS_S3_ACCESS_KEY`, `GODS_MLOPS_S3_SECRET_KEY`, `GODS_MLOPS_S3_BUCKET`, and optional `GODS_MLOPS_S3_REGION`. Tests use only isolated `GODS_MLOPS_TEST_*` PostgreSQL and S3 endpoints.
