CREATE TABLE IF NOT EXISTS gods_mlops_schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS review_storage_usage (
    singleton BOOLEAN PRIMARY KEY CHECK (singleton),
    active_bytes BIGINT NOT NULL DEFAULT 0
        CHECK (active_bytes BETWEEN 0 AND 107374182400)
);
INSERT INTO review_storage_usage (singleton, active_bytes)
VALUES (TRUE, 0)
ON CONFLICT (singleton) DO NOTHING;

CREATE TABLE IF NOT EXISTS review_assignments (
    assignment_id UUID PRIMARY KEY,
    revision UUID NOT NULL UNIQUE,
    sample_id UUID NOT NULL REFERENCES ingestion_samples(sample_id),
    stage TEXT NOT NULL CHECK (stage IN ('bbox', 'caption', 'relevance')),
    bbox_revision TEXT,
    label_studio_task_id BIGINT NOT NULL,
    media_object_key TEXT NOT NULL,
    required_bytes BIGINT NOT NULL CHECK (required_bytes > 0),
    state TEXT NOT NULL CHECK (state IN ('active', 'finalized', 'cancelled', 'rejected', 'expired')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_review_assignment_active_sample_stage
    ON review_assignments (sample_id, stage) WHERE state = 'active';
CREATE INDEX IF NOT EXISTS ix_review_assignments_sample_state
    ON review_assignments (sample_id, state);

CREATE TABLE IF NOT EXISTS annotation_revisions (
    annotation_revision_id UUID PRIMARY KEY,
    sample_id UUID NOT NULL REFERENCES ingestion_samples(sample_id),
    assignment_revision UUID NOT NULL UNIQUE REFERENCES review_assignments(revision),
    stage TEXT NOT NULL CHECK (stage IN ('bbox', 'caption', 'relevance')),
    label_studio_task_id BIGINT NOT NULL,
    label_studio_annotation_id BIGINT NOT NULL,
    result JSONB NOT NULL,
    result_sha256 CHAR(64) NOT NULL CHECK (result_sha256 ~ '^[0-9a-f]{64}$'),
    provenance JSONB NOT NULL,
    submitted_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (label_studio_task_id, label_studio_annotation_id, result_sha256)
);
