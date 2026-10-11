ALTER TABLE annotation_crops DROP CONSTRAINT IF EXISTS annotation_crops_state_check;
ALTER TABLE annotation_crops ADD CONSTRAINT annotation_crops_state_check
    CHECK (state IN ('pending', 'ready', 'purge_pending', 'deleted'));

CREATE TABLE IF NOT EXISTS relevance_matrix_revisions (
    relevance_revision_id UUID PRIMARY KEY,
    query_id TEXT NOT NULL CHECK (length(query_id) BETWEEN 1 AND 255),
    query_text TEXT NOT NULL CHECK (length(query_text) BETWEEN 1 AND 4096),
    query_revision TEXT NOT NULL CHECK (length(query_revision) BETWEEN 1 AND 255),
    query_sha256 CHAR(64) NOT NULL CHECK (query_sha256 ~ '^[0-9a-f]{64}$'),
    gallery_sha256 CHAR(64) NOT NULL CHECK (gallery_sha256 ~ '^[0-9a-f]{64}$'),
    judgments_sha256 CHAR(64) NOT NULL CHECK (judgments_sha256 ~ '^[0-9a-f]{64}$'),
    status TEXT NOT NULL CHECK (status IN ('complete', 'unresolved')),
    evaluation_eligible BOOLEAN NOT NULL,
    provenance JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK ((status = 'complete') = evaluation_eligible)
);
CREATE INDEX IF NOT EXISTS ix_relevance_query_revision
    ON relevance_matrix_revisions (query_id, query_revision, created_at DESC);

CREATE TABLE IF NOT EXISTS relevance_judgments (
    relevance_revision_id UUID NOT NULL REFERENCES relevance_matrix_revisions(relevance_revision_id),
    crop_id UUID NOT NULL REFERENCES annotation_crops(crop_id),
    crop_sha256 CHAR(64) NOT NULL CHECK (crop_sha256 ~ '^[0-9a-f]{64}$'),
    judgment TEXT NOT NULL CHECK (judgment IN ('relevant', 'not_relevant', 'uncertain')),
    PRIMARY KEY (relevance_revision_id, crop_id)
);
