CREATE TABLE IF NOT EXISTS relevance_matrix_review_drafts (
    review_id UUID PRIMARY KEY,
    query_id TEXT NOT NULL CHECK (length(query_id) BETWEEN 1 AND 255),
    query_text TEXT NOT NULL CHECK (length(query_text) BETWEEN 1 AND 4096),
    query_revision TEXT NOT NULL CHECK (length(query_revision) BETWEEN 1 AND 255),
    query_source_caption_revision_id UUID REFERENCES annotation_revisions(annotation_revision_id),
    query_sha256 CHAR(64) NOT NULL CHECK (query_sha256 ~ '^[0-9a-f]{64}$'),
    gallery_sha256 CHAR(64) NOT NULL CHECK (gallery_sha256 ~ '^[0-9a-f]{64}$'),
    state TEXT NOT NULL CHECK (state IN ('pending', 'needs_review', 'frozen')),
    invalidation_reason TEXT,
    frozen_relevance_revision_id UUID REFERENCES relevance_matrix_revisions(relevance_revision_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK ((state = 'needs_review') = (invalidation_reason IS NOT NULL)),
    CHECK ((state = 'frozen') = (frozen_relevance_revision_id IS NOT NULL))
);

CREATE INDEX IF NOT EXISTS ix_relevance_review_state
    ON relevance_matrix_review_drafts (state, created_at);

CREATE INDEX IF NOT EXISTS ix_relevance_review_query_caption
    ON relevance_matrix_review_drafts (query_source_caption_revision_id)
    WHERE state = 'pending';

CREATE TABLE IF NOT EXISTS relevance_matrix_review_crops (
    review_id UUID NOT NULL REFERENCES relevance_matrix_review_drafts(review_id),
    crop_id UUID NOT NULL REFERENCES annotation_crops(crop_id),
    crop_sha256 CHAR(64) NOT NULL CHECK (crop_sha256 ~ '^[0-9a-f]{64}$'),
    PRIMARY KEY (review_id, crop_id)
);

CREATE INDEX IF NOT EXISTS ix_relevance_review_crop_dependency
    ON relevance_matrix_review_crops (crop_id, review_id);
