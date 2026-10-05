CREATE TABLE IF NOT EXISTS annotation_crops (
    crop_id UUID PRIMARY KEY,
    sample_id UUID NOT NULL REFERENCES ingestion_samples(sample_id),
    bbox_revision UUID NOT NULL REFERENCES annotation_revisions(annotation_revision_id),
    region_index INTEGER NOT NULL CHECK (region_index >= 0),
    region_id TEXT NOT NULL,
    object_key TEXT NOT NULL UNIQUE,
    sha256 CHAR(64) NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    object_size_bytes BIGINT NOT NULL CHECK (object_size_bytes > 0),
    state TEXT NOT NULL CHECK (state IN ('pending', 'ready', 'deleted')),
    caption_state TEXT NOT NULL DEFAULT 'needs_review'
        CHECK (caption_state IN ('needs_review', 'reviewed', 'rejected', 'cancelled')),
    caption_revision_id UUID REFERENCES annotation_revisions(annotation_revision_id),
    provenance JSONB NOT NULL,
    parent_available BOOLEAN NOT NULL DEFAULT TRUE,
    regeneration_available BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (sample_id, bbox_revision, region_index)
);
CREATE INDEX IF NOT EXISTS ix_annotation_crops_parent
    ON annotation_crops (sample_id, state);
CREATE INDEX IF NOT EXISTS ix_annotation_crops_caption
    ON annotation_crops (caption_state, state);
