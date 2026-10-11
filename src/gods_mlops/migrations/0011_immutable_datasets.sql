CREATE TABLE IF NOT EXISTS dataset_versions (
    dataset_version TEXT PRIMARY KEY CHECK (length(dataset_version) BETWEEN 1 AND 255),
    input_sha256 CHAR(64) NOT NULL UNIQUE CHECK (input_sha256 ~ '^[0-9a-f]{64}$'),
    target TEXT NOT NULL CHECK (target IN ('detr', 'clip', 'both')),
    config_version TEXT NOT NULL CHECK (length(config_version) BETWEEN 1 AND 255),
    config_sha256 CHAR(64) NOT NULL CHECK (config_sha256 ~ '^[0-9a-f]{64}$'),
    code_sha256 CHAR(64) NOT NULL CHECK (code_sha256 ~ '^[0-9a-f]{64}$'),
    state TEXT NOT NULL CHECK (state IN ('publishing', 'published', 'blocked', 'invalidated')),
    manifest_object_key TEXT NOT NULL,
    manifest_sha256 CHAR(64) NOT NULL CHECK (manifest_sha256 ~ '^[0-9a-f]{64}$'),
    manifest_size_bytes BIGINT NOT NULL CHECK (manifest_size_bytes > 0),
    manifest_json JSONB NOT NULL,
    split_counts JSONB NOT NULL,
    training_ready BOOLEAN NOT NULL,
    training_reasons JSONB NOT NULL,
    evaluation_eligible BOOLEAN NOT NULL,
    evaluation_reasons JSONB NOT NULL,
    relevance_revision_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    last_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    published_at TIMESTAMPTZ,
    invalidated_at TIMESTAMPTZ,
    CHECK (state <> 'published' OR published_at IS NOT NULL),
    CHECK ((state = 'invalidated') = (invalidated_at IS NOT NULL))
);

CREATE TABLE IF NOT EXISTS dataset_sample_splits (
    sample_id UUID PRIMARY KEY REFERENCES ingestion_samples(sample_id) ON DELETE RESTRICT,
    split TEXT NOT NULL CHECK (split IN ('train', 'validation', 'test')),
    group_id TEXT NOT NULL CHECK (length(group_id) BETWEEN 1 AND 255),
    camera_id UUID NOT NULL,
    capture_day DATE NOT NULL,
    first_dataset_version TEXT NOT NULL REFERENCES dataset_versions(dataset_version) ON DELETE RESTRICT,
    assigned_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_dataset_sample_splits_camera_day
    ON dataset_sample_splits (camera_id, capture_day, split);

CREATE TABLE IF NOT EXISTS dataset_items (
    dataset_version TEXT NOT NULL REFERENCES dataset_versions(dataset_version) ON DELETE RESTRICT,
    item_kind TEXT NOT NULL CHECK (item_kind IN ('frame', 'crop')),
    item_id UUID NOT NULL,
    sample_id UUID NOT NULL REFERENCES ingestion_samples(sample_id) ON DELETE RESTRICT,
    target TEXT NOT NULL CHECK (target IN ('detr', 'clip')),
    split TEXT NOT NULL CHECK (split IN ('train', 'validation', 'test')),
    group_id TEXT NOT NULL,
    component_id TEXT NOT NULL,
    source_object_key TEXT NOT NULL,
    object_key TEXT NOT NULL,
    source_sha256 CHAR(64) NOT NULL CHECK (source_sha256 ~ '^[0-9a-f]{64}$'),
    object_size_bytes BIGINT NOT NULL CHECK (object_size_bytes > 0),
    snapshot JSONB NOT NULL,
    PRIMARY KEY (dataset_version, item_kind, item_id)
);
CREATE INDEX IF NOT EXISTS ix_dataset_items_sample ON dataset_items (sample_id, dataset_version);

CREATE TABLE IF NOT EXISTS dataset_objects (
    dataset_version TEXT NOT NULL REFERENCES dataset_versions(dataset_version) ON DELETE RESTRICT,
    object_key TEXT NOT NULL,
    purpose TEXT NOT NULL CHECK (purpose IN ('frame', 'crop', 'config', 'manifest')),
    sha256 CHAR(64) NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    size_bytes BIGINT NOT NULL CHECK (size_bytes > 0),
    state TEXT NOT NULL CHECK (state IN ('reserved', 'verified', 'removed')),
    verified_at TIMESTAMPTZ,
    PRIMARY KEY (dataset_version, object_key),
    CHECK ((state = 'verified') = (verified_at IS NOT NULL))
);

CREATE TABLE IF NOT EXISTS dataset_publication_blocks (
    input_sha256 CHAR(64) PRIMARY KEY CHECK (input_sha256 ~ '^[0-9a-f]{64}$'),
    dataset_version TEXT NOT NULL,
    reason_codes JSONB NOT NULL,
    details JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS dataset_split_leakage_impacts (
    impact_id UUID PRIMARY KEY,
    input_sha256 CHAR(64) NOT NULL CHECK (input_sha256 ~ '^[0-9a-f]{64}$'),
    link_ids JSONB NOT NULL,
    sample_ids JSONB NOT NULL,
    splits JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (input_sha256, link_ids)
);
CREATE INDEX IF NOT EXISTS ix_dataset_leakage_input ON dataset_split_leakage_impacts (input_sha256);

CREATE TABLE IF NOT EXISTS dataset_version_leakage_impacts (
    dataset_version TEXT NOT NULL REFERENCES dataset_versions(dataset_version) ON DELETE RESTRICT,
    impact_id UUID NOT NULL REFERENCES dataset_split_leakage_impacts(impact_id) ON DELETE RESTRICT,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (dataset_version, impact_id)
);
CREATE INDEX IF NOT EXISTS ix_dataset_version_leakage_impact
    ON dataset_version_leakage_impacts (impact_id, dataset_version);

CREATE TABLE IF NOT EXISTS dataset_invalidations (
    dataset_version TEXT NOT NULL REFERENCES dataset_versions(dataset_version) ON DELETE RESTRICT,
    sample_id UUID NOT NULL REFERENCES ingestion_samples(sample_id) ON DELETE RESTRICT,
    reason TEXT NOT NULL,
    invalidated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (dataset_version, sample_id)
);

CREATE TABLE IF NOT EXISTS dataset_model_lineage (
    model_id TEXT NOT NULL CHECK (length(model_id) BETWEEN 1 AND 255),
    dataset_version TEXT NOT NULL REFERENCES dataset_versions(dataset_version) ON DELETE RESTRICT,
    registered_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (model_id, dataset_version)
);
CREATE INDEX IF NOT EXISTS ix_dataset_model_lineage_version ON dataset_model_lineage (dataset_version, model_id);

CREATE TABLE IF NOT EXISTS dataset_model_impacts (
    model_id TEXT NOT NULL,
    dataset_version TEXT NOT NULL REFERENCES dataset_versions(dataset_version) ON DELETE RESTRICT,
    sample_id UUID NOT NULL REFERENCES ingestion_samples(sample_id) ON DELETE RESTRICT,
    reason TEXT NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (model_id, dataset_version, sample_id),
    FOREIGN KEY (model_id, dataset_version)
        REFERENCES dataset_model_lineage(model_id, dataset_version) ON DELETE RESTRICT
);
