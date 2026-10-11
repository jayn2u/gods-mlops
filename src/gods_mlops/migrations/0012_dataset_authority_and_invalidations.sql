-- Persist split authority independently of an individual publication selection.
CREATE TABLE IF NOT EXISTS dataset_group_link_members (
    link_kind TEXT NOT NULL CHECK (link_kind IN ('event', 'clip')),
    link_id TEXT NOT NULL CHECK (length(link_id) BETWEEN 1 AND 255),
    sample_id UUID NOT NULL REFERENCES ingestion_samples(sample_id) ON DELETE RESTRICT,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (link_kind, link_id, sample_id)
);
CREATE INDEX IF NOT EXISTS ix_dataset_group_link_members_sample
    ON dataset_group_link_members (sample_id, link_kind, link_id);

-- Retain link authority already present in immutable manifests from earlier schema versions.
INSERT INTO dataset_group_link_members (link_kind, link_id, sample_id)
SELECT edge.link_kind, edge.value ->> 'link_id', member.sample_id::uuid
FROM dataset_versions AS version
CROSS JOIN LATERAL (
    SELECT 'event'::text AS link_kind, item.value
    FROM jsonb_array_elements(
        COALESCE(version.manifest_json #> '{split_policy,event_links}', '[]'::jsonb)
    ) AS item(value)
    UNION ALL
    SELECT 'clip'::text AS link_kind, item.value
    FROM jsonb_array_elements(
        COALESCE(version.manifest_json #> '{split_policy,clip_links}', '[]'::jsonb)
    ) AS item(value)
) AS edge
CROSS JOIN LATERAL jsonb_array_elements_text(edge.value -> 'sample_ids') AS member(sample_id)
ON CONFLICT DO NOTHING;

-- A source can be invalidated before it belongs to any published dataset.
CREATE TABLE IF NOT EXISTS dataset_source_invalidations (
    sample_id UUID PRIMARY KEY REFERENCES ingestion_samples(sample_id) ON DELETE RESTRICT,
    reason TEXT NOT NULL,
    invalidated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
INSERT INTO dataset_source_invalidations (sample_id, reason, invalidated_at)
SELECT sample_id, min(reason), min(invalidated_at)
FROM dataset_invalidations
GROUP BY sample_id
ON CONFLICT (sample_id) DO NOTHING;

-- Keep independent reasons when the same sample has multiple lineage impacts.
ALTER TABLE dataset_model_impacts DROP CONSTRAINT IF EXISTS dataset_model_impacts_pkey;
ALTER TABLE dataset_model_impacts
    ADD PRIMARY KEY (model_id, dataset_version, sample_id, reason);
