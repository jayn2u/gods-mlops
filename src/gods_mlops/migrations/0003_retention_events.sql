CREATE TABLE IF NOT EXISTS retention_policy_events (
    event_id BIGSERIAL PRIMARY KEY,
    sample_id UUID,
    action TEXT NOT NULL,
    reason TEXT NOT NULL,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_retention_policy_events_sample_created
    ON retention_policy_events (sample_id, created_at DESC);
