CREATE TABLE IF NOT EXISTS gods_mlops_operator_retry_intent_generations (
    scope_sha256 CHAR(64) PRIMARY KEY CHECK (scope_sha256 ~ '^[0-9a-f]{64}$'),
    generation BIGINT NOT NULL CHECK (generation >= 1),
    expires_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS gods_mlops_operator_retry_intent_expiry_idx
    ON gods_mlops_operator_retry_intent_generations (expires_at);
