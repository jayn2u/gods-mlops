CREATE TABLE IF NOT EXISTS gods_mlops_worker_artifact_deadlines (
    job_id UUID NOT NULL REFERENCES gods_mlops_jobs(job_id) ON DELETE RESTRICT,
    fencing_token BIGINT NOT NULL CHECK (fencing_token > 0),
    lease_token UUID NOT NULL,
    controller_invocation_id UUID NOT NULL,
    artifact_deadline_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (job_id, fencing_token)
);

CREATE INDEX IF NOT EXISTS ix_gods_mlops_worker_artifact_deadlines_invocation
    ON gods_mlops_worker_artifact_deadlines (job_id, controller_invocation_id, created_at, fencing_token);
