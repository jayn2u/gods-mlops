CREATE TABLE IF NOT EXISTS gods_mlops_resource_profiles (
    phase TEXT NOT NULL CHECK (phase IN ('preparation', 'training', 'evaluation', 'probe')),
    target_phase TEXT CHECK (target_phase IN ('preparation', 'training', 'evaluation')),
    model_kind TEXT NOT NULL CHECK (model_kind IN ('detr', 'clip', 'qwen')),
    config_version VARCHAR(255) NOT NULL CHECK (length(config_version) > 0),
    config_sha256 CHAR(64) NOT NULL CHECK (config_sha256 ~ '^[0-9a-f]{64}$'),
    memory_requirement_mib INTEGER NOT NULL CHECK (memory_requirement_mib BETWEEN 1 AND 49140),
    artifact_reservation_bytes BIGINT NOT NULL CHECK (artifact_reservation_bytes BETWEEN 1 AND 1099511627776),
    checkpoint_reservation_bytes BIGINT NOT NULL CHECK (checkpoint_reservation_bytes > 0),
    result_reservation_bytes BIGINT NOT NULL CHECK (result_reservation_bytes > 0),
    config_json JSONB NOT NULL,
    profile_state TEXT NOT NULL CHECK (profile_state IN ('candidate', 'measured', 'rejected')),
    measurement_id UUID,
    oom_alternatives JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (phase, model_kind, config_version),
    CHECK ((profile_state = 'measured') = (measurement_id IS NOT NULL)),
    CHECK ((phase = 'probe' AND target_phase IS NOT NULL) OR (phase <> 'probe' AND target_phase IS NULL)),
    CHECK (NOT (target_phase = 'training' AND model_kind = 'qwen')),
    CHECK (2 * checkpoint_reservation_bytes + result_reservation_bytes <= artifact_reservation_bytes)
);

CREATE TABLE IF NOT EXISTS gods_mlops_jobs (
    job_id UUID PRIMARY KEY,
    phase TEXT NOT NULL CHECK (phase IN ('preparation', 'training', 'evaluation', 'probe')),
    input_kind TEXT NOT NULL CHECK (input_kind IN ('dataset_version', 'annotation_batch', 'probe_input', 'checkpoint')),
    input_id TEXT NOT NULL CHECK (length(input_id) BETWEEN 1 AND 255),
    input_sha256 CHAR(64) NOT NULL CHECK (input_sha256 ~ '^[0-9a-f]{64}$'),
    dataset_version TEXT REFERENCES dataset_versions(dataset_version) ON DELETE RESTRICT,
    source_refs JSONB NOT NULL DEFAULT '{}'::jsonb,
    model_kind TEXT NOT NULL CHECK (model_kind IN ('detr', 'clip', 'qwen')),
    target_phase TEXT NOT NULL CHECK (target_phase IN ('preparation', 'training', 'evaluation')),
    config_version VARCHAR(255) NOT NULL CHECK (length(config_version) > 0),
    config_sha256 CHAR(64) NOT NULL CHECK (config_sha256 ~ '^[0-9a-f]{64}$'),
    profile_state_snapshot TEXT NOT NULL CHECK (profile_state_snapshot IN ('candidate', 'measured')),
    state TEXT NOT NULL CHECK (state IN (
        'queued', 'waiting_profile', 'waiting_gpu', 'waiting_storage', 'waiting_capacity',
        'running', 'yield_requested', 'retrying', 'completed', 'failed', 'cancelled'
    )),
    reason_code TEXT,
    reason_detail JSONB NOT NULL DEFAULT '{}'::jsonb,
    retryable BOOLEAN NOT NULL DEFAULT FALSE,
    rerun BOOLEAN NOT NULL DEFAULT FALSE,
    dedupe_key CHAR(64),
    parent_job_id UUID REFERENCES gods_mlops_jobs(job_id) ON DELETE RESTRICT,
    retry_root_id UUID,
    retry_job_id UUID REFERENCES gods_mlops_jobs(job_id) ON DELETE RESTRICT,
    oom_retries SMALLINT NOT NULL DEFAULT 0 CHECK (oom_retries BETWEEN 0 AND 2),
    communication_retries SMALLINT NOT NULL DEFAULT 0 CHECK (communication_retries BETWEEN 0 AND 3),
    next_retry_at TIMESTAMPTZ,
    lease_token UUID,
    lease_generation BIGINT NOT NULL DEFAULT 0 CHECK (lease_generation >= 0),
    lease_expires_at TIMESTAMPTZ,
    owner_pid INTEGER,
    owner_start_ticks BIGINT,
    owner_uid INTEGER,
    artifact_reservation_bytes BIGINT NOT NULL CHECK (artifact_reservation_bytes BETWEEN 1 AND 1099511627776),
    checkpoint_uri TEXT,
    checkpoint_sha256 CHAR(64),
    checkpoint_identity JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    CHECK ((lease_token IS NULL) = (lease_expires_at IS NULL)),
    CHECK ((owner_pid IS NULL) = (owner_start_ticks IS NULL)),
    CHECK ((checkpoint_uri IS NULL) = (checkpoint_sha256 IS NULL)),
    CHECK ((checkpoint_sha256 IS NULL) = (checkpoint_identity IS NULL)),
    CHECK (dedupe_key IS NULL OR dedupe_key ~ '^[0-9a-f]{64}$')
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_gods_mlops_jobs_dedupe_key
    ON gods_mlops_jobs (dedupe_key) WHERE dedupe_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_gods_mlops_jobs_queue
    ON gods_mlops_jobs (state, created_at, job_id);

CREATE TABLE IF NOT EXISTS gods_mlops_job_events (
    event_id BIGSERIAL PRIMARY KEY,
    job_id UUID NOT NULL REFERENCES gods_mlops_jobs(job_id) ON DELETE RESTRICT,
    event_type TEXT NOT NULL,
    state TEXT NOT NULL,
    reason_code TEXT,
    observation_id UUID,
    fencing_token BIGINT,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_gods_mlops_job_events_job
    ON gods_mlops_job_events (job_id, event_id);

CREATE TABLE IF NOT EXISTS gods_mlops_gpu_observation_current (
    node_id TEXT PRIMARY KEY,
    observation_id UUID NOT NULL UNIQUE,
    observed_at TIMESTAMPTZ NOT NULL,
    received_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    observation JSONB,
    failure_code TEXT,
    failure_count INTEGER NOT NULL DEFAULT 0 CHECK (failure_count >= 0),
    idle_since TIMESTAMPTZ,
    idle_observation_count INTEGER NOT NULL DEFAULT 0 CHECK (idle_observation_count >= 0),
    last_observed_at TIMESTAMPTZ,
    CHECK ((observation IS NULL) <> (failure_code IS NULL))
);

CREATE TABLE IF NOT EXISTS gods_mlops_gpu_leases (
    gpu_uuid TEXT PRIMARY KEY,
    job_id UUID NOT NULL UNIQUE REFERENCES gods_mlops_jobs(job_id) ON DELETE RESTRICT,
    lease_token UUID NOT NULL UNIQUE,
    fencing_token BIGINT NOT NULL CHECK (fencing_token > 0),
    memory_requirement_mib INTEGER NOT NULL CHECK (memory_requirement_mib BETWEEN 1 AND 49140),
    expires_at TIMESTAMPTZ NOT NULL,
    owner_pid INTEGER,
    owner_start_ticks BIGINT,
    owner_uid INTEGER,
    granted_observation_id UUID NOT NULL,
    yield_reason TEXT,
    granted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK ((owner_pid IS NULL) = (owner_start_ticks IS NULL))
);

CREATE TABLE IF NOT EXISTS gods_mlops_artifact_reservations (
    job_id UUID PRIMARY KEY REFERENCES gods_mlops_jobs(job_id) ON DELETE RESTRICT,
    config_version VARCHAR(255) NOT NULL,
    reserved_bytes BIGINT NOT NULL CHECK (reserved_bytes BETWEEN 1 AND 1099511627776),
    consumed_bytes BIGINT NOT NULL DEFAULT 0 CHECK (consumed_bytes >= 0 AND consumed_bytes <= reserved_bytes),
    state TEXT NOT NULL CHECK (state IN ('reserved', 'settled', 'released')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    settled_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS gods_mlops_profile_measurements (
    measurement_id UUID PRIMARY KEY,
    job_id UUID NOT NULL UNIQUE REFERENCES gods_mlops_jobs(job_id) ON DELETE RESTRICT,
    input_sha256 CHAR(64) NOT NULL CHECK (input_sha256 ~ '^[0-9a-f]{64}$'),
    config_sha256 CHAR(64) NOT NULL CHECK (config_sha256 ~ '^[0-9a-f]{64}$'),
    model_kind TEXT NOT NULL CHECK (model_kind IN ('detr', 'clip', 'qwen')),
    target_phase TEXT NOT NULL CHECK (target_phase IN ('preparation', 'training', 'evaluation')),
    result_state TEXT NOT NULL CHECK (result_state IN ('succeeded', 'failed')),
    peak_allocated_mib INTEGER CHECK (peak_allocated_mib > 0),
    peak_reserved_mib INTEGER CHECK (peak_reserved_mib > 0),
    optimizer_steps INTEGER NOT NULL DEFAULT 0 CHECK (optimizer_steps >= 0),
    inference_steps INTEGER NOT NULL DEFAULT 0 CHECK (inference_steps >= 0),
    checkpoint_resumed BOOLEAN NOT NULL DEFAULT FALSE,
    verification_details JSONB NOT NULL DEFAULT '{}'::jsonb,
    checkpoint_sha256 CHAR(64),
    exit_code INTEGER NOT NULL,
    measured_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (
        (result_state = 'succeeded' AND target_phase = 'training' AND peak_allocated_mib IS NOT NULL
         AND peak_reserved_mib IS NOT NULL AND checkpoint_sha256 IS NOT NULL
         AND checkpoint_resumed = TRUE AND optimizer_steps >= 3 AND exit_code = 0
         AND model_kind IN ('detr', 'clip'))
        OR (result_state = 'succeeded' AND target_phase IN ('preparation', 'evaluation')
            AND peak_allocated_mib IS NOT NULL
            AND peak_reserved_mib IS NOT NULL AND inference_steps >= 1 AND exit_code = 0
            AND model_kind IN ('detr', 'clip', 'qwen'))
        OR result_state = 'failed'
    )
);
