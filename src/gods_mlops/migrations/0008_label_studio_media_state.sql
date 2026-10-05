ALTER TABLE review_assignments ALTER COLUMN label_studio_task_id DROP NOT NULL;
ALTER TABLE review_assignments DROP CONSTRAINT IF EXISTS review_assignments_state_check;
ALTER TABLE review_assignments ADD CONSTRAINT review_assignments_state_check
    CHECK (state IN ('active', 'provisioning', 'finalized', 'cancelled', 'rejected', 'expired', 'superseded'));
DROP INDEX IF EXISTS uq_review_assignment_active_sample_stage;
CREATE UNIQUE INDEX uq_review_assignment_active_sample_stage
    ON review_assignments (sample_id, stage) WHERE state IN ('active', 'provisioning');

CREATE TABLE IF NOT EXISTS label_studio_media_uploads (
    assignment_revision UUID PRIMARY KEY REFERENCES review_assignments(revision),
    project_id BIGINT NOT NULL CHECK (project_id > 0),
    upload_filename TEXT NOT NULL,
    upload_path TEXT,
    file_upload_id BIGINT,
    task_id BIGINT,
    sha256 CHAR(64) NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    object_size_bytes BIGINT NOT NULL CHECK (object_size_bytes > 0),
    state TEXT NOT NULL CHECK (state IN ('reserved', 'uploaded', 'delete_pending', 'deleted')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_label_studio_media_cleanup
    ON label_studio_media_uploads (state, updated_at);
