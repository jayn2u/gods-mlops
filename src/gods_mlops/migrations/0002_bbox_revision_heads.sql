ALTER TABLE review_assignments DROP CONSTRAINT IF EXISTS review_assignments_state_check;
ALTER TABLE review_assignments ADD CONSTRAINT review_assignments_state_check
    CHECK (state IN ('active', 'finalized', 'cancelled', 'rejected', 'expired', 'superseded'));

CREATE TABLE IF NOT EXISTS sample_annotation_heads (
    sample_id UUID PRIMARY KEY REFERENCES ingestion_samples(sample_id),
    current_bbox_assignment_revision UUID REFERENCES review_assignments(revision),
    latest_bbox_revision UUID REFERENCES annotation_revisions(annotation_revision_id),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
