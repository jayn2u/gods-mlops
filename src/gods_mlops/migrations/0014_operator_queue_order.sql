ALTER TABLE gods_mlops_jobs
    ADD COLUMN IF NOT EXISTS queue_order BIGINT;

DO $$
BEGIN
    ALTER TABLE gods_mlops_jobs
        ADD CONSTRAINT ck_gods_mlops_jobs_queue_order_positive
        CHECK (queue_order IS NULL OR queue_order > 0);
EXCEPTION WHEN duplicate_object THEN
    NULL;
END $$;
