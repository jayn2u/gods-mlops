CREATE TABLE IF NOT EXISTS dataset_adoptions (
    adoption_id UUID PRIMARY KEY,
    dataset_version TEXT NOT NULL CHECK (length(dataset_version) BETWEEN 1 AND 255),
    target TEXT NOT NULL CHECK (target IN ('detr', 'clip')),
    sample_id UUID NOT NULL REFERENCES ingestion_samples(sample_id),
    annotation_revision_id UUID NOT NULL REFERENCES annotation_revisions(annotation_revision_id),
    crop_id UUID REFERENCES annotation_crops(crop_id),
    caption_revision_id UUID REFERENCES annotation_revisions(annotation_revision_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (
        (target = 'detr' AND crop_id IS NULL AND caption_revision_id IS NULL)
        OR (target = 'clip' AND crop_id IS NOT NULL AND caption_revision_id IS NOT NULL)
    )
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_dataset_adoptions_detr_sample
    ON dataset_adoptions (dataset_version, sample_id) WHERE target = 'detr';
CREATE UNIQUE INDEX IF NOT EXISTS uq_dataset_adoptions_clip_crop
    ON dataset_adoptions (dataset_version, crop_id) WHERE target = 'clip';
