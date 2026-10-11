# Gods MLOps integration checkpoint

This is a partial integration checkpoint, not full E2E completion.

## Verified live

- Legacy K3s was stopped with separate preservation evidence; a new two-node K3s environment was installed.
- The Ubuntu NVIDIA RTX A6000 device plugin registered one GPU; a bounded device-access smoke Job completed. This alone is not training evidence.
- Kubeflow resources were deployed and namespace/RBAC/service references corrected. Trainer webhook policy was corrected without disabling validation.
- Ingestion PostgreSQL and receiver became Ready. SeaweedFS authenticated writes and readbacks were verified.
- The resource observer was deployed on vis-lab and persisted fresh Ubuntu observations. Its image now includes a passwd entry for UID 10001 required by OpenSSH.
- Four frames were captured from the user-approved Ubuntu RTSP video-loop service. Provenance remains video-loop, not a physical-camera claim.
- Four samples were registered through the authenticated ingestion API, with durable receipts and S3 SHA-256 readback verification.
- Four bounding-box review assignments were provisioned through the quota-aware Gods workflow into Label Studio project 1, tasks 1–4; authenticated media readback matched source hashes.

## Not completed

- Human bounding-box/caption annotations have not been submitted. Source-use approval is not annotation approval.
- No immutable dataset has been published from these four samples; no corresponding RT-DETR/CLIP train/evaluation run has completed.
- Label Studio browser login remains unresolved; a chat-local manual review widget is not a substitute for Label Studio UI verification.
- Full Kubeflow/operator UI and repeated deployment/reclaim/reconnect acceptance remain incomplete.
- The automatic SeaweedFS bucket hook failed Istio initialization; buckets were created separately and verified.
- A preservation path-count regression was observed earlier; full test-suite success is not claimed.

## Preservation and cleanup

Secrets, kubeconfigs, captured frames, database volumes, model artifacts, backups, and operational receipts are excluded from Git changes. Do not remove runtime bind mounts or retained K3s/storage roots when cleaning development worktrees. The user requested a PR checkpoint and cleanup rather than continued E2E execution. Model auto-replacement remains out of scope.
