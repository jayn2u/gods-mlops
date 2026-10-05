# Candidate receiver deployment

This Kustomize component adds a private FastAPI receiver and its own retained PostgreSQL volume. It uses the existing `gods-samples` SeaweedFS bucket and the chart-generated S3 admin credential. Its HTTP Service stays `ClusterIP`; the sample endpoint requires a bearer token.

Build the API image from the repository root and import both locked images into the Ubuntu node's K3s containerd before applying the platform Kustomization:

```sh
docker build -f infra/kubeflow/ingestion/Dockerfile -t gods-mlops-ingestion:0.3.0 .
docker build -f infra/kubeflow/ingestion/Dockerfile.postgres -t gods-mlops-ingestion-postgres:0.8.1-pg17-10001 .
docker save gods-mlops-ingestion:0.3.0 gods-mlops-ingestion-postgres:0.8.1-pg17-10001 -o /tmp/gods-mlops-ingestion-images.tar
```

Copy that tarball to Ubuntu and import it into K3s containerd with `sudo k3s ctr images import /tmp/gods-mlops-ingestion-images.tar`. Both workloads are pinned to Ubuntu, and `imagePullPolicy: Never` prevents an accidental public image lookup.

Create the `gods-mlops-ingestion-credentials` Secret in namespace `gods-mlops` with keys `DATABASE_PASSWORD`, `DATABASE_URL`, and `INGESTION_TOKEN`. Keep the source file outside the repository with mode `0600`; the password in `DATABASE_URL` must be URL encoded. The existing S3 admin credential Secret is read directly and remains covered by the existing object-store recovery credential. Add the user kubeconfig saved at `~/.kube/gods-mlops.yaml` to the lifecycle recovery artifact as `gods-ingestion-port-forward-kubeconfig` so the supervised forward can reconnect after reclaim.

Install the user unit only after an authorized kubeconfig is saved as `~/.kube/gods-mlops.yaml` with mode `0600`:

```sh
mkdir -p ~/.config/systemd/user
cp infra/kubeflow/ingestion/gods-mlops-ingestion-port-forward.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user start gods-mlops-ingestion-port-forward.service
curl --fail http://127.0.0.1:18081/readyz
```

The worker uses `compose.candidate-export.yaml` with `GW_MLOPS_CANDIDATE_EXPORT_ENABLED=true`, `GW_MLOPS_RECEIVER_URL=http://127.0.0.1:18081/api/samples`, and the same `GW_MLOPS_INGESTION_TOKEN`. Camera collection still defaults to off and must be enabled per camera through the authenticated camera API.

Before Task 3 reclaim, stop this user unit so it cannot reconnect to a stopped cluster. Start it again after reconnect and confirm `/readyz` before enabling product collection. The unit binds only to loopback; it does not add a NodePort, ingress, firewall rule, or public listener.

## Ubuntu resource observer and collection gate

The `gods-mlops-resource-observer` Deployment runs on `vis-lab` without a GPU request. Every five seconds it uses OpenSSH `BatchMode=yes` and `StrictHostKeyChecking=yes` to read the configured Ubuntu host. Its private key and pinned `known_hosts` file come from the `gods-mlops-ubuntu-observer-ssh` Secret; the deployment copies the key to a mode-0400 file in a memory-backed `emptyDir`, then mounts that directory read-only. Create the Secret from the existing approved SSH identity and matching trusted host-key entry, with the files kept outside the repository:

```sh
kubectl -n gods-mlops create secret generic gods-mlops-ubuntu-observer-ssh \
  --from-file=id_ed25519=/secure/path/to/approved-ubuntu-identity \
  --from-file=known_hosts=/secure/path/to/pinned-ubuntu-known-hosts
```

The observer asks the pinned UUID for `nvidia-smi` memory/process facts and reads the configured Ubuntu storage path's available bytes and filesystem UUID. It writes one fresh current observation into the shared PostgreSQL database. The receiver reads that durable observation before reserving a new sample. Missing, stale, failed, or mismatched observations return HTTP 503 with a retryable pause reason and `Retry-After: 5`; the worker's bounded spool keeps the candidate for retry. A received same-ID/same-hash retry returns its durable receipt before this gate, preserving lost-ack recovery.

The ConfigMap pins Ubuntu's machine fingerprint, A6000 UUID, `/data` filesystem UUID, and `/data/jayn2u/gods-mlops` as the configured collection path. Until that directory exists on the retained production filesystem and the observer reports it, new collection stays paused. The one-time `/data` probe used during Task 7 is scoped to the actual Ubuntu data filesystem; it does not establish that the Gods root has been deployed. The observer deployment and receiver config are source wiring only until an authorized K3s apply.

The receiver stores a seven-day `retention_until`, keeps a stable database tombstone, and returns `410 sample_expired` after object deletion; it still does not run an automatic cleanup loop. The hourly `gods-mlops-retention` CronJob calls the bounded `RetentionService` for frames and unadopted crops, retries terminal Label Studio media deletion, and preserves dataset adoptions, active bbox/caption assignments, and unacknowledged media reservations. See [the Label Studio runbook](../label-studio/README.md) for the 1.23.2 deployment, shared 100 GiB media volume, operator credentials, and project configuration.
