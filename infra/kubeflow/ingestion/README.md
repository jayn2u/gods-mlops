# Candidate receiver deployment

This Kustomize component adds a private FastAPI receiver and its own retained PostgreSQL volume. It uses the existing `gods-samples` SeaweedFS bucket and the chart-generated S3 admin credential. Its HTTP Service stays `ClusterIP`; the sample endpoint requires a bearer token.

Build the API image from the repository root and import both locked images into the Ubuntu node's K3s containerd before applying the platform Kustomization:

```sh
docker build -f infra/kubeflow/ingestion/Dockerfile -t gods-mlops-ingestion:0.1.0 .
docker build -f infra/kubeflow/ingestion/Dockerfile.postgres -t gods-mlops-ingestion-postgres:0.8.1-pg17-10001 .
docker save gods-mlops-ingestion:0.1.0 gods-mlops-ingestion-postgres:0.8.1-pg17-10001 -o /tmp/gods-mlops-ingestion-images.tar
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

Retention ownership remains with Task 5. The API stores a seven-day `retention_until`, keeps a stable database tombstone, and returns `410 sample_expired` after object deletion. Task 5 must extend the `selected` guard to its durable annotation and dataset adoption state before invoking `IngestionService.prune_expired`; the API does not run an automatic cleanup loop.
