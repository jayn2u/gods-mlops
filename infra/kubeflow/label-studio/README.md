# Label Studio CE review service

The `label-studio` Kustomize component deploys the immutable Label Studio Community Edition 1.23.2 image behind an internal `ClusterIP` and places its PostgreSQL tables in the existing `gods-mlops-ingestion-postgres` database. It does not add a database PVC or SQLite fallback. The labeler, review service, crop store, and retention runner share the existing database schema and `ingestion_storage_usage` byte ledger.

## Image and credentials

Build and import the Task 5 receiver/retention image, which also supplies the same-Pod media cleanup sidecar:

```sh
docker build -f infra/kubeflow/ingestion/Dockerfile -t gods-mlops-ingestion:0.3.0 .
docker save gods-mlops-ingestion:0.3.0 -o /tmp/gods-mlops-ingestion-0.3.0.tar
```

The Label Studio image is locked in `infra/versions.lock.yaml`; import the exact `heartexlabs/label-studio:1.23.2@sha256:afcc516a22775a39d0d66f4c2ddc95d01be3e3d3862b21fcf4f9de34b4ad4e12` image before applying Kustomize. Both workloads use `imagePullPolicy: Never` so an absent local image fails closed.

Create a mode-`0600` environment file outside the repository with the bundled prompt-based generator, which writes an exactly 40-character bootstrap user token (the pinned 1.23.2 auth-token column limit) and a 64-character cleanup token without printing either:

```sh
bash infra/kubeflow/label-studio/prepare-credentials.sh /tmp/gods-label-studio.env
kubectl --namespace gods-mlops create secret generic gods-label-studio-credentials \
  --from-env-file=/tmp/gods-label-studio.env
```

The script prompts for one operator username and password. The bootstrap token initializes the account only; keep the organization's legacy API-token authentication disabled. The Label Studio password and bootstrap token are never placed in the manifest, task payload, URL, or command history. Preserve the resulting Secret in the Ubuntu recovery artifact as `gods-label-studio-credentials`. The separate `gods-mlops-ingestion-credentials` Secret provides the existing PostgreSQL password.

After the first startup, sign in as that operator and create one JWT API refresh token from the account's API-token settings (the CE endpoint is `POST /api/token/`). On the pinned build this refresh token lasts 200 years; `/api/token/refresh/` returns a roughly five-minute access token and does not rotate the refresh token. Add the returned refresh token as `LABEL_STUDIO_API_TOKEN` to the same private environment file without printing it, then update the existing Secret through a file-backed apply:

```sh
kubectl --namespace gods-mlops create secret generic gods-label-studio-credentials \
  --from-env-file=/tmp/gods-label-studio.env --dry-run=client -o yaml |
  kubectl apply -f -
kubectl --namespace gods-mlops patch cronjob gods-mlops-retention \
  --type=merge -p '{"spec":{"suspend":false}}'
```

The retention job starts suspended until this JWT exists. The Gods client sends the refresh token only to `/api/token/refresh/`, then uses the returned short-lived access token with `Authorization: Bearer`. It does not persist the short-lived access token. `LABEL_STUDIO_USER_TOKEN` is used only for initial user setup and is not sent to review APIs. Do not set `LABEL_STUDIO_ENABLE_LEGACY_API_TOKEN` or enable legacy token authentication on the organization. If an operator revokes/replaces the refresh token, update the Secret from the private file and repeat the CronJob unsuspend step. The refresh token and bootstrap values are included in the same credential Secret recovery copy.

The deployment sets `DJANGO_DB=default`, `POSTGRE_NAME=gods_ingestion`, and the internal ingestion-PostgreSQL Service host/port. It sets `LABEL_STUDIO_BASE_DATA_DIR=/label-studio/data` and mounts `gods-mlops-label-studio-media` there. The Label Studio process and cleanup sidecar run as UID/GID `10001:10001` and mount the same PVC path. The sidecar's fixed root is `/label-studio/data/media/upload`.

Keep the pinned CE 1.23.2 setting `DISABLE_SIGNUP_WITHOUT_LINK=true`; the `LABEL_STUDIO_`-prefixed spelling in newer docs is not read by this pinned source. It hides the normal signup link and the view rejects account-creation POSTs without a valid invite token. No invite links are issued for the one-operator deployment. `SSRF_PROTECTION_ENABLED=true` remains enabled. `COLLECT_ANALYTICS=false` and pinned-source `LATEST_VERSION_CHECK=false` disable usage telemetry and the update lookup without changing vendor frontend code. Label Studio's security guidance documents the analytics and SSRF settings: [secure Label Studio](https://labelstud.io/guide/security.html). Its invite-only registration settings are documented in [signup configuration](https://labelstud.io/guide/signup), and PostgreSQL uses the documented `DJANGO_DB`/`POSTGRE_*` variables from [database storage setup](https://labelstud.io/guide/storedata).

## Project configurations

Create the following CE projects before routing reviews. Project IDs are deployment state and must be saved with the task/assignment mapping in PostgreSQL.

Bounding-box review:

```xml
<View>
  <Image name="image" value="$image" />
  <RectangleLabels name="bbox" toName="image">
    <Label value="person" />
  </RectangleLabels>
</View>
```

Crop caption review:

```xml
<View>
  <Image name="image" value="$image" />
  <TextArea name="caption" toName="image" rows="2"
    placeholder="Describe clothing and visible appearance" required="true" />
</View>
```

For manual relevance matrices, use one required single-choice `Choices` field with values `relevant`, `not_relevant`, and `uncertain`; keep each query/gallery hash and reviewer identity with the revision in Gods PostgreSQL. A complete matrix needs at least one positive and one negative judgment, and `uncertain` remains unresolved. Original caption pairing is not a relevance label.

Real assignments are created by the Gods workflow. It reserves the source and active-review bytes before upload, imports image bytes through Label Studio's authenticated multipart endpoint, attaches model pre-annotations only as predictions, then stores submitted human annotations as immutable revisions. Do not count a prediction as a label. Do not send API tokens in task data or URLs.

## Internal browser access and retention

Use a loopback-only port-forward for browser work:

```sh
kubectl --kubeconfig="$HOME/.kube/gods-mlops.yaml" \
  --namespace gods-mlops port-forward --address 127.0.0.1 \
  service/gods-mlops-label-studio 18082:8080
```

Then open `http://127.0.0.1:18082`. There is no ingress, NodePort, or LoadBalancer. The cleanup port `8090` is carried by the internal Service only and requires its bearer token.

Label Studio returns task paths such as `/data/upload/{project}/{file}`; with the pinned CE media backend, those bytes are stored at `{LABEL_STUDIO_BASE_DATA_DIR}/media/upload/{project}/{file}`. Deleting the task and FileUpload metadata does not guarantee physical file deletion. The colocated sidecar independently validates the project path, regular-file status, reserved size, and SHA-256 before unlinking and confirming absence. Gods releases the duplicate media bytes from the shared 1 TiB ledger only after that confirmation. The retained Label Studio volume is sized to the 100 GiB active-review exception; the active-review amount is also counted within the 1 TiB ledger.

The hourly `gods-mlops-retention` CronJob is the only scheduled expiry caller. It processes at most 100 records per run, forbids overlap, retries confirmed terminal media cleanup, deletes eligible seven-day frames and unadopted crops, and leaves stable expiry tombstones. Active assignments and dataset adoptions are protected. A reservation whose upload acknowledgment was lost and whose remote bytes cannot yet be reconciled remains charged; the job does not release ambiguous quota.

The media PVC is a retained recovery path at `/data/jayn2u/gods-mlops/metadata/label-studio-media`. Label Studio database state is covered by the existing ingestion-PostgreSQL logical backup/restore requirement. Keep the `gods-label-studio-credentials` recovery copy with the other Ubuntu-node credentials before reclaim.
