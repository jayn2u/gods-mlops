# Reproducible training environment

The training image uses Python 3.12, stable PyTorch 2.12.0 with CUDA 13.0 wheels, Transformers 4.57.1, `pycocotools` for detector evaluation, and the pinned model revisions in `models/lock.json`. `uv.lock` captures the complete Python dependency graph and package hashes. The image build installs only from that lock and uses digest-pinned Python and uv base images.

`infra/versions.lock.yaml` records the agreed platform versions and digest-pinned owned images. The storage image is upstream SeaweedFS 4.47 from the official `ghcr.io/chrislusf/seaweedfs` publisher. The upstream project's main license is Apache-2.0; the image also contains separately licensed static assets, so retain its bundled notices. It provides the approved S3-compatible object interface, and upstream Docker instructions document keyless cosign signatures for release images. Label Studio is pinned to the 1.23.2 Community Edition image and upstream Apache-2.0 source. The recorded OCI digests are multi-architecture index digests; Docker selects the platform manifest while preserving the exact index identity.

Validate the lock formats before building:

```bash
uv run gods-mlops check-locks
uv run pytest tests/test_locks.py -q
```

Build and check the training image's model-class and evaluation imports:

```bash
docker build -f images/training/Dockerfile -t gods-mlops/training:0.1.0 .
docker run --rm --network none gods-mlops/training:0.1.0 check-training-image
docker run --rm --gpus all --network none gods-mlops/training:0.1.0 check-training-image --require-cuda
```

The image import check proves that the pinned classes and evaluation modules load. The CUDA form also requires a visible GPU. Neither check loads full model weights or proves model-specific memory use, training convergence, or a completed evaluation.

## Prepare and verify model files

Mount a persistent, writable cache at `/mnt/model-cache`. The preparation command downloads each model and its processor files at the same immutable Hugging Face commit, into a unique directory under `.staging`. It hashes every required file and publishes the revision directory with an atomic rename only after all checks pass. A failed or incomplete download is removed from staging and never becomes a ready cache entry.

```bash
docker run --rm \
  -v /data/jayn2u/gods-mlops/cache:/mnt/model-cache \
  gods-mlops/training:0.1.0 prepare-models

docker run --rm --network none \
  -v /data/jayn2u/gods-mlops/cache:/mnt/model-cache:ro \
  gods-mlops/training:0.1.0 check-models
```

Preparation requires network access to Hugging Face. `check-models` requires every locked processor/configuration and weight file to exist as a regular local file and match its expected SHA-256. A missing file, mutable revision, unsafe path, symlinked artifact, or digest mismatch fails closed. The `check-locks` command validates lock structure only; it never reports the files ready.

The small JSON/tokenizer/processor hashes were computed from files served at the immutable revisions in `models/lock.json`. Large model blobs use the SHA-256 values published in Hugging Face's LFS metadata for those same commits; the preparation command checks the actual downloaded bytes against them. This checkout contains no prepared model cache. The initial lock is not evidence that the models have been downloaded or that the A6000 can train them.

## Image pin sources

- SeaweedFS 4.47 image and signature instructions: [upstream Docker guide](https://github.com/seaweedfs/seaweedfs/blob/4.47/docker/README.md); license: [upstream LICENSE](https://github.com/seaweedfs/seaweedfs/blob/4.47/LICENSE).
- Label Studio 1.23.2 image and license: [upstream release source](https://github.com/HumanSignal/label-studio/tree/1.23.2).
- Training base images: [official Python image](https://hub.docker.com/_/python) and [Astral uv image](https://github.com/astral-sh/uv/pkgs/container/uv).
- Model revisions and published LFS hashes: [RT-DETR API](https://huggingface.co/api/models/PekingU/rtdetr_v2_r18vd?revision=5650961749fa93567c0d46fc7f43ea4f9e914107&blobs=true), [CLIP API](https://huggingface.co/api/models/openai/clip-vit-base-patch16?revision=57c216476eefef5ab752ec549e440a49ae4ae5f3&blobs=true), and [Qwen API](https://huggingface.co/api/models/Qwen/Qwen2.5-VL-7B-Instruct?revision=cc594898137f460bfe9f0759e9844b3ce807cfb5&blobs=true).

## Gods-only Kubeflow infrastructure

The infrastructure uses the K3s, Kubeflow, NVIDIA device-plugin, SeaweedFS, and Label Studio versions recorded in `infra/versions.lock.yaml`. Kubeflow is rendered from the pinned `kubeflow/community-distribution` commit. The Kustomize overlay replaces the upstream example Profile with the `gods-mlops` namespace, scopes GPU admission to that namespace, and binds all 11 rendered PVCs to explicit Gods-only local PVs. Five upstream claims use the `gods-mlops-local-retain` StorageClass; the profile database is moved into the `gods-mlops` Profile namespace. Every local PV and its StorageClass use `Retain`; the existing K3s `local-path` StorageClass is left alone.

Label Studio CE 1.23.2 is internal-only behind a `ClusterIP` Service. It uses the existing retained ingestion PostgreSQL instance/database (`DJANGO_DB=default`, `POSTGRE_NAME=gods_ingestion`) for its Django state, the `gods-label-studio-credentials` Secret for the one operator account/API token and colocated cleanup token, and a new 100 GiB retained media PVC at `/data/jayn2u/gods-mlops/metadata/label-studio-media`. Its media container and the same-Pod cleanup sidecar share the PVC at `/label-studio/data`; the sidecar targets Label Studio's actual `media/upload` directory. The app keeps SSRF protection enabled, disables public signup and analytics, and exposes no NodePort or ingress. See [the Label Studio runbook](infra/kubeflow/label-studio/README.md) for project templates, secret preparation, and local port-forward access.

The receiver image now includes the annotation and retention modules as `gods-mlops-ingestion:0.2.0`; it still does not start an autonomous expiry loop. A bounded hourly `gods-mlops-retention` CronJob invokes the `RetentionService` with a 100-record batch limit, `concurrencyPolicy: Forbid`, and the same PostgreSQL/S3 quotas used by ingestion and review. It starts suspended until the operator creates a Label Studio JWT refresh token and stores it in `gods-label-studio-credentials`. Once resumed, it retries terminal Label Studio upload cleanup before deleting expired objects/crops. Ambiguous pre-ack upload reservations remain charged until an explicit reconciliation can identify and verify their bytes.

The storage roots and initial limits are:

| Node | New path | Claim size | Purpose |
|---|---|---:|---|
| `ubuntu` | `/data/jayn2u/gods-mlops/objects` | 1 TiB | SeaweedFS objects, datasets, models, and checkpoints |
| `ubuntu` | `/data/jayn2u/gods-mlops/cache` | 200 GiB | Re-creatable model and training cache |
| `ubuntu` | `/data/jayn2u/gods-mlops/metadata/platform` | 50 GiB | SeaweedFS master and filer state |
| `ubuntu` | `/data/jayn2u/gods-mlops/metadata/kubeflow-user/gods-mlops/metadata-postgres` | 10 GiB | Kubeflow Profile metadata database |
| `ubuntu` | `/data/jayn2u/gods-mlops/metadata/kubeflow/katib-mysql` | 10 GiB | Katib database |
| `ubuntu` | `/data/jayn2u/gods-mlops/metadata/kubeflow/model-catalog-postgres` | 5 GiB | Model Catalog database |
| `ubuntu` | `/data/jayn2u/gods-mlops/metadata/kubeflow/mysql-pv-claim` | 20 GiB | Kubeflow Pipelines metadata database |
| `ubuntu` | `/data/jayn2u/gods-mlops/metadata/kubeflow/seaweedfs-pvc` | 20 GiB | Kubeflow Pipelines SeaweedFS data |
| `vis-lab` | `/mnt/data/gods-mlops-runtime/spool` | 20 GiB | Future asynchronous intake spool |

Ansible preflight reads node addresses and routes, administrator privilege, K3s service state, GPU/runtime availability, free space, the Gods root marker, and the ownership of preserved paths. It writes a mode-`0600` JSON report to `infra/ansible/artifacts/preflight.json` on the controller before reporting readiness. The remote host tasks only run read-only commands and `stat`; the path/disk helper runs inline through Python and is not copied to the host. The helper requires the Gods root to be strictly beneath its declared mount, every storage and K3s path to stay inside the root, and any existing storage directory to be a real directory owned by UID/GID `10001`. Running `preflight.yml` directly uses the non-interactive `sudo -n` readiness check and remains blocked without privilege. `site.yml` enables Ansible become for its imported preflight, so its authenticated effective UID is checked when run with `--ask-become-pass`; it starts only after the report says both nodes are ready. It rejects an active K3s service unless a root-owned Gods marker proves that the service already belongs to this deployment.

Run the syntax check and read-only preflight with the copied inventory:

```bash
ansible-playbook --syntax-check \
  -i infra/ansible/inventory.example.yml infra/ansible/site.yml
ansible-playbook \
  -i infra/ansible/inventory.example.yml infra/ansible/preflight.yml
gods-mlops lifecycle verify-privilege --ask-become-pass
```

After the separate preservation/reclaim step is complete and a deployment window is approved, run the install playbook with normal become authentication:

```bash
ansible-playbook \
  -i infra/ansible/inventory.example.yml infra/ansible/site.yml \
  --ask-become-pass
```

The inventory uses the known SSH alias `jayn2u-179-pub-vis` for `ubuntu` and `203.253.25.54:2222` for `vis-lab` when using SSH. A controller running on `vis-lab` can use a temporary local-connection inventory for read-only preflight; do not change SSH known-host data to work around a refused route. The report identifies `sudo -n` failure as a blocker and is still written when a node is unreachable. Do not bypass that gate through Docker access. The current environment has an active non-Gods K3s cluster, an unmarked `/mnt/data/gods-mlops` directory, and no non-interactive sudo. That pre-existing checkout path is preserved; new spool data uses the separate `/mnt/data/gods-mlops-runtime` root. `site.yml` is expected to stop at preflight until the old path is reviewed and the cluster preservation/reclaim task is complete. No cluster, firewall, or host data is changed by static checks.

The install playbook reads K3s `v1.36.2+k3s1` from the version lock, installs server on `vis-lab` and joins only `ubuntu`, and disables K3s Traefik, ServiceLB, and the default `local-path` provisioner. Its persistent host directories are rooted at the new Gods paths. The marker is non-secret and records the project, root, K3s data directory, and pinned version. Storage directories use UID/GID `10001`, which matches the training image. The join token comes from `GODS_K3S_TOKEN`; supply it from the approved secret store and keep it there for later recovery.

The two-node K3s flows that the existing route must permit are:

| Purpose | Protocol / port | Source | Destination / interface |
|---|---|---|---|
| K3s API | TCP 6443 | `203.253.21.179/32` (`ubuntu`) | `203.253.25.54/32` (`vis-lab`), target interface reported by `ip route get` |
| Flannel VXLAN | UDP 8472 | `203.253.21.179/32` and `203.253.25.54/32` | peer node address on each node's route-selected interface |
| Kubelet API | TCP 10250 | `203.253.25.54/32` (`vis-lab`) | `203.253.21.179/32` (`ubuntu`), target interface reported by `ip route get` |
| Kubeflow browser access | TCP 18080, loopback only | local operator process | `127.0.0.1` port-forward to the in-cluster Istio gateway |

The JSON report records each host's IPv4 addresses, the selected peer-route interface, and preserved paths. The playbooks contain no firewall tasks; the table describes the existing node-to-node flows that must be available and does not open them. The default preflight uses `sudo -n` and remains non-interactive. The separate read-only become probe can prompt for normal Ansible authentication and requires the resulting effective UID to be `0`; it does not replace the other preflight checks. Do not grant blanket passwordless sudo as a workaround. Kubeflow remains internal: the rendered distribution has no `Ingress`, `NodePort`, `LoadBalancer`, pod `hostPort`, or public Service. Use a port-forward bound to `127.0.0.1:18080` for operator access; this avoids Gods Watching's local port `8080`.

After the separate existing-cluster preservation/reclaim task is complete and preflight is ready, prepare and review the render before applying it:

```bash
kubectl kustomize infra/kubeflow > /tmp/gods-kubeflow-render.yaml
kubectl apply -k infra/kubeflow
```

Terraform owns only the SeaweedFS and NVIDIA device-plugin releases; Kustomize owns the Profile, GPU policies, RuntimeClass, StorageClass, PVs, and PVCs. Apply the Kustomize resources first so Terraform can verify the `gods-mlops` namespace and consume those claims. Terraform reads both the preflight report and the Task 1 version lock, and its preconditions fail closed if either node is not ready or the locked versions change.

Provide S3 keys in a private `infra/terraform/terraform.tfvars` file with mode `0600`; all four values are required and marked sensitive. Terraform state also contains Helm release values and can contain those credentials, so keep state private and preserve an encrypted recovery copy. `.gitignore` excludes local Terraform state, variable files, Ansible reports, and generated kubeconfig material. Do not commit them or print them in logs.

The pinned Kubeflow distribution still supplies its upstream example Dex identity. Task 2 does not create an operator-specific Dex password or verify browser login. Configure explicit operator credentials through a private deployment input before runtime acceptance; the example identity and password are not claimed to be secure defaults.

Static IaC checks are not a deployment result. The runtime K3s/Kubeflow apply, NVIDIA device allocation, S3 write/read, and Retain-volume recovery still need the separate authorized lifecycle validation after administrator access and the preservation/reclaim gate are complete.

Data-preserving reclaim, recovery-manifest verification, reconnect, and the separately confirmed purge flow are documented in [recovery.md](recovery.md). Static retention checks do not prove that the cluster has been applied or that the upstream Dex example identity is ready for operator login.
