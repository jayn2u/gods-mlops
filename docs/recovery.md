# Data-preserving reclaim and reconnect

The default reclaim stops only the marked Gods K3s services. It retains all nine local-PV paths, the Gods K3s data directories, credentials, and the runtime configuration needed for reconnect. It does not call a K3s uninstall script, stop Docker, delete PVCs, or remove data paths. RTSP containers and `/data/docker/volumes/rtsp-video-loop_*` remain outside this lifecycle.

The paths under `/mnt/data` on `vis-lab` and `/data` on `ubuntu` are local to their owning nodes; the workflow does not assume a shared filesystem. Backup paths for each node must be beneath its restricted recovery root: `/mnt/data/gods-mlops-recovery` on `vis-lab` and `/data/jayn2u/gods-mlops-recovery` on `ubuntu`. Keep both roots outside Git and do not configure them as PVCs.

## Review the reclaim plan

From the repository root, save a mode-`0600` dry-run plan and review its `actions`, `retained_paths`, `preserved_paths`, `preserved_services`, and `impacts` fields:

```bash
gods-mlops lifecycle plan-reclaim \
  --inventory infra/ansible/inventory.example.yml \
  --output infra/ansible/artifacts/reclaim-plan.json
```

The plan targets only `k3s-agent` on `ubuntu` and `k3s` on `vis-lab`; it preserves their K3s data directories. It also identifies the captured Istio CNI and NVIDIA device-plugin DaemonSets that must stop before K3s. Any unrecognized DaemonSet blocks reclaim before nodes are cordoned.

Before changing either node, Ansible verifies effective UID `0`, both non-secret owner markers, the service `ExecStart` data directory, source-path ownership, backup hashes, database restore dumps, and private credential copies on the node that owns each path. These pre-stop checks do not hash active database or K3s trees. A real backup artifact must exist; there is no accepted `backup_ready` field.

The Kubeflow render has known disposable `emptyDir` volumes for Istio gateway sockets/certs/data, Istiod local certs, Model Catalog perf data, and Spark operator temporary data. A second allowlist covers the exact runtime-only volume names added by an injected Istio sidecar in the `gods-mlops` and `kubeflow` namespaces. Ansible deletes only the currently scheduled pod instances whose namespace, labels, owner kind, sidecar, and volume names match those policies, after cordoning both nodes and rechecking the live Pod list. Unknown `emptyDir` use blocks the change. It then drains without `--delete-emptydir-data`, so PodDisruptionBudgets remain in force and other ephemeral data cannot be discarded as a side effect.

After draining, Ansible removes only the two captured DaemonSet objects, waits for all scheduled non-mirror pods to stop, then stops the worker before the server. It verifies that each K3s service is inactive, the K3s CRI socket is not responsive, and a full `/proc` scan finds no Kubernetes pod processes, K3s service processes, or unclassified containerd shims. Docker processes in the `moby` namespace remain outside the K3s process check. A node is not marked complete unless the cold retained-state hashes also pass.

Cold evidence covers all nine PV paths after the workloads are quiescent. Database PVs use the real logical backup and restore checks below; their raw active files are not compared to SQL dumps. The K3s server invocation in `site.yml` has no external datastore or `--cluster-init` option, so it uses the default SQLite datastore. The cold-state check records the server SQLite database directory, server token, TLS and credential directories, packaged manifests, the server and agent configuration/certificate state, and the K3s service command digest. On both nodes it excludes only the reproducible `agent/containerd` runtime/cache directory. K3s symlinks are hashed by link text without traversal; unknown special files block completion. The site invocation and default datastore behavior follow the [K3s datastore guide](https://docs.k3s.io/datastore); the K3s docs identify `server/db` and the server token as the SQLite restore state in [backup and restore guidance](https://docs.k3s.io/datastore/backup-restore).

If a node is offline, a PodDisruptionBudget blocks eviction, an unknown emptyDir appears, a runtime process remains, or an unexpected special file is found, the playbook leaves a pending state instead of reporting completion. The reclaim-state file records the plan ID and any measured cold PV and K3s hashes. A retry uses the same digest when the inventory has not changed. It does not make a disconnected node appear reclaimed.

The plan preserves and reports existing MinIO/LabCLIP paths, prior K3s roots, RTSP Docker volumes/services, `/data/jayn2u/gods-mlops-model-preparation`, and each recovery root. Those paths do not become Gods cleanup targets by name similarity.

## Prepare and verify recovery artifacts

There are nine retained PV paths. Five require a local tree backup digest: Gods objects, cache, SeaweedFS platform metadata, the vis-lab spool, and Kubeflow Pipelines SeaweedFS data. The four database PVs use source ownership checks plus logical database backup/restore checks; they do not require raw file copies while the databases are active. The K3s data directories are preserved in place and verified with the cold-state allowlist above rather than copied while K3s is running.

Run `hash-tree` on the node that owns both the source and its backup copy. The command uses SSH/Ansible to read that node's local paths, checks the root UID/GID and mode, and returns a digest without traversing symlink targets:

```bash
gods-mlops lifecycle hash-tree \
  --inventory infra/ansible/inventory.example.yml \
  --node ubuntu \
  --path /data/jayn2u/gods-mlops/objects \
  --ask-become-pass

gods-mlops lifecycle hash-tree \
  --inventory infra/ansible/inventory.example.yml \
  --node ubuntu \
  --path /data/jayn2u/gods-mlops-recovery/objects \
  --ask-become-pass
```

Run the same command for each backup path under that same node's recovery root. Record the actual `uid`, `gid`, and tree SHA-256 returned on the node. Non-database backup roots must be root-owned mode `0700`. The digest includes relative paths and regular-file bytes; symlinks contribute their link text without following the target, and special files fail. Do not copy unrelated MinIO, LabCLIP, or RTSP data into the Gods backup set.

For each PostgreSQL/MySQL PV, create a logical dump and restore it into a separate validation database. Dump that validation database with the same database tool and options, then record both dump paths and their SHA-256 digest. Both files must be root-owned mode `0600`, and the restore dump must hash to the same value as the backup dump. The verifier reads actual dump bytes and accepts no manually asserted restore status.

Store these four credential copies as root-owned mode-`0600` files under the recovery root on their owning node: `gods-k3s-token` on `vis-lab`, plus `kubeflow-dex-operator`, `gods-object-store-credentials`, and `gods-database-credentials` on `ubuntu`. The K3s token entry also names the live server token path; verification proves that its backup copy matches the actual token bytes. Credential contents never appear in reports or logs.

The manifest has `schema_version: 1`, `owner: "gods-mlops"`, and a `plan_id` matching the current dry-run plan. Each retained-path, database-restore, and credential record has an `id` and `node`. Retained-path entries include `role`, `source_path`, and `source_owner`; non-database paths also include `backup_path`, `backup_owner`, `backup_mode`, and `expected_tree_sha256`. Database records include `backup_dump_path`, `restored_dump_path`, `expected_sha256`, `expected_uid: 0`, and `expected_mode: "0600"`. Credential records include `path`, `expected_sha256`, `expected_uid: 0`, and `expected_mode: "0600"`; the K3s token record also includes its node-local `source_path`, `source_expected_uid: 0`, and `source_expected_mode: "0600"`.

Use a node-routed structural check before reclaim; its `manifest_valid` result confirms IDs and paths are assigned correctly, not that backups exist or are ready:

```bash
gods-mlops lifecycle check-retained-manifest \
  --manifest /secure/recovery/gods-mlops/manifest.json \
  --inventory infra/ansible/inventory.example.yml
```

Actual artifact verification runs remotely on each owning node. It checks source ownership, backup hashes, database restore dumps, credentials, and the real cold PV/K3s state. The manifest and slices contain paths and digests only, not secrets:

```bash
gods-mlops lifecycle verify-retained \
  --manifest /secure/recovery/gods-mlops/manifest.json \
  --inventory infra/ansible/inventory.example.yml \
  --confirm-plan <plan_id> \
  --ask-become-pass
```

The command exits nonzero for missing IDs, inaccessible node-local paths, ownership mismatch, changed bytes, incomplete restore checks, missing token/config state, or unsafe credential permissions. An absent manifest never counts as verified.

## Reclaim and reconnect

Use the exact `plan_id` printed by `plan-reclaim`. The pre-stop gate checks actual backup and restore artifacts on both nodes before workloads are drained. Ansible can prompt for normal sudo authentication; no password is stored in inventory, command text, or logs.

```bash
gods-mlops lifecycle reclaim \
  --inventory infra/ansible/inventory.example.yml \
  --manifest /secure/recovery/gods-mlops/manifest.json \
  --confirm-plan <plan_id> \
  --ask-become-pass
```

Before deleting the Istio CNI and NVIDIA DaemonSets, reclaim captures their live manifests, strips server-managed metadata/status, binds the snapshot to the plan ID, and writes `infra/ansible/artifacts/reclaim-<plan_id>-daemonsets.json` mode `0600`. Copy that snapshot to a durable restricted location outside the checkout and keep its `snapshot_sha256` with the recovery manifest. Read the digest with `jq -r '.snapshot_sha256' <snapshot-path>`; reconnect requires the original plan ID and the exact snapshot digest.

Reconnect verifies cold PV/K3s hashes and database/credential artifacts on their owning nodes before starting K3s. It uses the same retained K3s roots, restores the exact captured DaemonSet objects, waits for both rollouts, and only then uncordons nodes. It does not reapply the upstream Kubeflow example overlay or force a Helm release replacement, which could overwrite configured Dex/auth settings. Supply both the original plan ID and the exact snapshot digest:

```bash
gods-mlops lifecycle reconnect \
  --inventory infra/ansible/inventory.example.yml \
  --manifest /secure/recovery/gods-mlops/manifest.json \
  --daemonset-snapshot /secure/recovery/gods-mlops/daemonsets.json \
  --confirm-plan <plan_id> \
  --confirm-snapshot-sha256 <snapshot_sha256> \
  --ask-become-pass
```

All nine rendered PVCs remain bound to explicit retained PV names and paths. The upstream Dex example identity remains a separate runtime-authentication gate; preserving a credential file does not configure that example identity.

## Separate permanent purge

Purge is a different command. Its JSON target list must name exact absolute paths beneath a marked Gods data root, specify the root owner marker, and list paths protected from deletion. Wildcards, root deletion, nested duplicate targets, old LabCLIP/K3s paths, and RTSP paths are rejected. Every target must be a child path; the root ownership marker remains in place.

First request a dry-run digest:

```bash
gods-mlops lifecycle purge --targets /secure/recovery/gods-mlops/purge-targets.json
```

Review the exact paths, then repeat the command with the printed digest:

```bash
gods-mlops lifecycle purge \
  --targets /secure/recovery/gods-mlops/purge-targets.json \
  --confirm-targets <target_sha256> \
  --ask-become-pass
```

The purge playbook revalidates the digest, each host's root marker, and inactive K3s service state before removing only those exact target children. No purge, reclaim, or reconnect command has been run on either host as part of this static implementation.
