# Gods lifecycle evidence runbook

**Current result: NOT RUN.** This document defines how an operator can collect and verify a completed, data-preserving lifecycle record. The pytest verifier reads local evidence only. It never runs Ansible, Terraform, kubectl, SSH, Docker, a database client, a browser, or a recovery command. A case marked `verified_evidence` means its hash-bound recorded inputs are internally consistent; it does not mean pytest performed the lifecycle or that every Task 11 product gate passed.

## Current prerequisites

The old `vis-lab:k3s.service` and `ubuntu:k3s-agent.service` handoff, protected K3s data-dir/datastore inspection, normal administrator authentication, and disposition of the unmarked Ubuntu Gods path are pending. No lifecycle action is authorized by setting the test acknowledgement. Complete and separately approve that preservation handoff first. Keep LabCLIP, MinIO, the old K3s roots, RTSP containers and volumes, and all Task 8/9 evidence outside Gods cleanup targets.

The collection also needs a reviewed inventory and storage manifest, an authenticated administrator window, a source-matched Gods image set, a generated kubeconfig for the new cluster, retained metadata/object samples, node-owned recovery artifacts, and a separately validated operator login. Use the current source and locks; do not treat the example inventory, an API response, a fixture, a hash by itself, or a successful shell exit as a completed receipt.

## Collection sequence

Keep receipts in a restricted local directory outside Git. Make the directory private to the operator and every evidence file mode `0600`. Do not put passwords, tokens, DSN values, private kubeconfig contents, Secret manifests, or credential-bearing URLs into receipts. Store only account identity hashes and non-secret references. Capture native output privately, review it, and create the sanitized observation records listed below with a reference to the native output hash.

1. **Deploy.** After the old-cluster preservation/reclaim task and preflight are approved, record the source SHA, lock/render/image digests, administrator-authenticated `site.yml` result, and node/path/marker readback. Use an explicit generated kubeconfig for the cluster commands; record its context, API identity and mode, never its contents.

   ```bash
   ansible-playbook --syntax-check \
     -i "$INVENTORY" infra/ansible/site.yml
   ansible-playbook \
     -i "$INVENTORY" infra/ansible/preflight.yml
   ansible-playbook \
     -i "$INVENTORY" infra/ansible/site.yml --ask-become-pass
   kubectl --kubeconfig "$KUBECONFIG" kustomize infra/kubeflow > "$EVIDENCE_ROOT/kubeflow-render.yaml"
   kubectl --kubeconfig "$KUBECONFIG" apply -k infra/kubeflow
   terraform -chdir=infra/terraform init
   terraform -chdir=infra/terraform plan \
     -var-file=terraform.tfvars -out="$EVIDENCE_ROOT/terraform.tfplan"
   terraform -chdir=infra/terraform apply "$EVIDENCE_ROOT/terraform.tfplan"
   ```

   Capture generated-kubeconfig cluster identity, Ready node UIDs, exact Gods marker/root ownership, rendered/image identities, workload readiness, and every retained PV/PVC binding including its claim, node, local path, `Bound` phase and `Retain` policy. The Terraform plan, variables, state and native command output remain private.

2. **Reapply.** Repeat the reviewed `site.yml`, explicit-kubeconfig Kustomize apply and Terraform plan/apply recipe with the same inventory, credentials, roots and locks. Capture the before and after identity/PV/path readbacks. Compare those bindings directly; `changed=0` is not a preservation check, and harmless controller status churn is not itself a failure.

3. **Baseline.** Save the reviewed dry-run plan, recovery manifest, captured DaemonSet snapshot and selected dataset, annotation revision, object/checkpoint and job-history identities. Derive required recovery IDs from the current inventory. Collect actual read-only `verify-recovery-artifacts` reports on each owning node, the original RTSP container/image/start-time/mount snapshots, a read-only stream probe, and a fresh successful login/protected-access receipt for the preserved operator identity. The `reclaim` command repeats the native backup gate immediately before any stop action.

   ```bash
   gods-mlops lifecycle plan-reclaim \
     --inventory "$INVENTORY" --output "$EVIDENCE_ROOT/reclaim-plan.json"
   gods-mlops lifecycle check-retained-manifest \
     --manifest "$MANIFEST" --inventory "$INVENTORY"
   ansible-playbook -i "$INVENTORY" infra/ansible/verify-retained.yml \
     --extra-vars "gods_recovery_manifest=$MANIFEST" \
     --extra-vars "gods_lifecycle_inventory_file=$INVENTORY" \
     --extra-vars "gods_reclaim_plan_id=$PLAN_ID" \
     --extra-vars "gods_recovery_operation=verify-recovery-artifacts" \
     --ask-become-pass
   ```

   `check-retained-manifest` reports structural coverage only. The native per-node reports must show actual backup/source ownership, database restore and credential checks with the exact node-sliced manifest digest, check counts, and an empty failure list. Do not replace them with a `backup_ready` or `restored` boolean.

4. **Interrupt and resume reclaim.** Capture the actual interrupted controller attempt, both owning-node state files, measured systemd/runtime state, plan and snapshot identities, and the durable boundary where the controller stopped. The current CLI has no safe pause hook: do not kill services, manufacture reclaim state, or race a signal from pytest. An administrator may supervise an actual Ansible interruption at an existing durable boundary during the reviewed window. If this cannot be observed safely, mark `reclaim_interrupted` not run and do not imply that a fabricated partial state proves it.

   Resume only with the same reviewed inventory, manifest, plan and snapshot. Capture the native node records and verification reports. Both nodes must reach `service_stopped`, `runtime_status: verified_stopped`, exact required cold retained-path hashes, complete role-specific cold K3s hashes, and no unresolved retained/runtime failures. Capture task times proving the worker completed before the server stop began.

   ```bash
   gods-mlops lifecycle reclaim \
     --inventory "$INVENTORY" --manifest "$MANIFEST" \
     --confirm-plan "$PLAN_ID" --ask-become-pass
   ```

   Once both nodes have completed the cold checks, capture the native cold reports with the existing read-only verifier:

   ```bash
   gods-mlops lifecycle verify-retained \
     --manifest "$MANIFEST" --inventory "$INVENTORY" \
     --confirm-plan "$PLAN_ID" --ask-become-pass
   ```

5. **Repeat reclaim.** Run the same reviewed reclaim recipe against that completed stopped state with the same plan, manifest and snapshot. Capture both node records and cold hashes again. The receipt must show the Ansible API-free offline-resume branch and its native task trace with no cluster API requests. A stopped service plus an unreachable API does not establish this branch.

   ```bash
   gods-mlops lifecycle reclaim \
     --inventory "$INVENTORY" --manifest "$MANIFEST" \
     --confirm-plan "$PLAN_ID" --ask-become-pass
   gods-mlops lifecycle verify-retained \
     --manifest "$MANIFEST" --inventory "$INVENTORY" \
     --confirm-plan "$PLAN_ID" --ask-become-pass
   ```

6. **Reconnect.** Use the supported reconnect command with the exact plan and snapshot digest. Capture cold verification before startup, the retained roots and credential identity, Ready node/workload/PVC readback, and the live owned DaemonSet document. The verifier checks the captured objects with the existing pure DaemonSet snapshot validator. This is retained-state reconnect, not a fresh upstream overlay/auth replacement.

   ```bash
   gods-mlops lifecycle reconnect \
     --inventory "$INVENTORY" --manifest "$MANIFEST" \
     --daemonset-snapshot "$DAEMONSET_SNAPSHOT" \
     --confirm-plan "$PLAN_ID" \
     --confirm-snapshot-sha256 "$SNAPSHOT_SHA256" \
     --ask-become-pass
   ```

7. **Continuity and RTSP.** After reconnect, compare the selected immutable object hashes, dataset manifests, annotation revisions and job history with baseline. Record a canonical logical dump/restore check using distinct source and restore validation databases and distinct backup/restored dump files, followed by real read access to restored metadata. Verify a fresh login and protected page access with the preserved account identity; a credential-file hash is not a login. At before, during and after points, record the two protected RTSP containers' IDs, image IDs, running state, start times and volume mounts, plus selected stable config/media hashes and a bounded read-only stream probe. Do not require byte equality for known mutable live files.

Do not include `purge`, `terraform destroy`, K3s uninstall, forced finalizer removal, PDB bypass, firewall change, unrelated service cleanup or volume deletion in this cycle. If a case lacks native evidence, leave it incomplete or not run.

## Protected config and bundle

`GODS_MLOPS_LIFECYCLE_E2E_CONFIG` points to a current-user-owned regular JSON file with mode `0600`. Its exact keys are:

| Key | Meaning |
|---|---|
| `schema_version` | Integer `1`. |
| `fixture_id` | Canonical UUID assigned to this collection. |
| `expected_source_commit` | Full 40-character source Git SHA under review. |
| `evidence_root` | Absolute current-user-owned private directory, with no symlink path components. |
| `bundle_path` | Safe relative path to the bundle JSON under `evidence_root`. |
| `bundle_sha256` | SHA-256 of the exact bundle bytes. |
| `plan_id` | Current inventory-derived reclaim plan SHA-256. |
| `snapshot_sha256` | Existing DaemonSet snapshot SHA-256. |

The bundle has `schema_version: 1`, the same `fixture_id`, `source_commit`, `plan_id` and `snapshot_sha256`, a `provenance` object, source `identities`, `bindings`, `observations_artifact_id`, `artifacts`, and the ordered `stages` list. `identities.images` maps names to `sha256:<64 lowercase hex>` digests. `identities.input_versions` maps each selected input name to `{ "version": "...", "sha256": "<64 lowercase hex>" }`; include the locked K3s version, which must match both non-secret root marker records. `bindings` names artifact IDs for `inventory_artifact_id`, `storage_artifact_id`, `manifest_artifact_id` and `snapshot_artifact_id`.

Every `artifacts` entry has exactly `id`, `path`, `sha256` and `media_type`. IDs are unique safe labels; paths are relative to the protected root and cannot contain `.` or `..` path components. The only media types are `json`, `yaml` and `bytes`. The loader rejects symlinks, path escape, non-regular files, changed hashes, malformed documents and files not owned privately by the current user. The inventory and storage YAML files are re-read by the existing `load_inventory` implementation; the verifier independently derives `plan_id`, validates the recovery manifest coverage and validates the snapshot digest/plan binding.

Each stage receipt contains `stage`, unique UUID `attempt_id`, timezone-qualified `started_at`/`ended_at`, integer `exit_status`, `source_commit`, `plan_id`, `recipe_artifact_id`, `recipe_sha256`, and `native_output_artifact_ids`. Stages `baseline` through `rtsp_preserved` also bind `snapshot_sha256`. The `reclaim_interrupted` receipt records the observed nonzero controller exit; all other completed stage receipts use exit status zero. Timestamps are ordered and non-overlapping. The recipe artifact is a record of what the administrator ran; pytest never interprets it as a command. Every stage references both the shared observations artifact and at least one separate native output artifact. Hash and retain the original output privately; add only sanitized summaries to `observations.json`.

The observations JSON has `schema_version: 1` and a `stages` object with one entry for each stage name below:

| Stage | Required observation |
|---|---|
| `deploy` | `preflight` status/authentication/effective UID/source/render/image hashes; generated kubeconfig context, cluster UID and mode; cluster UID; both nodes' UIDs/readiness/roots/marker identities and owners; all retained PV bindings; image and input identities; workload readiness. Record the source root modes (`0755` for the data root, `0700` for K3s state), root ownership, owner-marker mode `0644`, and cluster-marker mode `0600`. |
| `reapply` | `before` and `after` deployment readbacks with the same cluster/node/root/marker/PV/image/credential identities and ready workloads. Compare state; do not require zero change count. |
| `baseline` | Per-node `verify-recovery-artifacts` report artifact IDs; selected `datasets`, `annotation_revisions`, `immutable_objects` with IDs and SHA-256 values, `job_history` IDs/statuses, and the credential identity hash. The RTSP baseline is also in `rtsp_preserved.before`. |
| `reclaim_interrupted` | `controller` with the receipt attempt UUID, `interrupt_kind: externally_supervised_ansible_abort`, observed durable boundary and `observed: true`; both nodes' native state artifact IDs; each node's measured boolean service state, runtime status and observation time. At least one state remains incomplete under the same plan/snapshot. |
| `reclaim_resume` | Both node state artifact IDs, per-node `verify-retained` report IDs, `worker_verified_at`, `server_stop_started_at`, same-plan/snapshot service-stopped state, verified runtime, exact cold path hashes, role-specific cold K3s hashes and empty failures. |
| `reclaim_repeat` | Both completed node state and per-node verification report IDs, unchanged cold hashes, `offline_retry.selected: true`, the executed task name `Require matching saved node state and snapshot before an API-free resume`, and a native trace artifact whose `selected_branch` is `offline_resume` with an empty `api_request_events` list. |
| `reconnect` | Both retained node state IDs, pre-start per-node `verify-retained` report IDs, the live DaemonSet readback artifact ID, Ready node identities, workload readiness and deployment/PV readback. |
| `continuity` | The same selected records as baseline; `logical_restore` source/restore DB IDs, distinct validation DB IDs, distinct dump paths, equal source/restored query hashes and row counts, a verified restore and read-access report artifact; `authentication` with fresh login/protected-access HTTP results, session ID, matching credential identity hash, browser receipt and screenshot artifact IDs. |
| `rtsp_preserved` | `before`, `during` and `after` snapshots keyed by the protected service names from the inventory, with stable container/image IDs, running state, start time, mount identity, selected content hashes and readable nonempty stream probes. |

Node recovery-report artifacts must be the native command JSON summary, with `status: verified`, `failures: []`, the canonical manifest-slice SHA-256 and exact check counts. For cold `verify-retained`, that digest also binds the saved cold path/K3s hashes and service-command digest that Ansible adds to the node slice. The checker calls `slice_retained_manifest_for_node` to derive expected IDs/counts; it does not assume a fixed PV count. The stopped state must include the server's SQLite state hashes or worker's agent-state hash plus the service command hash. The original RTSP config/media comparison covers only selected immutable content; runtime files that legitimately change are recorded but not equality-gated.

Set `GODS_MLOPS_LIFECYCLE_E2E_ROOT_ACK` to `<fixture_id>:<bundle_sha256>` only after reviewing the exact protected bundle. This binds the bytes for reading; it grants no infrastructure permission and starts no operation. If config or matching acknowledgement is absent, the opt-in pytest is skipped. With both present, missing or invalid artifacts and any incomplete required case fail the test. The result includes `performed_by_this_test: false`, `lifecycle_pass_claimed: false`, and separate `not_run` product/UI/model/adoption gates.

Run only the read-only checker after collection:

```bash
uv run pytest tests/e2e/test_lifecycle.py -q
```

## Sanitized report template — not run

```text
Status: NOT RUN
Source commit: pending
Training/operator image digests: pending
Input versions: pending
Fixture UUID: pending
Plan ID / DaemonSet snapshot digest: pending
Evidence bundle digest: pending
Deploy: not_run — pending approved handoff/admin access
Reapply: not_run — pending deploy evidence
Baseline: not_run — pending native node recovery reports
Interrupted reclaim: not_run — no native interrupted attempt collected
Resume reclaim: not_run — no native stopped-state evidence collected
Repeat reclaim: not_run — no native API-free retry trace collected
Reconnect: not_run — no native retained-state reconnect collected
Continuity: not_run — no metadata/auth readback collected
RTSP preservation: not_run — no before/during/after probe collected
Product browser detection/search: not_run
Kubeflow/Label Studio/operator UI: not_run
Real operating-data model runs and quality: not_run
Submodule adoption: not_run
performed_by_this_test: false
```

The private bundle may contain restricted command receipts, but the report contains only identities, digests, statuses and evidence references. Screenshots and DOM captures are native artifacts only when they are bound by the bundle hash; synthetic test screenshots never establish an operator login or product pass.
