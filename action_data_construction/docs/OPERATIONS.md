# AUH operations guide

The commands below describe the audited AUH topology. Replace paths, account
names, partitions, and resources for another cluster.

## 1. CPU build and tests

```bash
export PROJECT=/path/to/VideoActionEditing/action_data_construction
export SOURCE=/path/to/read-only/MEV
export OUTPUT=/path/to/action-data-output

cd "$PROJECT"
python -m unittest discover -s tests -v
python build_candidates.py --source-root "$SOURCE" --output-root "$OUTPUT"
python verify_source_unchanged.py \
  --build-summary "$OUTPUT/metadata/build_summary.json" \
  --events "$SOURCE/metadata/events.csv"
```

## 2. Smoke-eight audit

```bash
TAG=$(date -u +%Y%m%dT%H%M%SZ)
export MEV_AUDIT_PHASE=smoke
export MEV_AUDIT_OUTPUT_ROOT="$OUTPUT/runs/smoke_${TAG}"
sbatch --export=ALL "$PROJECT/slurm/qwen_audit_8gpu.sbatch"
```

The success gate is Slurm `COMPLETED 0:0`, exactly eight successful terminal
receipts, zero missing/error rows, and `unchanged=true` in both source preflight
and postflight receipts.

## 3. Full audit

```bash
TAG=$(date -u +%Y%m%dT%H%M%SZ)
export MEV_AUDIT_OUTPUT_ROOT="$OUTPUT/runs/full_${TAG}"

J0=$(sbatch --parsable --export=ALL,MEV_WORKER_BASE=0 \
  "$PROJECT/slurm/qwen_audit_full_shard_8gpu.sbatch")
J2=$(sbatch --parsable --export=ALL,MEV_WORKER_BASE=2 \
  "$PROJECT/slurm/qwen_audit_full_shard_8gpu.sbatch")
J4=$(sbatch --parsable --export=ALL,MEV_WORKER_BASE=4 \
  "$PROJECT/slurm/qwen_audit_full_shard_8gpu.sbatch")
J6=$(sbatch --parsable --export=ALL,MEV_WORKER_BASE=6 \
  "$PROJECT/slurm/qwen_audit_full_shard_8gpu.sbatch")
JF=$(sbatch --parsable \
  --dependency="afterok:${J0}:${J2}:${J4}:${J6}" \
  --export=ALL "$PROJECT/slurm/qwen_audit_full_finalize_v2.sbatch")

printf 'shards=%s,%s,%s,%s finalize=%s output=%s\n' \
  "$J0" "$J2" "$J4" "$J6" "$JF" "$MEV_AUDIT_OUTPUT_ROOT"
```

The AUH design uses four independent eight-MI210 allocations. Each allocation
runs two persistent four-device workers, for eight global workers. Independent
jobs avoid the single-job GPU QoS ceiling and can queue separately. The
finalizer starts only after all shards succeed.

Each pair receives a create-only terminal receipt. A failed or timed-out shard
can be resubmitted with the same output root and worker base; rows whose content
and receipts already verify are skipped.

For an incomplete preview:

```bash
python "$PROJECT/finalize_metadata_v2.py" \
  --queue "$OUTPUT/metadata/qwen_audit_queue.jsonl" \
  --audit-root "$OUTPUT/runs/full_<TAG>" \
  --annotation-semantics \
    "$OUTPUT/metadata_annotation_v2/paired_annotation_semantics.jsonl" \
  --output-root \
    "$OUTPUT/runs/full_<TAG>/preview_metadata_annotation_v2" \
  --allow-incomplete
```

All shard contracts must agree on input digest, model, code digest, schema, and
the exact 0..7 worker partition. Finalization verifies terminal closure, source
inventory, annotation identity, and annotation-authoritative metadata before
publishing the completion marker.

## 4. Failure boundaries

- Never run `chmod`, `touch`, writes, links, copy targets, or cache writes in
  the protected MEV source tree.
- Never delete an existing run. Use a new output root when prompts or configs
  change.
- Preserve original model/schema errors and raw responses; do not hand-edit
  results or receipts.
- If the source inventory changes, stop and investigate the external change.
  Do not update an old receipt merely to hide the mismatch.
