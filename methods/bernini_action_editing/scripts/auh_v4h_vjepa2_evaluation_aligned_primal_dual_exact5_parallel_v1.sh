#!/bin/bash -p

# DRAFT-ONLY detached controller skeleton for v4-H.  This file intentionally
# contains no executable srun path.  The first normal-entry gate below is a
# permanent NO-GO until a later, independent audit completes the implementation
# and mechanically flips every construction flag.
#
# Planned formal caller interface (not active in this draft):
#   CONTROLLER RELEASE_ROOT PYTHON_BIN FEATURE_ROOT V4A_RECEIPT V4C_RECEIPT \
#     V4D_RECEIPT V4G_RECOVERY_ATTESTATION V4G_RECOVERY_EXECUTION_RECEIPT \
#     FRESH_RUN_ROOT EXPECTED_CONTROLLER_SHA256
#
# Dynamic inner, barrier, and fold receipt SHAs are controller outputs.  They
# must never be accepted from the formal caller.

set -Eeuo pipefail
umask 077

readonly tag=v4h-vjepa2-evaluation-aligned-primal-dual-exact5-parallel-v1

# Construction gates.  All remain false in this skeleton.
readonly release_sealed=false
readonly python_pin_sealed=false
readonly controller_contract_complete=false
readonly execution_body_complete=false
readonly resource_plan_audited=false

readonly to_be_pinned=TO_BE_PINNED
readonly expected_release_manifest_sha256=TO_BE_PINNED
readonly expected_release_manifest_digest=TO_BE_PINNED
readonly expected_release_tree_sha256=TO_BE_PINNED
readonly expected_runtime_sha256=TO_BE_PINNED
readonly expected_runtime_test_sha256=TO_BE_PINNED
readonly expected_controller_execution_receipt_sha256=TO_BE_PINNED

# Stable source/runtime entities already observed locally or on AUH.  These
# remain subject to the final pre/post authority audit even though their bytes
# are not placeholders.
readonly expected_python_sha256=8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a
readonly expected_srun_sha256=2b8f60b30edf7efed35bb00864651da1b0bec68e75f942ce58b5ff82bc43cd9e
readonly expected_feature_authority_sha256=74fe161f08eb92746ce22f00b50cca65f85ef05e4a4a6a5d0c7fb85867e94233
readonly expected_v2_runtime_sha256=46927772a1861354ad5edeb2072ae9b1b505d235de7c2615fb11a6648f2bddca
readonly expected_v4a_runtime_sha256=e7e755a430b79c34fdc86f5fceaba8a9f69c66dd1e66b47c8f4115eac5265973
readonly expected_extractor_sha256=720033ac069dd1ee33463d2c439199cfdce3a1c595d4252b7f395e68c56e1cfc
readonly expected_v4c_runtime_sha256=d286c23b0626aae2161deb12a465e8614fa1462dc74f3ab9b8afd88befee1cef
readonly expected_v4d_runtime_sha256=20934925e6c9bff364e6d00996f3713c9a2b254cf3ecfe0b506f03df35e146dc
readonly expected_v4e_burned_runtime_sha256=4d8b518122a01a294d6190732da14da9614b1f041cf72c1ca69e4574b72ee96a
readonly expected_v4f_runtime_sha256=97cd77e64a4dfaf3036e6c50a5b85060fd616f87371e5d967e69db1170466d74
readonly expected_feature_receipt_sha256=895fd7e9267c82477ffc11fbc1a11fdd89b276687d87c8e82e7d85d7cf62b54a
readonly expected_v4a_receipt_sha256=568ef85d9812bcc2a771952e1806392c80f8248f5597dd32e4c95e7e1f5a3fa2
readonly expected_v4c_frontier_receipt_sha256=8b7a38d0fd9e8b789cb47b1be58a0e35615f5f4dae54df956de4103f00e5fef9
readonly expected_v4d_receipt_sha256=53910bcb71ce02a193bd47e44c3a97de0ee24f431576db64a763637447720b6f

# Scientific-design origin: the two future exact-one recovery receipts must be
# hard-pinned in the v4-H runtime, release manifest, and controller.  The
# original burned exact26 identity is recorded as a transitive join, not as a
# substitute for either recovery receipt.
readonly expected_v4g_recovery_attestation_name=recovery-attestation.json
readonly expected_v4g_recovery_attestation_schema=v4g-scientific-no-go-sibling-recovery-attestation-v1
readonly expected_v4g_recovery_attestation_sha256=TO_BE_PINNED
readonly expected_v4g_recovery_attestation_digest=TO_BE_PINNED
readonly expected_v4g_recovery_execution_name=execution-receipt.json
readonly expected_v4g_recovery_execution_schema=v4g-scientific-no-go-recovery-controller-execution-v1
readonly expected_v4g_recovery_execution_status=SEALED_SCIENTIFIC_NO_GO_ATTESTED
readonly expected_v4g_recovery_execution_sha256=TO_BE_PINNED
readonly expected_v4g_recovery_execution_digest=TO_BE_PINNED
readonly expected_v4g_burned_exact26_manifest_sha256=14bf42749c97b20934a2a088a560fa23ed2b1e37555262e9d4c7f2f368e74265
readonly expected_v4g_burned_parent_signature_sha256=0540ac21631bc948db012c77003c99d0de32cb1f769ffe38e8ab8b8e380cac76

readonly expected_inner_schema=semantic-anchor-vjepa2-evaluation-aligned-primal-dual-inner-receipt-v4h
readonly expected_fold_schema=semantic-anchor-vjepa2-evaluation-aligned-primal-dual-fold-receipt-v4h
readonly expected_barrier_schema=semantic-anchor-vjepa2-evaluation-aligned-primal-dual-global-inner-barrier-v4h
readonly expected_aggregate_schema=semantic-anchor-vjepa2-evaluation-aligned-primal-dual-exact5-receipt-v4h
readonly expected_checkpoint_schema=semantic-anchor-vjepa2-evaluation-aligned-primal-dual-checkpoint-v4h
readonly expected_trace_schema=semantic-anchor-vjepa2-primal-dual-training-trace-v4h
readonly expected_inner_pass_status=V4H_FIXED1200_INNER_PASS_OOF_UNREAD
readonly expected_inner_no_go_status=V4H_FIXED1200_INNER_NO_GO_ALL_OOF_UNREAD
readonly expected_barrier_pass_status=V4H_EXACT5_INNER_BARRIER_PASS_OOF_UNREAD
readonly expected_aggregate_status=V4H_EVALUATION_ALIGNED_PRIMAL_DUAL_KNOWN_EXPOSED_DEVELOPMENT

readonly feature_authority_relative=methods/bernini_action_editing/semantic_action_cvae_canary_v1.py
readonly v2_runtime_relative=methods/bernini_action_editing/semantic_anchor_action_sequence_vae_v2.py
readonly v4a_runtime_relative=methods/bernini_action_editing/semantic_anchor_linear_frontier_v4_fast.py
readonly extractor_relative=methods/bernini_action_editing/extract_vjepa2_ordered_contextual_features_v4c.py
readonly v4c_runtime_relative=methods/bernini_action_editing/semantic_anchor_vjepa2_analytic_frontier_v4c.py
readonly v4d_runtime_relative=methods/bernini_action_editing/semantic_anchor_vjepa2_nonlinear_temporal_codec_v4d.py
readonly v4e_burned_runtime_relative=methods/bernini_action_editing/semantic_anchor_vjepa2_multiview_global_codec_v4e_alt.py
readonly v4f_runtime_relative=methods/bernini_action_editing/semantic_anchor_vjepa2_multiview_global_codec_v4f_residual_homotopy.py
readonly runtime_relative=methods/bernini_action_editing/semantic_anchor_vjepa2_evaluation_aligned_primal_dual_v4h.py
readonly runtime_test_relative=methods/bernini_action_editing/tests/test_semantic_anchor_vjepa2_evaluation_aligned_primal_dual_v4h.py
readonly release_manifest_relative=release-manifest-v4h.json
readonly expected_release_manifest_schema=v4h-vjepa2-evaluation-aligned-primal-dual-detached-release-manifest-v1
readonly expected_release_manifest_status=V4H_DETACHED_RELEASE_MANIFEST_SEALED

# The only available allocation is exact job 143808 with four registered
# nodes.  Folds 0..3 run concurrently in wave 0; fold 4 reuses node233 only
# after wave 0 is completely joined.  Evaluation repeats the same two-wave
# mapping.  No fifth GPU/node/job is inferred or discovered at runtime.
readonly -a worker_folds=(0 1 2 3 4)
readonly -a worker_jobs=(143808 143808 143808 143808 143808)
readonly -a worker_nodes=(
  auh7-1b-gpu-233
  auh7-1b-gpu-268
  auh7-1b-gpu-292
  auh7-1b-gpu-315
  auh7-1b-gpu-233
)
readonly -a worker_waves=(0 0 0 0 1)
readonly worker_cpus=8
readonly worker_memory=12G
readonly worker_gres=gpu:mi210:1
readonly max_parallel_gpu_steps=4
readonly preflight_job=143808
readonly preflight_node=auh7-1b-gpu-233
readonly preflight_cpus=4
readonly preflight_memory=4G
readonly barrier_job=143808
readonly barrier_node=auh7-1b-gpu-233
readonly aggregate_job=143808
readonly aggregate_node=auh7-1b-gpu-233
readonly aggregate_cpus=8
readonly aggregate_memory=12G

fail() {
  builtin printf '[%s] ERROR: %s\n' "${tag}" "$*" >&2
  builtin exit 2
}

validate_declarative_skeleton() {
  [[ ${#worker_folds[@]} -eq 5 && ${#worker_jobs[@]} -eq 5 \
      && ${#worker_nodes[@]} -eq 5 && ${#worker_waves[@]} -eq 5 ]] || \
    fail "worker ledger is not exact-five"
  [[ ${worker_folds[*]} == '0 1 2 3 4' \
      && ${worker_jobs[*]} == '143808 143808 143808 143808 143808' \
      && ${worker_waves[*]} == '0 0 0 0 1' \
      && ${worker_nodes[0]} == auh7-1b-gpu-233 \
      && ${worker_nodes[1]} == auh7-1b-gpu-268 \
      && ${worker_nodes[2]} == auh7-1b-gpu-292 \
      && ${worker_nodes[3]} == auh7-1b-gpu-315 \
      && ${worker_nodes[4]} == auh7-1b-gpu-233 \
      && ${max_parallel_gpu_steps} -eq 4 ]] || \
    fail "fixed four-node/two-wave resource ledger differs"
}

draft_audit() {
  validate_declarative_skeleton
  builtin printf '%s\n' \
    '{"allfold_oof_exact0_on_any_train_inner_or_barrier_failure":true,"architecture_source":"audited_v4g_exact5_controller_state_machine","controller":"v4h-vjepa2-evaluation-aligned-primal-dual-exact5-parallel-v1","controller_contract_complete":false,"dynamic_receipt_shas_are_controller_captured":true,"execution_body_complete":false,"fixed_no_fallback":true,"intentional_no_go":true,"launch_or_remote_action_performed":false,"manifest_sha256":"TO_BE_PINNED","max_parallel_gpu_steps":4,"normal_and_optimized_test_count":40,"official_caller_supplied_inner_barrier_or_fold_sha":false,"python_pin_sealed":false,"recovery_origin":{"attestation_sha256":"TO_BE_PINNED","execution_receipt_sha256":"TO_BE_PINNED","runtime_manifest_controller_exact_join_required":true},"release_sealed":false,"release_tree_sha256":"TO_BE_PINNED","resource_plan_audited":false,"state_machine":["cpu-preflight-exact1","train-fold-wave0-exact4","train-wave0-join","train-fold-wave1-exact1","verify-inner-barrier-exact1","evaluate-fold-wave0-exact4","evaluate-wave0-join","evaluate-fold-wave1-exact1","aggregate-exact1","branch-seal-exact1"],"train_fold_outputs":["primal_dual_trace.pt","preselection.pt","fixed1200.pt","inner.json"],"worker_ledger":[{"fold":0,"job":143808,"node":"auh7-1b-gpu-233","wave":0},{"fold":1,"job":143808,"node":"auh7-1b-gpu-268","wave":0},{"fold":2,"job":143808,"node":"auh7-1b-gpu-292","wave":0},{"fold":3,"job":143808,"node":"auh7-1b-gpu-315","wave":0},{"fold":4,"job":143808,"node":"auh7-1b-gpu-233","wave":1}]}'
}

if [[ ${1:-} == --draft-audit ]]; then
  [[ $# -eq 1 ]] || fail "--draft-audit takes no additional arguments"
  draft_audit
  builtin exit 0
fi

# First normal-entry gate.  It precedes argument parsing, path resolution,
# mkdir, chmod, srun, Python import, or any other persistent/remote action.
[[ ${release_sealed} == true \
    && ${python_pin_sealed} == true \
    && ${controller_contract_complete} == true \
    && ${execution_body_complete} == true \
    && ${resource_plan_audited} == true ]] || \
  fail "INTENTIONAL NO-GO: v4-H detached controller skeleton is not sealed"

# A later implementation must port these audited v4-G modules before the
# construction gates can be flipped: exact release/same-FD authority checks,
# CPU normal/-O/compile/AST preflight, fixed-holder rechecks, clean GPU child
# gates, create-only historical ledger, bounded NFS consumer stabilization,
# exact phase parsers, fail-closed all-OOF0 branch sealing, and final retained-
# FD exact-tree sealing.  It must additionally bind the v4-H primal-dual trace
# in train/barrier/evaluate/aggregate and reverify both v4-G recovery receipts
# before and after every persistent phase.
fail "INTENTIONAL NO-GO: v4-H executable body has not been installed"
