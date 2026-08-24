#!/usr/bin/env python3
"""R8 exact60 specialization of the frozen source-bound DINO raw diagnostic.

This diagnostic accepts only the completed Job 135056 r8 bank.  Before it
seals an input manifest, and again whenever that manifest is consumed, it
replays the terminal-evidence -> master/deep-audit -> exact60 artifact
binding.  The evidence grants decoded diagnostic input authority only; every
scientific, selection, training, and optimizer authority remains false.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


_BASE_BASENAME = "diagnose_saic_partial47_source_bound_dinov2_raw_v1.py"
_BASE_SHA256 = "ffbc9ba149d1ddadf704dd8258678a8893235e328da4c7601e98d63ba37aa7a2"
_BASE_PATH = Path(__file__).resolve().with_name(_BASE_BASENAME)
if not _BASE_PATH.is_file() or _BASE_PATH.is_symlink():
    raise RuntimeError("pinned exact47 source-bound evaluator is absent or not a plain file")
if hashlib.sha256(_BASE_PATH.read_bytes()).hexdigest() != _BASE_SHA256:
    raise RuntimeError("pinned exact47 source-bound evaluator SHA-256 differs")

import diagnose_saic_partial47_source_bound_dinov2_raw_v1 as core  # noqa: E402


SCHEMA_VERSION = "bernini-saic-r8-exact60-source-bound-dinov2-raw-v1"
INPUT_SCHEMA = f"{SCHEMA_VERSION}-input"
SHARD_SCHEMA = f"{SCHEMA_VERSION}-shard"
AGGREGATE_SCHEMA = f"{SCHEMA_VERSION}-aggregate"
PREFLIGHT_SCHEMA = f"{SCHEMA_VERSION}-preflight"
EXPECTED_ATTEMPT_COUNT = 60
EXPECTED_WORLD_SIZE = 8
EXPECTED_PARTITION_SIZES = (8, 8, 8, 8, 7, 7, 7, 7)

EXPECTED_RUN_ROOT = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_saic_v1_20260809/runs/"
    "t2v-events-topup-r8-ddc8a79-r1"
)
EXPECTED_ATTEMPTS_ROOT = f"{EXPECTED_RUN_ROOT}/attempts"
EXPECTED_ROOT_SPEC_PATH = f"{EXPECTED_RUN_ROOT}/sealed-saic-t2v-event-topup-v2-spec.json"
EXPECTED_ROOT_SPEC_SHA256 = "d693d0784530f007888e2825d15db3db808fdf4f1d111b5d080d968c894ff145"
EXPECTED_ROOT_SPEC_CONTENT_SHA256 = "af2dfc387a96ade19518c5bb5313d9485683510cdbd80a4f63b1cb0746683065"
EXPECTED_BASE_SPEC_PATH = f"{EXPECTED_RUN_ROOT}/sealed-base-saic-t2v-event-v1-spec.json"
EXPECTED_BASE_SPEC_SHA256 = "623a7ed8a2ce2d327247c541b59aa2d39f1fbfe4a480f7351d042c7ef7a47927"
EXPECTED_BASE_SPEC_CONTENT_SHA256 = "3920d5c121b75c6bbf984c24440c9773dfb49006778c61a671ae50963bb5456a"
EXPECTED_SOURCE_MANIFEST_PATH = f"{EXPECTED_RUN_ROOT}/sealed-saic-source-manifest.json"
EXPECTED_SOURCE_MANIFEST_SHA256 = "899b5a1dd66fc0bf6d4d0192fb6157f4afe691c50633246dddcaa1db2c2a98a9"
EXPECTED_SOURCE_MANIFEST_CONTENT_SHA256 = "9c2a3d6841951ea0ed050dc230630a1176460e25a979ec199eab575ad22f3c6f"
EXPECTED_JOB_ID = "135056"
EXPECTED_SOURCE_REVISION = "ddc8a79199aed1391cf089f51835c2bbfa74ae28"
EXPECTED_SOURCE_ARCHIVE_SHA256 = (
    "4038100b86655e5ea3e9a32432dc619c4b8d1a5d7859703c4cf06b77de0b934b"
)
EXPECTED_TERMINAL_EVIDENCE_PATH = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_saic_v1_20260809/releases/"
    "saic-t2v-topup-r8-ddc8a79-r1/evidence-terminal-ddc8a79/"
    "saic-exact60-terminal-evidence-135056.json"
)
EXPECTED_MASTER_RECEIPT_PATH = (
    f"{EXPECTED_RUN_ROOT}/saic-pure-t2v-event-bank-topup-receipt.json"
)
TERMINAL_EVIDENCE_SCHEMA = "saic-t2v-full60-terminal-evidence-v1"
TERMINAL_EVIDENCE_STATUS = (
    "terminal_technical_full60_complete_pending_detached_semantic_review"
)
MASTER_SCHEMA = "bernini-saic-pure-t2v-event-bank-topup-receipt-v2"
DEEP_AUDIT_SCHEMA = "saic-t2v-live-shard-prefix-audit-v1"
EXPECTED_TERMINAL_AUTHORITY = {
    "detached_decoded_event_review_input": True,
    "data_selection": False,
    "human_review": False,
    "optimizer": False,
    "scientific_action_editing_success_claim": False,
    "training": False,
    "training_target_admission": False,
}
EXPECTED_DEEP_AUDIT_AUTHORITY = {
    "detached_decoded_event_review_input": False,
    "merge_or_partial_reuse": False,
    "scientific_selection": False,
    "training": False,
    "optimizer": False,
}
DECODE_EVIDENCE_FIELDS = frozenset({
    "artifact_sha256", "decoded_rgb_sha256", "frame_count",
    "fps_numerator", "fps_denominator", "time_base_numerator",
    "time_base_denominator", "pts_step", "pts_sha256", "width",
    "height", "selected_frame_indices", "selected_rgb_sha256",
    "preprocessed_tensor_sha256",
})
FEATURE_EVIDENCE_FIELDS = frozenset({
    "global_feature_sha256", "dense_feature_sha256", "selected_frame_count",
    "dense_grid_height", "dense_grid_width", "feature_dimension",
})
RESULT_FIELDS = frozenset({
    "candidate_id", "candidate_binding", "candidate_decode",
    "candidate_features", "correct_source_evidence", "wrong_source_evidence",
    "raw_metrics", "authority",
})
RAW_METRIC_FIELDS = frozenset({
    "measurement_label", "global_candidate_correct", "global_candidate_wrong",
    "global_correct_minus_wrong_margin", "global_source_self_upper_bound",
    "dense_candidate_correct", "dense_candidate_wrong",
    "dense_correct_minus_wrong_margin", "dense_source_self_upper_bound",
    "thresholds", "absolute_preservation_authority", "identity_authority",
    "event_authority", "scientific_claim_authorized", "ranking_authorized",
    "selection_authorized", "training_target_authorized",
})
RAW_METRIC_AUTHORITY = {
    "absolute_preservation_authority": False,
    "identity_authority": False,
    "event_authority": False,
    "scientific_claim_authorized": False,
    "ranking_authorized": False,
    "selection_authorized": False,
    "training_target_authorized": False,
}
EXPECTED_EXPERIMENT_ROOT = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_pair_v5_source_bound_preservation_v1_20260808"
)
EXPECTED_CHECKPOINT_ROOT = f"{EXPECTED_EXPERIMENT_ROOT}/vendor/dinov2-base-f9e44c8"
EXPECTED_CHECKPOINT_MANIFEST_PATH = f"{EXPECTED_EXPERIMENT_ROOT}/inputs/dinov2-base-f9e44c8.sha256"
EXPECTED_CHECKPOINT_MANIFEST_SHA256 = (
    "b61f251411f0d8f6a617b67d0b903c333d16c77fb6b3f49507225884d4aed0ea"
)
EXPECTED_EVALUATOR_SPEC_PATH = f"{EXPECTED_EXPERIMENT_ROOT}/inputs/pair_v5_source_bound_preservation_evaluator_7c4c837_v1.json"
EXPECTED_VISUAL_SOURCE_ROOT = f"{EXPECTED_EXPERIMENT_ROOT}/inputs/source-preservation-7c4c837-minimal/methods/bernini_action_editing"
EXPECTED_VISUAL_SCORER_PATH = f"{EXPECTED_VISUAL_SOURCE_ROOT}/score_pair_v5_source_bound_preservation_v1.py"
EXPECTED_VISUAL_CONTRACT_PATH = f"{EXPECTED_VISUAL_SOURCE_ROOT}/pair_v5_source_bound_preservation_evaluator_v1.py"
EXPECTED_VISUAL_EVALUATOR_OBJECT_SHA256 = (
    "6a9232bdb17703747c76cd6eb9a5e7c92aa4fbcb4a0e85e77bd3cd960230dbaa"
)
EXPECTED_VISUAL_EVALUATOR = {
    "root": EXPECTED_CHECKPOINT_ROOT,
    "adapter_id": "hf-dinov2-last-hidden-state-square-patch-grid-v1",
    "architecture_id": "dinov2",
    "checkpoint_manifest_sha256": EXPECTED_CHECKPOINT_MANIFEST_SHA256,
    "checkpoint_config_sha256": (
        "f7ff4cfa73d2f70647dbf6950541ad25d73082d54c2e7e9bded160c7656b2a70"
    ),
    "preprocessor_config_sha256": (
        "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828"
    ),
    "checkpoint_file_count": 3,
    "verified_entries_digest": (
        "fb0c903b74ee42e14023585f2f45a1e3582e3e2ed8da0cdf622b734588445ae9"
    ),
    "every_checkpoint_file_verified": True,
    "preprocessor_golden_input_sha256": (
        "d8217ce3a86de051a4affd701c965befd12584cce51902c9f266fff952ebd18a"
    ),
    "preprocessor_golden_output_sha256": (
        "b5ef31a8754b854ce64dcf49a79949e22ff9219a7db5d2dfd5fec1ed0602fb6a"
    ),
    "preprocessor_golden_output_shape": [1, 3, 224, 224],
    "evaluator_spec_sha256": (
        "6b18b9bc10589325ee2c09af339ef43a3eff507bcc754a2a6984cb70f0afd736"
    ),
    "runtime_versions": {
        "python_version": "3.12.13",
        "torch_version": "2.7.1+rocm6.3",
        "torch_hip_version": "6.3.42131-fa1d09cbd",
        "transformers_version": "4.53.2",
        "safetensors_version": "0.8.0rc0",
        "av_version": "13.1.0",
        "numpy_version": "1.26.4",
        "pillow_version": "11.3.0",
    },
    "loading_counts": {
        "missing_key_count": 0,
        "unexpected_key_count": 0,
        "mismatched_key_count": 0,
        "loading_error_count": 0,
    },
    "frozen_eval": True,
    "trainable_parameter_tensors": 0,
    "identity_authority": False,
    "scientific_claim_authorized": False,
}

SOURCE_VIDEO_PREFIX = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/goku_action_wan22_20260730T043022Z/"
    "fullmotion_next1000_v17_20260803T133300Z/wan_next1000_v17/samples"
)
EXPECTED_SOURCE_SHA256_BY_IID = {
    "311c82f83eca4a7f": "cc329d61942f02e9e58c43e455e95bf892dcf8b8cd5c2785c3fff1e2e6fbd798",
    "31c34509415745ca": "2c34ec7f74faa624909f966d3d98f485f4b92250905843e2a7dd67a44d24fda6",
    "6d346c38cf504493": "99483c3d0dacb86658841df54452c0a3492563f197813c4a2526e4f096dba14f",
    "6ea45d35943742bb": "a4f0567c8c7a63421a34099bd16c13f4ca77dd4bbe9b0ce3725a363e96b6d193",
    "7b88a1ca1f804f41": "4d0c5cdfa9e0aae394af34a5bdda7de82ac770cd62cddbf3173ad2378458f3ed",
    "841b5e0080a1441d": "5f354b6b0f5cf49bf14d57a359bad03e90263d1a3965a57b1b89ce1a707f492a",
    "99cde432839f4240": "4fcd6d75b09e3b294d3dfff15fc7b523a536551e592484eed91de66e2e733a2c",
    "a35b590961d24694": "6e9381d3889437f618e1ec6b694703b10598c4b42d8b361b0442db7780be97ed",
}
EXPECTED_ACTOR_BY_IID = {
    "311c82f83eca4a7f": "human", "31c34509415745ca": "human",
    "6d346c38cf504493": "human", "a35b590961d24694": "human",
    "6ea45d35943742bb": "dog", "7b88a1ca1f804f41": "dog",
    "841b5e0080a1441d": "dog", "99cde432839f4240": "dog",
}
EXPECTED_WRONG_IID_BY_IID = {
    "311c82f83eca4a7f": "31c34509415745ca",
    "31c34509415745ca": "6d346c38cf504493",
    "6d346c38cf504493": "a35b590961d24694",
    "a35b590961d24694": "311c82f83eca4a7f",
    "6ea45d35943742bb": "7b88a1ca1f804f41",
    "7b88a1ca1f804f41": "841b5e0080a1441d",
    "841b5e0080a1441d": "99cde432839f4240",
    "99cde432839f4240": "6ea45d35943742bb",
}

AUTHORITY_CLOSURE = dict(core.AUTHORITY_CLOSURE)
SourceBoundRaw60Error = core.SourceBoundRawError
_base_build_manifest = core.build_manifest
_base_load_input_manifest = core.load_input_manifest
_base_worker_common = core._worker_common
_base_build_parser = core.build_parser


def _configure_core() -> None:
    core.__file__ = __file__
    core.core.__file__ = __file__
    core.SCHEMA_VERSION = SCHEMA_VERSION
    core.INPUT_SCHEMA = INPUT_SCHEMA
    core.SHARD_SCHEMA = SHARD_SCHEMA
    core.AGGREGATE_SCHEMA = AGGREGATE_SCHEMA
    core.PREFLIGHT_SCHEMA = PREFLIGHT_SCHEMA
    core.EXPECTED_ATTEMPT_COUNT = EXPECTED_ATTEMPT_COUNT
    core.EXPECTED_WORLD_SIZE = EXPECTED_WORLD_SIZE
    core.EXPECTED_PARTITION_SIZES = EXPECTED_PARTITION_SIZES
    core.AUTHORITY_CLOSURE = AUTHORITY_CLOSURE
    core.core.SCHEMA_VERSION = SCHEMA_VERSION
    core.core.INPUT_SCHEMA = INPUT_SCHEMA
    core.core.SHARD_SCHEMA = SHARD_SCHEMA
    core.core.AGGREGATE_SCHEMA = AGGREGATE_SCHEMA
    core.core.PREFLIGHT_SCHEMA = PREFLIGHT_SCHEMA
    core.core.EXPECTED_ATTEMPT_COUNT = EXPECTED_ATTEMPT_COUNT
    core.core.EXPECTED_WORLD_SIZE = EXPECTED_WORLD_SIZE
    core.core.AUTHORITY_CLOSURE = AUTHORITY_CLOSURE
    core.core.partition_indices = partition_indices

def partition_indices(count: int, rank: int, world_size: int) -> tuple[int, ...]:
    if count != EXPECTED_ATTEMPT_COUNT or world_size != EXPECTED_WORLD_SIZE:
        raise SourceBoundRaw60Error("r8 exact60 source-bound partition geometry differs")
    core.core._rank(rank, world_size=world_size)
    indices = tuple(index for index in range(count) if index % world_size == rank)
    sizes = tuple(
        len(tuple(index for index in range(count) if index % world_size == item))
        for item in range(world_size)
    )
    if sizes != EXPECTED_PARTITION_SIZES:
        raise SourceBoundRaw60Error("r8 exact60 source-bound partition sizes differ")
    return indices


def _configure_partitions() -> None:
    core._base_partition_indices = partition_indices
    core.partition_indices = partition_indices
    core.core.partition_indices = partition_indices


_configure_partitions()


def _install_build_parser() -> None:
    def build_parser() -> Any:
        parser = _base_build_parser()
        subparser_actions = [
            action for action in parser._actions
            if hasattr(action, "choices")
            and isinstance(action.choices, Mapping)
            and "build-manifest" in action.choices
        ]
        if len(subparser_actions) != 1:
            raise SourceBoundRaw60Error("r8 build parser subcommand closure differs")
        build = subparser_actions[0].choices["build-manifest"]
        if "--terminal-evidence" in build._option_string_actions:
            raise SourceBoundRaw60Error("r8 terminal-evidence parser option was preinstalled")
        build.add_argument("--terminal-evidence", required=True)
        return parser

    core.build_parser = build_parser


def _source_path(iid: str) -> str:
    return f"{SOURCE_VIDEO_PREFIX}/{iid}/samples/{iid}/source_video.mp4"


def _receipt_binding_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    bindings = []
    for row in rows:
        candidate_id = row.get("candidate_id")
        receipt_sha256 = row.get("receipt_sha256")
        if not isinstance(candidate_id, str) or not isinstance(receipt_sha256, str):
            raise SourceBoundRaw60Error("r8 receipt binding fields differ")
        bindings.append({"candidate_id": candidate_id, "receipt_sha256": receipt_sha256})
    bindings.sort(key=lambda item: item["candidate_id"])
    return core.core.object_sha256(bindings)


def _load_canonical_receipt(path_value: str | Path, *, label: str) -> tuple[dict[str, Any], str]:
    value, raw_sha = core.core._strict_json(
        path_value, expected_sha256=None, label=label,
    )
    unsigned = dict(value)
    claimed = unsigned.pop("receipt_digest", None)
    if claimed != core.core.object_sha256(unsigned):
        raise SourceBoundRaw60Error(f"{label} canonical receipt digest differs")
    return value, raw_sha


def _validate_terminal_evidence(path_value: str | Path) -> dict[str, Any]:
    if str(Path(path_value)) != EXPECTED_TERMINAL_EVIDENCE_PATH:
        raise SourceBoundRaw60Error("r8 terminal-evidence lexical path differs")
    terminal, terminal_sha = _load_canonical_receipt(
        path_value, label="r8 exact60 terminal evidence",
    )
    authority = terminal.get("authority")
    if (
        terminal.get("schema_version") != TERMINAL_EVIDENCE_SCHEMA
        or terminal.get("status") != TERMINAL_EVIDENCE_STATUS
        or terminal.get("job_id") != EXPECTED_JOB_ID
        or terminal.get("root") != EXPECTED_RUN_ROOT
        or terminal.get("candidate_count") != EXPECTED_ATTEMPT_COUNT
        or terminal.get("seed_cell_count") != 20
        or terminal.get("unique_mp4_sha256_count") != EXPECTED_ATTEMPT_COUNT
        or terminal.get("source_revision") != EXPECTED_SOURCE_REVISION
        or terminal.get("source_archive_sha256") != EXPECTED_SOURCE_ARCHIVE_SHA256
        or terminal.get("root_spec_raw_sha256") != EXPECTED_ROOT_SPEC_SHA256
        or authority != EXPECTED_TERMINAL_AUTHORITY
    ):
        raise SourceBoundRaw60Error("r8 terminal technical evidence differs")

    slurm = terminal.get("slurm_terminal_observation")
    parsed = slurm.get("parsed_row") if isinstance(slurm, Mapping) else None
    if (
        not isinstance(parsed, Mapping)
        or parsed.get("JobIDRaw") != EXPECTED_JOB_ID
        or parsed.get("State") != "COMPLETED"
        or parsed.get("ExitCode") != "0:0"
        or parsed.get("AllocNodes") != "1"
        or "gres/gpu:mi210=8" not in str(parsed.get("AllocTRES", ""))
        or f"SAIC_T2V_V3_OUTPUT_ROOT={EXPECTED_RUN_ROOT}" not in str(parsed.get("SubmitLine", ""))
        or f"SAIC_T2V_V3_SOURCE_REVISION={EXPECTED_SOURCE_REVISION}" not in str(parsed.get("SubmitLine", ""))
        or f"SAIC_T2V_V3_SOURCE_ARCHIVE_SHA256={EXPECTED_SOURCE_ARCHIVE_SHA256}" not in str(parsed.get("SubmitLine", ""))
    ):
        raise SourceBoundRaw60Error("r8 Slurm C0/job/input binding differs")

    master_evidence = terminal.get("master_receipt")
    if (
        not isinstance(master_evidence, Mapping)
        or master_evidence.get("path") != EXPECTED_MASTER_RECEIPT_PATH
    ):
        raise SourceBoundRaw60Error("r8 terminal/master path binding differs")
    master, master_sha = _load_canonical_receipt(
        EXPECTED_MASTER_RECEIPT_PATH, label="r8 exact60 master receipt",
    )
    attempts = master.get("attempts")
    if (
        master_sha != master_evidence.get("sha256")
        or master.get("receipt_digest") != master_evidence.get("receipt_digest")
        or master.get("schema_version") != MASTER_SCHEMA
        or master.get("topology") != "two_concurrent_world4_sp4_groups_on_one_8gpu_node"
        or master.get("attempt_count") != EXPECTED_ATTEMPT_COUNT
        or master.get("seed_cell_count") != 20
        or master.get("six_branch_spec_merge_cell_count") != 20
        or master.get("root_spec_raw_sha256") != EXPECTED_ROOT_SPEC_SHA256
        or not isinstance(attempts, list)
        or len(attempts) != EXPECTED_ATTEMPT_COUNT
        or len({row.get("candidate_id") for row in attempts if isinstance(row, Mapping)}) != EXPECTED_ATTEMPT_COUNT
        or len({row.get("mp4_sha256") for row in attempts if isinstance(row, Mapping)}) != EXPECTED_ATTEMPT_COUNT
        or {row.get("branch") for row in attempts if isinstance(row, Mapping)}
           != {"incomplete", "camera_only", "appearance_only"}
        or master.get("event_verified") is not False
        or master.get("identity_preservation_verified") is not False
        or master.get("seed_selection_authorized") is not False
        or master.get("training_target_authorized") is not False
        or master.get("optimizer_or_parameter_update_authorized") is not False
    ):
        raise SourceBoundRaw60Error("r8 exact60 master receipt differs")

    deep = terminal.get("deep_audits")
    if not isinstance(deep, Mapping) or set(deep) != {"sp4-a", "sp4-b"}:
        raise SourceBoundRaw60Error("r8 two-shard terminal evidence differs")
    deep_rows: list[Mapping[str, Any]] = []
    for group_id in ("sp4-a", "sp4-b"):
        evidence = deep[group_id]
        if not isinstance(evidence, Mapping):
            raise SourceBoundRaw60Error("r8 deep-audit evidence differs")
        audit, audit_sha = _load_canonical_receipt(
            evidence.get("path", ""), label=f"r8 {group_id} deep audit",
        )
        rows = audit.get("rows")
        if (
            audit_sha != evidence.get("sha256")
            or audit.get("receipt_digest") != evidence.get("receipt_digest")
            or audit.get("schema_version") != DEEP_AUDIT_SCHEMA
            or audit.get("root") != EXPECTED_RUN_ROOT
            or audit.get("group_id") != group_id
            or audit.get("slurm_job_id") != EXPECTED_JOB_ID
            or audit.get("planned_candidate_count") != 30
            or audit.get("completed_prefix_count") != 30
            or audit.get("completed_candidate_indices") != list(range(30))
            or audit.get("deep_generation_receipt_validation") is not True
            or audit.get("deep_rendezvous_completion_validation") is not True
            or audit.get("same_cell_gaussian_prefix_validation") is not True
            or audit.get("authority") != EXPECTED_DEEP_AUDIT_AUTHORITY
            or not isinstance(rows, list)
            or len(rows) != 30
        ):
            raise SourceBoundRaw60Error(f"r8 {group_id} exact30 deep audit differs")
        deep_rows.extend(rows)
    master_by_id = {row["candidate_id"]: row for row in attempts}
    if set(master_by_id) != {row.get("candidate_id") for row in deep_rows}:
        raise SourceBoundRaw60Error("r8 master/deep exact60 coverage differs")
    for row in deep_rows:
        bound = master_by_id[row["candidate_id"]]
        if (
            bound.get("receipt_sha256") != row.get("attempt_receipt_sha256")
            or bound.get("receipt_digest") != row.get("attempt_receipt_digest")
            or bound.get("mp4_sha256") != row.get("mp4_sha256")
            or bound.get("branch") != row.get("branch")
        ):
            raise SourceBoundRaw60Error("r8 master/deep artifact binding differs")
    return {
        "path": EXPECTED_TERMINAL_EVIDENCE_PATH,
        "raw_sha256": terminal_sha,
        "receipt_digest": terminal["receipt_digest"],
        "job_id": EXPECTED_JOB_ID,
        "status": TERMINAL_EVIDENCE_STATUS,
        "candidate_count": EXPECTED_ATTEMPT_COUNT,
        "master_receipt_sha256": master_sha,
    }


def _validate_fixed_build_inputs(args: Any) -> None:
    if str(Path(args.attempts_root)) != EXPECTED_ATTEMPTS_ROOT:
        raise SourceBoundRaw60Error("r8 attempts-root lexical path differs")
    attempts_root = core.core._plain_directory(args.attempts_root, label="r8 attempts root")
    if str(attempts_root) != EXPECTED_ATTEMPTS_ROOT:
        raise SourceBoundRaw60Error("r8 attempts-root resolved path differs")
    if str(Path(args.source_manifest)) != EXPECTED_SOURCE_MANIFEST_PATH:
        raise SourceBoundRaw60Error("r8 source-manifest lexical path differs")
    if args.expected_root_spec_sha256 != EXPECTED_ROOT_SPEC_SHA256:
        raise SourceBoundRaw60Error("r8 root-spec SHA-256 argument differs")
    if args.expected_source_manifest_sha256 != EXPECTED_SOURCE_MANIFEST_SHA256:
        raise SourceBoundRaw60Error("r8 source-manifest SHA-256 argument differs")

    _validate_terminal_evidence(args.terminal_evidence)

    root_spec, _ = core.core._strict_json(
        EXPECTED_ROOT_SPEC_PATH,
        expected_sha256=EXPECTED_ROOT_SPEC_SHA256,
        label="r8 sealed top-up spec",
    )
    base_spec, _ = core.core._strict_json(
        EXPECTED_BASE_SPEC_PATH,
        expected_sha256=EXPECTED_BASE_SPEC_SHA256,
        label="r8 sealed base spec",
    )
    if (
        core.core.object_sha256(root_spec) != EXPECTED_ROOT_SPEC_CONTENT_SHA256
        or core.core.object_sha256(base_spec) != EXPECTED_BASE_SPEC_CONTENT_SHA256
        or root_spec.get("source_manifest_file_sha256") != EXPECTED_SOURCE_MANIFEST_SHA256
        or root_spec.get("source_manifest_content_sha256") != EXPECTED_SOURCE_MANIFEST_CONTENT_SHA256
        or root_spec.get("base_v1_spec_raw_sha256") != EXPECTED_BASE_SPEC_SHA256
        or root_spec.get("base_v1_spec_content_sha256") != EXPECTED_BASE_SPEC_CONTENT_SHA256
        or root_spec.get("top_up_only") is not True
        or root_spec.get("artifact_authority") != core.core.topup_generate.contract.ARTIFACT_AUTHORITY
    ):
        raise SourceBoundRaw60Error("r8 sealed spec/source/authority closure differs")

    paths = sorted(
        attempts_root.rglob(core.core.ATTEMPT_BASENAME),
        key=lambda item: item.as_posix(),
    )
    if len(paths) != EXPECTED_ATTEMPT_COUNT:
        raise SourceBoundRaw60Error("r8 bank is not exact60")
    rows = []
    for path in paths:
        receipt, receipt_sha = core.core._strict_json(
            path, expected_sha256=None, label="r8 generation receipt",
        )
        candidate = receipt.get("candidate")
        rows.append({
            "candidate_id": candidate.get("candidate_id") if isinstance(candidate, Mapping) else None,
            "receipt_sha256": receipt_sha,
        })
    master, _ = _load_canonical_receipt(
        EXPECTED_MASTER_RECEIPT_PATH, label="r8 exact60 master receipt",
    )
    master_bindings = [
        {"candidate_id": row.get("candidate_id"), "receipt_sha256": row.get("receipt_sha256")}
        for row in master["attempts"]
    ]
    if _receipt_binding_digest(rows) != _receipt_binding_digest(master_bindings):
        raise SourceBoundRaw60Error("r8 exact60 candidate/receipt set differs from terminal master")


def build_manifest(args: Any) -> int:
    _validate_fixed_build_inputs(args)
    result = _base_build_manifest(args)
    manifest_path = Path(args.output_root) / "input-manifest.json"
    value, _ = _load_canonical_receipt(manifest_path, label="r8 source-bound input manifest")
    # The inherited create-only manifest has no terminal evidence field.  The
    # full terminal closure is instead replayed on every build and every load;
    # this avoids rewriting or weakening its closed schema.
    if value.get("attempt_count") != EXPECTED_ATTEMPT_COUNT:
        raise SourceBoundRaw60Error("r8 built manifest exact60 coverage differs")
    return result


def _validate_source_binding(binding: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(binding, Mapping):
        raise SourceBoundRaw60Error(f"{label} source binding is absent")
    iid = binding.get("iid")
    if not isinstance(iid, str) or iid not in EXPECTED_SOURCE_SHA256_BY_IID:
        raise SourceBoundRaw60Error(f"{label} source IID differs")
    if (
        binding.get("actor_family") != EXPECTED_ACTOR_BY_IID[iid]
        or binding.get("source_video_sha256") != EXPECTED_SOURCE_SHA256_BY_IID[iid]
        or binding.get("source_video") != _source_path(iid)
    ):
        raise SourceBoundRaw60Error(f"{label} source path/SHA/actor binding differs")
    return binding


def load_input_manifest(
    path: str | Path, *, expected_sha256: str, expected_source_sha256: str,
) -> tuple[dict[str, Any], str]:
    terminal_evidence = _validate_terminal_evidence(EXPECTED_TERMINAL_EVIDENCE_PATH)
    value, raw_sha = _base_load_input_manifest(
        path,
        expected_sha256=expected_sha256,
        expected_source_sha256=expected_source_sha256,
    )
    attempts = value.get("attempts")
    source_manifest = value.get("source_manifest")
    if (
        value.get("attempts_root") != EXPECTED_ATTEMPTS_ROOT
        or value.get("root_spec_raw_sha256") != EXPECTED_ROOT_SPEC_SHA256
        or not isinstance(attempts, list)
        or not isinstance(source_manifest, Mapping)
        or source_manifest.get("path") != EXPECTED_SOURCE_MANIFEST_PATH
        or source_manifest.get("raw_sha256") != EXPECTED_SOURCE_MANIFEST_SHA256
        or source_manifest.get("content_sha256") != EXPECTED_SOURCE_MANIFEST_CONTENT_SHA256
        or source_manifest.get("wrong_source_policy") != core.WRONG_SOURCE_POLICY
    ):
        raise SourceBoundRaw60Error("r8 source-bound input identity differs")
    master, master_sha = _load_canonical_receipt(
        EXPECTED_MASTER_RECEIPT_PATH, label="r8 exact60 master receipt",
    )
    master_attempts = master.get("attempts")
    if (
        master_sha != terminal_evidence["master_receipt_sha256"]
        or not isinstance(master_attempts, list)
        or _receipt_binding_digest(attempts)
        != _receipt_binding_digest(master_attempts)
    ):
        raise SourceBoundRaw60Error("r8 input/master exact60 binding differs")
    for row in attempts:
        if not isinstance(row, Mapping):
            raise SourceBoundRaw60Error("r8 attempt row differs")
        correct = _validate_source_binding(row.get("correct_source"), label="correct")
        wrong = _validate_source_binding(row.get("wrong_source"), label="wrong")
        if wrong.get("iid") != EXPECTED_WRONG_IID_BY_IID[correct["iid"]]:
            raise SourceBoundRaw60Error("r8 preregistered wrong-source policy differs")
    return value, raw_sha


def _worker_common(args: Any) -> tuple[str, dict[str, Any], str, Mapping[str, Any], dict[str, Any], Any]:
    if (
        str(Path(args.visual_checkpoint)) != EXPECTED_CHECKPOINT_ROOT
        or str(Path(args.visual_checkpoint_manifest)) != EXPECTED_CHECKPOINT_MANIFEST_PATH
        or str(Path(args.evaluator_spec)) != EXPECTED_EVALUATOR_SPEC_PATH
        or str(Path(args.visual_scorer_source)) != EXPECTED_VISUAL_SCORER_PATH
        or str(Path(args.visual_contract_source)) != EXPECTED_VISUAL_CONTRACT_PATH
        or args.expected_evaluator_spec_sha256 != core.EXPECTED_EVALUATOR_SPEC_SHA256
        or args.expected_visual_scorer_sha256 != core.EXPECTED_VISUAL_SCORER_SHA256
        or args.expected_visual_contract_sha256 != core.EXPECTED_VISUAL_CONTRACT_SHA256
        or core.core.file_sha256(args.visual_checkpoint_manifest)
        != EXPECTED_CHECKPOINT_MANIFEST_SHA256
    ):
        raise SourceBoundRaw60Error("r8 frozen visual evaluator path/SHA closure differs")
    return _base_worker_common(args)


def _validate_visual_evaluator(value: Any, *, rank: int) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(EXPECTED_VISUAL_EVALUATOR):
        raise SourceBoundRaw60Error(
            f"shard {rank} visual evaluator field closure differs"
        )
    runtime = value.get("runtime_versions")
    loading = value.get("loading_counts")
    expected_runtime = EXPECTED_VISUAL_EVALUATOR["runtime_versions"]
    expected_loading = EXPECTED_VISUAL_EVALUATOR["loading_counts"]
    if (
        not isinstance(runtime, Mapping)
        or set(runtime) != set(expected_runtime)
        or any(type(item) is not str for item in runtime.values())
        or dict(runtime) != expected_runtime
        or not isinstance(loading, Mapping)
        or set(loading) != set(expected_loading)
        or any(type(item) is not int for item in loading.values())
        or dict(loading) != expected_loading
        or type(value.get("checkpoint_file_count")) is not int
        or value.get("checkpoint_file_count") != 3
        or type(value.get("trainable_parameter_tensors")) is not int
        or value.get("trainable_parameter_tensors") != 0
        or value.get("every_checkpoint_file_verified") is not True
        or value.get("frozen_eval") is not True
        or value.get("identity_authority") is not False
        or value.get("scientific_claim_authorized") is not False
        or not isinstance(value.get("preprocessor_golden_output_shape"), list)
        or any(
            type(item) is not int
            for item in value["preprocessor_golden_output_shape"]
        )
    ):
        raise SourceBoundRaw60Error(
            f"shard {rank} visual evaluator runtime/freeze/authority differs"
        )
    checked = {
        **dict(value),
        "runtime_versions": dict(runtime),
        "loading_counts": dict(loading),
        "preprocessor_golden_output_shape": list(
            value["preprocessor_golden_output_shape"]
        ),
    }
    if (
        checked != EXPECTED_VISUAL_EVALUATOR
        or core.core.object_sha256(checked)
        != EXPECTED_VISUAL_EVALUATOR_OBJECT_SHA256
    ):
        raise SourceBoundRaw60Error(
            f"shard {rank} visual evaluator fixed identity differs"
        )
    return checked


def _require_identical_visual_evaluator(
    reference: Mapping[str, Any], current: Mapping[str, Any], *, rank: int,
) -> str:
    reference_sha = core.core.object_sha256(reference)
    current_sha = core.core.object_sha256(current)
    if current != reference or current_sha != reference_sha:
        raise SourceBoundRaw60Error(
            f"shard {rank} visual evaluator differs across ranks"
        )
    return current_sha


def _strict_number(value: Any, *, label: str) -> float:
    if type(value) not in {int, float}:
        raise SourceBoundRaw60Error(f"{label} is not a strict numeric scalar")
    result = float(value)
    if not math.isfinite(result):
        raise SourceBoundRaw60Error(f"{label} is non-finite")
    return result


def _validate_decode_evidence(
    value: Any, *, expected_artifact_sha256: str, label: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != DECODE_EVIDENCE_FIELDS:
        raise SourceBoundRaw60Error(f"{label} decode field closure differs")
    for field in (
        "artifact_sha256", "decoded_rgb_sha256", "pts_sha256",
        "selected_rgb_sha256", "preprocessed_tensor_sha256",
    ):
        digest = core.core._sha256(value.get(field), label=f"{label} {field}")
        if field != "preprocessed_tensor_sha256" and digest == "0" * 64:
            raise SourceBoundRaw60Error(f"{label} {field} is an empty placeholder")
    if (
        value.get("artifact_sha256") != expected_artifact_sha256
        or value.get("frame_count") != 81
        or value.get("fps_numerator") != 25
        or value.get("fps_denominator") != 1
        or value.get("selected_frame_indices") != list(core.core.EVAL_FRAME_INDICES)
        or value.get("preprocessed_tensor_sha256") != "0" * 64
        or any(
            type(value.get(field)) is not int or value[field] <= 0
            for field in (
                "time_base_numerator", "time_base_denominator", "pts_step",
                "width", "height",
            )
        )
        or value.get("time_base_numerator") * value.get("pts_step") * 25
        != value.get("time_base_denominator")
    ):
        raise SourceBoundRaw60Error(f"{label} exact81 decode binding differs")


def _validate_feature_evidence(value: Any, *, label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != FEATURE_EVIDENCE_FIELDS:
        raise SourceBoundRaw60Error(f"{label} feature field closure differs")
    for field in ("global_feature_sha256", "dense_feature_sha256"):
        if core.core._sha256(value.get(field), label=f"{label} {field}") == "0" * 64:
            raise SourceBoundRaw60Error(f"{label} {field} is an empty placeholder")
    if {
        "selected_frame_count": value.get("selected_frame_count"),
        "dense_grid_height": value.get("dense_grid_height"),
        "dense_grid_width": value.get("dense_grid_width"),
        "feature_dimension": value.get("feature_dimension"),
    } != {
        "selected_frame_count": 17,
        "dense_grid_height": 16,
        "dense_grid_width": 16,
        "feature_dimension": 768,
    }:
        raise SourceBoundRaw60Error(f"{label} frozen-DINO feature geometry differs")


def _validate_source_evidence(
    value: Any, *, expected_source_sha256: str, label: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != {"decode", "features"}:
        raise SourceBoundRaw60Error(f"{label} source evidence field closure differs")
    _validate_decode_evidence(
        value.get("decode"), expected_artifact_sha256=expected_source_sha256,
        label=label,
    )
    _validate_feature_evidence(value.get("features"), label=label)


def _validate_raw_metrics(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != RAW_METRIC_FIELDS:
        raise SourceBoundRaw60Error("r8 raw metric field closure differs")
    if (
        value.get("measurement_label")
        != "frozen_dinov2_source_bound_raw_proxy_only"
        or value.get("thresholds") is not None
        or {field: value.get(field) for field in RAW_METRIC_AUTHORITY}
        != RAW_METRIC_AUTHORITY
    ):
        raise SourceBoundRaw60Error("r8 raw metric label/authority closure differs")
    global_correct = _strict_number(
        value.get("global_candidate_correct"), label="global candidate/correct",
    )
    global_wrong = _strict_number(
        value.get("global_candidate_wrong"), label="global candidate/wrong",
    )
    global_self = _strict_number(
        value.get("global_source_self_upper_bound"), label="global source self",
    )
    dense_correct = _strict_number(
        value.get("dense_candidate_correct"), label="dense candidate/correct",
    )
    dense_wrong = _strict_number(
        value.get("dense_candidate_wrong"), label="dense candidate/wrong",
    )
    dense_self = _strict_number(
        value.get("dense_source_self_upper_bound"), label="dense source self",
    )
    global_margin = _strict_number(
        value.get("global_correct_minus_wrong_margin"), label="global margin",
    )
    dense_margin = _strict_number(
        value.get("dense_correct_minus_wrong_margin"), label="dense margin",
    )
    if (
        not all(0.0 <= item <= 1.0 for item in (
            global_correct, global_wrong, global_self,
            dense_correct, dense_wrong, dense_self,
        ))
        or not math.isclose(global_self, 1.0, rel_tol=0.0, abs_tol=1.0e-6)
        or not math.isclose(dense_self, 1.0, rel_tol=0.0, abs_tol=1.0e-6)
        or not math.isclose(
            global_margin, global_correct - global_wrong,
            rel_tol=0.0, abs_tol=1.0e-12,
        )
        or not math.isclose(
            dense_margin, dense_correct - dense_wrong,
            rel_tol=0.0, abs_tol=1.0e-12,
        )
    ):
        raise SourceBoundRaw60Error("r8 raw metric range/self/margin arithmetic differs")


def _validate_candidate_result(value: Any, *, expected: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping) or set(value) != RESULT_FIELDS:
        raise SourceBoundRaw60Error("r8 candidate result field closure differs")
    correct = expected.get("correct_source")
    wrong = expected.get("wrong_source")
    if (
        value.get("candidate_id") != expected.get("candidate_id")
        or value.get("candidate_binding") != expected
        or value.get("authority") != AUTHORITY_CLOSURE
        or not isinstance(correct, Mapping)
        or not isinstance(wrong, Mapping)
    ):
        raise SourceBoundRaw60Error("r8 candidate binding/nested authority differs")
    _validate_decode_evidence(
        value.get("candidate_decode"),
        expected_artifact_sha256=expected.get("mp4_sha256"),
        label="candidate",
    )
    _validate_feature_evidence(value.get("candidate_features"), label="candidate")
    _validate_source_evidence(
        value.get("correct_source_evidence"),
        expected_source_sha256=correct.get("source_video_sha256"),
        label="correct source",
    )
    _validate_source_evidence(
        value.get("wrong_source_evidence"),
        expected_source_sha256=wrong.get("source_video_sha256"),
        label="wrong source",
    )
    _validate_raw_metrics(value.get("raw_metrics"))


def aggregate(args: Any) -> int:
    source_sha = core.core._verify_self(args.expected_source_sha256)
    manifest, manifest_sha = load_input_manifest(
        args.input_manifest,
        expected_sha256=args.expected_input_manifest_sha256,
        expected_source_sha256=source_sha,
    )
    output_root = core.core._plain_directory(args.output_root, label="output root")
    shards: list[dict[str, Any]] = []
    by_index: dict[int, Mapping[str, Any]] = {}
    reference_visual_evaluator: dict[str, Any] | None = None
    rank_visual_evaluator_receipts: list[dict[str, Any]] = []
    for rank in range(EXPECTED_WORLD_SIZE):
        path = output_root / f"shard-{rank:02d}-of-{EXPECTED_WORLD_SIZE:02d}.json"
        value, raw_sha = core.core._strict_json(
            path, expected_sha256=None, label=f"shard {rank}",
        )
        unsigned = dict(value)
        declared = core.core._sha256(
            unsigned.pop("receipt_digest", None), label="shard digest",
        )
        indices = partition_indices(EXPECTED_ATTEMPT_COUNT, rank, EXPECTED_WORLD_SIZE)
        results = value.get("candidate_results")
        if (
            set(value) != {
                "schema_version", "diagnostic_source_sha256",
                "input_manifest_sha256", "rank", "world_size",
                "partition_indices", "candidate_count", "candidate_results",
                "visual_evaluator", "authority", "receipt_digest",
            }
            or value.get("schema_version") != SHARD_SCHEMA
            or value.get("diagnostic_source_sha256") != source_sha
            or value.get("input_manifest_sha256") != manifest_sha
            or value.get("rank") != rank
            or value.get("world_size") != EXPECTED_WORLD_SIZE
            or value.get("partition_indices") != list(indices)
            or value.get("candidate_count") != len(indices)
            or not isinstance(results, list)
            or len(results) != len(indices)
            or value.get("authority") != AUTHORITY_CLOSURE
            or declared != core.core.object_sha256(unsigned)
        ):
            raise SourceBoundRaw60Error(f"shard {rank} contract differs")
        visual_evaluator = _validate_visual_evaluator(
            value.get("visual_evaluator"), rank=rank,
        )
        if reference_visual_evaluator is None:
            reference_visual_evaluator = visual_evaluator
            visual_evaluator_sha = core.core.object_sha256(visual_evaluator)
        else:
            visual_evaluator_sha = _require_identical_visual_evaluator(
                reference_visual_evaluator, visual_evaluator, rank=rank,
            )
        rank_visual_evaluator_receipts.append({
            "rank": rank,
            "visual_evaluator_projection_sha256": visual_evaluator_sha,
        })
        shards.append({
            "rank": rank, "path": str(path.resolve(strict=True)),
            "sha256": raw_sha, "receipt_digest": declared,
        })
        for index, result in zip(indices, results):
            if index in by_index:
                raise SourceBoundRaw60Error("shard partition overlaps")
            _validate_candidate_result(
                result, expected=manifest["attempts"][index],
            )
            by_index[index] = result
    if set(by_index) != set(range(EXPECTED_ATTEMPT_COUNT)):
        raise SourceBoundRaw60Error("shards do not cover exact r8 exact60")
    ordered = [by_index[index] for index in range(EXPECTED_ATTEMPT_COUNT)]
    expected_ids = [row["candidate_id"] for row in manifest["attempts"]]
    if [row.get("candidate_id") for row in ordered] != expected_ids:
        raise SourceBoundRaw60Error("aggregate candidate order differs")
    if reference_visual_evaluator is None:
        raise SourceBoundRaw60Error("aggregate visual evaluator evidence is absent")
    unsigned = {
        "schema_version": AGGREGATE_SCHEMA,
        "diagnostic_source_sha256": source_sha,
        "input_manifest_sha256": manifest_sha,
        "world_size": EXPECTED_WORLD_SIZE,
        "candidate_count": EXPECTED_ATTEMPT_COUNT,
        "coverage": "exactly_once_complete_r8_exact60_source_bound_raw",
        "candidate_order": expected_ids,
        "shards": shards,
        "candidate_results": ordered,
        "visual_evaluator_evidence_projection": reference_visual_evaluator,
        "visual_evaluator_evidence_projection_sha256": core.core.object_sha256(
            reference_visual_evaluator
        ),
        "per_rank_visual_evaluator_projection_receipts":
            rank_visual_evaluator_receipts,
        "all8_visual_evaluator_projections_identical": True,
        "interpretation": {
            "measurement": "raw frozen-DINO candidate/correct/wrong source proxies and source-self upper bounds",
            "wrong_source_preregistered_without_candidate_metrics": True,
            "no_absolute_preservation_claim": True,
            "no_event_measurement": True,
            "no_threshold_or_ranking": True,
        },
        "authority": dict(AUTHORITY_CLOSURE),
    }
    core.core._write_create_only(
        output_root / "aggregate-receipt.json",
        {**unsigned, "receipt_digest": core.core.object_sha256(unsigned)},
    )
    return 0


def _install_specialization() -> None:
    _configure_core()
    _configure_partitions()
    _install_build_parser()
    core.build_manifest = build_manifest
    core.load_input_manifest = load_input_manifest
    core._worker_common = _worker_common
    core.aggregate = aggregate


def main(argv: Sequence[str] | None = None) -> int:
    _install_specialization()
    return core.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
