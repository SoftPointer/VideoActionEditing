#!/usr/bin/env python3
"""Fail-closed, zero-model-compute replay of the sealed r8 Qwen-v6 audit.

The original Qwen observations are immutable evidence.  This program does not
load a model and does not rewrite those observations.  It replays their sealed
record digests, generation metadata, original deterministic branch gate, and
the terminal/master/two-deep-audit exact60 closure.  It then closes one schema
hole: ``requested_attribute_already_present_at_start`` must be exactly
``not_applicable`` outside ``appearance_only``.  The existing v6 gate already
requires exactly ``no`` for an appearance-only pass.

The sole output is a create-only corrected replay receipt.  It grants no human
review, selection, training, optimizer, admission, or scientific authority.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, NoReturn, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
QWEN_SOURCE_NAME = "audit_saic_t2v_branch_semantics_qwen_v1.py"
QWEN_SOURCE_SHA256 = (
    "bca26ff69e29ada23e35610a26810643010b2cc4d9a9707b8a068c20b53cbe66"
)
_QWEN_SOURCE_PATH = METHOD_ROOT / QWEN_SOURCE_NAME
if (
    not _QWEN_SOURCE_PATH.is_file()
    or _QWEN_SOURCE_PATH.is_symlink()
    or hashlib.sha256(_QWEN_SOURCE_PATH.read_bytes()).hexdigest()
    != QWEN_SOURCE_SHA256
):
    raise RuntimeError("pinned Qwen-v6 deterministic-gate source differs")

_QWEN_SPEC = importlib.util.spec_from_file_location(
    "_saic_r8_pinned_qwen_v6_gate", _QWEN_SOURCE_PATH
)
if _QWEN_SPEC is None or _QWEN_SPEC.loader is None:
    raise RuntimeError("cannot load pinned Qwen-v6 deterministic-gate source")
qwen_v6 = importlib.util.module_from_spec(_QWEN_SPEC)
sys.modules[_QWEN_SPEC.name] = qwen_v6
_QWEN_SPEC.loader.exec_module(qwen_v6)


SCHEMA_VERSION = "saic-r8-exact60-qwen-v6-corrected-replay-v1"
EXPECTED_OLD_LAUNCHER_SHA256 = (
    "38f63226963b7d780639c3e7250916cde1f5a5e1012870d08cfcd8be03793a5a"
)
EXPECTED_RECORDS_SHA256 = (
    "d885317804e62d9f58f183476f538f3e5dbba9f21579ddb8971ad160a48f38c4"
)
EXPECTED_SUMMARY_SHA256 = (
    "c6e5a995267ddb5779481c9837bc5458a1b1217a25ea70c2ee99d5d8d02445c7"
)
EXPECTED_SUMMARY_RECEIPT_DIGEST = (
    "93eef610cd6cfe9aa1f24bb7c916712543683b497100d866d59844d37ceb9534"
)
EXPECTED_TERMINAL_SHA256 = (
    "07a6ec7ccbe165d89aa8757985537ef18d62eea5d08e245e452b607dee5bd29a"
)
EXPECTED_TERMINAL_RECEIPT_DIGEST = (
    "a8fe672840d597445a2164660a38bdfeb4fa51ccfbc3b822c3af8adb6d6519e5"
)
EXPECTED_MASTER_SHA256 = (
    "c5528a08fa976c0dbfb16984a35df3169c2d013a73fabd982ad45f45d5defc61"
)
EXPECTED_MASTER_RECEIPT_DIGEST = (
    "8d28c170f5c8fdc5e76bdfb55bb89a5a819f02beb483c005f87d6898c5d8ae33"
)
EXPECTED_DEEP = {
    "sp4-a": {
        "sha256": "2c5b47c306a7cd7895278c3bc668bc8c895328ff7c528afcab8b4ccbdd83a67e",
        "receipt_digest": "3a8bec49270bd426a360969141b404d21a43d37469b6876c0bc9d43e0124ac48",
    },
    "sp4-b": {
        "sha256": "fca0e039259babae6188a8912a5990d16fe6584d9e8d8092eb02e036d83865d3",
        "receipt_digest": "748a547808e47460292f409e37b7e9306f81907a6ffc463934dfee63740eb3ef",
    },
}

EXPECTED_RUN_ROOT = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_saic_v1_20260809/runs/"
    "t2v-events-topup-r8-ddc8a79-r1"
)
EXPECTED_ATTEMPTS_ROOT = f"{EXPECTED_RUN_ROOT}/attempts"
EXPECTED_QWEN_ROOT = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_saic_v1_20260809/runs/"
    "saic-fresh60-qwen4b-v6-69eec35-r1"
)
EXPECTED_RECORDS_PATH = (
    f"{EXPECTED_QWEN_ROOT}/qwen3_vl_4b_branch_semantics_records.jsonl"
)
EXPECTED_SUMMARY_PATH = (
    f"{EXPECTED_QWEN_ROOT}/qwen3_vl_4b_branch_semantics_summary.json"
)
EXPECTED_RELEASE_EVIDENCE_ROOT = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_saic_v1_20260809/releases/"
    "saic-t2v-topup-r8-ddc8a79-r1/evidence-terminal-ddc8a79"
)
EXPECTED_TERMINAL_PATH = (
    f"{EXPECTED_RELEASE_EVIDENCE_ROOT}/saic-exact60-terminal-evidence-135056.json"
)
EXPECTED_MASTER_PATH = (
    f"{EXPECTED_RUN_ROOT}/saic-pure-t2v-event-bank-topup-receipt.json"
)
EXPECTED_DEEP_PATHS = {
    group: f"{EXPECTED_RELEASE_EVIDENCE_ROOT}/deep-audit-{group}.json"
    for group in ("sp4-a", "sp4-b")
}

EXPECTED_SOURCE_REVISION = "ddc8a79199aed1391cf089f51835c2bbfa74ae28"
EXPECTED_SOURCE_ARCHIVE_SHA256 = (
    "4038100b86655e5ea3e9a32432dc619c4b8d1a5d7859703c4cf06b77de0b934b"
)
EXPECTED_ROOT_SPEC_SHA256 = (
    "d693d0784530f007888e2825d15db3db808fdf4f1d111b5d080d968c894ff145"
)
EXPECTED_OLD_PASS_IDS = frozenset({
    "saic-topup-v2-7b88a1ca1f804f41-camera_only-s2026082102",
    "saic-topup-v2-99cde432839f4240-appearance_only-s2026082203",
})
EXPECTED_CORRECTED_PASS_IDS = frozenset({
    "saic-topup-v2-99cde432839f4240-appearance_only-s2026082203",
})
EXPECTED_FIELD_VIOLATION_IDS = frozenset({
    "saic-topup-v2-31c34509415745ca-incomplete-s2026082131",
    "saic-topup-v2-7b88a1ca1f804f41-camera_only-s2026082102",
    "saic-topup-v2-7b88a1ca1f804f41-incomplete-s2026082102",
    "saic-topup-v2-841b5e0080a1441d-incomplete-s2026082111",
})

BRANCHES = ("incomplete", "camera_only", "appearance_only")
AUTHORITY = {
    "data_selection": False,
    "human_review": False,
    "optimizer": False,
    "scientific_claim": False,
    "selection": False,
    "training": False,
    "training_target_admission": False,
}
QWEN_AUTHORITY = {
    "data_selection": False,
    "human_review": False,
    "optimizer": False,
    "scientific_claim": False,
    "training": False,
}
TERMINAL_AUTHORITY = {
    "data_selection": False,
    "detached_decoded_event_review_input": True,
    "human_review": False,
    "optimizer": False,
    "scientific_action_editing_success_claim": False,
    "training": False,
    "training_target_admission": False,
}
DEEP_AUTHORITY = {
    "detached_decoded_event_review_input": False,
    "merge_or_partial_reuse": False,
    "optimizer": False,
    "scientific_selection": False,
    "training": False,
}

RECORD_FIELDS = frozenset({
    "action_family_id", "actor_family", "analysis_split", "authority",
    "branch", "branch_set", "candidate_id",
    "deterministic_branch_gate_passed", "deterministic_failure_codes",
    "generation_receipt_path", "generation_receipt_sha256", "iid",
    "parse_error", "raw_response", "receipt_digest",
    "registered_specification", "schema_version", "seed",
    "validated_observation", "video_path", "video_sha256", "visual_input",
})
SUMMARY_FIELDS = frozenset({
    "attempts_root", "authority", "branch_set",
    "deterministic_gate_pass_count", "failure_code_counts", "model",
    "output_jsonl", "output_jsonl_sha256", "passes_by_actor_family",
    "passes_by_analysis_split", "passes_by_branch", "receipt_digest",
    "record_count", "records_by_actor_family", "records_by_analysis_split",
    "records_by_branch", "schema_version", "status",
    "valid_model_output_count",
})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class QwenReplayError(RuntimeError):
    """Raised before a corrected receipt can be created."""


def fail(message: str) -> NoReturn:
    raise QwenReplayError(message)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        fail(f"{label} is not a lowercase SHA-256")
    return value


def _closed(value: Any, keys: Iterable[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != set(keys):
        fail(f"{label} fields differ")
    return value


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            fail(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def decode_json(raw: bytes, *, label: str) -> Any:
    try:
        text = raw.decode("ascii")
        return json.loads(text, object_pairs_hook=_reject_duplicate_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        fail(f"{label} is not strict ASCII JSON: {error}")


def _plain_file(path_value: str | Path, *, label: str) -> Path:
    path = Path(path_value)
    if not path.is_file() or path.is_symlink():
        fail(f"{label} is not a plain file")
    return path


def _load_raw_pinned(
    path_value: str | Path, expected_sha256: str, *, label: str
) -> tuple[bytes, str]:
    path = _plain_file(path_value, label=label)
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != _sha256(expected_sha256, label=f"expected {label} SHA-256"):
        fail(f"{label} raw SHA-256 differs")
    return raw, actual


def _load_receipt_pinned(
    path_value: str | Path,
    expected_sha256: str,
    expected_digest: str,
    *,
    label: str,
) -> tuple[dict[str, Any], str]:
    raw, actual = _load_raw_pinned(path_value, expected_sha256, label=label)
    value = decode_json(raw, label=label)
    if not isinstance(value, dict):
        fail(f"{label} is not an object")
    unsigned = dict(value)
    claimed = unsigned.pop("receipt_digest", None)
    if (
        claimed != _sha256(expected_digest, label=f"expected {label} digest")
        or object_sha256(unsigned) != claimed
    ):
        fail(f"{label} canonical receipt digest differs")
    return value, actual


def corrected_branch_gate(
    branch: str, observation: Mapping[str, Any]
) -> tuple[bool, list[str], bool]:
    """Replay v6 and add the missing non-appearance field constraint.

    The returned Boolean is the corrected gate, the list is the corrected
    failure-code set, and the final Boolean says whether this row exposes the
    previously ungated non-appearance field violation.
    """

    validated = qwen_v6.validate_model_output(dict(observation))
    old_passed, old_failures = qwen_v6.deterministic_branch_gate(branch, validated)
    attribute = validated["requested_attribute_already_present_at_start"]
    failures = set(old_failures)
    field_violation = branch != "appearance_only" and attribute != "not_applicable"
    if field_violation:
        failures.add("appearance_start_field_misapplied")
    # The old v6 gate must continue to enforce exactly `no` for an appearance
    # pass; guard that invariant explicitly so a dependency change fails shut.
    if branch == "appearance_only" and attribute != "no" and old_passed:
        fail("old v6 gate stopped enforcing appearance-start absence")
    corrected = sorted(failures)
    return not corrected, corrected, field_violation


def _validate_terminal_bundle(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    lexical = {
        "terminal evidence": (args.terminal_evidence, EXPECTED_TERMINAL_PATH),
        "master receipt": (args.master_receipt, EXPECTED_MASTER_PATH),
        "sp4-a deep audit": (args.deep_audit_sp4_a, EXPECTED_DEEP_PATHS["sp4-a"]),
        "sp4-b deep audit": (args.deep_audit_sp4_b, EXPECTED_DEEP_PATHS["sp4-b"]),
    }
    for label, (actual, expected) in lexical.items():
        if str(Path(actual)) != expected:
            fail(f"{label} lexical path differs")

    terminal, terminal_sha = _load_receipt_pinned(
        args.terminal_evidence,
        EXPECTED_TERMINAL_SHA256,
        EXPECTED_TERMINAL_RECEIPT_DIGEST,
        label="r8 terminal evidence",
    )
    if (
        terminal.get("schema_version") != "saic-t2v-full60-terminal-evidence-v1"
        or terminal.get("status")
        != "terminal_technical_full60_complete_pending_detached_semantic_review"
        or terminal.get("job_id") != "135056"
        or terminal.get("root") != EXPECTED_RUN_ROOT
        or terminal.get("candidate_count") != 60
        or terminal.get("seed_cell_count") != 20
        or terminal.get("unique_mp4_sha256_count") != 60
        or terminal.get("source_revision") != EXPECTED_SOURCE_REVISION
        or terminal.get("source_archive_sha256") != EXPECTED_SOURCE_ARCHIVE_SHA256
        or terminal.get("root_spec_raw_sha256") != EXPECTED_ROOT_SPEC_SHA256
        or terminal.get("authority") != TERMINAL_AUTHORITY
    ):
        fail("r8 terminal exact60 closure differs")
    parsed = (
        terminal.get("slurm_terminal_observation", {}).get("parsed_row")
        if isinstance(terminal.get("slurm_terminal_observation"), dict)
        else None
    )
    if (
        not isinstance(parsed, dict)
        or parsed.get("JobIDRaw") != "135056"
        or parsed.get("State") != "COMPLETED"
        or parsed.get("ExitCode") != "0:0"
        or parsed.get("AllocNodes") != "1"
        or "gres/gpu:mi210=8" not in str(parsed.get("AllocTRES", ""))
    ):
        fail("r8 terminal Slurm C0 closure differs")

    master, master_sha = _load_receipt_pinned(
        args.master_receipt,
        EXPECTED_MASTER_SHA256,
        EXPECTED_MASTER_RECEIPT_DIGEST,
        label="r8 master receipt",
    )
    master_ref = terminal.get("master_receipt")
    if (
        not isinstance(master_ref, dict)
        or master_ref
        != {
            "path": EXPECTED_MASTER_PATH,
            "receipt_digest": EXPECTED_MASTER_RECEIPT_DIGEST,
            "sha256": EXPECTED_MASTER_SHA256,
        }
        or master_sha != EXPECTED_MASTER_SHA256
    ):
        fail("terminal/master exact binding differs")
    attempts = master.get("attempts")
    if (
        master.get("schema_version")
        != "bernini-saic-pure-t2v-event-bank-topup-receipt-v2"
        or master.get("attempt_count") != 60
        or master.get("seed_cell_count") != 20
        or master.get("branch_order") != list(BRANCHES)
        or master.get("topology")
        != "two_concurrent_world4_sp4_groups_on_one_8gpu_node"
        or master.get("event_verified") is not False
        or master.get("identity_preservation_verified") is not False
        or master.get("seed_selection_authorized") is not False
        or master.get("training_target_authorized") is not False
        or master.get("optimizer_or_parameter_update_authorized") is not False
        or not isinstance(attempts, list)
        or len(attempts) != 60
    ):
        fail("r8 master exact60 metadata differs")
    master_by_id = {
        row.get("candidate_id"): row for row in attempts if isinstance(row, dict)
    }
    if (
        len(master_by_id) != 60
        or Counter(row.get("branch") for row in attempts) != Counter({b: 20 for b in BRANCHES})
        or len({row.get("mp4_sha256") for row in attempts}) != 60
    ):
        fail("r8 master candidate inventory differs")

    terminal_deep = terminal.get("deep_audits")
    if not isinstance(terminal_deep, dict) or set(terminal_deep) != set(EXPECTED_DEEP):
        fail("terminal deep-audit inventory differs")
    deep_rows: list[dict[str, Any]] = []
    for group, path_value in (
        ("sp4-a", args.deep_audit_sp4_a),
        ("sp4-b", args.deep_audit_sp4_b),
    ):
        pin = EXPECTED_DEEP[group]
        audit, audit_sha = _load_receipt_pinned(
            path_value,
            pin["sha256"],
            pin["receipt_digest"],
            label=f"r8 {group} deep audit",
        )
        if terminal_deep[group] != {
            "path": EXPECTED_DEEP_PATHS[group],
            "receipt_digest": pin["receipt_digest"],
            "sha256": pin["sha256"],
        }:
            fail(f"terminal/{group} deep-audit exact binding differs")
        rows = audit.get("rows")
        if (
            audit_sha != pin["sha256"]
            or audit.get("schema_version") != "saic-t2v-live-shard-prefix-audit-v1"
            or audit.get("root") != EXPECTED_RUN_ROOT
            or audit.get("group_id") != group
            or audit.get("slurm_job_id") != "135056"
            or audit.get("planned_candidate_count") != 30
            or audit.get("completed_prefix_count") != 30
            or audit.get("completed_candidate_indices") != list(range(30))
            or audit.get("deep_generation_receipt_validation") is not True
            or audit.get("deep_rendezvous_completion_validation") is not True
            or audit.get("same_cell_gaussian_prefix_validation") is not True
            or audit.get("authority") != DEEP_AUTHORITY
            or not isinstance(rows, list)
            or len(rows) != 30
        ):
            fail(f"r8 {group} exact30 deep closure differs")
        deep_rows.extend(rows)
    deep_by_id = {
        row.get("candidate_id"): row for row in deep_rows if isinstance(row, dict)
    }
    if set(deep_by_id) != set(master_by_id):
        fail("master/deep exact60 candidate coverage differs")
    for candidate_id, master_row in master_by_id.items():
        deep_row = deep_by_id[candidate_id]
        if (
            deep_row.get("attempt_receipt_sha256") != master_row.get("receipt_sha256")
            or deep_row.get("attempt_receipt_digest") != master_row.get("receipt_digest")
            or deep_row.get("mp4_sha256") != master_row.get("mp4_sha256")
            or deep_row.get("branch") != master_row.get("branch")
            or deep_row.get("seed") != master_row.get("seed")
        ):
            fail("master/deep artifact binding differs")
    projection = {
        "terminal": {
            "path": EXPECTED_TERMINAL_PATH,
            "raw_sha256": terminal_sha,
            "receipt_digest": EXPECTED_TERMINAL_RECEIPT_DIGEST,
        },
        "master": {
            "path": EXPECTED_MASTER_PATH,
            "raw_sha256": master_sha,
            "receipt_digest": EXPECTED_MASTER_RECEIPT_DIGEST,
        },
        "deep_audits": {
            group: {
                "path": EXPECTED_DEEP_PATHS[group],
                "raw_sha256": EXPECTED_DEEP[group]["sha256"],
                "receipt_digest": EXPECTED_DEEP[group]["receipt_digest"],
            }
            for group in ("sp4-a", "sp4-b")
        },
        "candidate_count": 60,
        "unique_mp4_count": 60,
        "all_master_deep_candidate_artifact_bindings_exact": True,
    }
    return master_by_id, projection


def _load_records(path_value: str | Path) -> tuple[list[dict[str, Any]], str]:
    if str(Path(path_value)) != EXPECTED_RECORDS_PATH:
        fail("Qwen records lexical path differs")
    raw, raw_sha = _load_raw_pinned(
        path_value, EXPECTED_RECORDS_SHA256, label="sealed Qwen-v6 records"
    )
    lines = raw.splitlines()
    if len(lines) != 60 or any(not line for line in lines):
        fail("Qwen-v6 JSONL exact60 line closure differs")
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        value = decode_json(line, label=f"Qwen record {index}")
        row = dict(_closed(value, RECORD_FIELDS, label=f"Qwen record {index}"))
        unsigned = dict(row)
        claimed = unsigned.pop("receipt_digest")
        if object_sha256(unsigned) != claimed:
            fail(f"Qwen record {index} receipt digest differs")
        records.append(row)
    return records, raw_sha


def _validate_records(
    records: Sequence[Mapping[str, Any]], master_by_id: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    ids = [row.get("candidate_id") for row in records]
    if len(records) != 60 or len(set(ids)) != 60 or set(ids) != set(master_by_id):
        fail("Qwen/master exact60 candidate inventory differs")

    old_pass_ids: set[str] = set()
    corrected_pass_ids: set[str] = set()
    field_violations: list[dict[str, Any]] = []
    existing_appearance_rejections = 0
    binding_projection: list[dict[str, Any]] = []
    for index, row in enumerate(records):
        candidate_id = row["candidate_id"]
        master = master_by_id[candidate_id]
        if (
            row["schema_version"] != qwen_v6.SCHEMA
            or row["branch_set"] != "counterfactual"
            or row["branch"] not in BRANCHES
            or row["actor_family"] not in {"dog", "human"}
            or row["analysis_split"] not in {"fit", "confirmation"}
            or not isinstance(row["iid"], str)
            or type(row["seed"]) is not int
            or not isinstance(row["action_family_id"], str)
            or not row["action_family_id"]
            or row["parse_error"] is not None
            or not isinstance(row["raw_response"], str)
            or row["authority"] != QWEN_AUTHORITY
        ):
            fail(f"Qwen record {index} closed metadata differs")
        visual = _closed(
            row["visual_input"],
            {"kind", "nframes", "tile_width", "columns", "sha256"},
            label=f"Qwen record {index} visual input",
        )
        if (
            visual["kind"] != "chronological_labeled_mosaic"
            or visual["nframes"] != 9
            or visual["tile_width"] != 384
            or visual["columns"] != 3
        ):
            fail(f"Qwen record {index} visual protocol differs")
        _sha256(visual["sha256"], label=f"Qwen record {index} mosaic SHA-256")
        spec = _closed(
            row["registered_specification"],
            {"start_state", "branch_instruction"},
            label=f"Qwen record {index} registered specification",
        )
        if any(not isinstance(spec[key], str) or not spec[key] for key in spec):
            fail(f"Qwen record {index} registered specification differs")

        if (
            row["iid"] != master.get("iid")
            or row["branch"] != master.get("branch")
            or row["analysis_split"] != master.get("analysis_split")
            or row["seed"] != master.get("seed")
            or row["video_path"] != master.get("mp4_path")
            or row["video_sha256"] != master.get("mp4_sha256")
            or row["generation_receipt_path"] != master.get("receipt_path")
            or row["generation_receipt_sha256"] != master.get("receipt_sha256")
        ):
            fail(f"Qwen record {index} master metadata binding differs")

        generation_raw, _ = _load_raw_pinned(
            row["generation_receipt_path"],
            row["generation_receipt_sha256"],
            label=f"Qwen record {index} generation receipt",
        )
        generation = decode_json(
            generation_raw, label=f"Qwen record {index} generation receipt"
        )
        candidate = generation.get("candidate") if isinstance(generation, dict) else None
        if not isinstance(candidate, dict) or any(
            candidate.get(record_key) != row[record_key]
            for record_key in (
                "candidate_id", "iid", "branch", "actor_family",
                "action_family_id", "analysis_split", "seed",
            )
        ):
            fail(f"Qwen record {index} live generation metadata differs")
        if (
            candidate.get("branch_start_state_caption") != spec["start_state"]
            or candidate.get("branch_instruction") != spec["branch_instruction"]
            or candidate.get("event_verified") is not False
            or candidate.get("event_audit_status")
            != "pending_detached_full81_review"
        ):
            fail(f"Qwen record {index} generation semantic boundary differs")
        video_path = _plain_file(row["video_path"], label=f"Qwen record {index} MP4")
        if file_sha256(video_path) != row["video_sha256"]:
            fail(f"Qwen record {index} live MP4 SHA-256 differs")

        observation = qwen_v6.validate_model_output(row["validated_observation"])
        replay_passed, replay_failures = qwen_v6.deterministic_branch_gate(
            row["branch"], observation
        )
        if (
            row["deterministic_branch_gate_passed"] is not replay_passed
            or row["deterministic_failure_codes"] != replay_failures
        ):
            fail(f"Qwen record {index} old deterministic gate replay differs")
        corrected_passed, corrected_failures, field_violation = corrected_branch_gate(
            row["branch"], observation
        )
        attribute = observation["requested_attribute_already_present_at_start"]
        if row["branch"] == "appearance_only" and attribute != "no":
            existing_appearance_rejections += 1
            if replay_passed:
                fail("appearance-start violation passed old deterministic gate")
        if replay_passed:
            old_pass_ids.add(candidate_id)
        if corrected_passed:
            corrected_pass_ids.add(candidate_id)
        if field_violation:
            field_violations.append({
                "candidate_id": candidate_id,
                "branch": row["branch"],
                "observed_value": attribute,
                "required_value": "not_applicable",
                "old_gate_passed": replay_passed,
                "corrected_gate_passed": corrected_passed,
                "corrected_failure_codes": corrected_failures,
            })
        binding_projection.append({
            "candidate_id": candidate_id,
            "generation_receipt_sha256": row["generation_receipt_sha256"],
            "video_sha256": row["video_sha256"],
            "record_receipt_digest": row["receipt_digest"],
        })

    if old_pass_ids != EXPECTED_OLD_PASS_IDS:
        fail("sealed old deterministic pass set differs")
    if corrected_pass_ids != EXPECTED_CORRECTED_PASS_IDS:
        fail("corrected deterministic pass set differs")
    if {row["candidate_id"] for row in field_violations} != EXPECTED_FIELD_VIOLATION_IDS:
        fail("new non-appearance field-violation set differs")
    if existing_appearance_rejections != 4:
        fail("existing appearance-start rejection count differs")

    branch_counts = Counter(row["branch"] for row in records)
    actor_counts = Counter(row["actor_family"] for row in records)
    split_counts = Counter(row["analysis_split"] for row in records)
    if (
        branch_counts != Counter({branch: 20 for branch in BRANCHES})
        or actor_counts != Counter({"dog": 30, "human": 30})
        or split_counts != Counter({"fit": 24, "confirmation": 36})
    ):
        fail("Qwen exact60 branch/actor/split counts differ")
    return {
        "record_count": 60,
        "all_record_receipt_digests_replayed": True,
        "all_generation_receipt_metadata_replayed": True,
        "all_live_generation_receipt_and_mp4_hashes_replayed": True,
        "all_old_deterministic_gates_replayed": True,
        "old_gate_pass_count": len(old_pass_ids),
        "old_gate_pass_candidate_ids": sorted(old_pass_ids),
        "existing_appearance_start_gate_rejection_count": existing_appearance_rejections,
        "field_violation_count": len(field_violations),
        "field_violations": sorted(field_violations, key=lambda row: row["candidate_id"]),
        "corrected_gate_pass_count": len(corrected_pass_ids),
        "corrected_gate_pass_candidate_ids": sorted(corrected_pass_ids),
        "candidate_binding_projection_sha256": object_sha256(
            sorted(binding_projection, key=lambda row: row["candidate_id"])
        ),
    }


def _count_by(records: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[key]) for row in records).items()))


def _pass_count_by(
    records: Sequence[Mapping[str, Any]], key: str
) -> dict[str, int]:
    output: Counter[str] = Counter()
    for row in records:
        if row["deterministic_branch_gate_passed"] is True:
            output[str(row[key])] += 1
    return {value: output[value] for value in sorted({str(row[key]) for row in records})}


def _validate_summary(
    path_value: str | Path,
    records_path: str | Path,
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], str]:
    if str(Path(path_value)) != EXPECTED_SUMMARY_PATH:
        fail("Qwen summary lexical path differs")
    summary, raw_sha = _load_receipt_pinned(
        path_value,
        EXPECTED_SUMMARY_SHA256,
        EXPECTED_SUMMARY_RECEIPT_DIGEST,
        label="sealed Qwen-v6 summary",
    )
    _closed(summary, SUMMARY_FIELDS, label="Qwen-v6 summary")
    failure_counts: Counter[str] = Counter()
    for row in records:
        failure_counts.update(set(row["deterministic_failure_codes"]))
    expected_model = {
        "config_sha256": "edac7703329133edfc53e46ac0081835144c99d7eebf28b71c732694d435224d",
        "do_sample": False,
        "index_sha256": "58a7841d7bff2548dd91577d216274a83cf1b500bc6a534b809d6c1b1707cf2b",
        "max_new_tokens": 768,
        "path": "/tmp/models/Qwen3-VL-4B-Instruct",
        "transformers_version": "5.5.4",
    }
    if (
        summary["schema_version"] != qwen_v6.SUMMARY_SCHEMA
        or summary["status"] != "diagnostic_vlm_triage_no_authority"
        or summary["attempts_root"] != EXPECTED_ATTEMPTS_ROOT
        or summary["branch_set"] != "counterfactual"
        or summary["record_count"] != 60
        or summary["valid_model_output_count"] != 60
        or summary["deterministic_gate_pass_count"] != 2
        or summary["records_by_branch"] != _count_by(records, "branch")
        or summary["passes_by_branch"] != _pass_count_by(records, "branch")
        or summary["records_by_actor_family"] != _count_by(records, "actor_family")
        or summary["passes_by_actor_family"] != _pass_count_by(records, "actor_family")
        or summary["records_by_analysis_split"] != _count_by(records, "analysis_split")
        or summary["passes_by_analysis_split"] != _pass_count_by(records, "analysis_split")
        or summary["failure_code_counts"] != dict(sorted(failure_counts.items()))
        or summary["output_jsonl"] != str(Path(records_path))
        or summary["output_jsonl_sha256"] != EXPECTED_RECORDS_SHA256
        or summary["model"] != expected_model
        or summary["authority"] != QWEN_AUTHORITY
    ):
        fail("Qwen-v6 summary aggregate replay differs")
    return summary, raw_sha


def _write_create_only(path_value: str | Path, value: Mapping[str, Any]) -> None:
    path = Path(path_value)
    if path.exists() or path.is_symlink():
        fail("corrected replay receipt output already exists")
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        fail("corrected replay receipt parent is not a plain directory")
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("ascii")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o400)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-launcher", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--terminal-evidence", required=True)
    parser.add_argument("--master-receipt", required=True)
    parser.add_argument("--deep-audit-sp4-a", required=True)
    parser.add_argument("--deep-audit-sp4-b", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    expected_self = _sha256(
        args.expected_source_sha256, label="expected replay source SHA-256"
    )
    actual_self = file_sha256(Path(__file__).resolve())
    if actual_self != expected_self:
        fail("replay source self SHA-256 differs")
    _, old_launcher_sha = _load_raw_pinned(
        args.old_launcher,
        EXPECTED_OLD_LAUNCHER_SHA256,
        label="sealed original Qwen-v6 launcher",
    )
    master_by_id, formal_projection = _validate_terminal_bundle(args)
    records, records_sha = _load_records(args.records)
    replay = _validate_records(records, master_by_id)
    summary, summary_sha = _validate_summary(args.summary, args.records, records)

    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "status": "corrected_deterministic_replay_diagnostic_no_authority",
        "source": {
            "path": str(Path(__file__).resolve()),
            "raw_sha256": actual_self,
            "qwen_v6_gate_source_name": QWEN_SOURCE_NAME,
            "qwen_v6_gate_source_sha256": QWEN_SOURCE_SHA256,
            "model_loaded_or_called": False,
        },
        "original_qwen_v6_evidence": {
            "launcher_path": str(Path(args.old_launcher)),
            "launcher_raw_sha256": old_launcher_sha,
            "records_path": str(Path(args.records)),
            "records_raw_sha256": records_sha,
            "summary_path": str(Path(args.summary)),
            "summary_raw_sha256": summary_sha,
            "summary_receipt_digest": summary["receipt_digest"],
        },
        "formal_r8_exact60_binding": formal_projection,
        "corrected_replay": replay,
        "interpretation": {
            "appearance_only_pass_requires_attribute_absent_at_start_exactly_no": True,
            "nonappearance_requires_attribute_field_exactly_not_applicable": True,
            "records_or_observations_modified": False,
            "corrected_receipt_is_diagnostic_only": True,
            "full_video_human_review_still_required": True,
            "corrected_pass_is_not_automatic_admission": True,
        },
        "authority": dict(AUTHORITY),
    }
    receipt = {**unsigned, "receipt_digest": object_sha256(unsigned)}
    _write_create_only(args.output, receipt)
    print(json.dumps({
        "output": str(Path(args.output)),
        "receipt_digest": receipt["receipt_digest"],
        "field_violation_count": replay["field_violation_count"],
        "corrected_gate_pass_count": replay["corrected_gate_pass_count"],
        "corrected_gate_pass_candidate_ids": replay[
            "corrected_gate_pass_candidate_ids"
        ],
        "authority": AUTHORITY,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
