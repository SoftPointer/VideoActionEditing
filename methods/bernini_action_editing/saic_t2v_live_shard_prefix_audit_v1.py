#!/usr/bin/env python3
"""Read-only deep audit of the completed prefix of one running SAIC shard.

The final split-root auditor intentionally requires an exact30 immutable shard.
That is too late to detect a bad attempt, rendezvous receipt, or artifact while a
multi-hour generation job is still running.  This tool validates only the
contiguous completed prefix and grants no merge, review, training, optimizer,
or scientific-selection authority.

It reuses the production generation and rendezvous validators.  No receipt is
written: the canonical JSON report is emitted to stdout and is explicitly a
live diagnostic whose content may grow while the producer is active.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import generate_saic_pure_t2v_event_bank_topup_v2 as generation
import saic_pure_t2v_event_bank_topup_v2 as contract
import saic_t2v_partial_group_pair_audit_v1 as final_audit
import saic_t2v_rendezvous_guard_v2 as rendezvous


SCHEMA_VERSION = "saic-t2v-live-shard-prefix-audit-v1"
STATUS = "running_prefix_deep_audited_no_authority"
PLAN_NAME = re.compile(r"(?P<index>[0-9]{4})-(?P<candidate>.+)\.json\Z")
AUTHORITY = {
    "detached_decoded_event_review_input": False,
    "merge_or_partial_reuse": False,
    "scientific_selection": False,
    "training": False,
    "optimizer": False,
}


class SAICT2VLiveShardPrefixAuditError(RuntimeError):
    """Raised when a running shard prefix is not deeply self-consistent."""


def _wrap(label: str, error: BaseException) -> SAICT2VLiveShardPrefixAuditError:
    return SAICT2VLiveShardPrefixAuditError(f"{label}: {error}")


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise _wrap(f"{label} is not ASCII JSON", error) from error
    if not isinstance(value, dict):
        raise SAICT2VLiveShardPrefixAuditError(f"{label} must be an object")
    return value


def _load_inputs(args: argparse.Namespace) -> tuple[
    Mapping[str, Any], Mapping[str, Any], str
]:
    try:
        spec, root_sha = contract.load_sealed_spec(
            args.root_spec,
            expected_raw_sha256=args.expected_root_spec_sha256,
            source_manifest_path=args.source_manifest,
            base_v1_spec_path=args.base_v1_spec,
        )
        base = contract.load_base_v1_spec(
            args.base_v1_spec, source_manifest_path=args.source_manifest
        )
        if len(contract.merge_six_branch_cells(base, spec)) != 20:
            raise SAICT2VLiveShardPrefixAuditError(
                "six-branch contract does not contain exact20 cells"
            )
        source = contract.source_set.load_manifest(args.source_manifest)
        contract.source_set.validate_manifest(source)
    except (
        contract.SAICPureT2VEventBankTopupError,
        contract.source_set.SAICReversibleSourceSetError,
    ) as error:
        raise _wrap("root/source contract differs", error) from error
    return spec, source, root_sha


def _plan_rows(root: Path, group: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    group_id = group["group_id"]
    plan_root = final_audit._plain_dir(
        root / "plan" / group_id, label=f"{group_id} plan root"
    )
    paths = sorted(plan_root.glob("*.json"), key=lambda path: path.name)
    candidates = group.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 30 or len(paths) != 30:
        raise SAICT2VLiveShardPrefixAuditError("live shard plan is not exact30")
    rows: list[Mapping[str, Any]] = []
    for expected_index, (path, candidate) in enumerate(zip(paths, candidates)):
        match = PLAN_NAME.fullmatch(path.name)
        if (
            match is None
            or int(match.group("index")) != expected_index
            or match.group("candidate") != candidate.get("candidate_id")
        ):
            raise SAICT2VLiveShardPrefixAuditError(
                f"plan filename/order differs at {expected_index}"
            )
        envelope = _load_json(path, label=f"plan {expected_index}")
        if (
            envelope.get("candidate") != candidate
            or envelope.get("group_id") != group_id
            or envelope.get("visible_gpus") != group.get("visible_gpus")
        ):
            raise SAICT2VLiveShardPrefixAuditError(
                f"plan content differs at {expected_index}"
            )
        rows.append(envelope)
    return rows


def audit_live_prefix(
    root_value: str | Path,
    *,
    group: Mapping[str, Any],
    root_spec_sha256: str,
    source_manifest: Mapping[str, Any],
    expected_min_completed: int = 0,
) -> dict[str, Any]:
    """Deeply validate the contiguous successful prefix of one live group."""

    if (
        isinstance(expected_min_completed, bool)
        or not isinstance(expected_min_completed, int)
        or not 0 <= expected_min_completed <= 30
    ):
        raise SAICT2VLiveShardPrefixAuditError(
            "expected minimum completed must be an integer in [0,30]"
        )
    group_id = group.get("group_id")
    if group_id not in final_audit.EXPECTED_GROUPS:
        raise SAICT2VLiveShardPrefixAuditError("group ID differs")
    root = final_audit._plain_dir(root_value, label=f"{group_id} live root")
    plans = _plan_rows(root, group)
    candidates = group["candidates"]
    lifecycle_root = final_audit._plain_dir(
        root / "logs" / "rendezvous" / group_id,
        label=f"{group_id} lifecycle root",
    )
    claim_root = final_audit._plain_dir(
        root / "logs" / "rendezvous" / "port-claims",
        label=f"{group_id} claim root",
    )
    completion_paths = sorted(lifecycle_root.glob("candidate-*/launch-*/completion.json"))
    completions_by_index: dict[int, Path] = {}
    slurm_job_ids: set[str] = set()
    for path in completion_paths:
        path = final_audit._plain_file(
            path, label=f"{group_id} live completion", mode=0o444
        )
        final_audit._under(path, root, label=f"{group_id} live completion")
        shallow = _load_json(path, label=f"{group_id} live completion")
        index = shallow.get("candidate_index")
        job_id = shallow.get("slurm_job_id")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < 30
            or not isinstance(job_id, str)
            or not job_id.isdigit()
            or index in completions_by_index
        ):
            raise SAICT2VLiveShardPrefixAuditError(
                "live completion index/job/uniqueness differs"
            )
        completions_by_index[index] = path
        slurm_job_ids.add(job_id)
    indices = sorted(completions_by_index)
    if indices != list(range(len(indices))):
        raise SAICT2VLiveShardPrefixAuditError(
            "successful candidates do not form a contiguous prefix"
        )
    if len(indices) < expected_min_completed:
        raise SAICT2VLiveShardPrefixAuditError(
            "completed prefix is shorter than the requested minimum"
        )
    if len(slurm_job_ids) > 1:
        raise SAICT2VLiveShardPrefixAuditError("live prefix spans multiple Slurm jobs")

    real_paths = {row["source_video"] for row in source_manifest["rows"]}
    real_hashes = {row["source_video_sha256"] for row in source_manifest["rows"]}
    seen_ports: set[int] = set()
    seen_rendezvous: set[str] = set()
    prefix_rows: list[dict[str, Any]] = []
    gaussian_cells: dict[tuple[str, int], set[str]] = {}
    proof_fields = (
        "raw_value_sha256",
        "content_sha256",
        "shape",
        "dtype",
        "stored_dtype",
        "generator_initial_seed",
    )
    for index in indices:
        candidate = candidates[index]
        completion_path = completions_by_index[index]
        job_id = next(iter(slurm_job_ids))
        try:
            completion = rendezvous._validate_completion(
                completion_path, claim_root=claim_root, slurm_job_id=job_id
            )
            attempt_path = final_audit._plain_file(
                completion["attempt_receipt_path"],
                label=f"{group_id} attempt {index}",
                mode=0o444,
            )
            final_audit._under(attempt_path, root, label=f"{group_id} attempt {index}")
            attempt = generation._load_attempt_receipt(
                attempt_path,
                candidate=candidate,
                group=group,
                root_spec_sha256=root_spec_sha256,
                real_source_paths=real_paths,
                real_source_hashes=real_hashes,
            )
        except (
            rendezvous.SAICT2VRendezvousGuardError,
            generation.SAICPureT2VTopupGenerationError,
            final_audit.SAICPartialPairAuditError,
            KeyError,
        ) as error:
            raise _wrap(f"deep audit failed for candidate {index}", error) from error
        expected_attempt = (
            root
            / "attempts"
            / candidate["candidate_id"]
            / generation.ATTEMPT_RECEIPT_BASENAME
        )
        port = completion.get("actual_master_port")
        rendezvous_id = completion.get("rdzv_id")
        if (
            completion.get("candidate_index") != index
            or completion.get("candidate_id") != candidate["candidate_id"]
            or attempt_path != expected_attempt
            or completion.get("attempt_receipt_sha256")
            != final_audit.file_sha256(attempt_path)
            or completion.get("attempt_receipt_digest") != attempt["receipt_digest"]
            or port in seen_ports
            or rendezvous_id in seen_rendezvous
        ):
            raise SAICT2VLiveShardPrefixAuditError(
                f"completion/attempt binding differs at candidate {index}"
            )
        seen_ports.add(port)
        seen_rendezvous.add(rendezvous_id)
        gaussian = attempt["artifacts"]["official_initial_gaussian"]
        identity = contract.object_sha256(
            {field: gaussian.get(field) for field in proof_fields}
        )
        gaussian_cells.setdefault((candidate["iid"], candidate["seed"]), set()).add(
            identity
        )
        if len(gaussian_cells[(candidate["iid"], candidate["seed"])]) != 1:
            raise SAICT2VLiveShardPrefixAuditError(
                f"same-cell Gaussian differs at candidate {index}"
            )
        prefix_rows.append(
            {
                "candidate_index": index,
                "candidate_id": candidate["candidate_id"],
                "branch": candidate["branch"],
                "seed": candidate["seed"],
                "plan_digest": contract.object_sha256(plans[index]),
                "attempt_receipt_sha256": final_audit.file_sha256(attempt_path),
                "attempt_receipt_digest": attempt["receipt_digest"],
                "completion_receipt_sha256": final_audit.file_sha256(completion_path),
                "completion_receipt_digest": completion["receipt_digest"],
                "mp4_sha256": attempt["artifacts"]["mp4"]["sha256"],
                "official_gaussian_identity_digest": identity,
                "actual_master_port": port,
                "rdzv_id": rendezvous_id,
            }
        )

    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "root": str(root),
        "group_id": group_id,
        "slurm_job_id": next(iter(slurm_job_ids), None),
        "root_spec_raw_sha256": root_spec_sha256,
        "planned_candidate_count": 30,
        "completed_prefix_count": len(prefix_rows),
        "completed_candidate_indices": indices,
        "deep_generation_receipt_validation": True,
        "deep_rendezvous_completion_validation": True,
        "same_cell_gaussian_prefix_validation": True,
        "rows": prefix_rows,
        "authority": AUTHORITY,
    }
    return {**unsigned, "receipt_digest": final_audit.object_sha256(unsigned)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--group-id", choices=final_audit.EXPECTED_GROUPS, required=True)
    parser.add_argument("--root-spec", required=True)
    parser.add_argument("--base-v1-spec", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--expected-root-spec-sha256", required=True)
    parser.add_argument("--expected-min-completed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec, source, root_sha = _load_inputs(args)
    groups = {group["group_id"]: group for group in spec["groups"]}
    if set(groups) != set(final_audit.EXPECTED_GROUPS):
        raise SAICT2VLiveShardPrefixAuditError("root spec group closure differs")
    receipt = audit_live_prefix(
        args.output_root,
        group=groups[args.group_id],
        root_spec_sha256=root_sha,
        source_manifest=source,
        expected_min_completed=args.expected_min_completed,
    )
    print(final_audit.canonical_bytes(receipt).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
