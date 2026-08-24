"""Audit two disjoint SAIC exact30 WORLD4 roots as one split-root exact60 bank.

This tool deliberately does not synthesize the legacy single-root master
receipt.  Attempt receipts bind absolute artifact paths inside their original
root, so copying or path-rewriting them would destroy their identity.  The
result instead authorizes detached decoded-video review only; training and
optimizer authority remain false until that review and a compatible consumer
admit the split-root receipt explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

import generate_saic_pure_t2v_event_bank_topup_v2 as generation
import saic_pure_t2v_event_bank_topup_v2 as contract
import saic_t2v_rendezvous_guard_v2 as rendezvous


SCHEMA_VERSION = "bernini-saic-t2v-topup-split-root-exact60-audit-v1"
PARTIAL_SCHEMA_VERSION = "saic-t2v-topup-existing-allocation-partial-group-v1"
PARTIAL_STATUS = "exact30_world4_group_complete_pending_disjoint_merge_and_full_audit"
PAIR_STATUS = "exact60_split_roots_verified_pending_detached_decoded_event_review"
PARTIAL_BASENAME = "saic-pure-t2v-event-bank-topup-partial-{group_id}-receipt.json"
MASTER_BASENAME = generation.MASTER_RECEIPT_BASENAME
SHA256 = "0123456789abcdef"
EXPECTED_GROUPS = ("sp4-a", "sp4-b")
EXPECTED_LOGICAL_DEVICES = {"sp4-a": [0, 1, 2, 3], "sp4-b": [4, 5, 6, 7]}
AUTHORITY = {
    "detached_decoded_event_review_input": True,
    "scientific_selection": False,
    "training": False,
    "optimizer": False,
    "single_root_master_receipt_emulated": False,
    "artifact_copy_or_path_rewrite": False,
}
PARTIAL_AUTHORITY = {
    "scientific_selection": False,
    "training": False,
    "optimizer": False,
    "merge_or_partial_reuse": False,
}
PARTIAL_FIELDS = {
    "schema_version", "status", "group_id", "slurm_job_id",
    "source_revision", "source_archive_sha256", "root_spec_raw_sha256",
    "candidate_count", "logical_visible_devices", "rows", "authority",
    "receipt_digest",
}
PARTIAL_ROW_FIELDS = {
    "candidate_index", "candidate_id", "plan_sha256",
    "attempt_receipt_path", "attempt_receipt_sha256",
    "completion_receipt_path", "completion_receipt_sha256",
}


class SAICPartialPairAuditError(RuntimeError):
    """Raised before an ambiguous split-root bank gains review authority."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _closed(value: Any, fields: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise SAICPartialPairAuditError(f"{label} fields differ")
    return value


def _plain_dir(path: str | Path, *, label: str) -> Path:
    value = Path(path)
    if not value.is_absolute() or not value.is_dir() or value.is_symlink():
        raise SAICPartialPairAuditError(f"{label} is not a plain absolute directory")
    try:
        resolved = value.resolve(strict=True)
    except OSError as error:
        raise SAICPartialPairAuditError(f"{label} cannot be resolved") from error
    if resolved != value:
        raise SAICPartialPairAuditError(f"{label} is not canonical")
    return value


def _plain_file(
    path: str | Path, *, label: str, mode: int | None = None
) -> Path:
    value = Path(path)
    if not value.is_absolute() or not value.is_file() or value.is_symlink():
        raise SAICPartialPairAuditError(f"{label} is not a plain absolute file")
    try:
        resolved = value.resolve(strict=True)
    except OSError as error:
        raise SAICPartialPairAuditError(f"{label} cannot be resolved") from error
    if resolved != value:
        raise SAICPartialPairAuditError(f"{label} is not canonical")
    observed = value.stat()
    if observed.st_nlink != 1:
        raise SAICPartialPairAuditError(f"{label} link count differs")
    if mode is not None and stat.S_IMODE(observed.st_mode) != mode:
        raise SAICPartialPairAuditError(f"{label} mode differs")
    return value


def _load_sealed(path: Path, *, fields: set[str], label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SAICPartialPairAuditError(f"{label} is not canonical JSON") from error
    value = _closed(value, fields, label=label)
    unsigned = dict(value)
    claimed = unsigned.pop("receipt_digest", None)
    if not isinstance(claimed, str) or len(claimed) != 64 or any(
        character not in SHA256 for character in claimed
    ) or claimed != object_sha256(unsigned):
        raise SAICPartialPairAuditError(f"{label} digest differs")
    if path.read_bytes() != canonical_bytes(value) + b"\n":
        raise SAICPartialPairAuditError(f"{label} bytes are not canonical")
    return value


def _under(path: Path, root: Path, *, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise SAICPartialPairAuditError(f"{label} escaped its shard root") from error


def _load_contract_inputs(args: argparse.Namespace) -> tuple[
    Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], str
]:
    try:
        spec, root_sha = contract.load_sealed_spec(
            args.root_spec,
            expected_raw_sha256=args.expected_root_spec_sha256,
            source_manifest_path=args.source_manifest,
            base_v1_spec_path=args.base_v1_spec,
        )
        base_spec = contract.load_base_v1_spec(
            args.base_v1_spec, source_manifest_path=args.source_manifest
        )
        merged = contract.merge_six_branch_cells(base_spec, spec)
        source_manifest = contract.source_set.load_manifest(args.source_manifest)
        contract.source_set.validate_manifest(source_manifest)
    except (
        contract.SAICPureT2VEventBankTopupError,
        contract.source_set.SAICReversibleSourceSetError,
    ) as error:
        raise SAICPartialPairAuditError("root/source contract differs") from error
    if len(merged) != 20:
        raise SAICPartialPairAuditError("six-branch cell cardinality differs")
    return spec, base_spec, source_manifest, root_sha


def validate_shard(
    root_value: str | Path,
    *,
    group: Mapping[str, Any],
    root_spec_sha256: str,
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    group_id = group.get("group_id")
    if group_id not in EXPECTED_GROUPS:
        raise SAICPartialPairAuditError("group ID differs")
    root = _plain_dir(root_value, label=f"{group_id} root")
    if (root / MASTER_BASENAME).exists() or (root / MASTER_BASENAME).is_symlink():
        raise SAICPartialPairAuditError(f"{group_id} root already claims a full master")
    partial_path = _plain_file(
        root / PARTIAL_BASENAME.format(group_id=group_id),
        label=f"{group_id} partial receipt",
        mode=0o444,
    )
    partial = _load_sealed(
        partial_path, fields=PARTIAL_FIELDS, label=f"{group_id} partial receipt"
    )
    slurm_job_id = partial.get("slurm_job_id")
    if (
        partial.get("schema_version") != PARTIAL_SCHEMA_VERSION
        or partial.get("status") != PARTIAL_STATUS
        or partial.get("group_id") != group_id
        or not isinstance(slurm_job_id, str)
        or not slurm_job_id.isdigit()
        or partial.get("root_spec_raw_sha256") != root_spec_sha256
        or partial.get("candidate_count") != 30
        or partial.get("logical_visible_devices")
        != EXPECTED_LOGICAL_DEVICES[group_id]
        or partial.get("authority") != PARTIAL_AUTHORITY
    ):
        raise SAICPartialPairAuditError(f"{group_id} partial authority differs")
    rows = partial.get("rows")
    candidates = group.get("candidates")
    if not isinstance(rows, list) or not isinstance(candidates, list) or len(rows) != 30 or len(candidates) != 30:
        raise SAICPartialPairAuditError(f"{group_id} exact30 cardinality differs")

    attempts_root = _plain_dir(root / "attempts", label=f"{group_id} attempts")
    expected_ids = [candidate.get("candidate_id") for candidate in candidates]
    observed_attempt_dirs = sorted(
        path.name for path in attempts_root.iterdir()
        if path.is_dir() and not path.is_symlink()
    )
    if sorted(expected_ids) != observed_attempt_dirs:
        raise SAICPartialPairAuditError(f"{group_id} attempt directory coverage differs")
    real_source_paths = {row["source_video"] for row in source_manifest["rows"]}
    real_source_hashes = {row["source_video_sha256"] for row in source_manifest["rows"]}
    claim_root = _plain_dir(
        root / "logs" / "rendezvous" / "port-claims",
        label=f"{group_id} claim root",
    )
    plan_paths = sorted((root / "plan" / group_id).glob("*.json"), key=lambda item: item.name)
    if len(plan_paths) != 30 or any(not path.is_file() or path.is_symlink() for path in plan_paths):
        raise SAICPartialPairAuditError(f"{group_id} plan coverage differs")

    candidate_rows: list[dict[str, Any]] = []
    gaussian_cells: dict[tuple[str, int], list[tuple[str, Mapping[str, Any]]]] = {}
    ports: set[int] = set()
    rendezvous_ids: set[str] = set()
    for index, (row_value, candidate, plan_path) in enumerate(zip(rows, candidates, plan_paths)):
        row = _closed(row_value, PARTIAL_ROW_FIELDS, label=f"{group_id} row {index}")
        candidate_id = candidate.get("candidate_id")
        attempt_path = _plain_file(
            row.get("attempt_receipt_path", ""),
            label=f"{group_id} attempt {index}",
            mode=0o444,
        )
        completion_path = _plain_file(
            row.get("completion_receipt_path", ""),
            label=f"{group_id} completion {index}",
            mode=0o444,
        )
        expected_attempt_path = (
            attempts_root / str(candidate_id) / generation.ATTEMPT_RECEIPT_BASENAME
        )
        if (
            row.get("candidate_index") != index
            or row.get("candidate_id") != candidate_id
            or attempt_path != expected_attempt_path
            or row.get("plan_sha256") != file_sha256(plan_path)
            or row.get("attempt_receipt_sha256") != file_sha256(attempt_path)
            or row.get("completion_receipt_sha256") != file_sha256(completion_path)
        ):
            raise SAICPartialPairAuditError(f"{group_id} row {index} binding differs")
        _under(attempt_path, root, label=f"{group_id} attempt {index}")
        _under(completion_path, root, label=f"{group_id} completion {index}")
        try:
            attempt = generation._load_attempt_receipt(
                attempt_path,
                candidate=candidate,
                group=group,
                root_spec_sha256=root_spec_sha256,
                real_source_paths=real_source_paths,
                real_source_hashes=real_source_hashes,
            )
            completion = rendezvous._validate_completion(
                completion_path,
                claim_root=claim_root,
                slurm_job_id=slurm_job_id,
            )
        except (
            generation.SAICPureT2VTopupGenerationError,
            rendezvous.SAICT2VRendezvousGuardError,
        ) as error:
            raise SAICPartialPairAuditError(
                f"{group_id} candidate {index} deep audit failed"
            ) from error
        if (
            completion.get("candidate_index") != index
            or completion.get("candidate_id") != candidate_id
            or Path(completion.get("attempt_receipt_path", "")) != attempt_path
            or completion.get("attempt_receipt_sha256") != file_sha256(attempt_path)
            or completion.get("attempt_receipt_digest") != attempt.get("receipt_digest")
        ):
            raise SAICPartialPairAuditError(f"{group_id} completion {index} differs")
        port = completion.get("actual_master_port")
        rdzv_id = completion.get("rdzv_id")
        if port in ports or rdzv_id in rendezvous_ids:
            raise SAICPartialPairAuditError(f"{group_id} rendezvous uniqueness differs")
        ports.add(port)
        rendezvous_ids.add(rdzv_id)
        gaussian = attempt["artifacts"]["official_initial_gaussian"]
        gaussian_cells.setdefault((candidate["iid"], candidate["seed"]), []).append(
            (candidate["branch"], gaussian)
        )
        candidate_rows.append(
            {
                "group_id": group_id,
                "candidate_index": index,
                "candidate_id": candidate_id,
                "iid": candidate["iid"],
                "seed": candidate["seed"],
                "branch": candidate["branch"],
                "attempt_receipt_path": str(attempt_path),
                "attempt_receipt_sha256": file_sha256(attempt_path),
                "attempt_receipt_digest": attempt["receipt_digest"],
                "completion_receipt_path": str(completion_path),
                "completion_receipt_sha256": file_sha256(completion_path),
                "mp4_path": attempt["artifacts"]["mp4"]["path"],
                "mp4_sha256": attempt["artifacts"]["mp4"]["sha256"],
            }
        )
    claim_paths = sorted(claim_root.glob("port-*.json"), key=lambda item: item.name)
    if len(claim_paths) != 30 or any(not path.is_file() or path.is_symlink() for path in claim_paths):
        raise SAICPartialPairAuditError(f"{group_id} permanent claim coverage differs")

    proof_fields = (
        "raw_value_sha256", "content_sha256", "shape", "dtype", "stored_dtype",
        "generator_initial_seed",
    )
    gaussian_proofs = []
    for (iid, seed), values in sorted(gaussian_cells.items()):
        if [branch for branch, _ in values] != list(contract.BRANCH_ORDER):
            raise SAICPartialPairAuditError(f"{group_id} same-cell branch order differs")
        identities = {
            contract.object_sha256({field: gaussian.get(field) for field in proof_fields})
            for _, gaussian in values
        }
        if len(identities) != 1:
            raise SAICPartialPairAuditError(f"{group_id} same-cell Gaussian differs")
        gaussian_proofs.append(
            {
                "group_id": group_id,
                "iid": iid,
                "seed": seed,
                "branch_order": list(contract.BRANCH_ORDER),
                "official_gaussian_identity_digest": next(iter(identities)),
            }
        )
    if len(gaussian_proofs) != 10:
        raise SAICPartialPairAuditError(f"{group_id} seed-cell count differs")
    return {
        "group_id": group_id,
        "root": str(root),
        "slurm_job_id": slurm_job_id,
        "source_revision": partial["source_revision"],
        "source_archive_sha256": partial["source_archive_sha256"],
        "root_spec_raw_sha256": partial["root_spec_raw_sha256"],
        "partial_receipt_path": str(partial_path),
        "partial_receipt_sha256": file_sha256(partial_path),
        "partial_receipt_digest": partial["receipt_digest"],
        "candidate_count": 30,
        "permanent_port_claim_count": len(claim_paths),
        "candidate_rows": candidate_rows,
        "gaussian_proofs": gaussian_proofs,
    }


def assemble_pair_receipt(shards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(shards) != 2 or tuple(shard.get("group_id") for shard in shards) != EXPECTED_GROUPS:
        raise SAICPartialPairAuditError("paired group order differs")
    first, second = shards
    for field in ("source_revision", "source_archive_sha256", "root_spec_raw_sha256"):
        if first.get(field) != second.get(field):
            raise SAICPartialPairAuditError(f"paired {field} differs")
    if first.get("root") == second.get("root"):
        raise SAICPartialPairAuditError("paired roots are not disjoint")
    rows = [row for shard in shards for row in shard.get("candidate_rows", [])]
    proofs = [row for shard in shards for row in shard.get("gaussian_proofs", [])]
    candidate_ids = [row.get("candidate_id") for row in rows]
    if (
        any(shard.get("candidate_count") != 30 for shard in shards)
        or any(shard.get("permanent_port_claim_count") != 30 for shard in shards)
        or len(rows) != 60
        or len(set(candidate_ids)) != 60
        or len(proofs) != 20
    ):
        raise SAICPartialPairAuditError("paired exact60 coverage differs")
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "status": PAIR_STATUS,
        "source_revision": first["source_revision"],
        "source_archive_sha256": first["source_archive_sha256"],
        "root_spec_raw_sha256": first["root_spec_raw_sha256"],
        "topology": "two_disjoint_world4_sp4_roots",
        "group_order": list(EXPECTED_GROUPS),
        "candidate_count": 60,
        "seed_cell_count": 20,
        "branch_order": list(contract.BRANCH_ORDER),
        "shards": [
            {key: value for key, value in shard.items() if key not in {"candidate_rows", "gaussian_proofs"}}
            for shard in shards
        ],
        "same_seed_official_gaussian_proofs": proofs,
        "candidates": rows,
        "detached_full81_event_review_complete": False,
        "event_verified": False,
        "identity_preservation_verified": False,
        "training_target_authorized": False,
        "optimizer_or_parameter_update_authorized": False,
        "authority": AUTHORITY,
    }
    return {**unsigned, "receipt_digest": object_sha256(unsigned)}


def write_create_only(path_value: str | Path, value: Mapping[str, Any]) -> Path:
    path = Path(path_value)
    if not path.is_absolute() or path == Path("/") or path.exists() or path.is_symlink():
        raise SAICPartialPairAuditError("output receipt must be fresh and absolute")
    parent = _plain_dir(path.parent, label="output receipt parent")
    if path != parent / path.name or not path.name.endswith(".json"):
        raise SAICPartialPairAuditError("output receipt path differs")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o444,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o444)
    except BaseException:
        raise
    observed = _plain_file(path, label="output receipt", mode=0o444)
    return observed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sp4-a-root", required=True)
    parser.add_argument("--sp4-b-root", required=True)
    parser.add_argument("--root-spec", required=True)
    parser.add_argument("--base-v1-spec", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--expected-root-spec-sha256", required=True)
    parser.add_argument("--output-receipt", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec, _, source_manifest, root_sha = _load_contract_inputs(args)
    groups = {group["group_id"]: group for group in spec["groups"]}
    if set(groups) != set(EXPECTED_GROUPS):
        raise SAICPartialPairAuditError("root spec group coverage differs")
    shard_a = validate_shard(
        args.sp4_a_root,
        group=groups["sp4-a"],
        root_spec_sha256=root_sha,
        source_manifest=source_manifest,
    )
    shard_b = validate_shard(
        args.sp4_b_root,
        group=groups["sp4-b"],
        root_spec_sha256=root_sha,
        source_manifest=source_manifest,
    )
    receipt = assemble_pair_receipt([shard_a, shard_b])
    write_create_only(args.output_receipt, receipt)
    print(canonical_bytes(receipt).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
