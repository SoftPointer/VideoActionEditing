#!/usr/bin/env python3
"""Profile arbitrary inherited banks or finalize detached MOSAIC audits.

Unlike the legacy core4 label author, this tool has no fixed candidate prefix,
bank hash, row count, or 40-row assumption.  The sealed composition plan is
the profile.  Templates contain no labels; finalization accepts only detached
per-row sidecars whose reviewed media and generation receipt are hash-bound.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import mosaic_event_population_authoring as population  # noqa: E402


class MosaicAuditAuthoringCLIError(RuntimeError):
    pass


_BINDINGS_ROOT_FIELDS = {"schema_version", "banks"}
_BANK_FIELDS = {
    "binding_id",
    "root_spec",
    "root_spec_raw_sha256",
    "bank_output_dir",
    "bank_receipt",
    "bank_receipt_raw_sha256",
}
_BINDINGS_SCHEMA = "bernini-mosaic-authenticated-bank-bindings-v1"


def _reject_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MosaicAuditAuthoringCLIError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json(path_text: str, expected_sha256: str, *, label: str) -> Any:
    path = Path(path_text)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise MosaicAuditAuthoringCLIError(f"{label} must be an absolute plain file")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise MosaicAuditAuthoringCLIError(f"{label} raw SHA-256 differs")
    try:
        return json.loads(
            raw,
            object_pairs_hook=_reject_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                MosaicAuditAuthoringCLIError(f"non-finite JSON constant: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MosaicAuditAuthoringCLIError(f"{label} is invalid JSON") from error


def _write(path_text: str, value: Mapping[str, Any]) -> str:
    path = Path(path_text)
    if (
        not path.is_absolute()
        or path == Path("/")
        or path.exists()
        or path.is_symlink()
        or not path.parent.is_dir()
        or path.parent.is_symlink()
    ):
        raise MosaicAuditAuthoringCLIError("output must be a fresh absolute plain file")
    raw = population.canonical_json_bytes(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
    return hashlib.sha256(raw).hexdigest()


def _load_authenticated_bank_rows(
    bindings: Any, composition_value: Any
) -> list[dict[str, Any]]:
    """Use the existing strong bank verifier; never trust a row-only JSON."""

    composition = population.validate_composition_plan(composition_value)
    if not isinstance(bindings, Mapping) or set(bindings) != _BINDINGS_ROOT_FIELDS:
        raise MosaicAuditAuthoringCLIError("bank bindings root closure differs")
    if bindings["schema_version"] != _BINDINGS_SCHEMA:
        raise MosaicAuditAuthoringCLIError("bank bindings schema differs")
    banks = bindings["banks"]
    if not isinstance(banks, list) or len(banks) != 4:
        raise MosaicAuditAuthoringCLIError("compact composition requires four banks")
    expected_hashes = {
        profile[key]
        for profile in composition["inherited_bank_profiles"]
        for key in ("seed1_root_spec_raw_sha256", "seed2_root_spec_raw_sha256")
    }
    if None in expected_hashes:
        raise MosaicAuditAuthoringCLIError(
            "composition has an unresolved root spec hash"
        )
    try:
        scorer = importlib.import_module("score_pair_v5_t2v_energy_bank_v3")
    except (ImportError, ModuleNotFoundError) as error:
        raise MosaicAuditAuthoringCLIError(
            "strong bank profiling requires the pinned vace/Torch runtime"
        ) from error
    observed_hashes = set()
    all_rows: list[dict[str, Any]] = []
    binding_ids: set[str] = set()
    for index, raw_bank in enumerate(banks):
        if not isinstance(raw_bank, Mapping) or set(raw_bank) != _BANK_FIELDS:
            raise MosaicAuditAuthoringCLIError(
                f"bank binding {index} field closure differs"
            )
        binding_id = raw_bank["binding_id"]
        if (
            not isinstance(binding_id, str)
            or not binding_id
            or binding_id in binding_ids
        ):
            raise MosaicAuditAuthoringCLIError("bank binding IDs repeat or are invalid")
        binding_ids.add(binding_id)
        root_digest = raw_bank["root_spec_raw_sha256"]
        if root_digest not in expected_hashes or root_digest in observed_hashes:
            raise MosaicAuditAuthoringCLIError(
                "bank binding root hash is unregistered or repeated"
            )
        observed_hashes.add(root_digest)
        common_spec = None
        common_bank_digest = None
        for group_id in ("sp4-a", "sp4-b"):
            try:
                spec, bank, rows = scorer.load_group_bank(
                    root_spec=raw_bank["root_spec"],
                    root_spec_sha256=root_digest,
                    bank_output_dir=raw_bank["bank_output_dir"],
                    bank_receipt=raw_bank["bank_receipt"],
                    bank_receipt_sha256=raw_bank["bank_receipt_raw_sha256"],
                    group_id=group_id,
                )
            except scorer.PairV5T2VEnergyScoringError as error:
                raise MosaicAuditAuthoringCLIError(str(error)) from error
            if common_spec is not None and spec != common_spec:
                raise MosaicAuditAuthoringCLIError("two SP4 root specs differ")
            if (
                common_bank_digest is not None
                and bank.get("receipt_digest") != common_bank_digest
            ):
                raise MosaicAuditAuthoringCLIError("two SP4 bank receipts differ")
            common_spec = spec
            common_bank_digest = bank.get("receipt_digest")
            all_rows.extend(
                {"group_id": group_id, **dict(row)} for row in rows
            )
    if observed_hashes != expected_hashes:
        raise MosaicAuditAuthoringCLIError("bank binding set is incomplete")
    return all_rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    profile = subparsers.add_parser("profile-inherited")
    for name in ("registry", "composition", "bank-bindings"):
        profile.add_argument(f"--{name}", required=True)
        profile.add_argument(f"--expected-{name}-sha256", required=True)
    profile.add_argument("--output", required=True)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--audit-requests", required=True)
    finalize.add_argument("--expected-audit-requests-sha256", required=True)
    finalize.add_argument("--audit-dir", required=True)
    finalize.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "profile-inherited":
        registry, _ = population.load_sealed_registry(
            args.registry, args.expected_registry_sha256
        )
        composition = _load_json(
            args.composition,
            args.expected_composition_sha256,
            label="composition plan",
        )
        bindings = _load_json(
            args.bank_bindings,
            args.expected_bank_bindings_sha256,
            label="bank bindings",
        )
        bound_rows = _load_authenticated_bank_rows(bindings, composition)
        value = population.build_inherited_audit_requests_from_authenticated_rows(
            registry, composition, bound_rows
        )
    else:
        requests = _load_json(
            args.audit_requests,
            args.expected_audit_requests_sha256,
            label="audit request manifest",
        )
        checked = population.validate_audit_request_manifest(requests)
        audit_dir = Path(args.audit_dir)
        if not audit_dir.is_absolute() or not audit_dir.is_dir() or audit_dir.is_symlink():
            raise MosaicAuditAuthoringCLIError("audit-dir must be an absolute plain directory")
        sidecars = {}
        for request in checked["candidate_requests"]:
            path = audit_dir / request["audit_sidecar_basename"]
            if path.is_file() and not path.is_symlink():
                sidecars[request["candidate_id"]] = _load_json(
                    str(path), hashlib.sha256(path.read_bytes()).hexdigest(), label="audit sidecar"
                )
        value = population.build_eligibility_index(checked, sidecars)
    digest = _write(args.output, value)
    print(
        json.dumps(
            {
                "command": args.command,
                "output": args.output,
                "output_sha256": digest,
                "prompt_or_requested_branch_used_as_label": False,
                "editor_optimizer_authorized": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
