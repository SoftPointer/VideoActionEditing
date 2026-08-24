#!/usr/bin/env python3
"""Read-only, CPU-only preflight for the e02/e03 oracle canary.

Neither the experiment spec nor a path embedded in it may authorize release.
Launch readiness additionally requires a compiled independent release trust
anchor, compiled per-case annotation/execution receipts, runner allowlist and
ABI receipt, plus a pre-run side-by-side review policy. None is compiled into
this version, so CLI-provided hashes cannot unlock the checked-in scaffold.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence


HERE = Path(__file__).resolve()
METHOD_ROOT = HERE.parents[1]
REPO_ROOT = METHOD_ROOT.parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import oracle_regeneration_canary_v1 as oracle  # noqa: E402


SCHEMA_VERSION = "bernini-oracle-regeneration-preflight-v2"
SPEC_SCHEMA_VERSION = "bernini-oracle-regeneration-e02-e03-canary-spec-v2"
RELEASE_AUTHORITY_SCHEMA_VERSION = (
    "bernini-oracle-regeneration-external-release-authority-v1"
)
REVIEW_POLICY_RECEIPT_SCHEMA_VERSION = (
    "bernini-oracle-regeneration-side-by-side-review-policy-receipt-v1"
)
RUNNER_ABI_RECEIPT_SCHEMA_VERSION = "bernini-oracle-native-runner-abi-receipt-v1"

# Intentionally unset.  The CLI hash is only a redundancy check and can never
# establish trust.  Exact authority SHA pinning is not an activation design for
# this component-pinned graph because it creates a hash cycle.  A future,
# separately reviewed version must instead compile a verification public key.
COMPILED_RELEASE_AUTHORITY_SHA256: Optional[str] = None
ACTIVATION_IMPLEMENTED_IN_THIS_VERSION = False
FUTURE_ACTIVATION_BLOCKERS = (
    "native source-reference VAE provenance/encoder/source-frame digests and "
    "storage/content independence are not yet receipt-verified",
    "FlowEdit keyed-noise domain/seed/generator provenance and inequality from "
    "correlated source noise are not yet receipt-verified",
    "selected-case physical native/Flow binding receipt verification is not "
    "implemented by this preflight version",
    "release authorization needs a new compiled-public-key signature verifier "
    "without an authority/spec/component exact-SHA cycle",
    "future FlowEdit callable code/closure identity must be pinned by runner ABI",
)
EXPECTED_COMPONENT_PATHS = {
    "oracle_core": "methods/bernini_action_editing/oracle_regeneration_canary_v1.py",
    "native_five_forward_runtime": (
        "methods/bernini_action_editing/native_branch_homotopy_runtime_v1.py"
    ),
    "native_homotopy_schedule": (
        "methods/bernini_action_editing/native_branch_homotopy_v1.py"
    ),
    "external_authority_contract": (
        "methods/bernini_action_editing/assets/"
        "oracle_regeneration_external_authority_contract_v1.json"
    ),
    "preflight_core": (
        "methods/bernini_action_editing/tools/"
        "preflight_oracle_regeneration_e02_e03_canary_v1.py"
    ),
    "preflight_launcher": (
        "methods/bernini_action_editing/scripts/"
        "auh_preflight_oracle_regeneration_e02_e03_canary_v1.sh"
    ),
}


class PreflightError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_sha(value: Any, *, label: str) -> str:
    try:
        return oracle._require_sha256(value, label=label)
    except Exception as error:
        raise PreflightError(str(error)) from error


def _compiled_case_pin_present(
    values: Mapping[str, str], *, case_id: str, label: str
) -> bool:
    value = values.get(case_id)
    if value is None:
        return False
    _require_sha(value, label=f"compiled {case_id} {label}")
    return True


def _strict_load(path: Path, *, label: str) -> Any:
    try:
        return oracle.strict_json_load_path_v1(path, label=label)
    except Exception as error:
        raise PreflightError(str(error)) from error


def _validate_component_pins(spec: Mapping[str, Any]) -> Mapping[str, str]:
    pins = spec.get("component_pins")
    if not isinstance(pins, Mapping):
        raise PreflightError("component pins are absent")
    observed: dict[str, str] = {}
    for name, expected_relative in EXPECTED_COMPONENT_PATHS.items():
        pin = pins.get(name)
        if not isinstance(pin, Mapping) or pin.get("path") != expected_relative:
            raise PreflightError(f"{name} pinned path differs")
        expected_sha = _require_sha(pin.get("sha256"), label=f"{name} pin")
        path = REPO_ROOT / expected_relative
        if not path.is_file() or _sha(path) != expected_sha:
            raise PreflightError(f"{name} pinned bytes differ")
        observed[name] = expected_sha
    return observed


def _validate_case_instruction_bindings(row: Mapping[str, Any]) -> None:
    caption = row.get("action_caption")
    program = row.get("structured_action_program")
    if not isinstance(caption, str) or not caption.strip() or not isinstance(program, Mapping):
        raise PreflightError("case action caption/program is absent")
    caption_sha = hashlib.sha256(caption.encode("utf-8")).hexdigest()
    program_sha = hashlib.sha256(oracle.canonical_json_bytes_v1(program)).hexdigest()
    if (
        row.get("action_caption_sha256") != caption_sha
        or row.get("structured_action_program_sha256") != program_sha
    ):
        raise PreflightError("case action caption/program digest differs")
    _require_sha(row.get("source_sha256"), label="case source")
    _require_sha(row.get("anchor_sha256"), label="case anchor")


def _validate_review_policy_receipt(
    authority: Mapping[str, Any],
    *,
    spec_sha256: str,
) -> Mapping[str, Any]:
    path_value = authority.get("review_policy_receipt_path")
    expected_sha = _require_sha(
        authority.get("review_policy_receipt_sha256"), label="review-policy receipt"
    )
    if not isinstance(path_value, str):
        raise PreflightError("external review-policy receipt path is absent")
    path = Path(path_value)
    if not path.is_absolute() or not path.is_file() or _sha(path) != expected_sha:
        raise PreflightError("external review-policy receipt bytes differ")
    receipt = _strict_load(path, label="review-policy receipt")
    authority_id = authority.get("authority_id")
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("schema_version") != REVIEW_POLICY_RECEIPT_SCHEMA_VERSION
        or receipt.get("authority_id") != authority_id
        or receipt.get("spec_sha256") != spec_sha256
        or receipt.get("cases") != ["e02", "e03"]
        or receipt.get("output_mode")
        != "base_and_regeneration_side_by_side_no_auto_replacement"
        or receipt.get("background_cosine_may_select") is not False
        or receipt.get("automatic_base_replacement_authorized") is not False
        or receipt.get("e03_policy")
        != "explicit_human_abstain_or_strict_action_non_regression"
        or receipt.get("accepted") is not True
        or receipt.get("pre_run_policy_only_not_an_outcome_decision") is not True
    ):
        raise PreflightError("external side-by-side review-policy receipt differs")
    return {
        "authority_id": authority_id,
        "receipt_path": str(path),
        "receipt_sha256": expected_sha,
    }


def _validate_external_release_authority(
    path: Path,
    *,
    expected_sha256: str,
    spec_sha256: str,
    component_pins: Mapping[str, str],
) -> Mapping[str, Any]:
    expected_sha256 = _require_sha(expected_sha256, label="external release authority")
    if not path.is_absolute() or not path.is_file() or _sha(path) != expected_sha256:
        raise PreflightError("external release authority bytes differ")
    authority = _strict_load(path, label="external release authority")
    if (
        not isinstance(authority, Mapping)
        or authority.get("schema_version") != RELEASE_AUTHORITY_SCHEMA_VERSION
        or authority.get("spec_sha256") != spec_sha256
        or authority.get("component_sha256") != dict(component_pins)
        or not isinstance(authority.get("authority_id"), str)
        or not authority.get("authority_id")
    ):
        raise PreflightError("external release authority identity/pins differ")
    roots = authority.get("annotation_ledger_roots")
    if not isinstance(roots, Mapping) or set(roots) != set(oracle.ALLOWED_CASES):
        raise PreflightError("external per-case annotation roots differ")
    for case_id in oracle.ALLOWED_CASES:
        _require_sha(roots.get(case_id), label=f"{case_id} annotation root")
    allowlist = authority.get("runner_allowlist")
    if not isinstance(allowlist, list) or not allowlist:
        raise PreflightError("external runner allowlist is absent")
    normalized_allowlist = []
    for item in allowlist:
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256"}:
            raise PreflightError("external runner allowlist entry differs")
        item_path = item.get("path")
        if not isinstance(item_path, str) or not Path(item_path).is_absolute():
            raise PreflightError("allowlisted runner path must be absolute")
        normalized_allowlist.append(
            {
                "path": item_path,
                "sha256": _require_sha(item.get("sha256"), label="runner"),
            }
        )
    abi_entries = authority.get("runner_abi_receipts")
    if not isinstance(abi_entries, list) or not abi_entries:
        raise PreflightError("external runner ABI receipts are absent")
    normalized_abi = []
    for item in abi_entries:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"runner_sha256", "receipt_path", "receipt_sha256"}
        ):
            raise PreflightError("external runner ABI receipt entry differs")
        receipt_path = item.get("receipt_path")
        if not isinstance(receipt_path, str) or not Path(receipt_path).is_absolute():
            raise PreflightError("runner ABI receipt path must be absolute")
        normalized_abi.append(
            {
                "runner_sha256": _require_sha(
                    item.get("runner_sha256"), label="ABI-bound runner"
                ),
                "receipt_path": receipt_path,
                "receipt_sha256": _require_sha(
                    item.get("receipt_sha256"), label="runner ABI receipt"
                ),
            }
        )
    review_policy = _validate_review_policy_receipt(
        authority, spec_sha256=spec_sha256
    )
    return {
        "authority_id": authority["authority_id"],
        "authority_path": str(path),
        "authority_sha256": expected_sha256,
        "annotation_ledger_roots": dict(roots),
        "runner_allowlist": normalized_allowlist,
        "runner_abi_receipts": normalized_abi,
        "review_policy": review_policy,
    }


def _validate_runner_abi_receipt(
    external_authority: Mapping[str, Any],
    *,
    runner_path: str,
    runner_sha256: str,
    component_pins: Mapping[str, str],
) -> Mapping[str, Any]:
    matches = [
        item
        for item in external_authority["runner_abi_receipts"]
        if item["runner_sha256"] == runner_sha256
    ]
    if len(matches) != 1:
        raise PreflightError("runner has no unique externally pinned ABI receipt")
    pin = matches[0]
    path = Path(pin["receipt_path"])
    if not path.is_file() or _sha(path) != pin["receipt_sha256"]:
        raise PreflightError("runner ABI receipt bytes differ")
    receipt = _strict_load(path, label="runner ABI receipt")
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("schema_version") != RUNNER_ABI_RECEIPT_SCHEMA_VERSION
        or receipt.get("runner_path") != runner_path
        or receipt.get("runner_sha256") != runner_sha256
        or receipt.get("component_sha256") != dict(component_pins)
        or receipt.get("supported_execution") != ["native_local_r2v4"]
        or receipt.get("native_binding_schema")
        != oracle.NATIVE_BINDING_RECEIPT_SCHEMA_VERSION
        or receipt.get("world_size") != 4
        or receipt.get("topology") != "one_node_one_sp4"
        or receipt.get("same_seed_and_initial_noise_for_base_and_regen") is not True
        or receipt.get("emits_official_base_and_local_regen_side_by_side") is not True
        or receipt.get("connected_flowedit_enabled") is not False
        or receipt.get("training_or_optimizer_available") is not False
        or receipt.get("accepted") is not True
    ):
        raise PreflightError("runner ABI receipt contract differs")
    return {
        "receipt_path": str(path),
        "receipt_sha256": pin["receipt_sha256"],
        "runner_sha256": runner_sha256,
    }


def preflight(
    spec_path: Path,
    *,
    case_id: str,
    execution: str,
    require_launch_ready: bool,
    release_authority_path: Optional[Path] = None,
    expected_release_authority_sha256: Optional[str] = None,
) -> Mapping[str, Any]:
    if not spec_path.is_absolute() or not spec_path.is_file():
        raise PreflightError("spec must be an existing absolute regular file")
    spec = _strict_load(spec_path, label="spec")
    if not isinstance(spec, Mapping) or spec.get("schema_version") != SPEC_SCHEMA_VERSION:
        raise PreflightError("oracle canary spec schema differs")
    if case_id not in oracle.ALLOWED_CASES:
        raise PreflightError("case must be e02 or e03")
    if execution not in ("native_local_r2v4", "flowedit_step0_noise", "connected"):
        raise PreflightError("execution choice differs")
    if (
        spec.get("training_authorized") is not False
        or spec.get("optimizer_authorized") is not False
        or spec.get("automatic_model_replacement_authorized") is not False
    ):
        raise PreflightError(
            "training/optimizer/automatic replacement must remain hard-disabled"
        )
    if spec.get("scientific_role") != "diagnostic_oracle_intervention_not_training":
        raise PreflightError("diagnostic-only scientific role differs")
    if type(spec.get("gpu_launch_authorized")) is not bool:
        raise PreflightError("gpu launch authorization must be exact boolean")
    selection_contract = spec.get("selection_contract")
    if (
        not isinstance(selection_contract, Mapping)
        or selection_contract.get("selection_authority") is not None
        or selection_contract.get("pre_run_review_policy_authority_source")
        != "future compiled signed release authority only"
        or selection_contract.get("post_run_outcome_selection_authority") is not None
        or selection_contract.get("background_cosine_may_choose_final_candidate")
        is not False
        or selection_contract.get(
            "base_and_regeneration_outputs_required_side_by_side"
        )
        is not True
        or selection_contract.get("successful_base_may_be_replaced_automatically")
        is not False
        or selection_contract.get(
            "e03_requires_explicit_abstain_or_non_regression_decision"
        )
        is not True
    ):
        raise PreflightError("fail-closed selection contract differs")
    component_pins = _validate_component_pins(spec)
    spec_sha256 = _sha(spec_path)
    blockers: list[str] = []
    if not ACTIVATION_IMPLEMENTED_IN_THIS_VERSION:
        blockers.append(
            "activation is hard-disabled in this preflight version; a new "
            "signed-authority implementation and audit are required"
        )
    external_authority = None
    if COMPILED_RELEASE_AUTHORITY_SHA256 is None:
        blockers.append(
            "compiled independent release-authority trust anchor is unset; "
            "caller CLI hashes cannot authorize this version"
        )
    elif release_authority_path is None or expected_release_authority_sha256 is None:
        blockers.append("compiled release authority bytes were not supplied")
    else:
        if expected_release_authority_sha256 != COMPILED_RELEASE_AUTHORITY_SHA256:
            raise PreflightError(
                "CLI release-authority hash differs from compiled trust anchor"
            )
        external_authority = _validate_external_release_authority(
            release_authority_path,
            expected_sha256=COMPILED_RELEASE_AUTHORITY_SHA256,
            spec_sha256=spec_sha256,
            component_pins=component_pins,
        )
        if external_authority["annotation_ledger_roots"] != dict(
            oracle.COMPILED_ANNOTATION_AUTHORITY_ROOTS
        ):
            raise PreflightError(
                "external annotation roots differ from compiled core roots"
            )
    if not _compiled_case_pin_present(
        oracle.COMPILED_ANNOTATION_AUTHORITY_ROOTS,
        case_id=case_id,
        label="annotation authority root",
    ):
        blockers.append(f"compiled {case_id} annotation authority root is absent")
    if execution in ("native_local_r2v4", "connected") and not (
        _compiled_case_pin_present(
            oracle.COMPILED_NATIVE_BINDING_RECEIPT_SHA256,
            case_id=case_id,
            label="native execution-binding receipt",
        )
    ):
        blockers.append(f"compiled {case_id} native execution-binding receipt is absent")
    if execution in ("flowedit_step0_noise", "connected") and not (
        _compiled_case_pin_present(
            oracle.COMPILED_FLOWEDIT_BINDING_RECEIPT_SHA256,
            case_id=case_id,
            label="FlowEdit execution-binding receipt",
        )
    ):
        blockers.append(f"compiled {case_id} FlowEdit execution-binding receipt is absent")
    cases = spec.get("cases")
    if not isinstance(cases, list):
        raise PreflightError("spec cases are absent")
    rows = [
        row
        for row in cases
        if isinstance(row, Mapping) and row.get("case_id") == case_id
    ]
    if len(rows) != 1:
        raise PreflightError("case closure differs")
    row = rows[0]
    _validate_case_instruction_bindings(row)
    e03_rows = [
        value
        for value in cases
        if isinstance(value, Mapping) and value.get("case_id") == "e03"
    ]
    if (
        len(e03_rows) != 1
        or "abstain" not in str(e03_rows[0].get("candidate_policy", "")).lower()
        or "non-regression"
        not in str(e03_rows[0].get("candidate_policy", "")).lower()
    ):
        raise PreflightError("e03 explicit abstain/non-regression policy differs")
    validated_gate = None
    gate_values = (
        row.get("manual_gate_manifest"),
        row.get("manual_gate_manifest_sha256"),
        row.get("manual_review_receipt"),
        row.get("manual_review_receipt_sha256"),
    )
    if not all(isinstance(value, str) and value for value in gate_values):
        blockers.append("reviewed manual exact-bool D/C gate and receipt are not bound")
    elif external_authority is None:
        blockers.append("manual gate cannot validate without external annotation root")
    else:
        gate_path, gate_sha, receipt_path, receipt_sha = gate_values
        if str(Path(gate_path)) == str(Path(receipt_path)):
            raise PreflightError("gate and review receipt must be distinct files")
        try:
            validated_gate = oracle.validate_oracle_gate_manifest_v1(
                Path(gate_path),
                expected_file_sha256=gate_sha,
                expected_review_receipt_sha256=receipt_sha,
                expected_case_id=case_id,
                expected_source_sha256=str(row.get("source_sha256")),
                expected_anchor_sha256=str(row.get("anchor_sha256")),
                expected_action_caption_sha256=str(
                    row.get("action_caption_sha256")
                ),
                expected_structured_action_program_sha256=str(
                    row.get("structured_action_program_sha256")
                ),
                expected_annotation_authority_root_sha256=(
                    external_authority["annotation_ledger_roots"][case_id]
                ),
                expected_latent_geometry=tuple(row.get("latent_geometry")),
            )
        except Exception as error:
            raise PreflightError(str(error)) from error
        if Path(receipt_path) != validated_gate.review_receipt_path:
            raise PreflightError("spec/gate review receipt paths differ")
    execution_rows = spec.get("execution_choices")
    selected = execution_rows.get(execution) if isinstance(execution_rows, Mapping) else None
    if execution == "connected":
        blockers.append("native-local and FlowEdit seams have no audited connected runner")
    elif not isinstance(selected, Mapping) or selected.get("implemented_core") is not True:
        blockers.append(f"{execution} core is not implemented")
    if execution == "flowedit_step0_noise":
        blockers.append(
            "case-specific pinned FlowEdit constructor/noise execution receipt is absent"
        )
    runner = spec.get("runner_entrypoint")
    runner_sha = spec.get("runner_entrypoint_sha256")
    runner_abi_receipt = None
    if not isinstance(runner, str) or not isinstance(runner_sha, str):
        blockers.append("real frozen outer-sampler runner is not bound")
    else:
        _require_sha(runner_sha, label="runner entrypoint")
        runner_path = Path(runner)
        if (
            not runner_path.is_absolute()
            or not runner_path.is_file()
            or _sha(runner_path) != runner_sha
        ):
            raise PreflightError("bound outer-sampler runner bytes differ")
        if external_authority is None or {
            "path": runner,
            "sha256": runner_sha,
        } not in external_authority["runner_allowlist"]:
            blockers.append("runner is not included in caller-pinned external allowlist")
        else:
            runner_abi_receipt = _validate_runner_abi_receipt(
                external_authority,
                runner_path=runner,
                runner_sha256=runner_sha,
                component_pins=component_pins,
            )
    if spec.get("gpu_launch_authorized") is not True:
        blockers.append("spec does not authorize GPU launch")
    if spec.get("status") != "EXTERNALLY_AUTHORIZED_NATIVE_LOCAL_CANARY":
        blockers.append("spec status is not external native-local authorization")
    report = {
        "schema_version": SCHEMA_VERSION,
        "spec_path": str(spec_path),
        "spec_sha256": spec_sha256,
        "case_id": case_id,
        "execution": execution,
        "component_pins_validated": component_pins,
        "activation_implemented_in_this_version": (
            ACTIVATION_IMPLEMENTED_IN_THIS_VERSION
        ),
        "future_activation_blockers": list(FUTURE_ACTIVATION_BLOCKERS),
        "compiled_release_authority_trust_anchor_present": (
            COMPILED_RELEASE_AUTHORITY_SHA256 is not None
        ),
        "external_release_authority_validated": external_authority is not None,
        "external_release_authority_sha256": (
            external_authority["authority_sha256"] if external_authority else None
        ),
        "manual_gate_validated": validated_gate is not None,
        "hard_gate_dtype": "bool" if validated_gate is not None else None,
        "review_policy_authority": (
            external_authority["review_policy"] if external_authority else None
        ),
        "selection_or_outcome_authority": None,
        "runner_abi_receipt": runner_abi_receipt,
        "output_mode": "base_and_regeneration_side_by_side_no_auto_replacement",
        "automatic_base_replacement_authorized": selection_contract[
            "successful_base_may_be_replaced_automatically"
        ],
        "background_cosine_may_choose_final_candidate": selection_contract[
            "background_cosine_may_choose_final_candidate"
        ],
        "training_authorized": spec["training_authorized"],
        "optimizer_authorized": spec["optimizer_authorized"],
        "gpu_launch_authorized": spec.get("gpu_launch_authorized"),
        "launch_ready": not blockers,
        "blockers": blockers,
    }
    if require_launch_ready and blockers:
        raise PreflightError("; ".join(blockers))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--case", required=True, choices=oracle.ALLOWED_CASES)
    parser.add_argument(
        "--execution",
        required=True,
        choices=("native_local_r2v4", "flowedit_step0_noise", "connected"),
    )
    parser.add_argument("--release-authority")
    parser.add_argument("--expected-release-authority-sha256")
    parser.add_argument("--require-launch-ready", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = preflight(
            Path(args.spec),
            case_id=args.case,
            execution=args.execution,
            require_launch_ready=args.require_launch_ready,
            release_authority_path=(
                Path(args.release_authority) if args.release_authority else None
            ),
            expected_release_authority_sha256=(
                args.expected_release_authority_sha256
            ),
        )
    except PreflightError as error:
        print(f"BLOCKED: {error}", file=sys.stderr)
        return 3
    print(oracle.canonical_json_bytes_v1(report).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
