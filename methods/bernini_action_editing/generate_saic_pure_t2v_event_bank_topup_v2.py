#!/usr/bin/env python3
"""Render and fail-closed audit the SAIC pure-T2V v2 top-up.

The native Bernini invocation is delegated to the already-tested v1 renderer.
Only v2 envelope/spec validation and v2 closed receipts are new.  Every call is
the official T2V arm with a launch-local constant-black exact81 shape proxy;
real source media, latents, noise, targets, references, and donors are absent.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(METHOD_ROOT))

import generate_saic_pure_t2v_event_bank_v1 as v1_generate  # noqa: E402
import saic_pure_t2v_event_bank_topup_v2 as contract  # noqa: E402


SCHEMA_VERSION = "bernini-saic-pure-t2v-event-topup-generation-receipt-v2"
MASTER_SCHEMA_VERSION = "bernini-saic-pure-t2v-event-bank-topup-receipt-v2"
ATTEMPT_RECEIPT_BASENAME = "saic-event-topup-generation-receipt.json"
MASTER_RECEIPT_BASENAME = "saic-pure-t2v-event-bank-topup-receipt.json"


class SAICPureT2VTopupGenerationError(RuntimeError):
    """Raised before unauthenticated v2 proposal evidence is accepted."""


def _wrap(message: str, error: Exception | None = None) -> SAICPureT2VTopupGenerationError:
    result = SAICPureT2VTopupGenerationError(message)
    if error is not None:
        result.__cause__ = error
    return result


def _plain_file(value: str | Path, *, label: str) -> Path:
    try:
        return v1_generate._plain_file(value, label=label)
    except v1_generate.SAICPureT2VGenerationError as error:
        raise _wrap(f"{label} must be an absolute plain file", error)


def _plain_dir(value: str | Path, *, label: str) -> Path:
    try:
        return v1_generate._plain_dir(value, label=label)
    except v1_generate.SAICPureT2VGenerationError as error:
        raise _wrap(f"{label} must be an absolute plain directory", error)


def file_sha256(path: str | Path) -> str:
    return v1_generate.file_sha256(path)


def _load_json(path: str | Path, *, label: str) -> dict[str, Any]:
    try:
        return v1_generate._load_json(path, label=label)
    except v1_generate.SAICPureT2VGenerationError as error:
        raise _wrap(f"cannot load {label}", error)


def _closed(value: Any, fields: set[str], *, label: str) -> Mapping[str, Any]:
    try:
        return contract._closed(value, fields, label=label)
    except contract.SAICPureT2VEventBankTopupError as error:
        raise _wrap(f"{label} keys differ", error)


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    try:
        contract._write_create_only(path, value)
    except contract.SAICPureT2VEventBankTopupError as error:
        raise _wrap(f"refusing to overwrite {path}", error)


def build_native_argv(args: argparse.Namespace, envelope: Mapping[str, Any]) -> list[str]:
    """Reuse the v1 native T2V entry and assert the v2 nonuse closure."""

    try:
        argv = v1_generate.build_native_argv(args, envelope)
    except v1_generate.SAICPureT2VGenerationError as error:
        raise _wrap("v2 native argv failed closed", error)
    if argv[argv.index("--arms") + 1] != "t2v":
        raise SAICPureT2VTopupGenerationError("v2 must use the pure T2V arm")
    if argv[argv.index("--num-inference-steps") + 1] != "40":
        raise SAICPureT2VTopupGenerationError("v2 must use forty inference steps")
    forbidden = {
        "--target-video", "--reference-image", "--reference-video", "--mask",
        "--flow", "--pose", "--track", "--motion-donor", "--source-latent",
        "--source-noise",
    }
    if forbidden.intersection(argv):
        raise SAICPureT2VTopupGenerationError("privileged input leaked into v2 argv")
    return argv


_ATTEMPT_FIELDS = {
    "schema_version",
    "bank_id",
    "top_up_only",
    "root_spec_raw_sha256",
    "base_v1_spec_raw_sha256",
    "candidate_envelope_path",
    "candidate_envelope_sha256",
    "group_id",
    "actor_family",
    "visible_gpus",
    "candidate",
    "sampling_contract",
    "semantic_input_closure",
    "geometry_proxy_contract",
    "artifact_authority",
    "runtime_topology",
    "real_source_nonuse_certificate",
    "native_receipt_path",
    "native_receipt_sha256",
    "native_receipt_digest",
    "bucket_hw",
    "latent_shape",
    "artifacts",
    "event_audit_status",
    "event_verified",
    "identity_preservation_verified",
    "seed_selection_authorized",
    "training_target_authorized",
    "optimizer_or_parameter_update_authorized",
    "receipt_digest",
}
_NONUSE_FIELDS = {
    "source_media_sha256_for_nonuse_audit",
    "real_source_path_present_in_candidate_envelope",
    "real_source_path_passed_to_native_runner",
    "real_source_rgb_read",
    "real_source_latent_read_or_created",
    "real_source_noise_read_or_created",
    "target_video_read_or_created",
    "reference_image_or_video_read",
    "motion_donor_read",
    "geometry_proxy_path",
    "geometry_proxy_sha256",
    "proxy_bytes_differ_from_real_source",
    "proxy_used_for_bucket_shape_only",
    "proxy_vae_latent_created",
    "proxy_pixels_entered_transformer",
    "native_content_conditioning_count",
}


def bind_attempt_receipt(
    *, args: argparse.Namespace, envelope: Mapping[str, Any], envelope_path: Path
) -> Path:
    output = _plain_dir(args.output_dir, label="v2 attempt output")
    native_path = _plain_file(output / "receipt.json", label="v2 native receipt")
    candidate = envelope["candidate"]
    proxy = envelope["geometry_proxy"]
    try:
        native_value = v1_generate.native_receipts._load_json(
            native_path, "v2 native receipt"
        )
        verified = v1_generate.native_receipts._verify_native_receipt(
            native_value,
            {
                "geometry_source_video_sha256": proxy["sha256"],
                "full_t2v_caption_utf8_sha256": candidate[
                    "full_t2v_caption_utf8_sha256"
                ],
                "seed": candidate["seed"],
            },
        )
    except v1_generate.pair_contract.PairT2VCalibrationSpecError as error:
        raise _wrap("native exact81 T2V evidence failed verification", error)

    canonical_proxy_path = str(
        _plain_file(proxy["path"], label="v2 receipt-bound black proxy")
    )
    native_input = native_value.get("input")
    if (
        not isinstance(native_input, Mapping)
        or native_input.get("source_video_path") != canonical_proxy_path
        or native_input.get("source_video_sha256") != proxy["sha256"]
        or proxy["sha256"] == candidate["source_media_sha256_for_nonuse_audit"]
    ):
        raise SAICPureT2VTopupGenerationError("v2 native runner did not bind black proxy")

    try:
        artifacts = {
            "mp4": v1_generate._artifact_inside(
                verified["mp4"], output, label="v2 proposal MP4"
            ),
            "predecode_clean_latent": v1_generate._artifact_inside(
                verified["predecode_clean_latent"],
                output,
                label="v2 proposal clean latent",
            ),
            "official_initial_gaussian": v1_generate._artifact_inside(
                verified["official_initial_gaussian"],
                output,
                label="v2 official Gaussian",
            ),
        }
    except v1_generate.SAICPureT2VGenerationError as error:
        raise _wrap("v2 native artifact escaped attempt root", error)

    nonuse = {
        "source_media_sha256_for_nonuse_audit": candidate[
            "source_media_sha256_for_nonuse_audit"
        ],
        "real_source_path_present_in_candidate_envelope": False,
        "real_source_path_passed_to_native_runner": False,
        "real_source_rgb_read": False,
        "real_source_latent_read_or_created": False,
        "real_source_noise_read_or_created": False,
        "target_video_read_or_created": False,
        "reference_image_or_video_read": False,
        "motion_donor_read": False,
        "geometry_proxy_path": canonical_proxy_path,
        "geometry_proxy_sha256": proxy["sha256"],
        "proxy_bytes_differ_from_real_source": True,
        "proxy_used_for_bucket_shape_only": True,
        "proxy_vae_latent_created": False,
        "proxy_pixels_entered_transformer": False,
        "native_content_conditioning_count": 0,
    }
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "bank_id": contract.BANK_ID,
        "top_up_only": True,
        "root_spec_raw_sha256": envelope["root_spec_raw_sha256"],
        "base_v1_spec_raw_sha256": contract.BASE_V1_SPEC_RAW_SHA256,
        "candidate_envelope_path": str(envelope_path),
        "candidate_envelope_sha256": file_sha256(envelope_path),
        "group_id": envelope["group_id"],
        "actor_family": envelope["actor_family"],
        "visible_gpus": envelope["visible_gpus"],
        "candidate": candidate,
        "sampling_contract": contract.SAMPLING_CONTRACT,
        "semantic_input_closure": contract.SEMANTIC_INPUT_CLOSURE,
        "geometry_proxy_contract": contract.GEOMETRY_PROXY_CONTRACT,
        "artifact_authority": contract.ARTIFACT_AUTHORITY,
        "runtime_topology": {
            "world_size": 4,
            "ulysses_size": 4,
            "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
        },
        "real_source_nonuse_certificate": nonuse,
        "native_receipt_path": str(native_path),
        "native_receipt_sha256": file_sha256(native_path),
        "native_receipt_digest": verified["native_receipt_digest"],
        "bucket_hw": verified["bucket_hw"],
        "latent_shape": verified["latent_shape"],
        "artifacts": artifacts,
        "event_audit_status": "pending_detached_full81_review",
        "event_verified": False,
        "identity_preservation_verified": False,
        "seed_selection_authorized": False,
        "training_target_authorized": False,
        "optimizer_or_parameter_update_authorized": False,
    }
    receipt = {**unsigned, "receipt_digest": contract.object_sha256(unsigned)}
    path = output / ATTEMPT_RECEIPT_BASENAME
    _write_create_only(path, receipt)
    return path


def generate_attempt(args: argparse.Namespace) -> int:
    envelope_path = _plain_file(args.candidate_envelope, label="v2 candidate envelope")
    try:
        envelope = contract.load_candidate_envelope(
            envelope_path,
            expected_root_spec_sha256=args.expected_root_spec_sha256,
        )
        spec, _ = contract.load_sealed_spec(
            args.root_spec,
            expected_raw_sha256=args.expected_root_spec_sha256,
            source_manifest_path=args.source_manifest,
            base_v1_spec_path=args.base_v1_spec,
        )
    except contract.SAICPureT2VEventBankTopupError as error:
        raise _wrap("v2 candidate envelope failed closed", error)
    matching_groups = [
        group for group in spec["groups"] if group["group_id"] == envelope["group_id"]
    ]
    if len(matching_groups) != 1:
        raise SAICPureT2VTopupGenerationError("v2 envelope group is absent from sealed spec")
    matching_candidates = [
        candidate
        for candidate in matching_groups[0]["candidates"]
        if candidate["candidate_id"] == envelope["candidate"]["candidate_id"]
    ]
    if len(matching_candidates) != 1 or matching_candidates[0] != envelope["candidate"]:
        raise SAICPureT2VTopupGenerationError(
            "v2 envelope candidate is not byte-semantic-equal to sealed spec"
        )
    expected_visible = ",".join(str(item) for item in envelope["visible_gpus"])
    if (
        os.environ.get("WORLD_SIZE") != "4"
        or os.environ.get("ROCR_VISIBLE_DEVICES") != expected_visible
    ):
        raise SAICPureT2VTopupGenerationError("v2 attempt requires sealed WORLD4/SP4")
    output = Path(args.output_dir)
    if (
        not output.is_absolute()
        or output == Path("/")
        or output.exists()
        or output.is_symlink()
    ):
        raise SAICPureT2VTopupGenerationError("v2 attempt output must be fresh and absolute")
    status = v1_generate.native.main(build_native_argv(args, envelope))
    if status == 0 and int(os.environ.get("RANK", "0")) == 0:
        bind_attempt_receipt(args=args, envelope=envelope, envelope_path=envelope_path)
    return status


def _artifact_inside(value: Any, root: Path, *, label: str) -> dict[str, Any]:
    try:
        return v1_generate._artifact_inside(value, root, label=label)
    except v1_generate.SAICPureT2VGenerationError as error:
        raise _wrap(f"{label} differs", error)


def _load_attempt_receipt(
    path: Path,
    *,
    candidate: Mapping[str, Any],
    group: Mapping[str, Any],
    root_spec_sha256: str,
    real_source_paths: set[str],
    real_source_hashes: set[str],
) -> dict[str, Any]:
    receipt = _load_json(path, label="v2 attempt receipt")
    _closed(receipt, _ATTEMPT_FIELDS, label="v2 attempt receipt")
    unsigned = dict(receipt)
    declared = unsigned.pop("receipt_digest")
    expected_visible = ",".join(str(item) for item in group["visible_gpus"])
    if (
        receipt["schema_version"] != SCHEMA_VERSION
        or receipt["bank_id"] != contract.BANK_ID
        or receipt["top_up_only"] is not True
        or declared != contract.object_sha256(unsigned)
        or receipt["root_spec_raw_sha256"] != root_spec_sha256
        or receipt["base_v1_spec_raw_sha256"] != contract.BASE_V1_SPEC_RAW_SHA256
        or receipt["candidate"] != candidate
        or receipt["group_id"] != group["group_id"]
        or receipt["actor_family"] != group["actor_family"]
        or receipt["visible_gpus"] != group["visible_gpus"]
        or receipt["sampling_contract"] != contract.SAMPLING_CONTRACT
        or receipt["semantic_input_closure"] != contract.SEMANTIC_INPUT_CLOSURE
        or receipt["geometry_proxy_contract"] != contract.GEOMETRY_PROXY_CONTRACT
        or receipt["artifact_authority"] != contract.ARTIFACT_AUTHORITY
        or receipt["runtime_topology"]
        != {
            "world_size": 4,
            "ulysses_size": 4,
            "rocr_visible_devices": expected_visible,
        }
        or receipt["event_audit_status"] != "pending_detached_full81_review"
        or receipt["event_verified"] is not False
        or receipt["identity_preservation_verified"] is not False
        or receipt["seed_selection_authorized"] is not False
        or receipt["training_target_authorized"] is not False
        or receipt["optimizer_or_parameter_update_authorized"] is not False
    ):
        raise SAICPureT2VTopupGenerationError("v2 attempt/spec binding differs")

    nonuse = _closed(
        receipt["real_source_nonuse_certificate"],
        _NONUSE_FIELDS,
        label="v2 source-nonuse certificate",
    )
    if (
        nonuse["source_media_sha256_for_nonuse_audit"]
        != candidate["source_media_sha256_for_nonuse_audit"]
        or nonuse["real_source_path_present_in_candidate_envelope"] is not False
        or nonuse["real_source_path_passed_to_native_runner"] is not False
        or nonuse["real_source_rgb_read"] is not False
        or nonuse["real_source_latent_read_or_created"] is not False
        or nonuse["real_source_noise_read_or_created"] is not False
        or nonuse["target_video_read_or_created"] is not False
        or nonuse["reference_image_or_video_read"] is not False
        or nonuse["motion_donor_read"] is not False
        or nonuse["proxy_bytes_differ_from_real_source"] is not True
        or nonuse["proxy_used_for_bucket_shape_only"] is not True
        or nonuse["proxy_vae_latent_created"] is not False
        or nonuse["proxy_pixels_entered_transformer"] is not False
        or nonuse["native_content_conditioning_count"] != 0
        or nonuse["geometry_proxy_path"] in real_source_paths
        or nonuse["geometry_proxy_sha256"] in real_source_hashes
    ):
        raise SAICPureT2VTopupGenerationError("v2 source-nonuse certificate differs")

    envelope_path = _plain_file(
        receipt["candidate_envelope_path"], label="v2 receipt-bound envelope"
    )
    if file_sha256(envelope_path) != receipt["candidate_envelope_sha256"]:
        raise SAICPureT2VTopupGenerationError("v2 envelope SHA-256 differs")
    try:
        envelope = contract.load_candidate_envelope(
            envelope_path, expected_root_spec_sha256=root_spec_sha256
        )
    except contract.SAICPureT2VEventBankTopupError as error:
        raise _wrap("v2 envelope re-audit failed", error)
    if (
        envelope["candidate"] != candidate
        or envelope["group_id"] != group["group_id"]
        or envelope["visible_gpus"] != group["visible_gpus"]
        or envelope["geometry_proxy"]["path"] in real_source_paths
        or envelope["geometry_proxy"]["sha256"] in real_source_hashes
    ):
        raise SAICPureT2VTopupGenerationError("v2 envelope source closure differs")

    native_path = _plain_file(receipt["native_receipt_path"], label="v2 native receipt")
    if file_sha256(native_path) != receipt["native_receipt_sha256"]:
        raise SAICPureT2VTopupGenerationError("v2 native receipt SHA-256 differs")
    native_value = v1_generate.native_receipts._load_json(native_path, "v2 native receipt")
    try:
        verified = v1_generate.native_receipts._verify_native_receipt(
            native_value,
            {
                "geometry_source_video_sha256": nonuse["geometry_proxy_sha256"],
                "full_t2v_caption_utf8_sha256": candidate[
                    "full_t2v_caption_utf8_sha256"
                ],
                "seed": candidate["seed"],
            },
        )
    except v1_generate.pair_contract.PairT2VCalibrationSpecError as error:
        raise _wrap("v2 native receipt re-audit failed", error)
    if (
        verified["native_receipt_digest"] != receipt["native_receipt_digest"]
        or native_value.get("input", {}).get("source_video_path")
        != nonuse["geometry_proxy_path"]
    ):
        raise SAICPureT2VTopupGenerationError("v2 native proxy binding differs")
    expected_artifacts = {
        "mp4": verified["mp4"],
        "predecode_clean_latent": verified["predecode_clean_latent"],
        "official_initial_gaussian": verified["official_initial_gaussian"],
    }
    if receipt["artifacts"] != expected_artifacts:
        raise SAICPureT2VTopupGenerationError("v2 native artifacts differ")
    for label, artifact in receipt["artifacts"].items():
        _artifact_inside(artifact, path.parent, label=f"{candidate['candidate_id']} {label}")
    return receipt


_MASTER_FIELDS = {
    "schema_version",
    "bank_id",
    "top_up_only",
    "root_spec_raw_sha256",
    "base_v1_spec_raw_sha256",
    "base_v1_spec_content_sha256",
    "source_manifest_content_sha256",
    "topology",
    "sampling_contract",
    "semantic_input_closure",
    "geometry_proxy_contract",
    "artifact_authority",
    "attempt_count",
    "row_count",
    "seed_cell_count",
    "branch_order",
    "merged_branch_order",
    "six_branch_spec_merge_cell_count",
    "same_seed_official_gaussian_proofs",
    "attempts",
    "detached_full81_event_review_complete",
    "event_verified",
    "identity_preservation_verified",
    "seed_selection_authorized",
    "training_target_authorized",
    "optimizer_or_parameter_update_authorized",
    "receipt_digest",
}


def audit_bank(args: argparse.Namespace) -> int:
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
        raise _wrap("v2 root/source audit failed", error)

    real_source_paths = {row["source_video"] for row in source_manifest["rows"]}
    real_source_hashes = {row["source_video_sha256"] for row in source_manifest["rows"]}
    root = _plain_dir(args.output_root, label="v2 event-bank output root")
    attempts_root = _plain_dir(root / "attempts", label="v2 attempts root")

    attempt_rows: list[dict[str, Any]] = []
    gaussian_cells: dict[tuple[str, int], list[tuple[str, Mapping[str, Any]]]] = {}
    for group in spec["groups"]:
        for candidate in group["candidates"]:
            receipt_path = (
                attempts_root / candidate["candidate_id"] / ATTEMPT_RECEIPT_BASENAME
            )
            receipt = _load_attempt_receipt(
                receipt_path,
                candidate=candidate,
                group=group,
                root_spec_sha256=root_sha,
                real_source_paths=real_source_paths,
                real_source_hashes=real_source_hashes,
            )
            gaussian = receipt["artifacts"]["official_initial_gaussian"]
            gaussian_cells.setdefault((candidate["iid"], candidate["seed"]), []).append(
                (candidate["branch"], gaussian)
            )
            attempt_rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "row_id": candidate["row_id"],
                    "iid": candidate["iid"],
                    "analysis_split": candidate["analysis_split"],
                    "branch": candidate["branch"],
                    "seed": candidate["seed"],
                    "receipt_path": str(receipt_path),
                    "receipt_sha256": file_sha256(receipt_path),
                    "receipt_digest": receipt["receipt_digest"],
                    "mp4_path": receipt["artifacts"]["mp4"]["path"],
                    "mp4_sha256": receipt["artifacts"]["mp4"]["sha256"],
                    "event_audit_status": "pending_detached_full81_review",
                }
            )
    if len(attempt_rows) != 60 or len(gaussian_cells) != 20:
        raise SAICPureT2VTopupGenerationError("v2 rendered cardinality differs")

    proof_fields = (
        "raw_value_sha256",
        "content_sha256",
        "shape",
        "dtype",
        "stored_dtype",
        "generator_initial_seed",
    )
    gaussian_proofs: list[dict[str, Any]] = []
    for (iid, seed), rows in gaussian_cells.items():
        if [branch for branch, _ in rows] != list(contract.BRANCH_ORDER):
            raise SAICPureT2VTopupGenerationError("v2 same-seed branch order differs")
        identities = {
            contract.object_sha256(
                {field: gaussian.get(field) for field in proof_fields}
            )
            for _, gaussian in rows
        }
        if len(identities) != 1:
            raise SAICPureT2VTopupGenerationError(
                "v2 same-cell branches did not use one official Gaussian"
            )
        gaussian_proofs.append(
            {
                "iid": iid,
                "seed": seed,
                "branch_order": list(contract.BRANCH_ORDER),
                "official_gaussian_tensor_values_byte_equal": True,
                "official_gaussian_identity_digest": next(iter(identities)),
            }
        )
    unsigned = {
        "schema_version": MASTER_SCHEMA_VERSION,
        "bank_id": contract.BANK_ID,
        "top_up_only": True,
        "root_spec_raw_sha256": root_sha,
        "base_v1_spec_raw_sha256": contract.BASE_V1_SPEC_RAW_SHA256,
        "base_v1_spec_content_sha256": contract.BASE_V1_SPEC_CONTENT_SHA256,
        "source_manifest_content_sha256": contract.SOURCE_MANIFEST_CONTENT_SHA256,
        "topology": "two_concurrent_world4_sp4_groups_on_one_8gpu_node",
        "sampling_contract": contract.SAMPLING_CONTRACT,
        "semantic_input_closure": contract.SEMANTIC_INPUT_CLOSURE,
        "geometry_proxy_contract": contract.GEOMETRY_PROXY_CONTRACT,
        "artifact_authority": contract.ARTIFACT_AUTHORITY,
        "attempt_count": 60,
        "row_count": 8,
        "seed_cell_count": 20,
        "branch_order": list(contract.BRANCH_ORDER),
        "merged_branch_order": list(contract.MERGED_BRANCH_ORDER),
        "six_branch_spec_merge_cell_count": len(merged),
        "same_seed_official_gaussian_proofs": gaussian_proofs,
        "attempts": attempt_rows,
        "detached_full81_event_review_complete": False,
        "event_verified": False,
        "identity_preservation_verified": False,
        "seed_selection_authorized": False,
        "training_target_authorized": False,
        "optimizer_or_parameter_update_authorized": False,
    }
    receipt = {**unsigned, "receipt_digest": contract.object_sha256(unsigned)}
    _closed(receipt, _MASTER_FIELDS, label="v2 master receipt")
    _write_create_only(root / MASTER_RECEIPT_BASENAME, receipt)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate-attempt")
    generate.add_argument("--candidate-envelope", required=True)
    generate.add_argument("--root-spec", required=True)
    generate.add_argument("--base-v1-spec", required=True)
    generate.add_argument("--source-manifest", required=True)
    generate.add_argument("--expected-root-spec-sha256", required=True)
    generate.add_argument("--bernini-root", required=True)
    generate.add_argument("--veomni-root", required=True)
    generate.add_argument("--checkpoint", required=True)
    generate.add_argument("--checkpoint-content-manifest", required=True)
    generate.add_argument("--output-dir", required=True)
    generate.add_argument("--method-source-revision", required=True)
    generate.add_argument("--method-source-archive-sha256", required=True)
    audit = commands.add_parser("audit-bank")
    audit.add_argument("--root-spec", required=True)
    audit.add_argument("--base-v1-spec", required=True)
    audit.add_argument("--source-manifest", required=True)
    audit.add_argument("--expected-root-spec-sha256", required=True)
    audit.add_argument("--output-root", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate-attempt":
        return generate_attempt(args)
    if args.command == "audit-bank":
        return audit_bank(args)
    raise SAICPureT2VTopupGenerationError("unknown v2 command")


if __name__ == "__main__":
    raise SystemExit(main())
