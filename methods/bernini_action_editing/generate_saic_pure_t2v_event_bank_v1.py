#!/usr/bin/env python3
"""Render and audit the sealed SAIC pure-T2V exact81 event bank.

Every GPU render delegates to the exact native Bernini entry point exercised
by owner job 131524.  The native ``--source-video`` slot receives only a
launch-local constant-black geometry proxy.  Candidate semantics contain text
from the sealed identity/scene projection and one forward/reverse/no-op event;
real source video paths and bytes never enter the native invocation.

Generation receipts stay pending detached full81 review.  They cannot qualify
an event, select a seed, authorize a training target, or authorize an optimizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(METHOD_ROOT))

import infer_native_identity_generation_canary as native  # noqa: E402
import infer_pair_v5_t2v_calibration_bank as native_receipts  # noqa: E402
import pair_v5_t2v_calibration_bank_spec as pair_contract  # noqa: E402
import saic_pure_t2v_event_bank_v1 as contract  # noqa: E402


SCHEMA_VERSION = "bernini-saic-pure-t2v-event-generation-receipt-v1"
MASTER_SCHEMA_VERSION = "bernini-saic-pure-t2v-event-bank-receipt-v1"
ATTEMPT_RECEIPT_BASENAME = "saic-event-generation-receipt.json"
MASTER_RECEIPT_BASENAME = "saic-pure-t2v-event-bank-receipt.json"
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SAICPureT2VGenerationError(RuntimeError):
    """Raised before unauthenticated generation evidence is accepted."""


def _plain_file(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise SAICPureT2VGenerationError(f"{label} must be an absolute plain file")
    return path.resolve(strict=True)


def _plain_dir(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_dir() or path.is_symlink():
        raise SAICPureT2VGenerationError(f"{label} must be an absolute plain directory")
    return path.resolve(strict=True)


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SAICPureT2VGenerationError(f"{label} must be lowercase SHA-256")
    return value


def _sha1(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA1.fullmatch(value) is None:
        raise SAICPureT2VGenerationError(f"{label} must be lowercase full SHA-1")
    return value


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    payload = contract.canonical_json_bytes(value) + b"\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as error:
        raise SAICPureT2VGenerationError(f"refusing to overwrite {path}") from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _reject_duplicate_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SAICPureT2VGenerationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json(path: str | Path, *, label: str) -> dict[str, Any]:
    source = _plain_file(path, label=label)
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                SAICPureT2VGenerationError(f"non-finite JSON token: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SAICPureT2VGenerationError(f"{label} is invalid JSON") from error
    if type(value) is not dict:
        raise SAICPureT2VGenerationError(f"{label} must be one object")
    return value


def _artifact_inside(value: Any, root: Path, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SAICPureT2VGenerationError(f"{label} must be an artifact object")
    path_value = value.get("path")
    expected = value.get("sha256")
    if not isinstance(path_value, str) or not isinstance(expected, str):
        raise SAICPureT2VGenerationError(f"{label} identity differs")
    path = _plain_file(path_value, label=label)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise SAICPureT2VGenerationError(f"{label} escaped attempt root") from error
    if file_sha256(path) != _sha(expected, label=f"{label} SHA-256"):
        raise SAICPureT2VGenerationError(f"{label} SHA-256 differs")
    return dict(value)


def native_candidate(envelope: Mapping[str, Any]) -> dict[str, Any]:
    candidate = envelope["candidate"]
    proxy = envelope["geometry_proxy"]
    return {
        "geometry_source_video_sha256": proxy["sha256"],
        "full_t2v_caption_utf8_sha256": candidate[
            "full_t2v_caption_utf8_sha256"
        ],
        "seed": candidate["seed"],
    }


def build_native_argv(args: argparse.Namespace, envelope: Mapping[str, Any]) -> list[str]:
    """Build the job-131524 native T2V call with a black proxy only."""

    candidate = envelope["candidate"]
    proxy = envelope["geometry_proxy"]
    proxy_path = _plain_file(proxy["path"], label="black geometry proxy")
    if (
        file_sha256(proxy_path) != proxy["sha256"]
        or proxy["sha256"] == candidate["source_media_sha256_for_nonuse_audit"]
    ):
        raise SAICPureT2VGenerationError("geometry proxy/source non-alias proof differs")
    argv = [
        "--bernini-root",
        str(_plain_dir(args.bernini_root, label="Bernini root")),
        "--veomni-root",
        str(_plain_dir(args.veomni_root, label="VeOmni root")),
        "--checkpoint",
        str(_plain_dir(args.checkpoint, label="Bernini checkpoint")),
        "--checkpoint-content-manifest",
        str(_plain_file(args.checkpoint_content_manifest, label="checkpoint manifest")),
        # The pinned runner calls this a source video.  It is a synthetic black
        # shape probe, not a source from the immutable SAIC manifest.
        "--source-video",
        str(proxy_path),
        "--expected-source-sha256",
        proxy["sha256"],
        "--action-prompt",
        candidate["full_t2v_caption"],
        "--expected-action-prompt-sha256",
        candidate["full_t2v_caption_utf8_sha256"],
        "--output-dir",
        args.output_dir,
        "--arms",
        "t2v",
        "--num-inference-steps",
        "40",
        "--seed",
        str(candidate["seed"]),
        "--method-source-revision",
        _sha1(args.method_source_revision, label="method source revision"),
        "--method-source-archive-sha256",
        _sha(args.method_source_archive_sha256, label="method archive SHA-256"),
    ]
    serialized = "\0".join(argv).encode("utf-8")
    # Only a hash of the real media appears in the envelope, never its path.
    # This last guard also catches accidental future path fields containing the
    # conventional source-video basename.
    if b"/source_video.mp4" in serialized:
        raise SAICPureT2VGenerationError("real source path leaked into native argv")
    return argv


def bind_attempt_receipt(
    *,
    args: argparse.Namespace,
    envelope: Mapping[str, Any],
    envelope_path: Path,
) -> Path:
    output = _plain_dir(args.output_dir, label="attempt output")
    native_path = _plain_file(output / "receipt.json", label="native receipt")
    try:
        native_value = native_receipts._load_json(native_path, "native receipt")
        verified = native_receipts._verify_native_receipt(
            native_value, native_candidate(envelope)
        )
    except pair_contract.PairT2VCalibrationSpecError as error:
        raise SAICPureT2VGenerationError(
            "job-131524 native exact81 T2V evidence failed verification"
        ) from error

    native_input = native_value.get("input")
    proxy = envelope["geometry_proxy"]
    candidate = envelope["candidate"]
    canonical_proxy_path = str(
        _plain_file(proxy["path"], label="receipt-bound black geometry proxy")
    )
    if (
        not isinstance(native_input, Mapping)
        or native_input.get("source_video_path") != canonical_proxy_path
        or native_input.get("source_video_sha256") != proxy["sha256"]
        or proxy["sha256"] == candidate["source_media_sha256_for_nonuse_audit"]
    ):
        raise SAICPureT2VGenerationError("native runner did not bind the black proxy")

    artifacts = {
        "mp4": _artifact_inside(verified["mp4"], output, label="proposal MP4"),
        "predecode_clean_latent": _artifact_inside(
            verified["predecode_clean_latent"], output, label="proposal clean latent"
        ),
        "official_initial_gaussian": _artifact_inside(
            verified["official_initial_gaussian"], output, label="official Gaussian"
        ),
    }
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "root_spec_raw_sha256": envelope["root_spec_raw_sha256"],
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
        "real_source_nonuse_certificate": {
            "source_media_sha256_for_nonuse_audit": candidate[
                "source_media_sha256_for_nonuse_audit"
            ],
            "real_source_path_present_in_candidate_envelope": False,
            "real_source_path_passed_to_native_runner": False,
            "real_source_rgb_read": False,
            "real_source_latent_read_or_created": False,
            "real_source_noise_read_or_created": False,
            "geometry_proxy_path": canonical_proxy_path,
            "geometry_proxy_sha256": proxy["sha256"],
            "proxy_bytes_differ_from_real_source": True,
            "proxy_used_for_bucket_shape_only": True,
            "proxy_vae_latent_created": False,
            "proxy_pixels_entered_transformer": False,
            "native_content_conditioning_count": 0,
        },
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
    envelope_path = _plain_file(args.candidate_envelope, label="candidate envelope")
    try:
        envelope = contract.load_candidate_envelope(
            envelope_path,
            expected_root_spec_sha256=args.expected_root_spec_sha256,
        )
    except contract.SAICPureT2VEventBankError as error:
        raise SAICPureT2VGenerationError("candidate envelope failed closed") from error
    expected_visible = ",".join(str(item) for item in envelope["visible_gpus"])
    if (
        os.environ.get("WORLD_SIZE") != "4"
        or os.environ.get("ROCR_VISIBLE_DEVICES") != expected_visible
    ):
        raise SAICPureT2VGenerationError("attempt must run in its sealed WORLD4/SP4 group")
    output = Path(args.output_dir)
    if not output.is_absolute() or output == Path("/") or output.exists() or output.is_symlink():
        raise SAICPureT2VGenerationError("attempt output must be fresh and absolute")
    status = native.main(build_native_argv(args, envelope))
    if status == 0 and int(os.environ.get("RANK", "0")) == 0:
        bind_attempt_receipt(
            args=args, envelope=envelope, envelope_path=envelope_path
        )
    return status


def _load_attempt_receipt(
    path: Path,
    *,
    candidate: Mapping[str, Any],
    group: Mapping[str, Any],
    root_spec_sha256: str,
    real_source_paths: set[str],
    real_source_hashes: set[str],
) -> dict[str, Any]:
    receipt = _load_json(path, label="SAIC attempt receipt")
    unsigned = dict(receipt)
    declared = unsigned.pop("receipt_digest", None)
    expected_visible = ",".join(str(item) for item in group["visible_gpus"])
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or declared != contract.object_sha256(unsigned)
        or receipt.get("root_spec_raw_sha256") != root_spec_sha256
        or receipt.get("candidate") != candidate
        or receipt.get("group_id") != group["group_id"]
        or receipt.get("actor_family") != group["actor_family"]
        or receipt.get("visible_gpus") != group["visible_gpus"]
        or receipt.get("runtime_topology")
        != {
            "world_size": 4,
            "ulysses_size": 4,
            "rocr_visible_devices": expected_visible,
        }
        or receipt.get("event_audit_status")
        != "pending_detached_full81_review"
        or receipt.get("event_verified") is not False
        or receipt.get("identity_preservation_verified") is not False
        or receipt.get("seed_selection_authorized") is not False
        or receipt.get("training_target_authorized") is not False
        or receipt.get("optimizer_or_parameter_update_authorized") is not False
    ):
        raise SAICPureT2VGenerationError("attempt receipt/spec binding differs")
    nonuse = receipt.get("real_source_nonuse_certificate")
    if (
        not isinstance(nonuse, Mapping)
        or nonuse.get("real_source_path_present_in_candidate_envelope") is not False
        or nonuse.get("real_source_path_passed_to_native_runner") is not False
        or nonuse.get("real_source_rgb_read") is not False
        or nonuse.get("real_source_latent_read_or_created") is not False
        or nonuse.get("real_source_noise_read_or_created") is not False
        or nonuse.get("proxy_bytes_differ_from_real_source") is not True
        or nonuse.get("proxy_used_for_bucket_shape_only") is not True
        or nonuse.get("proxy_vae_latent_created") is not False
        or nonuse.get("proxy_pixels_entered_transformer") is not False
        or nonuse.get("native_content_conditioning_count") != 0
        or nonuse.get("geometry_proxy_path") in real_source_paths
        or nonuse.get("geometry_proxy_sha256") in real_source_hashes
    ):
        raise SAICPureT2VGenerationError("attempt source-nonuse certificate differs")
    envelope_path = _plain_file(
        receipt.get("candidate_envelope_path", ""), label="receipt-bound envelope"
    )
    if file_sha256(envelope_path) != receipt.get("candidate_envelope_sha256"):
        raise SAICPureT2VGenerationError("candidate envelope SHA-256 differs")
    try:
        envelope = contract.load_candidate_envelope(
            envelope_path, expected_root_spec_sha256=root_spec_sha256
        )
    except contract.SAICPureT2VEventBankError as error:
        raise SAICPureT2VGenerationError("candidate envelope re-audit failed") from error
    if (
        envelope["candidate"] != candidate
        or envelope["group_id"] != group["group_id"]
        or envelope["visible_gpus"] != group["visible_gpus"]
        or envelope["geometry_proxy"]["path"] in real_source_paths
        or envelope["geometry_proxy"]["sha256"] in real_source_hashes
    ):
        raise SAICPureT2VGenerationError("candidate envelope source closure differs")
    native_path = _plain_file(receipt["native_receipt_path"], label="native receipt")
    if file_sha256(native_path) != receipt["native_receipt_sha256"]:
        raise SAICPureT2VGenerationError("native receipt SHA-256 differs")
    native_value = native_receipts._load_json(native_path, "native receipt")
    try:
        verified = native_receipts._verify_native_receipt(
            native_value,
            {
                "geometry_source_video_sha256": nonuse["geometry_proxy_sha256"],
                "full_t2v_caption_utf8_sha256": candidate[
                    "full_t2v_caption_utf8_sha256"
                ],
                "seed": candidate["seed"],
            },
        )
    except pair_contract.PairT2VCalibrationSpecError as error:
        raise SAICPureT2VGenerationError("native receipt re-audit failed") from error
    if (
        verified["native_receipt_digest"] != receipt["native_receipt_digest"]
        or native_value.get("input", {}).get("source_video_path")
        != nonuse["geometry_proxy_path"]
    ):
        raise SAICPureT2VGenerationError("native receipt proxy binding differs")
    root = path.parent
    expected_artifacts = {
        "mp4": verified["mp4"],
        "predecode_clean_latent": verified["predecode_clean_latent"],
        "official_initial_gaussian": verified["official_initial_gaussian"],
    }
    if receipt.get("artifacts") != expected_artifacts:
        raise SAICPureT2VGenerationError("attempt artifacts differ from native receipt")
    for label, artifact in receipt.get("artifacts", {}).items():
        _artifact_inside(artifact, root, label=f"{candidate['candidate_id']} {label}")
    return receipt


def audit_bank(args: argparse.Namespace) -> int:
    spec, root_sha = contract.load_sealed_spec(
        args.root_spec,
        expected_raw_sha256=args.expected_root_spec_sha256,
        source_manifest_path=args.source_manifest,
    )
    source_manifest = contract.source_set.load_manifest(args.source_manifest)
    contract.source_set.validate_manifest(source_manifest)
    real_source_paths = {row["source_video"] for row in source_manifest["rows"]}
    real_source_hashes = {
        row["source_video_sha256"] for row in source_manifest["rows"]
    }
    root = _plain_dir(args.output_root, label="event-bank output root")
    attempts_root = _plain_dir(root / "attempts", label="attempts root")

    attempt_rows = []
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
        raise SAICPureT2VGenerationError("rendered event-bank cardinality differs")

    gaussian_proofs = []
    proof_fields = (
        "raw_value_sha256",
        "content_sha256",
        "shape",
        "dtype",
        "stored_dtype",
        "generator_initial_seed",
    )
    for (iid, seed), rows in gaussian_cells.items():
        if [branch for branch, _ in rows] != list(contract.BRANCH_ORDER):
            raise SAICPureT2VGenerationError("same-seed branch order differs")
        identities = {
            contract.object_sha256(
                {field: gaussian.get(field) for field in proof_fields}
            )
            for _, gaussian in rows
        }
        if len(identities) != 1:
            raise SAICPureT2VGenerationError(
                "same row/seed branches did not use one official Gaussian value"
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
        "bank_id": spec["bank_id"],
        "root_spec_raw_sha256": root_sha,
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
    _write_create_only(root / MASTER_RECEIPT_BASENAME, receipt)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate-attempt")
    generate.add_argument("--candidate-envelope", required=True)
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
    raise SAICPureT2VGenerationError("unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
