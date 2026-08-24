#!/usr/bin/env python3
"""Post-save artifact finalizer for a formal Bernini RMC LoRA v7 checkpoint.

Training deliberately writes a pending receipt because source-level contract
tests cannot certify files that do not yet exist.  This program accepts only a
pending, target-only, exact-40 checkpoint.  It hashes the serialized adapter,
constructs a fresh pinned Bernini-R 1.3B base, and asks the production v7
loader to reload and compare the exact 46-module/92-tensor LoRA artifact.

Only after that strict reload succeeds are ``receipt.json`` and ``latest.json``
replaced.  No source or target video is accepted and no sampling path is
imported or executed.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Iterator, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_lora as infer_base  # noqa: E402
import infer_relational_motion_commutator as rmc  # noqa: E402
import train_relational_motion_commutator_auh as v7_train  # noqa: E402


legacy = rmc.trainer

ARTIFACT_VALIDATION_SCHEMA = v7_train.ARTIFACT_VALIDATION_SCHEMA
PENDING_ARTIFACT_STATUS = "pending_post_save_strict_reload"
READY_ARTIFACT_STATUS = "post_save_strict_reload_complete"
FORMAL_GLOBAL_STEP = rmc.NUM_DENOISING_STEPS
FORMAL_CHECKPOINT_NAME = f"checkpoint-{FORMAL_GLOBAL_STEP:08d}"

_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class RelationalMotionCommutatorFinalizationError(RuntimeError):
    """Raised without publishing a ready receipt when finalization fails."""


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RelationalMotionCommutatorFinalizationError(
            f"cannot read {label} {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise RelationalMotionCommutatorFinalizationError(
            f"{label} must contain one JSON object"
        )
    return value


def _require_sha(value: Any, *, label: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise RelationalMotionCommutatorFinalizationError(f"{label} is invalid")
    return value


def _require_plain_file(path: Path, *, label: str) -> Path:
    try:
        if not path.is_file() or path.is_symlink():
            raise RelationalMotionCommutatorFinalizationError(
                f"{label} must be one plain file: {path}"
            )
    except OSError as error:
        raise RelationalMotionCommutatorFinalizationError(
            f"cannot inspect {label}: {path}: {error}"
        ) from error
    return path


def _resolve_formal_checkpoint_root(value: str | Path) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute():
        raise RelationalMotionCommutatorFinalizationError(
            "formal checkpoint root must be absolute"
        )
    if requested.is_symlink():
        raise RelationalMotionCommutatorFinalizationError(
            "formal checkpoint root may not be a symlink"
        )
    try:
        root = requested.resolve(strict=True)
    except OSError as error:
        raise RelationalMotionCommutatorFinalizationError(
            f"cannot resolve formal checkpoint root: {error}"
        ) from error
    if not root.is_dir() or root.name != FORMAL_CHECKPOINT_NAME:
        raise RelationalMotionCommutatorFinalizationError(
            f"formal checkpoint root must be named {FORMAL_CHECKPOINT_NAME}"
        )
    _require_plain_file(root / "optimizer.pt", label="optimizer checkpoint")
    return root


def _validate_receipt_digest(receipt: Mapping[str, Any]) -> str:
    candidate = dict(receipt)
    digest = candidate.pop("receipt_digest", None)
    _require_sha(digest, label="pending receipt digest", pattern=_SHA256_RE)
    if legacy.object_sha256(candidate) != digest:
        raise RelationalMotionCommutatorFinalizationError(
            "pending receipt digest differs"
        )
    return digest


def validate_pending_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the release gate before any Bernini model is constructed."""

    if not isinstance(receipt, Mapping):
        raise RelationalMotionCommutatorFinalizationError(
            "pending receipt must be a mapping"
        )
    digest = _validate_receipt_digest(receipt)
    if receipt.get("schema_version") != rmc.TRAINING_RECEIPT_SCHEMA:
        raise RelationalMotionCommutatorFinalizationError(
            "checkpoint is not a v7 training receipt"
        )
    if receipt.get("method") != rmc.METHOD_NAME:
        raise RelationalMotionCommutatorFinalizationError(
            "checkpoint is not the v7 relational commutator method"
        )
    if (
        receipt.get("global_step") != FORMAL_GLOBAL_STEP
        or receipt.get("max_steps") != FORMAL_GLOBAL_STEP
        or receipt.get("formal_40_sigma_cycle_complete") is not True
        or receipt.get("accepted_sigma_schedule_indices")
        != list(range(FORMAL_GLOBAL_STEP))
    ):
        raise RelationalMotionCommutatorFinalizationError(
            "only one formal exact-40 sigma cycle can be finalized"
        )
    step_audit = receipt.get("step_audit")
    if (
        not isinstance(step_audit, list)
        or len(step_audit) != FORMAL_GLOBAL_STEP
        or receipt.get("step_audit_sha256") != legacy.object_sha256(step_audit)
    ):
        raise RelationalMotionCommutatorFinalizationError(
            "formal step audit is incomplete or altered"
        )
    for index, record in enumerate(step_audit):
        if (
            not isinstance(record, Mapping)
            or record.get("optimizer_step") != index + 1
            or record.get("sigma_schedule_index") != index
            or record.get("teacher_mode") != "target_only"
        ):
            raise RelationalMotionCommutatorFinalizationError(
                f"formal target-only step audit differs at index {index}"
            )

    immutable = receipt.get("immutable_contract")
    if not isinstance(immutable, Mapping) or not isinstance(
        immutable.get("value"), Mapping
    ):
        raise RelationalMotionCommutatorFinalizationError(
            "pending receipt lacks its immutable contract"
        )
    immutable_value = immutable["value"]
    if immutable.get("digest") != legacy.object_sha256(immutable_value):
        raise RelationalMotionCommutatorFinalizationError(
            "immutable training contract digest differs"
        )
    parity = immutable_value.get("inference_loader_parity")
    if (
        immutable_value.get("teacher_mode") != "target_only"
        or not isinstance(parity, Mapping)
        or parity.get("verified") is not True
        or receipt.get("inference_loader_parity") != parity
    ):
        raise RelationalMotionCommutatorFinalizationError(
            "pending receipt is not the source-preflight target-only arm"
        )
    if receipt.get("inference_loader_parity_pending") is not True:
        raise RelationalMotionCommutatorFinalizationError(
            "only a pending post-save receipt may be finalized"
        )
    artifact = receipt.get("artifact_validation")
    expected_pending_artifact = {
        "schema_version": ARTIFACT_VALIDATION_SCHEMA,
        "verified": False,
        "status": PENDING_ARTIFACT_STATUS,
    }
    if artifact != expected_pending_artifact:
        raise RelationalMotionCommutatorFinalizationError(
            "artifact validation is not in the pristine pending state"
        )
    if receipt.get("production_claim_forbidden") is not True:
        raise RelationalMotionCommutatorFinalizationError(
            "pending receipt lost its experimental-only restriction"
        )
    if receipt.get("scientific_claim_authorized") is not False:
        raise RelationalMotionCommutatorFinalizationError(
            "pending receipt carries an unsupported scientific claim"
        )
    return {
        "receipt_digest": digest,
        "immutable_value": immutable_value,
        "parity": parity,
    }


def _validate_invocation_identity(
    receipt: Mapping[str, Any],
    *,
    method_source_revision: str,
    method_source_archive_sha256: str,
    bernini_revision: str,
    veomni_revision: str,
    base_checkpoint: Path,
    expected_checkpoint_tree_sha256: str,
) -> None:
    source_revision = _require_sha(
        method_source_revision,
        label="method source revision",
        pattern=_SHA1_RE,
    )
    source_archive = _require_sha(
        method_source_archive_sha256,
        label="method source archive SHA-256",
        pattern=_SHA256_RE,
    )
    checkpoint_tree = _require_sha(
        expected_checkpoint_tree_sha256,
        label="base checkpoint tree SHA-256",
        pattern=_SHA256_RE,
    )
    immutable = receipt["immutable_contract"]["value"]
    checkpoint = receipt.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise RelationalMotionCommutatorFinalizationError(
            "pending receipt lacks base checkpoint identity"
        )
    if (
        immutable.get("method_source_revision") != source_revision
        or immutable.get("method_source_archive_sha256") != source_archive
        or receipt.get("bernini_commit") != bernini_revision
        or receipt.get("veomni_commit") != veomni_revision
        or immutable.get("bernini_commit") != bernini_revision
        or immutable.get("veomni_commit") != veomni_revision
        or immutable.get("checkpoint_path") != str(base_checkpoint)
        or checkpoint.get("path") != str(base_checkpoint)
        or immutable.get("checkpoint_tree_sha256") != checkpoint_tree
        or checkpoint.get("tree_sha256") != checkpoint_tree
    ):
        raise RelationalMotionCommutatorFinalizationError(
            "finalizer source or base-checkpoint identity differs from training"
        )


def build_ready_artifact_validation(
    *,
    receipt: Mapping[str, Any],
    adapter_config_sha256: str,
    adapter_model_sha256: str,
    method_source_revision: str,
    method_source_archive_sha256: str,
    bernini_revision: str,
    veomni_revision: str,
    expected_checkpoint_tree_sha256: str,
) -> dict[str, Any]:
    """Build the candidate proof consumed by the strict v7 loader."""

    adapter = receipt.get("adapter")
    if not isinstance(adapter, Mapping):
        raise RelationalMotionCommutatorFinalizationError(
            "pending receipt lacks adapter identity"
        )
    parameter_digest = _require_sha(
        adapter.get("checkpoint_parameter_digest"),
        label="training checkpoint parameter digest",
        pattern=_SHA256_RE,
    )
    pending_receipt_digest = _validate_receipt_digest(receipt)
    value: dict[str, Any] = {
        "schema_version": ARTIFACT_VALIDATION_SCHEMA,
        "verified": True,
        "status": READY_ARTIFACT_STATUS,
        "adapter_config_sha256": _require_sha(
            adapter_config_sha256,
            label="adapter config SHA-256",
            pattern=_SHA256_RE,
        ),
        "adapter_model_sha256": _require_sha(
            adapter_model_sha256,
            label="adapter model SHA-256",
            pattern=_SHA256_RE,
        ),
        "serialized_target_pattern_count": 17,
        "expanded_target_module_count": 46,
        "adapter_tensor_count": 92,
        "active_lora_module_count": 46,
        "strict_tensor_reload_equal": True,
        "parameter_digest_verified_after_safetensors_reload": True,
        "checkpoint_parameter_digest": parameter_digest,
        "pending_receipt_digest": pending_receipt_digest,
        "validator_method_source_revision": _require_sha(
            method_source_revision,
            label="validator method source revision",
            pattern=_SHA1_RE,
        ),
        "validator_method_source_archive_sha256": _require_sha(
            method_source_archive_sha256,
            label="validator method source archive SHA-256",
            pattern=_SHA256_RE,
        ),
        "bernini_commit": _require_sha(
            bernini_revision,
            label="validated Bernini commit",
            pattern=_SHA1_RE,
        ),
        "veomni_commit": _require_sha(
            veomni_revision,
            label="validated VeOmni commit",
            pattern=_SHA1_RE,
        ),
        "checkpoint_tree_sha256": _require_sha(
            expected_checkpoint_tree_sha256,
            label="validated checkpoint tree SHA-256",
            pattern=_SHA256_RE,
        ),
    }
    value["digest"] = legacy.object_sha256(value)
    return value


def build_ready_receipt_candidate(
    pending_receipt: Mapping[str, Any],
    *,
    artifact_validation: Mapping[str, Any],
) -> dict[str, Any]:
    """Return an unpublished ready receipt for use by the real strict loader."""

    validate_pending_receipt(pending_receipt)
    candidate = copy.deepcopy(dict(pending_receipt))
    candidate.pop("receipt_digest", None)
    candidate["inference_loader_parity_pending"] = False
    candidate["artifact_validation"] = copy.deepcopy(dict(artifact_validation))
    candidate["receipt_digest"] = legacy.object_sha256(candidate)
    return candidate


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = legacy.canonical_json_bytes(value) + b"\n"
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.finalize-",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


@contextmanager
def _exclusive_finalizer_lock(checkpoint_root: Path) -> Iterator[None]:
    lock = checkpoint_root / ".finalize-relational-motion-commutator.lock"
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RelationalMotionCommutatorFinalizationError(
            f"checkpoint finalization is already locked: {lock}"
        ) from error
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        yield
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def _build_fresh_bernini_base(
    *,
    bernini_root: Path,
    base_checkpoint: Path,
) -> Any:
    """Construct one untouched local Bernini-R 1.3B base on CPU."""

    try:
        import torch
        from bernini.models.renderer import (
            BerniniRendererConfig,
            BerniniRendererModel,
        )
    except Exception as error:
        raise RelationalMotionCommutatorFinalizationError(
            "cannot import the pinned Bernini runtime"
        ) from error
    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **infer_base.inference_renderer_config_overrides(base_checkpoint),
    )
    config.dtype = torch.bfloat16
    try:
        legacy.validate_renderer_config_mapping(config.to_dict(), base_checkpoint)
        base_model = BerniniRendererModel(config)
    except Exception as error:
        raise RelationalMotionCommutatorFinalizationError(
            "cannot construct the fresh pinned Bernini-R 1.3B base"
        ) from error
    if any("lora_" in name.lower() for name, _ in base_model.named_modules()):
        raise RelationalMotionCommutatorFinalizationError(
            "fresh Bernini base unexpectedly contains LoRA modules"
        )
    base_model.requires_grad_(False)
    base_model.eval()
    return base_model


def _publish_ready_receipt(
    *,
    receipt_path: Path,
    ready_receipt: Mapping[str, Any],
    latest_path: Path,
    pending_latest: Mapping[str, Any],
) -> None:
    artifact = ready_receipt["artifact_validation"]
    ready_latest = copy.deepcopy(dict(pending_latest))
    ready_latest.update(
        {
            "receipt_digest": ready_receipt["receipt_digest"],
            "inference_loader_parity_pending": False,
            "artifact_validation_digest": artifact["digest"],
        }
    )
    # Publish the pointer first and the authoritative receipt last.  A process
    # death between the two replacements leaves the receipt pending, hence the
    # checkpoint remains fail-closed even though latest.json carries the future
    # digest.  The inverse order could expose a ready receipt before its pointer
    # was synchronized.
    _atomic_write_json(latest_path, ready_latest)
    try:
        _atomic_write_json(receipt_path, ready_receipt)
    except Exception as error:
        try:
            _atomic_write_json(latest_path, pending_latest)
        except Exception as rollback_error:
            raise RelationalMotionCommutatorFinalizationError(
                "ready receipt publication failed and latest.json rollback failed"
            ) from rollback_error
        raise RelationalMotionCommutatorFinalizationError(
            "ready receipt publication failed; pending latest.json was restored"
        ) from error


def finalize_checkpoint(
    checkpoint_root: str | Path,
    *,
    bernini_root: str | Path,
    veomni_root: str | Path,
    base_checkpoint: str | Path,
    method_source_revision: str,
    method_source_archive_sha256: str,
    expected_bernini_commit: str = legacy.BERNINI_OFFICIAL_COMMIT,
    expected_veomni_commit: str = legacy.VEOMNI_TESTED_COMMIT,
    expected_checkpoint_tree_sha256: str = legacy.CHECKPOINT_TREE_SHA256,
) -> dict[str, Any]:
    """Strict-reload one pending artifact and atomically publish readiness."""

    root = _resolve_formal_checkpoint_root(checkpoint_root)
    try:
        bundle = infer_base.resolve_adapter_bundle(root)
    except infer_base.InferenceContractError as error:
        raise RelationalMotionCommutatorFinalizationError(str(error)) from error
    if bundle.checkpoint_root != root:
        raise RelationalMotionCommutatorFinalizationError(
            "adapter bundle did not resolve to the formal checkpoint root"
        )
    receipt_path = _require_plain_file(
        bundle.training_receipt_path, label="pending training receipt"
    )
    latest_path = _require_plain_file(root.parent / "latest.json", label="latest receipt")

    with _exclusive_finalizer_lock(root):
        receipt_bytes_before = receipt_path.read_bytes()
        latest_bytes_before = latest_path.read_bytes()
        pending_receipt = _read_json(receipt_path, label="pending training receipt")
        pending_state = validate_pending_receipt(pending_receipt)
        pending_latest = _read_json(latest_path, label="pending latest receipt")
        latest_checkpoint = pending_latest.get("checkpoint")
        try:
            latest_checkpoint_path = Path(str(latest_checkpoint)).expanduser().resolve(
                strict=True
            )
        except OSError as error:
            raise RelationalMotionCommutatorFinalizationError(
                "latest.json checkpoint cannot be resolved"
            ) from error
        if (
            latest_checkpoint_path != root
            or pending_latest.get("global_step") != FORMAL_GLOBAL_STEP
        ):
            raise RelationalMotionCommutatorFinalizationError(
                "latest.json does not identify this exact-40 checkpoint"
            )

        try:
            validated_bernini_root, validated_veomni_root, bernini_revision, veomni_revision = (
                legacy.validate_source_trees(
                    bernini_root,
                    veomni_root,
                    expected_bernini_commit=expected_bernini_commit,
                    expected_veomni_commit=expected_veomni_commit,
                )
            )
            validated_base_checkpoint, transformer_config = legacy.validate_checkpoint(
                base_checkpoint
            )
        except legacy.TrainingContractError as error:
            raise RelationalMotionCommutatorFinalizationError(str(error)) from error
        if transformer_config.get("num_attention_heads") != 12:
            raise RelationalMotionCommutatorFinalizationError(
                "fresh base is not the pinned Bernini-R 1.3B transformer"
            )
        _validate_invocation_identity(
            pending_receipt,
            method_source_revision=method_source_revision,
            method_source_archive_sha256=method_source_archive_sha256,
            bernini_revision=bernini_revision,
            veomni_revision=veomni_revision,
            base_checkpoint=validated_base_checkpoint,
            expected_checkpoint_tree_sha256=expected_checkpoint_tree_sha256,
        )
        legacy.activate_source_trees(validated_bernini_root, validated_veomni_root)

        adapter_config = _read_json(bundle.adapter_config_path, label="adapter config")
        config_sha256 = legacy.file_sha256(bundle.adapter_config_path)
        model_sha256 = legacy.file_sha256(bundle.adapter_model_path)
        artifact = build_ready_artifact_validation(
            receipt=pending_receipt,
            adapter_config_sha256=config_sha256,
            adapter_model_sha256=model_sha256,
            method_source_revision=method_source_revision,
            method_source_archive_sha256=method_source_archive_sha256,
            bernini_revision=bernini_revision,
            veomni_revision=veomni_revision,
            expected_checkpoint_tree_sha256=expected_checkpoint_tree_sha256,
        )
        ready_candidate = build_ready_receipt_candidate(
            pending_receipt,
            artifact_validation=artifact,
        )
        try:
            expected_identity = rmc.validate_training_adapter_contract(
                adapter_config,
                ready_candidate,
                expected_checkpoint_tree_sha256=expected_checkpoint_tree_sha256,
            )
        except rmc.RelationalMotionCommutatorInferenceError as error:
            raise RelationalMotionCommutatorFinalizationError(str(error)) from error

        # Normal entry has latest.json bound to the pending receipt.  A process
        # death after the deliberately latest-first publication leaves the
        # deterministic ready digest there while receipt.json remains pending;
        # accept exactly that state so a fresh strict reload can finish it.
        latest_digest = pending_latest.get("receipt_digest")
        latest_is_pending = (
            latest_digest == pending_state["receipt_digest"]
            and pending_latest.get("inference_loader_parity_pending")
            in (None, True)
            and pending_latest.get("artifact_validation_digest") is None
        )
        latest_is_interrupted_ready = (
            latest_digest == ready_candidate["receipt_digest"]
            and pending_latest.get("inference_loader_parity_pending") is False
            and pending_latest.get("artifact_validation_digest")
            == artifact["digest"]
        )
        if not (latest_is_pending or latest_is_interrupted_ready):
            raise RelationalMotionCommutatorFinalizationError(
                "latest.json digest or artifact release state differs"
            )

        base_model = _build_fresh_bernini_base(
            bernini_root=validated_bernini_root,
            base_checkpoint=validated_base_checkpoint,
        )
        try:
            loaded_model, tensor_count, active_count, loaded_identity = (
                rmc.strict_load_adapter(
                    base_model=base_model,
                    bundle=bundle,
                    adapter_config=adapter_config,
                    receipt=ready_candidate,
                    expected_checkpoint_tree_sha256=(
                        expected_checkpoint_tree_sha256
                    ),
                )
            )
        except rmc.RelationalMotionCommutatorInferenceError as error:
            raise RelationalMotionCommutatorFinalizationError(str(error)) from error
        if (
            loaded_model is None
            or tensor_count != 92
            or active_count != 46
            or loaded_identity != expected_identity
            or loaded_identity.get("checkpoint_parameter_digest")
            != artifact["checkpoint_parameter_digest"]
            or loaded_identity.get("adapter_config_sha256") != config_sha256
            or loaded_identity.get("adapter_model_sha256") != model_sha256
            or loaded_identity.get("artifact_validation_digest")
            != artifact["digest"]
        ):
            raise RelationalMotionCommutatorFinalizationError(
                "strict-reloaded adapter identity or exact46/92 counts differ"
            )

        if (
            legacy.file_sha256(bundle.adapter_config_path) != config_sha256
            or legacy.file_sha256(bundle.adapter_model_path) != model_sha256
            or receipt_path.read_bytes() != receipt_bytes_before
            or latest_path.read_bytes() != latest_bytes_before
        ):
            raise RelationalMotionCommutatorFinalizationError(
                "checkpoint artifacts changed during strict finalization"
            )
        _publish_ready_receipt(
            receipt_path=receipt_path,
            ready_receipt=ready_candidate,
            latest_path=latest_path,
            pending_latest=pending_latest,
        )
        return ready_candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Strict-reload and finalize one pending exact-40 Bernini RMC v7 "
            "checkpoint; no video is generated"
        )
    )
    parser.add_argument("--formal-checkpoint-root", required=True)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    parser.add_argument(
        "--expected-bernini-commit", default=legacy.BERNINI_OFFICIAL_COMMIT
    )
    parser.add_argument(
        "--expected-veomni-commit", default=legacy.VEOMNI_TESTED_COMMIT
    )
    parser.add_argument(
        "--expected-checkpoint-tree-sha256",
        default=legacy.CHECKPOINT_TREE_SHA256,
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    ready = finalize_checkpoint(
        args.formal_checkpoint_root,
        bernini_root=args.bernini_root,
        veomni_root=args.veomni_root,
        base_checkpoint=args.base_checkpoint,
        method_source_revision=args.method_source_revision,
        method_source_archive_sha256=args.method_source_archive_sha256,
        expected_bernini_commit=args.expected_bernini_commit,
        expected_veomni_commit=args.expected_veomni_commit,
        expected_checkpoint_tree_sha256=args.expected_checkpoint_tree_sha256,
    )
    print(legacy.canonical_json_bytes(ready).decode("utf-8"), flush=True)
    return 0


__all__ = [
    "ARTIFACT_VALIDATION_SCHEMA",
    "FORMAL_CHECKPOINT_NAME",
    "PENDING_ARTIFACT_STATUS",
    "READY_ARTIFACT_STATUS",
    "RelationalMotionCommutatorFinalizationError",
    "build_parser",
    "build_ready_artifact_validation",
    "build_ready_receipt_candidate",
    "finalize_checkpoint",
    "main",
    "validate_pending_receipt",
]


if __name__ == "__main__":
    raise SystemExit(main())
