#!/usr/bin/env python3
"""Post-save strict finalizer for a formal Bernini RS-FQT LoRA v8 pilot.

The trainer can prove its source/data/objective contract, but it cannot certify
PEFT files before they exist.  It therefore publishes an exact-40 checkpoint
with a pending receipt.  This finalizer hashes those files, constructs a fresh
pinned Bernini-R 1.3B base, invokes the production v8 exact46/92 loader, hashes
the files again, and only then publishes the ready receipt and ``latest.json``.

The ready receipt is reconstructed from the pending receipt by changing only
``inference_loader_parity_pending`` and ``artifact_validation``.  No video,
prompt, target, mask, tracking signal, or sampler is accepted here.
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

import infer_feasible_quotient_lora as v8_loader  # noqa: E402
import infer_lora as infer_base  # noqa: E402
import train_feasible_quotient_auh as v8_train  # noqa: E402


legacy = v8_train.legacy

ARTIFACT_VALIDATION_SCHEMA = v8_loader.ARTIFACT_VALIDATION_SCHEMA
PENDING_ARTIFACT_STATUS = v8_loader.PENDING_ARTIFACT_STATUS
READY_ARTIFACT_STATUS = v8_loader.READY_ARTIFACT_STATUS
FORMAL_GLOBAL_STEP = v8_loader.FORMAL_GLOBAL_STEP
FORMAL_CHECKPOINT_NAME = f"checkpoint-{FORMAL_GLOBAL_STEP:08d}"

_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class FeasibleQuotientFinalizationError(RuntimeError):
    """Raised while the durable training receipt remains fail-closed."""


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FeasibleQuotientFinalizationError(
            f"cannot read {label} {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise FeasibleQuotientFinalizationError(
            f"{label} must contain one JSON object"
        )
    return value


def _require_sha(
    value: Any, *, label: str, pattern: re.Pattern[str]
) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise FeasibleQuotientFinalizationError(f"{label} is invalid")
    return value


def _require_plain_file(path: Path, *, label: str) -> Path:
    try:
        if not path.is_file() or path.is_symlink():
            raise FeasibleQuotientFinalizationError(
                f"{label} must be one plain file: {path}"
            )
    except OSError as error:
        raise FeasibleQuotientFinalizationError(
            f"cannot inspect {label}: {path}: {error}"
        ) from error
    return path


def _resolve_formal_checkpoint_root(value: str | Path) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute():
        raise FeasibleQuotientFinalizationError(
            "formal v8 checkpoint root must be absolute"
        )
    if requested.is_symlink():
        raise FeasibleQuotientFinalizationError(
            "formal v8 checkpoint root may not be a symlink"
        )
    try:
        root = requested.resolve(strict=True)
    except OSError as error:
        raise FeasibleQuotientFinalizationError(
            f"cannot resolve formal v8 checkpoint root: {error}"
        ) from error
    if not root.is_dir() or root.name != FORMAL_CHECKPOINT_NAME:
        raise FeasibleQuotientFinalizationError(
            f"formal v8 checkpoint root must be named {FORMAL_CHECKPOINT_NAME}"
        )
    _require_plain_file(root / "optimizer.pt", label="optimizer checkpoint")
    return root


def _validate_receipt_digest(receipt: Mapping[str, Any]) -> str:
    candidate = dict(receipt)
    digest = candidate.pop("receipt_digest", None)
    _require_sha(digest, label="pending receipt digest", pattern=_SHA256_RE)
    if legacy.object_sha256(candidate) != digest:
        raise FeasibleQuotientFinalizationError(
            "pending v8 receipt digest differs"
        )
    return digest


def validate_pending_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Reject non-v8, nonformal, nonpristine, and already-ready receipts."""

    if not isinstance(receipt, Mapping):
        raise FeasibleQuotientFinalizationError(
            "pending v8 receipt must be a mapping"
        )
    digest = _validate_receipt_digest(receipt)
    if receipt.get("schema_version") != v8_loader.TRAINING_RECEIPT_SCHEMA:
        raise FeasibleQuotientFinalizationError(
            "checkpoint is not an RS-FQT v8 training receipt"
        )
    if receipt.get("method") != v8_loader.METHOD_NAME:
        raise FeasibleQuotientFinalizationError(
            "checkpoint is not the RS-FQT v8 training method"
        )
    if (
        receipt.get("global_step") != FORMAL_GLOBAL_STEP
        or receipt.get("max_steps") != FORMAL_GLOBAL_STEP
        or receipt.get("formal_40_sigma_cycle_complete") is not True
        or receipt.get("accepted_sigma_schedule_indices")
        != list(range(FORMAL_GLOBAL_STEP))
    ):
        raise FeasibleQuotientFinalizationError(
            "only one formal exact-40 RS-FQT cycle can be finalized"
        )
    step_audit = receipt.get("step_audit")
    if (
        not isinstance(step_audit, list)
        or len(step_audit) != FORMAL_GLOBAL_STEP
        or receipt.get("step_audit_sha256") != legacy.object_sha256(step_audit)
    ):
        raise FeasibleQuotientFinalizationError(
            "formal v8 step audit is incomplete or altered"
        )
    for index, record in enumerate(step_audit):
        if (
            not isinstance(record, Mapping)
            or record.get("optimizer_step") != index + 1
            or record.get("sigma_schedule_index") != index
            or record.get("teacher_mode") != "paired_displacement_only"
        ):
            raise FeasibleQuotientFinalizationError(
                f"formal v8 step audit differs at index {index}"
            )
    immutable = receipt.get("immutable_contract")
    if not isinstance(immutable, Mapping) or not isinstance(
        immutable.get("value"), Mapping
    ):
        raise FeasibleQuotientFinalizationError(
            "pending v8 receipt lacks its immutable contract"
        )
    value = immutable["value"]
    if immutable.get("digest") != legacy.object_sha256(value):
        raise FeasibleQuotientFinalizationError(
            "immutable v8 contract digest differs"
        )
    if (
        value.get("method") != v8_loader.METHOD_NAME
        or value.get("schema_version") != v8_loader.TRAINING_RECEIPT_SCHEMA
        or value.get("teacher_mode") != "paired_displacement_only"
        or value.get("pilot_scope") != "exact40_fixed_lr_falsification"
        or value.get("inference_loader_parity")
        != v8_loader._expected_parity_contract()
        or receipt.get("inference_loader_parity")
        != value.get("inference_loader_parity")
    ):
        raise FeasibleQuotientFinalizationError(
            "pending receipt is not the immutable RS-FQT pilot arm"
        )
    if receipt.get("inference_loader_parity_pending") is not True:
        raise FeasibleQuotientFinalizationError(
            "only a pending v8 receipt may be finalized"
        )
    expected_pending_artifact = {
        "schema_version": ARTIFACT_VALIDATION_SCHEMA,
        "verified": False,
        "status": PENDING_ARTIFACT_STATUS,
    }
    if receipt.get("artifact_validation") != expected_pending_artifact:
        raise FeasibleQuotientFinalizationError(
            "v8 artifact validation is not in its pristine pending state"
        )
    if (
        receipt.get("production_claim_forbidden") is not True
        or receipt.get("scientific_claim_authorized") is not False
    ):
        raise FeasibleQuotientFinalizationError(
            "pending v8 receipt lost its experimental-only restriction"
        )
    return {
        "receipt_digest": digest,
        "immutable_value": value,
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
    tree = _require_sha(
        expected_checkpoint_tree_sha256,
        label="base checkpoint tree SHA-256",
        pattern=_SHA256_RE,
    )
    immutable = receipt["immutable_contract"]["value"]
    checkpoint = receipt.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise FeasibleQuotientFinalizationError(
            "pending v8 receipt lacks base checkpoint identity"
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
        or immutable.get("checkpoint_tree_sha256") != tree
        or checkpoint.get("tree_sha256") != tree
    ):
        raise FeasibleQuotientFinalizationError(
            "v8 finalizer source or base-checkpoint identity differs"
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
    """Build a deterministic unpublished proof for the real v8 loader."""

    adapter = receipt.get("adapter")
    if not isinstance(adapter, Mapping):
        raise FeasibleQuotientFinalizationError(
            "pending v8 receipt lacks adapter identity"
        )
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
        "checkpoint_parameter_digest": _require_sha(
            adapter.get("checkpoint_parameter_digest"),
            label="training checkpoint parameter digest",
            pattern=_SHA256_RE,
        ),
        "pending_receipt_digest": _validate_receipt_digest(receipt),
        "validator_method_source_revision": _require_sha(
            method_source_revision,
            label="validator method source revision",
            pattern=_SHA1_RE,
        ),
        "validator_method_source_archive_sha256": _require_sha(
            method_source_archive_sha256,
            label="validator source archive SHA-256",
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
        "checkpoint_content_manifest_sha256": (
            v8_loader.v8_train.CHECKPOINT_CONTENT_MANIFEST_SHA256
        ),
        "checkpoint_content_file_count": (
            v8_loader.v8_train.CHECKPOINT_CONTENT_FILE_COUNT
        ),
        "loader_module": v8_loader.LOADER_MODULE,
        "finalizer_module": Path(__file__).name,
    }
    if value["finalizer_module"] != v8_loader.FINALIZER_MODULE:
        raise FeasibleQuotientFinalizationError(
            "v8 trainer names a different checkpoint finalizer"
        )
    value["digest"] = legacy.object_sha256(value)
    return value


def build_ready_receipt_candidate(
    pending_receipt: Mapping[str, Any],
    *,
    artifact_validation: Mapping[str, Any],
) -> dict[str, Any]:
    """Change only the pending gate/artifact proof and re-sign the receipt."""

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
            prefix=f".{path.name}.finalize-v8-",
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
    lock = checkpoint_root / ".finalize-feasible-quotient.lock"
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise FeasibleQuotientFinalizationError(
            f"v8 checkpoint finalization is already locked: {lock}"
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
    *, bernini_root: Path, base_checkpoint: Path
) -> Any:
    try:
        import torch
        from bernini.models.renderer import (
            BerniniRendererConfig,
            BerniniRendererModel,
        )
    except Exception as error:
        raise FeasibleQuotientFinalizationError(
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
        raise FeasibleQuotientFinalizationError(
            "cannot construct the fresh pinned Bernini-R 1.3B base"
        ) from error
    if any("lora_" in name.lower() for name, _ in base_model.named_modules()):
        raise FeasibleQuotientFinalizationError(
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
    # latest first, authoritative release gate last.  A crash between writes
    # leaves receipt.json pending and therefore cannot expose the checkpoint.
    _atomic_write_json(latest_path, ready_latest)
    try:
        _atomic_write_json(receipt_path, ready_receipt)
    except Exception as error:
        try:
            _atomic_write_json(latest_path, pending_latest)
        except Exception as rollback_error:
            raise FeasibleQuotientFinalizationError(
                "v8 receipt publication and latest.json rollback both failed"
            ) from rollback_error
        raise FeasibleQuotientFinalizationError(
            "v8 receipt publication failed; prior latest.json was restored"
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
    """Fresh-reload one pending exact-40 v8 artifact and publish readiness."""

    root = _resolve_formal_checkpoint_root(checkpoint_root)
    try:
        bundle = infer_base.resolve_adapter_bundle(root)
    except infer_base.InferenceContractError as error:
        raise FeasibleQuotientFinalizationError(str(error)) from error
    if bundle.checkpoint_root != root:
        raise FeasibleQuotientFinalizationError(
            "adapter bundle did not resolve to the formal v8 checkpoint"
        )
    receipt_path = _require_plain_file(
        bundle.training_receipt_path, label="pending v8 receipt"
    )
    latest_path = _require_plain_file(root.parent / "latest.json", label="latest index")

    with _exclusive_finalizer_lock(root):
        receipt_bytes_before = receipt_path.read_bytes()
        latest_bytes_before = latest_path.read_bytes()
        pending_receipt = _read_json(receipt_path, label="pending v8 receipt")
        pending_state = validate_pending_receipt(pending_receipt)
        pending_latest = _read_json(latest_path, label="pending latest index")
        try:
            indexed_root = Path(str(pending_latest.get("checkpoint"))).expanduser().resolve(
                strict=True
            )
        except OSError as error:
            raise FeasibleQuotientFinalizationError(
                "latest.json checkpoint cannot be resolved"
            ) from error
        if (
            indexed_root != root
            or pending_latest.get("global_step") != FORMAL_GLOBAL_STEP
        ):
            raise FeasibleQuotientFinalizationError(
                "latest.json does not identify this exact-40 v8 checkpoint"
            )

        try:
            validated_bernini, validated_veomni, bernini_revision, veomni_revision = (
                legacy.validate_source_trees(
                    bernini_root,
                    veomni_root,
                    expected_bernini_commit=expected_bernini_commit,
                    expected_veomni_commit=expected_veomni_commit,
                )
            )
            validated_base, transformer_config = legacy.validate_checkpoint(
                base_checkpoint
            )
        except legacy.TrainingContractError as error:
            raise FeasibleQuotientFinalizationError(str(error)) from error
        if transformer_config.get("num_attention_heads") != 12:
            raise FeasibleQuotientFinalizationError(
                "fresh base is not the pinned Bernini-R 1.3B transformer"
            )
        _validate_invocation_identity(
            pending_receipt,
            method_source_revision=method_source_revision,
            method_source_archive_sha256=method_source_archive_sha256,
            bernini_revision=bernini_revision,
            veomni_revision=veomni_revision,
            base_checkpoint=validated_base,
            expected_checkpoint_tree_sha256=expected_checkpoint_tree_sha256,
        )
        legacy.activate_source_trees(validated_bernini, validated_veomni)

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
            pending_receipt, artifact_validation=artifact
        )
        try:
            expected_identity = v8_loader.validate_training_adapter_contract(
                adapter_config,
                ready_candidate,
                expected_checkpoint_tree_sha256=expected_checkpoint_tree_sha256,
            )
        except v8_loader.FeasibleQuotientInferenceError as error:
            raise FeasibleQuotientFinalizationError(str(error)) from error

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
            raise FeasibleQuotientFinalizationError(
                "latest.json v8 release state differs"
            )

        base_model = _build_fresh_bernini_base(
            bernini_root=validated_bernini,
            base_checkpoint=validated_base,
        )
        try:
            loaded_model, tensor_count, active_count, loaded_identity = (
                v8_loader.strict_load_adapter(
                    base_model=base_model,
                    bundle=bundle,
                    adapter_config=adapter_config,
                    receipt=ready_candidate,
                    expected_checkpoint_tree_sha256=(
                        expected_checkpoint_tree_sha256
                    ),
                )
            )
        except v8_loader.FeasibleQuotientInferenceError as error:
            raise FeasibleQuotientFinalizationError(str(error)) from error
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
            raise FeasibleQuotientFinalizationError(
                "strict-reloaded v8 identity or exact46/92 counts differ"
            )
        if (
            legacy.file_sha256(bundle.adapter_config_path) != config_sha256
            or legacy.file_sha256(bundle.adapter_model_path) != model_sha256
            or receipt_path.read_bytes() != receipt_bytes_before
            or latest_path.read_bytes() != latest_bytes_before
        ):
            raise FeasibleQuotientFinalizationError(
                "v8 checkpoint artifacts changed during strict finalization"
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
            "Strict-reload and finalize one pending exact-40 Bernini RS-FQT "
            "v8 checkpoint; no video is generated"
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
    "FeasibleQuotientFinalizationError",
    "build_parser",
    "build_ready_artifact_validation",
    "build_ready_receipt_candidate",
    "finalize_checkpoint",
    "main",
    "validate_pending_receipt",
]


if __name__ == "__main__":
    raise SystemExit(main())
