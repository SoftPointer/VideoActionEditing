#!/usr/bin/env python3
"""Frozen Bernini owner-by-prompt same-state cross-query engineering micro.

This runtime replays four preregistered pure-T2V proposal trajectories from
one factorial-bank cell.  At four exact states of Bernini's native 40-step
UniPC schedule, the official owner-prompt field is evaluated first with hooks
off and retained unchanged for the native APG/scheduler path.  The frozen
transformer is then queried with all ten semantic prompts on the exact same
visual-token, rotary, timestep, and model-id objects.  Cross-query outputs are
observed and discarded; they never reach APG or ``scheduler.step``.

The executable is deliberately an engineering micro.  It writes FP32 FITQ
sufficient statistics and a hash-closed receipt, but it creates no optimizer,
performs no backward pass, writes no model weights, and cannot authorize a
scientific or training claim.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import timedelta
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import tempfile
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import dmiq_t2v_factorial_bank as factor_bank  # noqa: E402
import infer_native_identity_generation_canary as native  # noqa: E402
import inference_sigma_strata as sigma_strata  # noqa: E402
import internal_temporal_quotient_observer as fitq_observer  # noqa: E402
import tri_branch_unipc as sampler_contract  # noqa: E402


METHOD_NAME = "bernini-fitq-owner-prompt-cross-query-micro-v1"
RECEIPT_SCHEMA = "bernini-fitq-owner-prompt-cross-query-micro-receipt-v1"
STATISTICS_SCHEMA = "bernini-fitq-owner-prompt-cross-query-statistics-v1"

PROMPT_BRANCH_ORDER = tuple(factor_bank.BRANCH_ORDER)
OWNER_BRANCH_ORDER = (
    "full_action",
    "noop",
    "reverse_action",
    "camera_only",
)
SELECTED_SCHEDULE_INDICES = (22, 31, 36, 39)
EXPECTED_SELECTED_SIGMA_HEX = (
    "3f4dcdd4",
    "3f17da71",
    "3eb80796",
    "3df0f309",
)
EXPECTED_SELECTED_TIMESTEPS = (803, 593, 359, 117)
EXPECTED_SCHEDULE_SHA256 = (
    "3e5ad4473d133318026cc9e8f32399782bf06313691b58870c89d9c4c87c3d03"
)

EXPECTED_STEPS = 40
OFFICIAL_FORWARDS_PER_STEP = 2
EXTRA_QUERIES_PER_SELECTED_STEP = len(PROMPT_BRANCH_ORDER)
EXPECTED_OFFICIAL_FORWARDS_PER_OWNER = EXPECTED_STEPS * OFFICIAL_FORWARDS_PER_STEP
EXPECTED_EXTRA_FORWARDS_PER_OWNER = (
    len(SELECTED_SCHEDULE_INDICES) * EXTRA_QUERIES_PER_SELECTED_STEP
)
EXPECTED_TOTAL_FORWARDS_PER_OWNER = (
    EXPECTED_OFFICIAL_FORWARDS_PER_OWNER + EXPECTED_EXTRA_FORWARDS_PER_OWNER
)

_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_BASENAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class OwnerPromptCrossQueryError(RuntimeError):
    """Raised before ambiguous owner-by-prompt evidence can be emitted."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise OwnerPromptCrossQueryError(
            f"cross-query evidence is not canonical finite ASCII JSON: {error}"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha(value: Any, *, length: int, label: str) -> str:
    text = str(value)
    pattern = _SHA1_RE if length == 40 else _SHA256_RE
    if pattern.fullmatch(text) is None:
        raise OwnerPromptCrossQueryError(f"{label} must be lowercase SHA-{1 if length == 40 else 256}")
    return text


def _load_json(path: str | Path, *, label: str) -> tuple[dict[str, Any], Path, str]:
    requested = Path(path).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        raise OwnerPromptCrossQueryError(f"{label} must be an absolute non-symlink file")
    resolved = requested.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise OwnerPromptCrossQueryError(f"{label} must be a plain file")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OwnerPromptCrossQueryError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise OwnerPromptCrossQueryError(f"{label} root must be an object")
    return value, resolved, file_sha256(resolved)


def selected_schedule_contract() -> tuple[dict[str, Any], ...]:
    if sigma_strata.SCHEDULE_SHA256 != EXPECTED_SCHEDULE_SHA256:
        raise OwnerPromptCrossQueryError("pinned 40-step UniPC schedule digest changed")
    rows = []
    for index, expected_hex, expected_timestep in zip(
        SELECTED_SCHEDULE_INDICES,
        EXPECTED_SELECTED_SIGMA_HEX,
        EXPECTED_SELECTED_TIMESTEPS,
    ):
        observed_hex = sigma_strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[index]
        observed_timestep = sigma_strata.PINNED_TIMESTEPS[index]
        if observed_hex != expected_hex or observed_timestep != expected_timestep:
            raise OwnerPromptCrossQueryError("selected pinned UniPC states changed")
        rows.append(
            {
                "schedule_index": index,
                "timestep": observed_timestep,
                "sigma": sigma_strata.PINNED_POSITIVE_SIGMAS[index],
                "sigma_float32_be_hex": observed_hex,
            }
        )
    return tuple(rows)


def validate_micro_bank_bindings(
    manifest: Mapping[str, Any],
    bank_receipt: Mapping[str, Any],
    *,
    execution_group: str,
) -> dict[str, Any]:
    """Validate the immutable micro closure and return one group's four owners."""

    try:
        checked = factor_bank.validate_manifest(manifest)
    except Exception as error:
        raise OwnerPromptCrossQueryError(str(error)) from error
    if checked.get("profile") != "engineering_micro" or checked.get("attempt_rung") != 0:
        raise OwnerPromptCrossQueryError("owner cross-query runtime is micro/rung0 only")
    cells = checked.get("registered_design_cells")
    entries = checked.get("entries")
    if not isinstance(cells, list) or len(cells) != 2 or not isinstance(entries, list) or len(entries) != 20:
        raise OwnerPromptCrossQueryError("engineering micro must contain two cells and twenty entries")
    if execution_group not in factor_bank.GROUPS:
        raise OwnerPromptCrossQueryError("execution group must be one registered SP4 group")

    if bank_receipt.get("schema_version") != factor_bank.BANK_RECEIPT_SCHEMA:
        raise OwnerPromptCrossQueryError("factor bank receipt schema differs")
    declared = _require_sha(
        bank_receipt.get("receipt_digest"), length=64, label="bank receipt digest"
    )
    unsigned = dict(bank_receipt)
    unsigned.pop("receipt_digest", None)
    if object_sha256(unsigned) != declared:
        raise OwnerPromptCrossQueryError("bank receipt embedded digest differs")
    if (
        bank_receipt.get("manifest_digest") != checked["manifest_digest"]
        or bank_receipt.get("bank_id") != checked["bank_id"]
        or bank_receipt.get("profile") != "engineering_micro"
        or bank_receipt.get("entry_count") != 20
        or bank_receipt.get("proposal_cell_count") != 2
    ):
        raise OwnerPromptCrossQueryError("bank receipt manifest/count binding differs")
    closure = bank_receipt.get("condition_closure")
    interpretation = bank_receipt.get("interpretation")
    bank_entries = bank_receipt.get("entries")
    native_provenance = bank_receipt.get("native_method_provenance")
    renderer = checked.get("renderer_contract")
    if (
        not isinstance(closure, Mapping)
        or closure.get("all_native_entry_audits_pass") is not True
        or closure.get("all_cells_share_exact_initial_noise_across_ten_branches") is not True
        or closure.get("all_initial_noise_value_digests_independently_recomputed") is not True
        or closure.get("source_latent_or_reference_consumed") is not False
        or closure.get("target_video_consumed") is not False
        or closure.get("mask_flow_pose_track_trajectory_consumed") is not False
        or not isinstance(interpretation, Mapping)
        or interpretation.get("factorial_render_complete") is not True
        or interpretation.get("training_performed") is not False
        or interpretation.get("optimizer_update") != "null"
        or bank_receipt.get("attempt_rung") != 0
        or not isinstance(renderer, Mapping)
        or not isinstance(native_provenance, Mapping)
        or native_provenance.get("method_source_revision")
        != renderer.get("method_source_revision")
        or native_provenance.get("method_source_archive_sha256")
        != renderer.get("method_source_archive_sha256")
        or native_provenance.get("preregistered_in_manifest_before_render") is not True
        or native_provenance.get("all_entries_exact") is not True
        or not isinstance(bank_entries, list)
    ):
        raise OwnerPromptCrossQueryError("bank receipt frozen/render closure differs")
    if [row.get("entry_id") for row in bank_entries if isinstance(row, Mapping)] != [
        row["entry_id"] for row in entries
    ]:
        raise OwnerPromptCrossQueryError("bank receipt entry order differs from manifest")

    group_cells = [cell for cell in cells if cell.get("execution_group") == execution_group]
    if len(group_cells) != 1:
        raise OwnerPromptCrossQueryError("micro execution group must own exactly one cell")
    cell = group_cells[0]
    cell_entries = [
        entry for entry in entries if entry.get("proposal_cell_id") == cell.get("proposal_cell_id")
    ]
    if [entry.get("semantic_branch") for entry in cell_entries] != list(PROMPT_BRANCH_ORDER):
        raise OwnerPromptCrossQueryError("cell prompt branch order differs")
    receipt_by_id = {
        row["entry_id"]: row for row in bank_entries if isinstance(row, Mapping)
    }
    bound_rows = []
    for entry in cell_entries:
        rendered = receipt_by_id.get(entry["entry_id"])
        if not isinstance(rendered, Mapping):
            raise OwnerPromptCrossQueryError("bank entry is absent")
        if any(
            rendered.get(name) != entry.get(name)
            for name in (
                "entry_id",
                "semantic_branch",
                "proposal_cell_id",
                "execution_group",
                "seed",
            )
        ):
            raise OwnerPromptCrossQueryError("bank entry semantics differ from manifest")
        for name in (
            "design_slot_id",
            "analysis_split",
            "seed_replicate_id",
            "attempt_rung",
        ):
            if rendered.get(name) != entry.get(name):
                raise OwnerPromptCrossQueryError(
                    f"bank entry {name} differs from manifest"
                )
        for name in (
            "native_receipt_digest",
            "video_sha256",
            "clean_latent_sha256",
            "initial_noise_file_sha256",
            "initial_noise_tensor_value_sha256",
            "method_source_revision",
            "method_source_archive_sha256",
        ):
            _require_sha(
                rendered.get(name),
                length=40 if name == "method_source_revision" else 64,
                label=f"bank entry {name}",
            )
        if (
            rendered.get("method_source_revision")
            != renderer.get("method_source_revision")
            or rendered.get("method_source_archive_sha256")
            != renderer.get("method_source_archive_sha256")
            or rendered.get("initial_noise_value_digest_independently_recomputed")
            is not True
            or rendered.get("pure_t2v_condition_audit_pass") is not True
        ):
            raise OwnerPromptCrossQueryError("bank entry native provenance differs")
        bound_rows.append({"manifest": dict(entry), "bank": dict(rendered)})
    owner_rows = [
        row for branch in OWNER_BRANCH_ORDER for row in bound_rows
        if row["manifest"]["semantic_branch"] == branch
    ]
    if len(owner_rows) != len(OWNER_BRANCH_ORDER):
        raise OwnerPromptCrossQueryError("micro cell lacks the fixed four state owners")
    if len({row["manifest"]["entry_id"] for row in owner_rows}) != len(owner_rows):
        raise OwnerPromptCrossQueryError("micro owner IDs are not unique")
    if len({row["manifest"]["seed"] for row in bound_rows}) != 1:
        raise OwnerPromptCrossQueryError("same-cell prompt branches changed seed")
    if len({row["bank"]["initial_noise_tensor_value_sha256"] for row in bound_rows}) != 1:
        raise OwnerPromptCrossQueryError("same-cell bank entries changed initial Gaussian")
    return {
        "manifest": checked,
        "bank_receipt_digest": declared,
        "cell": dict(cell),
        "prompt_rows": tuple(bound_rows),
        "owner_rows": tuple(owner_rows),
    }


def validate_renderer_contract(renderer: Mapping[str, Any]) -> None:
    expected = {
        "implementation": "infer_native_identity_generation_canary.py",
        "implementation_arm": "t2v",
        "method_source_preregistered_before_render": True,
        "bernini_commit": factor_bank.BERNINI_COMMIT,
        "veomni_commit": factor_bank.VEOMNI_COMMIT,
        "checkpoint_tree_sha256": factor_bank.CHECKPOINT_TREE_SHA256,
        "frame_count": factor_bank.FRAME_COUNT,
        "fps": factor_bank.FPS,
        "video_height": factor_bank.VIDEO_HEIGHT,
        "video_width": factor_bank.VIDEO_WIDTH,
        "latent_shape": list(factor_bank.LATENT_SHAPE),
        "num_inference_steps": EXPECTED_STEPS,
        "guidance_mode": "t2v_apg",
        "omega_vid": 1.25,
        "omega_img": 4.5,
        "omega_txt": 4.0,
        "omega_scale": 0.8,
        "flow_shift": 5.0,
        "eta": 0.5,
        "norm_threshold": [50.0, 50.0],
        "momentum": 0.0,
        "target_initialization": "official_gen_wanx22_fresh_gaussian",
        "single_expert": "transformer_1",
        "ulysses_size": 4,
        "initial_noise_artifact_required": True,
        "initial_noise_raw_value_sha256_required": True,
        "full_source_video_count": 0,
        "source_derived_reference_count": 0,
        "target_mixed_with_source_latent": False,
        "source_or_reference_latent_forbidden": True,
        "external_mask_flow_pose_track_trajectory": False,
        "training_performed": False,
    }
    for name, wanted in expected.items():
        if renderer.get(name) != wanted:
            raise OwnerPromptCrossQueryError(
                f"factor-bank renderer contract {name} differs"
            )


def expected_cartesian_query_order(
    owner_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    rows = []
    for owner in owner_rows:
        entry = owner.get("manifest")
        if not isinstance(entry, Mapping):
            raise OwnerPromptCrossQueryError("owner row lacks manifest entry")
        for schedule_index in SELECTED_SCHEDULE_INDICES:
            for prompt_branch in PROMPT_BRANCH_ORDER:
                rows.append(
                    {
                        "owner_id": entry.get("entry_id"),
                        "owner_branch": entry.get("semantic_branch"),
                        "schedule_index": schedule_index,
                        "prompt_branch": prompt_branch,
                    }
                )
    if len(rows) != (
        len(OWNER_BRANCH_ORDER)
        * len(SELECTED_SCHEDULE_INDICES)
        * len(PROMPT_BRANCH_ORDER)
    ):
        raise OwnerPromptCrossQueryError("Cartesian query closure differs")
    return tuple(rows)


def observed_cartesian_query_order(
    owner_records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    rows = []
    for owner in owner_records:
        selected = owner.get("selected_states")
        if not isinstance(selected, list):
            raise OwnerPromptCrossQueryError("owner record lacks selected states")
        for state in selected:
            if not isinstance(state, Mapping):
                raise OwnerPromptCrossQueryError("selected state record is invalid")
            prompts = state.get("prompt_order")
            if not isinstance(prompts, list):
                raise OwnerPromptCrossQueryError("selected state lacks prompt order")
            for prompt_branch in prompts:
                rows.append(
                    {
                        "owner_id": owner.get("owner_id"),
                        "owner_branch": owner.get("owner_branch"),
                        "schedule_index": state.get("schedule_index"),
                        "prompt_branch": prompt_branch,
                    }
                )
    return tuple(rows)


def _tensor_identity(value: Any, *, label: str) -> dict[str, Any]:
    """Hash logical tensor values while retaining stable source-view metadata.

    Bernini constructs the per-step timestep with ``t.expand(1)``.  That is a
    one-element stride-0 view which PyTorch considers contiguous, so a plain
    ``tensor.contiguous().view(torch.uint8)`` can still fail: its last stride
    remains zero.  Only the private digest copy is normalized here.  The
    model-facing tensor object is never replaced or mutated.
    """

    torch = importlib.import_module("torch")
    if not isinstance(value, torch.Tensor) or value.numel() <= 0:
        raise OwnerPromptCrossQueryError(f"{label} must be a non-empty tensor")
    if value.layout != torch.strided:
        raise OwnerPromptCrossQueryError(f"{label} must use torch.strided layout")
    try:
        original_shape = [int(item) for item in value.shape]
        original_stride = [int(item) for item in value.stride()]
        original_storage_offset = int(value.storage_offset())
        original_metadata = {
            "object_type": f"{type(value).__module__}.{type(value).__qualname__}",
            "shape": original_shape,
            "stride": original_stride,
            "storage_offset": original_storage_offset,
            "layout": str(value.layout),
            "is_contiguous": bool(value.is_contiguous()),
            "requires_grad": bool(value.requires_grad),
        }

        cpu_source = value.detach().to(device="cpu")
        resolve_conj = getattr(cpu_source, "resolve_conj", None)
        if callable(resolve_conj):
            cpu_source = resolve_conj()
        resolve_neg = getattr(cpu_source, "resolve_neg", None)
        if callable(resolve_neg):
            cpu_source = resolve_neg()
        # ``clone(memory_format=contiguous_format)`` is intentional.  Unlike
        # ``contiguous()``, it materializes a standard stride even for a
        # size-one expanded dimension whose stride is zero.
        logical = cpu_source.clone(memory_format=torch.contiguous_format)
        if not logical.is_contiguous() or any(
            stride == 0 and size > 0
            for stride, size in zip(logical.stride(), logical.shape)
        ):
            raise OwnerPromptCrossQueryError(
                f"{label} logical digest clone is not physically contiguous"
            )
        if not bool(torch.isfinite(logical).all().item()):
            raise OwnerPromptCrossQueryError(f"{label} contains non-finite values")
        raw = logical.view(torch.uint8).numpy().tobytes(order="C")
    except OwnerPromptCrossQueryError:
        raise
    except Exception as error:
        raise OwnerPromptCrossQueryError(
            f"failed to audit logical tensor values for {label}: {error}"
        ) from error

    metadata = {
        "shape": [int(item) for item in logical.shape],
        "dtype": str(logical.dtype),
        "numel": int(logical.numel()),
        "byte_count": len(raw),
    }
    expected_byte_count = int(logical.numel() * logical.element_size())
    if len(raw) != expected_byte_count or metadata["shape"] != original_shape:
        raise OwnerPromptCrossQueryError(
            f"{label} logical digest clone geometry/byte count differs"
        )
    payload = canonical_json_bytes(metadata) + b"\0" + raw
    return {
        **metadata,
        "content_sha256": hashlib.sha256(payload).hexdigest(),
        # Retain this established key for _same_raw_tensor.  It denotes the
        # exact row-major logical value bytes, never padding/aliased storage.
        "raw_storage_sha256": hashlib.sha256(raw).hexdigest(),
        "finite": True,
        "label": label,
        "original_tensor": original_metadata,
        "original_tensor_object_preserved": True,
        "logical_value_digest_source": "detached_cpu_contiguous_clone",
        "python_object_id_or_data_ptr_serialized": False,
    }


def _nested_value_identity(value: Any, *, label: str) -> dict[str, Any]:
    """Hash a tensor/tree value without serializing rank-local object IDs."""

    torch = importlib.import_module("torch")
    if isinstance(value, torch.Tensor):
        identity = _tensor_identity(value, label=label)
        identity.pop("label", None)
        return {"kind": "tensor", "identity": identity}
    if isinstance(value, Mapping):
        rows = []
        for key in sorted(value):
            if not isinstance(key, str):
                raise OwnerPromptCrossQueryError(f"{label} mapping key must be text")
            rows.append(
                [key, _nested_value_identity(value[key], label=f"{label}.{key}")]
            )
        return {"kind": "mapping", "items": rows}
    if isinstance(value, (tuple, list)):
        return {
            "kind": "tuple" if isinstance(value, tuple) else "list",
            "items": [
                _nested_value_identity(item, label=f"{label}[{index}]")
                for index, item in enumerate(value)
            ],
        }
    if value is None or isinstance(value, (str, bool, int)):
        return {"kind": "scalar", "value": value}
    if isinstance(value, float) and math.isfinite(value):
        return {"kind": "scalar", "value": value}
    raise OwnerPromptCrossQueryError(f"{label} has unsupported identity type")


_RAW_IDENTITY_KEYS = ("shape", "dtype", "numel", "byte_count", "raw_storage_sha256")


def _same_raw_tensor(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(left.get(key) == right.get(key) for key in _RAW_IDENTITY_KEYS)


def _global_context(context: Any) -> dict[str, Any]:
    value = dict(context.as_dict())
    value.pop("sp_rank", None)
    value["rank_scope"] = "SP4_all_reduced"
    return value


@dataclass
class _ActiveOwner:
    owner_id: str
    owner_branch: str
    owner_prompt: Any
    uncond_prompt: Any
    official_forward_calls: int = 0
    extra_forward_calls: int = 0
    scheduler_calls: int = 0
    waiting_for_owner: bool = False
    pending_step_index: Optional[int] = None
    negative_bound: Optional[Mapping[str, Any]] = None
    selected_records: list[dict[str, Any]] = field(default_factory=list)


class OwnerPromptCrossQueryBridge:
    """Strict reversible read-only wrapper for one pinned T2V owner replay."""

    def __init__(
        self,
        diffusion: Any,
        *,
        prompt_embeds: Mapping[str, Any],
        prompt_lengths: Mapping[str, int],
        statistics_dir: Path,
        distributed_rank: int,
    ) -> None:
        self.diffusion = sampler_contract.resolve_diffusion_core(diffusion)
        self.scheduler = self.diffusion.scheduler
        self.prompt_embeds = dict(prompt_embeds)
        self.prompt_lengths = dict(prompt_lengths)
        if tuple(self.prompt_embeds) != PROMPT_BRANCH_ORDER:
            raise OwnerPromptCrossQueryError("prompt embedding order differs")
        if tuple(self.prompt_lengths) != PROMPT_BRANCH_ORDER:
            raise OwnerPromptCrossQueryError("prompt length order differs")
        self.prompt_value_identities = {
            branch: _nested_value_identity(value, label=f"prompt.{branch}")
            for branch, value in self.prompt_embeds.items()
        }
        self.statistics_dir = statistics_dir
        self.rank = int(distributed_rank)
        self._original_sample = self.diffusion.sample
        self._original_shared_step = self.diffusion.shared_step
        self._original_scheduler_step = self.scheduler.step
        self._active: Optional[_ActiveOwner] = None
        self._patches: list[tuple[Any, str, bool, Any]] = []
        self._installed = False
        self.artifacts: list[dict[str, Any]] = []
        self.owner_records: list[dict[str, Any]] = []

    def _set_patch(self, owner: Any, name: str, value: Any) -> None:
        instance = vars(owner)
        had = name in instance
        previous = instance.get(name)
        setattr(owner, name, value)
        self._patches.append((owner, name, had, previous))

    def install(self) -> None:
        if self._installed:
            raise OwnerPromptCrossQueryError("cross-query bridge is already installed")
        for owner, name in (
            (self.diffusion, "sample"),
            (self.diffusion, "shared_step"),
            (self.scheduler, "step"),
        ):
            if name in vars(owner):
                raise OwnerPromptCrossQueryError(f"refusing stacked {name} wrapper")

        def sample_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_sample(*args, **kwargs)

        def shared_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_shared_step(*args, **kwargs)

        def scheduler_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_scheduler_step(*args, **kwargs)

        for wrapper in (sample_wrapper, shared_wrapper, scheduler_wrapper):
            setattr(wrapper, "_fitq_owner_prompt_cross_query_micro", self)
        try:
            self._set_patch(self.diffusion, "sample", sample_wrapper)
            self._set_patch(self.diffusion, "shared_step", shared_wrapper)
            self._set_patch(self.scheduler, "step", scheduler_wrapper)
        except Exception:
            self.restore()
            raise
        self._installed = True

    def restore(self) -> None:
        errors = []
        while self._patches:
            owner, name, had, previous = self._patches.pop()
            try:
                if had:
                    setattr(owner, name, previous)
                else:
                    delattr(owner, name)
            except Exception as error:  # pragma: no cover - deployment failure
                errors.append(error)
        self._active = None
        self._installed = False
        if errors:
            raise OwnerPromptCrossQueryError("failed to restore cross-query wrappers") from errors[0]

    def run_owner(self, *, owner_id: str, owner_branch: str, sample_kwargs: Mapping[str, Any]) -> Any:
        if self._active is not None:
            raise OwnerPromptCrossQueryError("nested owner replay is forbidden")
        if owner_branch not in OWNER_BRANCH_ORDER:
            raise OwnerPromptCrossQueryError("owner branch is not preregistered")
        if _SAFE_BASENAME_RE.fullmatch(owner_id) is None:
            raise OwnerPromptCrossQueryError("owner ID is unsafe for an artifact basename")
        expected_prompt = self.prompt_embeds[owner_branch]
        if sample_kwargs.get("prompt_embeds") is not expected_prompt:
            raise OwnerPromptCrossQueryError("owner sample prompt object differs")
        state = _ActiveOwner(
            owner_id=owner_id,
            owner_branch=owner_branch,
            owner_prompt=expected_prompt,
            uncond_prompt=sample_kwargs.get("uncond_prompt_embeds"),
        )
        if state.uncond_prompt is None:
            raise OwnerPromptCrossQueryError("native T2V APG requires negative prompt")
        self._active = state
        before_artifacts = len(self.artifacts)
        try:
            result = self.diffusion.sample(**dict(sample_kwargs))
            if state.waiting_for_owner or state.pending_step_index is not None:
                raise OwnerPromptCrossQueryError("owner sample ended with incomplete step")
            if state.official_forward_calls != EXPECTED_OFFICIAL_FORWARDS_PER_OWNER:
                raise OwnerPromptCrossQueryError("official owner forward count differs")
            if state.extra_forward_calls != EXPECTED_EXTRA_FORWARDS_PER_OWNER:
                raise OwnerPromptCrossQueryError("owner cross-query count differs")
            if state.scheduler_calls != EXPECTED_STEPS:
                raise OwnerPromptCrossQueryError("original scheduler call count differs")
            if len(state.selected_records) != len(SELECTED_SCHEDULE_INDICES):
                raise OwnerPromptCrossQueryError("selected owner-state coverage differs")
            if len(self.artifacts) - before_artifacts != len(SELECTED_SCHEDULE_INDICES):
                raise OwnerPromptCrossQueryError("selected statistics artifact count differs")
            self.owner_records.append(
                {
                    "owner_id": owner_id,
                    "owner_branch": owner_branch,
                    "official_forward_calls": state.official_forward_calls,
                    "extra_cross_query_forward_calls": state.extra_forward_calls,
                    "total_transformer_forwards": (
                        state.official_forward_calls + state.extra_forward_calls
                    ),
                    "original_scheduler_calls": state.scheduler_calls,
                    "selected_states": list(state.selected_records),
                }
            )
            return result
        finally:
            self._active = None

    def _wrapped_sample(self, *args: Any, **kwargs: Any) -> Any:
        state = self._active
        if state is None:
            raise OwnerPromptCrossQueryError("sample ran outside explicit owner replay")
        values = sampler_contract._bind_call(self._original_sample, args, kwargs)
        expected = native.native_sampling_contract(
            "t2v", steps=EXPECTED_STEPS, seed=int(values.get("seed"))
        )
        for name in (
            "num_frames",
            "num_inference_steps",
            "guidance_mode",
            "omega_vid",
            "omega_img",
            "omega_txt",
            "omega_scale",
            "flow_shift",
            "seed",
            "eta",
            "norm_threshold",
            "momentum",
        ):
            observed = values.get(name)
            wanted = expected[name]
            if isinstance(wanted, tuple):
                observed = tuple(observed)
            if observed != wanted:
                raise OwnerPromptCrossQueryError(f"native sample {name} differs")
        if values.get("prompt_embeds") is not state.owner_prompt:
            raise OwnerPromptCrossQueryError("native owner prompt identity differs")
        if values.get("uncond_prompt_embeds") is not state.uncond_prompt:
            raise OwnerPromptCrossQueryError("native negative prompt identity differs")
        for name in (
            "image_vae_latents",
            "multi_video_vae_latents",
            "multi_image_vae_latents",
        ):
            if values.get(name) is not None:
                raise OwnerPromptCrossQueryError("source/reference latent entered pure T2V replay")
        return self._original_sample(*args, **kwargs)

    def _wrapped_shared_step(self, *args: Any, **kwargs: Any) -> Any:
        state = self._active
        if state is None:
            raise OwnerPromptCrossQueryError("shared_step ran outside owner replay")
        bound = sampler_contract._bind_call(self._original_shared_step, args, kwargs)
        prompt = bound.get("cond_embeds")
        if not state.waiting_for_owner:
            if prompt is not state.uncond_prompt:
                raise OwnerPromptCrossQueryError("expected native negative forward first")
            result = self._original_shared_step(*args, **kwargs)
            state.official_forward_calls += 1
            state.waiting_for_owner = True
            state.pending_step_index = None
            state.negative_bound = bound
            return result

        if prompt is not state.owner_prompt:
            raise OwnerPromptCrossQueryError("native owner conditional prompt differs")
        negative_bound = state.negative_bound
        if not isinstance(negative_bound, Mapping):
            raise OwnerPromptCrossQueryError("negative forward binding is absent")
        for name in ("noisy_latents", "timesteps", "rotary_embs"):
            if negative_bound.get(name) is not bound.get(name):
                raise OwnerPromptCrossQueryError(f"negative/owner {name} object differs")
        for name in ("model_id", "batch_vae_seqlen"):
            sampler_contract._equal_metadata(
                negative_bound.get(name), bound.get(name), label=name
            )

        step_index, sigma, sigma_float = sampler_contract._resolve_sigma(
            self.scheduler, bound["timesteps"]
        )
        if step_index != state.scheduler_calls:
            raise OwnerPromptCrossQueryError("shared_step schedule index differs from owner step")
        if sigma_strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[step_index] != (
            struct.pack(">f", sigma_float).hex()
        ):
            raise OwnerPromptCrossQueryError("runtime selected sigma bits differ")

        # Crucial noninterference order: this hook-off field is the only owner
        # conditional result returned to Bernini APG and later UniPC.
        official_owner = self._original_shared_step(*args, **kwargs)
        state.official_forward_calls += 1
        if step_index in SELECTED_SCHEDULE_INDICES:
            record = self._run_selected_cross_query(
                state=state,
                step_index=step_index,
                sigma=sigma,
                sigma_float=sigma_float,
                owner_args=args,
                owner_kwargs=kwargs,
                owner_bound=bound,
                official_owner=official_owner,
            )
            state.selected_records.append(record)
            state.pending_step_index = step_index
        else:
            state.pending_step_index = -step_index - 1
        state.waiting_for_owner = False
        return official_owner

    def _run_selected_cross_query(
        self,
        *,
        state: _ActiveOwner,
        step_index: int,
        sigma: Any,
        sigma_float: float,
        owner_args: Sequence[Any],
        owner_kwargs: Mapping[str, Any],
        owner_bound: Mapping[str, Any],
        official_owner: Any,
    ) -> dict[str, Any]:
        observer = fitq_observer.InternalTemporalQuotientObserver(
            self.diffusion, capture_exact_block0=True
        )
        owner_state_before = {
            name: _nested_value_identity(owner_bound.get(name), label=f"before.{name}")
            for name in (
                "noisy_latents",
                "timesteps",
                "rotary_embs",
                "model_id",
                "batch_vae_seqlen",
            )
        }
        local_results: dict[str, Any] = {}
        global_results: list[Any] = []
        field_identities: dict[str, dict[str, Any]] = {}
        observer.install()
        try:
            for branch in PROMPT_BRANCH_ORDER:
                query_args, query_kwargs = sampler_contract._replace_argument(
                    self._original_shared_step,
                    owner_args,
                    owner_kwargs,
                    name="cond_embeds",
                    value=self.prompt_embeds[branch],
                )
                query_args, query_kwargs = sampler_contract._replace_argument(
                    self._original_shared_step,
                    query_args,
                    query_kwargs,
                    name="batch_text_seqlen",
                    value=[self.prompt_lengths[branch]],
                )
                query_bound = sampler_contract._bind_call(
                    self._original_shared_step, query_args, query_kwargs
                )
                for name in ("noisy_latents", "timesteps", "rotary_embs"):
                    if query_bound.get(name) is not owner_bound.get(name):
                        raise OwnerPromptCrossQueryError(
                            f"owner/prompt query changed exact {name} object"
                        )
                for name in ("model_id", "batch_vae_seqlen"):
                    sampler_contract._equal_metadata(
                        owner_bound.get(name), query_bound.get(name), label=name
                    )
                context = fitq_observer.FITQObserverContext(
                    mode="t2v",
                    branch=f"{state.owner_id}--{branch}",
                    sigma=sigma_float,
                    lambda_value=1.0,
                    sp_rank=self.rank,
                )
                with observer.capture(context) as session:
                    field_value = self._original_shared_step(*query_args, **query_kwargs)
                local = session.result
                if local is None:
                    raise OwnerPromptCrossQueryError("FITQ observer returned no result")
                global_result = fitq_observer.all_reduce_local_sufficient_statistics(local)
                local_results[branch] = local
                global_results.append(global_result)
                field_identities[branch] = _tensor_identity(
                    field_value, label=f"cross_query_{state.owner_id}_{branch}_{step_index}"
                )
                state.extra_forward_calls += 1
                if (
                    _nested_value_identity(
                        self.prompt_embeds[branch], label=f"prompt_after.{branch}"
                    )
                    != self.prompt_value_identities[branch]
                ):
                    raise OwnerPromptCrossQueryError(
                        "cross-query forward mutated a prompt embedding"
                    )
        finally:
            if observer.active:
                observer.abort_forward()
            if observer.installed:
                observer.remove()

        owner_state_after = {
            name: _nested_value_identity(owner_bound.get(name), label=f"after.{name}")
            for name in owner_state_before
        }
        if owner_state_after != owner_state_before:
            raise OwnerPromptCrossQueryError(
                "cross-query forward mutated native owner state/RoPE/timestep metadata"
            )

        reference = local_results[PROMPT_BRANCH_ORDER[0]]
        parity = {}
        for branch in PROMPT_BRANCH_ORDER:
            result = fitq_observer.exact_block0_parity(reference, local_results[branch])
            if result.get("all") is not True:
                raise OwnerPromptCrossQueryError("same-state block0 parity failed")
            parity[branch] = dict(result)
        hook_off_identity = _tensor_identity(
            official_owner, label=f"official_owner_hook_off_{state.owner_id}_{step_index}"
        )
        hook_on_owner_identity = field_identities[state.owner_branch]
        if not _same_raw_tensor(hook_off_identity, hook_on_owner_identity):
            raise OwnerPromptCrossQueryError("hook-on owner query changed official field")
        artifact = self._write_state_artifact(
            state=state,
            step_index=step_index,
            sigma_float=sigma_float,
            results=global_results,
        )
        self.artifacts.append(artifact)
        return {
            "schedule_index": step_index,
            "timestep": sigma_strata.PINNED_TIMESTEPS[step_index],
            "sigma": sigma_float,
            "sigma_float32_be_hex": sigma_strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[step_index],
            "prompt_order": list(PROMPT_BRANCH_ORDER),
            "same_visual_token_object": True,
            "same_rotary_object": True,
            "same_timestep_object": True,
            "same_model_id_and_vae_lengths": True,
            "owner_state_value_identity_before": owner_state_before,
            "owner_state_value_identity_after": owner_state_after,
            "owner_state_value_byte_exact_after_all_queries": True,
            "prompt_embedding_values_byte_exact_after_query": True,
            "block0_input_and_attn1_exact_all_prompts": True,
            "block0_parity": parity,
            "hook_off_owner_field": hook_off_identity,
            "hook_on_owner_field": hook_on_owner_identity,
            "hook_on_off_owner_field_byte_exact": True,
            "cross_query_outputs_forwarded_to_apg_or_scheduler": False,
            "statistics_artifact": artifact,
        }

    def _write_state_artifact(
        self,
        *,
        state: _ActiveOwner,
        step_index: int,
        sigma_float: float,
        results: Sequence[Any],
    ) -> dict[str, Any]:
        torch = importlib.import_module("torch")
        order = fitq_observer.expected_site_order()
        if len(results) != len(PROMPT_BRANCH_ORDER):
            raise OwnerPromptCrossQueryError("statistics prompt coverage differs")
        means = []
        seconds = []
        shared_count = None
        for result in results:
            result.require_complete()
            if not result.globally_reduced:
                raise OwnerPromptCrossQueryError("statistics were not SP4 reduced")
            sums = torch.stack([result.sites[site].sum for site in order])
            sumsqs = torch.stack([result.sites[site].sumsq for site in order])
            count = torch.stack([result.sites[site].count for site in order])
            denominator = count[:, :, :, None].clamp_min(1.0)
            means.append((sums / denominator).detach().to(device="cpu"))
            seconds.append((sumsqs / denominator).detach().to(device="cpu"))
            if shared_count is None:
                shared_count = count.detach().to(device="cpu")
            elif not torch.equal(shared_count, count.detach().to(device="cpu")):
                raise OwnerPromptCrossQueryError("prompt statistics counts differ")
        mean_tensor = torch.stack(means)
        second_tensor = torch.stack(seconds)
        if mean_tensor.dtype != torch.float32 or second_tensor.dtype != torch.float32:
            raise OwnerPromptCrossQueryError("statistics artifacts must be FP32")
        filename = f"{state.owner_id}.step-{step_index:02d}.owner-prompt-fitq.pt"
        path = self.statistics_dir / filename
        metadata: list[Any] = [None]
        dist = torch.distributed
        if self.rank == 0:
            if path.exists() or path.is_symlink():
                raise OwnerPromptCrossQueryError("refusing to overwrite statistics artifact")
            payload = {
                "schema_version": STATISTICS_SCHEMA,
                "owner_id": state.owner_id,
                "owner_branch": state.owner_branch,
                "schedule_index": step_index,
                "sigma": sigma_float,
                "prompt_order": list(PROMPT_BRANCH_ORDER),
                "site_order": list(order),
                "phase_head_mean_fp32": mean_tensor,
                "phase_head_second_moment_fp32": second_tensor,
                "global_phase_head_count_fp32": shared_count,
            }
            temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
            try:
                torch.save(payload, str(temporary))
                os.replace(temporary, path)
            finally:
                if temporary.exists():
                    temporary.unlink()
            metadata[0] = {
                "path": str(path),
                "sha256": file_sha256(path),
                "size_bytes": int(path.stat().st_size),
                "schema_version": STATISTICS_SCHEMA,
                "owner_id": state.owner_id,
                "owner_branch": state.owner_branch,
                "schedule_index": step_index,
                "sigma": sigma_float,
                "prompt_order": list(PROMPT_BRANCH_ORDER),
                "site_order": list(order),
                "mean_shape": list(mean_tensor.shape),
                "second_moment_shape": list(second_tensor.shape),
                "count_shape": list(shared_count.shape),
                "dtype": "torch.float32",
                "contains_model_weights": False,
                "is_checkpoint": False,
            }
        dist.broadcast_object_list(metadata, src=0)
        if not isinstance(metadata[0], Mapping):
            raise OwnerPromptCrossQueryError("statistics metadata broadcast failed")
        return dict(metadata[0])

    def _wrapped_scheduler_step(self, *args: Any, **kwargs: Any) -> Any:
        state = self._active
        if state is None or state.waiting_for_owner or state.pending_step_index is None:
            raise OwnerPromptCrossQueryError("scheduler.step arrived before complete owner pair")
        model_output = sampler_contract._extract_argument(
            args, kwargs, index=0, name="model_output"
        )
        timestep = sampler_contract._extract_argument(args, kwargs, index=1, name="timestep")
        sample = sampler_contract._extract_argument(args, kwargs, index=2, name="sample")
        index, _, _ = sampler_contract._resolve_sigma(self.scheduler, timestep)
        decoded = (
            state.pending_step_index
            if state.pending_step_index >= 0
            else -state.pending_step_index - 1
        )
        if index != decoded or index != state.scheduler_calls:
            raise OwnerPromptCrossQueryError("scheduler index differs from queried owner state")
        if index in SELECTED_SCHEDULE_INDICES:
            selected = state.selected_records[-1]
            if selected["schedule_index"] != index:
                raise OwnerPromptCrossQueryError("selected scheduler record order differs")
            selected["scheduler_sample_packed"] = _tensor_identity(
                sample, label=f"scheduler_owner_state_{state.owner_id}_{index}"
            )
            selected["scheduler_model_output"] = _tensor_identity(
                model_output, label=f"scheduler_owner_output_{state.owner_id}_{index}"
            )
            selected["original_scheduler_arguments_replaced"] = False
        result = self._original_scheduler_step(*args, **kwargs)
        state.scheduler_calls += 1
        state.pending_step_index = None
        return result


def _fresh_output_directory(value: str | Path) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested == Path("/"):
        raise OwnerPromptCrossQueryError("output directory must be absolute and non-root")
    if _SAFE_BASENAME_RE.fullmatch(requested.name) is None:
        raise OwnerPromptCrossQueryError("output directory basename is unsafe")
    parent = requested.parent.resolve(strict=True)
    output = parent / requested.name
    if output.exists() or output.is_symlink():
        raise OwnerPromptCrossQueryError("refusing to reuse output directory")
    return output


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise OwnerPromptCrossQueryError("refusing to overwrite receipt")
    payload = canonical_json_bytes(value) + b"\n"
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--factor-manifest", required=True)
    parser.add_argument("--expected-factor-manifest-file-sha256", required=True)
    parser.add_argument("--factor-bank-receipt", required=True)
    parser.add_argument("--expected-factor-bank-receipt-file-sha256", required=True)
    parser.add_argument("--bank-output-root", required=True)
    parser.add_argument("--execution-group", required=True, choices=factor_bank.GROUPS)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runtime-source-revision", required=True)
    parser.add_argument("--runtime-source-archive-sha256", required=True)
    parser.add_argument("--launcher-source-sha256", required=True)
    parser.add_argument(
        "--expected-bernini-commit", default=native.legacy.trainer.BERNINI_OFFICIAL_COMMIT
    )
    parser.add_argument(
        "--expected-veomni-commit", default=native.legacy.trainer.VEOMNI_TESTED_COMMIT
    )
    parser.add_argument(
        "--expected-checkpoint-tree-sha256",
        default=native.legacy.trainer.CHECKPOINT_TREE_SHA256,
    )
    return parser


def _validate_cli(args: argparse.Namespace) -> None:
    for name in ("runtime_source_revision", "expected_bernini_commit", "expected_veomni_commit"):
        _require_sha(getattr(args, name), length=40, label=name)
    for name in (
        "expected_factor_manifest_file_sha256",
        "expected_factor_bank_receipt_file_sha256",
        "runtime_source_archive_sha256",
        "launcher_source_sha256",
        "expected_checkpoint_tree_sha256",
    ):
        _require_sha(getattr(args, name), length=64, label=name)
    if args.expected_bernini_commit != native.legacy.trainer.BERNINI_OFFICIAL_COMMIT:
        raise OwnerPromptCrossQueryError("Bernini commit differs from pinned release")
    if args.expected_veomni_commit != native.legacy.trainer.VEOMNI_TESTED_COMMIT:
        raise OwnerPromptCrossQueryError("VeOmni commit differs from pinned release")
    if args.expected_checkpoint_tree_sha256 != native.legacy.trainer.CHECKPOINT_TREE_SHA256:
        raise OwnerPromptCrossQueryError("checkpoint tree differs from pinned release")
    selected_schedule_contract()


def _load_clean_latent(path: Path, *, expected_file_sha256: str) -> Any:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise OwnerPromptCrossQueryError("bank clean latent must be an absolute plain file")
    if file_sha256(path) != expected_file_sha256:
        raise OwnerPromptCrossQueryError("bank clean latent file SHA-256 differs")
    try:
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - AUH runtime dependency
        raise OwnerPromptCrossQueryError("safetensors is required") from error
    with safe_open(str(path), framework="pt", device="cpu") as opened:
        if list(opened.keys()) != ["normalized_clean_latent"]:
            raise OwnerPromptCrossQueryError("bank clean latent tensor key differs")
        value = opened.get_tensor("normalized_clean_latent").contiguous()
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_cli(args)
    output_dir = _fresh_output_directory(args.output_dir)
    manifest, manifest_path, manifest_file_sha = _load_json(
        args.factor_manifest, label="factor manifest"
    )
    if manifest_file_sha != args.expected_factor_manifest_file_sha256:
        raise OwnerPromptCrossQueryError("factor manifest file SHA-256 differs")
    bank_receipt, bank_receipt_path, bank_receipt_file_sha = _load_json(
        args.factor_bank_receipt, label="factor bank receipt"
    )
    if bank_receipt_file_sha != args.expected_factor_bank_receipt_file_sha256:
        raise OwnerPromptCrossQueryError("factor bank receipt file SHA-256 differs")
    bindings = validate_micro_bank_bindings(
        manifest, bank_receipt, execution_group=args.execution_group
    )
    renderer_contract = bindings["manifest"]["renderer_contract"]
    validate_renderer_contract(renderer_contract)
    bank_root_requested = Path(args.bank_output_root).expanduser()
    if not bank_root_requested.is_absolute() or bank_root_requested.is_symlink():
        raise OwnerPromptCrossQueryError("bank output root must be absolute and non-symlink")
    bank_root = bank_root_requested.resolve(strict=True)
    if bank_root != bank_root_requested or not bank_root.is_dir():
        raise OwnerPromptCrossQueryError("bank output root must be canonical")

    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            native.legacy.trainer.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = native.legacy.trainer.validate_checkpoint(
            args.checkpoint
        )
    except Exception as error:
        raise OwnerPromptCrossQueryError(str(error)) from error
    if int(transformer_config["num_attention_heads"]) % 4:
        raise OwnerPromptCrossQueryError("Bernini attention heads do not divide Ulysses4")
    native.legacy.trainer.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer
    import bernini.models.wan_diffusion as wan_diffusion
    from bernini.cli import DEFAULT_NEG_PROMPT
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state

    distributed = native.legacy.inference_distributed_contract()
    if distributed.world_size != 4 or distributed.ulysses_size != 4:
        raise OwnerPromptCrossQueryError("runtime requires one exact WORLD4/Ulysses4 group")
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        raise OwnerPromptCrossQueryError("runtime requires four AUH ROCm GPUs")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=180),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=4)
    device = torch.device("cuda", distributed.local_rank)

    checkpoint_manifest = Path(args.checkpoint_content_manifest).expanduser()
    checkpoint_rows: list[Any] = [None]
    if distributed.rank == 0:
        try:
            checkpoint_rows[0] = {
                "ok": True,
                "identity": native.source_audit.validate_checkpoint_content(
                    checkpoint, checkpoint_manifest
                ),
            }
        except Exception as error:
            checkpoint_rows[0] = {"ok": False, "error": str(error)}
    dist.broadcast_object_list(checkpoint_rows, src=0)
    if not isinstance(checkpoint_rows[0], Mapping) or checkpoint_rows[0].get("ok") is not True:
        raise OwnerPromptCrossQueryError("checkpoint content validation failed")
    checkpoint_identity = dict(checkpoint_rows[0]["identity"])

    if distributed.rank == 0:
        output_dir.mkdir(mode=0o755, parents=False, exist_ok=False)
        (output_dir / "statistics").mkdir(mode=0o755)
    dist.barrier()
    statistics_dir = output_dir / "statistics"

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), subfolder="tokenizer", **native.legacy.tokenizer_load_kwargs()
    )
    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **native.legacy.inference_renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    native.legacy.trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    model = BerniniRendererModel(config)
    model.requires_grad_(False)
    model.eval()
    freeze_before = native.source_audit.model_freeze_certificate(model)
    model.to(device)
    resolved_transformer = fitq_observer.resolve_pinned_30_block_transformer(
        model.diff_dec
    )
    if resolved_transformer is None:
        raise OwnerPromptCrossQueryError("single active Bernini transformer is absent")

    prompt_rows = {row["manifest"]["semantic_branch"]: row for row in bindings["prompt_rows"]}
    prompt_tokens = {}
    for branch in PROMPT_BRANCH_ORDER:
        body = prompt_rows[branch]["manifest"]["prompt"]
        full = native.build_task_prompt("t2v", body, prompt_cleaner=prompt_clean)
        prompt_tokens[branch] = native.legacy._tokenize_training_prompt(tokenizer, full)
    negative_ids, negative_mask = native.legacy._tokenize_renderer_negative(
        tokenizer, DEFAULT_NEG_PROMPT
    )

    model.t5_text_encoder.to(device)
    prompt_embeds = {}
    prompt_lengths = {}
    with torch.inference_mode():
        for branch in PROMPT_BRANCH_ORDER:
            ids, mask = prompt_tokens[branch]
            embedding = model.encode_prompt(ids.to(device), mask.to(device))
            prompt_embeds[branch] = embedding
            prompt_lengths[branch] = int(embedding.shape[1])
        uncond_embeds = model.encode_prompt(
            negative_ids.to(device), negative_mask.to(device)
        )
    model.t5_text_encoder.to("cpu")
    torch.cuda.empty_cache()

    schedule_evidence = sigma_strata.audit_runtime_unipc_schedule(
        model.diff_dec.scheduler, initialize=True
    )
    sampler_contract.validate_runtime_source_identity(
        bernini_commit=bernini_revision,
        wan_diffusion_path=Path(wan_diffusion.__file__).resolve(),
    )
    bridge = OwnerPromptCrossQueryBridge(
        model.diff_dec,
        prompt_embeds=prompt_embeds,
        prompt_lengths=prompt_lengths,
        statistics_dir=statistics_dir,
        distributed_rank=distributed.rank,
    )
    bridge.install()
    owner_outputs = []
    try:
        with torch.inference_mode():
            for row in bindings["owner_rows"]:
                entry = row["manifest"]
                bank = row["bank"]
                owner_id = entry["entry_id"]
                sample_kwargs = {
                    "prompt_embeds": prompt_embeds[entry["semantic_branch"]],
                    "uncond_prompt_embeds": uncond_embeds,
                    "image_vae_latents": None,
                    "multi_video_vae_latents": None,
                    "multi_image_vae_latents": None,
                    "width": factor_bank.VIDEO_WIDTH,
                    "height": factor_bank.VIDEO_HEIGHT,
                    "device": device,
                    **native.native_sampling_contract(
                        "t2v", steps=EXPECTED_STEPS, seed=entry["seed"]
                    ),
                }
                generated, noise_capture = native._sample_with_native_initial_noise_observer(
                    sample_fn=lambda oid=owner_id, branch=entry["semantic_branch"], kw=sample_kwargs: bridge.run_owner(
                        owner_id=oid, owner_branch=branch, sample_kwargs=kw
                    ),
                    wan_diffusion_module=wan_diffusion,
                    expected_shape=factor_bank.LATENT_SHAPE,
                    expected_device=device,
                    expected_seed=entry["seed"],
                )
                if noise_capture.raw_value_sha256 != bank["initial_noise_tensor_value_sha256"]:
                    raise OwnerPromptCrossQueryError("cross replay initial Gaussian differs from bank")
                clean_requested = Path(bank["clean_latent_path"])
                if (
                    not clean_requested.is_absolute()
                    or clean_requested.is_symlink()
                ):
                    raise OwnerPromptCrossQueryError(
                        "bank clean latent path must be absolute and non-symlink"
                    )
                clean_path = clean_requested.resolve(strict=True)
                if clean_path != clean_requested:
                    raise OwnerPromptCrossQueryError(
                        "bank clean latent path must be canonical"
                    )
                try:
                    clean_path.relative_to(bank_root)
                except ValueError as error:
                    raise OwnerPromptCrossQueryError("bank clean latent escaped output root") from error
                entry_root = (bank_root / entry["output_subdir"]).resolve(strict=True)
                if clean_path.parent != entry_root:
                    raise OwnerPromptCrossQueryError(
                        "bank clean latent escaped its registered entry directory"
                    )
                expected_clean = _load_clean_latent(
                    clean_path, expected_file_sha256=bank["clean_latent_sha256"]
                )
                actual_clean = generated.detach().to(device="cpu", dtype=torch.float32).contiguous()
                if tuple(actual_clean.shape) != factor_bank.LATENT_SHAPE or not torch.equal(
                    actual_clean, expected_clean
                ):
                    raise OwnerPromptCrossQueryError("cross replay final latent differs byte-exactly from bank owner")
                owner_outputs.append(
                    {
                        "owner_id": owner_id,
                        "owner_branch": entry["semantic_branch"],
                        "seed": entry["seed"],
                        "bank_native_receipt_digest": bank["native_receipt_digest"],
                        "bank_initial_noise_tensor_value_sha256": bank["initial_noise_tensor_value_sha256"],
                        "observed_initial_noise_tensor_value_sha256": noise_capture.raw_value_sha256,
                        "bank_clean_latent_path": str(clean_path),
                        "bank_clean_latent_file_sha256": bank["clean_latent_sha256"],
                        "bank_clean_latent_tensor": _tensor_identity(
                            expected_clean, label=f"bank_clean_{owner_id}"
                        ),
                        "cross_replay_final_tensor": _tensor_identity(
                            actual_clean, label=f"cross_replay_clean_{owner_id}"
                        ),
                        "initial_noise_byte_exact_to_bank": True,
                        "final_latent_byte_exact_to_bank": True,
                    }
                )
    finally:
        bridge.restore()

    freeze_after = native.source_audit.model_freeze_certificate(model)
    if freeze_after != freeze_before:
        raise OwnerPromptCrossQueryError("frozen model certificate changed")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise OwnerPromptCrossQueryError("frozen model retained trainable parameters")

    checkpoint_after_rows: list[Any] = [None]
    if distributed.rank == 0:
        try:
            checkpoint_after_rows[0] = {
                "ok": True,
                "identity": native.source_audit.validate_checkpoint_content(
                    checkpoint, checkpoint_manifest
                ),
            }
        except Exception as error:
            checkpoint_after_rows[0] = {"ok": False, "error": str(error)}
    dist.broadcast_object_list(checkpoint_after_rows, src=0)
    if (
        not isinstance(checkpoint_after_rows[0], Mapping)
        or checkpoint_after_rows[0].get("ok") is not True
        or checkpoint_after_rows[0].get("identity") != checkpoint_identity
    ):
        raise OwnerPromptCrossQueryError("checkpoint content changed during runtime")

    cartesian_query_order = expected_cartesian_query_order(bindings["owner_rows"])
    if observed_cartesian_query_order(bridge.owner_records) != cartesian_query_order:
        raise OwnerPromptCrossQueryError(
            "observed owner/state/prompt Cartesian query order differs"
        )

    local_evidence = {
        "rank": distributed.rank,
        "world_size": distributed.world_size,
        "owner_record_digest": object_sha256(bridge.owner_records),
        "artifact_digest": object_sha256(bridge.artifacts),
        "owner_output_digest": object_sha256(owner_outputs),
        "freeze_certificate_digest": object_sha256(freeze_after),
    }
    gathered: list[Any] = [None] * 4
    dist.all_gather_object(gathered, local_evidence)
    if sorted(row.get("rank") for row in gathered if isinstance(row, Mapping)) != [0, 1, 2, 3]:
        raise OwnerPromptCrossQueryError("rank evidence closure differs")
    for field in (
        "owner_record_digest",
        "artifact_digest",
        "owner_output_digest",
        "freeze_certificate_digest",
    ):
        if len({row.get(field) for row in gathered if isinstance(row, Mapping)}) != 1:
            raise OwnerPromptCrossQueryError(f"ranks disagree on {field}")

    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "method": METHOD_NAME,
        "runtime_source_revision": args.runtime_source_revision,
        "runtime_source_archive_sha256": args.runtime_source_archive_sha256,
        "factor_bank_renderer_source_revision": renderer_contract[
            "method_source_revision"
        ],
        "factor_bank_renderer_source_archive_sha256": renderer_contract[
            "method_source_archive_sha256"
        ],
        "runtime_source_provenance_separate_from_factor_bank_renderer_provenance": True,
        "runtime_and_factor_bank_source_provenance_equal": (
            args.runtime_source_revision
            == renderer_contract["method_source_revision"]
            and args.runtime_source_archive_sha256
            == renderer_contract["method_source_archive_sha256"]
        ),
        "launcher_source_sha256": args.launcher_source_sha256,
        "runtime_source_path": str(Path(__file__).resolve()),
        "runtime_source_sha256": file_sha256(Path(__file__).resolve()),
        "observer_source_sha256": file_sha256(Path(fitq_observer.__file__).resolve()),
        "sampler_contract_source_sha256": file_sha256(Path(sampler_contract.__file__).resolve()),
        "bernini_commit": bernini_revision,
        "veomni_commit": veomni_revision,
        "checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
        "checkpoint_content_identity": checkpoint_identity,
        "checkpoint_content_rehashed_after_runtime": True,
        "factor_manifest_path": str(manifest_path),
        "factor_manifest_file_sha256": manifest_file_sha,
        "factor_manifest_digest": bindings["manifest"]["manifest_digest"],
        "factor_bank_receipt_path": str(bank_receipt_path),
        "factor_bank_receipt_file_sha256": bank_receipt_file_sha,
        "factor_bank_receipt_digest": bindings["bank_receipt_digest"],
        "bank_id": bindings["manifest"]["bank_id"],
        "profile": "engineering_micro",
        "execution_group": args.execution_group,
        "proposal_cell_id": bindings["cell"]["proposal_cell_id"],
        "prompt_branch_order": list(PROMPT_BRANCH_ORDER),
        "owner_branch_order": list(OWNER_BRANCH_ORDER),
        "owner_count": len(OWNER_BRANCH_ORDER),
        "prompt_count": len(PROMPT_BRANCH_ORDER),
        "selected_schedule": list(selected_schedule_contract()),
        "schedule_sha256": EXPECTED_SCHEDULE_SHA256,
        "cartesian_query_order": list(cartesian_query_order),
        "cartesian_query_count": len(cartesian_query_order),
        "cartesian_query_closure_complete": True,
        "scheduler": schedule_evidence,
        "owner_records": bridge.owner_records,
        "owner_outputs": owner_outputs,
        "statistics_artifacts": bridge.artifacts,
        "forward_contract": {
            "official_forwards_per_owner": EXPECTED_OFFICIAL_FORWARDS_PER_OWNER,
            "extra_cross_query_forwards_per_owner": EXPECTED_EXTRA_FORWARDS_PER_OWNER,
            "total_forwards_per_owner": EXPECTED_TOTAL_FORWARDS_PER_OWNER,
            "original_scheduler_calls_per_owner": EXPECTED_STEPS,
            "hook_off_owner_field_retained_for_native_apg": True,
            "cross_query_outputs_discarded": True,
            "original_scheduler_arguments_replaced": False,
            "read_only_nn_module_hooks": True,
            "hooks_return_none": True,
            "collectives_inside_hooks": False,
        },
        "same_state_owner_by_prompt_cross_query_verified": True,
        "same_initial_noise_only_claim": False,
        "leave_one_owner_out_evaluated": False,
        "leave_one_owner_out_status": "runtime_tensor_bank_emitted_analysis_pending",
        "cross_mode_gate_evaluated": False,
        "causal_intervention_gate_evaluated": False,
        "identity_preservation_gate_evaluated": False,
        "fitq_go_authorized": False,
        "training_authorized": False,
        "scientific_claim_authorized": False,
        "optimizer_update": "null",
        "production_claim_forbidden": True,
        "freeze_certificate": freeze_after,
        "distributed": {
            "world_size": 4,
            "ulysses_size": 4,
            "rank_evidence": gathered,
        },
        "training": {
            "forward_only": True,
            "backward_performed": False,
            "optimizer_present": False,
            "adapter_present": False,
            "checkpoint_saved": False,
            "model_weights_written": False,
            "analysis_statistics_written": True,
        },
    }
    receipt["receipt_digest"] = object_sha256(receipt)
    dist.barrier()
    if distributed.rank == 0:
        _write_json_atomic(output_dir / "receipt.json", receipt)
        print(
            canonical_json_bytes(
                {
                    "receipt": str(output_dir / "receipt.json"),
                    "receipt_digest": receipt["receipt_digest"],
                    "owner_count": len(OWNER_BRANCH_ORDER),
                    "prompt_count": len(PROMPT_BRANCH_ORDER),
                    "selected_state_count": len(SELECTED_SCHEDULE_INDICES),
                    "optimizer_update": "null",
                    "training_authorized": False,
                }
            ).decode("ascii")
        )
    dist.barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
