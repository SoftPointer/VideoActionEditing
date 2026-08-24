#!/usr/bin/env python3
"""Read-only exact81 FITQ observation of Bernini's official forward path.

This executable deliberately reuses the already audited HAT/IAR runtime for
input packing, provenance checks, and calls to
``renderer.diff_dec.shared_step``.  It adds only PyTorch *observation* hooks at
official ``nn.Module`` boundaries.  The hooks return ``None``, never replace an
attention processor, never alter a field, and never run a collective.

One extra hook-off reference forward is made before the observed grid.  Its
output must be byte-exactly equal to the corresponding hook-on field.  The
receipt therefore reports the real count: 84 grid forwards, one hooked action
duplicate, and one hook-off reference for the pinned S=4, L=3, K=2, M=1
configuration.

The emitted ``*.fitq-stats.pt`` files contain FP32 phase-head means and second
moments only.  They are analysis evidence, not model weights or checkpoints.
Passing this engineering scan does not authorize Stage-1 training.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import internal_temporal_quotient_observer as fitq_observer  # noqa: E402


METHOD_NAME = "bernini-fitq-official-runtime-scan-v1"
RECEIPT_SCHEMA = "bernini-fitq-official-runtime-scan-receipt-v1"
STATISTICS_SCHEMA = "bernini-fitq-phase-head-mean-second-moment-v1"
EXPECTED_GRID_FORWARDS = 84
EXPECTED_DUPLICATE_FORWARDS = 1
EXPECTED_OBSERVED_FORWARDS = EXPECTED_GRID_FORWARDS + EXPECTED_DUPLICATE_FORWARDS
EXPECTED_REFERENCE_FORWARDS = 1
EXPECTED_TOTAL_FORWARDS = EXPECTED_OBSERVED_FORWARDS + EXPECTED_REFERENCE_FORWARDS
_SAFE_BASENAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class FITQOfficialRuntimeScanError(RuntimeError):
    """Raised before ambiguous FITQ evidence can be emitted."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise FITQOfficialRuntimeScanError(
            "FITQ evidence is not canonical finite ASCII JSON: {}".format(error)
        ) from error


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_absolute_new_directory(value: str) -> Path:
    if not isinstance(value, str) or not value:
        raise FITQOfficialRuntimeScanError(
            "output statistics directory must be non-empty text"
        )
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise FITQOfficialRuntimeScanError(
            "output statistics directory must be absolute"
        )
    if _SAFE_BASENAME_RE.fullmatch(path.name) is None:
        raise FITQOfficialRuntimeScanError(
            "output statistics directory basename is unsafe"
        )
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise FITQOfficialRuntimeScanError(
            "output statistics parent must be an existing non-symlink directory"
        )
    if path.exists() or path.is_symlink():
        raise FITQOfficialRuntimeScanError(
            "refusing to reuse an output statistics directory"
        )
    return path


def _branch_mode(branch: str) -> str:
    if branch.startswith("frozen_t2v_"):
        return "t2v"
    if branch.startswith("frozen_identity_"):
        return "mv2v"
    raise FITQOfficialRuntimeScanError(
        "unrecognized official branch for FITQ context: {!r}".format(branch)
    )


def build_explicit_branch_plan(branch_names: Sequence[str]) -> tuple[dict[str, str], ...]:
    """Bind every official field call to an explicit FITQ mode and branch."""

    names = tuple(branch_names)
    if len(names) < 7 or len(set(names)) != len(names):
        raise FITQOfficialRuntimeScanError(
            "FITQ requires distinct K>=2/M>=1 official branch names"
        )
    plan = tuple({"mode": _branch_mode(name), "branch": name} for name in names)
    if plan[0] != {"mode": "t2v", "branch": "frozen_t2v_action"}:
        raise FITQOfficialRuntimeScanError("first FITQ branch must be T2V action")
    if sum(item["mode"] == "t2v" for item in plan) < 3:
        raise FITQOfficialRuntimeScanError("FITQ branch plan lacks hard negatives")
    if sum(item["mode"] == "mv2v" for item in plan) < 4:
        raise FITQOfficialRuntimeScanError("FITQ branch plan lacks MV2V controls")
    return plan


def _fingerprint_dict(value: Any) -> dict[str, Any]:
    return {
        "shape": [int(item) for item in value.shape],
        "dtype": str(value.dtype),
        "nbytes": int(value.nbytes),
        "sha256": str(value.sha256),
    }


def _fingerprint_pair_digest(left: Any, right: Any) -> str:
    sites = fitq_observer.EXACT_PROBE_SITES
    payload = {
        "left": {site: _fingerprint_dict(left.exact_fingerprints[site]) for site in sites},
        "right": {site: _fingerprint_dict(right.exact_fingerprints[site]) for site in sites},
    }
    return _object_sha256(payload)


_RAW_TENSOR_IDENTITY_KEYS = (
    "shape",
    "dtype",
    "numel",
    "byte_count",
    "raw_storage_sha256",
)


def _raw_tensor_identities_equal(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    """Compare exact value bytes and dtype/shape, ignoring diagnostic labels.

    ``torch.equal`` treats signed zero as equal and historically accepted some
    cross-dtype comparisons.  FITQ hook parity is a byte-level noninterference
    claim, so the runtime uses the official raw-storage identities instead.
    """

    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        raise FITQOfficialRuntimeScanError("tensor identity must be an object")
    for key in _RAW_TENSOR_IDENTITY_KEYS:
        if key not in left or key not in right:
            raise FITQOfficialRuntimeScanError(
                "tensor identity lacks exact raw-value fields"
            )
    return all(left[key] == right[key] for key in _RAW_TENSOR_IDENTITY_KEYS)


def _global_context_dict(context: Any) -> dict[str, Any]:
    """Drop the rank-local coordinate after an exact SP4 SUM reduction."""

    value = dict(context.as_dict())
    value.pop("sp_rank", None)
    value["rank_scope"] = "SP4_all_reduced"
    return value


def _truthful_field_evidence(
    local_evidence: Mapping[str, Any],
    *,
    underlying_schema_validation_shadow_digest: str,
) -> dict[str, Any]:
    """Replace legacy no-callback claims with the actual hook provenance."""

    evidence = copy.deepcopy(dict(local_evidence))
    observed = evidence.get("forwards_per_rank")
    if type(observed) is not int or observed <= 0:
        raise FITQOfficialRuntimeScanError("underlying observed forward count is invalid")
    cells = evidence.get("cell_records")
    if not isinstance(cells, list) or not cells:
        raise FITQOfficialRuntimeScanError("underlying official cell grid is absent")

    observed_cell_digests = []
    for cell in cells:
        if not isinstance(cell, dict):
            raise FITQOfficialRuntimeScanError("official FITQ cell must be an object")
        old_digest = cell.pop("cell_digest", None)
        if not isinstance(old_digest, str) or len(old_digest) != 64:
            raise FITQOfficialRuntimeScanError("underlying IAR cell digest is invalid")
        cell["underlying_iar_cell_digest"] = old_digest
        cell["no_forward_callback"] = False
        cell["read_only_forward_hooks_present"] = True
        cell["forward_outputs_observed_not_modified"] = True
        cell["custom_forward_core_present"] = False
        cell["custom_analysis_core_present"] = True
        cell["cell_digest"] = _object_sha256(cell)
        observed_cell_digests.append(cell["cell_digest"])

    core = evidence.get("iar_core")
    if not isinstance(core, dict):
        raise FITQOfficialRuntimeScanError("underlying IAR core evidence is absent")
    closure = core.pop("full_cell_grid_closure", None)
    closure_digest = core.pop("full_cell_grid_digest", None)
    core["underlying_iar_full_cell_grid_closure"] = closure
    core["underlying_iar_full_cell_grid_digest"] = closure_digest
    core["fitq_observed_cell_digests"] = observed_cell_digests
    core["fitq_observed_cell_grid_digest"] = _object_sha256(observed_cell_digests)

    evidence["underlying_method"] = evidence.get("method")
    evidence["method"] = METHOD_NAME
    evidence["field_grid_forwards_per_rank"] = observed
    evidence["hooked_action_duplicate_forwards_per_rank"] = EXPECTED_DUPLICATE_FORWARDS
    evidence["hook_off_reference_forwards_per_rank"] = EXPECTED_REFERENCE_FORWARDS
    evidence["forwards_per_rank"] = (
        observed + EXPECTED_DUPLICATE_FORWARDS + EXPECTED_REFERENCE_FORWARDS
    )
    evidence["forward_callback_present"] = True
    evidence["forward_callback_kind"] = "read_only_nn_module_hooks"
    evidence["read_only_forward_hooks_present"] = True
    evidence["hooks_return_none"] = True
    evidence["collectives_inside_hooks"] = False
    evidence["custom_core_present"] = False
    evidence["custom_forward_core_present"] = False
    evidence["custom_analysis_core_present"] = True
    evidence["underlying_iar_schema_validation_shadow_digest"] = (
        underlying_schema_validation_shadow_digest
    )
    evidence["underlying_iar_shadow_is_hypothetical_not_runtime_provenance"] = True
    evidence["underlying_iar_shadow_emitted_as_receipt"] = False
    return evidence


class FITQOfficialRuntimeAdapter:
    """Narrow adapter around IAR's direct official calls and receipt builder."""

    def __init__(
        self,
        *,
        iar_module: Any,
        statistics_dir: Path,
        original_direct_prediction: Any,
        original_run_cell: Any,
        original_assemble_receipt: Any,
    ) -> None:
        self.iar = iar_module
        self.statistics_dir = statistics_dir
        self.original_direct_prediction = original_direct_prediction
        self.original_run_cell = original_run_cell
        self.original_assemble_receipt = original_assemble_receipt
        self.observer: Optional[Any] = None
        self._statistics_dir_created = False
        self._current_plan: list[dict[str, str]] = []
        self._current_lambda: Optional[float] = None
        self._current_sigma: Optional[float] = None
        self._current_local_results: dict[str, Any] = {}
        self._observed_forward_count = 0
        self._reference_forward_count = 0
        self._artifact_records: list[dict[str, Any]] = []
        self._context_records: list[dict[str, Any]] = []
        self._cell_parity_records: list[dict[str, Any]] = []
        self._hook_output_parity: Optional[dict[str, Any]] = None
        self._action_duplicate_floor: Optional[dict[str, Any]] = None
        self._duplicate_local_fingerprint_pair_digest: Optional[str] = None

    @property
    def torch(self) -> Any:
        try:
            return importlib.import_module("torch")
        except ImportError as error:  # pragma: no cover - AUH runtime dependency
            raise FITQOfficialRuntimeScanError("FITQ runtime requires torch") from error

    def _distributed_rank(self) -> int:
        dist = self.torch.distributed
        if not dist.is_available() or not dist.is_initialized():
            raise FITQOfficialRuntimeScanError(
                "FITQ adapter requires initialized WORLD4 distributed state"
            )
        world = int(dist.get_world_size())
        rank = int(dist.get_rank())
        if world != fitq_observer.EXPECTED_SP_WORLD or not 0 <= rank < world:
            raise FITQOfficialRuntimeScanError(
                "FITQ adapter requires exact WORLD4/Ulysses4"
            )
        return rank

    def _ensure_statistics_directory(self) -> None:
        if self._statistics_dir_created:
            return
        dist = self.torch.distributed
        rank = self._distributed_rank()
        if rank == 0:
            self.statistics_dir.mkdir(mode=0o755, parents=False, exist_ok=False)
        dist.barrier()
        if not self.statistics_dir.is_dir() or self.statistics_dir.is_symlink():
            raise FITQOfficialRuntimeScanError(
                "rank zero did not create a plain FITQ statistics directory"
            )
        self._statistics_dir_created = True

    def _install_observer(self, renderer: Any) -> None:
        if self.observer is not None:
            return
        self.observer = fitq_observer.install_internal_temporal_quotient_observer(
            renderer, capture_exact_block0=True
        )
        if not self.observer.installed or self.observer.trainable_parameters != ():
            raise FITQOfficialRuntimeScanError("FITQ observer installation is ambiguous")

    def _write_statistics_artifact(self, result: Any, ordinal: int) -> dict[str, Any]:
        result.require_complete()
        if not result.globally_reduced:
            raise FITQOfficialRuntimeScanError(
                "refusing to serialize rank-local FITQ statistics"
            )
        order = fitq_observer.expected_site_order()
        expected_sum_shape = (
            len(order),
            fitq_observer.EXPECTED_PHASE_COUNT,
            fitq_observer.EXPECTED_HEAD_COUNT,
            fitq_observer.EXPECTED_HEAD_DIM,
        )
        torch = self.torch
        sums = torch.stack([result.sites[site].sum for site in order]).detach()
        sumsqs = torch.stack([result.sites[site].sumsq for site in order]).detach()
        count = result.sites[order[0]].count.detach()
        if tuple(sums.shape) != expected_sum_shape or tuple(sumsqs.shape) != expected_sum_shape:
            raise FITQOfficialRuntimeScanError("FITQ statistic tensor geometry differs")
        if any(value.dtype != torch.float32 for value in (sums, sumsqs, count)):
            raise FITQOfficialRuntimeScanError("FITQ artifacts must remain FP32")
        if any(value.requires_grad for value in (sums, sumsqs, count)):
            raise FITQOfficialRuntimeScanError("FITQ artifacts retained autograd")
        first_count = count
        for site in order[1:]:
            if not torch.equal(result.sites[site].count, first_count):
                raise FITQOfficialRuntimeScanError(
                    "FITQ global phase count differs across hook sites"
                )
        denominator = count[:, :, None].clamp_min(1.0)[None]
        means = (sums / denominator).detach()
        second_moments = (sumsqs / denominator).detach()
        if not bool(torch.isfinite(means).all().item()) or not bool(
            torch.isfinite(second_moments).all().item()
        ):
            raise FITQOfficialRuntimeScanError(
                "FITQ phase-head mean/second moment is non-finite"
            )
        feature_shape = (
            len(order),
            1,
            fitq_observer.EXPECTED_PHASE_COUNT,
            1,
            fitq_observer.EXPECTED_HIDDEN_DIM,
        )
        feature_means = means.reshape(feature_shape)
        feature_second_moments = second_moments.reshape(feature_shape)

        context = _global_context_dict(result.context)
        branch_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", result.context.branch).strip("-")
        filename = "{:04d}-{}-sigma-{:.8f}-lambda-{:.8f}.fitq-stats.pt".format(
            ordinal,
            branch_slug,
            result.context.sigma,
            result.context.lambda_value,
        )
        output_path = self.statistics_dir / filename
        rank = self._distributed_rank()
        metadata: list[Any] = [None]
        if rank == 0:
            if output_path.exists() or output_path.is_symlink():
                raise FITQOfficialRuntimeScanError(
                    "refusing to overwrite FITQ statistics artifact"
                )
            temporary = output_path.with_name(".{}.tmp-{}".format(filename, os.getpid()))
            if temporary.exists() or temporary.is_symlink():
                raise FITQOfficialRuntimeScanError("FITQ temporary artifact already exists")
            payload = {
                "schema_version": STATISTICS_SCHEMA,
                "site_order": list(order),
                "context": context,
                "phase_feature_mean_fp32": feature_means.to(device="cpu"),
                "phase_feature_second_moment_fp32": feature_second_moments.to(device="cpu"),
                "global_count_fp32_shared_by_all_sites": count.to(device="cpu"),
            }
            try:
                torch.save(payload, str(temporary))
                os.replace(str(temporary), str(output_path))
            finally:
                if temporary.exists():
                    temporary.unlink()
            metadata[0] = {
                "ordinal": ordinal,
                "path": str(output_path),
                "sha256": _file_sha256(output_path),
                "size_bytes": int(output_path.stat().st_size),
                "schema_version": STATISTICS_SCHEMA,
                "context": context,
                "site_count": len(order),
                "mean_shape": list(feature_shape),
                "second_moment_shape": list(feature_shape),
                "count_shape": list(count.shape),
                "dtype": "torch.float32",
                "contains_model_weights": False,
                "is_checkpoint": False,
            }
        self.torch.distributed.broadcast_object_list(metadata, src=0)
        record = metadata[0]
        if not isinstance(record, Mapping):
            raise FITQOfficialRuntimeScanError("statistics artifact metadata broadcast failed")
        return dict(record)

    def _duplicate_numerical_floor(self, left: Any, right: Any) -> dict[str, Any]:
        """Measure, but do not hide, numerical variation of a duplicate query."""

        torch = self.torch
        order = fitq_observer.expected_site_order()
        max_mean = 0.0
        max_second = 0.0
        mean_square_total = 0.0
        second_square_total = 0.0
        element_count = 0
        counts_exact = True
        all_statistics_exact = True
        for site in order:
            left_stats = left.sites[site]
            right_stats = right.sites[site]
            counts_exact = counts_exact and _raw_tensor_identities_equal(
                self.iar._tensor_identity(
                    left_stats.count, label="fitq_duplicate_count_left"
                ),
                self.iar._tensor_identity(
                    right_stats.count, label="fitq_duplicate_count_right"
                ),
            )
            denominator = left_stats.count[:, :, None].clamp_min(1.0)
            left_mean = left_stats.sum / denominator
            right_mean = right_stats.sum / denominator
            left_second = left_stats.sumsq / denominator
            right_second = right_stats.sumsq / denominator
            mean_delta = (left_mean - right_mean).float()
            second_delta = (left_second - right_second).float()
            max_mean = max(max_mean, float(mean_delta.abs().amax().item()))
            max_second = max(max_second, float(second_delta.abs().amax().item()))
            mean_square_total += float(mean_delta.square().sum().item())
            second_square_total += float(second_delta.square().sum().item())
            element_count += int(mean_delta.numel())
            all_statistics_exact = all_statistics_exact and (
                _raw_tensor_identities_equal(
                    self.iar._tensor_identity(
                        left_stats.sum, label="fitq_duplicate_sum_left"
                    ),
                    self.iar._tensor_identity(
                        right_stats.sum, label="fitq_duplicate_sum_right"
                    ),
                )
                and _raw_tensor_identities_equal(
                    self.iar._tensor_identity(
                        left_stats.sumsq, label="fitq_duplicate_sumsq_left"
                    ),
                    self.iar._tensor_identity(
                        right_stats.sumsq, label="fitq_duplicate_sumsq_right"
                    ),
                )
            )
        if not counts_exact or element_count <= 0:
            raise FITQOfficialRuntimeScanError("duplicate FITQ statistic geometry differs")
        return {
            "phase_head_counts_exact": True,
            "all_phase_head_statistics_byte_exact": all_statistics_exact,
            "byte_exact_comparison": "shape_dtype_and_raw_storage_sha256",
            "phase_head_mean_max_abs_delta": max_mean,
            "phase_head_mean_rms_delta": math.sqrt(mean_square_total / element_count),
            "phase_head_second_moment_max_abs_delta": max_second,
            "phase_head_second_moment_rms_delta": math.sqrt(
                second_square_total / element_count
            ),
            "null_floor_is_measurement_not_success_gate": True,
        }

    def direct_prediction(self, *args: Any, **kwargs: Any) -> Any:
        if not self._current_plan or self._current_lambda is None or self._current_sigma is None:
            raise FITQOfficialRuntimeScanError(
                "official prediction escaped an explicit branch/sigma/lambda context"
            )
        branch_spec = self._current_plan.pop(0)
        renderer = args[0] if args else kwargs.get("renderer")
        if renderer is None:
            raise FITQOfficialRuntimeScanError("official prediction lacks renderer")

        hook_off_reference = None
        if self._reference_forward_count == 0:
            if self.observer is not None:
                raise FITQOfficialRuntimeScanError(
                    "hook-off reference must precede observer installation"
                )
            hook_off_reference = self.original_direct_prediction(*args, **kwargs)
            self._reference_forward_count += 1

        self._install_observer(renderer)
        context = fitq_observer.FITQObserverContext(
            mode=branch_spec["mode"],
            branch=branch_spec["branch"],
            sigma=self._current_sigma,
            lambda_value=self._current_lambda,
            sp_rank=self._distributed_rank(),
        )
        assert self.observer is not None
        with self.observer.capture(context) as session:
            field = self.original_direct_prediction(*args, **kwargs)
        local_result = session.result
        if local_result is None:
            raise FITQOfficialRuntimeScanError("FITQ observer returned no local result")
        global_result = fitq_observer.all_reduce_local_sufficient_statistics(
            local_result
        )

        if hook_off_reference is not None:
            hook_off_identity = self.iar._tensor_identity(
                hook_off_reference, label="fitq_hook_off_reference"
            )
            hook_on_identity = self.iar._tensor_identity(
                field, label="fitq_hook_on_observed"
            )
            exact = _raw_tensor_identities_equal(
                hook_off_identity, hook_on_identity
            )
            if not exact:
                raise FITQOfficialRuntimeScanError(
                    "hook-on official field differs from hook-off reference"
                )
            self._hook_output_parity = {
                "mode": context.mode,
                "branch": context.branch,
                "sigma": context.sigma,
                "lambda": context.lambda_value,
                "byte_exact_equal": True,
                "comparison": "shape_dtype_and_raw_storage_sha256",
                "hook_off_field": hook_off_identity,
                "hook_on_field": hook_on_identity,
            }

        self._ensure_statistics_directory()
        ordinal = self._observed_forward_count
        artifact = self._write_statistics_artifact(global_result, ordinal)
        self._artifact_records.append(artifact)
        self._context_records.append(_global_context_dict(context))
        self._current_local_results[context.branch] = local_result
        self._observed_forward_count += 1

        if self._action_duplicate_floor is None:
            duplicate_context = fitq_observer.FITQObserverContext(
                mode=context.mode,
                branch="frozen_t2v_action_duplicate",
                sigma=context.sigma,
                lambda_value=context.lambda_value,
                sp_rank=context.sp_rank,
            )
            with self.observer.capture(duplicate_context) as duplicate_session:
                duplicate_field = self.original_direct_prediction(*args, **kwargs)
            duplicate_local = duplicate_session.result
            if duplicate_local is None:
                raise FITQOfficialRuntimeScanError(
                    "FITQ duplicate action observer returned no result"
                )
            duplicate_global = fitq_observer.all_reduce_local_sufficient_statistics(
                duplicate_local
            )
            duplicate_artifact = self._write_statistics_artifact(
                duplicate_global, self._observed_forward_count
            )
            self._artifact_records.append(duplicate_artifact)
            self._context_records.append(_global_context_dict(duplicate_context))
            self._observed_forward_count += 1
            duplicate_block0 = fitq_observer.exact_block0_parity(
                local_result, duplicate_local
            )
            if duplicate_block0.get("all") is not True:
                raise FITQOfficialRuntimeScanError(
                    "duplicate action block0 same-state parity failed"
                )
            floor = self._duplicate_numerical_floor(global_result, duplicate_global)
            duplicate_left_identity = self.iar._tensor_identity(
                field, label="fitq_action_duplicate_left"
            )
            duplicate_right_identity = self.iar._tensor_identity(
                duplicate_field, label="fitq_action_duplicate_right"
            )
            floor.update(
                {
                    "mode": context.mode,
                    "branch": context.branch,
                    "duplicate_branch": duplicate_context.branch,
                    "sigma": context.sigma,
                    "lambda": context.lambda_value,
                    "final_field_byte_exact": _raw_tensor_identities_equal(
                        duplicate_left_identity, duplicate_right_identity
                    ),
                    "final_field_comparison": (
                        "shape_dtype_and_raw_storage_sha256"
                    ),
                    "final_field_left": duplicate_left_identity,
                    "final_field_right": duplicate_right_identity,
                    "block0_same_state_exact": dict(duplicate_block0),
                }
            )
            self._action_duplicate_floor = floor
            self._duplicate_local_fingerprint_pair_digest = _fingerprint_pair_digest(
                local_result, duplicate_local
            )
        return field

    def _same_state_pairs(self, branches: Sequence[str]) -> tuple[tuple[str, str], ...]:
        names = tuple(branches)
        action = "frozen_t2v_action"
        negatives = tuple(name for name in names if name.startswith("frozen_t2v_hard_negative["))
        pairs: list[tuple[str, str]] = [(action, name) for name in negatives]
        pairs.append(("frozen_identity_noop_correct", "frozen_identity_action_correct"))
        wrong_noop = sorted(
            name for name in names if name.startswith("frozen_identity_noop_wrong_source[")
        )
        wrong_action = sorted(
            name for name in names if name.startswith("frozen_identity_action_wrong_source[")
        )
        if len(wrong_noop) != len(wrong_action) or not wrong_noop:
            raise FITQOfficialRuntimeScanError("MV2V wrong-source parity pairs differ")
        pairs.extend(zip(wrong_noop, wrong_action))
        return tuple(pairs)

    def run_cell(self, *args: Any, **kwargs: Any) -> Any:
        if self._current_plan:
            raise FITQOfficialRuntimeScanError("nested or incomplete official FITQ cell")
        bundles = kwargs.get("bundles")
        bridge_fraction = kwargs.get("bridge_fraction")
        if not isinstance(bundles, Sequence) or not bundles:
            raise FITQOfficialRuntimeScanError("FITQ cell lacks official query bundles")
        if isinstance(bridge_fraction, bool) or not isinstance(bridge_fraction, (int, float)):
            raise FITQOfficialRuntimeScanError("FITQ cell lambda is invalid")
        sigma = float(bundles[0].point.sigma)
        lambda_value = float(bridge_fraction)
        if not math.isfinite(sigma) or not math.isfinite(lambda_value):
            raise FITQOfficialRuntimeScanError("FITQ cell coordinate is non-finite")
        branch_names = self.iar._branch_names(
            len(kwargs.get("hard_negative_t2v_conditions", ())), len(bundles)
        )
        self._current_plan = list(build_explicit_branch_plan(branch_names))
        self._current_lambda = lambda_value
        self._current_sigma = sigma
        self._current_local_results = {}
        try:
            result = self.original_run_cell(*args, **kwargs)
            if self._current_plan:
                raise FITQOfficialRuntimeScanError(
                    "official cell did not consume every explicit FITQ branch"
                )
            if set(self._current_local_results) != set(branch_names):
                raise FITQOfficialRuntimeScanError(
                    "FITQ local results do not cover the official branch set"
                )
            parity_rows = []
            for left_name, right_name in self._same_state_pairs(branch_names):
                left = self._current_local_results[left_name]
                right = self._current_local_results[right_name]
                parity = fitq_observer.exact_block0_parity(left, right)
                if parity.get("all") is not True:
                    raise FITQOfficialRuntimeScanError(
                        "same-state block0 parity failed for {} vs {}".format(
                            left_name, right_name
                        )
                    )
                parity_rows.append(
                    {
                        "left_branch": left_name,
                        "right_branch": right_name,
                        "mode": left.context.mode,
                        "sigma": sigma,
                        "lambda": lambda_value,
                        "exact": dict(parity),
                        "fingerprint_pair_digest": _fingerprint_pair_digest(left, right),
                    }
                )
            self._cell_parity_records.append(
                {
                    "sigma": sigma,
                    "lambda": lambda_value,
                    "same_state_pairs": parity_rows,
                    "all_pairs_byte_exact": True,
                }
            )
            return result
        finally:
            self._current_plan = []
            self._current_lambda = None
            self._current_sigma = None
            self._current_local_results = {}

    def _fitq_common_evidence(self, local_evidence: Mapping[str, Any]) -> dict[str, Any]:
        expected = int(local_evidence.get("forwards_per_rank", -1))
        if expected != EXPECTED_GRID_FORWARDS:
            raise FITQOfficialRuntimeScanError(
                "pinned FITQ scan requires exactly 84 observed grid forwards"
            )
        if self._observed_forward_count != EXPECTED_OBSERVED_FORWARDS:
            raise FITQOfficialRuntimeScanError(
                "hooked FITQ forward count differs from grid plus action duplicate"
            )
        if self._reference_forward_count != EXPECTED_REFERENCE_FORWARDS:
            raise FITQOfficialRuntimeScanError("FITQ hook-off reference count differs")
        if self._hook_output_parity is None:
            raise FITQOfficialRuntimeScanError("FITQ hook-on/off parity is absent")
        if (
            len(self._artifact_records) != EXPECTED_OBSERVED_FORWARDS
            or len(self._context_records) != EXPECTED_OBSERVED_FORWARDS
        ):
            raise FITQOfficialRuntimeScanError("FITQ context/artifact closure differs")
        if self._action_duplicate_floor is None:
            raise FITQOfficialRuntimeScanError("FITQ duplicate-action null floor is absent")
        if self._duplicate_local_fingerprint_pair_digest is None:
            raise FITQOfficialRuntimeScanError(
                "FITQ duplicate-action local fingerprint evidence is absent"
            )
        cell_count = len(local_evidence.get("cell_records", ()))
        if len(self._cell_parity_records) != cell_count:
            raise FITQOfficialRuntimeScanError("FITQ block0 parity cell count differs")
        return {
            "observer_source_path": str(Path(fitq_observer.__file__).resolve()),
            "observer_source_sha256": _file_sha256(Path(fitq_observer.__file__).resolve()),
            "runtime_source_path": str(Path(__file__).resolve()),
            "runtime_source_sha256": _file_sha256(Path(__file__).resolve()),
            "hook_sites": list(fitq_observer.expected_site_order()),
            "hook_site_count": len(fitq_observer.expected_site_order()),
            "read_only_forward_hooks_present": True,
            "hooks_return_none": True,
            "collectives_inside_hooks": False,
            "all_reduce_after_each_official_forward": True,
            "custom_forward_core_present": False,
            "custom_analysis_core_present": True,
            "official_field_output_modified": False,
            "hook_on_off_field_parity": dict(self._hook_output_parity),
            "action_duplicate_numerical_floor": dict(self._action_duplicate_floor),
            "field_grid_forwards_per_rank": expected,
            "hooked_action_duplicate_forwards_per_rank": EXPECTED_DUPLICATE_FORWARDS,
            "observed_hooked_forwards_per_rank": self._observed_forward_count,
            "hook_off_reference_forwards_per_rank": self._reference_forward_count,
            "total_official_forwards_per_rank": (
                self._observed_forward_count + self._reference_forward_count
            ),
            "contexts": list(self._context_records),
            "context_count": len(self._context_records),
            "statistics_artifacts": list(self._artifact_records),
            "statistics_artifact_count": len(self._artifact_records),
            "statistics_directory": str(self.statistics_dir),
            "statistics_are_analysis_evidence_not_checkpoint": True,
            "statistics_contain_model_weights": False,
            "analysis_statistics": "phase_head_mean_second_moment",
            "mean_reshape_contract": "per_site_[1,21,1,1536]_from_heads_12x128",
            "tokenwise_localization_available": False,
            "second_moments_are_not_full_A1_evidence": True,
            "same_state_block0_parity_is_rank_local": True,
            "proposal_bank_size": 1,
            "proposal_bank_status": "insufficient_bank",
            "decision_scope": "engineering_N0_like_diagnostic_only",
            "scientific_fitq_outcome": "not_evaluated_single_proposal",
            "cross_mode_transport_gate_evaluated": False,
            "causal_intervention_gate_evaluated": False,
            "fitq_go_authorized": False,
            "no_training_authorization_from_observation": True,
        }

    def assemble_receipt(
        self,
        local_evidence: Mapping[str, Any],
        rank_records: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        # First retain all fail-closed checks from the official HAT/IAR runtime.
        # This call retains the old IAR input/grid/schema checks.  Its payload
        # says hooks are absent, so it is strictly a hypothetical validation
        # shadow and is never emitted or presented as runtime provenance.
        schema_validation_shadow = self.original_assemble_receipt(
            local_evidence, rank_records
        )
        common = self._fitq_common_evidence(local_evidence)
        common_digest = _object_sha256(common)
        field_evidence = _truthful_field_evidence(
            local_evidence,
            underlying_schema_validation_shadow_digest=(
                schema_validation_shadow["receipt_digest"]
            ),
        )
        field_evidence_digest = _object_sha256(field_evidence)
        actual_runtime_evidence_digest = _object_sha256(
            {
                "official_field_evidence": field_evidence,
                "fitq_observation": common,
            }
        )
        rank = self._distributed_rank()
        rank_fitq = {
            "rank": rank,
            "world_size": int(self.torch.distributed.get_world_size()),
            "common_evidence_digest": common_digest,
            "truthful_field_evidence_digest": field_evidence_digest,
            "actual_runtime_evidence_digest": actual_runtime_evidence_digest,
            "same_state_block0_parity": list(self._cell_parity_records),
            "same_state_block0_parity_digest": _object_sha256(
                self._cell_parity_records
            ),
            "action_duplicate_block0_fingerprint_pair_digest": (
                self._duplicate_local_fingerprint_pair_digest
            ),
        }
        gathered: list[Any] = [None] * fitq_observer.EXPECTED_SP_WORLD
        self.torch.distributed.all_gather_object(gathered, rank_fitq)
        if any(not isinstance(item, Mapping) for item in gathered):
            raise FITQOfficialRuntimeScanError("FITQ rank evidence gather is incomplete")
        gathered_rows = [dict(item) for item in gathered]
        if sorted(item.get("rank") for item in gathered_rows) != [0, 1, 2, 3]:
            raise FITQOfficialRuntimeScanError("FITQ rank set is not exact WORLD4")
        if any(item.get("world_size") != 4 for item in gathered_rows):
            raise FITQOfficialRuntimeScanError("FITQ gathered rank has wrong world size")
        if any(item.get("common_evidence_digest") != common_digest for item in gathered_rows):
            raise FITQOfficialRuntimeScanError(
                "FITQ ranks disagree on global statistics artifact evidence"
            )
        if any(
            item.get("truthful_field_evidence_digest") != field_evidence_digest
            or item.get("actual_runtime_evidence_digest")
            != actual_runtime_evidence_digest
            for item in gathered_rows
        ):
            raise FITQOfficialRuntimeScanError(
                "FITQ ranks disagree on truthful hooked-runtime evidence"
            )
        for item in gathered_rows:
            cells = item.get("same_state_block0_parity")
            if not isinstance(cells, list) or len(cells) != len(self._cell_parity_records):
                raise FITQOfficialRuntimeScanError("FITQ rank parity coverage differs")
            if any(cell.get("all_pairs_byte_exact") is not True for cell in cells):
                raise FITQOfficialRuntimeScanError("FITQ same-state parity is not exact")

        if self.observer is None or not self.observer.installed or self.observer.active:
            raise FITQOfficialRuntimeScanError("FITQ observer terminal state is ambiguous")
        self.observer.remove()
        if self.observer.installed:
            raise FITQOfficialRuntimeScanError("FITQ hooks remained installed")

        receipt: dict[str, Any] = {
            "schema_version": RECEIPT_SCHEMA,
            "method": METHOD_NAME,
            "official_field_evidence": field_evidence,
            "fitq_observation": common,
            "underlying_iar_schema_validation_shadow": {
                "digest": schema_validation_shadow["receipt_digest"],
                "hypothetical_no_hook_schema_check_only": True,
                "is_runtime_provenance": False,
                "emitted_as_receipt": False,
            },
            "distributed": {
                "world_size": 4,
                "ulysses_size": 4,
                "topology": "one_independent_WORLD4/Ulysses4_group",
                "statistics_all_reduce_outside_hooks": True,
                "truthful_field_evidence_digest": field_evidence_digest,
                "actual_runtime_evidence_digest": actual_runtime_evidence_digest,
                "rank_fitq_records": sorted(gathered_rows, key=lambda item: item["rank"]),
            },
            "exact81": True,
            "engineering_scan_only": True,
            "read_only_forward_hooks_present": True,
            "forward_callback_present": True,
            "forward_callback_kind": "read_only_nn_module_hooks",
            "custom_forward_core_present": False,
            "custom_analysis_core_present": True,
            "official_field_output_modified": False,
            "training_authorized": False,
            "fitq_stage1_authorized": False,
            "fitq_stage0_scientific_outcome": "not_evaluated_single_proposal",
            "optimizer_update": "null",
            "scientific_claim_authorized": False,
            "production_claim_forbidden": True,
            "training": {
                "forward_only": True,
                "backward_performed": False,
                "optimizer_present": False,
                "checkpoint_saved": False,
                "adapter_present": False,
                "model_weights_written": False,
                "analysis_statistics_written": True,
            },
        }
        receipt["receipt_digest"] = _object_sha256(receipt)
        return receipt

    def close(self) -> None:
        if self.observer is not None and self.observer.active:
            self.observer.abort_forward()
        if self.observer is not None and self.observer.installed:
            self.observer.remove()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        add_help=False,
        description="Add read-only FITQ hooks to the exact81 official IAR runtime",
    )
    parser.add_argument("--output-statistics-dir", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    fitq_args, iar_argv = build_parser().parse_known_args(raw_argv)
    statistics_dir = _require_absolute_new_directory(fitq_args.output_statistics_dir)
    try:
        iar = importlib.import_module("infer_iar_official_runtime_smoke")
    except ImportError as error:  # pragma: no cover - AUH dependency closure
        raise FITQOfficialRuntimeScanError(
            "official FITQ runtime could not import the pinned IAR closure"
        ) from error

    original_direct = iar._direct_target_prediction
    original_cell = iar._run_official_cell
    original_assemble = iar.assemble_sp4_receipt
    adapter = FITQOfficialRuntimeAdapter(
        iar_module=iar,
        statistics_dir=statistics_dir,
        original_direct_prediction=original_direct,
        original_run_cell=original_cell,
        original_assemble_receipt=original_assemble,
    )
    iar._direct_target_prediction = adapter.direct_prediction
    iar._run_official_cell = adapter.run_cell
    iar.assemble_sp4_receipt = adapter.assemble_receipt
    try:
        return int(iar.main(iar_argv))
    finally:
        iar._direct_target_prediction = original_direct
        iar._run_official_cell = original_cell
        iar.assemble_sp4_receipt = original_assemble
        adapter.close()


if __name__ == "__main__":
    raise SystemExit(main())
