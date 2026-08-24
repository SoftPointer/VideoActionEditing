#!/usr/bin/env python3
"""Diagnose Q-MOSAIC zero-route parity on one real WORLD4/SP4 group.

This program is deliberately separate from the production direction runner.
It authenticates the same frozen owner/editor/checkpoint coordinate, explicitly
reproduces the historical no-grad/adapter-off/detached measurement ``M``, and
then executes the graph-mode
ABBA sequence ``O0,N0,N1,O1`` for both prompt roles.  A fifth, excluded call
``P`` runs the existing per-projection zero-route proof.  Only create-only JSON
receipts are written: there is no decode, VJP, optimizer, parameter update, or
semantic/action-editing success authority.

The diagnostic distinguishes four outcomes and never turns any of them into
training authority:

* ``A_WRAPPER_ROUTE`` -- the projection-local exact-zero/base-byte proof fails;
* ``B_REPEATABILITY`` -- the projection proof passes and cross-route drift is
  contained by independently observed same-route/grad-mode repeatability;
* ``EXACT_TRANSIENT`` -- every measured comparison is byte exact;
* ``INCONCLUSIVE`` -- all remaining cases.

The production B0/Z0 gate is imported but not called or modified.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import timedelta
import hashlib
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

import run_qmosaic_editor_direction_sp4_v1 as production  # noqa: E402


qmosaic = production.qmosaic
legacy = production.legacy

METHOD_NAME = "bernini-qmosaic-abba-zero-route-parity-diagnostic-sp4"
RANK_RECEIPT_SCHEMA = "bernini-qmosaic-abba-parity-rank-v1"
WORLD4_RECEIPT_SCHEMA = "bernini-qmosaic-abba-parity-world4-v1"
ALL8_MANIFEST_SCHEMA = "bernini-qmosaic-abba-parity-all8-v1"
RANK_RECEIPT_BASENAME = "rank-{sp_rank}.receipt.json"
WORLD4_RECEIPT_BASENAME = "world4.receipt.json"
ALL8_MANIFEST_BASENAME = "all8.manifest.json"

WORLD_SIZE = production.WORLD_SIZE
SP_SIZE = production.SP_SIZE
CHECKPOINT_CONTENT_FILE_COUNT = production.CHECKPOINT_CONTENT_FILE_COUNT
FIXED_REGISTRY_SHA256 = production.FIXED_REGISTRY_SHA256
FIXED_QUERY_SEEDS = production.FIXED_QUERY_SEEDS

A_WRAPPER_ROUTE = "A_WRAPPER_ROUTE"
B_REPEATABILITY = "B_REPEATABILITY"
EXACT_TRANSIENT = "EXACT_TRANSIENT"
INCONCLUSIVE = "INCONCLUSIVE"
VERDICTS = frozenset(
    {A_WRAPPER_ROUTE, B_REPEATABILITY, EXACT_TRANSIENT, INCONCLUSIVE}
)
ROLE_ORDER = ("action", "noop")
ABBA_CALL_ORDER = ("O0", "N0", "N1", "O1")
COMPLETE_CALL_ORDER = ("M", *ABBA_CALL_ORDER, "P")

PAIR_SPECS = (
    ("off_off_O0_O1", "O0", "O1", "within_off"),
    ("on0_on0_N0_N1", "N0", "N1", "within_on0"),
    ("off_on0_O0_N0", "O0", "N0", "cross"),
    ("off_on0_O0_N1", "O0", "N1", "cross"),
    ("off_on0_O1_N0", "O1", "N0", "cross"),
    ("off_on0_O1_N1", "O1", "N1", "cross"),
    ("mode_M_O0", "M", "O0", "mode"),
    ("mode_M_O1", "M", "O1", "mode"),
    ("failed_gate_M_N0", "M", "N0", "failed_gate"),
    ("failed_gate_M_N1", "M", "N1", "failed_gate"),
)

_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class QMosaicABBAParityDiagnosticError(RuntimeError):
    """A diagnostic input, live runtime, receipt, or aggregation differed."""


class DiagnosticRawBlock15TargetObserver(qmosaic.Block15TargetSuffixObserver):
    """Capture the pre-sketch local block-15 target rows without a GPU copy.

    The production observer is left untouched.  This subclass only retains a
    detached reference to the block output and its version counter in the hook;
    target-row selection and the device-to-CPU copy happen after the complete
    native forward.  Consequently the hook inserts no extra GPU reduction or
    host synchronization before the existing FP32 ``index_add_`` sketch.
    """

    def __init__(self, transformer: Any, *, spatial_sketch: Any):
        super().__init__(transformer, spatial_sketch=spatial_sketch)
        self._diagnostic_expected: Optional[tuple[str, str]] = None
        self._diagnostic_raw: dict[tuple[str, str], tuple[Any, int, Any]] = {}

    @contextmanager
    def expect_replay(self, *, role: str, call: str):
        if (
            role not in ROLE_ORDER
            or call not in ABBA_CALL_ORDER + ("P",)
            or self._diagnostic_expected is not None
        ):
            raise QMosaicABBAParityDiagnosticError(
                "raw block15 replay expectation differs"
            )
        self._diagnostic_expected = (role, call)
        try:
            yield
        finally:
            self._diagnostic_expected = None

    def _hook(self, module: Any, inputs: Any, output: Any) -> None:
        import torch

        pending = self._pending  # noqa: SLF001 - diagnostic subclass of pinned observer
        if (
            pending is None
            or not isinstance(output, torch.Tensor)
            or output.ndim != 3
            or int(output.shape[0]) != 1
        ):
            raise QMosaicABBAParityDiagnosticError(
                "raw block15 hook tensor/pending state differs"
            )
        observed_role, layout, _ = pending
        role = observed_role.split("-", 1)[0]
        if role not in ROLE_ORDER:
            raise QMosaicABBAParityDiagnosticError("raw block15 role differs")
        if observed_role.endswith("-measure"):
            key = (role, "M")
            if self._diagnostic_expected is not None:
                raise QMosaicABBAParityDiagnosticError(
                    "sealed measurement occurred inside a replay expectation"
                )
        elif observed_role.endswith("-replay"):
            key = self._diagnostic_expected
            if key is None or key[0] != role:
                raise QMosaicABBAParityDiagnosticError(
                    "raw block15 replay call label differs"
                )
        else:
            raise QMosaicABBAParityDiagnosticError("raw block15 capture role differs")
        if key in self._diagnostic_raw:
            raise QMosaicABBAParityDiagnosticError(
                "raw block15 diagnostic call repeated"
            )
        # No clone, index-select, digest, scalar read, or CPU transfer occurs in
        # this pre-sketch hook.  The detached view shares the version counter,
        # which is checked before it is consumed after the forward.
        detached = output.detach()
        self._diagnostic_raw[key] = (detached, int(detached._version), layout)
        super()._hook(module, inputs, output)

    def consume_raw_target(self, *, role: str, call: str) -> Any:
        import torch

        key = (role, call)
        row = self._diagnostic_raw.pop(key, None)
        if row is None:
            raise QMosaicABBAParityDiagnosticError(
                "raw block15 diagnostic call is absent"
            )
        hidden, version, layout = row
        if int(hidden._version) != version:
            raise QMosaicABBAParityDiagnosticError(
                "block15 output was mutated after the diagnostic hook"
            )
        indices = layout.local_target_indices.to(device=hidden.device)
        raw = hidden[0].index_select(0, indices).detach().cpu().contiguous()
        if (
            raw.ndim != 2
            or int(raw.shape[1]) != qmosaic.HIDDEN_SIZE
            or not raw.is_floating_point()
            or not bool(torch.isfinite(raw).all().item())
        ):
            raise QMosaicABBAParityDiagnosticError(
                "raw local block15 target suffix differs"
            )
        return raw

    def assert_diagnostic_empty(self) -> None:
        if self._diagnostic_expected is not None or self._diagnostic_raw:
            raise QMosaicABBAParityDiagnosticError(
                "raw block15 diagnostic capture was not completely consumed"
            )

    def discard_failed_call(self, *, role: str, call: str) -> None:
        """Drop only a failed P capture before sealing A_WRAPPER_ROUTE evidence."""

        if call != "P" or role not in ROLE_ORDER:
            raise QMosaicABBAParityDiagnosticError(
                "only a failed projection-proof call may be discarded"
            )
        self._diagnostic_raw.pop((role, call), None)


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
        raise QMosaicABBAParityDiagnosticError(
            "receipt is not finite canonical ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def seal_receipt(unsigned: Mapping[str, Any]) -> dict[str, Any]:
    if "receipt_digest" in unsigned:
        raise QMosaicABBAParityDiagnosticError("receipt is already sealed")
    value = dict(unsigned)
    return {**value, "receipt_digest": object_sha256(value)}


def validate_sealed_receipt(value: Any, *, schema: str, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise QMosaicABBAParityDiagnosticError(f"{label} is not a mapping")
    normalized = dict(value)
    digest = normalized.pop("receipt_digest", None)
    if (
        normalized.get("schema_version") != schema
        or not isinstance(digest, str)
        or _SHA256_RE.fullmatch(digest) is None
        or object_sha256(normalized) != digest
    ):
        raise QMosaicABBAParityDiagnosticError(f"{label} seal differs")
    return dict(value)


def write_create_only_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    if (
        not target.is_absolute()
        or not target.parent.is_dir()
        or target.parent.is_symlink()
        or target.exists()
        or target.is_symlink()
    ):
        raise QMosaicABBAParityDiagnosticError(
            "receipt path must be fresh absolute under a plain parent"
        )
    payload = canonical_json_bytes(value) + b"\n"
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def file_sha256(path: str | Path) -> str:
    source = Path(path)
    if not source.is_absolute() or not source.is_file() or source.is_symlink():
        raise QMosaicABBAParityDiagnosticError(
            "hashed receipt must be an absolute plain file"
        )
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: Any, *, label: str) -> str:
    try:
        return qmosaic.tensor_sha256(value, label=label)
    except Exception as error:
        raise QMosaicABBAParityDiagnosticError(f"{label} cannot be hashed") from error


def tensor_record(value: Any, *, label: str) -> Mapping[str, Any]:
    import torch

    if (
        not isinstance(value, torch.Tensor)
        or not value.is_floating_point()
        or value.layout != torch.strided
        or not bool(torch.isfinite(value).all().item())
    ):
        raise QMosaicABBAParityDiagnosticError(
            f"{label} must be a finite strided floating tensor"
        )
    contiguous = value.detach().cpu().contiguous()
    return {
        "shape": list(map(int, contiguous.shape)),
        "dtype": str(contiguous.dtype),
        "numel": int(contiguous.numel()),
        "tensor_sha256": _tensor_sha256(contiguous, label=label),
    }


def compare_fp32_tensors(
    left: Any,
    right: Any,
    *,
    left_label: str,
    right_label: str,
    left_record: Optional[Mapping[str, Any]] = None,
    right_record: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Return exact-byte and numerical metrics for one preregistered pair."""

    return compare_floating_tensors(
        left,
        right,
        left_label=left_label,
        right_label=right_label,
        left_record=left_record,
        right_record=right_record,
        require_fp32=True,
        existing_replay_bound_applicable=True,
    )


def compare_floating_tensors(
    left: Any,
    right: Any,
    *,
    left_label: str,
    right_label: str,
    left_record: Optional[Mapping[str, Any]] = None,
    right_record: Optional[Mapping[str, Any]] = None,
    require_fp32: bool = False,
    existing_replay_bound_applicable: bool = False,
) -> Mapping[str, Any]:
    """Compare raw floating tensors while preserving their original bytes."""

    import torch

    if (
        not isinstance(left, torch.Tensor)
        or not isinstance(right, torch.Tensor)
        or not left.is_floating_point()
        or not right.is_floating_point()
        or left.dtype != right.dtype
        or (require_fp32 and left.dtype != torch.float32)
        or left.shape != right.shape
        or not bool(torch.isfinite(left).all().item())
        or not bool(torch.isfinite(right).all().item())
    ):
        raise QMosaicABBAParityDiagnosticError(
            "pair comparison requires equal finite floating tensors"
        )
    lhs = left.detach().cpu().contiguous()
    rhs = right.detach().cpu().contiguous()
    lhs_record = dict(left_record or tensor_record(lhs, label=left_label))
    rhs_record = dict(right_record or tensor_record(rhs, label=right_label))
    if (
        lhs_record.get("shape") != list(map(int, lhs.shape))
        or rhs_record.get("shape") != list(map(int, rhs.shape))
        or lhs_record.get("dtype") != str(lhs.dtype)
        or rhs_record.get("dtype") != str(rhs.dtype)
        or lhs_record.get("numel") != int(lhs.numel())
        or rhs_record.get("numel") != int(rhs.numel())
        or _SHA256_RE.fullmatch(str(lhs_record.get("tensor_sha256"))) is None
        or _SHA256_RE.fullmatch(str(rhs_record.get("tensor_sha256"))) is None
    ):
        raise QMosaicABBAParityDiagnosticError("pair tensor record differs")

    if lhs.numel() == 0:
        maximum = rms = difference_l2 = left_l2 = reference_l2 = relative_l2 = 0.0
        reference_max_abs = 0.0
        raw_mismatch_count = numerical_mismatch_count = 0
    else:
        difference = lhs.double() - rhs.double()
        absolute = difference.abs()
        maximum = float(absolute.max().item())
        rms = float(torch.sqrt(difference.square().mean()).item())
        difference_l2 = float(torch.linalg.vector_norm(difference).item())
        left_l2 = float(torch.linalg.vector_norm(lhs.double()).item())
        reference_l2 = float(torch.linalg.vector_norm(rhs.double()).item())
        denominator = max(left_l2, reference_l2, 1.0e-30)
        relative_l2 = difference_l2 / denominator
        reference_max_abs = float(rhs.double().abs().max().item())
        bit_dtype = {
            2: torch.int16,
            4: torch.int32,
            8: torch.int64,
        }.get(int(lhs.element_size()))
        if bit_dtype is None:
            raise QMosaicABBAParityDiagnosticError(
                "floating tensor element width is unsupported"
            )
        raw_mismatch_count = int(
            torch.count_nonzero(lhs.view(bit_dtype) != rhs.view(bit_dtype)).item()
        )
        numerical_mismatch_count = int(torch.count_nonzero(lhs != rhs).item())
    existing_threshold = float(
        qmosaic.REPLAY_ATOL + qmosaic.REPLAY_RTOL * reference_max_abs
    )
    raw_exact = lhs_record["tensor_sha256"] == rhs_record["tensor_sha256"]
    torch_equal = bool(torch.equal(lhs, rhs))
    values = (
        maximum,
        rms,
        difference_l2,
        left_l2,
        reference_l2,
        relative_l2,
        reference_max_abs,
        existing_threshold,
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise QMosaicABBAParityDiagnosticError("pair metric is non-finite")
    return {
        "left": left_label,
        "right": right_label,
        "shape": list(map(int, lhs.shape)),
        "dtype": str(lhs.dtype),
        "numel": int(lhs.numel()),
        "empty_tensor": lhs.numel() == 0,
        "left_tensor_sha256": lhs_record["tensor_sha256"],
        "right_tensor_sha256": rhs_record["tensor_sha256"],
        "raw_exact_equal": raw_exact,
        "torch_equal": torch_equal,
        "exact_equal": raw_exact and torch_equal and raw_mismatch_count == 0,
        "raw_mismatch_count": raw_mismatch_count,
        "numerical_mismatch_count": numerical_mismatch_count,
        "max_abs": maximum,
        "rms": rms,
        "difference_l2": difference_l2,
        "left_l2": left_l2,
        "reference_l2": reference_l2,
        "relative_l2": relative_l2,
        "relative_l2_policy": "l2_difference/max(l2_left,l2_right,1e-30)",
        "symmetric_relative_l2_denominator": max(
            left_l2, reference_l2, 1.0e-30
        ),
        "reference_max_abs": reference_max_abs,
        "existing_replay_atol": float(qmosaic.REPLAY_ATOL),
        "existing_replay_rtol": float(qmosaic.REPLAY_RTOL),
        "existing_replay_bound": existing_threshold,
        "existing_replay_bound_applicable": existing_replay_bound_applicable,
        "within_existing_replay_bound": (
            maximum <= existing_threshold
            if existing_replay_bound_applicable
            else None
        ),
    }


def projection_proof_passes(value: Any, *, role: str, sp_rank: int) -> bool:
    if not isinstance(value, Mapping):
        return False
    return (
        value.get("schema_version") == qmosaic.ZERO_ROUTE_PROOF_SCHEMA_VERSION
        and value.get("role") == role
        and value.get("sp_rank") == sp_rank
        and value.get("wrapper_count") == len(qmosaic.CANONICAL_B_PARAMETER_NAMES)
        and value.get("missing_wrapper_count") == 0
        and value.get("repeated_wrapper_count") == 0
        and value.get("all_selected_deltas_numerically_exact_zero") is True
        and value.get("all_base_result_raw_bytes_equal") is True
        and value.get("b_unchanged") is True
        and value.get("b_state_before_sha256")
        == value.get("b_state_after_sha256")
        and _SHA256_RE.fullmatch(str(value.get("digest"))) is not None
        and qmosaic.object_sha256(
            {key: item for key, item in value.items() if key != "digest"}
        )
        == value.get("digest")
    )


def classify_role(
    pairs: Mapping[str, Mapping[str, Any]], *, projection_passed: bool
) -> tuple[str, Mapping[str, Any]]:
    """Classify one role using only preregistered pair categories."""

    expected_names = {name for name, _, _, _ in PAIR_SPECS}
    if set(pairs) != expected_names:
        raise QMosaicABBAParityDiagnosticError("role pair closure differs")
    categories = {
        category: [pairs[name] for name, _, _, observed in PAIR_SPECS if observed == category]
        for category in {row[3] for row in PAIR_SPECS}
    }
    for rows in categories.values():
        if any(
            not isinstance(row, Mapping)
            or type(row.get("exact_equal")) is not bool
            or not math.isfinite(float(row.get("max_abs", float("nan"))))
            or not math.isfinite(float(row.get("relative_l2", float("nan"))))
            for row in rows
        ):
            raise QMosaicABBAParityDiagnosticError("role pair metric differs")

    within_rows = categories["within_off"] + categories["within_on0"]
    mode_rows = categories["mode"]
    cross_rows = categories["cross"]
    failed_gate_rows = categories["failed_gate"]
    repeat_max_abs = max(float(row["max_abs"]) for row in within_rows)
    repeat_relative_l2 = max(float(row["relative_l2"]) for row in within_rows)
    mode_max_abs = max(float(row["max_abs"]) for row in mode_rows)
    mode_relative_l2 = max(float(row["relative_l2"]) for row in mode_rows)
    envelope_max_abs = max(repeat_max_abs, mode_max_abs)
    envelope_relative_l2 = max(repeat_relative_l2, mode_relative_l2)
    cross_max_abs = max(float(row["max_abs"]) for row in cross_rows)
    cross_relative_l2 = max(float(row["relative_l2"]) for row in cross_rows)
    baseline_nonexact = any(
        row["exact_equal"] is not True for row in (*within_rows, *mode_rows)
    )
    failed_gate_nonexact = any(
        row["exact_equal"] is not True for row in failed_gate_rows
    )
    all_exact = all(row["exact_equal"] is True for row in pairs.values())
    cross_inside = (
        cross_max_abs <= envelope_max_abs
        and cross_relative_l2 <= envelope_relative_l2
    )
    evidence = {
        "repeat_envelope": {
            "max_abs": repeat_max_abs,
            "relative_l2": repeat_relative_l2,
        },
        "measurement_mode_envelope": {
            "max_abs": mode_max_abs,
            "relative_l2": mode_relative_l2,
        },
        "combined_envelope": {
            "max_abs": envelope_max_abs,
            "relative_l2": envelope_relative_l2,
            "inflation_factor": 1.0,
        },
        "cross_route_observed": {
            "max_abs": cross_max_abs,
            "relative_l2": cross_relative_l2,
        },
        "baseline_nonexact": baseline_nonexact,
        "failed_gate_nonexact": failed_gate_nonexact,
        "cross_route_inside_predeclared_envelope": cross_inside,
        "projection_proof_passed": projection_passed,
    }
    if not projection_passed:
        return A_WRAPPER_ROUTE, evidence
    if all_exact:
        return EXACT_TRANSIENT, evidence
    if baseline_nonexact and failed_gate_nonexact and cross_inside:
        return B_REPEATABILITY, evidence
    return INCONCLUSIVE, evidence


def combine_verdicts(verdicts: Sequence[str]) -> str:
    if not verdicts or any(value not in VERDICTS for value in verdicts):
        raise QMosaicABBAParityDiagnosticError("verdict closure differs")
    if A_WRAPPER_ROUTE in verdicts:
        return A_WRAPPER_ROUTE
    if all(value == EXACT_TRANSIENT for value in verdicts):
        return EXACT_TRANSIENT
    if all(value in {B_REPEATABILITY, EXACT_TRANSIENT} for value in verdicts) and any(
        value == B_REPEATABILITY for value in verdicts
    ):
        return B_REPEATABILITY
    return INCONCLUSIVE


def build_role_diagnostic(
    *,
    role: str,
    sp_rank: int,
    tensors: Mapping[str, Any],
    raw_target_tensors: Mapping[str, Any],
    call_execution: Mapping[str, Mapping[str, Any]],
    projection_proof: Any,
    projection_error: Optional[Mapping[str, str]] = None,
) -> Mapping[str, Any]:
    if role not in ROLE_ORDER or sp_rank not in range(SP_SIZE):
        raise QMosaicABBAParityDiagnosticError("role/SP rank differs")
    if (
        set(tensors) != set(COMPLETE_CALL_ORDER)
        or set(raw_target_tensors) != set(COMPLETE_CALL_ORDER)
        or set(call_execution) != set(COMPLETE_CALL_ORDER)
    ):
        raise QMosaicABBAParityDiagnosticError("diagnostic call closure differs")
    for call in COMPLETE_CALL_ORDER:
        row = call_execution[call]
        expected_adapter = call.startswith("N") or call == "P"
        expected_grad = call != "M"
        if (
            not isinstance(row, Mapping)
            or row.get("adapter_enabled") is not expected_adapter
            or row.get("grad_enabled") is not expected_grad
            or row.get("detach_observer") is not (call == "M")
            or row.get("inference_mode_enabled") is not False
        ):
            raise QMosaicABBAParityDiagnosticError(
                f"diagnostic {call} execution mode differs"
            )
    records = {
        name: tensor_record(value, label=f"{role}-{name}")
        for name, value in tensors.items()
    }
    pairs: dict[str, Mapping[str, Any]] = {}
    raw_records = {
        name: tensor_record(value, label=f"{role}-{name}-raw-block15-target")
        for name, value in raw_target_tensors.items()
    }
    raw_pairs: dict[str, Mapping[str, Any]] = {}
    pair_categories: dict[str, str] = {}
    for name, left, right, category in PAIR_SPECS:
        pairs[name] = compare_fp32_tensors(
            tensors[left],
            tensors[right],
            left_label=left,
            right_label=right,
            left_record=records[left],
            right_record=records[right],
        )
        raw_pairs[name] = compare_floating_tensors(
            raw_target_tensors[left],
            raw_target_tensors[right],
            left_label=left,
            right_label=right,
            left_record=raw_records[left],
            right_record=raw_records[right],
            require_fp32=False,
            existing_replay_bound_applicable=False,
        )
        pair_categories[name] = category
    proof_passed = projection_error is None and projection_proof_passes(
        projection_proof, role=role, sp_rank=sp_rank
    )
    verdict, envelope = classify_role(pairs, projection_passed=proof_passed)
    sketch_nonexact_names = sorted(
        name for name, value in pairs.items() if value["exact_equal"] is not True
    )
    raw_nonexact_names = sorted(
        name for name, value in raw_pairs.items() if value["exact_equal"] is not True
    )
    raw_exact_while_sketch_nonexact = bool(sketch_nonexact_names) and not raw_nonexact_names
    baseline_names = {
        name
        for name, _, _, category in PAIR_SPECS
        if category in {"within_off", "within_on0", "mode"}
    }
    raw_baseline_nonexact = any(name in baseline_names for name in raw_nonexact_names)
    if raw_exact_while_sketch_nonexact:
        numerical_attribution = "RAW_EXACT_SKETCH_NONEXACT"
    elif raw_baseline_nonexact:
        numerical_attribution = "RAW_NATIVE_PATH_NONEXACT"
    elif not sketch_nonexact_names and not raw_nonexact_names:
        numerical_attribution = "NO_NONEXACT_OBSERVED"
    else:
        numerical_attribution = "INCONCLUSIVE"
    return {
        "role": role,
        "call_order": list(COMPLETE_CALL_ORDER),
        "abba_envelope_call_order": list(ABBA_CALL_ORDER),
        "projection_call_excluded_from_envelope": True,
        "calls": records,
        "call_execution": {
            name: dict(call_execution[name]) for name in COMPLETE_CALL_ORDER
        },
        "raw_block15_target_calls": raw_records,
        "pair_categories": pair_categories,
        "pairs": pairs,
        "raw_block15_target_pairs": raw_pairs,
        "raw_vs_sketch_attribution": {
            "raw_capture_coordinate": (
                "local_block15_target_rows_before_FP32_index_add_sketch"
            ),
            "sketch_nonexact_pair_names": sketch_nonexact_names,
            "raw_nonexact_pair_names": raw_nonexact_names,
            "raw_exact_while_sketch_nonexact": raw_exact_while_sketch_nonexact,
            "raw_same_route_or_mode_nonexact": raw_baseline_nonexact,
            "classification": numerical_attribution,
            "P_excluded_from_all_pair_envelopes": True,
        },
        "projection_proof": (
            dict(projection_proof) if isinstance(projection_proof, Mapping) else None
        ),
        "projection_error": None if projection_error is None else dict(projection_error),
        "projection_proof_passed": proof_passed,
        "envelope": dict(envelope),
        "verdict": verdict,
    }


def build_rank_receipt(
    *,
    cell_id: str,
    query_seed: int,
    sp_rank: int,
    world_rank: int,
    role_diagnostics: Mapping[str, Mapping[str, Any]],
    provenance: Mapping[str, Any],
    runtime_environment: Mapping[str, Any],
    parameter_invariance: Mapping[str, Any],
    terminal_full_seal: Mapping[str, Any],
    output_path: str,
) -> Mapping[str, Any]:
    if (
        cell_id not in FIXED_QUERY_SEEDS
        or query_seed not in FIXED_QUERY_SEEDS[cell_id]
        or sp_rank not in range(SP_SIZE)
        or world_rank not in range(WORLD_SIZE)
        or world_rank != sp_rank
        or tuple(role_diagnostics) != ROLE_ORDER
        or any(
            role_diagnostics[role].get("role") != role
            or role_diagnostics[role].get("verdict") not in VERDICTS
            for role in ROLE_ORDER
        )
    ):
        raise QMosaicABBAParityDiagnosticError("rank receipt coordinate differs")
    verdict = combine_verdicts(
        [str(role_diagnostics[role]["verdict"]) for role in ROLE_ORDER]
    )
    unsigned = {
        "schema_version": RANK_RECEIPT_SCHEMA,
        "method_name": METHOD_NAME,
        "cell_id": cell_id,
        "query_seed": query_seed,
        "world_size": WORLD_SIZE,
        "sp_size": SP_SIZE,
        "world_rank": world_rank,
        "sp_rank": sp_rank,
        "call_contract": {
            "measurement": (
                "M=explicit_historical_job131900_no_grad_"
                "adapter_off_detach_true"
            ),
            "abba": list(ABBA_CALL_ORDER),
            "O": "graph_enabled_adapter_off",
            "N": "graph_enabled_adapter_on_exact_zero_B",
            "P": "separate_projection_local_zero_route_proof",
            "P_excluded_from_envelope": True,
        },
        "role_diagnostics": {
            role: dict(role_diagnostics[role]) for role in ROLE_ORDER
        },
        "verdict": verdict,
        "provenance": dict(provenance),
        "runtime_environment": dict(runtime_environment),
        "parameter_invariance": dict(parameter_invariance),
        "terminal_full_seal": dict(terminal_full_seal),
        "rank_receipt_path": output_path,
        "execution_authority": {
            "diagnostic_only": True,
            "decode_executed": False,
            "vjp_executed": False,
            "optimizer_created": False,
            "parameter_update_authorized": False,
            "parameter_update_performed": False,
            "adapter_checkpoint_written": False,
            "semantic_action_editing_success_claim": False,
        },
    }
    return seal_receipt(unsigned)


def build_world4_receipt(
    *,
    rank_receipts: Sequence[Mapping[str, Any]],
    rank_artifacts: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if len(rank_receipts) != SP_SIZE or len(rank_artifacts) != SP_SIZE:
        raise QMosaicABBAParityDiagnosticError("WORLD4 rank closure differs")
    normalized = [
        validate_sealed_receipt(row, schema=RANK_RECEIPT_SCHEMA, label="rank receipt")
        for row in rank_receipts
    ]
    if (
        [row.get("sp_rank") for row in normalized] != list(range(SP_SIZE))
        or [row.get("world_rank") for row in normalized] != list(range(WORLD_SIZE))
        or len({row.get("cell_id") for row in normalized}) != 1
        or len({row.get("query_seed") for row in normalized}) != 1
        or any(
            row.get("execution_authority", {}).get("diagnostic_only") is not True
            or row.get("execution_authority", {}).get("decode_executed") is not False
            or row.get("execution_authority", {}).get("vjp_executed") is not False
            or row.get("execution_authority", {}).get("parameter_update_performed")
            is not False
            for row in normalized
        )
    ):
        raise QMosaicABBAParityDiagnosticError("WORLD4 rank identity differs")
    artifacts = [dict(row) for row in rank_artifacts]
    if (
        [row.get("sp_rank") for row in artifacts] != list(range(SP_SIZE))
        or any(
            _SHA256_RE.fullmatch(str(row.get("file_sha256"))) is None
            or _SHA256_RE.fullmatch(str(row.get("receipt_digest"))) is None
            for row in artifacts
        )
        or [row["receipt_digest"] for row in artifacts]
        != [row["receipt_digest"] for row in normalized]
    ):
        raise QMosaicABBAParityDiagnosticError("WORLD4 rank artifact differs")
    verdict = combine_verdicts([str(row["verdict"]) for row in normalized])
    first = normalized[0]
    unsigned = {
        "schema_version": WORLD4_RECEIPT_SCHEMA,
        "method_name": METHOD_NAME,
        "cell_id": first["cell_id"],
        "query_seed": first["query_seed"],
        "topology": "WORLD4_SP4",
        "rank_receipts": artifacts,
        "rank_verdicts": [row["verdict"] for row in normalized],
        "verdict": verdict,
        "all_rank_projection_proofs_passed": all(
            all(
                row["role_diagnostics"][role]["projection_proof_passed"] is True
                for role in ROLE_ORDER
            )
            for row in normalized
        ),
        "provenance_consensus": {
            "method_source_revision": first["provenance"].get(
                "method_source_revision"
            ),
            "method_source_archive_sha256": first["provenance"].get(
                "method_source_archive_sha256"
            ),
            "checkpoint_content_receipt_digest": first["provenance"].get(
                "checkpoint_content_receipt_digest"
            ),
        },
        "execution_authority": {
            "diagnostic_only": True,
            "decode_executed": False,
            "vjp_executed": False,
            "optimizer_created": False,
            "parameter_update_authorized": False,
            "parameter_update_performed": False,
            "scientific_action_editing_success_claim": False,
        },
    }
    return seal_receipt(unsigned)


def _load_json_receipt(path: str | Path, *, schema: str, label: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_absolute() or not source.is_file() or source.is_symlink():
        raise QMosaicABBAParityDiagnosticError(f"{label} path differs")
    try:
        raw = source.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise QMosaicABBAParityDiagnosticError(f"{label} JSON differs") from error
    if raw != canonical_json_bytes(value) + b"\n":
        raise QMosaicABBAParityDiagnosticError(f"{label} is not canonical JSON")
    return validate_sealed_receipt(value, schema=schema, label=label)


def aggregate_all8(
    *, dog_world4_receipt: str | Path, human_world4_receipt: str | Path, output: str | Path
) -> Mapping[str, Any]:
    sources = [Path(dog_world4_receipt), Path(human_world4_receipt)]
    rows = [
        _load_json_receipt(path, schema=WORLD4_RECEIPT_SCHEMA, label="WORLD4 receipt")
        for path in sources
    ]
    if (
        [row.get("cell_id") for row in rows] != ["dog", "human"]
        or any(
            row.get("query_seed") not in FIXED_QUERY_SEEDS[row["cell_id"]]
            or row.get("topology") != "WORLD4_SP4"
            or len(row.get("rank_receipts", ())) != SP_SIZE
            or row.get("execution_authority", {}).get("diagnostic_only") is not True
            for row in rows
        )
    ):
        raise QMosaicABBAParityDiagnosticError("all8 dog/human closure differs")
    for world4, source in zip(rows, sources):
        if source.name != WORLD4_RECEIPT_BASENAME:
            raise QMosaicABBAParityDiagnosticError("WORLD4 receipt basename differs")
        for artifact in world4["rank_receipts"]:
            rank_path = Path(str(artifact.get("path")))
            if (
                not rank_path.is_absolute()
                or rank_path.parent != source.parent
                or file_sha256(rank_path) != artifact.get("file_sha256")
            ):
                raise QMosaicABBAParityDiagnosticError(
                    "all8 rank receipt artifact changed"
                )
            rank = _load_json_receipt(
                rank_path, schema=RANK_RECEIPT_SCHEMA, label="rank receipt"
            )
            if rank.get("receipt_digest") != artifact.get("receipt_digest"):
                raise QMosaicABBAParityDiagnosticError(
                    "all8 rank receipt digest differs"
                )
    output_path = Path(output)
    if (
        not output_path.is_absolute()
        or output_path.name != ALL8_MANIFEST_BASENAME
        or sources[0].parent.parent != output_path.parent
        or sources[1].parent.parent != output_path.parent
    ):
        raise QMosaicABBAParityDiagnosticError("all8 output topology differs")
    consensus = [row["provenance_consensus"] for row in rows]
    if consensus[0] != consensus[1]:
        raise QMosaicABBAParityDiagnosticError("all8 provenance differs")
    unsigned = {
        "schema_version": ALL8_MANIFEST_SCHEMA,
        "method_name": METHOD_NAME,
        "topology": "one_node_8xMI210_two_concurrent_WORLD4_SP4",
        "cells": [
            {
                "cell_id": row["cell_id"],
                "query_seed": row["query_seed"],
                "world4_receipt_path": str(source),
                "world4_receipt_file_sha256": file_sha256(source),
                "world4_receipt_digest": row["receipt_digest"],
                "verdict": row["verdict"],
                "rank_receipt_digests": [
                    artifact["receipt_digest"] for artifact in row["rank_receipts"]
                ],
            }
            for row, source in zip(rows, sources)
        ],
        "verdict": combine_verdicts([str(row["verdict"]) for row in rows]),
        "provenance_consensus": consensus[0],
        "rank_receipt_count": 2 * SP_SIZE,
        "execution_authority": {
            "diagnostic_only": True,
            "decode_executed": False,
            "vjp_executed": False,
            "optimizer_created": False,
            "parameter_update_authorized": False,
            "parameter_update_performed": False,
            "scientific_action_editing_success_claim": False,
        },
    }
    receipt = seal_receipt(unsigned)
    write_create_only_json(output_path, receipt)
    return receipt


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--expected-checkpoint-content-manifest-sha256", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--expected-registry-sha256", default=FIXED_REGISTRY_SHA256)
    parser.add_argument("--cell-id", choices=tuple(FIXED_QUERY_SEEDS), required=True)
    parser.add_argument("--query-seed", type=int, required=True)
    parser.add_argument("--owner-root", required=True)
    parser.add_argument("--owner-master-receipt", required=True)
    parser.add_argument("--expected-owner-master-receipt-sha256", required=True)
    parser.add_argument("--owner-audit-sidecar", required=True)
    parser.add_argument("--expected-owner-audit-sidecar-sha256", required=True)
    parser.add_argument("--owner-audit-evidence", required=True)
    parser.add_argument("--owner-audit-public-key", required=True)
    parser.add_argument("--expected-owner-audit-public-key-sha256", required=True)
    parser.add_argument("--owner-cell-root", required=True)
    parser.add_argument("--owner-cell-receipt", required=True)
    parser.add_argument("--expected-owner-cell-receipt-sha256", required=True)
    parser.add_argument("--editor-receipt", required=True)
    parser.add_argument("--expected-editor-receipt-sha256", required=True)
    parser.add_argument("--editor-public-key", required=True)
    parser.add_argument("--expected-editor-public-key-sha256", required=True)
    parser.add_argument("--editor-artifact-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    parser.add_argument(
        "--expected-bernini-commit", default=legacy.trainer.BERNINI_OFFICIAL_COMMIT
    )
    parser.add_argument(
        "--expected-veomni-commit", default=legacy.trainer.VEOMNI_TESTED_COMMIT
    )
    parser.add_argument("--diagnostic-only", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runtime = commands.add_parser("run-world4", help="run one real WORLD4/SP4 cell")
    _add_runtime_arguments(runtime)
    aggregate = commands.add_parser(
        "aggregate-all8", help="seal dog+human WORLD4 receipts into one all8 manifest"
    )
    aggregate.add_argument("--dog-world4-receipt", required=True)
    aggregate.add_argument("--human-world4-receipt", required=True)
    aggregate.add_argument("--output", required=True)
    return parser


def validate_runtime_cli(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.diagnostic_only is not True:
        raise QMosaicABBAParityDiagnosticError(
            "run-world4 requires explicit --diagnostic-only"
        )
    if args.expected_registry_sha256 != FIXED_REGISTRY_SHA256:
        raise QMosaicABBAParityDiagnosticError("registry SHA is fixed")
    if args.query_seed not in FIXED_QUERY_SEEDS.get(args.cell_id, ()):
        raise QMosaicABBAParityDiagnosticError("cell/query seed is not preregistered")
    for name in (
        "expected_bernini_commit",
        "expected_veomni_commit",
        "method_source_revision",
    ):
        if _SHA1_RE.fullmatch(str(getattr(args, name))) is None:
            raise QMosaicABBAParityDiagnosticError(f"{name} must be full SHA-1")
    for name in (
        "expected_checkpoint_content_manifest_sha256",
        "expected_owner_master_receipt_sha256",
        "expected_owner_audit_sidecar_sha256",
        "expected_owner_audit_public_key_sha256",
        "expected_owner_cell_receipt_sha256",
        "expected_editor_receipt_sha256",
        "expected_editor_public_key_sha256",
        "method_source_archive_sha256",
    ):
        if _SHA256_RE.fullmatch(str(getattr(args, name))) is None:
            raise QMosaicABBAParityDiagnosticError(f"{name} must be SHA-256")
    output = Path(args.output_dir)
    if (
        not output.is_absolute()
        or output == Path("/")
        or output.name in {"", ".", ".."}
        or not output.parent.is_dir()
        or output.parent.is_symlink()
    ):
        raise QMosaicABBAParityDiagnosticError(
            "output directory must be absolute under a plain existing parent"
        )
    try:
        return production._strict_registry_cell(  # noqa: SLF001 - pinned registry
            args.registry, cell_id=args.cell_id, query_seed=args.query_seed
        )
    except production.QMosaicEditorDirectionError as error:
        raise QMosaicABBAParityDiagnosticError(str(error)) from error


def _create_output_consensus(output: Path, *, dist: Any, rank: int) -> None:
    result: list[Any] = [None]
    if rank == 0:
        try:
            if output.exists() or output.is_symlink():
                raise QMosaicABBAParityDiagnosticError(
                    "output directory must be create-only"
                )
            output.mkdir(mode=0o700)
            result[0] = {"ok": True}
        except Exception as error:
            result[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    dist.broadcast_object_list(result, src=0)
    if not isinstance(result[0], Mapping) or result[0].get("ok") is not True:
        raise QMosaicABBAParityDiagnosticError(
            f"rank-zero output creation failed: {result[0]}"
        )
    dist.barrier()


def _historical_measure_to_cpu(
    *,
    runner: Any,
    role: str,
    observer: DiagnosticRawBlock15TargetObserver,
    torch: Any,
    dist: Any,
) -> tuple[Any, Any, Mapping[str, Any]]:
    """Reproduce job 131900's no-grad/off/detached local M coordinate."""

    dist.barrier()
    torch.cuda.synchronize()
    with torch.no_grad():
        execution = {
            "adapter_enabled": False,
            "grad_enabled": bool(torch.is_grad_enabled()),
            "inference_mode_enabled": bool(torch.is_inference_mode_enabled()),
            "detach_observer": True,
            "source": "explicit_historical_job131900_measurement",
        }
        local = runner._forward_local(  # noqa: SLF001 - diagnostic pinned runner
            role=role, adapter_enabled=False, detach=True
        )
    torch.cuda.synchronize()
    value = local.detach().float().cpu().contiguous()
    raw_target = observer.consume_raw_target(role=role, call="M")
    del local
    dist.barrier()
    return value, raw_target, execution


def _replay_to_cpu(
    *,
    runner: Any,
    role: str,
    call: str,
    enabled: bool,
    observer: DiagnosticRawBlock15TargetObserver,
    torch: Any,
    dist: Any,
) -> tuple[Any, Any, Mapping[str, Any]]:
    expected_enabled = call.startswith("N") or call == "P"
    if call not in ABBA_CALL_ORDER + ("P",) or enabled is not expected_enabled:
        raise QMosaicABBAParityDiagnosticError("ABBA replay route/call differs")
    dist.barrier()
    torch.cuda.synchronize()
    with torch.enable_grad():
        execution = {
            "adapter_enabled": enabled,
            "grad_enabled": bool(torch.is_grad_enabled()),
            "inference_mode_enabled": bool(torch.is_inference_mode_enabled()),
            "detach_observer": False,
            "source": "explicit_diagnostic_graph_forward",
        }
        with observer.expect_replay(role=role, call=call):
            graph = runner._forward_local(  # noqa: SLF001 - diagnostic pinned runner
                role=role, adapter_enabled=enabled, detach=False
            )
    torch.cuda.synchronize()
    value = graph.detach().float().cpu().contiguous()
    raw_target = observer.consume_raw_target(role=role, call=call)
    del graph
    dist.barrier()
    return value, raw_target, execution


def run_world4(args: argparse.Namespace) -> Mapping[str, Any]:
    cell = validate_runtime_cli(args)
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            legacy.trainer.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = legacy.trainer.validate_checkpoint(
            args.checkpoint
        )
    except legacy.trainer.TrainingContractError as error:
        raise QMosaicABBAParityDiagnosticError(str(error)) from error
    if transformer_config.get("num_attention_heads") != 12:
        raise QMosaicABBAParityDiagnosticError(
            "pinned Bernini attention geometry differs"
        )
    legacy.trainer.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import get_parallel_state, init_parallel_state

    distributed = legacy.inference_distributed_contract()
    if (
        distributed.world_size != WORLD_SIZE
        or distributed.ulysses_size != SP_SIZE
        or not torch.cuda.is_available()
        or getattr(torch.version, "hip", None) is None
    ):
        raise QMosaicABBAParityDiagnosticError(
            "diagnostic requires one AUH WORLD4/SP4 ROCm group"
        )
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=180),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=SP_SIZE)
    device = torch.device("cuda", distributed.local_rank)
    output = Path(args.output_dir)
    observer: Any = None
    action_handle: Any = None
    renderer: Any = None
    try:
        _create_output_consensus(output, dist=dist, rank=distributed.rank)
        checkpoint_packet = qmosaic.load_validated_checkpoint_content_manifest(
            checkpoint_root=checkpoint,
            content_manifest_path=args.checkpoint_content_manifest,
            expected_manifest_sha256=args.expected_checkpoint_content_manifest_sha256,
            expected_file_count=CHECKPOINT_CONTENT_FILE_COUNT,
        )
        owner = qmosaic.load_authenticated_owner_quotient_packet(
            registry=args.registry,
            expected_registry_sha256=FIXED_REGISTRY_SHA256,
            owner_root=args.owner_root,
            owner_master_receipt=args.owner_master_receipt,
            expected_owner_master_receipt_sha256=args.expected_owner_master_receipt_sha256,
            audit_sidecar=args.owner_audit_sidecar,
            expected_audit_sidecar_sha256=args.expected_owner_audit_sidecar_sha256,
            audit_evidence=args.owner_audit_evidence,
            audit_public_key=args.owner_audit_public_key,
            expected_audit_public_key_sha256=args.expected_owner_audit_public_key_sha256,
            cell_root=args.owner_cell_root,
            receipt_path=args.owner_cell_receipt,
            expected_receipt_file_sha256=args.expected_owner_cell_receipt_sha256,
            query_seed=args.query_seed,
        )
        if (
            owner.cell_id != args.cell_id
            or owner.query_seed != args.query_seed
            or owner.source_iid != cell["source_iid"]
            or owner.source_video_sha256 != cell["source_video_sha256"]
        ):
            raise QMosaicABBAParityDiagnosticError(
                "owner differs from fixed registry cell"
            )
        runtime_inputs = qmosaic.load_authenticated_editor_runtime_input_packet(
            receipt_path=args.editor_receipt,
            expected_receipt_file_sha256=args.expected_editor_receipt_sha256,
            public_key_path=args.editor_public_key,
            expected_public_key_file_sha256=args.expected_editor_public_key_sha256,
            artifact_root=args.editor_artifact_root,
            owner=owner,
            checkpoint=checkpoint_packet,
        )
        editor_source_binding = runtime_inputs.receipt()
        if (
            editor_source_binding.get("method_source_revision")
            != args.method_source_revision
            or editor_source_binding.get("method_source_archive_sha256")
            != args.method_source_archive_sha256
        ):
            raise QMosaicABBAParityDiagnosticError(
                "editor packet source archive differs from diagnostic source"
            )

        config = BerniniRendererConfig.from_pretrained(
            str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
            local_files_only=True,
            **legacy.inference_renderer_config_overrides(checkpoint),
        )
        config.dtype = torch.bfloat16
        legacy.trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
        renderer = BerniniRendererModel(config).requires_grad_(False).eval().to(device)
        renderer.t5_text_encoder.to("cpu")
        torch.cuda.empty_cache()
        diffusion = renderer.diff_dec
        transformer = diffusion.transformer
        if transformer is None or diffusion.transformer_2 is not None:
            raise QMosaicABBAParityDiagnosticError(
                "renderer is not frozen transformer_1-only"
            )
        action_handle = qmosaic.install_core16_fixed_a_b_only_action_lora(transformer)
        action_handle.assert_fixed_gauge()
        parameter_before = action_handle.state_digest()
        b_before = action_handle.b_parameter_state_sha256()
        snapshot = action_handle.adapter_state_snapshot()

        clean_shape = runtime_inputs.tensors["clean_latent"].shape
        patch_positions = (int(clean_shape[3]) // 2) * (int(clean_shape[4]) // 2)
        observer = DiagnosticRawBlock15TargetObserver(
            transformer,
            spatial_sketch=qmosaic.make_fixed_spatial_sketch(patch_positions),
        )
        observer.install()
        collective = qmosaic.authenticate_live_bernini_sp4_collective(
            parallel_state=get_parallel_state()
        )
        runner = qmosaic.NativeSharedStepSP4ReplayRunner(
            diffusion=diffusion,
            transformer=transformer,
            owner=owner,
            runtime_inputs=runtime_inputs,
            action_handle=action_handle,
            observer=observer,
            sp4_collective=collective,
            sp_rank=collective.sp_rank,
            checkpoint_content=checkpoint_packet,
        )

        role_tensors: dict[str, dict[str, Any]] = {}
        role_raw_targets: dict[str, dict[str, Any]] = {}
        role_execution: dict[str, dict[str, Mapping[str, Any]]] = {}
        for role in ROLE_ORDER:
            measurement, raw_measurement, measurement_execution = (
                _historical_measure_to_cpu(
                    runner=runner,
                    role=role,
                    observer=observer,
                    torch=torch,
                    dist=dist,
                )
            )
            tensors = {"M": measurement}
            raw_targets = {"M": raw_measurement}
            executions = {"M": measurement_execution}
            for call, enabled in (("O0", False), ("N0", True), ("N1", True), ("O1", False)):
                tensors[call], raw_targets[call], executions[call] = _replay_to_cpu(
                    runner=runner,
                    role=role,
                    call=call,
                    enabled=enabled,
                    observer=observer,
                    torch=torch,
                    dist=dist,
                )
            role_tensors[role] = tensors
            role_raw_targets[role] = raw_targets
            role_execution[role] = executions

        projection_proofs: dict[str, Any] = {}
        projection_errors: dict[str, Optional[Mapping[str, str]]] = {}
        for role in ROLE_ORDER:
            holder: Any = None
            try:
                with action_handle.capture_zero_route_proof(
                    role=role, sp_rank=collective.sp_rank
                ) as holder:
                    (
                        role_tensors[role]["P"],
                        role_raw_targets[role]["P"],
                        role_execution[role]["P"],
                    ) = _replay_to_cpu(
                        runner=runner,
                        role=role,
                        call="P",
                        enabled=True,
                        observer=observer,
                        torch=torch,
                        dist=dist,
                    )
                projection_proofs[role] = holder.require_receipt()
                projection_errors[role] = None
            except Exception as error:
                observer.discard_failed_call(role=role, call="P")
                projection_proofs[role] = None
                projection_errors[role] = {
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                # Preserve the closed call schema even when proof finalization
                # rejects evidence after a completed P forward.
                if "P" not in role_tensors[role]:
                    role_tensors[role]["P"] = role_tensors[role]["N1"].clone()
                if "P" not in role_raw_targets[role]:
                    role_raw_targets[role]["P"] = role_raw_targets[role]["N1"].clone()
                if "P" not in role_execution[role]:
                    role_execution[role]["P"] = {
                        "adapter_enabled": True,
                        "grad_enabled": True,
                        "inference_mode_enabled": False,
                        "detach_observer": False,
                        "source": "failed_projection_proof_call",
                    }

        observer.assert_diagnostic_empty()

        role_diagnostics = {
            role: build_role_diagnostic(
                role=role,
                sp_rank=collective.sp_rank,
                tensors=role_tensors[role],
                raw_target_tensors=role_raw_targets[role],
                call_execution=role_execution[role],
                projection_proof=projection_proofs[role],
                projection_error=projection_errors[role],
            )
            for role in ROLE_ORDER
        }
        if not action_handle.adapter_state_matches(snapshot):
            raise QMosaicABBAParityDiagnosticError(
                "diagnostic changed Action-LoRA runtime state"
            )
        parameter_after = action_handle.state_digest()
        b_after = action_handle.b_parameter_state_sha256()
        action_handle.assert_fixed_gauge()
        if parameter_after != parameter_before or b_after != b_before:
            raise QMosaicABBAParityDiagnosticError(
                "diagnostic changed Action-LoRA parameter bytes"
            )
        parameter_invariance = {
            "action_lora_state_sha256_before": parameter_before,
            "action_lora_state_sha256_after": parameter_after,
            "lora_b_state_sha256_before": b_before,
            "lora_b_state_sha256_after": b_after,
            "parameter_bytes_unchanged": True,
            "lora_b_exact_zero_before": True,
            "lora_b_exact_zero_after": True,
            "optimizer_created": False,
            "parameter_update_performed": False,
        }
        provenance = {
            "method_source_revision": args.method_source_revision,
            "method_source_archive_sha256": args.method_source_archive_sha256,
            "bernini_revision": bernini_revision,
            "veomni_revision": veomni_revision,
            "checkpoint_content_receipt_digest": checkpoint_packet.receipt()["digest"],
            "owner_packet_receipt_digest": owner.receipt()["digest"],
            "editor_runtime_input_receipt_digest": editor_source_binding["digest"],
            "runner_contract_digest": runner.contract_receipt(deep=False)["digest"],
            "collective_receipt_digest": collective.receipt()["digest"],
            "historical_M_source": (
                "explicit_runner_forward_not_current_measure_role"
            ),
        }
        runtime_environment = {
            "torch_version": str(torch.__version__),
            "torch_hip_version": str(torch.version.hip),
            "device_name": str(torch.cuda.get_device_name(device)),
            "device_index": distributed.local_rank,
            "deterministic_algorithms_enabled": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "grad_enabled_at_diagnostic_boundary": bool(torch.is_grad_enabled()),
            "explicit_cuda_synchronize_before_and_after_every_replay": True,
        }

        # This is the final deep model/checkpoint/signed-input byte validation.
        terminal_local = production.assert_terminal_full_seal_before_publish(
            runner, sp_rank=collective.sp_rank
        )
        terminal_rows: list[Any] = [None] * SP_SIZE
        dist.all_gather_object(terminal_rows, terminal_local)
        if [row.get("sp_rank") for row in terminal_rows] != list(range(SP_SIZE)):
            raise QMosaicABBAParityDiagnosticError(
                "WORLD4 terminal full-seal rank order differs"
            )

        if observer is not None:
            observer.remove()
            observer = None
        if action_handle is not None and not action_handle.restored:
            action_handle.restore()
        action_handle = None
        renderer.to("cpu")
        torch.cuda.empty_cache()
        dist.barrier()

        rank_path = output / RANK_RECEIPT_BASENAME.format(
            sp_rank=collective.sp_rank
        )
        local_receipt = build_rank_receipt(
            cell_id=args.cell_id,
            query_seed=args.query_seed,
            sp_rank=collective.sp_rank,
            world_rank=distributed.rank,
            role_diagnostics=role_diagnostics,
            provenance=provenance,
            runtime_environment=runtime_environment,
            parameter_invariance=parameter_invariance,
            terminal_full_seal=terminal_local,
            output_path=str(rank_path),
        )
        rank_receipts: list[Any] = [None] * SP_SIZE
        dist.all_gather_object(rank_receipts, local_receipt)
        if [row.get("sp_rank") for row in rank_receipts] != list(range(SP_SIZE)):
            raise QMosaicABBAParityDiagnosticError(
                "WORLD4 rank receipt order differs"
            )
        write_create_only_json(rank_path, local_receipt)
        dist.barrier()

        world4_result: list[Any] = [None]
        if distributed.rank == 0:
            try:
                artifacts = []
                for rank, receipt in enumerate(rank_receipts):
                    path = output / RANK_RECEIPT_BASENAME.format(sp_rank=rank)
                    artifacts.append(
                        {
                            "sp_rank": rank,
                            "path": str(path),
                            "file_sha256": file_sha256(path),
                            "receipt_digest": receipt["receipt_digest"],
                        }
                    )
                world4 = build_world4_receipt(
                    rank_receipts=rank_receipts, rank_artifacts=artifacts
                )
                world4_path = output / WORLD4_RECEIPT_BASENAME
                write_create_only_json(world4_path, world4)
                world4_result[0] = {"ok": True, "receipt": world4}
            except Exception as error:
                world4_result[0] = {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
        dist.broadcast_object_list(world4_result, src=0)
        if (
            not isinstance(world4_result[0], Mapping)
            or world4_result[0].get("ok") is not True
        ):
            raise QMosaicABBAParityDiagnosticError(
                f"WORLD4 receipt publication failed: {world4_result[0]}"
            )
        dist.barrier()
        return dict(world4_result[0]["receipt"])
    finally:
        if observer is not None:
            observer.remove()
        if action_handle is not None and not action_handle.restored:
            action_handle.restore()
        if renderer is not None:
            renderer.to("cpu")
        if dist.is_initialized():
            dist.destroy_process_group()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "aggregate-all8":
        receipt = aggregate_all8(
            dog_world4_receipt=args.dog_world4_receipt,
            human_world4_receipt=args.human_world4_receipt,
            output=args.output,
        )
    else:
        receipt = run_world4(args)
    if os.environ.get("RANK", "0") == "0":
        print(canonical_json_bytes(receipt).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())


__all__ = [
    "ABBA_CALL_ORDER",
    "ALL8_MANIFEST_BASENAME",
    "ALL8_MANIFEST_SCHEMA",
    "A_WRAPPER_ROUTE",
    "B_REPEATABILITY",
    "COMPLETE_CALL_ORDER",
    "DiagnosticRawBlock15TargetObserver",
    "EXACT_TRANSIENT",
    "INCONCLUSIVE",
    "PAIR_SPECS",
    "QMosaicABBAParityDiagnosticError",
    "RANK_RECEIPT_SCHEMA",
    "ROLE_ORDER",
    "VERDICTS",
    "WORLD4_RECEIPT_SCHEMA",
    "aggregate_all8",
    "build_parser",
    "build_rank_receipt",
    "build_role_diagnostic",
    "build_world4_receipt",
    "classify_role",
    "combine_verdicts",
    "compare_floating_tensors",
    "compare_fp32_tensors",
    "main",
    "projection_proof_passes",
    "run_world4",
    "seal_receipt",
    "tensor_record",
    "validate_runtime_cli",
    "write_create_only_json",
]
