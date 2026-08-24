#!/usr/bin/env python3
"""Pinned d541801 diagnostic boundary for historical PAIR-v5 T2V v3 scores.

The active repository may contain later action-energy arithmetic experiments.
The historical two-log scalar is not active calibration truth.  This module
therefore provides two deliberately narrow, non-authorizing diagnostics:

* validate all formal score receipts in one isolated Python subprocess whose
  import root is an extracted d541801 source archive; and
* reproduce the d541801 live FP32 MACE scalar directly from branch energies.

It never accepts a v4 receipt and must not authorize an optimizer or an active
native-RV2V score.  Legacy public names remain only for archival readers.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
from typing import Any, Mapping, Sequence


PINNED_SOURCE_REVISION = "d541801a162796aacde34c2bfc2b1f0472d954d2"
PINNED_SCORER_SOURCE_SHA256 = (
    "3d7ce459ddb9a014873acd6384c7c4030b4e3aca9004c1b8486ebbc1f0f5d32e"
)
PINNED_MACE_SOURCE_SHA256 = (
    "0cd4b2c86aa9ccdd353010a8750eeecfebfb28425b44616e9fd52810dd90e986"
)
FORMAL_SCORE_SCHEMA = "bernini-pair-v5-frozen-t2v-global-energy-score-v3"
FORMAL_SCORE_FILENAME = "pair-v5-t2v-global-energy-score-v3.json"
HISTORICAL_SCORE_SCHEMA = FORMAL_SCORE_SCHEMA
HISTORICAL_SCORE_FILENAME = FORMAL_SCORE_FILENAME
V3_SCALAR_DEFINITION = (
    "diagnostic_non_authorizing_d541801_live_fp32_log_difference_first_argmin"
)
DIAGNOSTIC_NON_AUTHORIZING = True
ENERGY_EPSILON = 1.0e-8
MACE_CROSS_DEVICE_REPLAY_RTOL = 1.0e-5
MACE_CROSS_DEVICE_REPLAY_ATOL = 1.0e-6

BRANCH_ORDER = (
    "action",
    "noop",
    "incomplete",
    "reverse",
    "shuffle",
    "wrong_actor",
    "wrong_object",
    "camera_only",
    "appearance_only",
    "generic_wrong_motion",
)
HARD_NEGATIVE_BRANCHES = BRANCH_ORDER[1:]

_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_LIVE_PROOF_FIELDS = frozenset(
    {
        "branch_order",
        "hard_negative_order",
        "branch_energy_tensor_sha256",
        "negative_log_energy_ratio_tensor_sha256",
        "reward_tensor_sha256",
        "hardest_negative_index_tensor_sha256",
        "tensor_dtype",
        "formula_recomputed_on_origin_device_bit_exact",
        "reward_and_first_argmin_recomputed_on_origin_device_bit_exact",
        "digest",
    }
)
_PACKET_FIELDS = frozenset(
    {
        "definition",
        "energy_epsilon",
        "global_action_energy",
        "global_hard_negative_energy_by_branch",
        "global_negative_log_energy_ratio_by_branch",
        "global_hardest_negative_branch",
        "raw_global_action_energy_score",
        "mace_live_tensor_formula_proof",
        "compatibility_source_revision",
        "compatibility_scalar_definition",
        "diagnostic_non_authorizing",
        "optimizer_authorized",
        "scientific_action_editing_claim",
        "packet_digest",
    }
)


class PairV5T2VScoreV3CompatibilityError(RuntimeError):
    """The isolated v3 source or live v3 scalar failed closed."""


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
        raise PairV5T2VScoreV3CompatibilityError(
            "value is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    before = path.stat()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise PairV5T2VScoreV3CompatibilityError(
            f"file changed while hashing: {path}"
        )
    return digest.hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PairV5T2VScoreV3CompatibilityError(
            f"{label} must be lowercase SHA-256"
        )
    return value


def _plain_file(value: Any, *, label: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise PairV5T2VScoreV3CompatibilityError(f"{label} path differs")
    path = Path(value)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise PairV5T2VScoreV3CompatibilityError(
            f"{label} must be an absolute plain file"
        )
    return path.resolve(strict=True)


def _plain_directory(value: Any, *, label: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise PairV5T2VScoreV3CompatibilityError(f"{label} path differs")
    path = Path(value)
    if not path.is_absolute() or not path.is_dir() or path.is_symlink():
        raise PairV5T2VScoreV3CompatibilityError(
            f"{label} must be an absolute plain directory"
        )
    return path.resolve(strict=True)


def _critical_source_manifest(method_root: Path) -> dict[str, str]:
    scorer_path = _plain_file(
        method_root / "score_pair_v5_t2v_energy_bank_v3.py",
        label="pinned v3 scorer source",
    )
    mace_path = _plain_file(
        method_root / "mace_candidate_action_energy.py",
        label="pinned v3 MACE source",
    )
    manifest = {
        "score_pair_v5_t2v_energy_bank_v3.py": file_sha256(scorer_path),
        "mace_candidate_action_energy.py": file_sha256(mace_path),
    }
    if manifest != {
        "score_pair_v5_t2v_energy_bank_v3.py": PINNED_SCORER_SOURCE_SHA256,
        "mace_candidate_action_energy.py": PINNED_MACE_SOURCE_SHA256,
    }:
        raise PairV5T2VScoreV3CompatibilityError(
            "isolated formal-v3 critical source manifest differs from d541801"
        )
    return manifest


_ISOLATED_VALIDATOR = r"""
import json
from pathlib import Path
import sys

method_root = Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0, str(method_root))
import score_pair_v5_t2v_energy_bank_v3 as scorer

if scorer.SCORE_RECEIPT_SCHEMA != "bernini-pair-v5-frozen-t2v-global-energy-score-v3":
    raise SystemExit("isolated scorer schema is not formal v3")
rows = []
for raw_path in sys.argv[2:]:
    path = Path(raw_path)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise SystemExit("formal v3 score path differs")
    value = json.loads(path.read_bytes())
    checked = scorer.validate_score_receipt(value)
    if checked.get("schema_version") != scorer.SCORE_RECEIPT_SCHEMA:
        raise SystemExit("formal v3 score schema differs after validation")
    rows.append(checked)
sys.stdout.buffer.write(scorer.canonical_json_bytes(rows))
"""


def validate_formal_score_receipts_isolated(
    *,
    formal_v3_method_root: str | Path,
    formal_v3_source_revision: str,
    formal_v3_source_archive_sha256: str,
    score_paths: Sequence[str | Path],
    python_executable: str | Path = sys.executable,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run d541801's exact validator in an import-isolated subprocess."""

    if formal_v3_source_revision != PINNED_SOURCE_REVISION:
        raise PairV5T2VScoreV3CompatibilityError(
            "formal v3 source revision is not pinned d541801"
        )
    archive_sha = _sha256(
        formal_v3_source_archive_sha256,
        label="formal v3 source archive SHA-256",
    )
    method_root = _plain_directory(
        formal_v3_method_root, label="formal v3 isolated method root"
    )
    critical_manifest = _critical_source_manifest(method_root)
    paths = [
        _plain_file(value, label=f"formal v3 score {index}")
        for index, value in enumerate(score_paths)
    ]
    if len(paths) != 40 or len(set(paths)) != 40:
        raise PairV5T2VScoreV3CompatibilityError(
            "formal v3 validation requires exactly forty unique score files"
        )
    python_path = _plain_file(python_executable, label="isolated validator Python")
    # Preserve the selected runtime's loader/ROCm environment while removing
    # every Python import injection point.  ``-I`` provides the interpreter
    # boundary; this scrub prevents an inherited PYTHON* variable from making
    # the receipt validator depend on the active checkout.
    environment = dict(os.environ)
    for name in (
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "PYTHONINSPECT",
    ):
        environment.pop(name, None)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    completed = subprocess.run(
        [
            str(python_path),
            "-I",
            "-B",
            "-c",
            _ISOLATED_VALIDATOR,
            str(method_root),
            *(str(path) for path in paths),
        ],
        cwd=str(method_root),
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=600,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace")[-4000:]
        raise PairV5T2VScoreV3CompatibilityError(
            f"isolated d541801 formal-v3 validation failed: {stderr}"
        )
    try:
        rows = json.loads(completed.stdout)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PairV5T2VScoreV3CompatibilityError(
            "isolated d541801 validator emitted invalid JSON"
        ) from error
    if (
        not isinstance(rows, list)
        or len(rows) != 40
        or any(
            not isinstance(row, dict)
            or row.get("schema_version") != FORMAL_SCORE_SCHEMA
            for row in rows
        )
    ):
        raise PairV5T2VScoreV3CompatibilityError(
            "isolated validator output is not exact formal v3"
        )
    source_binding_unsigned = {
        "source_revision": formal_v3_source_revision,
        "source_archive_sha256": archive_sha,
        "critical_source_manifest": critical_manifest,
        "formal_score_schema": FORMAL_SCORE_SCHEMA,
        "formal_score_filename": FORMAL_SCORE_FILENAME,
        "receipt_validator_executed_in_isolated_python": True,
        "active_repository_scorer_imported_by_isolated_validator": False,
        "diagnostic_non_authorizing": True,
        "optimizer_authorized": False,
        "scientific_action_editing_claim": False,
    }
    source_binding = {
        **source_binding_unsigned,
        "binding_digest": object_sha256(source_binding_unsigned),
    }
    return rows, source_binding


def _tensor_sha256(value: Any) -> str:
    import torch

    if not isinstance(value, torch.Tensor) or value.device.type == "meta":
        raise PairV5T2VScoreV3CompatibilityError(
            "tensor hash requires a real tensor"
        )
    cpu = value.detach().to(device="cpu").contiguous().clone()
    metadata = {
        "shape": [int(item) for item in cpu.shape],
        "dtype": str(cpu.dtype),
        "layout": str(cpu.layout),
    }
    raw = cpu.view(torch.uint8).reshape(-1).numpy().tobytes()
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes(metadata))
    digest.update(b"\x00")
    digest.update(raw)
    return digest.hexdigest()


def make_native_v3_energy_packet(energy: Any) -> dict[str, Any]:
    """Recompute d541801's live FP32 log-difference from branch energies."""

    import torch

    branch = getattr(energy, "branch_energies", None)
    if (
        not isinstance(branch, torch.Tensor)
        or tuple(int(item) for item in branch.shape) != (len(BRANCH_ORDER), 1)
        or branch.dtype != torch.float32
        or branch.device.type == "meta"
        or branch.requires_grad
        or branch.grad_fn is not None
        or not bool(torch.isfinite(branch).all().item())
        or bool((branch < 0.0).any().item())
    ):
        raise PairV5T2VScoreV3CompatibilityError(
            "native v3 branch-energy tensor closure differs"
        )
    with torch.no_grad():
        # This expression and operation order are copied from d541801.  Do not
        # replace it with log1p, FP64, or Decimal arithmetic on the formal path.
        ratios = torch.log(branch[1:] + ENERGY_EPSILON) - torch.log(
            branch[:1] + ENERGY_EPSILON
        )
        reward, hardest = ratios.min(dim=0)
    if (
        ratios.dtype != torch.float32
        or reward.dtype != torch.float32
        or hardest.dtype != torch.int64
    ):
        raise PairV5T2VScoreV3CompatibilityError(
            "native v3 live arithmetic dtype differs"
        )
    proof_unsigned = {
        "branch_order": list(BRANCH_ORDER),
        "hard_negative_order": list(HARD_NEGATIVE_BRANCHES),
        "branch_energy_tensor_sha256": _tensor_sha256(branch),
        "negative_log_energy_ratio_tensor_sha256": _tensor_sha256(ratios),
        "reward_tensor_sha256": _tensor_sha256(reward),
        "hardest_negative_index_tensor_sha256": _tensor_sha256(hardest),
        "tensor_dtype": "torch.float32",
        "formula_recomputed_on_origin_device_bit_exact": True,
        "reward_and_first_argmin_recomputed_on_origin_device_bit_exact": True,
    }
    proof = {**proof_unsigned, "digest": object_sha256(proof_unsigned)}
    values = [float(item) for item in branch[:, 0].tolist()]
    ratio_by_branch = {
        name: float(ratios[index, 0].item())
        for index, name in enumerate(HARD_NEGATIVE_BRANCHES)
    }
    hardest_index = int(hardest.item())
    unsigned = {
        "definition": V3_SCALAR_DEFINITION,
        "energy_epsilon": float(ENERGY_EPSILON),
        "global_action_energy": values[0],
        "global_hard_negative_energy_by_branch": {
            name: values[index + 1]
            for index, name in enumerate(HARD_NEGATIVE_BRANCHES)
        },
        "global_negative_log_energy_ratio_by_branch": ratio_by_branch,
        "global_hardest_negative_branch": HARD_NEGATIVE_BRANCHES[hardest_index],
        "raw_global_action_energy_score": float(reward.item()),
        "mace_live_tensor_formula_proof": proof,
        "compatibility_source_revision": PINNED_SOURCE_REVISION,
        "compatibility_scalar_definition": V3_SCALAR_DEFINITION,
        "diagnostic_non_authorizing": True,
        "optimizer_authorized": False,
        "scientific_action_editing_claim": False,
    }
    packet = {**unsigned, "packet_digest": object_sha256(unsigned)}
    return validate_native_v3_energy_packet(packet)


def _exact_fp32(value: Any, *, label: str, nonnegative: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise PairV5T2VScoreV3CompatibilityError(f"{label} differs")
    try:
        roundtrip = struct.unpack("!f", struct.pack("!f", value))[0]
    except (OverflowError, struct.error) as error:
        raise PairV5T2VScoreV3CompatibilityError(
            f"{label} is not exact FP32"
        ) from error
    if float(roundtrip) != value:
        raise PairV5T2VScoreV3CompatibilityError(f"{label} is not exact FP32")
    if nonnegative and value < 0.0:
        raise PairV5T2VScoreV3CompatibilityError(f"{label} is negative")
    return value


def validate_native_v3_energy_packet(value: Any) -> dict[str, Any]:
    """Replay the d541801 scalar receipt without importing active v4 code."""

    import torch

    if not isinstance(value, Mapping) or set(value) != set(_PACKET_FIELDS):
        raise PairV5T2VScoreV3CompatibilityError(
            "native v3 energy packet field closure differs"
        )
    row = dict(value)
    unsigned = dict(row)
    declared = _sha256(
        unsigned.pop("packet_digest", None), label="native v3 packet digest"
    )
    if object_sha256(unsigned) != declared:
        raise PairV5T2VScoreV3CompatibilityError(
            "native v3 energy packet digest differs"
        )
    if (
        row["definition"] != V3_SCALAR_DEFINITION
        or row["compatibility_source_revision"] != PINNED_SOURCE_REVISION
        or row["compatibility_scalar_definition"] != V3_SCALAR_DEFINITION
        or row["diagnostic_non_authorizing"] is not True
        or row["optimizer_authorized"] is not False
        or row["scientific_action_editing_claim"] is not False
        or row["energy_epsilon"] != float(ENERGY_EPSILON)
    ):
        raise PairV5T2VScoreV3CompatibilityError(
            "native v3 scalar authority differs"
        )
    action = _exact_fp32(
        row["global_action_energy"], label="v3 action energy", nonnegative=True
    )
    negatives_raw = row["global_hard_negative_energy_by_branch"]
    ratios_raw = row["global_negative_log_energy_ratio_by_branch"]
    if (
        not isinstance(negatives_raw, Mapping)
        or set(negatives_raw) != set(HARD_NEGATIVE_BRANCHES)
        or not isinstance(ratios_raw, Mapping)
        or set(ratios_raw) != set(HARD_NEGATIVE_BRANCHES)
    ):
        raise PairV5T2VScoreV3CompatibilityError(
            "native v3 negative branch closure differs"
        )
    negatives = {
        name: _exact_fp32(
            negatives_raw[name], label=f"v3 {name} energy", nonnegative=True
        )
        for name in HARD_NEGATIVE_BRANCHES
    }
    ratios = {
        name: _exact_fp32(ratios_raw[name], label=f"v3 {name} ratio")
        for name in HARD_NEGATIVE_BRANCHES
    }
    reward = _exact_fp32(
        row["raw_global_action_energy_score"], label="v3 raw action score"
    )
    energy_tensor = torch.tensor(
        [action, *(negatives[name] for name in HARD_NEGATIVE_BRANCHES)],
        dtype=torch.float32,
    ).reshape(-1, 1)
    ratio_tensor = torch.tensor(
        [ratios[name] for name in HARD_NEGATIVE_BRANCHES], dtype=torch.float32
    ).reshape(-1, 1)
    reward_tensor = torch.tensor([reward], dtype=torch.float32)
    expected_reward, expected_index = ratio_tensor[:, 0].min(dim=0)
    index = int(expected_index.item())
    if (
        reward != float(expected_reward.item())
        or row["global_hardest_negative_branch"]
        != HARD_NEGATIVE_BRANCHES[index]
    ):
        raise PairV5T2VScoreV3CompatibilityError(
            "native v3 reward/first-argmin closure differs"
        )
    proof = row["mace_live_tensor_formula_proof"]
    if not isinstance(proof, Mapping) or set(proof) != set(_LIVE_PROOF_FIELDS):
        raise PairV5T2VScoreV3CompatibilityError(
            "native v3 live proof closure differs"
        )
    proof_unsigned = dict(proof)
    proof_digest = _sha256(
        proof_unsigned.pop("digest", None), label="native v3 live proof digest"
    )
    if (
        object_sha256(proof_unsigned) != proof_digest
        or proof["branch_order"] != list(BRANCH_ORDER)
        or proof["hard_negative_order"] != list(HARD_NEGATIVE_BRANCHES)
        or proof["tensor_dtype"] != "torch.float32"
        or proof["formula_recomputed_on_origin_device_bit_exact"] is not True
        or proof["reward_and_first_argmin_recomputed_on_origin_device_bit_exact"]
        is not True
        or proof["branch_energy_tensor_sha256"] != _tensor_sha256(energy_tensor)
        or proof["negative_log_energy_ratio_tensor_sha256"]
        != _tensor_sha256(ratio_tensor)
        or proof["reward_tensor_sha256"] != _tensor_sha256(reward_tensor)
        or proof["hardest_negative_index_tensor_sha256"]
        != _tensor_sha256(torch.tensor([index], dtype=torch.int64))
    ):
        raise PairV5T2VScoreV3CompatibilityError(
            "native v3 live proof binding differs"
        )
    cpu_formula = torch.log(energy_tensor[1:, 0] + ENERGY_EPSILON) - torch.log(
        energy_tensor[0] + ENERGY_EPSILON
    )
    if not torch.allclose(
        ratio_tensor[:, 0],
        cpu_formula,
        rtol=MACE_CROSS_DEVICE_REPLAY_RTOL,
        atol=MACE_CROSS_DEVICE_REPLAY_ATOL,
    ):
        raise PairV5T2VScoreV3CompatibilityError(
            "native v3 serialized ratio differs from auxiliary replay"
        )
    row["packet_digest"] = declared
    return row


__all__ = [
    "DIAGNOSTIC_NON_AUTHORIZING",
    "ENERGY_EPSILON",
    "FORMAL_SCORE_FILENAME",
    "FORMAL_SCORE_SCHEMA",
    "HARD_NEGATIVE_BRANCHES",
    "HISTORICAL_SCORE_FILENAME",
    "HISTORICAL_SCORE_SCHEMA",
    "PINNED_MACE_SOURCE_SHA256",
    "PINNED_SCORER_SOURCE_SHA256",
    "PINNED_SOURCE_REVISION",
    "PairV5T2VScoreV3CompatibilityError",
    "V3_SCALAR_DEFINITION",
    "make_native_v3_energy_packet",
    "validate_formal_score_receipts_isolated",
    "validate_native_v3_energy_packet",
]
