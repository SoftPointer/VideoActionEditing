#!/usr/bin/env python3
"""WORLD4/SP4 Phase-A CAGE cotangent probe on a sealed native RV2V candidate.

This is a read-only reward probe, not a trainer.  It combines exactly two
authorities:

* ``y`` and ``epsilon`` are the normalized pre-decode clean latent and the
  official initial Gaussian captured from one sealed native Bernini RV2V-4
  rollout; and
* the action plus nine hard-negative captions come from the matching cell of
  the sealed core4-v2 pure-T2V specification.

At native UniPC40 indices 20, 28, and 33 the probe first runs all ten frozen
T2V branches under ``no_grad`` (30 model calls).  It detaches the global worst
``(sigma, negative)`` cell and then replays only its action and negative calls
with a graph with respect to the noisy input (two model calls).  The resulting
candidate-clean cotangent includes both the denoiser input VJP and the direct
``epsilon-y`` flow-target derivative implemented by
``cage_candidate_action_energy_vjp``.

The old ``FrozenBerniniT2VScorer`` cannot be used here: it both supports only
the index-33 pilot and explicitly wraps patching/shared_step in ``no_grad``.
This module therefore reproduces only Bernini's native target-only packing,
exact discrete timestep, and spatial unpacking while keeping model parameters
frozen.  No pure-T2V MP4/latent/noise, target video, donor, mask, flow, pose,
track, or trajectory can enter the public tensor boundary.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import struct
import tempfile
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(METHOD_ROOT))

import torch
from torch import nn

import cage_candidate_action_energy_vjp as cage
import dclr_runtime_contract as runtime_contract
import infer_native_identity_generation_canary as native_generation
import mace_candidate_action_energy as mace
import pair_v5_native_rollout_spec as rollout_contract
import pair_v5_native_rv2v_action_score_v3 as native_score
import pair_v5_t2v_calibration_bank_spec as t2v_bank_contract
import score_pair_v5_t2v_energy_bank_v3 as frozen_runtime
import source_self_native_ref_contrastive_v3 as native_schedule


SCHEMA_VERSION = "bernini-cage-native-candidate-vjp-probe-v1"
ARTIFACT_SCHEMA = "bernini-cage-native-candidate-cotangent-v1"
RECEIPT_FILENAME = "cage-native-candidate-vjp-receipt.json"
COTANGENT_FILENAME = "cage-native-candidate-cotangent.safetensors"
COTANGENT_KEY = "candidate_action_cotangent"

WORLD_SIZE = 4
SP_SIZE = 4
PROBE_SCHEDULE_INDICES = (20, 28, 33)
PINNED_T2V_CORE4_V2_SHA256 = (
    "a18387b383fb11f19279c67694089754ff84b51e939e7a92b51a7e35a0743a95"
)
PINNED_NATIVE_CORE4_POPULATION_SHA256 = (
    "525d727951ee05d7aac27f47d294e3604996781106dfc710087d4029a1bbd8f0"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}")

FORBIDDEN_PUBLIC_INPUT_NAMES = frozenset(
    {
        "proposal",
        "proposal_media",
        "proposal_video",
        "proposal_latent",
        "proposal_noise",
        "t2v_video",
        "t2v_latent",
        "t2v_noise",
        "target",
        "target_video",
        "target_latent",
        "donor",
        "donor_video",
        "donor_latent",
        "mask",
        "motion_mask",
        "flow",
        "pose",
        "track",
        "trajectory",
    }
)


class CAGENativeCandidateVJPProbeError(RuntimeError):
    """The sealed Phase-A probe contract was not satisfied."""


@dataclass(frozen=True)
class World4SP4Contract:
    world_size: int
    rank: int
    local_rank: int
    local_world_size: int
    ulysses_size: int = SP_SIZE


@dataclass(frozen=True)
class NativeProbeCoordinate:
    schedule_index: int
    coordinate_id: str
    sigma: float
    native_timestep: int

    def receipt(self) -> dict[str, Any]:
        sigma_fp32 = struct.unpack("!f", struct.pack("!f", self.sigma))[0]
        value = {
            "schedule_name": "pinned_bernini_unipc40_flow_shift5",
            "schedule_digest": native_schedule.PINNED_NATIVE_UNIPC40_SCHEDULE_DIGEST,
            "schedule_index": self.schedule_index,
            "coordinate_id": self.coordinate_id,
            "physical_sigma_float64_hex": float(self.sigma).hex(),
            "physical_sigma_float32_be_hex": struct.pack(
                "!f", sigma_fp32
            ).hex(),
            "native_scheduler_timestep": self.native_timestep,
            "frozen_t2v_model_timestep": float(self.native_timestep),
            "frozen_t2v_model_timestep_float32_be_hex": struct.pack(
                "!f", float(self.native_timestep)
            ).hex(),
            "timestep_mapping": (
                "direct_native_unipc40_discrete_timestep_same_schedule_index"
            ),
            "legacy_1000_sigma_timestep_rejected": True,
        }
        return {**value, "digest": object_sha256(value)}


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise CAGENativeCandidateVJPProbeError(
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
        raise CAGENativeCandidateVJPProbeError(
            f"file changed while hashing: {path}"
        )
    return digest.hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CAGENativeCandidateVJPProbeError(
            f"{label} must be lowercase SHA-256"
        )
    return value


def _safe_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise CAGENativeCandidateVJPProbeError(f"{label} is not path-safe")
    return value


def _plain_file(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise CAGENativeCandidateVJPProbeError(
            f"{label} must be an absolute plain file"
        )
    return path.resolve(strict=True)


def _plain_directory(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_dir() or path.is_symlink():
        raise CAGENativeCandidateVJPProbeError(
            f"{label} must be an absolute plain directory"
        )
    return path.resolve(strict=True)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    raw = path.read_bytes()

    def reject_constant(token: str) -> None:
        raise CAGENativeCandidateVJPProbeError(
            f"{label} contains non-finite constant {token}"
        )

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CAGENativeCandidateVJPProbeError(
                    f"{label} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise CAGENativeCandidateVJPProbeError(
            f"{label} is not valid UTF-8 JSON"
        ) from error
    if not isinstance(value, dict):
        raise CAGENativeCandidateVJPProbeError(f"{label} root must be an object")
    return value


def world4_sp4_contract(
    environment: Mapping[str, str] = os.environ,
) -> World4SP4Contract:
    values: dict[str, int] = {}
    for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE"):
        try:
            values[name] = int(environment.get(name, ""))
        except (TypeError, ValueError) as error:
            raise CAGENativeCandidateVJPProbeError(
                f"invalid torchrun environment field {name}"
            ) from error
    if (
        values["WORLD_SIZE"] != WORLD_SIZE
        or values["LOCAL_WORLD_SIZE"] != WORLD_SIZE
    ):
        raise CAGENativeCandidateVJPProbeError(
            "CAGE Phase-A requires one exact WORLD4 node"
        )
    if (
        values["RANK"] != values["LOCAL_RANK"]
        or not 0 <= values["RANK"] < WORLD_SIZE
    ):
        raise CAGENativeCandidateVJPProbeError(
            "CAGE Phase-A rank/local-rank topology differs"
        )
    return World4SP4Contract(
        world_size=WORLD_SIZE,
        rank=values["RANK"],
        local_rank=values["LOCAL_RANK"],
        local_world_size=WORLD_SIZE,
    )


def require_sp4_object_consensus(
    value: Any,
    *,
    label: str,
    distributed_module: Any,
    world_size: int = WORLD_SIZE,
) -> Any:
    """Require exact replicated objects; never average an SP cotangent."""

    if world_size != WORLD_SIZE:
        raise CAGENativeCandidateVJPProbeError(
            f"{label} consensus requires exactly four ranks"
        )
    if (
        not callable(getattr(distributed_module, "get_world_size", None))
        or not callable(getattr(distributed_module, "all_gather_object", None))
        or int(distributed_module.get_world_size()) != WORLD_SIZE
    ):
        raise CAGENativeCandidateVJPProbeError(
            f"{label} distributed WORLD4 state differs"
        )
    rows: list[Any] = [None] * WORLD_SIZE
    distributed_module.all_gather_object(rows, value)
    if any(row != rows[0] for row in rows[1:]):
        raise CAGENativeCandidateVJPProbeError(
            f"{label} differs across SP4 ranks"
        )
    return rows[0]


def native_probe_coordinates() -> tuple[NativeProbeCoordinate, ...]:
    result = tuple(
        NativeProbeCoordinate(
            schedule_index=index,
            coordinate_id=f"native-unipc40-index-{index:02d}",
            sigma=float(native_schedule.NATIVE_UNIPC40_SIGMAS[index]),
            native_timestep=int(native_schedule.NATIVE_UNIPC40_TIMESTEPS[index]),
        )
        for index in PROBE_SCHEDULE_INDICES
    )
    if (
        tuple(item.native_timestep for item in result) != (833, 682, 516)
        or tuple(item.schedule_index for item in result)
        != PROBE_SCHEDULE_INDICES
        or len({item.coordinate_id for item in result}) != len(result)
    ):
        raise CAGENativeCandidateVJPProbeError(
            "pinned Phase-A native coordinate registry differs"
        )
    return result


def make_energy_coordinates(
    official_gaussian: torch.Tensor,
) -> tuple[cage.EnergyCoordinate, ...]:
    if (
        not isinstance(official_gaussian, torch.Tensor)
        or official_gaussian.dtype != torch.float32
        or official_gaussian.ndim != 5
        or tuple(int(item) for item in official_gaussian.shape[:3])
        != (1, 16, 21)
        or official_gaussian.requires_grad
        or official_gaussian.grad_fn is not None
        or not bool(torch.isfinite(official_gaussian).all().item())
    ):
        raise CAGENativeCandidateVJPProbeError(
            "official Gaussian must be detached FP32 exact81 [1,16,21,H,W]"
        )
    return tuple(
        cage.EnergyCoordinate(
            coordinate_id=item.coordinate_id,
            sigma=item.sigma,
            # The same candidate-own official draw is deliberately reused at
            # all three native schedule coordinates (common random numbers).
            epsilon=official_gaussian,
        )
        for item in native_probe_coordinates()
    )


def _family_order_from_t2v_spec(t2v_spec: Mapping[str, Any]) -> list[str]:
    result: list[str] = []
    for group in t2v_spec["groups"]:
        for candidate in group["candidates"]:
            family = candidate["action_family_id"]
            if family not in result:
                result.append(family)
    return result


def match_core4_native_candidate(
    population_spec: Mapping[str, Any],
    t2v_spec: Mapping[str, Any],
    *,
    candidate_id: str,
) -> dict[str, Any]:
    """Match one native action rollout to its core4-v2 action+9 cell."""

    selected_id = _safe_id(candidate_id, label="candidate_id")
    try:
        population = rollout_contract.validate_root_spec(population_spec)
        t2v = t2v_bank_contract.validate_root_spec(t2v_spec)
    except (
        rollout_contract.PairRolloutSpecError,
        t2v_bank_contract.PairT2VCalibrationSpecError,
    ) as error:
        raise CAGENativeCandidateVJPProbeError(str(error)) from error
    if t2v.get("schema_version") != t2v_bank_contract.SCHEMA_VERSION_V2:
        raise CAGENativeCandidateVJPProbeError(
            "Phase-A requires the formal core4-v2 T2V spec"
        )
    t2v_ids = [
        row["candidate_id"]
        for group in t2v["groups"]
        for row in group["candidates"]
    ]
    population_ids = [
        row["candidate_id"]
        for group in population["groups"]
        for row in group["candidates"]
    ]
    if (
        len(t2v_ids) != 40
        or not all(value.startswith("pair5-t2v-core4-v2-") for value in t2v_ids)
        or len(population_ids) != 8
        or not all(
            value.startswith("pair5-native-core4-v1-")
            for value in population_ids
        )
    ):
        raise CAGENativeCandidateVJPProbeError(
            "sealed core4 native/T2V population closure differs"
        )
    try:
        binding = native_score.bind_population_to_calibration(
            population,
            t2v,
            calibration_family_order=_family_order_from_t2v_spec(t2v),
        )
    except native_score.PairV5NativeRV2VActionScoreError as error:
        raise CAGENativeCandidateVJPProbeError(str(error)) from error
    rows = [
        row
        for row in binding["bound_rows"]
        if row["candidate"]["candidate_id"] == selected_id
    ]
    if len(rows) != 1:
        raise CAGENativeCandidateVJPProbeError(
            "candidate_id does not select exactly one sealed native core4 row"
        )
    row = rows[0]
    cell = row["cell"]
    branches = [item["semantic_branch"] for item in cell["rows"]]
    if (
        branches != list(mace.BRANCH_ORDER)
        or cell["action_candidate"]["full_t2v_caption"]
        != row["candidate"]["complete_caption"]
        or cell["action_candidate"]["full_t2v_caption_utf8_sha256"]
        != row["candidate"]["complete_caption_sha256"]
        or population["semantic_input_closure"].get("t2v_proposal_media")
        is not False
        or t2v["semantic_input_closure"].get("proposal_media_as_condition")
        is not False
        or t2v["semantic_input_closure"].get("proposal_media_as_student_input")
        is not False
        or t2v["semantic_input_closure"].get("proposal_media_as_noise")
        is not False
        or t2v["semantic_input_closure"].get(
            "proposal_media_as_donor_or_pseudo_target"
        )
        is not False
    ):
        raise CAGENativeCandidateVJPProbeError(
            "core4 prompt cell or proposal-media closure differs"
        )
    return {
        **row,
        "population_binding": binding,
        "caption_by_branch": {
            branch: cell["caption_by_branch"][branch]
            for branch in mace.BRANCH_ORDER
        },
        "caption_sha256_by_branch": {
            branch: cell["caption_sha256_by_branch"][branch]
            for branch in mace.BRANCH_ORDER
        },
    }


def _load_sealed_probe_specs(
    *,
    population_spec_path: str | Path,
    population_spec_sha256: str,
    t2v_spec_path: str | Path,
    t2v_spec_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    population_digest = _sha256(
        population_spec_sha256, label="native population spec SHA-256"
    )
    t2v_digest = _sha256(t2v_spec_sha256, label="T2V core4-v2 spec SHA-256")
    if population_digest != PINNED_NATIVE_CORE4_POPULATION_SHA256:
        raise CAGENativeCandidateVJPProbeError(
            "native population is not the pinned core4 action population"
        )
    if t2v_digest != PINNED_T2V_CORE4_V2_SHA256:
        raise CAGENativeCandidateVJPProbeError(
            "T2V spec is not the pinned core4-v2 prompt authority"
        )
    population_path = _plain_file(
        population_spec_path, label="native population spec"
    )
    t2v_path = _plain_file(t2v_spec_path, label="T2V core4-v2 spec")
    try:
        population, observed_population = rollout_contract.load_sealed_spec(
            population_path, population_digest
        )
        t2v, observed_t2v = t2v_bank_contract.load_sealed_spec(
            t2v_path, t2v_digest
        )
    except (
        rollout_contract.PairRolloutSpecError,
        t2v_bank_contract.PairT2VCalibrationSpecError,
    ) as error:
        raise CAGENativeCandidateVJPProbeError(str(error)) from error
    if (
        observed_population != population_digest
        or observed_t2v != t2v_digest
    ):
        raise CAGENativeCandidateVJPProbeError("sealed spec digest differs")
    return population, t2v, population_path, t2v_path


def load_sealed_native_candidate(
    *,
    population_spec_path: str | Path,
    population_spec_sha256: str,
    t2v_spec_path: str | Path,
    t2v_spec_sha256: str,
    rollout_root: str | Path,
    candidate_id: str,
    checkpoint_tree_sha256: str,
) -> dict[str, Any]:
    """Authenticate one RV2V clean/Gaussian pair and matching prompt cell."""

    population, t2v, population_path, t2v_path = _load_sealed_probe_specs(
        population_spec_path=population_spec_path,
        population_spec_sha256=population_spec_sha256,
        t2v_spec_path=t2v_spec_path,
        t2v_spec_sha256=t2v_spec_sha256,
    )
    row = match_core4_native_candidate(
        population, t2v, candidate_id=candidate_id
    )
    checkpoint_tree = _sha256(
        checkpoint_tree_sha256, label="checkpoint tree SHA-256"
    )
    root = _plain_directory(rollout_root, label="native rollout root")
    candidate = row["candidate"]
    candidate_dir = root / candidate["candidate_id"]
    if (
        not candidate_dir.is_dir()
        or candidate_dir.is_symlink()
        or candidate_dir.parent != root
    ):
        raise CAGENativeCandidateVJPProbeError(
            "native candidate rollout directory differs"
        )
    pair_path = _plain_file(
        candidate_dir / "pair-v5-rollout-receipt.json",
        label="PAIR native rollout receipt",
    )
    pair_receipt = _read_json(pair_path, label="PAIR native rollout receipt")
    try:
        native_score._closed(
            pair_receipt,
            native_score._PAIR_ROLLOUT_FIELDS,
            label="PAIR native rollout receipt",
        )
        pair_digest = native_score._verify_embedded_with_canonicalizer(
            pair_receipt,
            field="receipt_digest",
            label="PAIR native rollout receipt",
            canonicalizer=rollout_contract.canonical_json_bytes,
        )
        expected_envelope = native_score._expected_candidate_envelope_sha256(
            row=row, root_spec_sha256=population_spec_sha256
        )
    except native_score.PairV5NativeRV2VActionScoreError as error:
        raise CAGENativeCandidateVJPProbeError(str(error)) from error
    expected_topology = {
        "world_size": WORLD_SIZE,
        "ulysses_size": SP_SIZE,
        "rocr_visible_devices": ",".join(str(item) for item in row["visible_gpus"]),
    }
    if (
        pair_receipt.get("schema_version")
        != rollout_contract.RECEIPT_SCHEMA_VERSION
        or pair_receipt.get("root_spec_raw_sha256")
        != population_spec_sha256
        or pair_receipt.get("candidate_envelope_sha256") != expected_envelope
        or pair_receipt.get("group_id") != row["group_id"]
        or pair_receipt.get("visible_gpus") != row["visible_gpus"]
        or pair_receipt.get("runtime_topology") != expected_topology
        or pair_receipt.get("ordinal") != row["ordinal"]
        or pair_receipt.get("candidate") != candidate
        or pair_receipt.get("sampling_contract")
        != rollout_contract.SAMPLING_CONTRACT
        or pair_receipt.get("semantic_input_closure")
        != rollout_contract.SEMANTIC_INPUT_CLOSURE
    ):
        raise CAGENativeCandidateVJPProbeError(
            "PAIR native rollout/spec binding differs"
        )
    native_path = _plain_file(
        pair_receipt.get("native_receipt_path"), label="native RV2V receipt"
    )
    if native_path != candidate_dir / "receipt.json":
        raise CAGENativeCandidateVJPProbeError(
            "native RV2V receipt escaped candidate directory"
        )
    native_file_digest = file_sha256(native_path)
    if native_file_digest != pair_receipt.get("native_receipt_sha256"):
        raise CAGENativeCandidateVJPProbeError(
            "native RV2V receipt file hash differs"
        )
    native_receipt = _read_json(native_path, label="native RV2V receipt")
    try:
        native_artifacts = native_score._verify_native_rv2v_receipt(
            native_receipt,
            candidate=candidate,
            checkpoint_tree_sha256=checkpoint_tree,
            candidate_dir=candidate_dir,
        )
    except native_score.PairV5NativeRV2VActionScoreError as error:
        raise CAGENativeCandidateVJPProbeError(str(error)) from error
    expected_artifacts = {
        "mp4": native_artifacts["mp4"],
        "predecode_clean_latent": native_artifacts["predecode_clean_latent"],
        "official_initial_gaussian": native_artifacts[
            "official_initial_gaussian"
        ],
    }
    if (
        pair_receipt.get("native_receipt_digest")
        != native_artifacts["native_receipt_digest"]
        or pair_receipt.get("artifacts") != expected_artifacts
    ):
        raise CAGENativeCandidateVJPProbeError(
            "PAIR/native artifact binding differs"
        )
    source = _plain_file(candidate["source_video"], label="native source video")
    if file_sha256(source) != candidate["source_video_sha256"]:
        raise CAGENativeCandidateVJPProbeError("native source video hash differs")
    return {
        **row,
        "population_spec_path": str(population_path),
        "population_spec_sha256": population_spec_sha256,
        "t2v_spec_path": str(t2v_path),
        "t2v_spec_sha256": t2v_spec_sha256,
        "rollout_root": str(root),
        "pair_receipt_path": str(pair_path),
        "pair_receipt_file_sha256": file_sha256(pair_path),
        "pair_receipt_digest": pair_digest,
        "native_receipt_path": str(native_path),
        "native_receipt_file_sha256": native_file_digest,
        "native_receipt": native_receipt,
        "native_artifacts": native_artifacts,
        "source_video_path": str(source),
    }


def _patch_size(transformer: nn.Module) -> tuple[int, int, int]:
    config = getattr(transformer, "config", None)
    raw = getattr(config, "patch_size", (1, 2, 2))
    try:
        result = tuple(int(item) for item in raw)
    except (TypeError, ValueError) as error:
        raise CAGENativeCandidateVJPProbeError(
            "transformer patch_size is invalid"
        ) from error
    if result != (1, 2, 2):
        raise CAGENativeCandidateVJPProbeError(
            "Phase-A requires Bernini Wan patch_size=(1,2,2)"
        )
    return result


def _unpack_target_prediction(
    packed: torch.Tensor,
    *,
    reference_shape: Sequence[int],
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    shape = tuple(int(item) for item in reference_shape)
    if len(shape) != 5 or shape[:3] != (1, 16, 21):
        raise CAGENativeCandidateVJPProbeError(
            "candidate reference is not exact81 latent geometry"
        )
    batch, channels, frames, height, width = shape
    pt, ph, pw = patch_size
    if frames % pt or height % ph or width % pw:
        raise CAGENativeCandidateVJPProbeError(
            "candidate geometry is not patch divisible"
        )
    tp, hp, wp = frames // pt, height // ph, width // pw
    expected = (batch, tp * hp * wp, pt * ph * pw * channels)
    if not isinstance(packed, torch.Tensor) or tuple(packed.shape) != expected:
        raise CAGENativeCandidateVJPProbeError(
            f"packed T2V prediction shape differs: expected {expected}"
        )
    return (
        packed.reshape(batch, tp, hp, wp, pt, ph, pw, channels)
        .permute(0, 7, 1, 4, 2, 5, 3, 6)
        .reshape(shape)
        .contiguous()
    )


class NativeMultiSigmaFrozenT2VInputVJPBridge(nn.Module):
    """Frozen target-only Bernini bridge with a graph only for input replay."""

    def __init__(
        self,
        diffusion: nn.Module,
        transformer: nn.Module,
        prompt_by_branch: Mapping[str, str],
        condition_by_branch: Mapping[str, torch.Tensor],
        *,
        frozen_model_receipt_digest: str,
        model_id: str = "transformer_1",
    ) -> None:
        super().__init__()
        if (
            not isinstance(diffusion, nn.Module)
            or not isinstance(transformer, nn.Module)
            or not callable(getattr(diffusion, "shared_step", None))
            or not callable(getattr(transformer, "patch_vae_latent", None))
        ):
            raise CAGENativeCandidateVJPProbeError(
                "bridge requires Bernini diffusion/shared_step and transformer patching"
            )
        if model_id != "transformer_1":
            raise CAGENativeCandidateVJPProbeError(
                "Phase-A supports the frozen Bernini 1.3B transformer_1 only"
            )
        if any(parameter.requires_grad for parameter in diffusion.parameters()):
            raise CAGENativeCandidateVJPProbeError(
                "frozen T2V bridge contains trainable parameters"
            )
        try:
            prompts = mace.validate_prompt_closure(prompt_by_branch)
        except mace.MACECandidateActionEnergyError as error:
            raise CAGENativeCandidateVJPProbeError(str(error)) from error
        if not isinstance(condition_by_branch, Mapping) or set(
            condition_by_branch
        ) != set(mace.BRANCH_ORDER):
            raise CAGENativeCandidateVJPProbeError(
                "text condition branch closure differs"
            )
        conditions: dict[str, torch.Tensor] = {}
        for branch in mace.BRANCH_ORDER:
            value = condition_by_branch[branch]
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != (1, 512, 4096)
                or value.dtype
                not in (torch.float16, torch.bfloat16, torch.float32)
                or value.requires_grad
                or value.grad_fn is not None
                or not bool(torch.isfinite(value).all().item())
            ):
                raise CAGENativeCandidateVJPProbeError(
                    f"condition {branch} is not frozen [1,512,4096]"
                )
            # The existing no-grad scorer encodes prompts under
            # ``torch.inference_mode``.  Such tensors are numerically valid,
            # but autograd may not save them for replay backward (attention
            # needs frozen K/V to differentiate with respect to its query).
            # Clone only inference tensors outside inference mode.  This
            # preserves exact values and a frozen condition while removing
            # the inference-only tensor flag.
            normal_value = value.clone() if torch.is_inference(value) else value
            if (
                torch.is_inference(normal_value)
                or normal_value.requires_grad
                or normal_value.grad_fn is not None
                or not torch.equal(normal_value, value)
            ):
                raise CAGENativeCandidateVJPProbeError(
                    f"condition {branch} could not enter the input-VJP graph safely"
                )
            conditions[branch] = normal_value
        self.diffusion = diffusion
        self.transformer = transformer
        self.model_id = model_id
        self._prompts = prompts
        self._branch_by_prompt = {
            prompt: branch for branch, prompt in prompts.items()
        }
        self._conditions = conditions
        self._coordinates = native_probe_coordinates()
        self._coordinate_by_id = {
            item.coordinate_id: item for item in self._coordinates
        }
        self._frozen_model_receipt_digest = _sha256(
            frozen_model_receipt_digest,
            label="frozen model receipt digest",
        )
        self._patch_size = _patch_size(transformer)
        self._scan_coordinate_position = 0
        self._scan_branch_position = 0
        self._scan_x_key: Optional[tuple[int, int]] = None
        self._scan_tokens: Optional[torch.Tensor] = None
        self._scan_rotary: Optional[torch.Tensor] = None
        self._scan_shape: Optional[tuple[int, ...]] = None
        self._events: list[dict[str, Any]] = []
        self._patch_events: list[dict[str, Any]] = []
        self.eval()

    @property
    def events(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(dict(item) for item in self._events)

    def _coordinate(self, request: cage.DenoiseRequest) -> NativeProbeCoordinate:
        coordinate = self._coordinate_by_id.get(request.coordinate_id)
        if (
            coordinate is None
            or request.coordinate_index != self._coordinates.index(coordinate)
            or struct.pack("!f", float(request.sigma))
            != struct.pack("!f", float(coordinate.sigma))
        ):
            raise CAGENativeCandidateVJPProbeError(
                "CAGE request/native schedule coordinate binding differs"
            )
        return coordinate

    @staticmethod
    def _validate_x(value: Any, *, replay: bool) -> torch.Tensor:
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or value.ndim != 5
            or tuple(int(item) for item in value.shape[:3]) != (1, 16, 21)
            or int(value.shape[3]) <= 0
            or int(value.shape[4]) <= 0
            or int(value.shape[3]) % 2
            or int(value.shape[4]) % 2
            or not bool(torch.isfinite(value).all().item())
            or bool(value.requires_grad) is not replay
        ):
            raise CAGENativeCandidateVJPProbeError(
                "T2V input must be finite FP32 exact81 with mode-correct graph state"
            )
        return value

    def _patch(
        self, x_sigma: torch.Tensor, *, replay: bool, request: cage.DenoiseRequest
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = getattr(self.transformer, "dtype", None)
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise CAGENativeCandidateVJPProbeError(
                "transformer exposes no supported dtype"
            )
        result = self.transformer.patch_vae_latent(
            x_sigma.to(dtype=dtype), source_id=0
        )
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise CAGENativeCandidateVJPProbeError(
                "patch_vae_latent must return target tokens and rotary"
            )
        tokens, rotary = result
        expected_tokens = int(x_sigma.shape[2]) * (
            int(x_sigma.shape[3]) // 2
        ) * (int(x_sigma.shape[4]) // 2)
        if (
            not isinstance(tokens, torch.Tensor)
            or tuple(tokens.shape) != (1, expected_tokens, 1536)
            or tokens.device != x_sigma.device
            or not tokens.is_floating_point()
            or bool(tokens.requires_grad) is not replay
            or (replay and tokens.grad_fn is None)
            or not bool(torch.isfinite(tokens).all().item())
            or not isinstance(rotary, torch.Tensor)
            or tuple(rotary.shape) != (1, 1, expected_tokens, 64)
            or rotary.dtype != torch.complex128
            or rotary.device != x_sigma.device
            or not bool(torch.isfinite(rotary).all().item())
        ):
            raise CAGENativeCandidateVJPProbeError(
                "native T2V patch tensor/rotary contract differs"
            )
        self._patch_events.append(
            {
                "mode": request.mode,
                "coordinate_index": request.coordinate_index,
                "branch": request.branch,
                "grad_enabled": bool(torch.is_grad_enabled()),
                "input_requires_grad": bool(x_sigma.requires_grad),
            }
        )
        return tokens, rotary

    def _shared_step(
        self,
        *,
        request: cage.DenoiseRequest,
        coordinate: NativeProbeCoordinate,
        tokens: torch.Tensor,
        rotary: torch.Tensor,
        reference_shape: Sequence[int],
        replay: bool,
    ) -> torch.Tensor:
        timestep = torch.tensor(
            [float(coordinate.native_timestep)],
            dtype=torch.float32,
            device=tokens.device,
        )
        condition = self._conditions[request.branch]
        if condition.device != tokens.device:
            raise CAGENativeCandidateVJPProbeError(
                "text condition and candidate devices differ"
            )
        prediction = self.diffusion.shared_step(
            model_id=self.model_id,
            noisy_latents=tokens,
            timesteps=timestep,
            cond_embeds=condition,
            rotary_embs=rotary,
            batch_vae_seqlen=[int(tokens.shape[1])],
            batch_text_seqlen=[runtime_contract.PINNED_TEXT_TOKENS],
        )
        if (
            not isinstance(prediction, torch.Tensor)
            or tuple(prediction.shape)
            != (1, int(tokens.shape[1]), runtime_contract.PINNED_PATCH_DIM)
            or prediction.device != tokens.device
            or prediction.dtype
            not in (torch.float16, torch.bfloat16, torch.float32)
            or bool(prediction.requires_grad) is not replay
            or (replay and prediction.grad_fn is None)
            or not bool(torch.isfinite(prediction).all().item())
        ):
            raise CAGENativeCandidateVJPProbeError(
                "native frozen T2V shared_step output graph/shape differs"
            )
        spatial = _unpack_target_prediction(
            prediction,
            reference_shape=reference_shape,
            patch_size=self._patch_size,
        )
        if bool(spatial.requires_grad) is not replay:
            raise CAGENativeCandidateVJPProbeError(
                "spatial prediction graph state differs"
            )
        self._events.append(
            {
                "mode": request.mode,
                "coordinate_index": request.coordinate_index,
                "coordinate_id": request.coordinate_id,
                "schedule_index": coordinate.schedule_index,
                "branch": request.branch,
                "native_timestep": coordinate.native_timestep,
                "sigma_float32_be_hex": struct.pack(
                    "!f", float(coordinate.sigma)
                ).hex(),
                "grad_enabled": bool(torch.is_grad_enabled()),
                "input_requires_grad": bool(request.x_sigma.requires_grad),
                "prediction_requires_grad": bool(spatial.requires_grad),
            }
        )
        return spatial

    def _scan(self, request: cage.DenoiseRequest) -> torch.Tensor:
        if request.coordinate_index != self._scan_coordinate_position:
            raise CAGENativeCandidateVJPProbeError(
                "no-grad scan coordinate order differs"
            )
        expected_branch = mace.BRANCH_ORDER[self._scan_branch_position]
        if request.branch != expected_branch:
            raise CAGENativeCandidateVJPProbeError(
                f"no-grad scan expected branch {expected_branch}"
            )
        coordinate = self._coordinate(request)
        x_sigma = self._validate_x(request.x_sigma, replay=False)
        key = (id(x_sigma), int(x_sigma._version))
        if self._scan_branch_position == 0:
            tokens, rotary = self._patch(x_sigma, replay=False, request=request)
            self._scan_x_key = key
            self._scan_tokens = tokens
            self._scan_rotary = rotary
            self._scan_shape = tuple(int(item) for item in x_sigma.shape)
        elif key != self._scan_x_key:
            raise CAGENativeCandidateVJPProbeError(
                "all ten scan branches must reuse one x_sigma object"
            )
        if (
            self._scan_tokens is None
            or self._scan_rotary is None
            or self._scan_shape is None
        ):
            raise CAGENativeCandidateVJPProbeError(
                "no-grad scan packet is uninitialized"
            )
        result = self._shared_step(
            request=request,
            coordinate=coordinate,
            tokens=self._scan_tokens,
            rotary=self._scan_rotary,
            reference_shape=self._scan_shape,
            replay=False,
        )
        self._scan_branch_position += 1
        if self._scan_branch_position == len(mace.BRANCH_ORDER):
            self._scan_branch_position = 0
            self._scan_coordinate_position += 1
            self._scan_x_key = None
            self._scan_tokens = None
            self._scan_rotary = None
            self._scan_shape = None
        return result

    def _replay(self, request: cage.DenoiseRequest) -> torch.Tensor:
        coordinate = self._coordinate(request)
        x_sigma = self._validate_x(request.x_sigma, replay=True)
        tokens, rotary = self._patch(x_sigma, replay=True, request=request)
        return self._shared_step(
            request=request,
            coordinate=coordinate,
            tokens=tokens,
            rotary=rotary,
            reference_shape=tuple(int(item) for item in x_sigma.shape),
            replay=True,
        )

    def forward(self, request: cage.DenoiseRequest) -> torch.Tensor:
        if not isinstance(request, cage.DenoiseRequest):
            raise CAGENativeCandidateVJPProbeError(
                "bridge accepts only a CAGE DenoiseRequest"
            )
        branch = self._branch_by_prompt.get(request.prompt)
        if branch != request.branch or branch not in mace.BRANCH_ORDER:
            raise CAGENativeCandidateVJPProbeError(
                "request prompt/semantic branch binding differs"
            )
        if request.mode == cage.SCAN_MODE:
            return self._scan(request)
        if request.mode == cage.REPLAY_MODE:
            return self._replay(request)
        raise CAGENativeCandidateVJPProbeError(
            "native Phase-A bridge accepts only scan or selected replay"
        )

    def execution_receipt(
        self, result: cage.CandidateActionEnergyVJPResult
    ) -> dict[str, Any]:
        if not isinstance(result, cage.CandidateActionEnergyVJPResult):
            raise CAGENativeCandidateVJPProbeError(
                "execution receipt requires a CAGE VJP result"
            )
        scan_events = [
            item for item in self._events if item["mode"] == cage.SCAN_MODE
        ]
        replay_events = [
            item for item in self._events if item["mode"] == cage.REPLAY_MODE
        ]
        scan_patch_events = [
            item
            for item in self._patch_events
            if item["mode"] == cage.SCAN_MODE
        ]
        replay_patch_events = [
            item
            for item in self._patch_events
            if item["mode"] == cage.REPLAY_MODE
        ]
        expected_scan = [
            (coordinate_index, branch)
            for coordinate_index in range(len(self._coordinates))
            for branch in mace.BRANCH_ORDER
        ]
        actual_scan = [
            (int(item["coordinate_index"]), str(item["branch"]))
            for item in scan_events
        ]
        expected_replay = [
            (
                result.scan.selected_coordinate_index,
                mace.ACTION_BRANCH,
            ),
            (
                result.scan.selected_coordinate_index,
                result.scan.selected_negative_branch,
            ),
        ]
        actual_replay = [
            (int(item["coordinate_index"]), str(item["branch"]))
            for item in replay_events
        ]
        if (
            self._scan_coordinate_position != len(self._coordinates)
            or self._scan_branch_position != 0
            or actual_scan != expected_scan
            or actual_replay != expected_replay
            or len(scan_patch_events) != len(self._coordinates)
            or len(replay_patch_events) != 2
            or any(item["grad_enabled"] for item in scan_events)
            or any(item["input_requires_grad"] for item in scan_events)
            or any(item["prediction_requires_grad"] for item in scan_events)
            or any(not item["grad_enabled"] for item in replay_events)
            or any(not item["input_requires_grad"] for item in replay_events)
            or any(not item["prediction_requires_grad"] for item in replay_events)
        ):
            raise CAGENativeCandidateVJPProbeError(
                "30-scan/2-replay native execution contract differs"
            )
        value = {
            "frozen_model_receipt_digest": self._frozen_model_receipt_digest,
            "model_id": self.model_id,
            "branch_order": list(mace.BRANCH_ORDER),
            "hard_negative_order": list(mace.HARD_NEGATIVE_BRANCHES),
            "coordinates": [item.receipt() for item in self._coordinates],
            "scan_shared_step_calls": len(scan_events),
            "scan_patch_vae_latent_calls": len(scan_patch_events),
            "scan_all_branches_all_sigmas_under_no_grad": True,
            "scan_shared_x_sigma_per_ten_branches": True,
            "replay_shared_step_calls": len(replay_events),
            "replay_patch_vae_latent_calls": len(replay_patch_events),
            "replay_branches": [item[1] for item in expected_replay],
            "replay_coordinate_index": result.scan.selected_coordinate_index,
            "replay_input_graph_enabled": True,
            "existing_no_grad_scorer_bypassed": True,
            "native_target_only_source_id": 0,
            "native_timestep_not_1000_sigma": True,
            "pure_t2v_proposal_media_consumed": False,
        }
        return {**value, "digest": object_sha256(value)}


def compute_native_candidate_cotangent(
    normalized_clean_latent: torch.Tensor,
    official_initial_gaussian: torch.Tensor,
    prompt_by_branch: Mapping[str, str],
    bridge: NativeMultiSigmaFrozenT2VInputVJPBridge,
    *,
    config: Optional[cage.EnergyVJPConfig] = None,
) -> tuple[cage.CandidateActionEnergyVJPResult, dict[str, Any]]:
    """Run the exact 30-call scan followed by the selected two input VJPs."""

    if (
        not isinstance(normalized_clean_latent, torch.Tensor)
        or normalized_clean_latent.dtype != torch.float32
        or normalized_clean_latent.ndim != 5
        or tuple(int(item) for item in normalized_clean_latent.shape[:3])
        != (1, 16, 21)
        or normalized_clean_latent.requires_grad
        or normalized_clean_latent.grad_fn is not None
        or not bool(torch.isfinite(normalized_clean_latent).all().item())
        or not isinstance(bridge, NativeMultiSigmaFrozenT2VInputVJPBridge)
    ):
        raise CAGENativeCandidateVJPProbeError(
            "clean latent/bridge Phase-A input contract differs"
        )
    if normalized_clean_latent.shape != official_initial_gaussian.shape:
        raise CAGENativeCandidateVJPProbeError(
            "clean latent and official Gaussian geometry differs"
        )
    coordinates = make_energy_coordinates(official_initial_gaussian)
    try:
        result = cage.compute_candidate_action_energy_vjp(
            normalized_clean_latent,
            coordinates,
            prompt_by_branch,
            bridge,
            config=config,
        )
    except cage.CAGECandidateActionEnergyVJPError as error:
        raise CAGENativeCandidateVJPProbeError(str(error)) from error
    return result, bridge.execution_receipt(result)


def verify_authenticated_native_clean_tensor_identity(
    value: torch.Tensor,
    artifact: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    """Bind one loaded clean latent to its authenticated safetensors file.

    The pre-decode clean-latent receipt predates raw/content value fields.  It
    authenticates the complete single-tensor container by SHA-256 and records
    an exact native coordinate/role plus a byte-exact FP32 save/reopen round
    trip.  Treating its absent value fields as expected ``None`` values is an
    API error, not evidence of corruption.

    This verifier reopens that authenticated container, checks the single key,
    exact value, metadata and native provenance, then seals newly recomputed
    raw/content identities as observed evidence.  The Gaussian uses the
    separate full recorded-value identity verifier and never enters here.
    """

    from safetensors import safe_open

    tensor_key = "normalized_clean_latent"
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.float32
        or value.ndim != 5
        or tuple(int(item) for item in value.shape[:3]) != (1, 16, 21)
        or value.requires_grad
        or value.grad_fn is not None
        or not value.is_contiguous()
        or not bool(torch.isfinite(value).all().item())
        or not isinstance(artifact, Mapping)
    ):
        raise CAGENativeCandidateVJPProbeError(
            f"{label} must be one detached contiguous CPU FP32 exact81 tensor"
        )
    expected_artifact_fields = {
        "artifact_role",
        "coordinate",
        "mp4_decode_reencode_used",
        "native_sampler_before_vae_decode",
        "origin",
        "path",
        "roundtrip_byte_exact_fp32",
        "sampler_return_dtype",
        "sha256",
        "shape",
        "source_video_vae_encode_before_any_decode",
        "stored_dtype",
        "tensor_key",
    }
    if set(artifact) != expected_artifact_fields:
        raise CAGENativeCandidateVJPProbeError(
            f"{label} native artifact field closure differs"
        )
    raw_path = artifact.get("path")
    if not isinstance(raw_path, (str, Path)):
        raise CAGENativeCandidateVJPProbeError(
            f"{label} artifact path differs"
        )
    path = _plain_file(raw_path, label=f"{label} artifact")
    container_sha256 = _sha256(
        artifact.get("sha256"), label=f"{label} container SHA-256"
    )
    if file_sha256(path) != container_sha256:
        raise CAGENativeCandidateVJPProbeError(
            f"{label} authenticated container SHA-256 differs"
        )
    with safe_open(str(path), framework="pt", device="cpu") as opened:
        if list(opened.keys()) != [tensor_key]:
            raise CAGENativeCandidateVJPProbeError(
                f"{label} authenticated container key closure differs"
            )
        stored = opened.get_tensor(tensor_key).contiguous()
        metadata = dict(opened.metadata() or {})
    if file_sha256(path) != container_sha256:
        raise CAGENativeCandidateVJPProbeError(
            f"{label} authenticated container changed while reopening"
        )
    if (
        stored.dtype != torch.float32
        or stored.shape != value.shape
        or not torch.equal(stored, value)
    ):
        raise CAGENativeCandidateVJPProbeError(
            f"{label} loaded value differs from authenticated container"
        )
    try:
        actual = frozen_runtime.native_tensor_value_identity(value)
        reopened = frozen_runtime.native_tensor_value_identity(stored)
    except frozen_runtime.PairV5T2VEnergyScoringError as error:
        raise CAGENativeCandidateVJPProbeError(str(error)) from error
    expected_identity_keys = {
        "shape",
        "dtype",
        "numel",
        "byte_count",
        "raw_value_sha256",
        "content_sha256",
    }
    if (
        set(actual) != expected_identity_keys
        or set(reopened) != expected_identity_keys
        or reopened != actual
        or actual.get("shape") != artifact.get("shape")
        or actual.get("dtype") != artifact.get("stored_dtype")
        or _SHA256_RE.fullmatch(str(actual.get("raw_value_sha256"))) is None
        or _SHA256_RE.fullmatch(str(actual.get("content_sha256"))) is None
    ):
        raise CAGENativeCandidateVJPProbeError(
            f"{label} actual tensor/container metadata differs"
        )

    def require_field(field: str, expected: Any) -> None:
        recorded = artifact.get(field)
        if recorded != expected:
            raise CAGENativeCandidateVJPProbeError(
                f"{label} artifact field {field} differs: "
                f"recorded={recorded!r} actual={expected!r}"
            )

    require_field("tensor_key", tensor_key)
    clean_fields = {
        "stored_dtype": "torch.float32",
        "sampler_return_dtype": "torch.float32",
        "coordinate": "bernini_normalized_clean_vae_latent",
        "artifact_role": "native_sampler_proposal",
        "origin": "native_sampler_before_vae_decode",
        "native_sampler_before_vae_decode": True,
        "source_video_vae_encode_before_any_decode": False,
        "mp4_decode_reencode_used": False,
        "roundtrip_byte_exact_fp32": True,
    }
    expected_metadata = {
        "coordinate": "bernini_normalized_clean_vae_latent",
        "frame_contract": "exact81_latent21",
        "artifact_role": "native_sampler_proposal",
        "source": "native_sampler_before_vae_decode",
    }
    for field, expected in clean_fields.items():
        require_field(field, expected)
    if metadata != expected_metadata:
        raise CAGENativeCandidateVJPProbeError(
            f"{label} safetensors metadata differs"
        )
    binding = {
        **actual,
        "authenticated_container_path": str(path),
        "authenticated_container_sha256": container_sha256,
        "single_tensor_container_reopened_byte_exact": True,
        "safetensors_metadata": metadata,
        "recorded_value_hashes_present": False,
        "native_receipt_value_hashes_synthesized": False,
        "observed_value_hashes_recomputed_after_authenticated_reopen": True,
        "identity_authority": (
            "authenticated_single_tensor_container_sha256_and_native_fp32_roundtrip"
        ),
    }
    return {**binding, "binding_digest": object_sha256(binding)}


def model_freeze_runtime_certificate(
    model: nn.Module,
    *,
    base_certificate: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        checked = frozen_runtime._validated_freeze_certificate(base_certificate)
    except frozen_runtime.PairV5T2VEnergyScoringError as error:
        raise CAGENativeCandidateVJPProbeError(str(error)) from error
    trainable = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    gradients = [
        name for name, parameter in model.named_parameters() if parameter.grad is not None
    ]
    if trainable or gradients or model.training:
        raise CAGENativeCandidateVJPProbeError(
            "frozen model acquired trainable parameters, gradients, or train mode"
        )
    value = {
        "standard_freeze_certificate": checked,
        "model_eval": True,
        "parameter_requires_grad_tensors": 0,
        "parameter_grad_tensors": 0,
        "input_vjp_only": True,
        "optimizer_created": False,
        "optimizer_steps": 0,
    }
    return {**value, "digest": object_sha256(value)}


def _float_matrix(value: torch.Tensor, *, label: str) -> list[list[float]]:
    tensor = value.detach().to(device="cpu", dtype=torch.float32)
    if tensor.ndim != 2 or not bool(torch.isfinite(tensor).all().item()):
        raise CAGENativeCandidateVJPProbeError(f"{label} must be finite rank two")
    return [[float(item) for item in row] for row in tensor.tolist()]


def vjp_result_receipt(
    result: cage.CandidateActionEnergyVJPResult,
) -> dict[str, Any]:
    scan = result.scan
    value = {
        "cage_core_schema": cage.SCHEMA_VERSION,
        "cage_core_contract": cage.contract_receipt(),
        "coordinate_ids": list(scan.coordinate_ids),
        "sigmas": [float(item) for item in scan.sigmas],
        "branch_order": list(mace.BRANCH_ORDER),
        "hard_negative_order": list(mace.HARD_NEGATIVE_BRANCHES),
        "branch_energies_fp32": _float_matrix(
            scan.branch_energies, label="branch energies"
        ),
        "negative_log_energy_ratios_fp32": _float_matrix(
            scan.negative_log_energy_ratios, label="energy margins"
        ),
        "selection_detached": scan.selection_detached,
        "selected_coordinate_index": scan.selected_coordinate_index,
        "selected_coordinate_id": scan.selected_coordinate_id,
        "selected_sigma": scan.selected_sigma,
        "selected_negative_index": scan.selected_negative_index,
        "selected_negative_branch": scan.selected_negative_branch,
        "selected_action_energy": scan.selected_action_energy,
        "selected_negative_energy": scan.selected_negative_energy,
        "selected_margin": scan.selected_margin,
        "selected_loss": scan.selected_loss,
        "replay_call_order": list(result.replay_call_order),
        "replay_action_energy": result.replay_action_energy,
        "replay_negative_energy": result.replay_negative_energy,
        "margin_derivative": result.margin_derivative,
        "action_energy_derivative": result.action_energy_derivative,
        "negative_energy_derivative": result.negative_energy_derivative,
        "gradient_norm": result.gradient_norm,
        "gradient_finite": result.finite,
        "gradient_nonzero": result.nonzero,
        "direct_flow_target_gradient_identity": frozen_runtime.native_tensor_value_identity(
            result.direct_flow_target_gradient
        ),
        "action_input_vjp_identity": frozen_runtime.native_tensor_value_identity(
            result.action_input_vjp
        ),
        "negative_input_vjp_identity": frozen_runtime.native_tensor_value_identity(
            result.negative_input_vjp
        ),
        "candidate_clean_cotangent_identity": frozen_runtime.native_tensor_value_identity(
            result.gradient
        ),
        "gradient_formula": (
            "direct_flow_target_term+(1-sigma)*(action_input_vjp+negative_input_vjp)"
        ),
    }
    return {**value, "digest": object_sha256(value)}


def save_cotangent_safetensors(path: Path, value: torch.Tensor) -> dict[str, Any]:
    """Create and verify one exact FP32 cotangent artifact atomically."""

    from safetensors import safe_open
    from safetensors.torch import save_file

    if (
        not path.is_absolute()
        or path.suffix != ".safetensors"
        or path.exists()
        or path.is_symlink()
        or not path.parent.is_dir()
        or path.parent.is_symlink()
    ):
        raise CAGENativeCandidateVJPProbeError(
            "cotangent path must be a fresh absolute safetensors file"
        )
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or value.ndim != 5
        or tuple(int(item) for item in value.shape[:3]) != (1, 16, 21)
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise CAGENativeCandidateVJPProbeError(
            "cotangent must be detached finite FP32 exact81"
        )
    stored = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    identity = frozen_runtime.native_tensor_value_identity(stored)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".safetensors",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        save_file(
            {COTANGENT_KEY: stored},
            str(temporary),
            metadata={
                "schema_version": ARTIFACT_SCHEMA,
                "coordinate": "normalized_clean_latent_cotangent",
                "source": "frozen_t2v_candidate_own_action_energy_input_vjp",
                "proposal_media_consumed": "false",
            },
        )
        with safe_open(str(temporary), framework="pt", device="cpu") as opened:
            if list(opened.keys()) != [COTANGENT_KEY]:
                raise CAGENativeCandidateVJPProbeError(
                    "cotangent safetensors key closure differs"
                )
            restored = opened.get_tensor(COTANGENT_KEY).contiguous()
            metadata = dict(opened.metadata() or {})
        if (
            restored.dtype != torch.float32
            or not torch.equal(restored, stored)
            or metadata.get("schema_version") != ARTIFACT_SCHEMA
            or metadata.get("proposal_media_consumed") != "false"
        ):
            raise CAGENativeCandidateVJPProbeError(
                "cotangent safetensors round trip differs"
            )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
    artifact = {
        "schema_version": ARTIFACT_SCHEMA,
        "path": str(path),
        "sha256": file_sha256(path),
        "tensor_key": COTANGENT_KEY,
        "shape": identity["shape"],
        "stored_dtype": identity["dtype"],
        "raw_value_sha256": identity["raw_value_sha256"],
        "content_sha256": identity["content_sha256"],
        "roundtrip_byte_exact_fp32": True,
        "proposal_media_consumed": False,
    }
    return {**artifact, "artifact_digest": object_sha256(artifact)}


def _write_create_only_json(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise CAGENativeCandidateVJPProbeError(
            "receipt path must be a fresh file in an existing directory"
        )
    raw = canonical_json_bytes(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        if path.exists() or path.is_symlink():
            path.unlink()
        raise
    return hashlib.sha256(raw).hexdigest()


def contract_receipt() -> dict[str, Any]:
    public_tensor_functions = (
        make_energy_coordinates,
        compute_native_candidate_cotangent,
        verify_authenticated_native_clean_tensor_identity,
        save_cotangent_safetensors,
    )
    offending = {
        function.__name__: sorted(
            set(inspect.signature(function).parameters)
            & FORBIDDEN_PUBLIC_INPUT_NAMES
        )
        for function in public_tensor_functions
        if set(inspect.signature(function).parameters)
        & FORBIDDEN_PUBLIC_INPUT_NAMES
    }
    if offending:
        raise CAGENativeCandidateVJPProbeError(
            f"public tensor API exposes forbidden inputs: {offending}"
        )
    value = {
        "schema_version": SCHEMA_VERSION,
        "topology": "WORLD4_SP4_single_candidate",
        "schedule_indices": list(PROBE_SCHEDULE_INDICES),
        "native_timesteps": [
            item.native_timestep for item in native_probe_coordinates()
        ],
        "branch_order": list(mace.BRANCH_ORDER),
        "scan_calls": len(PROBE_SCHEDULE_INDICES) * len(mace.BRANCH_ORDER),
        "selected_input_vjp_replay_calls": 2,
        "accepted_tensor_inputs": [
            "sealed_native_rv2v_normalized_clean_latent",
            "same_candidate_official_initial_gaussian",
        ],
        "accepted_semantic_inputs": [
            "matching_core4_v2_action_plus_nine_caption_cell"
        ],
        "same_clean_and_official_gaussian_across_sigma_coordinates": True,
        "native_clean_identity_authority": (
            "authenticated_container_sha256_single_key_shape_dtype_coordinate_"
            "origin_role_fp32_roundtrip_plus_recomputed_value_hashes"
        ),
        "native_clean_absent_value_hashes_never_compared_as_none": True,
        "official_gaussian_recorded_raw_content_identity_required": True,
        "pure_t2v_media_consumed": False,
        "proposal_media_target_condition_noise_or_donor_consumed": False,
        "source_video_tensor_consumed": False,
        "source_video_hash_bound_as_provenance": True,
        "mask_flow_pose_track_trajectory_consumed": False,
        "existing_no_grad_scorer_used_for_replay": False,
        "native_discrete_timestep_used": True,
        "legacy_1000_sigma_timestep_rejected": True,
        "cotangent_all_rank_exact_consensus_required": True,
        "training_performed": False,
        "optimizer_created": False,
    }
    return {**value, "digest": object_sha256(value)}


def _make_receipt(
    *,
    args: argparse.Namespace,
    row: Mapping[str, Any],
    prompt_binding: Mapping[str, Any],
    checkpoint_identity: Mapping[str, Any],
    checkpoint_binding: Mapping[str, Any],
    freeze_before: Mapping[str, Any],
    freeze_after: Mapping[str, Any],
    clean_identity: Mapping[str, Any],
    gaussian_identity: Mapping[str, Any],
    result_receipt: Mapping[str, Any],
    execution_receipt: Mapping[str, Any],
    cotangent_artifact: Mapping[str, Any],
    runtime_versions: Mapping[str, str],
    bernini_revision: str,
    veomni_revision: str,
) -> dict[str, Any]:
    candidate = row["candidate"]
    native_artifacts = row["native_artifacts"]
    value = {
        "schema_version": SCHEMA_VERSION,
        "contract": contract_receipt(),
        "candidate": {
            "candidate_id": candidate["candidate_id"],
            "group_id": row["group_id"],
            "ordinal": row["ordinal"],
            "seed": candidate["seed"],
            "analysis_split": row["cell"]["analysis_split"],
            "action_family_id": row["cell"]["action_family_id"],
            "calibration_group_id": row["cell"]["calibration_group_id"],
            "source_video_sha256": candidate["source_video_sha256"],
            "action_caption_sha256": candidate["complete_caption_sha256"],
        },
        "sealed_specs": {
            "native_population_path": row["population_spec_path"],
            "native_population_sha256": row["population_spec_sha256"],
            "t2v_core4_v2_path": row["t2v_spec_path"],
            "t2v_core4_v2_sha256": row["t2v_spec_sha256"],
            "branch_order": list(mace.BRANCH_ORDER),
        },
        "native_rollout": {
            "pair_receipt_path": row["pair_receipt_path"],
            "pair_receipt_file_sha256": row["pair_receipt_file_sha256"],
            "pair_receipt_digest": row["pair_receipt_digest"],
            "native_receipt_path": row["native_receipt_path"],
            "native_receipt_file_sha256": row["native_receipt_file_sha256"],
            "native_receipt_digest": native_artifacts["native_receipt_digest"],
            "source_video_path": row["source_video_path"],
            "source_video_consumed_as_tensor": False,
            "generated_rv2v_mp4_consumed_as_tensor": False,
        },
        "candidate_coordinate": {
            "normalized_clean_latent_artifact": dict(
                native_artifacts["predecode_clean_latent"]
            ),
            "official_initial_gaussian_artifact": dict(
                native_artifacts["official_initial_gaussian"]
            ),
            "normalized_clean_latent_identity": dict(clean_identity),
            "official_initial_gaussian_identity": dict(gaussian_identity),
            "same_candidate_artifacts": True,
            "same_gaussian_reused_across_three_sigmas": True,
        },
        "prompts": dict(prompt_binding),
        "coordinates": [
            item.receipt() for item in native_probe_coordinates()
        ],
        "frozen_runtime": {
            "checkpoint_identity": dict(checkpoint_identity),
            "checkpoint_binding": dict(checkpoint_binding),
            "freeze_before": dict(freeze_before),
            "freeze_after": dict(freeze_after),
            "freeze_certificates_identical": freeze_before == freeze_after,
            "bernini_revision": bernini_revision,
            "veomni_revision": veomni_revision,
            "runtime_versions": dict(runtime_versions),
        },
        "execution": dict(execution_receipt),
        "vjp": dict(result_receipt),
        "output": dict(cotangent_artifact),
        "distributed": {
            "world_size": WORLD_SIZE,
            "ulysses_size": SP_SIZE,
            "candidate_input_all_rank_exact": True,
            "cotangent_all_rank_exact": True,
            "receipt_digest_all_rank_exact": True,
            "cotangent_reduction_or_averaging_used": False,
        },
        "input_closure": {
            "pure_t2v_caption_text_only": True,
            "pure_t2v_generated_media_consumed": False,
            "pure_t2v_predecode_latent_consumed": False,
            "pure_t2v_gaussian_consumed": False,
            "target_video": False,
            "donor": False,
            "mask": False,
            "flow": False,
            "pose": False,
            "track": False,
            "trajectory": False,
        },
        "training_performed": False,
        "optimizer_created": False,
        "optimizer_steps": 0,
        "scientific_action_editing_success_claim": False,
        "method_source_sha256": file_sha256(Path(__file__).resolve()),
        "cli": {
            "candidate_id": args.candidate_id,
            "expected_checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
        },
    }
    if freeze_before != freeze_after:
        raise CAGENativeCandidateVJPProbeError(
            "freeze runtime certificate changed during probe"
        )
    return {**value, "receipt_digest": object_sha256(value)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--population-spec",
        default=str(
            METHOD_ROOT
            / "assets/pair_v5_native_rv2v4_core4_action_population_v1.json"
        ),
    )
    parser.add_argument(
        "--expected-population-spec-sha256",
        default=PINNED_NATIVE_CORE4_POPULATION_SHA256,
    )
    parser.add_argument(
        "--t2v-spec",
        default=str(METHOD_ROOT / "assets/pair_v5_t2v_calibration_core4_bank_v2.json"),
    )
    parser.add_argument(
        "--expected-t2v-spec-sha256", default=PINNED_T2V_CORE4_V2_SHA256
    )
    parser.add_argument("--rollout-root", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument(
        "--expected-checkpoint-tree-sha256",
        default=native_generation.legacy.trainer.CHECKPOINT_TREE_SHA256,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--expected-bernini-commit",
        default=native_generation.legacy.trainer.BERNINI_OFFICIAL_COMMIT,
    )
    parser.add_argument(
        "--expected-veomni-commit",
        default=native_generation.legacy.trainer.VEOMNI_TESTED_COMMIT,
    )
    parser.add_argument("--target-margin", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--ack-probe-not-training", action="store_true")
    return parser


def _validate_cli(args: argparse.Namespace) -> None:
    _safe_id(args.candidate_id, label="candidate_id")
    for name in (
        "expected_population_spec_sha256",
        "expected_t2v_spec_sha256",
        "expected_checkpoint_tree_sha256",
    ):
        _sha256(getattr(args, name), label=name)
    if (
        args.expected_population_spec_sha256
        != PINNED_NATIVE_CORE4_POPULATION_SHA256
        or args.expected_t2v_spec_sha256 != PINNED_T2V_CORE4_V2_SHA256
        or args.expected_checkpoint_tree_sha256
        != native_generation.legacy.trainer.CHECKPOINT_TREE_SHA256
    ):
        raise CAGENativeCandidateVJPProbeError(
            "CLI sealed population/T2V/checkpoint identity differs"
        )
    for name in ("target_margin", "temperature"):
        value = getattr(args, name)
        if not math.isfinite(value) or (name == "temperature" and value <= 0.0):
            raise CAGENativeCandidateVJPProbeError(f"{name} differs")
    if args.ack_probe_not_training is not True:
        raise CAGENativeCandidateVJPProbeError(
            "--ack-probe-not-training is mandatory"
        )


def _fresh_output_path(value: str | Path) -> Path:
    requested = Path(value)
    if not requested.is_absolute() or requested == Path("/"):
        raise CAGENativeCandidateVJPProbeError(
            "output-dir must be absolute and non-root"
        )
    parent = requested.parent.resolve(strict=True)
    if not parent.is_dir() or parent.is_symlink():
        raise CAGENativeCandidateVJPProbeError(
            "output-dir parent must be a plain directory"
        )
    output = parent / requested.name
    if output.exists() or output.is_symlink():
        raise CAGENativeCandidateVJPProbeError("output-dir must be fresh")
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_cli(args)
    distributed = world4_sp4_contract()
    output = _fresh_output_path(args.output_dir)
    row = load_sealed_native_candidate(
        population_spec_path=args.population_spec,
        population_spec_sha256=args.expected_population_spec_sha256,
        t2v_spec_path=args.t2v_spec,
        t2v_spec_sha256=args.expected_t2v_spec_sha256,
        rollout_root=args.rollout_root,
        candidate_id=args.candidate_id,
        checkpoint_tree_sha256=args.expected_checkpoint_tree_sha256,
    )
    legacy = native_generation.legacy
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
        raise CAGENativeCandidateVJPProbeError(str(error)) from error
    if int(transformer_config.get("num_attention_heads", -1)) != 12:
        raise CAGENativeCandidateVJPProbeError(
            "pinned Bernini 1.3B attention-head count differs"
        )
    legacy.trainer.activate_source_trees(bernini_root, veomni_root)

    import torch.distributed as dist
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state

    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        raise CAGENativeCandidateVJPProbeError(
            "native CAGE probe requires four AUH ROCm GPUs"
        )
    if torch.cuda.device_count() != WORLD_SIZE:
        raise CAGENativeCandidateVJPProbeError(
            "visible accelerator count differs from WORLD4"
        )
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=120),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=SP_SIZE)
    device = torch.device("cuda", distributed.local_rank)

    checkpoint_rows: list[Any] = [None]
    if distributed.rank == 0:
        try:
            checkpoint_rows[0] = {
                "ok": True,
                "identity": native_generation.source_audit.validate_checkpoint_content(
                    checkpoint, Path(args.checkpoint_content_manifest)
                ),
            }
        except Exception as error:
            checkpoint_rows[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    dist.broadcast_object_list(checkpoint_rows, src=0)
    checkpoint_result = checkpoint_rows[0]
    if (
        not isinstance(checkpoint_result, Mapping)
        or checkpoint_result.get("ok") is not True
    ):
        raise CAGENativeCandidateVJPProbeError(
            f"rank-zero checkpoint audit failed: {checkpoint_result}"
        )
    checkpoint_identity = dict(checkpoint_result["identity"])

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.inference_renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    legacy.trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    renderer = BerniniRendererModel(config).requires_grad_(False).eval().to(device)
    standard_freeze_before = (
        native_generation.source_audit.model_freeze_certificate(renderer)
    )
    freeze_before = model_freeze_runtime_certificate(
        renderer, base_certificate=standard_freeze_before
    )
    checkpoint_binding = frozen_runtime.checkpoint_content_binding(
        checkpoint_identity, standard_freeze_before
    )
    diffusion = renderer.diff_dec
    transformer = diffusion.transformer
    if transformer is None or diffusion.transformer_2 is not None:
        raise CAGENativeCandidateVJPProbeError(
            "Phase-A requires frozen transformer_1 only"
        )

    prompt_binding = native_score.prompt_binding_from_cell(
        row["cell"], prompt_cleaner=prompt_clean
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), subfolder="tokenizer", **legacy.tokenizer_load_kwargs()
    )
    conditions = frozen_runtime._encode_prompt_bank(
        renderer,
        tokenizer,
        prompt_binding["prompt_by_branch"],
        device=device,
    )
    clean_cpu = frozen_runtime._load_exact81_tensor(
        row["native_artifacts"]["predecode_clean_latent"],
        key="normalized_clean_latent",
        label=f"{args.candidate_id} normalized clean latent",
    )
    gaussian_cpu = frozen_runtime._load_exact81_tensor(
        row["native_artifacts"]["official_initial_gaussian"],
        key="official_initial_gaussian",
        label=f"{args.candidate_id} official initial Gaussian",
    )
    clean_identity = verify_authenticated_native_clean_tensor_identity(
        clean_cpu,
        row["native_artifacts"]["predecode_clean_latent"],
        label="normalized clean latent",
    )
    gaussian_identity = frozen_runtime.verify_native_tensor_value_identity(
        gaussian_cpu,
        row["native_artifacts"]["official_initial_gaussian"],
        label="official initial Gaussian",
    )
    if clean_cpu.shape != gaussian_cpu.shape:
        raise CAGENativeCandidateVJPProbeError(
            "native clean/Gaussian tensor geometry differs"
        )
    require_sp4_object_consensus(
        {"clean": clean_identity, "gaussian": gaussian_identity},
        label="native candidate coordinate",
        distributed_module=dist,
    )
    clean = clean_cpu.to(device=device).contiguous()
    gaussian = gaussian_cpu.to(device=device).contiguous()
    del clean_cpu, gaussian_cpu

    bridge = NativeMultiSigmaFrozenT2VInputVJPBridge(
        diffusion,
        transformer,
        prompt_binding["prompt_by_branch"],
        conditions,
        frozen_model_receipt_digest=checkpoint_binding["binding_digest"],
    )
    # The bridge owns graph-safe frozen copies when prompt encoding returned
    # inference tensors; release the original registry before VJP replay.
    del conditions
    energy_config = cage.EnergyVJPConfig(
        target_margin=float(args.target_margin),
        temperature=float(args.temperature),
    )
    result, execution_receipt = compute_native_candidate_cotangent(
        clean,
        gaussian,
        prompt_binding["prompt_by_branch"],
        bridge,
        config=energy_config,
    )
    result_receipt = vjp_result_receipt(result)
    require_sp4_object_consensus(
        result_receipt,
        label="candidate action cotangent",
        distributed_module=dist,
    )

    standard_freeze_after = (
        native_generation.source_audit.model_freeze_certificate(renderer)
    )
    freeze_after = model_freeze_runtime_certificate(
        renderer, base_certificate=standard_freeze_after
    )
    if standard_freeze_after != standard_freeze_before or freeze_after != freeze_before:
        raise CAGENativeCandidateVJPProbeError(
            "frozen Bernini certificate changed during input VJP"
        )
    runtime_versions = frozen_runtime.current_runtime_versions()

    if distributed.rank == 0:
        output.mkdir(parents=False, exist_ok=False)
        cotangent_artifact: Any = save_cotangent_safetensors(
            output / COTANGENT_FILENAME, result.gradient
        )
    else:
        cotangent_artifact = None
    artifact_rows: list[Any] = [cotangent_artifact]
    dist.broadcast_object_list(artifact_rows, src=0)
    cotangent_artifact = artifact_rows[0]
    if not isinstance(cotangent_artifact, Mapping):
        raise CAGENativeCandidateVJPProbeError(
            "rank-zero cotangent artifact publication failed"
        )
    require_sp4_object_consensus(
        dict(cotangent_artifact),
        label="cotangent artifact metadata",
        distributed_module=dist,
    )
    receipt = _make_receipt(
        args=args,
        row=row,
        prompt_binding=prompt_binding,
        checkpoint_identity=checkpoint_identity,
        checkpoint_binding=checkpoint_binding,
        freeze_before=freeze_before,
        freeze_after=freeze_after,
        clean_identity=clean_identity,
        gaussian_identity=gaussian_identity,
        result_receipt=result_receipt,
        execution_receipt=execution_receipt,
        cotangent_artifact=cotangent_artifact,
        runtime_versions=runtime_versions,
        bernini_revision=bernini_revision,
        veomni_revision=veomni_revision,
    )
    require_sp4_object_consensus(
        receipt["receipt_digest"],
        label="probe receipt digest",
        distributed_module=dist,
    )
    if distributed.rank == 0:
        _write_create_only_json(output / RECEIPT_FILENAME, receipt)
        print(canonical_json_bytes(receipt).decode("ascii"), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0


__all__ = [
    "ARTIFACT_SCHEMA",
    "COTANGENT_FILENAME",
    "COTANGENT_KEY",
    "CAGENativeCandidateVJPProbeError",
    "NativeMultiSigmaFrozenT2VInputVJPBridge",
    "NativeProbeCoordinate",
    "PINNED_NATIVE_CORE4_POPULATION_SHA256",
    "PINNED_T2V_CORE4_V2_SHA256",
    "PROBE_SCHEDULE_INDICES",
    "RECEIPT_FILENAME",
    "SCHEMA_VERSION",
    "SP_SIZE",
    "WORLD_SIZE",
    "World4SP4Contract",
    "compute_native_candidate_cotangent",
    "contract_receipt",
    "load_sealed_native_candidate",
    "make_energy_coordinates",
    "match_core4_native_candidate",
    "model_freeze_runtime_certificate",
    "native_probe_coordinates",
    "require_sp4_object_consensus",
    "save_cotangent_safetensors",
    "verify_authenticated_native_clean_tensor_identity",
    "vjp_result_receipt",
    "world4_sp4_contract",
]


if __name__ == "__main__":
    raise SystemExit(main())
