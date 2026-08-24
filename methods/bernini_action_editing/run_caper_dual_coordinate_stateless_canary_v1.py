#!/usr/bin/env python3
"""CAPER fail-closed two-wave preflight and immutable experiment plan.

This file deliberately implements only the first, read-only half of the
CAPER experiment:

1. validate the pinned CAPER authority/specification; and
2. preregister the two WORLD8 waves which feed the existing
   :mod:`run_qmosaic_editor_direction_sp4_v1` clean-latent direction runner.

Production packet materialization is intentionally delegated to
``materialize_qmosaic_editor_runtime_v1.py``.  The former transient,
ephemeral-key materializer is not exposed by this command-line interface.

This is not a trainer.  It contains no optimizer, LoRA VJP, parameter update,
adapter publication, mask, track, pose, flow, or trajectory path.  The
generation Gaussian is authorized only as direction-measurement state and is
explicitly forbidden as a later training epsilon.  Semantic action authority
remains false until an independently registered, action-specific exact81
segment evaluator has assessed every base/+q/-q arm.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import asdict
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_lora as legacy  # noqa: E402
import infer_native_identity_generation_canary as native  # noqa: E402
import infer_source_kv_carrier_oracle as source_audit  # noqa: E402
import self_imagined_native_rv2v_hidden_vjp_v1 as qmosaic  # noqa: E402


ASSET_SCHEMA_VERSION = "bernini-caper-dual-coordinate-core4-canary-v1"
MATERIALIZER_SCHEMA_VERSION = "bernini-caper-editor-runtime-materializer-v1"
PREFLIGHT_SCHEMA_VERSION = "bernini-caper-dual-coordinate-preflight-v1"
METHOD_NAME = "bernini-caper-dual-coordinate-stateless-canary"
PINNED_ASSET_SHA256 = (
    "b8fe179905cf77951fda2fdc6cf18622b11510263c56df4b306457a3ce717f57"
)
PINNED_ASSET_OBJECT_DIGEST = (
    "8a6519ade2464f5f431c6df558035e263dff3f05cc88f7b59c9e47545d048487"
)
PINNED_REGISTRY_SHA256 = (
    "01fe53b02fa42da8eb5c187a81e6737f323604e7dc26b3eee4f941ad4de82d96"
)
PINNED_CHECKPOINT_MANIFEST_SHA256 = (
    "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
)
PINNED_OWNER_MASTER_FILE_SHA256 = (
    "c0d8f3e4a7f3b95269b5196c0d8844327d9e7296dda1828493683a9ae7d707de"
)
PINNED_OWNER_MASTER_RECEIPT_DIGEST = (
    "b71d726c7001c57da80391b18c5c82b8fe0910a62f8cd99484d3a90d218347ab"
)
PINNED_OWNER_AUDIT_SIDECAR_SHA256 = (
    "24746c91e88e4051c49fe18b06e0e58bb2c4b119b3d946586d9dd6092308030b"
)
PINNED_OWNER_AUDIT_EVIDENCE_SHA256 = (
    "3e2335d4d335a9ee8262aa319fc2790dbac3e59b20e554f54b4dc1273f259dc3"
)
PINNED_OWNER_AUDIT_PUBLIC_KEY_SHA256 = (
    "d1bba83ca1d162128bda71e21c419c476b9328c7892bd1998adcd24c09c577ec"
)
PINNED_QUOTIENT_MASTER_FILE_SHA256 = (
    "fde8de229135bf46682681bbc83fc39d7554e144b6d97991741e39a2ebfe98c3"
)
PINNED_QUOTIENT_MASTER_RECEIPT_DIGEST = (
    "8fa7e4cf01d9fa49b506aa66b50932c2ac767faa4727dc772e407d41052652e6"
)
PINNED_CELL_RECEIPT_FILE_SHA256 = {
    "dog": "5630c0f511360a6ae0386855f4c00e78e226fea32f71d340773db83ab5c49bd2",
    "human": "fb6a37464e98841fe340e5a1411dffe8135640410fd0cef5c1f89b86fe81184e",
}
PINNED_CELL_RECEIPT_DIGEST = {
    "dog": "6970b785eda453afa7a382c2ab6638e6f286ba8115bdcfb632af4eefd02bdf90",
    "human": "5471955cfc8a67ec4aef0e414815fc1dec763db9322951e2f485bfa790fe97f3",
}
PINNED_GENERATION_RECEIPT_FILE_SHA256 = {
    "dog": "e6e6cdcd7ffbb6c3fcbaad52ac3ed088429c842777238aa021fc07de1fa67cf2",
    "human": "b1fd8e8a8296bb8e688fb09e84141ce67a5f5416ccd0e1cdca39ca18f307ac9b",
}
PINNED_GENERATION_RECEIPT_DIGEST = {
    "dog": "10acf8d383185ae87f19a966ae2d9a524d793ca315ce7e1779a18de62661cafe",
    "human": "6ff703a713b92faf8e0e1ff83c76ceb45715ef821ce025b210a5e4e13e7b5b01",
}
PINNED_DIRECTION_GATE_PRIVATE_KEY = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_qmosaic_editor_authority_20260809/"
    "direction-gate-ed25519-private.pem"
)
PINNED_DIRECTION_GATE_PUBLIC_KEY = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_qmosaic_editor_authority_20260809/"
    "direction-gate-ed25519-public.pem"
)
PINNED_DIRECTION_GATE_PUBLIC_KEY_SHA256 = (
    "655befbbf0deea1006e33e9656127e11753d85a0ae22d84619bfbcb8185dcdbd"
)
PINNED_EDITOR_RUNTIME_PRIVATE_KEY = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_qmosaic_editor_authority_20260809/"
    "editor-runtime-ed25519-private.pem"
)
PINNED_EDITOR_RUNTIME_PUBLIC_KEY = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_qmosaic_editor_authority_20260809/"
    "editor-runtime-ed25519-public.pem"
)
PINNED_EDITOR_RUNTIME_PUBLIC_KEY_SHA256 = (
    "b1357fcf5d3b30e51d686a2f1170bc139a7d8c5ea3ef99dc7cc9b2b008d3052d"
)
QUERY_SEEDS = {
    "dog": (2026081502, 2026081503),
    "human": (2026081505, 2026081506),
}
EDITOR_NOISE_SEEDS = {
    "dog": (2026082502, 2026082503),
    "human": (2026082505, 2026082506),
}
WAVE_PLAN = (
    (("dog", 2026081502), ("human", 2026081505)),
    (("dog", 2026081503), ("human", 2026081506)),
)
WORLD_SIZE = 4
SP_SIZE = 4
EXACT_FRAMES = 81
LATENT_PHASES = 21
NATIVE_STEPS = 40
NATIVE_TIMESTEP = 516
RUNTIME_TENSOR_KEYS = (
    "source_latent",
    "image_reference_0",
    "image_reference_1",
    "image_reference_2",
    "image_reference_3",
    "clean_latent",
    "official_initial_noise",
    "action_condition",
    "noop_condition",
    "timestep",
)
_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class CAPERRuntimeError(RuntimeError):
    """A pinned authority, actual sampler observation, or packet seal differed."""


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
        raise CAPERRuntimeError("value is not finite canonical ASCII JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    source = Path(path)
    if not source.is_absolute() or not source.is_file() or source.is_symlink():
        raise CAPERRuntimeError("hashed artifact must be an absolute plain file")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CAPERRuntimeError(f"{label} must be lowercase SHA-256")
    return value


def _sha1(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA1_RE.fullmatch(value) is None:
        raise CAPERRuntimeError(f"{label} must be full lowercase SHA-1")
    return value


def _sealed(unsigned: Mapping[str, Any]) -> dict[str, Any]:
    if "receipt_digest" in unsigned:
        raise CAPERRuntimeError("cannot reseal a receipt")
    row = dict(unsigned)
    return {**row, "receipt_digest": object_sha256(row)}


def _write_create_only(path: Path, payload: bytes, *, mode: int = 0o400) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise CAPERRuntimeError("output must be a fresh absolute plain path")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _write_json_create_only(path: Path, value: Mapping[str, Any]) -> str:
    payload = canonical_json_bytes(value) + b"\n"
    _write_create_only(path, payload)
    return hashlib.sha256(payload).hexdigest()


def _read_canonical_json(path: Path, *, expected_sha256: str) -> Mapping[str, Any]:
    if file_sha256(path) != _sha256(expected_sha256, label="JSON file SHA-256"):
        raise CAPERRuntimeError("JSON artifact bytes changed")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise CAPERRuntimeError("JSON artifact is not canonical ASCII") from error
    if not isinstance(value, Mapping):
        raise CAPERRuntimeError("JSON artifact root is not an object")
    # The externally pinned raw-file digest is the byte authority.  Requiring
    # a particular whitespace layout here would add no integrity and would
    # prevent the preregistration from remaining human-reviewable.  This call
    # still rejects NaN/non-ASCII-incompatible values through canonicalization.
    canonical_json_bytes(value)
    return dict(value)


_ASSET_FIELDS = {
    "schema_version",
    "method_name",
    "status",
    "topology",
    "model",
    "owner_authority",
    "quotient_authority",
    "editor_runtime_authority",
    "cells",
    "waves",
    "direction_phase",
    "later_update_phase",
    "forbidden_inputs",
    "hard_negative_authority",
}


def load_and_validate_asset(
    path: str | Path, expected_sha256: Optional[str] = None
) -> Mapping[str, Any]:
    """Load the immutable preregistration and enforce fixed authority bytes."""

    source = Path(path).resolve(strict=True)
    expected = PINNED_ASSET_SHA256 if expected_sha256 is None else expected_sha256
    if expected != PINNED_ASSET_SHA256:
        raise CAPERRuntimeError("asset SHA-256 is pinned, not caller-selectable")
    value = _read_canonical_json(source, expected_sha256=expected)
    if set(value) != _ASSET_FIELDS:
        raise CAPERRuntimeError("asset field closure differs")
    topology = value.get("topology")
    model = value.get("model")
    owner = value.get("owner_authority")
    quotient = value.get("quotient_authority")
    editor_runtime = value.get("editor_runtime_authority")
    direction = value.get("direction_phase")
    later = value.get("later_update_phase")
    hard_negative = value.get("hard_negative_authority")
    if not all(
        isinstance(row, Mapping)
        for row in (
            topology, model, owner, quotient, editor_runtime, direction, later,
            hard_negative,
        )
    ):
        raise CAPERRuntimeError("asset nested closure differs")
    expected_nested_fields = {
        "topology": {
            "node_gpu_count", "dp_arms", "sp_size_per_arm",
            "world_size_per_process_group", "parallel_contract",
        },
        "model": {
            "bernini_commit", "veomni_commit", "checkpoint_tree_sha256",
            "checkpoint_content_manifest_sha256", "frame_count", "latent_phases",
            "sampler_steps", "native_schedule_index", "native_timestep",
            "direction_relative_l2_dose",
        },
        "owner": {
            "root", "registry", "registry_file_sha256", "master_receipt",
            "master_receipt_file_sha256", "master_receipt_digest", "audit_sidecar",
            "audit_sidecar_file_sha256", "audit_evidence",
            "audit_evidence_file_sha256", "audit_public_key",
            "audit_public_key_file_sha256",
            "owner_rgb_latent_noise_velocity_in_editor_graph",
            "only_detached_normalized_hidden_quotient_authorized",
        },
        "quotient": {
            "root", "master_receipt", "master_receipt_file_sha256",
            "master_receipt_digest", "query_specific_detached_unit_only",
        },
        "editor_runtime": {"private_key", "public_key", "public_key_sha256"},
        "direction": {
            "runner", "coordinate", "no_lora_vjp_required", "optimizer",
            "parameter_update", "semantic_authority_after_decode",
            "requires_independent_segment_evaluator", "required_evaluator_cells",
            "required_evaluator_index", "batch_or_segment_mean_compensation",
        },
        "later": {
            "authorized", "reason", "direction_gate_authority_private_key",
            "direction_gate_authority_public_key",
            "direction_gate_authority_public_key_sha256", "required_order",
            "optimizer", "generation_gaussian_reuse",
        },
        "hard_negative": {
            "root", "bank_receipt_file_sha256", "bank_receipt_digest",
            "semantic_audit", "gradient_source_authorized", "use",
        },
    }
    for label, row in (
        ("topology", topology), ("model", model), ("owner", owner),
        ("quotient", quotient), ("editor_runtime", editor_runtime),
        ("direction", direction), ("later", later),
        ("hard_negative", hard_negative),
    ):
        if set(row) != expected_nested_fields[label]:
            raise CAPERRuntimeError(f"asset {label} field closure differs")
    if (
        value.get("schema_version") != ASSET_SCHEMA_VERSION
        or value.get("method_name") != METHOD_NAME
        or value.get("status") != "DIRECTION_MATERIALIZATION_ONLY_NO_TRAINING_AUTHORITY"
        or dict(topology)
        != {
            "node_gpu_count": 8,
            "dp_arms": 2,
            "sp_size_per_arm": 4,
            "world_size_per_process_group": 4,
            "parallel_contract": "two_concurrent_WORLD4_SP4_groups",
        }
        or model.get("checkpoint_content_manifest_sha256")
        != PINNED_CHECKPOINT_MANIFEST_SHA256
        or owner.get("master_receipt_file_sha256")
        != PINNED_OWNER_MASTER_FILE_SHA256
        or owner.get("master_receipt_digest")
        != PINNED_OWNER_MASTER_RECEIPT_DIGEST
        or owner.get("audit_sidecar_file_sha256")
        != PINNED_OWNER_AUDIT_SIDECAR_SHA256
        or owner.get("audit_evidence_file_sha256")
        != PINNED_OWNER_AUDIT_EVIDENCE_SHA256
        or owner.get("audit_public_key_file_sha256")
        != PINNED_OWNER_AUDIT_PUBLIC_KEY_SHA256
        or quotient.get("master_receipt_file_sha256")
        != PINNED_QUOTIENT_MASTER_FILE_SHA256
        or quotient.get("master_receipt_digest")
        != PINNED_QUOTIENT_MASTER_RECEIPT_DIGEST
        or editor_runtime.get("private_key")
        != PINNED_EDITOR_RUNTIME_PRIVATE_KEY
        or editor_runtime.get("public_key")
        != PINNED_EDITOR_RUNTIME_PUBLIC_KEY
        or editor_runtime.get("public_key_sha256")
        != PINNED_EDITOR_RUNTIME_PUBLIC_KEY_SHA256
    ):
        raise CAPERRuntimeError("pinned model/owner/quotient authority differs")
    raw_cells = value.get("cells")
    raw_waves = value.get("waves")
    if not isinstance(raw_cells, list) or len(raw_cells) != 2:
        raise CAPERRuntimeError("asset must contain exactly dog and human cells")
    cells = {row.get("cell_id"): row for row in raw_cells if isinstance(row, Mapping)}
    if set(cells) != set(QUERY_SEEDS):
        raise CAPERRuntimeError("asset dog/human cell closure differs")
    for cell_id, seeds in QUERY_SEEDS.items():
        row = cells[cell_id]
        if set(row) != {
            "cell_id", "source_iid", "source_video", "source_video_sha256",
            "latent_shape", "action_family_id", "action_prompt_utf8",
            "action_prompt_utf8_sha256", "noop_prompt_utf8",
            "noop_prompt_utf8_sha256", "role_prompt_binding_digest",
            "query_seeds", "editor_noise_seeds", "generation_mode", "editor_mode",
            "segment_order", "generation_receipt", "generation_receipt_file_sha256",
            "generation_receipt_digest", "quotient_cell_root", "quotient_receipt",
            "quotient_receipt_file_sha256", "quotient_receipt_digest",
        }:
            raise CAPERRuntimeError(f"asset {cell_id} field closure differs")
        if (
            row.get("query_seeds") != list(seeds)
            or row.get("editor_noise_seeds") != list(EDITOR_NOISE_SEEDS[cell_id])
            or row.get("quotient_receipt_file_sha256")
            != PINNED_CELL_RECEIPT_FILE_SHA256[cell_id]
            or row.get("quotient_receipt_digest")
            != PINNED_CELL_RECEIPT_DIGEST[cell_id]
            or row.get("generation_receipt_file_sha256")
            != PINNED_GENERATION_RECEIPT_FILE_SHA256[cell_id]
            or row.get("generation_receipt_digest")
            != PINNED_GENERATION_RECEIPT_DIGEST[cell_id]
            or row.get("generation_mode") != "pure_t2v_exact81"
            or row.get("editor_mode") != "native_rv2v_exact81"
            or row.get("segment_order")
            != ["onset", "transition", "completion", "hold"]
        ):
            raise CAPERRuntimeError(f"asset {cell_id} authority differs")
        for name in (
            "source_video_sha256",
            "action_prompt_utf8_sha256",
            "noop_prompt_utf8_sha256",
            "role_prompt_binding_digest",
        ):
            _sha256(row.get(name), label=f"{cell_id} {name}")
    expected_waves = [
        {
            "wave_id": index + 1,
            "groups": [
                {
                    "cell_id": cell,
                    "query_seed": seed,
                    "cuda_visible_devices": "0,1,2,3" if slot == 0 else "4,5,6,7",
                    "master_port_offset": slot,
                }
                for slot, (cell, seed) in enumerate(groups)
            ],
        }
        for index, groups in enumerate(WAVE_PLAN)
    ]
    if raw_waves != expected_waves:
        raise CAPERRuntimeError("two-wave WORLD8 schedule differs")
    if (
        direction.get("runner") != "run_qmosaic_editor_direction_sp4_v1.py"
        or direction.get("no_lora_vjp_required") is not True
        or direction.get("parameter_update") is not False
        or direction.get("optimizer") is not False
        or direction.get("semantic_authority_after_decode") is not False
        or direction.get("requires_independent_segment_evaluator") is not True
        or later.get("authorized") is not False
        or later.get("reason")
        != "missing_registered_per_sample_per_segment_exact81_direction_gates"
        or later.get("direction_gate_authority_private_key")
        != PINNED_DIRECTION_GATE_PRIVATE_KEY
        or later.get("direction_gate_authority_public_key")
        != PINNED_DIRECTION_GATE_PUBLIC_KEY
        or later.get("direction_gate_authority_public_key_sha256")
        != PINNED_DIRECTION_GATE_PUBLIC_KEY_SHA256
        or hard_negative.get("gradient_source_authorized") is not False
        or hard_negative.get("use") != "NO_GO_OR_HARD_NEGATIVE_ONLY"
    ):
        raise CAPERRuntimeError("direction/update authority boundary differs")
    forbidden = value.get("forbidden_inputs")
    required_forbidden = {
        "mask",
        "track",
        "pose",
        "flow",
        "trajectory",
        "owner_rgb_in_editor_graph",
        "owner_latent_in_editor_graph",
        "owner_noise_in_editor_graph",
        "generation_gaussian_as_training_epsilon",
        "t2v_pixels_in_v2v_graph",
    }
    if not isinstance(forbidden, list) or set(forbidden) != required_forbidden:
        raise CAPERRuntimeError("forbidden input closure differs")
    return value


def two_wave_plan(asset: Mapping[str, Any]) -> tuple[tuple[tuple[str, int], ...], ...]:
    """Return the exact preregistered dog/human concurrent wave plan."""

    # The caller must have obtained ``asset`` through load_and_validate_asset.
    if (
        asset.get("schema_version") != ASSET_SCHEMA_VERSION
        or object_sha256(asset) != PINNED_ASSET_OBJECT_DIGEST
    ):
        raise CAPERRuntimeError("unvalidated asset supplied to two_wave_plan")
    observed = tuple(
        tuple((str(row["cell_id"]), int(row["query_seed"])) for row in wave["groups"])
        for wave in asset["waves"]
    )
    if observed != WAVE_PLAN:
        raise CAPERRuntimeError("two-wave plan changed")
    return observed


def build_preflight_decision(asset: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a machine-readable direction-only/no-training decision."""

    waves = two_wave_plan(asset)
    unsigned = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "method_name": METHOD_NAME,
        "world8_direction_materialization_runnable": True,
        "wave_count": 2,
        "query_count": 4,
        "waves": [
            [
                {
                    "cell_id": cell,
                    "owner_query_seed": seed,
                    "editor_noise_seed": editor_noise_seed(cell, seed),
                    "owner_editor_noise_seed_shared": False,
                }
                for cell, seed in wave
            ]
            for wave in waves
        ],
        "editor_packet_required_before_direction": True,
        "editor_runtime_materializer": "materialize_qmosaic_editor_runtime_v1.py",
        "legacy_ephemeral_materializer_authorized": False,
        "editor_runtime_authority": dict(asset["editor_runtime_authority"]),
        "actual_native_sampler_noise_observation_required": True,
        "cpu_gaussian_replay_is_only_an_equality_check": True,
        "editor_noise_domain_separation": "owner_query_seed_plus_1000",
        "semantic_direction_gate_materialized": False,
        "lora_b_vjp_authorized": False,
        "source_preservation_qp_authorized": False,
        "candidate_direct_add_authorized": False,
        "parameter_update_authorized": False,
        "training_claim_authorized": False,
        "next_blocker": (
            "independent_registered_per_sample_per_segment_exact81_"
            "base_plus_minus_evaluator"
        ),
    }
    return _sealed(unsigned)


def validate_observed_gaussian_provenance(
    provenance: Mapping[str, Any], *, editor_noise_seed: int
) -> Mapping[str, Any]:
    """Validate provenance produced by the read-only native sampler observer."""

    expected_keys = {
        "observer_call_count",
        "observer_only",
        "observer_returned_original_tensor_object",
        "observer_replaced_or_injected_noise",
        "requested_shape",
        "requested_dtype",
        "requested_device",
        "returned_dtype",
        "returned_device",
        "generator_device",
        "generator_initial_seed",
        "raw_value_sha256",
        "content_sha256",
        "cpu_generator_replay_exact_equal",
        "persisted_tensor_is_observer_capture_not_replay",
        "runtime",
        "role",
        "training_epsilon_reuse_authorized",
    }
    if set(provenance) != expected_keys:
        raise CAPERRuntimeError("Gaussian provenance field closure differs")
    runtime = provenance.get("runtime")
    if (
        provenance.get("observer_call_count") != 1
        or provenance.get("observer_only") is not True
        or provenance.get("observer_returned_original_tensor_object") is not True
        or provenance.get("observer_replaced_or_injected_noise") is not False
        or provenance.get("generator_device") != "cpu"
        or provenance.get("generator_initial_seed") != editor_noise_seed
        or provenance.get("requested_dtype") != "torch.float32"
        or provenance.get("returned_dtype") != "torch.float32"
        or provenance.get("requested_device") != provenance.get("returned_device")
        or provenance.get("cpu_generator_replay_exact_equal") is not True
        or provenance.get("persisted_tensor_is_observer_capture_not_replay") is not True
        or provenance.get("role") != "native_sampler_initial_noise_only"
        or provenance.get("training_epsilon_reuse_authorized") is not False
        or not isinstance(runtime, Mapping)
        or not runtime.get("torch")
        or not runtime.get("torch_hip")
        or not runtime.get("diffusers")
    ):
        raise CAPERRuntimeError("Gaussian is not a pinned observer-only capture")
    _sha256(provenance.get("raw_value_sha256"), label="Gaussian raw value")
    _sha256(provenance.get("content_sha256"), label="Gaussian content")
    shape = provenance.get("requested_shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 5
        or shape[:3] != [1, 16, LATENT_PHASES]
        or any(type(item) is not int or item <= 0 for item in shape)
    ):
        raise CAPERRuntimeError("Gaussian exact81 latent shape differs")
    return dict(provenance)


def _cell(asset: Mapping[str, Any], cell_id: str, query_seed: int) -> Mapping[str, Any]:
    if query_seed not in QUERY_SEEDS.get(cell_id, ()):
        raise CAPERRuntimeError("cell/query seed is outside fixed preregistration")
    cells = {row["cell_id"]: row for row in asset["cells"]}
    row = cells[cell_id]
    if query_seed not in row["query_seeds"]:
        raise CAPERRuntimeError("asset cell/query seed differs")
    return row


def editor_noise_seed(cell_id: str, owner_query_seed: int) -> int:
    """Map one owner-quotient query to its preregistered editor-noise domain."""

    owner_seeds = QUERY_SEEDS.get(cell_id)
    noise_seeds = EDITOR_NOISE_SEEDS.get(cell_id)
    if owner_seeds is None or noise_seeds is None or owner_query_seed not in owner_seeds:
        raise CAPERRuntimeError("owner query is outside editor-noise preregistration")
    value = noise_seeds[owner_seeds.index(owner_query_seed)]
    expected = qmosaic.editor_noise_seed_from_owner_query_seed(owner_query_seed)
    if value != expected or value == owner_query_seed:
        raise CAPERRuntimeError("owner/editor noise domains are not separated")
    return value


def _runtime_tensor_sha256(value: Any, *, label: str) -> str:
    return qmosaic.tensor_sha256(value.detach().float().cpu().contiguous(), label=label)


def _save_runtime_tensors_create_only(
    path: Path, tensors: Mapping[str, Any]
) -> Mapping[str, Any]:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    if set(tensors) != set(RUNTIME_TENSOR_KEYS):
        raise CAPERRuntimeError("runtime tensor key closure differs")
    normalized: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    for name in RUNTIME_TENSOR_KEYS:
        value = tensors[name]
        if not isinstance(value, torch.Tensor):
            raise CAPERRuntimeError(f"runtime tensor {name} is absent")
        stored = value.detach().float().cpu().contiguous().clone()
        if stored.requires_grad or not bool(torch.isfinite(stored).all().item()):
            raise CAPERRuntimeError(f"runtime tensor {name} differs")
        normalized[name] = stored
        hashes[name] = qmosaic.tensor_sha256(stored, label=f"stored runtime {name}")
    if path.exists() or path.is_symlink():
        raise CAPERRuntimeError("runtime safetensors path is not fresh")
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".safetensors", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        save_file(
            normalized,
            str(temporary),
            metadata={
                "schema_version": MATERIALIZER_SCHEMA_VERSION,
                "coordinate": "native_rv2v_source_clean_latent_sigma33",
                "official_initial_noise": "actual_sampler_observer_capture",
                "training_epsilon_reuse_authorized": "false",
            },
        )
        with safe_open(str(temporary), framework="pt", device="cpu") as opened:
            if list(opened.keys()) != sorted(RUNTIME_TENSOR_KEYS):
                raise CAPERRuntimeError("runtime safetensors key closure differs")
            for name in RUNTIME_TENSOR_KEYS:
                restored = opened.get_tensor(name).contiguous()
                if (
                    restored.dtype != torch.float32
                    or not torch.equal(restored, normalized[name])
                    or qmosaic.tensor_sha256(restored, label=f"roundtrip {name}")
                    != hashes[name]
                ):
                    raise CAPERRuntimeError(f"runtime tensor {name} round trip differs")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
    return {
        "path": str(path),
        "file_sha256": file_sha256(path),
        "tensor_keys": list(RUNTIME_TENSOR_KEYS),
        "tensor_sha256_by_key": hashes,
        "create_only": True,
    }


def _copy_source_create_only(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise CAPERRuntimeError("source snapshot path is not fresh")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".mp4"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as target, source.open("rb") as original:
            shutil.copyfileobj(original, target, length=1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _build_signed_editor_packet(
    *,
    cell: Mapping[str, Any],
    query_seed: int,
    owner_packet: Any,
    checkpoint_packet: Any,
    source_artifact: Mapping[str, Any],
    tensor_artifact: Mapping[str, Any],
    materializer_receipt_path: str | Path,
    materializer_receipt_file_sha256: str,
    materializer_receipt_digest: str,
    tokenizer_receipt_digest: str,
    text_encoder_receipt_digest: str,
) -> tuple[Mapping[str, Any], bytes]:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private = Ed25519PrivateKey.generate()
    public_bytes = private.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    public_sha = hashlib.sha256(public_bytes).hexdigest()
    artifact = {
        **dict(tensor_artifact),
        "materializer_receipt_digest": materializer_receipt_digest,
    }
    unsigned = {
        "schema_version": "bernini-qmosaic-signed-editor-runtime-input-v2",
        "cell_id": owner_packet.cell_id,
        "owner_query_seed": query_seed,
        "editor_noise_seed": editor_noise_seed(owner_packet.cell_id, query_seed),
        "source_iid": owner_packet.source_iid,
        "source_video_sha256": cell["source_video_sha256"],
        "action_prompt_utf8": cell["action_prompt_utf8"],
        "noop_prompt_utf8": cell["noop_prompt_utf8"],
        "action_prompt_sha256": cell["action_prompt_utf8_sha256"],
        "noop_prompt_sha256": cell["noop_prompt_utf8_sha256"],
        "owner_packet_receipt_digest": owner_packet.receipt()["digest"],
        "checkpoint_content_receipt_digest": checkpoint_packet.receipt()["digest"],
        "tokenizer_receipt_digest": tokenizer_receipt_digest,
        "text_encoder_receipt_digest": text_encoder_receipt_digest,
        "source_video_artifact": dict(source_artifact),
        "materialization_receipt_artifact": {
            "path": str(materializer_receipt_path),
            "file_sha256": materializer_receipt_file_sha256,
            "receipt_digest": materializer_receipt_digest,
        },
        "runtime_tensor_artifact": artifact,
        "authority_public_key_sha256": public_sha,
        "authority_signature_scheme": "Ed25519/canonical-json-ascii-v1",
    }
    signed_core = {**unsigned, "receipt_digest": object_sha256(unsigned)}
    signature = private.sign(canonical_json_bytes(signed_core))
    receipt = {
        **signed_core,
        "authority_signature_ed25519_base64": base64.b64encode(signature).decode("ascii"),
    }
    return receipt, public_bytes


def _all_rank_equal(value: Any, *, dist: Any, label: str) -> None:
    rows: list[Any] = [None] * int(dist.get_world_size())
    dist.all_gather_object(rows, value)
    if any(row != rows[0] for row in rows[1:]):
        raise CAPERRuntimeError(f"{label} differs across SP4 ranks")


def _unauthorized_legacy_materialize_editor_runtime_packet(
    args: argparse.Namespace,
) -> Mapping[str, Any]:
    """Permanently reject the superseded ephemeral-key/source-latent path.

    Historical implementation text remains below temporarily for audit
    archaeology, but this unconditional boundary makes it unreachable.  The
    only production packet path is ``materialize_qmosaic_editor_runtime_v1``.
    """

    raise CAPERRuntimeError(
        "legacy CAPER materializer is unauthorized; use "
        "materialize_qmosaic_editor_runtime_v1.py"
    )

    asset = load_and_validate_asset(args.asset, args.expected_asset_sha256)
    cell = _cell(asset, args.cell_id, args.query_seed)
    noise_seed = editor_noise_seed(args.cell_id, args.query_seed)
    output = Path(args.output_dir)
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise CAPERRuntimeError("output directory must be a fresh absolute path")
    _sha1(args.method_source_revision, label="method source revision")
    _sha256(args.method_source_archive_sha256, label="method source archive SHA-256")
    if args.expected_checkpoint_content_manifest_sha256 != PINNED_CHECKPOINT_MANIFEST_SHA256:
        raise CAPERRuntimeError("checkpoint content manifest is fixed")
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            legacy.trainer.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=legacy.trainer.BERNINI_OFFICIAL_COMMIT,
                expected_veomni_commit=legacy.trainer.VEOMNI_TESTED_COMMIT,
            )
        )
        checkpoint, transformer_config = legacy.trainer.validate_checkpoint(args.checkpoint)
    except legacy.trainer.TrainingContractError as error:
        raise CAPERRuntimeError(str(error)) from error
    if int(transformer_config.get("num_attention_heads", -1)) != 12:
        raise CAPERRuntimeError("pinned Bernini attention geometry differs")
    legacy.trainer.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from diffusers import __version__ as diffusers_version
    from diffusers.models import AutoencoderKLWan
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer, __version__ as transformers_version

    import bernini.models.wan_diffusion as wan_diffusion
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state
    from bernini.pipeline import _vae_encode

    distributed = legacy.inference_distributed_contract()
    if (
        distributed.world_size != WORLD_SIZE
        or distributed.ulysses_size != SP_SIZE
        or not torch.cuda.is_available()
        or getattr(torch.version, "hip", None) is None
    ):
        raise CAPERRuntimeError("packet materializer requires AUH WORLD4/SP4 ROCm")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=180),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=SP_SIZE)
    device = torch.device("cuda", distributed.local_rank)
    model: Any = None
    vae: Any = None
    try:
        checkpoint_packet = qmosaic.load_validated_checkpoint_content_manifest(
            checkpoint_root=checkpoint,
            content_manifest_path=args.checkpoint_content_manifest,
            expected_manifest_sha256=PINNED_CHECKPOINT_MANIFEST_SHA256,
            expected_file_count=23,
        )
        owner_root = asset["owner_authority"]["root"]
        owner = qmosaic.load_authenticated_owner_quotient_packet(
            registry=asset["owner_authority"]["registry"],
            expected_registry_sha256=PINNED_REGISTRY_SHA256,
            owner_root=owner_root,
            owner_master_receipt=asset["owner_authority"]["master_receipt"],
            expected_owner_master_receipt_sha256=PINNED_OWNER_MASTER_FILE_SHA256,
            audit_sidecar=asset["owner_authority"]["audit_sidecar"],
            expected_audit_sidecar_sha256=PINNED_OWNER_AUDIT_SIDECAR_SHA256,
            audit_evidence=asset["owner_authority"]["audit_evidence"],
            audit_public_key=asset["owner_authority"]["audit_public_key"],
            expected_audit_public_key_sha256=PINNED_OWNER_AUDIT_PUBLIC_KEY_SHA256,
            cell_root=cell["quotient_cell_root"],
            receipt_path=cell["quotient_receipt"],
            expected_receipt_file_sha256=PINNED_CELL_RECEIPT_FILE_SHA256[args.cell_id],
            query_seed=args.query_seed,
        )
        if (
            owner.cell_id != args.cell_id
            or owner.query_seed != args.query_seed
            or owner.source_iid != cell["source_iid"]
            or owner.source_video_sha256 != cell["source_video_sha256"]
            or owner.action_prompt_sha256 != cell["action_prompt_utf8_sha256"]
            or owner.noop_prompt_sha256 != cell["noop_prompt_utf8_sha256"]
            or owner.action_family_id != cell["action_family_id"]
        ):
            raise CAPERRuntimeError("authenticated owner and asset cell differ")

        source_path = Path(cell["source_video"])
        source_tensor, source_metadata, source_sha = source_audit.prepare_hashed_source_snapshot(
            source_path
        )
        if source_sha != cell["source_video_sha256"]:
            raise CAPERRuntimeError("source video bytes changed")
        bucket_hw = tuple(int(item) for item in source_metadata["source_derived_bucket_hw"])
        if list(cell["latent_shape"])[3:] != [bucket_hw[0] // 8, bucket_hw[1] // 8]:
            raise CAPERRuntimeError("source-derived bucket differs from preregistered latent")

        tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint), subfolder="tokenizer", **legacy.tokenizer_load_kwargs()
        )
        if tokenizer.padding_side != "right" or tokenizer.init_kwargs.get(
            "fix_mistral_regex"
        ) is not True:
            raise CAPERRuntimeError("pinned tokenizer contract differs")
        task_prompts = {
            role: native.build_task_prompt(
                "rv2v", cell[f"{role}_prompt_utf8"], prompt_cleaner=prompt_clean
            )
            for role in ("action", "noop")
        }
        token_pairs = {
            role: legacy._tokenize_training_prompt(tokenizer, prompt)
            for role, prompt in task_prompts.items()
        }
        action_ids, action_mask = token_pairs["action"]

        config = BerniniRendererConfig.from_pretrained(
            str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
            local_files_only=True,
            **legacy.inference_renderer_config_overrides(checkpoint),
        )
        config.dtype = torch.bfloat16
        legacy.trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
        if float(config.shift) != native.FLOW_SHIFT or config.use_unipc is not True:
            raise CAPERRuntimeError("renderer scheduler contract differs")
        model = BerniniRendererModel(config).requires_grad_(False).eval()
        model.to(device)
        conditions: dict[str, Any] = {}
        with torch.inference_mode():
            for role, (ids, mask) in token_pairs.items():
                condition = model.encode_prompt(ids.to(device), mask.to(device)).detach().float()
                native._broadcast_condition_from_rank_zero(  # noqa: SLF001
                    condition, label=f"CAPER {role} condition", world_size=WORLD_SIZE
                )
                if tuple(map(int, condition.shape)) != (1, 512, 4096):
                    raise CAPERRuntimeError(f"{role} prompt embedding geometry differs")
                conditions[role] = condition.contiguous()
        if torch.equal(conditions["action"], conditions["noop"]):
            raise CAPERRuntimeError("action and no-op prompt embeddings alias")

        vae = AutoencoderKLWan.from_pretrained(
            str(checkpoint),
            subfolder="vae",
            torch_dtype=torch.float32,
            local_files_only=True,
        ).eval().requires_grad_(False).to(device)
        source_pixels = source_tensor.to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            source_latent = _vae_encode(vae, source_pixels).detach().float().contiguous()
            references = [
                _vae_encode(
                    vae,
                    source_pixels[:, :, frame : frame + 1, :, :].contiguous(),
                ).detach().float().contiguous()
                for frame in native.RV2V_REFERENCE_INDICES
            ]
        native._broadcast_condition_from_rank_zero(  # noqa: SLF001
            source_latent, label="CAPER full source latent", world_size=WORLD_SIZE
        )
        for index, reference in enumerate(references):
            native._broadcast_condition_from_rank_zero(  # noqa: SLF001
                reference, label=f"CAPER source reference {index}", world_size=WORLD_SIZE
            )
        expected_shape = tuple(int(item) for item in cell["latent_shape"])
        if tuple(source_latent.shape) != expected_shape or any(
            tuple(reference.shape) != (1, 16, 1, expected_shape[3], expected_shape[4])
            for reference in references
        ):
            raise CAPERRuntimeError("source/reference VAE geometry differs")
        vae.to("cpu")
        del source_pixels
        torch.cuda.empty_cache()

        condition_kwargs = native.select_native_conditions(
            "rv2v",
            full_source_latent=source_latent,
            reference_latents={
                frame: reference
                for frame, reference in zip(native.RV2V_REFERENCE_INDICES, references)
            },
        )
        negative_ids, negative_mask = legacy._tokenize_renderer_negative(
            tokenizer, legacy.DEFAULT_NEGATIVE_PROMPT
        )
        with torch.inference_mode():
            generated_latent, capture = native._sample_with_native_initial_noise_observer(  # noqa: SLF001
                sample_fn=lambda: model.sample(
                    input_ids=action_ids.to(device),
                    attention_mask=action_mask.to(device),
                    uncond_input_ids=negative_ids.to(device),
                    uncond_attention_mask=negative_mask.to(device),
                    **condition_kwargs,
                    width=bucket_hw[1],
                    height=bucket_hw[0],
                    device=device,
                    **native.native_sampling_contract(
                        "rv2v", steps=NATIVE_STEPS, seed=noise_seed
                    ),
                ),
                wan_diffusion_module=wan_diffusion,
                expected_shape=expected_shape,
                expected_device=device,
                expected_seed=noise_seed,
            )
        if tuple(generated_latent.shape) != expected_shape:
            raise CAPERRuntimeError("native RV2V sampler output geometry differs")
        capture_identity = native._all_rank_tensor_identity(  # noqa: SLF001
            capture.tensor,
            label="CAPER observed official Gaussian",
            world_size=WORLD_SIZE,
        )
        if capture_identity.get("all_rank_exact") is not True:
            raise CAPERRuntimeError("observed official Gaussian differs across SP4 ranks")
        replay_generator = torch.Generator(device="cpu")
        replay_generator.manual_seed(noise_seed)
        replay = torch.randn(expected_shape, generator=replay_generator, dtype=torch.float32)
        if not torch.equal(replay, capture.tensor):
            raise CAPERRuntimeError(
                "actual sampler Gaussian differs from pinned CPU-generator replay"
            )
        gaussian_provenance = validate_observed_gaussian_provenance(
            {
                "observer_call_count": capture.call_count,
                "observer_only": True,
                "observer_returned_original_tensor_object": True,
                "observer_replaced_or_injected_noise": False,
                "requested_shape": list(capture.requested_shape),
                "requested_dtype": capture.requested_dtype,
                "requested_device": capture.requested_device,
                "returned_dtype": capture.returned_dtype,
                "returned_device": capture.returned_device,
                "generator_device": capture.generator_device,
                "generator_initial_seed": capture.generator_initial_seed,
                "raw_value_sha256": capture.raw_value_sha256,
                "content_sha256": capture.content_sha256,
                "cpu_generator_replay_exact_equal": True,
                "persisted_tensor_is_observer_capture_not_replay": True,
                "runtime": {
                    "torch": torch.__version__,
                    "torch_hip": str(torch.version.hip),
                    "diffusers": diffusers_version,
                    "transformers": transformers_version,
                },
                "role": "native_sampler_initial_noise_only",
                "training_epsilon_reuse_authorized": False,
            },
            editor_noise_seed=noise_seed,
        )
        del generated_latent, replay

        tensors = {
            "source_latent": source_latent,
            "image_reference_0": references[0],
            "image_reference_1": references[1],
            "image_reference_2": references[2],
            "image_reference_3": references[3],
            "clean_latent": source_latent.clone(),
            "official_initial_noise": capture.tensor,
            "action_condition": conditions["action"],
            "noop_condition": conditions["noop"],
            "timestep": torch.tensor([float(NATIVE_TIMESTEP)], dtype=torch.float32),
        }
        tensor_digests = {
            name: _runtime_tensor_sha256(value, label=f"CAPER prepublish {name}")
            for name, value in tensors.items()
        }
        _all_rank_equal(tensor_digests, dist=dist, label="runtime tensor values")
        tokenizer_receipt_digest = object_sha256(
            {
                "schema_version": "bernini-caper-tokenizer-binding-v1",
                "checkpoint_content_receipt_digest": checkpoint_packet.receipt()["digest"],
                "task": "vr2v",
                "raw_prompt_sha256_by_role": {
                    role: cell[f"{role}_prompt_utf8_sha256"]
                    for role in ("action", "noop")
                },
                "task_prompt_sha256_by_role": {
                    role: hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                    for role, prompt in task_prompts.items()
                },
                "input_ids_sha256_by_role": {
                    role: qmosaic.tensor_sha256(ids.detach().cpu(), label=f"{role} ids")
                    for role, (ids, _mask) in token_pairs.items()
                },
                "attention_mask_sha256_by_role": {
                    role: qmosaic.tensor_sha256(mask.detach().cpu(), label=f"{role} mask")
                    for role, (_ids, mask) in token_pairs.items()
                },
            }
        )
        text_encoder_receipt_digest = object_sha256(
            {
                "schema_version": "bernini-caper-text-encoder-binding-v1",
                "checkpoint_content_receipt_digest": checkpoint_packet.receipt()["digest"],
                "action_condition_sha256": tensor_digests["action_condition"],
                "noop_condition_sha256": tensor_digests["noop_condition"],
                "shape": [1, 512, 4096],
                "all_rank_exact": True,
            }
        )
        _all_rank_equal(
            {
                "tokenizer": tokenizer_receipt_digest,
                "text_encoder": text_encoder_receipt_digest,
            },
            dist=dist,
            label="prompt encoder receipts",
        )

        publish_result: list[Any] = [None]
        if distributed.rank == 0:
            try:
                output.mkdir(mode=0o700)
                source_snapshot = output / "source.mp4"
                _copy_source_create_only(source_path, source_snapshot)
                if file_sha256(source_snapshot) != cell["source_video_sha256"]:
                    raise CAPERRuntimeError("source snapshot copy differs")
                tensor_path = output / "editor-runtime-input.safetensors"
                tensor_artifact = _save_runtime_tensors_create_only(tensor_path, tensors)
                materializer_unsigned = {
                    "schema_version": MATERIALIZER_SCHEMA_VERSION,
                    "method_name": METHOD_NAME,
                    "cell_id": args.cell_id,
                    "query_seed": args.query_seed,
                    "owner_query_seed": args.query_seed,
                    "editor_noise_seed": noise_seed,
                    "owner_editor_noise_seed_shared": False,
                    "source_iid": cell["source_iid"],
                    "action_family_id": cell["action_family_id"],
                    "owner_master_receipt_digest": PINNED_OWNER_MASTER_RECEIPT_DIGEST,
                    "owner_generation_receipt_digest": PINNED_GENERATION_RECEIPT_DIGEST[
                        args.cell_id
                    ],
                    "owner_packet_receipt_digest": owner.receipt()["digest"],
                    "quotient_master_receipt_digest": PINNED_QUOTIENT_MASTER_RECEIPT_DIGEST,
                    "quotient_cell_receipt_digest": PINNED_CELL_RECEIPT_DIGEST[args.cell_id],
                    "checkpoint_content_receipt_digest": checkpoint_packet.receipt()["digest"],
                    "source_video_sha256": cell["source_video_sha256"],
                    "role_prompt_binding_digest": cell["role_prompt_binding_digest"],
                    "tokenizer_receipt_digest": tokenizer_receipt_digest,
                    "text_encoder_receipt_digest": text_encoder_receipt_digest,
                    "runtime_tensor_sha256_by_key": tensor_digests,
                    "native_sampler_gaussian": gaussian_provenance,
                    "source_condition": {
                        "full_source_latent": True,
                        "independently_vae_encoded_reference_frame_indices": list(
                            native.RV2V_REFERENCE_INDICES
                        ),
                        "reference_count": 4,
                        "clean_latent_is_source_video_vae_encode": True,
                    },
                    "sampler": {
                        "mode": "rv2v",
                        "frame_count": EXACT_FRAMES,
                        "latent_phases": LATENT_PHASES,
                        "steps": NATIVE_STEPS,
                        "owner_query_seed": args.query_seed,
                        "editor_noise_seed": noise_seed,
                        "owner_editor_noise_seed_shared": False,
                        "native_sampler_output_persisted": False,
                        "native_sampler_output_used_as_supervision": False,
                    },
                    "authority_boundary": {
                        "packet_materialization_only": True,
                        "direction_measurement_authorized": True,
                        "semantic_action_gate_passed": False,
                        "lora_vjp_authorized": False,
                        "optimizer_created": False,
                        "parameter_update_performed": False,
                        "adapter_checkpoint_written": False,
                        "generation_gaussian_as_training_epsilon": False,
                        "masks_tracks_pose_flow_trajectory_consumed": False,
                    },
                    "runtime_source": {
                        "method_revision": args.method_source_revision,
                        "method_archive_sha256": args.method_source_archive_sha256,
                        "bernini_revision": bernini_revision,
                        "veomni_revision": veomni_revision,
                    },
                }
                materializer_receipt = _sealed(materializer_unsigned)
                materializer_path = output / "materializer.receipt.json"
                materializer_file_sha = _write_json_create_only(
                    materializer_path, materializer_receipt
                )
                editor_receipt, public_bytes = _build_signed_editor_packet(
                    cell=cell,
                    query_seed=args.query_seed,
                    owner_packet=owner,
                    checkpoint_packet=checkpoint_packet,
                    source_artifact={
                        "path": str(source_snapshot),
                        "file_sha256": cell["source_video_sha256"],
                    },
                    tensor_artifact=tensor_artifact,
                    materializer_receipt_path=materializer_path,
                    materializer_receipt_file_sha256=materializer_file_sha,
                    materializer_receipt_digest=materializer_receipt["receipt_digest"],
                    tokenizer_receipt_digest=tokenizer_receipt_digest,
                    text_encoder_receipt_digest=text_encoder_receipt_digest,
                )
                public_path = output / "editor-runtime-ed25519-public.pem"
                _write_create_only(public_path, public_bytes)
                receipt_path = output / "editor-runtime-input.receipt.json"
                receipt_file_sha = _write_json_create_only(receipt_path, editor_receipt)
                loaded = qmosaic.load_authenticated_editor_runtime_input_packet(
                    receipt_path=receipt_path,
                    expected_receipt_file_sha256=receipt_file_sha,
                    public_key_path=public_path,
                    expected_public_key_file_sha256=file_sha256(public_path),
                    artifact_root=output,
                    owner=owner,
                    checkpoint=checkpoint_packet,
                )
                if loaded.payload["runtime_tensor_artifact"][
                    "materializer_receipt_digest"
                ] != materializer_receipt["receipt_digest"]:
                    raise CAPERRuntimeError("editor packet lost materializer binding")
                publish_result[0] = {
                    "ok": True,
                    "editor_receipt_path": str(receipt_path),
                    "editor_receipt_file_sha256": receipt_file_sha,
                    "editor_receipt_digest": editor_receipt["receipt_digest"],
                    "editor_public_key_path": str(public_path),
                    "editor_public_key_file_sha256": file_sha256(public_path),
                    "editor_artifact_root": str(output),
                    "materializer_receipt_path": str(materializer_path),
                    "materializer_receipt_file_sha256": materializer_file_sha,
                    "materializer_receipt_digest": materializer_receipt["receipt_digest"],
                    "direction_measurement_authorized": True,
                    "lora_vjp_authorized": False,
                    "parameter_update_authorized": False,
                }
            except Exception as error:
                publish_result[0] = {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
        dist.broadcast_object_list(publish_result, src=0)
        result = publish_result[0]
        if not isinstance(result, Mapping) or result.get("ok") is not True:
            raise CAPERRuntimeError(f"rank-zero packet publication failed: {result}")
        _all_rank_equal(
            result["editor_receipt_digest"], dist=dist, label="editor packet receipt"
        )
        return dict(result)
    finally:
        if vae is not None:
            try:
                vae.to("cpu")
            except Exception:
                pass
        if model is not None:
            try:
                model.to("cpu")
            except Exception:
                pass
        if dist.is_initialized():
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--asset", required=True)
    preflight.add_argument("--expected-asset-sha256", required=True)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        asset = load_and_validate_asset(args.asset, args.expected_asset_sha256)
        decision = build_preflight_decision(asset)
        print(canonical_json_bytes(decision).decode("ascii"), flush=True)
        return 0
    raise CAPERRuntimeError("unsupported CAPER preflight command")


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())


__all__ = [
    "ASSET_SCHEMA_VERSION",
    "CAPERRuntimeError",
    "METHOD_NAME",
    "MATERIALIZER_SCHEMA_VERSION",
    "PINNED_ASSET_SHA256",
    "PINNED_EDITOR_RUNTIME_PRIVATE_KEY",
    "PINNED_EDITOR_RUNTIME_PUBLIC_KEY",
    "PINNED_EDITOR_RUNTIME_PUBLIC_KEY_SHA256",
    "PINNED_CHECKPOINT_MANIFEST_SHA256",
    "QUERY_SEEDS",
    "RUNTIME_TENSOR_KEYS",
    "WAVE_PLAN",
    "build_preflight_decision",
    "canonical_json_bytes",
    "editor_noise_seed",
    "file_sha256",
    "load_and_validate_asset",
    "object_sha256",
    "two_wave_plan",
    "validate_observed_gaussian_provenance",
]
