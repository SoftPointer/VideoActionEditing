#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import full30_action_data_teacher_authority_v1 as authority
import full30_action_psiout_materializer_v1 as materializer


EVENTS = (
    ("dog-rise", "dog", "dog-low-q0"),
    ("dog-reach", "dog", "dog-standing-q0"),
    ("human-stand", "human", "human-kneeling-q0"),
    ("human-wave", "human", "human-arms-low-q0"),
)


def _write(root: Path, relative: str, payload: bytes) -> tuple[str, str]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return str(path.resolve()), hashlib.sha256(payload).hexdigest()


def _write_json(root: Path, relative: str, value: object, *, mode: int | None = None) -> tuple[str, str]:
    raw = authority.canonical_json_bytes(value) + b"\n"
    path, digest = _write(root, relative, raw)
    if mode is not None:
        Path(path).chmod(mode)
    return path, digest


def _write_fp32_artifact(
    root: Path,
    relative: str,
    *,
    tensor_key: str,
    seed_value: float,
) -> dict[str, object]:
    shape = [1, 16, 21, 2, 2]
    count = math.prod(shape)
    values = [0.0] * count
    values[0] = float(seed_value)
    values[1] = float(seed_value) + 0.125
    payload = struct.pack(f"<{count}f", *values)
    header = {
        tensor_key: {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [0, len(payload)],
        }
    }
    header_bytes = authority.canonical_json_bytes(header)
    raw = struct.pack("<Q", len(header_bytes)) + header_bytes + payload
    path, file_sha = _write(root, relative, raw)
    return {
        "schema_version": authority._MATERIALIZATION_ARTIFACT_SCHEMA,
        "path": path,
        "file_sha256": file_sha,
        "tensor_key": tensor_key,
        "tensor_raw_sha256": hashlib.sha256(payload).hexdigest(),
        "dtype": "float32-le",
        "shape": shape,
    }


def _pack_f32(values: object) -> bytes:
    sequence = tuple(values)  # type: ignore[arg-type]
    if len(sequence) != authority.TENSOR_ELEMENTS:
        raise AssertionError("fixture tensor shape differs")
    return struct.pack(f"<{authority.TENSOR_ELEMENTS}f", *sequence)


def _unpack_f32(payload: bytes) -> tuple[float, ...]:
    return struct.unpack(f"<{authority.TENSOR_ELEMENTS}f", payload)


def _fixture_norm(values: tuple[float, ...]) -> float:
    return math.sqrt(math.fsum(float(value) * float(value) for value in values))


def _fixture_cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.fsum(float(a) * float(b) for a, b in zip(left, right)) / (
        _fixture_norm(left) * _fixture_norm(right)
    )


def _basis(index: int, scale: float = 1.0) -> tuple[float, ...]:
    values = [0.0] * authority.TENSOR_ELEMENTS
    values[index] = scale
    return tuple(values)


def _unit_with_cosine(cosine: float, primary: int, orthogonal: int) -> tuple[float, ...]:
    values = [0.0] * authority.TENSOR_ELEMENTS
    values[primary] = cosine
    values[orthogonal] = math.sqrt(1.0 - cosine * cosine)
    return tuple(values)


def _representation_material() -> tuple[
    dict[str, tuple[float, ...]],
    dict[str, tuple[float, ...]],
    dict[str, tuple[float, ...]],
    dict[str, tuple[float, ...]],
]:
    origin: dict[str, tuple[float, ...]] = {}
    cross: dict[str, tuple[float, ...]] = {}
    origin_nuisance: dict[str, tuple[float, ...]] = {}
    cross_nuisance: dict[str, tuple[float, ...]] = {}
    zero = tuple(0.0 for _ in range(authority.TENSOR_ELEMENTS))
    for ordinal, sigma_index in enumerate(authority.SIGMA_INDICES):
        primary = ordinal * 12
        origin_values = {
            "projected_unit": _basis(primary),
            "projected_raw": _basis(primary, 1.0e-3),
            "duplicate_forward_first": _basis(primary, 0.25),
            "duplicate_forward_second": _basis(primary, 0.25),
            "noop_forward_first": zero,
            "noop_forward_second": zero,
            "wrong_actor_projected_unit": _unit_with_cosine(0.40, primary, primary + 2),
            "wrong_object_projected_unit": _unit_with_cosine(0.40, primary, primary + 3),
            "generic_wrong_motion_projected_unit": _unit_with_cosine(
                0.40, primary, primary + 4
            ),
        }
        for kind in authority._ORIGIN_PSIOUT_TENSOR_KINDS:
            origin[authority._tensor_name(sigma_index, kind)] = origin_values[kind]
        cross[authority._tensor_name(sigma_index, "projected_unit")] = _unit_with_cosine(
            0.75, primary, primary + 1
        )
        nuisance_values = {
            "camera_unit": _basis(primary + 8),
            "appearance_unit": _basis(primary + 9),
        }
        for kind in authority._NUISANCE_TENSOR_KINDS:
            name = authority._tensor_name(sigma_index, kind)
            origin_nuisance[name] = nuisance_values[kind]
            cross_nuisance[name] = nuisance_values[kind]
    return origin, cross, origin_nuisance, cross_nuisance


def _write_tensor_container(
    root: Path,
    relative: str,
    *,
    container_kind: str,
    evidence_id: str,
    evidence_role: str,
    teacher_cell_id: str,
    branch: str,
    tensors: dict[str, tuple[float, ...]],
) -> tuple[str, str, dict[str, bytes]]:
    expected_names = authority._expected_tensor_names(container_kind, evidence_role)
    if tuple(tensors) != expected_names:
        raise AssertionError("fixture tensor names/order differ")
    payload_parts: list[bytes] = []
    packed: dict[str, bytes] = {}
    entries: list[dict[str, object]] = []
    for ordinal, name in enumerate(expected_names):
        tensor_bytes = _pack_f32(tensors[name])
        packed[name] = tensor_bytes
        payload_parts.append(tensor_bytes)
        entries.append(
            {
                "name": name,
                "dtype": authority.TENSOR_DTYPE,
                "shape": list(authority.TENSOR_SHAPE),
                "offset": ordinal * authority.TENSOR_SLICE_BYTES,
                "length": authority.TENSOR_SLICE_BYTES,
                "sha256": hashlib.sha256(tensor_bytes).hexdigest(),
            }
        )
    payload = b"".join(payload_parts)
    header = {
        "schema_version": authority.TENSOR_CONTAINER_SCHEMA,
        "container_kind": container_kind,
        "evidence_id": evidence_id,
        "evidence_role": evidence_role,
        "teacher_cell_id": teacher_cell_id,
        "branch": branch,
        "dtype": authority.TENSOR_DTYPE,
        "shape": list(authority.TENSOR_SHAPE),
        "sigma_indices": list(authority.SIGMA_INDICES),
        "layout": authority.TENSOR_CONTAINER_LAYOUT,
        "tensor_count": len(entries),
        "payload_bytes": len(payload),
        "entries": entries,
    }
    header_bytes = authority.canonical_json_bytes(header)
    raw = (
        authority.TENSOR_CONTAINER_MAGIC
        + struct.pack(">I", len(header_bytes))
        + header_bytes
        + payload
    )
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(authority.TENSOR_CONTAINER_MODE)
    return str(path.resolve()), hashlib.sha256(raw).hexdigest(), packed


def _rewrite_tensor(
    path_value: object, name: str, values: tuple[float, ...]
) -> str:
    path = Path(path_value)  # type: ignore[arg-type]
    raw = path.read_bytes()
    prefix_bytes = len(authority.TENSOR_CONTAINER_MAGIC) + 4
    header_length = struct.unpack(
        ">I", raw[len(authority.TENSOR_CONTAINER_MAGIC) : prefix_bytes]
    )[0]
    header = json.loads(raw[prefix_bytes : prefix_bytes + header_length])
    payload = bytearray(raw[prefix_bytes + header_length :])
    entry = next(item for item in header["entries"] if item["name"] == name)
    tensor_bytes = _pack_f32(values)
    start = entry["offset"]
    payload[start : start + entry["length"]] = tensor_bytes
    entry["sha256"] = hashlib.sha256(tensor_bytes).hexdigest()
    header_bytes = authority.canonical_json_bytes(header)
    rewritten = (
        authority.TENSOR_CONTAINER_MAGIC
        + struct.pack(">I", len(header_bytes))
        + header_bytes
        + bytes(payload)
    )
    path.write_bytes(rewritten)
    path.chmod(authority.TENSOR_CONTAINER_MODE)
    return hashlib.sha256(rewritten).hexdigest()


def _rewrite_container_header(path_value: object, mutate: object) -> str:
    path = Path(path_value)  # type: ignore[arg-type]
    raw = path.read_bytes()
    prefix_bytes = len(authority.TENSOR_CONTAINER_MAGIC) + 4
    header_length = struct.unpack(
        ">I", raw[len(authority.TENSOR_CONTAINER_MAGIC) : prefix_bytes]
    )[0]
    header = json.loads(raw[prefix_bytes : prefix_bytes + header_length])
    mutate(header)  # type: ignore[operator]
    header_bytes = authority.canonical_json_bytes(header)
    rewritten = (
        authority.TENSOR_CONTAINER_MAGIC
        + struct.pack(">I", len(header_bytes))
        + header_bytes
        + raw[prefix_bytes + header_length :]
    )
    path.write_bytes(rewritten)
    path.chmod(authority.TENSOR_CONTAINER_MODE)
    return hashlib.sha256(rewritten).hexdigest()


def _seal(value: dict[str, object], field: str) -> dict[str, object]:
    return authority.seal_record(value, field)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _resign_top(manifest: dict[str, object]) -> None:
    manifest["manifest_digest"] = authority.object_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_digest"}
    )


def _resign_nested(row: dict[str, object], field: str) -> None:
    row[field] = authority.object_sha256(
        {key: value for key, value in row.items() if key != field}
    )


def _representation_anchor_evidence(
    root: Path,
    *,
    origin: dict[str, object],
    branch: str,
    cell_ordinal: int,
    evidence_role: str,
    psiout_tensors: dict[str, tuple[float, ...]],
    nuisance_tensors: dict[str, tuple[float, ...]],
) -> tuple[dict[str, object], dict[str, bytes], dict[str, bytes]]:
    cell = str(origin["teacher_cell_id"])
    if evidence_role == "teacher_origin":
        anchor_id = f"origin-anchor:{cell}:{branch}"
        anchor_iid = str(origin["origin_iid"])
        anchor_split = str(origin["analysis_split"])
        actor_id = str(origin["actor_id"])
        scene_id = str(origin["scene_id"])
    else:
        anchor_id = f"cross-anchor:{cell}:{branch}"
        intrinsic_key = f"{origin['analysis_split']}:{origin['event_id']}:{branch}"
        anchor_iid = hashlib.sha256(intrinsic_key.encode("utf-8")).hexdigest()[:16]
        anchor_split = "confirmation" if origin["analysis_split"] == "fit" else "fit"
        actor_id = f"cross-actor-{origin['analysis_split']}-{origin['event_id']}"
        scene_id = f"cross-scene-{origin['analysis_split']}-{origin['event_id']}"
    evidence_id = f"evidence:{evidence_role}:{cell}:{branch}"
    prefix = f"{evidence_role}-{cell}-{branch}"
    video_prefix = (
        prefix
        if evidence_role == "teacher_origin"
        else f"{evidence_role}-{origin['analysis_split']}-{origin['event_id']}-{branch}"
    )
    video_path, video_sha = _write(
        root,
        f"representation-video/{video_prefix}.mp4",
        f"representation-video-{video_prefix}".encode(),
    )
    sidecar_path, sidecar_sha, packed_psiout = _write_tensor_container(
        root,
        f"psiout/{prefix}.f30tc",
        container_kind="psiout",
        evidence_id=evidence_id,
        evidence_role=evidence_role,
        teacher_cell_id=cell,
        branch=branch,
        tensors=psiout_tensors,
    )
    nuisance_path, nuisance_sha, packed_nuisance = _write_tensor_container(
        root,
        f"nuisance/{prefix}.f30tc",
        container_kind="nuisance",
        evidence_id=evidence_id,
        evidence_role=evidence_role,
        teacher_cell_id=cell,
        branch=branch,
        tensors=nuisance_tensors,
    )
    evidence_unsigned: dict[str, object] = {
        "schema_version": authority.REPRESENTATION_EVIDENCE_SCHEMA,
        "evidence_id": evidence_id,
        "evidence_role": evidence_role,
        "teacher_cell_id": cell,
        "anchor_id": anchor_id,
        "anchor_iid": anchor_iid,
        "anchor_split": anchor_split,
        "branch": branch,
        "event_id": origin["event_id"],
        "actor_kind": origin["actor_kind"],
        "q0_id": origin["q0_id"],
        "actor_id": actor_id,
        "scene_id": scene_id,
        "anchor_video_path": video_path,
        "anchor_video_sha256": video_sha,
        "psiout_sidecar_path": sidecar_path,
        "psiout_sidecar_sha256": sidecar_sha,
        "nuisance_packet_path": nuisance_path,
        "nuisance_packet_sha256": nuisance_sha,
        "all_tensor_values_finite": True,
    }
    review_unsigned = {
        "schema_version": authority.REPRESENTATION_REVIEW_SCHEMA,
        "review_id": f"representation-review:{evidence_role}:{cell}:{branch}",
        "evidence_id": evidence_id,
        "anchor_id": anchor_id,
        "anchor_video_sha256": video_sha,
        "anchor_split": anchor_split,
        "branch": branch,
        "event_id": origin["event_id"],
        "actor_kind": origin["actor_kind"],
        "q0_id": origin["q0_id"],
        "actor_id": actor_id,
        "scene_id": scene_id,
        "frame_count": 81,
        "fps": 25.0,
        "entire_full81_video_viewed": True,
        "independent_reviewer": True,
        "reviewer_blinded_to_teacher_cell": True,
        "reviewer_blinded_to_representation_metrics": True,
        "sealed_before_sidecar_extraction": True,
        "sealed_before_representation_admission": True,
        "target_event_verified": True,
        "actor_identity_verified": True,
        "scene_verified": True,
    }
    evidence_unsigned["pre_admission_blind_review"] = _seal(
        review_unsigned, "review_digest"
    )
    return (
        _seal(evidence_unsigned, "evidence_digest"),
        packed_psiout,
        packed_nuisance,
    )


def _materialization_runtime_fixture(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    method_root = METHOD_ROOT.resolve()
    runtime_root = root / "materialization-runtime"
    bernini_root = runtime_root / "bernini"
    veomni_root = runtime_root / "veomni"
    checkpoint_root = runtime_root / "checkpoint"
    for path in (bernini_root, veomni_root, checkpoint_root):
        path.mkdir(parents=True, exist_ok=True)
    transformer_config_path, transformer_config_sha = _write_json(
        root,
        "materialization-runtime/checkpoint/transformer/config.json",
        {"fixture": "official-materializer-runtime"},
    )
    if Path(transformer_config_path) != (
        checkpoint_root / "transformer" / "config.json"
    ).resolve():
        raise AssertionError("fixture transformer config path differs")
    checkpoint_manifest_path, checkpoint_manifest_sha = _write_json(
        root,
        "materialization-runtime/checkpoint-content.json",
        {"tree_sha256": "2" * 64},
    )
    helpers = [
        {
            "module": module,
            "path": str((method_root / f"{module}.py").resolve()),
            "file_sha256": authority.file_sha256(method_root / f"{module}.py"),
        }
        for module in authority._REQUIRED_HELPER_MODULES
    ]
    compute = dict(materializer.frozen_compute_contract_v1())
    runtime_identity = _seal(
        {
            "schema_version": authority.MATERIALIZATION_RUNTIME_SCHEMA,
            "bernini_revision": "1" * 40,
            "veomni_revision": "6" * 40,
            "official_checkpoint_tree_sha256": "2" * 64,
            "transformer_config_sha256": transformer_config_sha,
            "sigma_table_sha256": authority.PINNED_SIGMA_TABLE_SHA256,
            "psiout_protocol_sha256": authority.PINNED_PSIOUT_PROTOCOL_SHA256,
            "official_provider_source_sha256": authority.file_sha256(
                method_root / "full30_action_psiout_materializer_v1.py"
            ),
            "official_provider_abi": authority.MATERIALIZATION_PROVIDER_ABI,
            "compute_contract": compute,
            "compute_contract_digest": authority.object_sha256(compute),
            "frame_count": 81,
            "fps": 25.0,
            "sampler_steps": 40,
        },
        "runtime_digest",
    )
    runtime_plan = _seal(
        {
            "schema_version": authority._MATERIALIZATION_RUNTIME_PLAN_SCHEMA,
            "frozen_runtime_identity": runtime_identity,
            "bernini_root": str(bernini_root.resolve()),
            "veomni_root": str(veomni_root.resolve()),
            "checkpoint_root": str(checkpoint_root.resolve()),
            "checkpoint_content_manifest_path": checkpoint_manifest_path,
            "checkpoint_content_manifest_sha256": checkpoint_manifest_sha,
            "psiout_protocol_path": str(
                (method_root / "full30_action_learning_v1.py").resolve()
            ),
            "official_provider_source_path": str(
                (method_root / "full30_action_psiout_materializer_v1.py").resolve()
            ),
            "official_helper_sources": helpers,
        },
        "runtime_plan_digest",
    )
    return runtime_identity, runtime_plan


def _materialization_output_policy() -> dict[str, object]:
    return {
        "schema_version": authority.MATERIALIZATION_OUTPUT_POLICY_SCHEMA,
        "create_only": True,
        "container_mode_octal": "0600",
        "generated_rgb_decoded": False,
        "generated_rgb_used_as_model_input": False,
        "generated_rgb_used_as_regression_target": False,
        "generated_latent_used_as_absolute_regression_target": False,
        "model_parameters_updated": False,
        "optimizer_created": False,
        "persisted_tensor_role": (
            "detached-post-head-psiout-or-same-mode-amplitude-evidence-only"
        ),
    }


def _materialization_condition_fixture(
    root: Path,
    *,
    record_id: str,
    role: str,
    control_anchor_id: object,
    instruction_override: str | None = None,
) -> dict[str, object]:
    instruction = (
        f"Materialized condition {role} for {record_id}."
        if instruction_override is None
        else instruction_override
    )
    instruction_sha = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
    authority_value = _seal(
        {
            "condition": {
                "instruction": instruction,
                "instruction_utf8_sha256": instruction_sha,
            }
        },
        "authority_digest",
    )
    path, file_sha = _write_json(
        root,
        f"materialization-authority/condition-{hashlib.sha256((record_id + role).encode()).hexdigest()}.json",
        authority_value,
    )
    return {
        "schema_version": authority._MATERIALIZATION_CONDITION_SCHEMA,
        "role": role,
        "instruction": instruction,
        "instruction_utf8_sha256": instruction_sha,
        "authority_path": path,
        "authority_file_sha256": file_sha,
        "authority_digest_field": "authority_digest",
        "authority_digest": authority_value["authority_digest"],
        "json_pointer": "/condition",
        "text_field": "instruction",
        "sha256_field": "instruction_utf8_sha256",
        "control_anchor_id": control_anchor_id,
    }


def _materialization_state_and_forwards(
    *,
    record: dict[str, object],
    runtime_digest: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    states: list[dict[str, object]] = []
    forwards: list[dict[str, object]] = []
    condition_by_role = {
        str(row["role"]): row for row in record["conditions"]  # type: ignore[index]
    }
    if record["record_kind"] == "teacher_anchor":
        roles = ["branch", "branch", "noop", "noop", "camera_only", "appearance_only"]
        if record["evidence_role"] == "teacher_origin":
            roles.extend(authority.WRONG_CONTROL_TYPES)
    else:
        roles = ["branch", "noop"]
    source = record["source_clean_latent"]
    source_sha = None if source is None else source["tensor_raw_sha256"]  # type: ignore[index]
    for sigma_index in authority.SIGMA_INDICES:
        sigma_hex, timestep = authority._SIGMA_RUNTIME_BINDINGS[sigma_index]
        state_unsigned = {
            "schema_version": authority.MATERIALIZATION_STATE_RECEIPT_SCHEMA,
            "provider_abi": authority.MATERIALIZATION_PROVIDER_ABI,
            "official_provider": True,
            "runtime_digest": runtime_digest,
            "record_id": record["record_id"],
            "record_kind": record["record_kind"],
            "teacher_cell_id": record["teacher_cell_id"],
            "branch": record["branch"],
            "sigma_index": sigma_index,
            "sigma_float32_be_hex": sigma_hex,
            "timestep": timestep,
            "clean_raw_sha256": record["target_clean_latent"]["tensor_raw_sha256"],  # type: ignore[index]
            "source_raw_sha256": source_sha,
            "noise_raw_sha256": record["noise"]["artifact"]["tensor_raw_sha256"],  # type: ignore[index]
            "x_sigma_raw_sha256": _digest(f"x-sigma:{record['record_id']}:{sigma_index}"),
            "input_hashes": {
                "noisy_latents": _digest(f"noisy:{record['record_id']}:{sigma_index}"),
                "rotary_embs": _digest(f"rotary:{record['record_id']}:{sigma_index}"),
                "target_mask": _digest(f"mask:{record['record_id']}:{sigma_index}"),
                "timestep": _digest(f"timestep:{record['record_id']}:{sigma_index}"),
            },
            "target_tokens": 84,
            "spatial_shape": record["target_clean_latent"]["shape"],  # type: ignore[index]
            "same_x_sigma_object_for_all_counterfactuals": True,
            "all_rank_consensus": True,
            "model_parameters_updated": False,
            "optimizer_created": False,
        }
        state = _seal(state_unsigned, "state_digest")
        states.append(state)
        velocity_by_role: dict[str, str] = {}
        for role in roles:
            velocity_sha = velocity_by_role.setdefault(
                role, _digest(f"velocity:{record['record_id']}:{sigma_index}:{role}")
            )
            forward_unsigned = {
                "schema_version": authority.MATERIALIZATION_FORWARD_RECEIPT_SCHEMA,
                "provider_abi": authority.MATERIALIZATION_PROVIDER_ABI,
                "official_provider": True,
                "record_id": record["record_id"],
                "condition_role": role,
                "condition_utf8_sha256": condition_by_role[role][
                    "instruction_utf8_sha256"
                ],
                "shared_state_digest": state["state_digest"],
                "runtime_digest": runtime_digest,
                "sigma_index": sigma_index,
                "sigma_float32_be_hex": sigma_hex,
                "timestep": timestep,
                "output_stage": "post-final-norm-proj-out-target-velocity",
                "official_frozen_native_only": True,
                "model_eval": True,
                "torch_inference_mode": True,
                "calibrator_peft_adapter_present": False,
                "frozen_effective_adapter_enabled": False,
                "frozen_effective_typed_patch_role_enabled": False,
                "base_compute_dtype": "torch.bfloat16",
                "autocast_dtype": "torch.bfloat16",
                "observer_output_dtype": "torch.float32",
                "observer_output_detached": True,
                "observer_output_contiguous": True,
                "same_state_input_objects_reused": True,
                "same_state_input_bytes_unchanged": True,
                "all_rank_consensus": True,
                "post_head_velocity_raw_sha256": velocity_sha,
                "model_parameters_updated": False,
                "optimizer_created": False,
            }
            forwards.append(_seal(forward_unsigned, "forward_digest"))
    return states, forwards


def _pair_v5_seed_truth_fixture(
    root: Path,
    *,
    candidate: dict[str, object],
    target: dict[str, object],
    seed: int,
    record_slug: str,
) -> tuple[dict[str, str], dict[str, str]]:
    branch = str(candidate["branch"])
    caption = f"fixture PAIR-v5 {branch} teacher candidate"
    caption_sha = hashlib.sha256(caption.encode("utf-8")).hexdigest()
    candidate_id = f"pair-v5-canary-{record_slug[:24]}"
    envelope = {
        "schema_version": "pair-v5-frozen-bernini-t2v-calibration-candidate-v1",
        "root_spec_raw_sha256": hashlib.sha256(b"fixture-pair-v5-root").hexdigest(),
        "group_id": "sp4-a",
        "visible_gpus": [0, 1, 2, 3],
        "ordinal": 0,
        "sampling_contract": {},
        "semantic_input_closure": {},
        "artifact_use_contract": {},
        "split_contract": {},
        "candidate": {
            "candidate_id": candidate_id,
            "analysis_split": str(candidate["anchor_split"]),
            "action_family_id": str(candidate["event_id"]),
            "calibration_group_id": str(candidate["teacher_cell_id"]),
            "prompt_group_id": f"prompt-{record_slug[:24]}",
            "action_family_group_id": str(candidate["event_id"]),
            "actor_group_id": str(candidate["actor_id"]),
            "scene_group_id": str(candidate["scene_id"]),
            "action_group_id": str(candidate["q0_id"]),
            "geometry_source_video": str(candidate["anchor_video_path"]),
            "geometry_source_video_sha256": str(candidate["anchor_video_sha256"]),
            "geometry_contract": "decode_exact81_for_bucket_shape_only_never_encode_or_condition",
            "semantic_branch": branch,
            "full_t2v_caption": caption,
            "full_t2v_caption_utf8_sha256": caption_sha,
            "caption_contract": "complete_standalone_t2v_generation_caption",
            "seed": seed,
        },
    }
    envelope_path, envelope_sha = _write_json(
        root,
        f"pair-v5-candidates/{record_slug}.json",
        envelope,
    )
    latent = {
        "path": target["path"],
        "sha256": target["file_sha256"],
        "tensor_key": target["tensor_key"],
        "raw_value_sha256": target["tensor_raw_sha256"],
        "shape": target["shape"],
        "stored_dtype": "torch.float32",
        "coordinate": "bernini_normalized_clean_vae_latent",
        "native_sampler_before_vae_decode": True,
        "mp4_decode_reencode_used": False,
    }
    gaussian_sha = hashlib.sha256(
        f"official-initial-gaussian:{seed}".encode("ascii")
    ).hexdigest()
    gaussian_content_sha = hashlib.sha256(
        f"official-initial-gaussian-content:{seed}".encode("ascii")
    ).hexdigest()
    native = _seal(
        {
            "schema_version": "bernini-native-identity-generation-canary-v2",
            "method": "frozen-bernini-native-identity-generation-canary",
            "method_source_revision": "1" * 40,
            "method_source_archive_sha256": "2" * 64,
            "bernini_commit": "3" * 40,
            "veomni_commit": "4" * 40,
            "bernini_inference_files": {"sampling.py": "5" * 64},
            "checkpoint": {"tree_sha256": "6" * 64},
            "runtime_versions": {"torch": "fixture"},
            "freeze_certificate": {"all_model_parameters_frozen": True},
            "arms": ["t2v"],
            "input": {
                "source_video_sha256": candidate["anchor_video_sha256"],
                "action_prompt_utf8_sha256": caption_sha,
            },
            "sampling": {
                "t2v": {
                    "seed": seed,
                    "num_frames": 81,
                    "num_inference_steps": 40,
                    "scheduler": "pinned-native-bernini",
                }
            },
            "initial_noise_artifacts": {
                "t2v": {
                    "generator_initial_seed": seed,
                    "raw_value_sha256": gaussian_sha,
                    "content_sha256": gaussian_content_sha,
                    "shape": [1, 16, 21, 2, 2],
                    "dtype": "torch.float32",
                    "stored_dtype": "torch.float32",
                    "captured_from_native_sampler": True,
                    "external_initial_noise_injection": False,
                    "source_or_target_derived": False,
                    "observer_changed_return_value": False,
                    "official_randn_tensor_call_count": 1,
                }
            },
            "outputs": {
                "t2v": {
                    "path": candidate["anchor_video_path"],
                    "sha256": candidate["anchor_video_sha256"],
                    "normalized_clean_latent": latent,
                }
            },
        },
        "receipt_digest",
    )
    native_path, native_sha = _write_json(
        root,
        f"pair-v5-native-receipts/{record_slug}.json",
        native,
    )
    return (
        {"path": envelope_path, "sha256": envelope_sha},
        {
            "path": native_path,
            "sha256": native_sha,
            "receipt_digest": str(native["receipt_digest"]),
        },
    )


def _attach_teacher_materialization_provenance(
    root: Path,
    representations: list[dict[str, object]],
) -> dict[str, object]:
    runtime_identity, runtime_plan = _materialization_runtime_fixture(root)
    sigma_authority = dict(materializer.sigma_authority_receipt_v1())
    plan_records: list[dict[str, object]] = []
    record_inputs: list[
        tuple[dict[str, object], dict[str, object], dict[str, object]]
    ] = []
    cell_order = tuple(
        dict.fromkeys(str(row["teacher_cell_id"]) for row in representations)
    )
    seed_by_cell = {cell: 7000 + index for index, cell in enumerate(cell_order)}
    for representation in representations:
        controls = {
            str(row["control_type"]): row["control_anchor_id"]
            for row in representation["sigma_evidence"][0]["wrong_controls"]  # type: ignore[index]
        }
        for evidence_field in ("origin_evidence", "cross_anchor_evidence"):
            evidence = representation[evidence_field]  # type: ignore[index]
            candidate = copy.deepcopy(evidence)
            record_id = f"materialize:{candidate['evidence_id']}"
            record_slug = hashlib.sha256(record_id.encode("utf-8")).hexdigest()
            review_path, review_sha = _write_json(
                root,
                f"materialization-review/{record_slug}.json",
                candidate["pre_admission_blind_review"],
            )
            target = _write_fp32_artifact(
                root,
                f"materialization-latent/{record_slug}-target.safetensors",
                tensor_key="latent",
                seed_value=1.0 + len(plan_records) * 0.01,
            )
            noise = _write_fp32_artifact(
                root,
                f"materialization-latent/{record_slug}-noise.safetensors",
                tensor_key="noise",
                seed_value=3.0 + len(plan_records) * 0.01,
            )
            pair_v5_candidate, native_receipt = _pair_v5_seed_truth_fixture(
                root,
                candidate=candidate,
                target=target,
                seed=seed_by_cell[str(candidate["teacher_cell_id"])],
                record_slug=record_slug,
            )
            latent_authority = _seal(
                {
                    "media": {
                        "path": candidate["anchor_video_path"],
                        "sha256": candidate["anchor_video_sha256"],
                    },
                    "latent": {
                        "path": target["path"],
                        "sha256": target["file_sha256"],
                        "tensor_key": target["tensor_key"],
                        "raw_value_sha256": target["tensor_raw_sha256"],
                        "shape": target["shape"],
                        "stored_dtype": "torch.float32",
                        "coordinate": "bernini_normalized_clean_vae_latent",
                        "native_sampler_before_vae_decode": True,
                        "mp4_decode_reencode_used": False,
                    },
                    "pair_v5_candidate": pair_v5_candidate,
                    "native_receipt": native_receipt,
                    "checkpoint_tree_sha256": runtime_identity[
                        "official_checkpoint_tree_sha256"
                    ],
                },
                "authority_digest",
            )
            latent_authority_path, latent_authority_sha = _write_json(
                root,
                f"materialization-authority/latent-{record_slug}.json",
                latent_authority,
            )
            condition_roles = (
                "branch",
                "noop",
                "camera_only",
                "appearance_only",
                *authority.WRONG_CONTROL_TYPES,
            )
            conditions = [
                _materialization_condition_fixture(
                    root,
                    record_id=record_id,
                    role=role,
                    control_anchor_id=controls[role]
                    if role in authority.WRONG_CONTROL_TYPES
                    else None,
                )
                for role in condition_roles
            ]
            record = _seal(
                {
                    "schema_version": authority._MATERIALIZATION_PLAN_RECORD_SCHEMA,
                    "record_id": record_id,
                    "record_kind": "teacher_anchor",
                    "evidence_id": candidate["evidence_id"],
                    "evidence_role": candidate["evidence_role"],
                    "teacher_cell_id": candidate["teacher_cell_id"],
                    "analysis_split": candidate["anchor_split"],
                    "branch": candidate["branch"],
                    "event_id": candidate["event_id"],
                    "actor_kind": candidate["actor_kind"],
                    "q0_id": candidate["q0_id"],
                    "actor_id": candidate["actor_id"],
                    "scene_id": candidate["scene_id"],
                    "anchor_id": candidate["anchor_id"],
                    "anchor_iid": candidate["anchor_iid"],
                    "pair_id": None,
                    "source_iid": None,
                    "review": {
                        "schema_version": authority._MATERIALIZATION_REVIEW_BINDING_SCHEMA,
                        "path": review_path,
                        "file_sha256": review_sha,
                        "review_digest": candidate["pre_admission_blind_review"][
                            "review_digest"
                        ],
                    },
                    "reviewed_media": {
                        "path": candidate["anchor_video_path"],
                        "file_sha256": candidate["anchor_video_sha256"],
                    },
                    "target_clean_latent": target,
                    "target_clean_latent_authority": {
                        "schema_version": authority._MATERIALIZATION_LATENT_AUTHORITY_SCHEMA,
                        "path": latent_authority_path,
                        "file_sha256": latent_authority_sha,
                        "digest_field": "authority_digest",
                        "digest": latent_authority["authority_digest"],
                        "media_json_pointer": "/media",
                        "latent_json_pointer": "/latent",
                        "checkpoint_tree_sha256_json_pointer": "/checkpoint_tree_sha256",
                    },
                    "source_clean_latent": None,
                    "source_posterior_index0_path": None,
                    "source_posterior_index0_sha256": None,
                    "source_posterior_tensor_key": None,
                    "noise": {
                        "artifact": noise,
                        "seed": authority._teacher_noise_seed(
                            str(candidate["teacher_cell_id"]),
                            str(candidate["branch"]),
                        ),
                        "generator": "torch-cpu-generator-manual-seed-randn-fp32-v1",
                    },
                    "conditions": conditions,
                },
                "record_digest",
            )
            plan_records.append(record)
            record_inputs.append((representation, evidence, candidate))

    population = _seal(
        {
            "schema_version": authority._MATERIALIZATION_POPULATION_SCHEMA,
            "population_id": "teacher-authority-fixture",
            "record_count": len(plan_records),
            "teacher_record_count": len(plan_records),
            "amplitude_record_count": 0,
            "teacher_cell_ids": sorted(
                {str(row["teacher_cell_id"]) for row in plan_records},
                key=lambda item: item.encode("utf-8"),
            ),
            "record_order_sha256": authority.object_sha256(
                [str(row["record_id"]) for row in plan_records]
            ),
            "finite_closed_population": True,
            "block_probe": False,
        },
        "population_digest",
    )
    plan = _seal(
        {
            "schema_version": authority._MATERIALIZATION_PLAN_SCHEMA,
            "plan_id": "teacher-authority-fixture-plan",
            "status": "SEALED_REVIEWED_PRE_OPTIMIZER",
            "runtime": runtime_plan,
            "population": population,
            "records": plan_records,
            "output_policy": _materialization_output_policy(),
        },
        "plan_digest",
    )

    references: list[dict[str, object]] = []
    receipt_by_evidence: dict[str, tuple[str, str, str, str]] = {}
    representation_fragments: list[dict[str, object]] = []
    materialization_by_evidence: dict[str, dict[str, object]] = {}
    for ordinal, (record, inputs) in enumerate(zip(plan_records, record_inputs)):
        representation, _live_evidence, candidate = inputs
        psiout = authority._validate_tensor_container(
            candidate["psiout_sidecar_path"],
            candidate["psiout_sidecar_sha256"],
            container_kind="psiout",
            evidence_id=str(candidate["evidence_id"]),
            evidence_role=str(candidate["evidence_role"]),
            teacher_cell_id=str(candidate["teacher_cell_id"]),
            branch=str(candidate["branch"]),
            label="fixture psiout",
        )
        nuisance = authority._validate_tensor_container(
            candidate["nuisance_packet_path"],
            candidate["nuisance_packet_sha256"],
            container_kind="nuisance",
            evidence_id=str(candidate["evidence_id"]),
            evidence_role=str(candidate["evidence_role"]),
            teacher_cell_id=str(candidate["teacher_cell_id"]),
            branch=str(candidate["branch"]),
            label="fixture nuisance",
        )
        states, forwards = _materialization_state_and_forwards(
            record=record,
            runtime_digest=str(runtime_identity["runtime_digest"]),
        )
        noise_receipt = _seal(
            {
                "schema_version": authority.MATERIALIZATION_NOISE_RECEIPT_SCHEMA,
                "provider_abi": authority.MATERIALIZATION_PROVIDER_ABI,
                "official_provider": True,
                "record_id": record["record_id"],
                "seed": record["noise"]["seed"],
                "generator": record["noise"]["generator"],
                "shape": record["noise"]["artifact"]["shape"],
                "artifact_raw_sha256": record["noise"]["artifact"][
                    "tensor_raw_sha256"
                ],
                "replayed_raw_sha256": record["noise"]["artifact"][
                    "tensor_raw_sha256"
                ],
                "byte_exact_replay": True,
            },
            "noise_digest",
        )
        sigma_metrics: list[dict[str, object]] = []
        external_rows = representation["sigma_evidence"]
        for sigma_ordinal, sigma_index in enumerate(authority.SIGMA_INDICES):
            prefix = f"sigma_{sigma_index:02d}:"
            external = external_rows[sigma_ordinal]
            if candidate["evidence_role"] == "teacher_origin":
                sigma_metrics.append(
                    {
                        "sigma_index": sigma_index,
                        "state_digest": states[sigma_ordinal]["state_digest"],
                        "projected_unit_sha256": psiout[prefix + "projected_unit"][2],
                        "projected_raw_sha256": psiout[prefix + "projected_raw"][2],
                        "duplicate_forward_first_sha256": psiout[
                            prefix + "duplicate_forward_first"
                        ][2],
                        "duplicate_forward_second_sha256": psiout[
                            prefix + "duplicate_forward_second"
                        ][2],
                        "duplicate_forward_bytes_identical": True,
                        "noop_forward_first_sha256": psiout[
                            prefix + "noop_forward_first"
                        ][2],
                        "noop_forward_second_sha256": psiout[
                            prefix + "noop_forward_second"
                        ][2],
                        "same_state_noop_minus_noop_null_norm": external[
                            "same_state_noop_minus_noop_null_norm"
                        ],
                        "projected_teacher_raw_norm": external[
                            "projected_teacher_raw_norm"
                        ],
                        "signal_to_null_snr": external["signal_to_null_snr"],
                        "camera_unit_sha256": nuisance[prefix + "camera_unit"][2],
                        "appearance_unit_sha256": nuisance[
                            prefix + "appearance_unit"
                        ][2],
                        "camera_residual_cosine": external[
                            "camera_residual_cosine"
                        ],
                        "appearance_residual_cosine": external[
                            "appearance_residual_cosine"
                        ],
                        "wrong_controls": copy.deepcopy(external["wrong_controls"]),
                    }
                )
            else:
                sigma_metrics.append(
                    {
                        "sigma_index": sigma_index,
                        "state_digest": states[sigma_ordinal]["state_digest"],
                        "projected_unit_sha256": psiout[prefix + "projected_unit"][2],
                        "camera_unit_sha256": nuisance[prefix + "camera_unit"][2],
                        "appearance_unit_sha256": nuisance[
                            prefix + "appearance_unit"
                        ][2],
                    }
                )
        receipt = _seal(
            {
                "schema_version": authority.MATERIALIZATION_RECORD_RECEIPT_SCHEMA,
                "plan_id": plan["plan_id"],
                "plan_digest": plan["plan_digest"],
                "runtime_digest": runtime_identity["runtime_digest"],
                "provider_abi": authority.MATERIALIZATION_PROVIDER_ABI,
                "official_provider": True,
                "test_only": False,
                "record_ordinal": ordinal,
                "record_id": record["record_id"],
                "record_digest": record["record_digest"],
                "record_kind": record["record_kind"],
                "evidence_id": record["evidence_id"],
                "evidence_role": record["evidence_role"],
                "teacher_cell_id": record["teacher_cell_id"],
                "branch": record["branch"],
                "record_authority": record,
                "record_conditions": record["conditions"],
                "review_digest": record["review"]["review_digest"],
                "reviewed_media_sha256": record["reviewed_media"]["file_sha256"],
                "target_clean_latent_raw_sha256": record["target_clean_latent"][
                    "tensor_raw_sha256"
                ],
                "target_clean_latent_authority_digest": record[
                    "target_clean_latent_authority"
                ]["digest"],
                "source_clean_latent_raw_sha256": None,
                "source_posterior_index0_sha256": None,
                "noise_seed": record["noise"]["seed"],
                "noise_raw_sha256": record["noise"]["artifact"][
                    "tensor_raw_sha256"
                ],
                "noise_replay_receipt": noise_receipt,
                "sigma_authority_digest": sigma_authority[
                    "sigma_authority_digest"
                ],
                "state_receipts": states,
                "forward_receipts": forwards,
                "container_bindings": [
                    {
                        "container_kind": "psiout",
                        "path": candidate["psiout_sidecar_path"],
                        "file_sha256": candidate["psiout_sidecar_sha256"],
                        "slice_sha256": {
                            name: tensor[2] for name, tensor in psiout.items()
                        },
                    },
                    {
                        "container_kind": "nuisance",
                        "path": candidate["nuisance_packet_path"],
                        "file_sha256": candidate["nuisance_packet_sha256"],
                        "slice_sha256": {
                            name: tensor[2] for name, tensor in nuisance.items()
                        },
                    },
                ],
                "sigma_metrics": sigma_metrics,
                "candidate_authority_evidence": candidate,
                "generated_rgb_decoded": False,
                "generated_rgb_used_as_model_input": False,
                "generated_rgb_used_as_regression_target": False,
                "generated_latent_used_as_absolute_regression_target": False,
                "model_parameters_updated": False,
                "optimizer_created": False,
            },
            "record_receipt_digest",
        )
        receipt_path, receipt_sha = _write_json(
            root,
            f"materialization-receipts/{ordinal:04d}.json",
            receipt,
            mode=authority.MATERIALIZATION_RECEIPT_MODE,
        )
        reference = {
            "record_id": record["record_id"],
            "record_kind": record["record_kind"],
            "path": receipt_path,
            "file_sha256": receipt_sha,
            "record_receipt_digest": receipt["record_receipt_digest"],
            "candidate_evidence_digest": candidate["evidence_digest"],
        }
        references.append(reference)
        materialization_by_evidence[str(candidate["evidence_id"])] = receipt
        receipt_by_evidence[str(candidate["evidence_id"])] = (
            receipt_path,
            receipt_sha,
            str(receipt["record_receipt_digest"]),
            str(record["record_id"]),
        )

    for representation in representations:
        origin = representation["origin_evidence"]
        cross = representation["cross_anchor_evidence"]
        origin_receipt = materialization_by_evidence[str(origin["evidence_id"])]
        cross_receipt = materialization_by_evidence[str(cross["evidence_id"])]
        representation_fragments.append(
            {
                "teacher_cell_id": representation["teacher_cell_id"],
                "branch": representation["branch"],
                "origin_record_id": origin_receipt["record_id"],
                "cross_anchor_record_id": cross_receipt["record_id"],
                "origin_evidence_digest": origin_receipt[
                    "candidate_authority_evidence"
                ]["evidence_digest"],
                "cross_anchor_evidence_digest": cross_receipt[
                    "candidate_authority_evidence"
                ]["evidence_digest"],
                "sigma_evidence": copy.deepcopy(representation["sigma_evidence"]),
            }
        )

    run = _seal(
        {
            "schema_version": authority.MATERIALIZATION_RUN_RECEIPT_SCHEMA,
            "plan_id": plan["plan_id"],
            "plan_digest": plan["plan_digest"],
            "plan_authority": plan,
            "population_digest": population["population_digest"],
            "record_order_sha256": population["record_order_sha256"],
            "runtime_identity": runtime_identity,
            "runtime_plan_digest": runtime_plan["runtime_plan_digest"],
            "official_helper_sources": runtime_plan["official_helper_sources"],
            "provider_abi": authority.MATERIALIZATION_PROVIDER_ABI,
            "official_provider": True,
            "test_only": False,
            "world_size": 4,
            "dp_size": 1,
            "sp_size": 4,
            "sigma_indices": list(authority.SIGMA_INDICES),
            "sigma_authority": sigma_authority,
            "record_count": len(plan_records),
            "computation_digest": _digest("teacher-materialization-computation"),
            "record_receipts": references,
            "representation_sigma_evidence_candidates": representation_fragments,
            "amplitude_sigma_calibration_candidates": [],
            "output_policy": _materialization_output_policy(),
            "generated_rgb_decoded": False,
            "generated_rgb_used_as_model_input": False,
            "generated_rgb_used_as_regression_target": False,
            "generated_latent_used_as_absolute_regression_target": False,
            "model_parameters_updated": False,
            "optimizer_created": False,
        },
        "run_digest",
    )
    run_path, run_sha = _write_json(
        root,
        "materialization-receipts/materialization-run.json",
        run,
        mode=authority.MATERIALIZATION_RECEIPT_MODE,
    )
    for representation in representations:
        for evidence_field in ("origin_evidence", "cross_anchor_evidence"):
            evidence = representation[evidence_field]
            receipt_path, receipt_sha, receipt_digest, _record_id = receipt_by_evidence[
                str(evidence["evidence_id"])
            ]
            evidence.update(
                {
                    "materialization_record_receipt_path": receipt_path,
                    "materialization_record_receipt_sha256": receipt_sha,
                    "materialization_record_receipt_digest": receipt_digest,
                    "materialization_run_digest": run["run_digest"],
                }
            )
            _resign_nested(evidence, "evidence_digest")
        _resign_nested(representation, "admission_digest")
    return _seal(
        {
            "schema_version": authority.MATERIALIZATION_RUN_BINDING_SCHEMA,
            "path": run_path,
            "file_sha256": run_sha,
            "run_digest": run["run_digest"],
        },
        "binding_digest",
    )


def _build_manifest(root: Path) -> dict[str, object]:
    teacher_origins: list[dict[str, object]] = []
    teacher_ordinal = 0
    for split in ("fit", "confirmation"):
        for event_index, (event_id, actor_kind, q0_id) in enumerate(EVENTS):
            for seed in range(2):
                path, digest = _write(
                    root,
                    f"teachers/{split}-{event_index}-{seed}.mp4",
                    f"teacher-origin-{split}-{event_index}-{seed}".encode(),
                )
                unsigned = {
                    "schema_version": authority.TEACHER_ORIGIN_SCHEMA,
                    "teacher_cell_id": f"{split}-teacher-{event_index}-{seed}",
                    "analysis_split": split,
                    "origin_iid": f"{0x10000 + teacher_ordinal:016x}",
                    "origin_source_path": path,
                    "origin_source_sha256": digest,
                    "origin_group_id": f"teacher-group-{split}-{event_index}-{seed}",
                    "event_id": event_id,
                    "actor_kind": actor_kind,
                    "q0_id": q0_id,
                    "actor_id": f"teacher-actor-{split}-{event_index}-{seed}",
                    "scene_id": f"teacher-scene-{split}-{event_index}-{seed}",
                }
                teacher_origins.append(_seal(unsigned, "origin_digest"))
                teacher_ordinal += 1

    sources: list[dict[str, object]] = []
    source_ordinal = 0
    per_event = {"fit": 16, "confirmation": 4, "heldout": 2}
    for split in authority.SPLITS:
        for event_index, (event_id, actor_kind, q0_id) in enumerate(EVENTS):
            for local_index in range(per_event[split]):
                iid = f"{source_ordinal + 1:016x}"
                video_path, video_sha = _write(
                    root,
                    f"sources/{iid}.mp4",
                    f"source-video-{iid}".encode(),
                )
                index0_path, index0_sha = _write(
                    root,
                    f"index0/{iid}.source-posterior-index0.pt",
                    f"source-index0-{iid}".encode(),
                )
                unsigned = {
                    "schema_version": authority.SOURCE_SCHEMA,
                    "source_iid": iid,
                    "analysis_split": split,
                    "source_group_id": f"real-group-{split}-{event_index}-{local_index}",
                    "source_video_path": video_path,
                    "source_video_sha256": video_sha,
                    "source_posterior_index0_path": index0_path,
                    "source_posterior_index0_sha256": index0_sha,
                    "source_posterior_tensor_key": "latent",
                    "posterior_index_decoded": 0,
                    "physical_index0_only": True,
                    "synthetic_target_index1_bytes_read": False,
                    "synthetic_target_index1_decoded": False,
                    "synthetic_target_index1_hashed": False,
                    "actor_id": f"real-actor-{split}-{event_index}-{local_index}",
                    "scene_id": f"real-scene-{split}-{event_index}-{local_index}",
                    "event_id": event_id,
                    "actor_kind": actor_kind,
                    "q0_id": q0_id,
                    "source_motion_label": f"incompatible-motion-{event_index}-{local_index}",
                }
                sources.append(_seal(unsigned, "source_digest"))
                source_ordinal += 1

    assignments = authority.deterministic_teacher_assignment_v1(sources, teacher_origins)
    origins_by_cell = {
        str(row["teacher_cell_id"]): row for row in teacher_origins
    }
    pairs: list[dict[str, object]] = []
    for source in sources:
        iid = str(source["source_iid"])
        teacher_cell = assignments[iid]
        origin = origins_by_cell[teacher_cell]
        for branch in authority.BRANCHES:
            pair_id = f"pair:{source['analysis_split']}:{iid}:{branch}"
            instruction = f"For {iid}, perform {origin['event_id']} as {branch}."
            review_unsigned = {
                "schema_version": authority.REVIEW_SCHEMA,
                "review_id": f"review:{source['analysis_split']}:{iid}:{branch}",
                "pair_id": pair_id,
                "source_iid": iid,
                "source_video_sha256": source["source_video_sha256"],
                "branch": branch,
                "frame_count": 81,
                "fps": 25.0,
                "entire_full81_video_viewed": True,
                "independent_reviewer": True,
                "reviewer_blinded_to_teacher_cell": True,
                "sealed_before_pair_admission": True,
                "actor_kind_compatible": True,
                "q0_compatible": True,
                "owner_object_verified": True,
                "source_motion_verified": True,
                "target_event_incompatible_with_source_motion": True,
            }
            pair_unsigned = {
                "schema_version": authority.PAIR_SCHEMA,
                "pair_id": pair_id,
                "analysis_split": source["analysis_split"],
                "source_iid": iid,
                "source_video_sha256": source["source_video_sha256"],
                "branch": branch,
                "teacher_cell_id": teacher_cell,
                "event_id": source["event_id"],
                "actor_kind": source["actor_kind"],
                "q0_id": source["q0_id"],
                "source_motion_label": source["source_motion_label"],
                "instruction": instruction,
                "instruction_utf8_sha256": hashlib.sha256(
                    instruction.encode("utf-8")
                ).hexdigest(),
                "target_event_incompatible_with_source_motion": True,
                "optimizer_admitted": source["analysis_split"] == "fit",
                "pre_admission_full81_review": _seal(
                    review_unsigned, "review_digest"
                ),
            }
            pairs.append(_seal(pair_unsigned, "pair_digest"))

    representations: list[dict[str, object]] = []
    for cell_ordinal, origin in enumerate(teacher_origins):
        for branch in authority.BRANCHES:
            cell = str(origin["teacher_cell_id"])
            (
                origin_material,
                cross_material,
                origin_nuisance_material,
                cross_nuisance_material,
            ) = _representation_material()
            origin_evidence, origin_packed, origin_nuisance_packed = _representation_anchor_evidence(
                root,
                origin=origin,
                branch=branch,
                cell_ordinal=cell_ordinal,
                evidence_role="teacher_origin",
                psiout_tensors=origin_material,
                nuisance_tensors=origin_nuisance_material,
            )
            cross_evidence, cross_packed, _cross_nuisance_packed = _representation_anchor_evidence(
                root,
                origin=origin,
                branch=branch,
                cell_ordinal=cell_ordinal,
                evidence_role="same_event_cross_anchor",
                psiout_tensors=cross_material,
                nuisance_tensors=cross_nuisance_material,
            )
            sigma_evidence: list[dict[str, object]] = []
            for sigma_index in authority.SIGMA_INDICES:
                origin_projected = origin_packed[
                    authority._tensor_name(sigma_index, "projected_unit")
                ]
                cross_projected = cross_packed[
                    authority._tensor_name(sigma_index, "projected_unit")
                ]
                projected_raw = origin_packed[
                    authority._tensor_name(sigma_index, "projected_raw")
                ]
                duplicate_first = origin_packed[
                    authority._tensor_name(sigma_index, "duplicate_forward_first")
                ]
                duplicate_second = origin_packed[
                    authority._tensor_name(sigma_index, "duplicate_forward_second")
                ]
                noop_first = origin_packed[
                    authority._tensor_name(sigma_index, "noop_forward_first")
                ]
                noop_second = origin_packed[
                    authority._tensor_name(sigma_index, "noop_forward_second")
                ]
                origin_values = _unpack_f32(origin_projected)
                cross_values = _unpack_f32(cross_projected)
                raw_values = _unpack_f32(projected_raw)
                noop_first_values = _unpack_f32(noop_first)
                noop_second_values = _unpack_f32(noop_second)
                null_norm = math.sqrt(
                    math.fsum(
                        (float(first) - float(second)) ** 2
                        for first, second in zip(noop_first_values, noop_second_values)
                    )
                )
                raw_norm = _fixture_norm(raw_values)
                same_event_cosine = _fixture_cosine(origin_values, cross_values)
                wrong_controls: list[dict[str, object]] = []
                for control_type in authority.WRONG_CONTROL_TYPES:
                    wrong_tensor = origin_packed[
                        authority._tensor_name(
                            sigma_index, f"{control_type}_projected_unit"
                        )
                    ]
                    wrong_controls.append({
                        "control_type": control_type,
                        "control_anchor_id": f"control:{cell}:{branch}:{control_type}",
                        "wrong_projected_slice_sha256": hashlib.sha256(wrong_tensor).hexdigest(),
                        "wrong_event_cosine": _fixture_cosine(
                            origin_values, _unpack_f32(wrong_tensor)
                        ),
                    })
                camera = origin_nuisance_packed[
                    authority._tensor_name(sigma_index, "camera_unit")
                ]
                appearance = origin_nuisance_packed[
                    authority._tensor_name(sigma_index, "appearance_unit")
                ]
                sigma_evidence.append(
                    {
                        "sigma_index": sigma_index,
                        "origin_projected_slice_sha256": hashlib.sha256(
                            origin_projected
                        ).hexdigest(),
                        "cross_anchor_projected_slice_sha256": hashlib.sha256(
                            cross_projected
                        ).hexdigest(),
                        "same_event_cosine": same_event_cosine,
                        "duplicate_forward_first_sha256": hashlib.sha256(
                            duplicate_first
                        ).hexdigest(),
                        "duplicate_forward_second_sha256": hashlib.sha256(
                            duplicate_second
                        ).hexdigest(),
                        "duplicate_forward_bytes_identical": duplicate_first
                        == duplicate_second,
                        "same_state_noop_minus_noop_null_norm": null_norm,
                        "projected_teacher_raw_norm": raw_norm,
                        "signal_to_null_snr": raw_norm
                        / max(null_norm, authority.DUPLICATE_SNR_DENOMINATOR_FLOOR),
                        "camera_residual_cosine": _fixture_cosine(
                            origin_values, _unpack_f32(camera)
                        ),
                        "appearance_residual_cosine": _fixture_cosine(
                            origin_values, _unpack_f32(appearance)
                        ),
                        "wrong_controls": wrong_controls,
                    }
                )
            unsigned = {
                "schema_version": authority.REPRESENTATION_SCHEMA,
                "admission_id": f"admit:{cell}:{branch}",
                "teacher_cell_id": cell,
                "analysis_split": origin["analysis_split"],
                "branch": branch,
                "event_id": origin["event_id"],
                "origin_evidence": origin_evidence,
                "cross_anchor_evidence": cross_evidence,
                "sigma_evidence": sigma_evidence,
                "optimizer_admitted": origin["analysis_split"] == "fit",
            }
            representations.append(_seal(unsigned, "admission_digest"))

    materialization_run_receipt = _attach_teacher_materialization_provenance(
        root, representations
    )

    unsigned_manifest: dict[str, object] = {
        "schema_version": authority.SCHEMA_VERSION,
        "materialization_run_receipt": materialization_run_receipt,
        "authority": {
            "status": "optimizer_admitted",
            "data_authority_complete": True,
            "teacher_authority_complete": True,
            "current_optimizer_pair_rows": 128,
            "current_optimizer_teacher_bundles": 16,
            "current_authority_nonzero": True,
            "optimizer_authorized": True,
        },
        "source_io_policy": {
            "physical_payload": "source_posterior_index_0_only",
            "posterior_index_decoded": 0,
            "synthetic_target_index1_path_present": False,
            "synthetic_target_index1_bytes_read": False,
            "synthetic_target_index1_decoded": False,
            "synthetic_target_index1_hashed": False,
        },
        "teacher_origins": teacher_origins,
        "sources": sources,
        "pairs": pairs,
        "representation_admissions": representations,
        "authority_counts": authority._expected_counts(),
    }
    return _seal(unsigned_manifest, "manifest_digest")


class Full30ActionDataTeacherAuthorityTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.manifest = _build_manifest(self.root)

    def _reject(self, manifest: dict[str, object], pattern: str) -> None:
        with self.assertRaisesRegex(authority.Full30ActionAuthorityError, pattern):
            authority.validate_full30_action_authority_v1(manifest)

    def test_complete_positive_authority_closes_all_counts(self) -> None:
        receipt = authority.validate_full30_action_authority_v1(self.manifest)
        self.assertTrue(receipt["optimizer_authorized"])
        self.assertTrue(receipt["deterministic_assignment_verified"])
        self.assertTrue(receipt["representation_admission_verified"])
        self.assertFalse(receipt["synthetic_target_index1_bytes_read"])
        self.assertEqual(receipt["source_counts"], {"fit": 64, "confirmation": 16, "heldout": 8})
        self.assertEqual(receipt["pair_counts"], {"fit": 128, "confirmation": 32, "heldout": 16})
        self.assertEqual(receipt["representation_anchor_evidence"], 64)
        self.assertEqual(receipt["representation_blind_reviews"], 64)
        self.assertEqual(receipt["representation_sigma_rows"], 192)
        self.assertEqual(receipt["representation_wrong_control_rows"], 576)
        sigma_rows = 0
        control_rows = 0
        for admission in self.manifest["representation_admissions"]:  # type: ignore[union-attr]
            rows = admission["sigma_evidence"]
            self.assertEqual(
                [row["sigma_index"] for row in rows], list(authority.SIGMA_INDICES)
            )
            for row in rows:
                sigma_rows += 1
                self.assertGreaterEqual(
                    row["same_event_cosine"], authority.SAME_EVENT_MINIMUM_COSINE
                )
                computed_snr = row["projected_teacher_raw_norm"] / max(
                    row["same_state_noop_minus_noop_null_norm"],
                    authority.DUPLICATE_SNR_DENOMINATOR_FLOOR,
                )
                self.assertEqual(row["signal_to_null_snr"], computed_snr)
                self.assertGreaterEqual(computed_snr, authority.DUPLICATE_MIN_SNR)
                self.assertLessEqual(
                    abs(row["camera_residual_cosine"]), authority.NUISANCE_MAX_ABS_COSINE
                )
                self.assertLessEqual(
                    abs(row["appearance_residual_cosine"]),
                    authority.NUISANCE_MAX_ABS_COSINE,
                )
                self.assertEqual(
                    [control["control_type"] for control in row["wrong_controls"]],
                    list(authority.WRONG_CONTROL_TYPES),
                )
                for control in row["wrong_controls"]:
                    control_rows += 1
                    self.assertGreaterEqual(
                        row["same_event_cosine"] - control["wrong_event_cosine"],
                        authority.WRONG_CONTROL_MINIMUM_MARGIN,
                    )
        self.assertEqual(sigma_rows, 32 * 6)
        self.assertEqual(control_rows, 32 * 18)

    def test_zero_or_incomplete_current_authority_rejects(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["authority"].update(  # type: ignore[union-attr]
            {
                "current_optimizer_pair_rows": 0,
                "current_optimizer_teacher_bundles": 0,
                "current_authority_nonzero": False,
            }
        )
        _resign_top(manifest)
        self._reject(manifest, "not admitted|incomplete|zero")

        missing = copy.deepcopy(self.manifest)
        missing["sources"].pop()  # type: ignore[union-attr]
        _resign_top(missing)
        self._reject(missing, "exactly 88")

    def test_global_iid_sha_group_and_teacher_origin_exclusion(self) -> None:
        cases: list[tuple[dict[str, object], str]] = []

        duplicate_iid = copy.deepcopy(self.manifest)
        duplicate_iid["sources"][1]["source_iid"] = duplicate_iid["sources"][0]["source_iid"]  # type: ignore[index]
        _resign_nested(duplicate_iid["sources"][1], "source_digest")  # type: ignore[index]
        _resign_top(duplicate_iid)
        cases.append((duplicate_iid, "IID is duplicated"))

        duplicate_sha = copy.deepcopy(self.manifest)
        duplicate_sha["sources"][1]["source_video_path"] = duplicate_sha["sources"][0]["source_video_path"]  # type: ignore[index]
        duplicate_sha["sources"][1]["source_video_sha256"] = duplicate_sha["sources"][0]["source_video_sha256"]  # type: ignore[index]
        _resign_nested(duplicate_sha["sources"][1], "source_digest")  # type: ignore[index]
        _resign_top(duplicate_sha)
        cases.append((duplicate_sha, "video SHA-256 is duplicated"))

        group_leak = copy.deepcopy(self.manifest)
        first_confirmation = next(
            row for row in group_leak["sources"] if row["analysis_split"] == "confirmation"  # type: ignore[union-attr]
        )
        first_confirmation["source_group_id"] = group_leak["sources"][0]["source_group_id"]  # type: ignore[index]
        _resign_nested(first_confirmation, "source_digest")
        _resign_top(group_leak)
        cases.append((group_leak, "group crosses analysis splits"))

        origin_overlap = copy.deepcopy(self.manifest)
        origin_overlap["teacher_origins"][0]["origin_iid"] = origin_overlap["sources"][0]["source_iid"]  # type: ignore[index]
        _resign_nested(origin_overlap["teacher_origins"][0], "origin_digest")  # type: ignore[index]
        _resign_top(origin_overlap)
        cases.append((origin_overlap, "overlaps a teacher origin"))

        teacher_split_leak = copy.deepcopy(self.manifest)
        fit_origin = next(
            row for row in teacher_split_leak["teacher_origins"] if row["analysis_split"] == "fit"  # type: ignore[union-attr]
        )
        confirmation_origin = next(
            row for row in teacher_split_leak["teacher_origins"] if row["analysis_split"] == "confirmation"  # type: ignore[union-attr]
        )
        confirmation_origin["origin_group_id"] = fit_origin["origin_group_id"]
        _resign_nested(confirmation_origin, "origin_digest")
        _resign_top(teacher_split_leak)
        cases.append((teacher_split_leak, "teacher origin group crosses"))

        for manifest, pattern in cases:
            with self.subTest(pattern=pattern):
                self._reject(manifest, pattern)

    def test_index0_only_and_never_read_synthetic_are_hard_gates(self) -> None:
        global_read = copy.deepcopy(self.manifest)
        global_read["source_io_policy"]["synthetic_target_index1_bytes_read"] = True  # type: ignore[index]
        _resign_top(global_read)
        self._reject(global_read, "bytes_read.*not false")

        wrong_index = copy.deepcopy(self.manifest)
        wrong_index["sources"][0]["posterior_index_decoded"] = 1  # type: ignore[index]
        _resign_nested(wrong_index["sources"][0], "source_digest")  # type: ignore[index]
        _resign_top(wrong_index)
        self._reject(wrong_index, "posterior index is not zero")

        row_read = copy.deepcopy(self.manifest)
        row_read["sources"][0]["synthetic_target_index1_bytes_read"] = True  # type: ignore[index]
        _resign_nested(row_read["sources"][0], "source_digest")  # type: ignore[index]
        _resign_top(row_read)
        self._reject(row_read, "bytes_read.*not false")

        wrong_filename = copy.deepcopy(self.manifest)
        source = wrong_filename["sources"][0]  # type: ignore[index]
        original = Path(source["source_posterior_index0_path"])
        renamed = original.with_name("ambiguous-posterior.pt")
        renamed.write_bytes(original.read_bytes())
        source["source_posterior_index0_path"] = str(renamed)
        _resign_nested(source, "source_digest")
        _resign_top(wrong_filename)
        self._reject(wrong_filename, "index0 filename differs")

    def test_every_pair_requires_sealed_pre_admission_full81_review(self) -> None:
        for field, value, pattern in (
            ("frame_count", 80, "not full81"),
            ("entire_full81_video_viewed", False, "not true"),
            ("sealed_before_pair_admission", False, "not true"),
            ("actor_kind_compatible", False, "not true"),
            ("target_event_incompatible_with_source_motion", False, "not true"),
        ):
            manifest = copy.deepcopy(self.manifest)
            pair = manifest["pairs"][0]  # type: ignore[index]
            review = pair["pre_admission_full81_review"]
            review[field] = value
            _resign_nested(review, "review_digest")
            _resign_nested(pair, "pair_digest")
            _resign_top(manifest)
            with self.subTest(field=field):
                self._reject(manifest, pattern)

        stale = copy.deepcopy(self.manifest)
        stale["pairs"][0]["pre_admission_full81_review"]["frame_count"] = 80  # type: ignore[index]
        _resign_nested(stale["pairs"][0], "pair_digest")  # type: ignore[index]
        _resign_top(stale)
        self._reject(stale, "review.*digest differs")

        reused = copy.deepcopy(self.manifest)
        reused["pairs"][1]["pre_admission_full81_review"]["review_id"] = reused["pairs"][0]["pre_admission_full81_review"]["review_id"]  # type: ignore[index]
        _resign_nested(reused["pairs"][1]["pre_admission_full81_review"], "review_digest")  # type: ignore[index]
        _resign_nested(reused["pairs"][1], "pair_digest")  # type: ignore[index]
        _resign_top(reused)
        self._reject(reused, "review id is reused")

    def test_assignment_is_deterministic_and_actor_q0_compatible(self) -> None:
        wrong_assignment = copy.deepcopy(self.manifest)
        pair = wrong_assignment["pairs"][0]  # type: ignore[index]
        origin = next(
            row
            for row in wrong_assignment["teacher_origins"]  # type: ignore[union-attr]
            if row["event_id"] == pair["event_id"]
            and row["actor_kind"] == pair["actor_kind"]
            and row["q0_id"] == pair["q0_id"]
            and row["analysis_split"] == "fit"
            and row["teacher_cell_id"] != pair["teacher_cell_id"]
        )
        pair["teacher_cell_id"] = origin["teacher_cell_id"]
        _resign_nested(pair, "pair_digest")
        _resign_top(wrong_assignment)
        self._reject(wrong_assignment, "deterministic teacher assignment")

        wrong_actor = copy.deepcopy(self.manifest)
        wrong_actor["pairs"][0]["actor_kind"] = "incompatible-actor"  # type: ignore[index]
        _resign_nested(wrong_actor["pairs"][0], "pair_digest")  # type: ignore[index]
        _resign_top(wrong_actor)
        self._reject(wrong_actor, "actor_kind compatibility differs")

        duplicate_instruction = copy.deepcopy(self.manifest)
        first, second = duplicate_instruction["pairs"][:2]  # type: ignore[index]
        second["instruction"] = first["instruction"]
        second["instruction_utf8_sha256"] = first["instruction_utf8_sha256"]
        _resign_nested(second, "pair_digest")
        _resign_top(duplicate_instruction)
        self._reject(duplicate_instruction, "instructions are not distinct")

    def test_every_sigma_threshold_is_recomputed_from_explicit_rows(self) -> None:
        cases: list[tuple[dict[str, object], str]] = []

        same_event = copy.deepcopy(self.manifest)
        admission = same_event["representation_admissions"][0]  # type: ignore[index]
        admission["sigma_evidence"][2]["same_event_cosine"] = 0.549
        _resign_nested(admission, "admission_digest")
        _resign_top(same_event)
        cases.append((same_event, "same_event_cosine is below 0.55"))

        wrong_margin = copy.deepcopy(self.manifest)
        admission = wrong_margin["representation_admissions"][0]  # type: ignore[index]
        admission["sigma_evidence"][4]["wrong_controls"][1]["wrong_event_cosine"] = 0.56
        _resign_nested(admission, "admission_digest")
        _resign_top(wrong_margin)
        cases.append((wrong_margin, "correct-minus-wrong margin is below 0.2"))

        nondeterministic = copy.deepcopy(self.manifest)
        admission = nondeterministic["representation_admissions"][0]  # type: ignore[index]
        admission["sigma_evidence"][1]["duplicate_forward_second_sha256"] = _digest(
            "different-forward-bytes"
        )
        _resign_nested(admission, "admission_digest")
        _resign_top(nondeterministic)
        cases.append((nondeterministic, "differs from tensor-container bytes"))

        bad_null = copy.deepcopy(self.manifest)
        admission = bad_null["representation_admissions"][0]  # type: ignore[index]
        sigma = admission["sigma_evidence"][3]
        sigma["same_state_noop_minus_noop_null_norm"] = 1.01e-7
        sigma["signal_to_null_snr"] = sigma["projected_teacher_raw_norm"] / 1.01e-7
        _resign_nested(admission, "admission_digest")
        _resign_top(bad_null)
        cases.append((bad_null, "null norm exceeds 1e-07"))

        bad_raw = copy.deepcopy(self.manifest)
        admission = bad_raw["representation_admissions"][0]  # type: ignore[index]
        sigma = admission["sigma_evidence"][5]
        sigma["projected_teacher_raw_norm"] = 9.99e-5
        sigma["signal_to_null_snr"] = 9.99e-5 / max(
            sigma["same_state_noop_minus_noop_null_norm"],
            authority.DUPLICATE_SNR_DENOMINATOR_FLOOR,
        )
        _resign_nested(admission, "admission_digest")
        _resign_top(bad_raw)
        cases.append((bad_raw, "projected raw norm is below 0.0001"))

        bad_snr = copy.deepcopy(self.manifest)
        admission = bad_snr["representation_admissions"][0]  # type: ignore[index]
        admission["sigma_evidence"][0]["signal_to_null_snr"] = 99.0
        _resign_nested(admission, "admission_digest")
        _resign_top(bad_snr)
        cases.append((bad_snr, "SNR arithmetic differs"))

        bad_nuisance = copy.deepcopy(self.manifest)
        admission = bad_nuisance["representation_admissions"][0]  # type: ignore[index]
        admission["sigma_evidence"][0]["camera_residual_cosine"] = -1.01e-5
        _resign_nested(admission, "admission_digest")
        _resign_top(bad_nuisance)
        cases.append((bad_nuisance, "camera_residual_cosine exceeds 1e-05"))

        for manifest, pattern in cases:
            with self.subTest(pattern=pattern):
                self._reject(manifest, pattern)

    def test_cross_anchor_identity_review_and_control_closure_are_hard(self) -> None:
        first = self.manifest["representation_admissions"][0]  # type: ignore[index]
        other_cell = self.manifest["representation_admissions"][2]  # type: ignore[index]
        source_cross = first["cross_anchor_evidence"]
        target_cross = other_cell["cross_anchor_evidence"]
        self.assertEqual(source_cross["anchor_video_sha256"], target_cross["anchor_video_sha256"])
        self.assertEqual(source_cross["anchor_iid"], target_cross["anchor_iid"])
        self.assertNotEqual(source_cross["evidence_id"], target_cross["evidence_id"])
        shared_receipt = authority.validate_full30_action_authority_v1(self.manifest)
        self.assertTrue(shared_receipt["optimizer_authorized"])

        reused = copy.deepcopy(self.manifest)
        first = reused["representation_admissions"][0]  # type: ignore[index]
        other_cell = reused["representation_admissions"][2]  # type: ignore[index]
        cross = other_cell["cross_anchor_evidence"]
        cross["anchor_id"] = first["cross_anchor_evidence"]["anchor_id"]
        review = cross["pre_admission_blind_review"]
        review["anchor_id"] = cross["anchor_id"]
        _resign_nested(review, "review_digest")
        _resign_nested(cross, "evidence_digest")
        _resign_nested(other_cell, "admission_digest")
        _resign_top(reused)
        self._reject(
            reused,
            "representation anchor id is reused|differs from materializer base candidate",
        )

        reused_bytes = copy.deepcopy(self.manifest)
        first = reused_bytes["representation_admissions"][0]  # type: ignore[index]
        other_cell = reused_bytes["representation_admissions"][4]  # type: ignore[index]
        source_cross = first["cross_anchor_evidence"]
        target_cross = other_cell["cross_anchor_evidence"]
        target_cross["anchor_video_path"] = source_cross["anchor_video_path"]
        target_cross["anchor_video_sha256"] = source_cross["anchor_video_sha256"]
        review = target_cross["pre_admission_blind_review"]
        review["anchor_video_sha256"] = target_cross["anchor_video_sha256"]
        _resign_nested(review, "review_digest")
        _resign_nested(target_cross, "evidence_digest")
        _resign_nested(other_cell, "admission_digest")
        _resign_top(reused_bytes)
        self._reject(
            reused_bytes,
            "reused representation anchor video identity differs|differs from materializer base candidate",
        )

        same_identity = copy.deepcopy(self.manifest)
        admission = same_identity["representation_admissions"][0]  # type: ignore[index]
        origin = admission["origin_evidence"]
        cross = admission["cross_anchor_evidence"]
        cross["actor_id"] = origin["actor_id"]
        cross["scene_id"] = origin["scene_id"]
        review = cross["pre_admission_blind_review"]
        review["actor_id"] = cross["actor_id"]
        review["scene_id"] = cross["scene_id"]
        _resign_nested(review, "review_digest")
        _resign_nested(cross, "evidence_digest")
        _resign_nested(admission, "admission_digest")
        _resign_top(same_identity)
        self._reject(
            same_identity,
            "actor is not different|scene is not different|differs from materializer base candidate",
        )

        review_tamper = copy.deepcopy(self.manifest)
        admission = review_tamper["representation_admissions"][0]  # type: ignore[index]
        cross = admission["cross_anchor_evidence"]
        cross["pre_admission_blind_review"]["target_event_verified"] = False
        _resign_nested(cross, "evidence_digest")
        _resign_nested(admission, "admission_digest")
        _resign_top(review_tamper)
        self._reject(review_tamper, "blind_review digest differs")

        control_missing = copy.deepcopy(self.manifest)
        admission = control_missing["representation_admissions"][0]  # type: ignore[index]
        admission["sigma_evidence"][0]["wrong_controls"].pop()
        _resign_nested(admission, "admission_digest")
        _resign_top(control_missing)
        self._reject(control_missing, "wrong_controls must contain exactly 3 rows")

    def test_aggregate_representation_summary_cannot_replace_sigma_rows(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        admission = manifest["representation_admissions"][0]  # type: ignore[index]
        del admission["sigma_evidence"]
        admission["aggregate_q10_count_summary"] = {
            "pair_count": 999,
            "q10_cosine": 1.0,
            "q10_correct_minus_wrong_margin": 2.0,
            "passed": True,
        }
        _resign_nested(admission, "admission_digest")
        _resign_top(manifest)
        self._reject(manifest, "field closure differs")

    def test_representation_artifact_sha_and_bundle_closure_are_hard(self) -> None:
        bad_sha = copy.deepcopy(self.manifest)
        admission = bad_sha["representation_admissions"][0]  # type: ignore[index]
        evidence = admission["origin_evidence"]
        evidence["psiout_sidecar_sha256"] = "0" * 64
        _resign_nested(evidence, "evidence_digest")
        _resign_nested(admission, "admission_digest")
        _resign_top(bad_sha)
        self._reject(bad_sha, "sidecar file SHA-256 differs")

        missing = copy.deepcopy(self.manifest)
        missing["representation_admissions"].pop()  # type: ignore[union-attr]
        _resign_top(missing)
        self._reject(missing, "exactly 32")

        confirmation_leak = copy.deepcopy(self.manifest)
        confirmation = next(
            row
            for row in confirmation_leak["representation_admissions"]  # type: ignore[union-attr]
            if row["analysis_split"] == "confirmation"
        )
        confirmation["optimizer_admitted"] = True
        _resign_nested(confirmation, "admission_digest")
        _resign_top(confirmation_leak)
        self._reject(confirmation_leak, "leaks confirmation")

        heldout_pair_leak = copy.deepcopy(self.manifest)
        heldout_pair = next(
            row
            for row in heldout_pair_leak["pairs"]  # type: ignore[union-attr]
            if row["analysis_split"] == "heldout"
        )
        heldout_pair["optimizer_admitted"] = True
        _resign_nested(heldout_pair, "pair_digest")
        _resign_top(heldout_pair_leak)
        self._reject(heldout_pair_leak, "leaks a non-fit row")

    def test_tensor_container_is_strict_and_materially_bound(self) -> None:
        def admission_and_evidence(manifest: dict[str, object]):
            admission = manifest["representation_admissions"][0]  # type: ignore[index]
            return admission, admission["origin_evidence"]

        def bind_rewritten_sidecar(manifest: dict[str, object], digest: str) -> None:
            admission, evidence = admission_and_evidence(manifest)
            evidence["psiout_sidecar_sha256"] = digest
            _resign_nested(evidence, "evidence_digest")
            _resign_nested(admission, "admission_digest")
            _resign_top(manifest)

        def nonfinite(manifest: dict[str, object], _root: Path) -> None:
            _admission, evidence = admission_and_evidence(manifest)
            values = list(_basis(0))
            values[0] = float("nan")
            digest = _rewrite_tensor(
                evidence["psiout_sidecar_path"],
                authority._tensor_name(authority.SIGMA_INDICES[0], "projected_unit"),
                tuple(values),
            )
            bind_rewritten_sidecar(manifest, digest)

        def arbitrary_slice_digest(manifest: dict[str, object], _root: Path) -> None:
            admission, _evidence = admission_and_evidence(manifest)
            admission["sigma_evidence"][0]["origin_projected_slice_sha256"] = _digest(  # type: ignore[index]
                "not-a-tensor-slice"
            )
            _resign_nested(admission, "admission_digest")
            _resign_top(manifest)

        def extra_bytes(manifest: dict[str, object], _root: Path) -> None:
            _admission, evidence = admission_and_evidence(manifest)
            path = Path(evidence["psiout_sidecar_path"])
            path.write_bytes(path.read_bytes() + b"hostile-extra-byte")
            path.chmod(authority.TENSOR_CONTAINER_MODE)
            bind_rewritten_sidecar(manifest, authority.file_sha256(path))

        def symlink(manifest: dict[str, object], root: Path) -> None:
            admission, evidence = admission_and_evidence(manifest)
            target = Path(evidence["psiout_sidecar_path"])
            link = root / "hostile-sidecar-symlink.f30tc"
            link.symlink_to(target)
            evidence["psiout_sidecar_path"] = str(link.absolute())
            _resign_nested(evidence, "evidence_digest")
            _resign_nested(admission, "admission_digest")
            _resign_top(manifest)

        def wrong_mode(manifest: dict[str, object], _root: Path) -> None:
            _admission, evidence = admission_and_evidence(manifest)
            Path(evidence["psiout_sidecar_path"]).chmod(0o640)

        def payload_tamper_with_honest_file_hash(
            manifest: dict[str, object], _root: Path
        ) -> None:
            _admission, evidence = admission_and_evidence(manifest)
            path = Path(evidence["psiout_sidecar_path"])
            payload = bytearray(path.read_bytes())
            payload[-1] ^= 0x01
            path.write_bytes(payload)
            path.chmod(authority.TENSOR_CONTAINER_MODE)
            bind_rewritten_sidecar(manifest, authority.file_sha256(path))

        def noncanonical_offset(manifest: dict[str, object], _root: Path) -> None:
            _admission, evidence = admission_and_evidence(manifest)

            def mutate(header: dict[str, object]) -> None:
                header["entries"][1]["offset"] = 0  # type: ignore[index]

            digest = _rewrite_container_header(evidence["psiout_sidecar_path"], mutate)
            bind_rewritten_sidecar(manifest, digest)

        def wrong_shape(manifest: dict[str, object], _root: Path) -> None:
            _admission, evidence = admission_and_evidence(manifest)

            def mutate(header: dict[str, object]) -> None:
                header["entries"][0]["shape"] = [32, 21]  # type: ignore[index]

            digest = _rewrite_container_header(evidence["psiout_sidecar_path"], mutate)
            bind_rewritten_sidecar(manifest, digest)

        def wrong_schema(manifest: dict[str, object], _root: Path) -> None:
            _admission, evidence = admission_and_evidence(manifest)

            def mutate(header: dict[str, object]) -> None:
                header["schema_version"] = "bernini-full30-action-tensor-container-v999"

            digest = _rewrite_container_header(evidence["psiout_sidecar_path"], mutate)
            bind_rewritten_sidecar(manifest, digest)

        cases = (
            ("nonfinite", nonfinite, "non-finite FP32"),
            ("arbitrary-slice-digest", arbitrary_slice_digest, "differs from tensor-container bytes"),
            ("extra-bytes", extra_bytes, "payload length/extra bytes differ"),
            ("symlink", symlink, "plain non-symlink file"),
            ("mode", wrong_mode, "mode must be exactly 0o600"),
            ("payload-tamper", payload_tamper_with_honest_file_hash, "tensor byte SHA-256 differs"),
            ("offset", noncanonical_offset, "offset is not canonical"),
            ("shape", wrong_shape, "shape differs"),
            ("schema", wrong_schema, "schema differs"),
        )
        for case_name, mutate, pattern in cases:
            with self.subTest(case=case_name), tempfile.TemporaryDirectory() as temporary:
                manifest = _build_manifest(Path(temporary).resolve())
                mutate(manifest, Path(temporary).resolve())
                self._reject(manifest, pattern)

    def test_materialization_run_record_and_provenance_links_are_fail_closed(self) -> None:
        wrong_record = copy.deepcopy(self.manifest)
        first = wrong_record["representation_admissions"][0]["origin_evidence"]  # type: ignore[index]
        second = wrong_record["representation_admissions"][0]["cross_anchor_evidence"]  # type: ignore[index]
        for field in (
            "materialization_record_receipt_path",
            "materialization_record_receipt_sha256",
            "materialization_record_receipt_digest",
        ):
            first[field] = second[field]
        _resign_nested(first, "evidence_digest")
        _resign_nested(
            wrong_record["representation_admissions"][0], "admission_digest"  # type: ignore[index]
        )
        _resign_top(wrong_record)
        self._reject(wrong_record, "base candidate|record authority")

        wrong_run = copy.deepcopy(self.manifest)
        evidence = wrong_run["representation_admissions"][0]["origin_evidence"]  # type: ignore[index]
        evidence["materialization_run_digest"] = "0" * 64
        _resign_nested(evidence, "evidence_digest")
        _resign_nested(
            wrong_run["representation_admissions"][0], "admission_digest"  # type: ignore[index]
        )
        _resign_top(wrong_run)
        self._reject(wrong_run, "materialization run digest differs")

        for kind in ("run-mode", "run-symlink", "run-tamper", "record-mode", "record-symlink"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                manifest = _build_manifest(root)
                binding = manifest["materialization_run_receipt"]
                run_path = Path(binding["path"])
                first_evidence = manifest["representation_admissions"][0]["origin_evidence"]  # type: ignore[index]
                if kind == "run-mode":
                    run_path.chmod(0o640)
                    pattern = "mode must be exactly"
                elif kind == "run-symlink":
                    link = root / "run-receipt-symlink.json"
                    link.symlink_to(run_path)
                    binding["path"] = str(link.absolute())
                    _resign_nested(binding, "binding_digest")
                    _resign_top(manifest)
                    pattern = "plain non-symlink"
                elif kind == "run-tamper":
                    run_path.write_bytes(run_path.read_bytes() + b"x")
                    run_path.chmod(authority.MATERIALIZATION_RECEIPT_MODE)
                    pattern = "file SHA-256 differs"
                elif kind == "record-mode":
                    Path(first_evidence["materialization_record_receipt_path"]).chmod(
                        0o640
                    )
                    pattern = "mode must be exactly"
                else:
                    target = Path(
                        first_evidence["materialization_record_receipt_path"]
                    )
                    link = root / "record-receipt-symlink.json"
                    link.symlink_to(target)
                    first_evidence["materialization_record_receipt_path"] = str(
                        link.absolute()
                    )
                    _resign_nested(first_evidence, "evidence_digest")
                    _resign_nested(
                        manifest["representation_admissions"][0],  # type: ignore[index]
                        "admission_digest",
                    )
                    _resign_top(manifest)
                    pattern = "plain non-symlink"
                with self.assertRaisesRegex(
                    authority.Full30ActionAuthorityError, pattern
                ):
                    authority.validate_full30_action_authority_v1(manifest)

    def test_materialization_receipt_field_closures_reject_extra_and_missing(self) -> None:
        run = json.loads(
            Path(
                self.manifest["materialization_run_receipt"]["path"]  # type: ignore[index]
            ).read_bytes()
        )
        forged_runtime = copy.deepcopy(run["runtime_identity"])
        forged_runtime["official_provider_source_sha256"] = "f" * 64
        _resign_nested(forged_runtime, "runtime_digest")
        with self.assertRaisesRegex(
            authority.Full30ActionAuthorityError,
            "provider source differs from the physical materializer",
        ):
            authority._validate_materialization_runtime_identity(forged_runtime)

        for level, mutation in (
            ("run-extra", lambda value: value.__setitem__("unexpected", True)),
            ("run-missing", lambda value: value.pop("sigma_authority")),
            ("record-extra", lambda value: value.__setitem__("unexpected", True)),
            ("record-missing", lambda value: value.pop("record_conditions")),
        ):
            with self.subTest(level=level), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                manifest = _build_manifest(root)
                binding = manifest["materialization_run_receipt"]
                run_path = Path(binding["path"])
                run = json.loads(run_path.read_bytes())
                if level.startswith("record"):
                    reference = run["record_receipts"][0]
                    record_path = Path(reference["path"])
                    record = json.loads(record_path.read_bytes())
                    mutation(record)
                    if "record_receipt_digest" in record:
                        _resign_nested(record, "record_receipt_digest")
                    raw = authority.canonical_json_bytes(record) + b"\n"
                    record_path.write_bytes(raw)
                    record_path.chmod(authority.MATERIALIZATION_RECEIPT_MODE)
                    reference["file_sha256"] = hashlib.sha256(raw).hexdigest()
                    if "record_receipt_digest" in record:
                        reference["record_receipt_digest"] = record[
                            "record_receipt_digest"
                        ]
                else:
                    mutation(run)
                if "run_digest" in run:
                    _resign_nested(run, "run_digest")
                run_raw = authority.canonical_json_bytes(run) + b"\n"
                run_path.write_bytes(run_raw)
                run_path.chmod(authority.MATERIALIZATION_RECEIPT_MODE)
                binding["file_sha256"] = hashlib.sha256(run_raw).hexdigest()
                if "run_digest" in run:
                    binding["run_digest"] = run["run_digest"]
                _resign_nested(binding, "binding_digest")
                with self.assertRaisesRegex(
                    authority.Full30ActionAuthorityError, "field closure differs"
                ):
                    authority._load_materialization_run_v1(binding)

    def test_materialization_review_latent_noise_and_condition_files_are_reopened(self) -> None:
        for artifact_kind in ("review", "target-latent", "noise", "condition"):
            with self.subTest(artifact_kind=artifact_kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                manifest = _build_manifest(root)
                evidence = manifest["representation_admissions"][0]["origin_evidence"]  # type: ignore[index]
                receipt = json.loads(
                    Path(evidence["materialization_record_receipt_path"]).read_bytes()
                )
                record = receipt["record_authority"]
                if artifact_kind == "review":
                    path = Path(record["review"]["path"])
                elif artifact_kind == "target-latent":
                    path = Path(record["target_clean_latent"]["path"])
                elif artifact_kind == "noise":
                    path = Path(record["noise"]["artifact"]["path"])
                else:
                    path = Path(record["conditions"][0]["authority_path"])
                path.write_bytes(path.read_bytes() + b"tamper")
                with self.assertRaisesRegex(
                    authority.Full30ActionAuthorityError, "SHA-256 differs"
                ):
                    authority.validate_full30_action_authority_v1(manifest)

    def test_metrics_are_recomputed_from_fp32_tensor_bytes(self) -> None:
        cases: list[tuple[str, object]] = []

        same_event = copy.deepcopy(self.manifest)
        admission = same_event["representation_admissions"][0]  # type: ignore[index]
        admission["sigma_evidence"][0]["same_event_cosine"] = 0.65
        _resign_nested(admission, "admission_digest")
        _resign_top(same_event)
        cases.append(("same-event", same_event))

        raw = copy.deepcopy(self.manifest)
        admission = raw["representation_admissions"][0]  # type: ignore[index]
        sigma = admission["sigma_evidence"][0]
        sigma["projected_teacher_raw_norm"] *= 2.0
        sigma["signal_to_null_snr"] = sigma["projected_teacher_raw_norm"] / max(
            sigma["same_state_noop_minus_noop_null_norm"],
            authority.DUPLICATE_SNR_DENOMINATOR_FLOOR,
        )
        _resign_nested(admission, "admission_digest")
        _resign_top(raw)
        cases.append(("raw-and-snr", raw))

        null = copy.deepcopy(self.manifest)
        admission = null["representation_admissions"][0]  # type: ignore[index]
        sigma = admission["sigma_evidence"][0]
        sigma["same_state_noop_minus_noop_null_norm"] = 5.0e-8
        sigma["signal_to_null_snr"] = sigma["projected_teacher_raw_norm"] / 5.0e-8
        _resign_nested(admission, "admission_digest")
        _resign_top(null)
        cases.append(("null-and-snr", null))

        nuisance = copy.deepcopy(self.manifest)
        admission = nuisance["representation_admissions"][0]  # type: ignore[index]
        admission["sigma_evidence"][0]["camera_residual_cosine"] = 1.0e-6
        _resign_nested(admission, "admission_digest")
        _resign_top(nuisance)
        cases.append(("nuisance", nuisance))

        control = copy.deepcopy(self.manifest)
        admission = control["representation_admissions"][0]  # type: ignore[index]
        admission["sigma_evidence"][0]["wrong_controls"][0]["wrong_event_cosine"] = 0.30
        _resign_nested(admission, "admission_digest")
        _resign_top(control)
        cases.append(("control", control))

        for case_name, manifest in cases:
            with self.subTest(case=case_name):
                self._reject(manifest, "does not match tensor-container bytes")

    def test_source_artifact_sha_is_verified_not_merely_declared(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        source_path = Path(manifest["sources"][0]["source_video_path"])  # type: ignore[index]
        source_path.write_bytes(b"tampered-after-seal")
        self._reject(manifest, "source_video file SHA-256 differs")

    def test_cli_requires_manifest_file_sha_binding(self) -> None:
        path = self.root / "authority.json"
        path.write_bytes(authority.canonical_json_bytes(self.manifest) + b"\n")
        file_sha = authority.file_sha256(path)
        script = METHOD_ROOT / "full30_action_data_teacher_authority_v1.py"
        success = subprocess.run(
            [
                sys.executable,
                str(script),
                "--manifest",
                str(path),
                "--expected-sha256",
                file_sha,
            ],
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        self.assertEqual(success.returncode, 0, success.stderr)
        receipt = json.loads(success.stdout)
        self.assertTrue(receipt["optimizer_authorized"])

        rejected = subprocess.run(
            [
                sys.executable,
                str(script),
                "--manifest",
                str(path),
                "--expected-sha256",
                "0" * 64,
            ],
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("file SHA-256 differs", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
