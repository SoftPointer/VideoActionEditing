#!/usr/bin/env python3
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys
import tempfile
import unittest
from typing import Any, Mapping, Optional, Sequence


METHODS = Path(__file__).resolve().parents[1]
if str(METHODS) not in sys.path:
    sys.path.insert(0, str(METHODS))

import full30_action_amplitude_authority_v1 as amplitude_authority
import full30_action_data_teacher_authority_v1 as teacher_authority
import full30_action_psiout_materializer_v1 as materializer
import inference_sigma_strata as sigma_authority


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_bytes(path: Path, raw: bytes, mode: int = 0o600) -> str:
    path.write_bytes(raw)
    path.chmod(mode)
    return _sha(raw)


def _write_json(path: Path, value: Mapping[str, Any]) -> str:
    return _write_bytes(path, materializer.canonical_json_bytes(value) + b"\n")


class FixtureBuilder:
    shape = (1, 16, 21, 2, 2)

    def __init__(self, root: Path) -> None:
        self.root = root
        self.counter = 0
        self.bernini_root = root / "Bernini"
        self.veomni_root = root / "VeOmni"
        self.checkpoint_root = root / "checkpoint"
        self.bernini_root.mkdir()
        self.veomni_root.mkdir()
        (self.checkpoint_root / "transformer").mkdir(parents=True)
        self.transformer_config = self.checkpoint_root / "transformer" / "config.json"
        transformer_sha = _write_bytes(self.transformer_config, b"{}\n")
        self.checkpoint_manifest = root / "checkpoint-content.json"
        checkpoint_manifest_sha = _write_bytes(self.checkpoint_manifest, b"{}\n")
        self.psiout_path = METHODS / "full30_action_learning_v1.py"
        self.provider_path = Path(materializer.__file__).resolve()
        compute = dict(materializer.frozen_compute_contract_v1())
        runtime_unsigned = {
            "schema_version": amplitude_authority.RUNTIME_IDENTITY_SCHEMA_VERSION,
            "bernini_revision": "a" * 40,
            "veomni_revision": "b" * 40,
            "official_checkpoint_tree_sha256": "c" * 64,
            "transformer_config_sha256": transformer_sha,
            "sigma_table_sha256": sigma_authority.SCHEDULE_SHA256,
            "psiout_protocol_sha256": materializer.file_sha256(self.psiout_path),
            "official_provider_source_sha256": materializer.file_sha256(self.provider_path),
            "official_provider_abi": materializer.OFFICIAL_PROVIDER_ABI,
            "compute_contract": compute,
            "compute_contract_digest": materializer.object_sha256(compute),
            "frame_count": 81,
            "fps": 25.0,
            "sampler_steps": 40,
        }
        self.runtime_identity = materializer.seal_record(runtime_unsigned, "runtime_digest")
        helpers = []
        for module_name in materializer.REQUIRED_HELPER_MODULES:
            path = METHODS / f"{module_name}.py"
            helpers.append(
                {
                    "module": module_name,
                    "path": str(path),
                    "file_sha256": materializer.file_sha256(path),
                }
            )
        runtime_plan_unsigned = {
            "schema_version": materializer.PLAN_RUNTIME_SCHEMA_VERSION,
            "frozen_runtime_identity": self.runtime_identity,
            "bernini_root": str(self.bernini_root),
            "veomni_root": str(self.veomni_root),
            "checkpoint_root": str(self.checkpoint_root),
            "checkpoint_content_manifest_path": str(self.checkpoint_manifest),
            "checkpoint_content_manifest_sha256": checkpoint_manifest_sha,
            "psiout_protocol_path": str(self.psiout_path),
            "official_provider_source_path": str(self.provider_path),
            "official_helper_sources": helpers,
        }
        self.runtime = materializer.seal_record(
            runtime_plan_unsigned, "runtime_plan_digest"
        )

    def _path(self, stem: str, suffix: str) -> Path:
        self.counter += 1
        return self.root / f"{self.counter:03d}-{stem}{suffix}"

    def artifact(self, stem: str, offset: float) -> Mapping[str, Any]:
        values = tuple(
            materializer._f32(offset + ((index % 29) - 14) * 0.003)
            for index in range(math.prod(self.shape))
        )
        tensor = materializer.FP32TensorV1(self.shape, values)
        key = "normalized_clean_latent"
        header = materializer.canonical_json_bytes(
            {
                key: {
                    "dtype": "F32",
                    "shape": list(self.shape),
                    "data_offsets": [0, len(tensor.bytes_le())],
                }
            }
        )
        raw = struct.pack("<Q", len(header)) + header + tensor.bytes_le()
        path = self._path(stem, ".safetensors")
        file_sha = _write_bytes(path, raw)
        return {
            "schema_version": materializer.PLAN_ARTIFACT_SCHEMA_VERSION,
            "path": str(path),
            "file_sha256": file_sha,
            "tensor_key": key,
            "tensor_raw_sha256": tensor.raw_sha256(),
            "dtype": "float32-le",
            "shape": list(self.shape),
        }

    def media(self, stem: str) -> Mapping[str, Any]:
        path = self._path(stem, ".mp4")
        raw = b"fixture-full81-mp4\x00" + stem.encode("ascii")
        return {"path": str(path), "file_sha256": _write_bytes(path, raw)}

    def condition_rows(
        self, stem: str, roles: Sequence[str], *, amplitude: bool = False
    ) -> list[Mapping[str, Any]]:
        texts: dict[str, str] = {}
        for role in roles:
            if amplitude and role == "noop":
                text = materializer.EXACT_NOOP_INSTRUCTION
            else:
                text = f"{stem} instruction for {role}."
            texts[role] = text
        bound = {
            role: {
                "instruction": text,
                "instruction_utf8_sha256": _sha(text.encode("utf-8")),
            }
            for role, text in texts.items()
        }
        authority = materializer.seal_record(
            {"schema_version": "fixture-condition-authority-v1", "conditions": bound},
            "authority_digest",
        )
        authority_path = self._path(stem + "-conditions", ".json")
        authority_file_sha = _write_json(authority_path, authority)
        rows = []
        for role in roles:
            rows.append(
                {
                    "schema_version": materializer.PLAN_CONDITION_SCHEMA_VERSION,
                    "role": role,
                    "instruction": texts[role],
                    "instruction_utf8_sha256": bound[role]["instruction_utf8_sha256"],
                    "authority_path": str(authority_path),
                    "authority_file_sha256": authority_file_sha,
                    "authority_digest_field": "authority_digest",
                    "authority_digest": authority["authority_digest"],
                    "json_pointer": f"/conditions/{role}",
                    "text_field": "instruction",
                    "sha256_field": "instruction_utf8_sha256",
                    "control_anchor_id": (
                        f"{stem}-{role}-control"
                        if role in teacher_authority.WRONG_CONTROL_TYPES
                        else None
                    ),
                }
            )
        return rows

    def teacher_review(
        self, record: Mapping[str, Any], media: Mapping[str, Any], stem: str
    ) -> Mapping[str, Any]:
        unsigned = {
            "schema_version": teacher_authority.REPRESENTATION_REVIEW_SCHEMA,
            "review_id": f"review-{stem}",
            "evidence_id": record["evidence_id"],
            "anchor_id": record["anchor_id"],
            "anchor_video_sha256": media["file_sha256"],
            "anchor_split": record["analysis_split"],
            "branch": record["branch"],
            "event_id": record["event_id"],
            "actor_kind": record["actor_kind"],
            "q0_id": record["q0_id"],
            "actor_id": record["actor_id"],
            "scene_id": record["scene_id"],
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
        review = materializer.seal_record(unsigned, "review_digest")
        path = self._path(stem + "-review", ".json")
        file_sha = _write_json(path, review)
        return {
            "schema_version": materializer.PLAN_REVIEW_SCHEMA_VERSION,
            "path": str(path),
            "file_sha256": file_sha,
            "review_digest": review["review_digest"],
        }

    def amplitude_review(
        self, record: Mapping[str, Any], media: Mapping[str, Any], stem: str
    ) -> Mapping[str, Any]:
        unsigned = {
            "schema_version": amplitude_authority.REVIEW_SCHEMA_VERSION,
            "review_id": f"review-{stem}",
            "evidence_id": record["evidence_id"],
            "pair_id": record["pair_id"],
            "source_iid": record["source_iid"],
            "branch": record["branch"],
            "baseline_output_sha256": media["file_sha256"],
            "frame_count": 81,
            "fps": 25.0,
            "sampler_steps": 40,
            "entire_full81_video_viewed": True,
            "independent_reviewer": True,
            "reviewer_blinded_to_amplitude_metrics": True,
            "sealed_before_sidecar_extraction": True,
            "sealed_before_optimizer_authority": True,
            "action_result": "partial",
        }
        review = materializer.seal_record(unsigned, "review_digest")
        path = self._path(stem + "-review", ".json")
        file_sha = _write_json(path, review)
        return {
            "schema_version": materializer.PLAN_REVIEW_SCHEMA_VERSION,
            "path": str(path),
            "file_sha256": file_sha,
            "review_digest": review["review_digest"],
        }

    def latent_authority(
        self,
        stem: str,
        media: Mapping[str, Any],
        artifact: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        authority = materializer.seal_record(
            {
                "schema_version": "fixture-native-generation-receipt-v1",
                "official_checkpoint_tree_sha256": self.runtime_identity[
                    "official_checkpoint_tree_sha256"
                ],
                "artifacts": {
                    "mp4": {
                        "path": media["path"],
                        "sha256": media["file_sha256"],
                    },
                    "predecode_clean_latent": {
                        "path": artifact["path"],
                        "sha256": artifact["file_sha256"],
                        "tensor_key": artifact["tensor_key"],
                        "raw_value_sha256": artifact["tensor_raw_sha256"],
                        "shape": artifact["shape"],
                        "stored_dtype": "torch.float32",
                        "coordinate": "bernini_normalized_clean_vae_latent",
                        "native_sampler_before_vae_decode": True,
                        "mp4_decode_reencode_used": False,
                    },
                },
            },
            "receipt_digest",
        )
        path = self._path(stem + "-latent-authority", ".json")
        file_sha = _write_json(path, authority)
        return {
            "schema_version": materializer.PLAN_LATENT_AUTHORITY_SCHEMA_VERSION,
            "path": str(path),
            "file_sha256": file_sha,
            "digest_field": "receipt_digest",
            "digest": authority["receipt_digest"],
            "media_json_pointer": "/artifacts/mp4",
            "latent_json_pointer": "/artifacts/predecode_clean_latent",
            "checkpoint_tree_sha256_json_pointer": "/official_checkpoint_tree_sha256",
        }

    def teacher_record(
        self,
        stem: str,
        evidence_role: str,
        anchor_iid: str,
        actor_id: str,
        scene_id: str,
        clean_offset: float,
        noise: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        roles = (
            materializer.TEACHER_CONDITION_ROLES
            if evidence_role == "teacher_origin"
            else materializer.CROSS_CONDITION_ROLES
        )
        media = self.media(stem)
        clean = self.artifact(stem + "-clean", clean_offset)
        row: dict[str, Any] = {
            "schema_version": materializer.PLAN_RECORD_SCHEMA_VERSION,
            "record_id": f"record-{stem}",
            "record_kind": "teacher_anchor",
            "evidence_id": f"evidence-{stem}",
            "evidence_role": evidence_role,
            "teacher_cell_id": "cell-one",
            "analysis_split": "fit",
            "branch": "action",
            "event_id": "wave",
            "actor_kind": "person",
            "q0_id": "q0-wave",
            "actor_id": actor_id,
            "scene_id": scene_id,
            "anchor_id": f"anchor-{stem}",
            "anchor_iid": anchor_iid,
            "pair_id": None,
            "source_iid": None,
            "review": None,
            "reviewed_media": media,
            "target_clean_latent": clean,
            "target_clean_latent_authority": self.latent_authority(
                stem, media, clean
            ),
            "source_clean_latent": None,
            "source_posterior_index0_path": None,
            "source_posterior_index0_sha256": None,
            "source_posterior_tensor_key": None,
            "noise": {
                "artifact": noise,
                "seed": materializer.teacher_noise_seed_v1("cell-one", "action"),
                "generator": "torch-cpu-generator-manual-seed-randn-fp32-v1",
            },
            "conditions": self.condition_rows(stem, roles),
        }
        row["review"] = self.teacher_review(row, media, stem)
        return materializer.seal_record(row, "record_digest")

    def amplitude_record(self, stem: str, source_iid: str, offset: float) -> Mapping[str, Any]:
        media = self.media(stem)
        clean = self.artifact(stem + "-clean", offset)
        noise = self.artifact(stem + "-noise", offset + 2.0)
        posterior_path = self.root / f"{source_iid}.source-posterior-index0.pt"
        posterior_sha = _write_bytes(
            posterior_path, b"fixture-physical-posterior-index0\x00" + stem.encode("ascii")
        )
        row: dict[str, Any] = {
            "schema_version": materializer.PLAN_RECORD_SCHEMA_VERSION,
            "record_id": f"record-{stem}",
            "record_kind": "amplitude_calibrator",
            "evidence_id": f"evidence-{stem}",
            "evidence_role": "calibrator",
            "teacher_cell_id": "cell-one",
            "analysis_split": "fit",
            "branch": "action",
            "event_id": "wave",
            "actor_kind": "person",
            "q0_id": "q0-wave",
            "actor_id": f"actor-{stem}",
            "scene_id": f"scene-{stem}",
            "anchor_id": None,
            "anchor_iid": None,
            "pair_id": f"pair-{stem}",
            "source_iid": source_iid,
            "review": None,
            "reviewed_media": media,
            "target_clean_latent": clean,
            "target_clean_latent_authority": None,
            "source_clean_latent": dict(clean),
            "source_posterior_index0_path": str(posterior_path),
            "source_posterior_index0_sha256": posterior_sha,
            "source_posterior_tensor_key": "posterior_parameters",
            "noise": {
                "artifact": noise,
                "seed": materializer.amplitude_noise_seed_v1(f"pair-{stem}"),
                "generator": "torch-cpu-generator-manual-seed-randn-fp32-v1",
            },
            "conditions": self.condition_rows(
                stem, materializer.AMPLITUDE_CONDITION_ROLES, amplitude=True
            ),
        }
        row["review"] = self.amplitude_review(row, media, stem)
        return materializer.seal_record(row, "record_digest")

    def build(self) -> Mapping[str, Any]:
        teacher_noise = self.artifact("teacher-noise", 3.0)
        records = [
            self.teacher_record(
                "origin", "teacher_origin", "0000000000000001", "actor-one", "scene-one", 0.1, teacher_noise
            ),
            self.teacher_record(
                "cross", "same_event_cross_anchor", "0000000000000002", "actor-two", "scene-two", 0.2, teacher_noise
            ),
            self.amplitude_record("amplitude-a", "1000000000000001", 0.3),
            self.amplitude_record("amplitude-b", "1000000000000002", 0.4),
        ]
        population = materializer.seal_record(
            {
                "schema_version": materializer.PLAN_POPULATION_SCHEMA_VERSION,
                "population_id": "fixture-population",
                "record_count": len(records),
                "teacher_record_count": 2,
                "amplitude_record_count": 2,
                "teacher_cell_ids": ["cell-one"],
                "record_order_sha256": materializer.object_sha256(
                    [row["record_id"] for row in records]
                ),
                "finite_closed_population": True,
                "block_probe": False,
            },
            "population_digest",
        )
        policy = {
            "schema_version": materializer.PLAN_OUTPUT_POLICY_SCHEMA_VERSION,
            "create_only": True,
            "container_mode_octal": "0600",
            "generated_rgb_decoded": False,
            "generated_rgb_used_as_model_input": False,
            "generated_rgb_used_as_regression_target": False,
            "generated_latent_used_as_absolute_regression_target": False,
            "model_parameters_updated": False,
            "optimizer_created": False,
            "persisted_tensor_role": "detached-post-head-psiout-or-same-mode-amplitude-evidence-only",
        }
        return materializer.seal_record(
            {
                "schema_version": materializer.PLAN_SCHEMA_VERSION,
                "plan_id": "fixture-plan",
                "status": "SEALED_REVIEWED_PRE_OPTIMIZER",
                "runtime": self.runtime,
                "population": population,
                "records": records,
                "output_policy": policy,
            },
            "plan_digest",
        )


class FakeFrozenProvider:
    is_official = False
    rank = 0

    def __init__(
        self,
        plan: Mapping[str, Any],
        *,
        wrong_stage: bool = False,
        nondeterministic: bool = False,
        bad_noise: bool = False,
    ) -> None:
        self.identity = plan["runtime"]["frozen_runtime_identity"]
        self.wrong_stage = wrong_stage
        self.nondeterministic = nondeterministic
        self.bad_noise = bad_noise
        self.prepare_calls = 0
        self.forward_calls = 0
        self.forward_state_objects: dict[str, set[int]] = {}
        self.role_calls: dict[tuple[str, str], int] = {}
        self.consensus: list[str] = []
        self.barriers = 0
        self.closed = False

    def verify_noise_authority_v1(
        self, *, record: Mapping[str, Any], noise: Any
    ) -> Mapping[str, Any]:
        artifact_sha = materializer._tensor_raw_sha256(noise, label="fake noise")
        replayed_sha = "f" * 64 if self.bad_noise else artifact_sha
        return materializer.seal_record(
            {
                "schema_version": materializer.NOISE_RECEIPT_SCHEMA_VERSION,
                "provider_abi": materializer.OFFICIAL_PROVIDER_ABI,
                "official_provider": False,
                "record_id": record["record_id"],
                "seed": record["noise"]["seed"],
                "generator": record["noise"]["generator"],
                "shape": list(materializer._tensor_shape(noise, label="fake noise")),
                "artifact_raw_sha256": artifact_sha,
                "replayed_raw_sha256": replayed_sha,
                "byte_exact_replay": not self.bad_noise,
            },
            "noise_digest",
        )

    def prepare_same_state_v1(
        self,
        *,
        record: Mapping[str, Any],
        clean: Any,
        source: Optional[Any],
        noise: Any,
        x_sigma: Any,
        sigma_index: int,
        sigma: float,
        timestep: int,
    ) -> materializer.PreparedSameStateV1:
        self.prepare_calls += 1
        input_hashes = {
            name: _sha(f"{record['record_id']}:{sigma_index}:{name}".encode("ascii"))
            for name in ("noisy_latents", "rotary_embs", "target_mask", "timestep")
        }
        unsigned = {
            "schema_version": materializer.STATE_RECEIPT_SCHEMA_VERSION,
            "provider_abi": materializer.OFFICIAL_PROVIDER_ABI,
            "official_provider": False,
            "runtime_digest": self.identity["runtime_digest"],
            "record_id": record["record_id"],
            "record_kind": record["record_kind"],
            "teacher_cell_id": record["teacher_cell_id"],
            "branch": record["branch"],
            "sigma_index": sigma_index,
            "sigma_float32_be_hex": struct.pack(">f", sigma).hex(),
            "timestep": timestep,
            "clean_raw_sha256": materializer._tensor_raw_sha256(clean, label="fake clean"),
            "source_raw_sha256": (
                None
                if source is None
                else materializer._tensor_raw_sha256(source, label="fake source")
            ),
            "noise_raw_sha256": materializer._tensor_raw_sha256(noise, label="fake noise"),
            "x_sigma_raw_sha256": materializer._tensor_raw_sha256(x_sigma, label="fake x"),
            "input_hashes": input_hashes,
            "target_tokens": 21,
            "spatial_shape": list(materializer._tensor_shape(clean, label="fake clean")),
            "same_x_sigma_object_for_all_counterfactuals": True,
            "all_rank_consensus": True,
            "model_parameters_updated": False,
            "optimizer_created": False,
        }
        receipt = materializer.seal_record(unsigned, "state_digest")
        opaque = {
            "record": record,
            "shape": tuple(unsigned["spatial_shape"]),
            "x_object": id(x_sigma),
        }
        return materializer.PreparedSameStateV1(
            record_id=str(record["record_id"]),
            sigma_index=sigma_index,
            sigma_float32_be_hex=unsigned["sigma_float32_be_hex"],
            timestep=timestep,
            clean_raw_sha256=unsigned["clean_raw_sha256"],
            source_raw_sha256=unsigned["source_raw_sha256"],
            noise_raw_sha256=unsigned["noise_raw_sha256"],
            x_sigma_raw_sha256=unsigned["x_sigma_raw_sha256"],
            state_digest=receipt["state_digest"],
            receipt=receipt,
            opaque_state=opaque,
        )

    def _velocity(
        self, state: materializer.PreparedSameStateV1, role: str, role_ordinal: int
    ) -> materializer.FP32TensorV1:
        shape = state.opaque_state["shape"]
        _, channels, phases, height, width = shape
        channel_by_role = {
            "branch": 0,
            "camera_only": 1,
            "appearance_only": 2,
            "wrong_actor": 3,
            "wrong_object": 4,
            "generic_wrong_motion": 5,
        }
        values = []
        for channel in range(channels):
            for phase in range(phases):
                for _y in range(height):
                    for _x in range(width):
                        value = 0.001 * phase if channel == 15 else 0.0
                        if role in channel_by_role and channel == channel_by_role[role]:
                            value += 0.05 * phase
                        if (
                            self.nondeterministic
                            and role == "branch"
                            and role_ordinal == 2
                            and channel == 6
                        ):
                            value += 0.02 * phase
                        values.append(materializer._f32(value))
        return materializer.FP32TensorV1(shape, tuple(values))

    def forward_post_head_v1(
        self,
        *,
        state: materializer.PreparedSameStateV1,
        condition: Mapping[str, Any],
    ) -> materializer.FrozenForwardResultV1:
        self.forward_calls += 1
        self.forward_state_objects.setdefault(state.state_digest, set()).add(
            id(state.opaque_state)
        )
        key = (state.state_digest, str(condition["role"]))
        ordinal = self.role_calls.get(key, 0) + 1
        self.role_calls[key] = ordinal
        velocity = self._velocity(state, str(condition["role"]), ordinal)
        velocity_sha = velocity.raw_sha256()
        unsigned = {
            "schema_version": materializer.FORWARD_RECEIPT_SCHEMA_VERSION,
            "provider_abi": materializer.OFFICIAL_PROVIDER_ABI,
            "official_provider": False,
            "record_id": state.record_id,
            "condition_role": condition["role"],
            "condition_utf8_sha256": condition["instruction_utf8_sha256"],
            "shared_state_digest": state.state_digest,
            "runtime_digest": self.identity["runtime_digest"],
            "sigma_index": state.sigma_index,
            "sigma_float32_be_hex": state.sigma_float32_be_hex,
            "timestep": state.timestep,
            "output_stage": "wrong-stage" if self.wrong_stage else materializer.POST_HEAD_STAGE,
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
        return materializer.FrozenForwardResultV1(
            velocity=velocity,
            receipt=materializer.seal_record(unsigned, "forward_digest"),
        )

    def consensus_digest_v1(self, digest: str, label: str) -> None:
        del label
        self.consensus.append(digest)

    def broadcast_rank0_v1(self, value: Any) -> Any:
        return value

    def barrier_v1(self) -> None:
        self.barriers += 1

    def close(self) -> None:
        self.closed = True


class Full30PsiOutMaterializerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.builder = FixtureBuilder(self.root)
        self.plan = self.builder.build()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _output(self, name: str = "output") -> Path:
        return self.root / name

    def test_positive_full_population_writes_current_validator_containers(self) -> None:
        provider = FakeFrozenProvider(self.plan)
        result = materializer.materialize_with_test_provider_v1(
            self.plan, output_directory=self._output(), provider=provider
        )
        self.assertTrue(result.test_only)
        self.assertFalse(result.official_provider)
        self.assertEqual(result.record_count, 4)
        self.assertEqual(provider.prepare_calls, 24)
        self.assertEqual(provider.forward_calls, 114)
        self.assertTrue(all(len(ids) == 1 for ids in provider.forward_state_objects.values()))
        self.assertEqual(provider.barriers, 1)
        run = json.loads(result.run_receipt_path.read_bytes())
        self.assertEqual(set(run), materializer._RUN_RECEIPT_FIELDS)
        materializer._verify_seal(run, "run_digest", "test run receipt")
        self.assertEqual(run["plan_authority"], self.plan)
        self.assertTrue(run["test_only"])
        self.assertFalse(run["official_provider"])
        self.assertEqual(run["computation_digest"], provider.consensus[-1])
        sigma_receipt = run["sigma_authority"]
        self.assertEqual(
            set(sigma_receipt), materializer._SIGMA_AUTHORITY_RECEIPT_FIELDS
        )
        materializer._verify_seal(
            sigma_receipt, "sigma_authority_digest", "test sigma receipt"
        )
        self.assertEqual(
            sigma_receipt["schedule_sha256"], sigma_authority.SCHEDULE_SHA256
        )
        self.assertEqual(
            [row["sigma_index"] for row in sigma_receipt["materialized_rows"]],
            list(materializer.SIGMA_INDICES),
        )
        self.assertEqual(len(run["representation_sigma_evidence_candidates"]), 1)
        representation = run["representation_sigma_evidence_candidates"][0]
        self.assertEqual(len(representation["sigma_evidence"]), 6)
        self.assertTrue(
            all(row["same_event_cosine"] > 0.999 for row in representation["sigma_evidence"])
        )
        self.assertEqual(len(run["amplitude_sigma_calibration_candidates"]), 1)
        parsed_teacher: dict[str, dict[str, Any]] = {}
        teacher_evidence: dict[str, Mapping[str, Any]] = {}
        parsed_amplitude: dict[str, Mapping[int, Any]] = {}
        for reference in run["record_receipts"]:
            self.assertEqual(
                set(reference), materializer._RUN_RECORD_REFERENCE_FIELDS
            )
            receipt_path = Path(reference["path"])
            self.assertEqual(stat.S_IMODE(receipt_path.stat().st_mode), 0o600)
            self.assertEqual(
                materializer.file_sha256(receipt_path), reference["file_sha256"]
            )
            receipt = json.loads(receipt_path.read_bytes())
            self.assertEqual(set(receipt), materializer._RECORD_RECEIPT_FIELDS)
            materializer._verify_seal(
                receipt, "record_receipt_digest", "test record receipt"
            )
            self.assertEqual(
                receipt["record_receipt_digest"],
                reference["record_receipt_digest"],
            )
            materializer._verify_seal(
                receipt["record_authority"], "record_digest", "test plan record"
            )
            self.assertEqual(
                receipt["record_conditions"],
                receipt["record_authority"]["conditions"],
            )
            self.assertEqual(
                receipt["sigma_authority_digest"],
                sigma_receipt["sigma_authority_digest"],
            )
            self.assertEqual(len(receipt["state_receipts"]), 6)
            self.assertTrue(
                all(
                    set(row) == materializer._STATE_RECEIPT_FIELDS
                    for row in receipt["state_receipts"]
                )
            )
            self.assertTrue(
                all(
                    set(row) == materializer._FORWARD_RECEIPT_FIELDS
                    for row in receipt["forward_receipts"]
                )
            )
            self.assertEqual(
                set(receipt["noise_replay_receipt"]),
                materializer._NOISE_RECEIPT_FIELDS,
            )
            self.assertEqual(
                receipt["source_posterior_index0_sha256"],
                receipt["record_authority"]["source_posterior_index0_sha256"],
            )
            for binding in receipt["container_bindings"]:
                self.assertEqual(
                    set(binding), materializer._CONTAINER_BINDING_FIELDS
                )
                path = Path(binding["path"])
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
                if binding["container_kind"] in ("psiout", "nuisance"):
                    parsed = teacher_authority._validate_tensor_container(
                        str(path),
                        binding["file_sha256"],
                        container_kind=binding["container_kind"],
                        evidence_id=receipt["evidence_id"],
                        evidence_role=receipt["evidence_role"],
                        teacher_cell_id=receipt["teacher_cell_id"],
                        branch=receipt["branch"],
                        label="test consumed teacher container",
                    )
                    parsed_teacher.setdefault(receipt["evidence_role"], {})[
                        binding["container_kind"]
                    ] = parsed
                    teacher_evidence[receipt["evidence_role"]] = receipt[
                        "candidate_authority_evidence"
                    ]
                else:
                    evidence = receipt["candidate_authority_evidence"]
                    parsed_amplitude[evidence["evidence_id"]] = (
                        amplitude_authority._validate_container(
                        str(path),
                        binding["file_sha256"],
                        evidence_id=evidence["evidence_id"],
                        pair_id=evidence["pair_id"],
                        source_iid=evidence["source_iid"],
                        teacher_cell_id=evidence["teacher_cell_id"],
                        branch=evidence["branch"],
                        label="test consumed amplitude container",
                        )
                    )
                    self.assertFalse(evidence["official_frozen_native_only"])
        teacher_authority._validate_sigma_evidence(
            representation["sigma_evidence"],
            origin_anchor_id=teacher_evidence["teacher_origin"]["anchor_id"],
            cross_anchor_id=teacher_evidence["same_event_cross_anchor"]["anchor_id"],
            origin_sidecar=parsed_teacher["teacher_origin"]["psiout"],
            cross_sidecar=parsed_teacher["same_event_cross_anchor"]["psiout"],
            origin_nuisance=parsed_teacher["teacher_origin"]["nuisance"],
            label="test consumed representation sigma fragment",
        )
        amplitude_fragment = run["amplitude_sigma_calibration_candidates"][0]
        for sigma_row in amplitude_fragment["sigma_calibrations"]:
            observed_norms = []
            for metric in sigma_row["calibrator_metrics"]:
                tensor = parsed_amplitude[metric["evidence_id"]][
                    sigma_row["sigma_index"]
                ]
                actual_norm = amplitude_authority._norm(tensor.values)
                self.assertEqual(metric["projected_slice_sha256"], tensor.sha256)
                self.assertTrue(
                    math.isclose(
                        metric["amplitude_norm"],
                        actual_norm,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                )
                observed_norms.append(actual_norm)
            self.assertTrue(
                math.isclose(
                    sigma_row["median_amplitude"],
                    math.fsum(sorted(observed_norms)) / 2.0,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            )

    def test_wrong_post_head_stage_fails_before_output(self) -> None:
        output = self._output()
        with self.assertRaisesRegex(
            materializer.Full30PsiOutMaterializationError, "output_stage"
        ):
            materializer.materialize_with_test_provider_v1(
                self.plan,
                output_directory=output,
                provider=FakeFrozenProvider(self.plan, wrong_stage=True),
            )
        self.assertFalse(output.exists())

    def test_nondeterministic_duplicate_fails_before_output(self) -> None:
        output = self._output()
        with self.assertRaisesRegex(
            materializer.Full30PsiOutMaterializationError, "not byte deterministic"
        ):
            materializer.materialize_with_test_provider_v1(
                self.plan,
                output_directory=output,
                provider=FakeFrozenProvider(self.plan, nondeterministic=True),
            )
        self.assertFalse(output.exists())

    def test_noise_replay_mismatch_fails_before_any_forward(self) -> None:
        provider = FakeFrozenProvider(self.plan, bad_noise=True)
        with self.assertRaisesRegex(
            materializer.Full30PsiOutMaterializationError, "noise replay receipt"
        ):
            materializer.materialize_with_test_provider_v1(
                self.plan, output_directory=self._output(), provider=provider
            )
        self.assertEqual(provider.forward_calls, 0)
        self.assertFalse(self._output().exists())

    def test_source_target_clean_byte_mismatch_is_rejected(self) -> None:
        plan = deepcopy(self.plan)
        amplitude = plan["records"][2]
        amplitude["source_clean_latent"] = self.builder.artifact("wrong-source", 9.0)
        plan["records"][2] = materializer.seal_record(amplitude, "record_digest")
        plan = materializer.seal_record(plan, "plan_digest")
        with self.assertRaisesRegex(
            materializer.Full30PsiOutMaterializationError,
            "same-mode source/target clean bytes differ",
        ):
            materializer.validate_materialization_plan_v1(plan)

    def test_runtime_helper_tamper_is_rejected(self) -> None:
        plan = deepcopy(self.plan)
        plan["runtime"]["official_helper_sources"][0]["file_sha256"] = "0" * 64
        plan["runtime"] = materializer.seal_record(
            plan["runtime"], "runtime_plan_digest"
        )
        plan = materializer.seal_record(plan, "plan_digest")
        with self.assertRaisesRegex(
            materializer.Full30PsiOutMaterializationError, "SHA-256 differs"
        ):
            materializer.validate_materialization_plan_v1(plan)

    def test_reviewed_media_tamper_is_rejected(self) -> None:
        media_path = Path(self.plan["records"][0]["reviewed_media"]["path"])
        media_path.write_bytes(b"tampered")
        with self.assertRaisesRegex(
            materializer.Full30PsiOutMaterializationError, "SHA-256 differs"
        ):
            materializer.validate_materialization_plan_v1(self.plan)

    def test_physical_source_posterior_tamper_is_rejected(self) -> None:
        source_path = Path(
            self.plan["records"][2]["source_posterior_index0_path"]
        )
        source_path.write_bytes(b"tampered-source-posterior")
        with self.assertRaisesRegex(
            materializer.Full30PsiOutMaterializationError, "SHA-256 differs"
        ):
            materializer.validate_materialization_plan_v1(self.plan)

    def test_create_only_output_rejects_overwrite(self) -> None:
        output = self._output()
        materializer.materialize_with_test_provider_v1(
            self.plan, output_directory=output, provider=FakeFrozenProvider(self.plan)
        )
        with self.assertRaisesRegex(
            materializer.Full30PsiOutMaterializationError, "already exists"
        ):
            materializer.materialize_with_test_provider_v1(
                self.plan, output_directory=output, provider=FakeFrozenProvider(self.plan)
            )

    def test_formal_cli_has_no_fake_switch_and_fails_closed_off_world4(self) -> None:
        plan_path = self.root / "plan.json"
        plan_sha = _write_json(plan_path, self.plan)
        output = self._output("formal-output")
        self.assertEqual(
            materializer.main(
                [
                    "--plan",
                    str(plan_path),
                    "--plan-sha256",
                    plan_sha,
                    "--output-dir",
                    str(output),
                ]
            ),
            2,
        )
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
