#!/usr/bin/env python3
"""CPU/authority tests for the isolated Round37 activation-v2 candidate."""

from __future__ import annotations

from contextlib import contextmanager
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import oracle_regeneration_activation_v2 as activation
import self_guided_action_field_v1 as sgaf


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object, *, frozen: bool = True) -> Path:
    path.write_bytes(activation.safe_core.canonical_json_bytes_v1(value))
    if frozen:
        path.chmod(0o444)
    return path


def _file(path: Path, payload: bytes, *, frozen: bool = False) -> tuple[Path, str]:
    path.write_bytes(payload)
    if frozen:
        path.chmod(0o444)
    return path, _sha(path)


class ActivationV2StaticTests(unittest.TestCase):
    def test_checked_in_candidate_is_hard_disabled(self) -> None:
        self.assertIsNone(activation.COMPILED_AUTHORITY_PACKET_SHA256)
        self.assertIsNone(activation.COMPILED_EXTERNAL_LEDGER_RECEIPT_SHA256)
        self.assertFalse(activation.compiled_activation_available_v2())
        with self.assertRaisesRegex(activation.OracleActivationV2Error, "not compiled"):
            activation.load_compiled_activation_authority_v2(
                Path("/not/used"), Path("/not/used")
            )

    def test_new_python_sources_have_no_literal_duplicate_dict_keys(self) -> None:
        paths = (
            METHOD_ROOT / "oracle_regeneration_activation_v2.py",
            METHOD_ROOT / "tools/materialize_oracle_regeneration_vae_refs_activation_v2.py",
            METHOD_ROOT / "tools/materialize_oracle_regeneration_prompts_activation_v2.py",
        )
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Dict):
                    continue
                keys = [
                    key.value
                    for key in node.keys
                    if isinstance(key, ast.Constant)
                    and isinstance(key.value, (str, int, float, bytes, bool))
                ]
                self.assertEqual(
                    len(keys), len(set(keys)), f"duplicate literal dict key in {path}:{node.lineno}"
                )


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch required")
class ActivationV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch

        cls.torch = torch

    def _tensor_identity(self, value):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "content_sha256": activation.safe_core.tensor_content_sha256_v1(value),
        }

    def _make_gate(
        self,
        root: Path,
        *,
        case_row: dict,
        annotator: str,
        reviewer: str,
        geometry: tuple[int, int, int, int, int],
        contact_start: int = 4,
    ) -> tuple[Path, str, Path, str, str]:
        review_path = root / "e02-gate-review.json"
        delete = [[] for _ in range(21)]
        create = [[] for _ in range(21)]
        contact = [[] for _ in range(21)]
        for phase in range(1, 21):
            delete[phase] = [[0, 1]]
            if phase >= contact_start:
                contact[phase] = [[1, 4]]
            if phase >= 5:
                create[phase] = [[5, 1]]
        mask_payload = activation._manual_gate_payload_v2(
            geometry=geometry,
            delete_rle=delete,
            create_rle=create,
            contact_rle=contact,
        )
        mask_sha = hashlib.sha256(
            activation.safe_core.canonical_json_bytes_v1(mask_payload)
        ).hexdigest()
        leaf_payload = {
            "schema_version": activation.MANUAL_GATE_LEAF_SCHEMA_VERSION,
            "case_id": "e02",
            "source_sha256": case_row["source_video"]["sha256"],
            "anchor_sha256": case_row["self_generated_anchor"]["sha256"],
            "action_caption_sha256": case_row["action_caption_sha256"],
            "structured_action_program_sha256": case_row[
                "structured_action_program_sha256"
            ],
            "mask_sha256": mask_sha,
            "annotator": annotator,
            "reviewer": reviewer,
        }
        leaf_sha = hashlib.sha256(
            b"\x00" + activation.safe_core.canonical_json_bytes_v1(leaf_payload)
        ).hexdigest()
        gate = {
            "schema_version": activation.MANUAL_GATE_SCHEMA_VERSION,
            "case_id": "e02",
            "source_sha256": case_row["source_video"]["sha256"],
            "anchor_sha256": case_row["self_generated_anchor"]["sha256"],
            "action_caption_sha256": case_row["action_caption_sha256"],
            "structured_action_program_sha256": case_row[
                "structured_action_program_sha256"
            ],
            "latent_geometry": list(geometry),
            "flattening": "per_phase_row_major_yx",
            "dtype": "bool",
            "hard_support": True,
            "phase_zero_empty": True,
            "delete_rle": delete,
            "create_rle": create,
            "contact_rle": contact,
            "typed_semantics": {
                "delete_D": "obsolete_source_state_occupancy_to_delete",
                "create_C": "new_actor_object_state_occupancy_to_create",
                "contact_ownership_K": "contact_and_ownership_transition_permission_corridor",
                "execution_support_G": "exact_boolean_union_D_or_C_or_K",
                "coordinate_system": "source_latent_phase_y_x",
                "expected_nonempty_phase_windows": {
                    "delete_D": [1, 20],
                    "create_C": [5, 20],
                    "contact_ownership_K": [4, 20],
                    "execution_support_G": [1, 20],
                },
            },
            "mask_sha256": mask_sha,
            "annotation_authority": {
                "tree_shape": activation.safe_core.ANNOTATION_TREE_SHAPE,
                "ledger_root_sha256": leaf_sha,
                "leaf_sha256": leaf_sha,
                "leaf_index": 0,
                "tree_size": 1,
                "inclusion_proof": [],
            },
            "authority": {
                "role": "source_only_model_proposal_diagnostic_intervention_only",
                "training_target_authorized": False,
                "action_representation_claimed": False,
                "forbidden_inputs_absent": {
                    "failed_active_video_or_latent": True,
                    "raw_anchor_source_pixel_or_latent_difference": True,
                    "predicted_soft_gate": True,
                    "target_video_or_latent": True,
                    "self_generated_anchor_tensor": True,
                },
            },
            "qualification": {
                "status": "independent_model_reviewed_diagnostic_exact_gate",
                "annotator": annotator,
                "reviewer": reviewer,
                "author_kind": "AI_AGENT",
                "reviewer_kind": "AI_AGENT",
                "review_receipt_path": str(review_path),
            },
        }
        gate_path = root / "e02-gate.json"
        _write(gate_path, gate, frozen=False)
        gate_sha = _sha(gate_path)
        review = {
            "schema_version": activation.MANUAL_GATE_REVIEW_SCHEMA_VERSION,
            "case_id": "e02",
            "source_sha256": case_row["source_video"]["sha256"],
            "anchor_sha256": case_row["self_generated_anchor"]["sha256"],
            "action_caption_sha256": case_row["action_caption_sha256"],
            "structured_action_program_sha256": case_row[
                "structured_action_program_sha256"
            ],
            "gate_manifest_sha256": gate_sha,
            "mask_sha256": mask_sha,
            "annotation_authority_root_sha256": leaf_sha,
            "annotation_authority_leaf_sha256": leaf_sha,
            "annotator": annotator,
            "reviewer": reviewer,
            "author_kind": "AI_AGENT",
            "reviewer_kind": "AI_AGENT",
            "source_only_model_proposal": True,
            "independent_model_review": True,
            "accepted": True,
            "phase_zero_source_authority_checked": True,
            "source_coordinate_authoring_checked": True,
            "delete_create_contact_semantics_checked": True,
            "D_C_disjoint_checked": True,
            "K_preserved_as_independent_channel_checked": True,
            "G_exact_union_D_C_K_checked": True,
            "channel_active_windows_checked": True,
            "no_large_rectangle_shortcut_checked": True,
            "single_actor_object_component_checked": True,
            "duplicate_actor_or_object_rejected": True,
            "terminal_hold_semantics_checked": True,
            "anchor_terminal_disappearance_observed": True,
            "anchor_strict_target_pass": False,
            "anchor_used_only_as_review_context": True,
            "failed_active_used_to_author_mask": False,
            "anchor_difference_used_to_author_mask": False,
            "predicted_soft_gate_used_to_author_mask": False,
            "target_video_or_latent_used_to_author_mask": False,
            "self_generated_anchor_tensor_used_to_author_mask": False,
        }
        _write(review_path, review)
        gate_path.chmod(0o444)
        return gate_path, gate_sha, review_path, _sha(review_path), leaf_sha

    def _make_reference_receipt(
        self,
        root: Path,
        *,
        case_id: str,
        source_iid: str,
        source_sha: str,
        source,
        references,
        bucket_hw: tuple[int, int],
    ) -> tuple[Path, str]:
        dependency = {}
        for name in (
            "decode",
            "source-prepare",
            "checkpoint-manifest",
            "config",
            "vae-code",
            "vae-materializer",
        ):
            path, digest = _file(
                root / f"{case_id}-vae-{name}.pin", name.encode(), frozen=True
            )
            dependency[name] = (path, digest)
        raw_sha = tuple(hashlib.sha256(f"raw-{index}".encode()).hexdigest() for index in range(4))
        pre_sha = tuple(hashlib.sha256(f"pre-{index}".encode()).hexdigest() for index in range(4))
        source_sha_tensor = activation.safe_core.tensor_content_sha256_v1(source)
        ref_sha = tuple(
            activation.safe_core.tensor_content_sha256_v1(value) for value in references
        )
        rows = [
            {
                "frame_index": frame_index,
                "raw_rgb_sha256": raw_sha[position],
                "preprocessed_rgb_sha256": pre_sha[position],
                "shape": list(references[position].shape),
                "dtype": str(references[position].dtype),
                "content_sha256": ref_sha[position],
                "independently_vae_encoded": True,
            }
            for position, frame_index in enumerate(activation.REFERENCE_RGB_INDICES)
        ]
        payload = {
            "schema_version": activation.REFERENCE_RECEIPT_SCHEMA_VERSION,
            "case_id": case_id,
            "source_iid": source_iid,
            "source_video_sha256": source_sha,
            "source_frame_count": 81,
            "source_fps_numerator": 25,
            "source_fps_denominator": 1,
            "source_input_frame_geometry": [16, 16, 3],
            "source_bucket_hw": list(bucket_hw),
            "reference_rgb_indices": list(activation.REFERENCE_RGB_INDICES),
            "reference_raw_rgb_sha256": list(raw_sha),
            "full_preprocessed_source_identity": {
                "shape": [1, 3, 81, *bucket_hw],
                "dtype": "torch.float32",
                "content_sha256": "a" * 64,
            },
            "reference_preprocessed_rgb_sha256": list(pre_sha),
            "preprocess_contract": {
                "frame_decode_backend": "decord_cpu0_num_threads1_private_source_snapshot",
                "frame_decode_code_path": str(dependency["decode"][0]),
                "frame_decode_code_sha256": dependency["decode"][1],
                "source_prepare_code_path": str(dependency["source-prepare"][0]),
                "source_prepare_code_sha256": dependency["source-prepare"][1],
                "rgb_dtype": "uint8",
                "rgb_channel_order": "RGB",
                "resize_policy": "torchvision_bicubic_antialias_true_source_aspect_bucket",
                "normalization": "uint8_div255_mul2_sub1_float32",
            },
            "vae_contract": {
                "checkpoint_content_manifest_path": str(
                    dependency["checkpoint-manifest"][0]
                ),
                "checkpoint_content_manifest_sha256": dependency[
                    "checkpoint-manifest"
                ][1],
                "checkpoint_content_identity_sha256": "c" * 64,
                "config_path": str(dependency["config"][0]),
                "config_sha256": dependency["config"][1],
                "vae_code_path": str(dependency["vae-code"][0]),
                "vae_code_sha256": dependency["vae-code"][1],
                "encode_function": "bernini.pipeline._vae_encode",
                "encode_dtype": "torch.float32",
                "latent_coordinate": "official_bernini_vae_encode_output",
            },
            "full_source_latent_identity": self._tensor_identity(source),
            "reference_latent_identities": rows,
            "materializer_code_path": str(dependency["vae-materializer"][0]),
            "materializer_code_sha256": dependency["vae-materializer"][1],
            "rank_world_receipt": {
                "world_size": 4,
                "sequence_parallel_size": 4,
                "rank0_only_vae_encode": True,
                "broadcast_exact": True,
                "all_rank_full_source_latent_sha256": [source_sha_tensor] * 4,
                "all_rank_reference_latent_sha256": [list(ref_sha) for _ in range(4)],
            },
            "references_encoded_as_four_independent_rgb_frames": True,
            "references_not_sliced_from_full_source_latent": True,
            "source_reference_storage_alias_rejected": True,
            "reference_content_duplicates_rejected": True,
            "target_video_or_latent_used": False,
            "self_generated_anchor_tensor_used": False,
            "materialization_checks_passed": True,
        }
        path = _write(root / f"{case_id}-vae-receipt.json", payload)
        return path, _sha(path)

    def _make_prompt_receipt(
        self,
        root: Path,
        *,
        case_id: str,
        source_iid: str,
        caption: str,
        prompts,
    ) -> tuple[Path, str]:
        dependency = {}
        for name in (
            "tokenizer-config",
            "tokenizer-code",
            "checkpoint-manifest",
            "text-config",
            "renderer-code",
            "prompt-builder-code",
            "native-prompt-code",
            "prompt-cleaner-code",
            "prompt-materializer",
        ):
            path, digest = _file(
                root / f"{case_id}-prompt-{name}.pin", name.encode(), frozen=True
            )
            dependency[name] = (path, digest)
        modes = ("low-vr2v", "high-r2v4", "renderer-negative")
        keys = ("low_action", "high_action", "negative")
        role_rows = {}
        digests = []
        for key, mode, tensor in zip(keys, modes, prompts):
            rendered = f"{mode}:{caption}"
            identity = self._tensor_identity(tensor)
            digests.append(identity["content_sha256"])
            role_rows[key] = {
                "mode": mode,
                "rendered_text": rendered,
                "rendered_text_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                "token_ids_sha256": hashlib.sha256(f"ids-{mode}".encode()).hexdigest(),
                "attention_mask_sha256": hashlib.sha256(f"mask-{mode}".encode()).hexdigest(),
                "embedding_identity": identity,
            }
        payload = {
            "schema_version": activation.PROMPT_RECEIPT_SCHEMA_VERSION,
            "case_id": case_id,
            "source_iid": source_iid,
            "action_caption": caption,
            "action_caption_sha256": hashlib.sha256(caption.encode()).hexdigest(),
            "prompt_contract": {
                "tokenizer_config_path": str(dependency["tokenizer-config"][0]),
                "tokenizer_config_sha256": dependency["tokenizer-config"][1],
                "tokenizer_code_path": str(dependency["tokenizer-code"][0]),
                "tokenizer_code_sha256": dependency["tokenizer-code"][1],
                "checkpoint_content_manifest_path": str(
                    dependency["checkpoint-manifest"][0]
                ),
                "checkpoint_content_manifest_sha256": dependency[
                    "checkpoint-manifest"
                ][1],
                "checkpoint_content_identity_sha256": "c" * 64,
                "text_encoder_config_path": str(dependency["text-config"][0]),
                "text_encoder_config_sha256": dependency["text-config"][1],
                "renderer_code_path": str(dependency["renderer-code"][0]),
                "renderer_code_sha256": dependency["renderer-code"][1],
                "prompt_builder_code_path": str(
                    dependency["prompt-builder-code"][0]
                ),
                "prompt_builder_code_sha256": dependency["prompt-builder-code"][1],
                "native_prompt_code_path": str(dependency["native-prompt-code"][0]),
                "native_prompt_code_sha256": dependency["native-prompt-code"][1],
                "prompt_cleaner_code_path": str(
                    dependency["prompt-cleaner-code"][0]
                ),
                "prompt_cleaner_code_sha256": dependency["prompt-cleaner-code"][1],
                "tokenizer_function": "infer_lora._tokenize_training_prompt+_tokenize_renderer_negative",
                "text_encoder_function": "bernini.models.renderer.BerniniRendererModel.encode_prompt",
                "max_length": 512,
                "embedding_dtype": "torch.bfloat16",
            },
            **role_rows,
            "materializer_code_path": str(dependency["prompt-materializer"][0]),
            "materializer_code_sha256": dependency["prompt-materializer"][1],
            "rank_world_receipt": {
                "world_size": 4,
                "sequence_parallel_size": 4,
                "rank0_only_text_encode": True,
                "broadcast_exact": True,
                "all_rank_low_action_sha256": [digests[0]] * 4,
                "all_rank_high_action_sha256": [digests[1]] * 4,
                "all_rank_negative_sha256": [digests[2]] * 4,
            },
            "rank0_only_text_encoder_load": True,
            "nonzero_ranks_never_deserialized_text_encoder": True,
            "self_generated_anchor_tensor_used": False,
            "target_video_or_latent_used": False,
            "materialization_checks_passed": True,
        }
        path = _write(root / f"{case_id}-prompt-receipt.json", payload)
        return path, _sha(path)

    @contextmanager
    def _authority_fixture(
        self,
        *,
        cryptographic_signature_claimed: bool = False,
        contact_start: int = 4,
    ):
        torch = self.torch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(strict=True)
            geometry = (1, 16, 21, 6, 6)
            gate_geometry = (1, 1, 21, 6, 6)
            reference_geometry = (1, 16, 1, 6, 6)
            source = torch.zeros(geometry, dtype=torch.float32).contiguous()
            references = tuple(
                torch.full(reference_geometry, float(index + 1), dtype=torch.float32)
                for index in range(4)
            )
            prompts = tuple(
                torch.full(
                    (1, 512, 4096), float(index + 1), dtype=torch.bfloat16
                ).contiguous()
                for index in range(3)
            )
            media = {}
            for case_id in activation.ALLOWED_CASES:
                source_path, source_sha = _file(
                    root / f"{case_id}-source.mp4", f"{case_id}-source".encode()
                )
                anchor_path, anchor_sha = _file(
                    root / f"{case_id}-anchor.mp4", f"{case_id}-anchor".encode()
                )
                media[case_id] = (source_path, source_sha, anchor_path, anchor_sha)
            captions = {
                "e02": (
                    "The same pale bare hand firmly grips the same red mushroom at its "
                    "lower stem, twists and pulls it free from the same soil, lifts the "
                    "same intact mushroom above the newly empty root hole, and holds it "
                    "there. Exactly one hand and one mushroom remain visible; do not "
                    "duplicate, split, or fuse either one."
                ),
                "e03": (
                    "The same farm worker moves the same harvested root cluster over the "
                    "same woven basket, lowers it past the rim, opens the same hand, and "
                    "releases it. The cluster falls, bounces slightly, and settles inside "
                    "while the now-empty hand withdraws. Do not duplicate the hand or "
                    "cluster."
                ),
            }
            programs = {
                "e02": {
                    "entities": ["same_pale_bare_hand", "same_red_mushroom"],
                    "states": ["rooted", "same_hand_contact", "detached", "held"],
                },
                "e03": {
                    "entities": ["same_farm_worker_hand", "same_harvested_root_cluster"],
                    "states": ["held", "released", "settled", "empty_hand"],
                },
            }
            cases = []
            expected_bindings = {}
            for index, case_id in enumerate(activation.ALLOWED_CASES):
                source_path, source_sha, anchor_path, anchor_sha = media[case_id]
                caption = captions[case_id]
                caption_sha = hashlib.sha256(caption.encode()).hexdigest()
                program_sha = activation._canonical_object_sha256(programs[case_id])
                source_iid = f"unit-{case_id}-source"
                expected_bindings[case_id] = {
                    "source_iid": source_iid,
                    "source_sha256": source_sha,
                    "anchor_sha256": anchor_sha,
                    "action_caption_sha256": caption_sha,
                    "structured_action_program_sha256": program_sha,
                }
                reference_path, reference_sha = self._make_reference_receipt(
                    root,
                    case_id=case_id,
                    source_iid=source_iid,
                    source_sha=source_sha,
                    source=source,
                    references=references,
                    bucket_hw=(48, 48),
                )
                prompt_path, prompt_sha = self._make_prompt_receipt(
                    root,
                    case_id=case_id,
                    source_iid=source_iid,
                    caption=caption,
                    prompts=prompts,
                )
                common = {
                    "case_id": case_id,
                    "source_iid": source_iid,
                    "decision": (
                        "ACTIVE_DIAGNOSTIC" if case_id == "e02" else "ABSTAIN_KEEP_BASE"
                    ),
                    "source_video": {"path": str(source_path), "sha256": source_sha},
                    "self_generated_anchor": {
                        "path": str(anchor_path),
                        "sha256": anchor_sha,
                    },
                    "anchor_used_only_as_review_context": True,
                    "self_generated_anchor_tensor_used_by_native_expert": False,
                    "target_video_or_latent_used": False,
                    "failed_active_used_to_author_gate": False,
                    "anchor_source_difference_used_to_author_gate": False,
                    "predicted_soft_gate_used_to_author_gate": False,
                    "automatic_model_replacement_authorized": False,
                    "action_caption": caption,
                    "action_caption_sha256": caption_sha,
                    "structured_action_program": programs[case_id],
                    "structured_action_program_sha256": program_sha,
                    "seed": 101 + index,
                    "full_source_latent_geometry": list(geometry),
                    "hard_gate_geometry": list(gate_geometry),
                    "reference_latent_geometry": list(reference_geometry),
                    "reference_rgb_indices": list(activation.REFERENCE_RGB_INDICES),
                    "run_arms": list(
                        activation.EXPECTED_ARMS_E02
                        if case_id == "e02"
                        else activation.EXPECTED_ARMS_E03
                    ),
                    "manual_gate_manifest": None,
                    "independent_review_receipt": None,
                    "annotation_authority_root_sha256": None,
                    "vae_reference_receipt": {
                        "path": str(reference_path),
                        "sha256": reference_sha,
                    },
                    "prompt_receipt": {
                        "path": str(prompt_path),
                        "sha256": prompt_sha,
                    },
                }
                if case_id == "e02":
                    common.update(
                        {
                            "anchor_terminal_disappearance_observed": True,
                            "anchor_strict_target_pass": False,
                        }
                    )
                    gate_path, gate_sha, review_path, review_sha, root_sha = (
                        self._make_gate(
                            root,
                            case_row=common,
                            annotator="annotator.unit",
                            reviewer="reviewer.unit",
                            geometry=gate_geometry,
                            contact_start=contact_start,
                        )
                    )
                    common["manual_gate_manifest"] = {
                        "path": str(gate_path),
                        "sha256": gate_sha,
                    }
                    common["independent_review_receipt"] = {
                        "path": str(review_path),
                        "sha256": review_sha,
                    }
                    common["annotation_authority_root_sha256"] = root_sha
                else:
                    common["local_regeneration_selection_authorized"] = False
                cases.append(common)
            packet = {
                "schema_version": activation.AUTHORITY_PACKET_SCHEMA_VERSION,
                "status": "INDEPENDENT_MODEL_REVIEWED_DIAGNOSTIC_EXPERIMENTAL_PACKET",
                "packet_id": "round37-unit-authority",
                "execution_contract": {
                    "native_only": True,
                    "flowedit_enabled": False,
                    "connected_runner_enabled": False,
                    "learned_gate_enabled": False,
                    "world_size": 4,
                    "sequence_parallel_size": 4,
                    "one_node": True,
                    "same_seed_and_official_gaussian": True,
                    "candidate_count_per_arm": 1,
                    "source_reference_r2v4_regeneration_expert": True,
                    "self_generated_anchor_tensor_used_by_native_expert": False,
                    "anchor_reference_or_quotient_arm_deferred": True,
                    "global_source_reference_r2v4_upper_bound_arm_deferred": True,
                },
                "safety_contract": {
                    "training_authorized": False,
                    "optimizer_authorized": False,
                    "automatic_model_replacement_authorized": False,
                    "background_cosine_selection_authorized": False,
                    "target_video_or_latent_used": False,
                },
                "cases": cases,
            }
            packet_path = _write(root / "authority.json", packet)
            packet_sha = _sha(packet_path)
            ledger = {
                "schema_version": activation.LEDGER_RECEIPT_SCHEMA_VERSION,
                "authority_packet_sha256": packet_sha,
                "packet_id": packet["packet_id"],
                "annotator": "annotator.unit",
                "reviewer": "reviewer.unit",
                "issuer": "issuer.unit",
                "annotator_kind": "AI_AGENT",
                "reviewer_kind": "AI_AGENT",
                "issuer_kind": "AI_AGENT",
                "trust_root_kind": "COMPILED_EXACT_PACKET_AND_LEDGER_SHA256_CODE_REVIEW",
                "accepted": True,
                "e02_exact_gate_reviewed": True,
                "e03_abstain_keep_base_reviewed": True,
                "authority_packet_contains_no_activation_code_hashes": True,
                "private_signing_material_present": False,
                "cryptographic_signature_claimed": cryptographic_signature_claimed,
                "diagnostic_experimental_canary_only": True,
                "formal_authority": False,
                "training_authority": False,
            }
            ledger_path = _write(root / "ledger.json", ledger)
            expected_latent = {case_id: geometry for case_id in activation.ALLOWED_CASES}
            expected_gate = {case_id: gate_geometry for case_id in activation.ALLOWED_CASES}
            expected_reference = {
                case_id: reference_geometry for case_id in activation.ALLOWED_CASES
            }
            expected_input_hw = {
                case_id: (16, 16) for case_id in activation.ALLOWED_CASES
            }
            with mock.patch.object(
                activation, "COMPILED_AUTHORITY_PACKET_SHA256", packet_sha
            ), mock.patch.object(
                activation,
                "COMPILED_EXTERNAL_LEDGER_RECEIPT_SHA256",
                _sha(ledger_path),
            ), mock.patch.object(
                activation, "EXPECTED_CASE_BINDINGS", expected_bindings
            ), mock.patch.object(
                activation, "EXPECTED_LATENT_GEOMETRY", expected_latent
            ), mock.patch.object(
                activation, "EXPECTED_GATE_GEOMETRY", expected_gate
            ), mock.patch.object(
                activation, "EXPECTED_REFERENCE_GEOMETRY", expected_reference
            ), mock.patch.object(
                activation, "EXPECTED_SOURCE_INPUT_HW", expected_input_hw
            ):
                yield {
                    "root": root,
                    "packet_path": packet_path,
                    "ledger_path": ledger_path,
                    "source": source,
                    "references": references,
                    "prompts": prompts,
                    "media": media,
                }

    def test_compiled_packet_validates_dck_reference_prompt_and_mints_e02_only(self) -> None:
        with self._authority_fixture() as fixture:
            authority = activation.load_compiled_activation_authority_v2(
                fixture["packet_path"], fixture["ledger_path"]
            )
            gate = activation.validate_manual_gate_v2(authority)
            self.assertEqual(gate.support_count, 104)
            self.assertEqual(gate.contact_count, 68)
            self.assertFalse(bool(gate.owned_gate.contact[:, :, 0:4].any().item()))
            self.assertEqual(
                {
                    phase
                    for phase in range(21)
                    if bool(gate.owned_gate.contact[:, :, phase].any().item())
                },
                set(range(4, 21)),
            )
            reference = activation.validate_reference_receipt_v2(
                authority,
                case_id="e02",
                source_video_latent=fixture["source"],
                source_reference_latents=fixture["references"],
            )
            prompt = activation.validate_prompt_receipt_v2(
                authority,
                case_id="e02",
                low_action_prompt_embeds=fixture["prompts"][0],
                high_action_prompt_embeds=fixture["prompts"][1],
                negative_prompt_embeds=fixture["prompts"][2],
            )
            capability = activation.mint_native_local_execution_capability_v2(
                authority,
                case_id="e02",
                source_video_latent=fixture["source"],
                source_reference_latents=fixture["references"],
                low_action_prompt_embeds=fixture["prompts"][0],
                high_action_prompt_embeds=fixture["prompts"][1],
                negative_prompt_embeds=fixture["prompts"][2],
            )
            self.assertEqual(capability.source_latent_sha256, reference.source_latent_sha256)
            self.assertEqual(capability.r2v_action_prompt_sha256, prompt.high_action_sha256)
            with self.assertRaisesRegex(activation.OracleActivationV2Error, "e03"):
                activation.mint_native_local_execution_capability_v2(
                    authority,
                    case_id="e03",
                    source_video_latent=fixture["source"],
                    source_reference_latents=fixture["references"],
                    low_action_prompt_embeds=fixture["prompts"][0],
                    high_action_prompt_embeds=fixture["prompts"][1],
                    negative_prompt_embeds=fixture["prompts"][2],
                )

    def test_wrong_media_prompt_ref_alias_and_authority_mutation_fail_closed(self) -> None:
        torch = self.torch
        with self._authority_fixture() as fixture:
            authority = activation.load_compiled_activation_authority_v2(
                fixture["packet_path"], fixture["ledger_path"]
            )
            wrong_source = fixture["source"].clone()
            wrong_source[0, 0, 1, 0, 0] = 9.0
            with self.assertRaises(activation.OracleActivationV2Error):
                activation.validate_reference_receipt_v2(
                    authority,
                    case_id="e02",
                    source_video_latent=wrong_source,
                    source_reference_latents=fixture["references"],
                )
            aliases = (
                fixture["references"][0],
                fixture["references"][0],
                fixture["references"][2],
                fixture["references"][3],
            )
            with self.assertRaises(activation.OracleActivationV2Error):
                activation.validate_reference_receipt_v2(
                    authority,
                    case_id="e02",
                    source_video_latent=fixture["source"],
                    source_reference_latents=aliases,
                )
            wrong_high = fixture["prompts"][1].clone()
            wrong_high[0, 0, 0] = torch.tensor(7.0, dtype=torch.bfloat16)
            with self.assertRaises(activation.OracleActivationV2Error):
                activation.validate_prompt_receipt_v2(
                    authority,
                    case_id="e02",
                    low_action_prompt_embeds=fixture["prompts"][0],
                    high_action_prompt_embeds=wrong_high,
                    negative_prompt_embeds=fixture["prompts"][2],
                )
            source_path = fixture["media"]["e02"][0]
            source_path.write_bytes(b"wrong-media")
            with self.assertRaisesRegex(
                activation.OracleActivationV2Error, "graph|bytes|changed"
            ):
                activation.revalidate_compiled_activation_authority_v2(authority)

    def test_cryptographic_signature_claim_cannot_upgrade_code_review_root(self) -> None:
        with self._authority_fixture(cryptographic_signature_claimed=True) as fixture:
            with self.assertRaisesRegex(
                activation.OracleActivationV2Error, "external ledger"
            ):
                activation.load_compiled_activation_authority_v2(
                    fixture["packet_path"], fixture["ledger_path"]
                )

    def test_K_active_window_is_exact_and_rejects_early_contact_permission(self) -> None:
        with self._authority_fixture(contact_start=3) as fixture:
            authority = activation.load_compiled_activation_authority_v2(
                fixture["packet_path"], fixture["ledger_path"]
            )
            with self.assertRaisesRegex(
                activation.OracleActivationV2Error, "topology|phase semantics"
            ):
                activation.validate_manual_gate_v2(authority)

    def _gate_tensor(self, *, null: bool):
        torch = self.torch
        delete = torch.zeros(1, 1, 21, 2, 2, dtype=torch.bool)
        create = torch.zeros_like(delete)
        contact = torch.zeros_like(delete)
        if not null:
            delete[:, :, 1, 0, 0] = True
            contact[:, :, 1, 0, 1] = True
            create[:, :, 1, 1, 1] = True
        support = torch.logical_or(torch.logical_or(delete, create), contact).contiguous()
        preserve = torch.logical_not(support).contiguous()
        provisional = activation._OwnedHardStateChangeGateV2(
            delete=delete,
            create=create,
            contact=contact,
            support=support,
            preserve=preserve,
            source_mask_sha256="b" * 64,
            realized_gate_sha256="0" * 64,
            delete_count=int(delete.sum()),
            create_count=int(create.sum()),
            contact_count=int(contact.sum()),
            support_count=int(support.sum()),
        )
        return activation._OwnedHardStateChangeGateV2(
            **{
                **provisional.__dict__,
                "realized_gate_sha256": activation._realized_gate_sha256_v2(provisional),
            }
        )

    def test_null_G_returns_original_signed_zero_object_before_high_validation(self) -> None:
        torch = self.torch
        shape = (1, 16, 21, 2, 2)
        official = torch.full((1, 21, 64), -0.0, dtype=torch.float32)
        executed, trace = activation._scheduled_local_velocity_v2(
            sample=torch.zeros_like(official),
            high_r2v4_velocity=object(),
            official_v2v_velocity=official,
            sigma=torch.tensor(1.0, dtype=torch.float32),
            gate=self._gate_tensor(null=True),
            target_latent_shape=shape,
        )
        self.assertIs(executed, official)
        self.assertFalse(trace["scheduled_expert_evaluated"])
        self.assertTrue(trace["scheduler_received_original_official_object"])
        self.assertTrue(
            torch.equal(
                executed.contiguous().view(torch.uint8),
                official.contiguous().view(torch.uint8),
            )
        )

    def test_active_DCK_route_keeps_bytes_outside_G(self) -> None:
        torch = self.torch
        shape = (1, 16, 21, 2, 2)
        official = torch.full((1, 21, 64), -0.0, dtype=torch.float32)
        high = torch.ones_like(official)
        gate = self._gate_tensor(null=False)
        executed, trace = activation._scheduled_local_velocity_v2(
            sample=torch.zeros_like(official),
            high_r2v4_velocity=high,
            official_v2v_velocity=official,
            sigma=torch.tensor(1.0, dtype=torch.float32),
            gate=gate,
            target_latent_shape=shape,
        )
        packed = sgaf._spatial_to_packed(gate.support.expand(shape), shape)
        outside = torch.logical_not(packed)
        self.assertTrue(
            torch.equal(
                executed[outside].contiguous().view(torch.uint8),
                official[outside].contiguous().view(torch.uint8),
            )
        )
        self.assertTrue(torch.equal(executed[packed], high[packed]))
        self.assertEqual(trace["hard_support_definition"], "G=D_or_C_or_K")


if __name__ == "__main__":
    unittest.main()
