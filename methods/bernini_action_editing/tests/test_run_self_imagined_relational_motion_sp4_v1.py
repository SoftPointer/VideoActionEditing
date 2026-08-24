from __future__ import annotations

import importlib
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))


RUNTIME_MODULE = "run_self_imagined_relational_motion_sp4_v1"
try:
    runtime = importlib.import_module(RUNTIME_MODULE)
except ModuleNotFoundError as error:  # Runtime is authored after this test contract.
    if error.name not in (RUNTIME_MODULE, "torch"):
        raise
    runtime = None

try:
    import torch
    import self_imagined_relational_motion as relational
except ModuleNotFoundError:  # pragma: no cover - dependency-light contract host
    torch = None
    relational = None


def _parser_destinations(parser) -> set[str]:
    """Return destinations from a parser and all of its subcommands."""

    result: set[str] = set()
    pending = [parser]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        for action in current._actions:
            result.add(action.dest)
            choices = getattr(action, "choices", None)
            if isinstance(choices, dict):
                pending.extend(choices.values())
    return result


def _positive_teacher_helper():
    if runtime is None:
        raise AssertionError("runtime is unavailable")
    for name in (
        "validate_positive_teacher_binding",
        "strict_positive_teacher_binding",
        "load_positive_teacher_binding",
    ):
        candidate = getattr(runtime, name, None)
        if callable(candidate):
            return candidate
    raise AssertionError("runtime has no strict positive-teacher binding helper")


def _mock_positive_teacher_receipt() -> dict[str, object]:
    prompt = {
        "action_raw_caption_utf8_sha256": "4" * 64,
        "noop_raw_caption_utf8_sha256": "5" * 64,
        "action_full_prompt_utf8_sha256": "6" * 64,
        "noop_full_prompt_utf8_sha256": "7" * 64,
        "action_condition_tensor_sha256": "8" * 64,
        "noop_condition_tensor_sha256": "9" * 64,
        "prompt_builder_contract_digest": "a" * 64,
        "all_13_arms_use_cell_fixed_prompt_pair": True,
        "branch_caption_never_used_as_condition": True,
        "detached_labels_never_used_as_condition": True,
        "target_action_candidate_id": "teacher-action",
        "target_noop_candidate_id": "teacher-noop",
    }
    pair_value = {
        "action_full_prompt_utf8_sha256": prompt[
            "action_full_prompt_utf8_sha256"
        ],
        "noop_full_prompt_utf8_sha256": prompt["noop_full_prompt_utf8_sha256"],
        "action_condition_tensor_sha256": prompt[
            "action_condition_tensor_sha256"
        ],
        "noop_condition_tensor_sha256": prompt[
            "noop_condition_tensor_sha256"
        ],
    }
    prompt["prompt_pair_digest"] = hashlib.sha256(
        json.dumps(
            pair_value, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    materializer = runtime.materializer
    freeze_certificate = {
        "base_frozen": True,
        "trainable_parameter_tensors": 0,
        "trainable_parameter_elements": 0,
        "lora_module_count": 0,
    }
    checkpoint_unsigned = {
        "manifest_sha256": (
            runtime.live_bridge.BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256
        ),
        "verified_file_count": 23,
        "verified_entries_digest": (
            "676e6104eebee3ab1066c70f40af385346b013a3afcab8cafb06c5290994d9ba"
        ),
        "every_file_sha256_verified": True,
        "loaded_components": ["transformer_1", "umt5_text_encoder"],
        "all_loaded_parameters_frozen": True,
        "freeze_certificate": freeze_certificate,
    }
    return {
        "schema_version": "bernini-starc-core4-same-state-hidden-arm-v1",
        "group_id": "sp4-a",
        "episode_id": "dog-fit",
        "split": "fit",
        "role": "positive",
        "label": 1,
        "artifact": {
            "path": "/sealed/dog-fit/starc-block15-hidden-residual.safetensors",
            "file_sha256": "1" * 64,
            "tensor_key": "sketched_action_minus_noop_hidden_residual",
            "tensor_shape": [1, 21, 16, 1536],
            "tensor_dtype": "torch.float32",
            "tensor_sha256": "2" * 64,
            "detached_finite_fp32": True,
        },
        "event_label_binding": {
            "complete_target_transition_observed": True,
            "terminal_hold_observed": True,
            "full_target_action_observed": True,
            "full_target_action_false_confirmed": False,
            "labels_are_external_and_detached": True,
            "labels_may_enter_model_condition": False,
        },
        "same_state_query_binding": {
            "native_schedule_index": materializer.SCHEDULE_INDEX,
            "native_timestep": materializer.NATIVE_TIMESTEP,
            "sigma": materializer.SIGMA,
            "action_and_noop_share_exact_x_sigma_object": True,
            "action_and_noop_share_exact_rotary_object": True,
            "action_and_noop_share_exact_timestep_object": True,
            "shared_tensor_bytes_unchanged": True,
            "block0_input_and_attn1_exact_parity": True,
            "source_condition_consumed": False,
            "mask_flow_pose_track_or_trajectory_consumed": False,
            "event_labels_consumed": False,
        },
        "hidden_binding": {
            "hook_coordinate": materializer.HOOK_COORDINATE,
            "residual_shape": [1, 21, 16, 1536],
            "full_hidden_persisted": False,
        },
        "model_binding": {
            "bernini_revision": runtime.live_bridge.BERNINI_OFFICIAL_COMMIT,
            "veomni_revision": runtime.live_bridge.VEOMNI_TESTED_COMMIT,
            "native_schedule_digest": (
                runtime.temporal_scorer.contract.NATIVE_SCHEDULE_DIGEST
            ),
            "native_schedule_index": materializer.SCHEDULE_INDEX,
            "native_timestep": materializer.NATIVE_TIMESTEP,
            "sigma": materializer.SIGMA,
            "hook_coordinate": materializer.HOOK_COORDINATE,
            "transformer_1_only": True,
            "adapter_loaded": False,
            "all_parameters_frozen": True,
            "frozen_checkpoint_receipt_digest": "b" * 64,
            "checkpoint_content_binding": {
                **checkpoint_unsigned,
                "binding_digest": runtime._object_sha256(checkpoint_unsigned),
            },
        },
        "prompt_binding": prompt,
        "source_candidate_binding": {
            "candidate_id": "teacher-action",
            "semantic_branch": runtime.materializer.dataset_contract.ACTION_BRANCH,
        },
        "training_performed": False,
        "optimizer_authorized": False,
        "editor_optimizer_authorized": False,
        "scientific_critic_claim_authorized": False,
        "generated_media_editor_use_authorized": False,
        "receipt_digest": "3" * 64,
    }


def _mock_current_base_receipt() -> dict[str, object]:
    source = "/sealed/source.mp4"
    clean = "/sealed/base.normalized-clean-latent.safetensors"
    noise = "/sealed/base.official-initial-gaussian.safetensors"
    base_mp4 = "/sealed/base.mp4"
    source_sha = "1" * 64
    action_sha = "2" * 64
    clean_sha = "3" * 64
    noise_sha = "4" * 64
    base_sha = "5" * 64
    row = {
        "schema_version": (
            "bernini-identity-orbit-heldout-role-composition-receipt-v1"
        ),
        "cell_spec": {
            "cell": {
                "source_video": source,
                "source_video_sha256": source_sha,
                "action_caption_utf8_sha256": action_sha,
                "target_seed": 2026080907,
            }
        },
        "source": {"path": source, "sha256": source_sha},
        "model": {
            "bernini_commit": runtime.live_bridge.BERNINI_OFFICIAL_COMMIT,
            "veomni_commit": runtime.live_bridge.VEOMNI_TESTED_COMMIT,
            "checkpoint_tree_sha256": (
                runtime.live_bridge.BERNINI_CHECKPOINT_TREE_SHA256
            ),
            "checkpoint_unchanged": True,
            "checkpoint_content": {
                "manifest_sha256_computed": (
                    runtime.live_bridge.BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256
                ),
                "manifest_sha256_expected": (
                    runtime.live_bridge.BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256
                ),
                "every_file_sha256_verified": True,
            },
        },
        "sampling": {
            "exact81": True,
            "frame_count": 81,
            "fps": 25,
            "latent_phases": 21,
            "num_inference_steps": 40,
            "native_unipc_shift5": True,
            "source_rich_noise": False,
        },
        "outputs": {
            "base": {
                "path": base_mp4,
                "sha256": base_sha,
                "frame_count": 81,
                "fps": 25,
                "height": 32,
                "width": 48,
                "normalized_clean_latent": {
                    "path": clean,
                    "sha256": clean_sha,
                    "shape": [1, 16, 21, 4, 6],
                    "tensor_key": runtime.TENSOR_KEY_CLEAN,
                    "native_sampler_before_vae_decode": True,
                    "coordinate": "bernini_normalized_clean_vae_latent",
                    "stored_dtype": "torch.float32",
                    "roundtrip_byte_exact_fp32": True,
                },
            }
        },
        "initial_noise_artifacts": {
            "base": {
                "path": noise,
                "sha256": noise_sha,
                "shape": [1, 16, 21, 4, 6],
                "tensor_key": runtime.TENSOR_KEY_NOISE,
                "captured_from_native_sampler": True,
                "observer_only": True,
                "original_return_tensor_forwarded_by_identity": True,
                "roundtrip_raw_value_exact": True,
                "source_or_target_derived": False,
                "generator_initial_seed": 2026080907,
                "external_initial_noise_injection": False,
                "sampler_noise_replacement": False,
            }
        },
    }
    row["receipt_digest"] = runtime._producer_ascii_object_sha256(row)
    return row


@unittest.skipIf(runtime is None, "SIRM SP4 runtime has not been authored yet")
class SIRMRuntimeStaticContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = METHOD_ROOT / f"{RUNTIME_MODULE}.py"
        cls.source = cls.path.read_text(encoding="utf-8")

    def test_parser_has_no_learned_or_starc_critic_head_surface(self) -> None:
        destinations = _parser_destinations(runtime.build_parser())
        forbidden = {
            "critic_checkpoint",
            "expected_critic_checkpoint_sha256",
            "critic_checkpoint_receipt",
            "expected_critic_checkpoint_receipt_sha256",
            "critic_config_receipt",
            "expected_critic_config_receipt_sha256",
            "critic_head",
            "critic_head_checkpoint",
            "head_checkpoint",
            "learned_critic",
            "starc_checkpoint",
            "starc_critic",
        }
        self.assertFalse(destinations & forbidden)
        lowered = self.source.lower()
        for option in (
            "--critic-checkpoint",
            "--critic-config-receipt",
            "--critic-head",
            "--head-checkpoint",
            "--learned-critic",
            "--starc-critic",
        ):
            self.assertNotIn(option, lowered)

    def test_runtime_uses_parameter_free_relational_scorer_not_starc_head(self) -> None:
        self.assertIn("FrozenRelationalMotionScorer", self.source)
        for forbidden in (
            "FrozenHiddenTemporalEventCritic",
            "verify_frozen_starc_critic_artifact",
            "run_starc_core4_critic_pilot_v1",
            "critic.load_state_dict",
        ):
            self.assertNotIn(forbidden, self.source)

    def test_teacher_checkpoint_identity_maps_to_live_authenticated_content(self) -> None:
        for teacher_key, live_key in (
            ("manifest_sha256", "checkpoint_content_manifest_file_sha256"),
            ("verified_file_count", "checkpoint_content_verified_file_count"),
            ("verified_entries_digest", "checkpoint_content_verified_entries_digest"),
        ):
            self.assertIn(f'teacher_checkpoint.get("{teacher_key}")', self.source)
            self.assertIn(f'checkpoint_binding["{live_key}"]', self.source)

    def test_teacher_fixed_sketch_is_cross_bound_to_current_geometry(self) -> None:
        for token in (
            "live_bridge.geometry_spatial_sketch_binding(",
            'teacher_receipt.get("spatial_sketch_binding") != expected_spatial_sketch',
            'teacher_hidden.get("latent_shape")',
            "list(candidate.geometry.latent_shape)",
            'teacher_hidden.get("patch_positions")',
            "candidate.geometry.patch_positions",
            'teacher_hidden.get("patch_grid_height_width")',
            "[candidate.geometry.patch_rows, candidate.geometry.patch_columns]",
            '"teacher_current_geometry_and_sketch_exact_match": True',
        ):
            self.assertIn(token, self.source)
        self.assertIn(
            "authenticate_frozen_bernini_checkpoint_content", self.source
        )

    def test_positive_teacher_helper_accepts_only_bound_fit_positive(self) -> None:
        helper = _positive_teacher_helper()
        receipt = _mock_positive_teacher_receipt()
        materializer = getattr(runtime, "materializer", None)
        self.assertIsNotNone(materializer)
        with mock.patch.object(
            materializer,
            "validate_arm_receipt",
            side_effect=lambda value, **_kwargs: dict(value),
        ) as validator:
            checked = helper(receipt, expected_episode_id="dog-fit")
        validator.assert_called_once()
        if isinstance(checked, dict):
            self.assertEqual(checked.get("episode_id"), "dog-fit")
            self.assertEqual(checked.get("role"), "positive")
            self.assertEqual(checked.get("label"), 1)

    def test_positive_teacher_helper_rejects_negative_alias_and_excess_authority(self) -> None:
        helper = _positive_teacher_helper()
        materializer = getattr(runtime, "materializer", None)
        self.assertIsNotNone(materializer)
        mutations = {
            "negative role": {"role": "same_video_reverse", "label": 0},
            "wrong episode": {"episode_id": "human-fit"},
            "wrong split": {"split": "confirmation"},
            "training authority": {"training_performed": True},
            "editor authority": {"editor_optimizer_authorized": True},
            "claim authority": {"scientific_critic_claim_authorized": True},
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label):
                receipt = _mock_positive_teacher_receipt()
                receipt.update(mutation)
                with mock.patch.object(
                    materializer,
                    "validate_arm_receipt",
                    side_effect=lambda value, **_kwargs: dict(value),
                ):
                    with self.assertRaises(Exception):
                        helper(receipt, expected_episode_id="dog-fit")

    def test_dose_and_projector_are_fixed_not_cli_tunable(self) -> None:
        self.assertTrue(hasattr(runtime, "FIXED_DOSE_RMS"))
        self.assertTrue(
            math.isclose(
                float(runtime.FIXED_DOSE_RMS), 0.03, rel_tol=0.0, abs_tol=0.0
            )
        )
        destinations = _parser_destinations(runtime.build_parser())
        self.assertFalse(
            destinations
            & {
                "dose",
                "dose_rms",
                "intervention_dose",
                "projector",
                "projection_mode",
                "projected_max_rms",
            }
        )
        self.assertIn("symmetric_latent_interventions", self.source)
        self.assertIn("FIXED_DOSE_RMS", self.source)

    def test_runtime_static_closure_has_no_optimizer_or_auxiliary_projector(self) -> None:
        destinations = _parser_destinations(runtime.build_parser())
        forbidden_destinations = {
            "optimizer",
            "learning_rate",
            "adapter",
            "lora",
            "mask",
            "track",
            "pose",
            "flow",
            "detector_box",
            "swept_tube",
        }
        self.assertFalse(destinations & forbidden_destinations)
        for token in (
            "optimizer.step(",
            "loss.backward(",
            "load_adapter(",
            "project_source_safe_cotangent =",
            "temporal_scorer._frozen_d541801_runtime()",
        ):
            self.assertNotIn(token, self.source)
        for token in (
            "_frozen_d541801_runtime_facade()",
            "_FROZEN_D541801_RUNTIME_DEPENDENCIES",
            '"infer_lora.py"',
            '"infer_source_kv_carrier_oracle.py"',
            '"source_self_native_ref_contrastive_v3.py"',
        ):
            self.assertIn(token, self.source)

    def test_current_base_receipt_closes_one_native_rv2v_rollout(self) -> None:
        row = _mock_current_base_receipt()
        checked = runtime.validate_current_base_provenance_receipt(
            row,
            source_path=Path("/sealed/source.mp4"),
            source_sha256="1" * 64,
            action_caption_sha256="2" * 64,
            clean_path=Path("/sealed/base.normalized-clean-latent.safetensors"),
            clean_file_sha256="3" * 64,
            noise_path=Path("/sealed/base.official-initial-gaussian.safetensors"),
            noise_file_sha256="4" * 64,
            base_mp4_path=Path("/sealed/base.mp4"),
            base_mp4_sha256="5" * 64,
            latent_shape=[1, 16, 21, 4, 6],
        )
        self.assertTrue(checked["same_native_rv2v_base_rollout"])

        tampered = _mock_current_base_receipt()
        tampered["initial_noise_artifacts"]["base"][
            "external_initial_noise_injection"
        ] = True
        tampered["receipt_digest"] = runtime._producer_ascii_object_sha256(
            {key: value for key, value in tampered.items() if key != "receipt_digest"}
        )
        with self.assertRaises(Exception):
            runtime.validate_current_base_provenance_receipt(
                tampered,
                source_path=Path("/sealed/source.mp4"),
                source_sha256="1" * 64,
                action_caption_sha256="2" * 64,
                clean_path=Path("/sealed/base.normalized-clean-latent.safetensors"),
                clean_file_sha256="3" * 64,
                noise_path=Path("/sealed/base.official-initial-gaussian.safetensors"),
                noise_file_sha256="4" * 64,
                base_mp4_path=Path("/sealed/base.mp4"),
                base_mp4_sha256="5" * 64,
                latent_shape=[1, 16, 21, 4, 6],
            )

    def test_base_receipt_and_delta_artifact_are_explicit(self) -> None:
        destinations = _parser_destinations(runtime.build_parser())
        self.assertIn("base_receipt", destinations)
        self.assertIn("expected_base_receipt_sha256", destinations)
        for token in (
            'staging / "fixed-dose-delta.safetensors"',
            '"fixed_dose_delta_artifact"',
            '"fixed_dose_delta_tensor_sha256"',
            '"current_base_provenance"',
        ):
            self.assertIn(token, self.source)

    @unittest.skipIf(torch is None, "Torch is required for prompt hashing")
    def test_frozen_facade_supplies_prompt_pair_tensor_hash(self) -> None:
        diffusers = ModuleType("diffusers")
        pipelines = ModuleType("diffusers.pipelines")
        wan = ModuleType("diffusers.pipelines.wan")
        pipeline_wan = ModuleType("diffusers.pipelines.wan.pipeline_wan")
        pipeline_wan.prompt_clean = lambda value: value
        modules = {
            "diffusers": diffusers,
            "diffusers.pipelines": pipelines,
            "diffusers.pipelines.wan": wan,
            "diffusers.pipelines.wan.pipeline_wan": pipeline_wan,
        }
        with mock.patch.dict(sys.modules, modules):
            frozen = runtime._frozen_d541801_runtime_facade()

        class Renderer:
            @staticmethod
            def encode_prompt(ids, _mask):
                return ids.unsqueeze(-1).expand(1, 512, 4096).float()

        calls = iter((1, 2))

        def tokenize(_tokenizer, _prompt):
            value = next(calls)
            return (
                torch.full((1, 512), value, dtype=torch.long),
                torch.ones((1, 512), dtype=torch.long),
            )

        with mock.patch.object(
            frozen.native_generation.legacy,
            "_tokenize_training_prompt",
            side_effect=tokenize,
        ):
            conditions, hashes = runtime.temporal_scorer._encode_prompt_pair(
                Renderer(),
                object(),
                action_prompt="action",
                noop_prompt="noop",
                device=torch.device("cpu"),
                frozen=frozen,
            )
        self.assertEqual(set(conditions), {"target_action", "noop"})
        self.assertEqual(set(hashes), {"target_action", "noop"})
        self.assertNotEqual(hashes["target_action"], hashes["noop"])


@unittest.skipIf(
    runtime is None or torch is None or relational is None,
    "SIRM runtime or Torch tensor core is unavailable",
)
class SIRMFixedDoseTensorClosureTests(unittest.TestCase):
    def test_fixed_point03_pair_closes_projector_and_symmetric_dose(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(20260809031)
        clean = torch.randn(
            1, 16, 21, 4, 6, generator=generator, dtype=torch.float32
        )
        raw = torch.randn(
            1, 16, 21, 4, 6, generator=generator, dtype=torch.float32
        )
        pair = relational.symmetric_latent_interventions(
            clean,
            raw,
            dose_rms=float(runtime.FIXED_DOSE_RMS),
        )
        self.assertTrue(
            torch.equal(
                pair.projected_cotangent[:, :, 0],
                torch.zeros_like(pair.projected_cotangent[:, :, 0]),
            )
        )
        self.assertLess(
            float(pair.projected_cotangent.to(torch.float64).sum(dim=2).abs().max()),
            3.0e-6,
        )
        height, width = pair.projected_cotangent.shape[-2:]
        x = torch.linspace(-1.0, 1.0, width, dtype=torch.float64)
        y = torch.linspace(-1.0, 1.0, height, dtype=torch.float64)
        basis = torch.stack(
            (
                torch.ones(height, width, dtype=torch.float64),
                x.unsqueeze(0).expand(height, width),
                y.unsqueeze(1).expand(height, width),
            )
        ).reshape(3, height * width)
        basis = basis / torch.linalg.vector_norm(basis, dim=1, keepdim=True)
        affine_dot = (
            pair.projected_cotangent.to(torch.float64).reshape(-1, height * width)
            @ basis.transpose(0, 1)
        )
        self.assertLess(float(affine_dot.abs().max()), 3.0e-6)
        self.assertTrue(
            torch.allclose(pair.plus - clean, pair.delta, rtol=0.0, atol=2.0e-7)
        )
        self.assertTrue(
            torch.allclose(clean - pair.minus, pair.delta, rtol=0.0, atol=2.0e-7)
        )
        observed = float(pair.delta.to(torch.float64).square().mean().sqrt())
        self.assertEqual(pair.dose_rms, 0.03)
        self.assertTrue(math.isclose(observed, 0.03, rel_tol=1.0e-6, abs_tol=1.0e-8))


if __name__ == "__main__":
    unittest.main()
