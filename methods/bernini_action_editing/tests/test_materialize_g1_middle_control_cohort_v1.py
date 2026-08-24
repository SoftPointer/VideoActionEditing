#!/usr/bin/env python3

"""Tests for the G1 projected-middle control cohort."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]
SUBJECT_PATH = (
    REPO_ROOT
    / "methods"
    / "bernini_action_editing"
    / "materialize_g1_middle_control_cohort_v1.py"
)
EXTRACTOR_PATH = (
    REPO_ROOT
    / "methods"
    / "bernini_action_editing"
    / "materialize_decoded_middle_action_repr_v1.py"
)


def _load_subject():
    spec = importlib.util.spec_from_file_location(
        "materialize_g1_middle_control_cohort_v1_test", SUBJECT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_extractor():
    spec = importlib.util.spec_from_file_location(
        "materialize_decoded_middle_action_repr_v1_contract_test", EXTRACTOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


subject = _load_subject()

try:
    import torch
    from safetensors.torch import load_file, save_file
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    load_file = None  # type: ignore[assignment]
    save_file = None  # type: ignore[assignment]


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _explicit_gaussian_match(name: str) -> dict[str, object]:
    canonical_sha = _digest(f"canonical-gaussian-{name}")
    raw_sigma_sha = _digest(f"raw-noise-sigma-{name}")
    authority = {
        "authority_kind": subject.EXPLICIT_GAUSSIAN_AUTHORITY_KIND,
        "domain": subject.EXPLICIT_GAUSSIAN_DOMAIN,
        "producer_rank": 0,
        "base_seed": 2026082401,
        "derived_seed": 2026082402,
        "dtype": "torch.float32",
        "shape": [63, 16, 1, 2, 2],
        "canonical_gaussian_sha256": canonical_sha,
        "broadcast_transport": "torch_distributed_nccl_fp32_tensor_broadcast",
        "world_size": 4,
        "world4_raw_sha256_consensus": True,
        "action_injection_count": 1,
        "noop_injection_count": 1,
        "action_gaussian_sha256": canonical_sha,
        "noop_gaussian_sha256": canonical_sha,
        "raw_noise_sigma_dtype": "torch.bfloat16",
        "raw_noise_sigma_shape": [1],
        "action_raw_noise_sigma_sha256": raw_sigma_sha,
        "noop_raw_noise_sigma_sha256": raw_sigma_sha,
        "clean_capture_stage": "inside_cloned_pack_before_fm_interpolation",
        "packed_state_original_op_order_bit_exact": True,
        "target_velocity_bit_exact": True,
        "recovered_from_x_or_velocity": False,
        "vendor_data_file_sha256": subject.PINNED_BERNINI_DATA_SHA256,
        "pack_vae_latents_source_sha256": (
            subject.PINNED_PACK_VAE_LATENTS_SOURCE_SHA256
        ),
        "process_renderer_sample_source_sha256": (
            subject.PINNED_PROCESS_RENDERER_SAMPLE_SOURCE_SHA256
        ),
        "vendor_module_mutated": False,
        "original_function_globals_mutated": False,
        "trainer_received_authority": False,
    }
    return {
        "comparison_stage": "before_fm_interpolation",
        "criterion": subject.EXPLICIT_GAUSSIAN_MATCH_CRITERION,
        "inverse_recovery_numerical_fields_applicable": False,
        "canonical_gaussian_sha256": canonical_sha,
        "both_branches_retimed_from_canonical_gaussian": True,
        "fixed_absolute_tolerance_is_authority": False,
        "authority": authority,
    }


def _deterministic_vae_authority(name: str) -> dict[str, object]:
    phase0_sha = _digest(f"posterior-phase0-{name}")
    before = {
        "deterministic_algorithms_enabled": False,
        "deterministic_algorithms_warn_only": False,
        "cudnn_deterministic": False,
        "cudnn_benchmark": False,
    }
    return {
        "authority_kind": subject.DETERMINISTIC_VAE_AUTHORITY_KIND,
        "policy": subject.DETERMINISTIC_VAE_POLICY,
        "producer_rank": 0,
        "encode_call_count": 2,
        "scope": subject.DETERMINISTIC_VAE_SCOPE,
        "before_flags": dict(before),
        "during_flags": {
            "deterministic_algorithms_enabled": True,
            "deterministic_algorithms_warn_only": False,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
        },
        "restored_flags": dict(before),
        "flags_restored_exact": True,
        "posterior_phase0_max_abs_error": 0.0,
        "posterior_phase0_bit_exact": True,
        "action_phase0_posterior_sha256": phase0_sha,
        "noop_phase0_posterior_sha256": phase0_sha,
        "posterior_modified_after_encode": False,
        "posterior_copy_or_splice_used": False,
        "trainer_received_posterior": False,
    }


@unittest.skipUnless(
    torch is not None and load_file is not None and save_file is not None,
    "PyTorch/safetensors are unavailable",
)
class G1MiddleControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.correct = self._middle("correct", case_id="case-a", role="real_forward", instruction="turn", factor=1.0)
        self.shuffle = self._middle("shuffle", case_id="case-a", role="temporal_shuffle", instruction="turn", factor=0.8, phase_shift=3)
        self.reverse = self._middle("reverse", case_id="case-a", role="reverse", instruction="turn", factor=-0.7, reverse=True)
        self.wrong = self._middle("wrong", case_id="case-b", role="real_forward", instruction="lift", factor=2.5, phase_shift=5)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _middle(
        self,
        name: str,
        *,
        case_id: str,
        role: str,
        instruction: str,
        factor: float,
        phase_shift: int = 0,
        reverse: bool = False,
        projection_seed: int = 2026082401,
    ) -> tuple[Path, Path]:
        assert torch is not None and save_file is not None
        cache_root = self.root / name
        cache_root.mkdir()
        cache = cache_root / "middle_repr.safetensors"
        tensors = {}
        for block in subject.BLOCK_INDICES:
            value = torch.arange(2 * 21 * 3 * 5, dtype=torch.float32).reshape(2, 21, 3, 5)
            value = value.mul(factor).add(float(block))
            value[:, 0] = 0
            if phase_shift:
                tail = torch.roll(value[:, 1:], shifts=phase_shift, dims=1)
                value = torch.cat((value[:, :1], tail), dim=1)
            if reverse:
                value = torch.cat((value[:, :1], torch.flip(value[:, 1:], dims=(1,))), dim=1)
            tensors[f"middle_block_{block:02d}"] = value.contiguous()
        metadata = {
            "schema_version": subject.UPSTREAM_CACHE_SCHEMA,
            "method": "bernini-decoded-middle-action-representation-v1",
            "representation_origin": "decoded_video_reencode",
            "anchor_source_role": role,
            "blocks": ",".join(map(str, subject.BLOCK_INDICES)),
            "sigmas": "0.85,0.55",
            "projection_width": "5",
            "contains_detached_projected_residuals_only": "true",
            "contains_rgb_latent_absolute_hidden_qkv_or_endpoint": "false",
        }
        save_file(tensors, str(cache), metadata=metadata)
        cache_sha = subject._sha256_file(cache)
        tensor_rows = {
            key: {
                "shape": list(map(int, value.shape)),
                "dtype": str(value.dtype),
                "sha256": subject._tensor_sha256(value),
                "detached": True,
                "phase0_hard_zero": True,
            }
            for key, value in tensors.items()
        }
        receipt = {
            "schema_version": subject.UPSTREAM_RECEIPT_SCHEMA,
            "method": "bernini-decoded-middle-action-representation-v1",
            "complete": True,
            "scientific_claim_authorized": False,
            "case_id": case_id,
            "representation_origin": "decoded_video_reencode",
            "anchor_source_role": role,
            "input_video_sha256": _digest(f"video-{name}"),
            "instruction_sha256": _digest(instruction),
            "cache": {
                "filename": cache.name,
                "sha256": cache_sha,
                "schema_version": subject.UPSTREAM_CACHE_SCHEMA,
                "tensor_key_allowlist": sorted(tensors),
                "tensors": tensor_rows,
            },
            "representation": {
                "blocks": list(subject.BLOCK_INDICES),
                "capture": "post_transformer_block_output",
                "contrast": "decoded_action_minus_exact_first_frame_repeat",
                "decoded_video_reencode": True,
                "selfgen_native_trajectory": False,
                "noop_constructed_inside_extractor": True,
                "first_frame_repeat_rgb_exact": True,
                "same_caption": True,
                "same_gaussian": True,
                "same_timestep": True,
                "same_rotary": True,
                "sigmas": [0.85, 0.55],
                "noise_max_abs_error": 0.0,
                "gaussian_match": _explicit_gaussian_match(case_id),
                "deterministic_vae_authority": (
                    _deterministic_vae_authority(case_id)
                ),
                "phase0_clean_max_abs_error": 0.0,
                "phase0_match_atol": 0.0,
                "patch_grid": [21, 1, 3],
                "projection": {
                    "kind": "case_independent_fixed_rademacher_jl",
                    "width": 5,
                    "seed": projection_seed,
                    "sha256": _digest(f"projection-{projection_seed}"),
                    "fitted_on_input_video": False,
                },
            },
            "information_firewall": {
                "input_video_accessed_by_frozen_extractor": True,
                "target_video_accessed_by_extractor": role not in subject.UPSTREAM_SELFGEN_ROLES,
                "target_rgb_or_vae_used_by_frozen_extractor": role not in subject.UPSTREAM_SELFGEN_ROLES,
                "trainer_receives_detached_representation_cache_only": True,
                "target_video_accessed_by_trainer": False,
                "target_rgb_or_vae_target_used_by_trainer": False,
                "anchor_role": "detached_action_representation_only",
                "input_video_path_persisted": False,
                "input_rgb_frames_persisted": False,
                "input_vae_or_clean_latent_persisted": False,
                "absolute_action_hidden_persisted": False,
                "absolute_noop_hidden_persisted": False,
                "raw_q_or_k_or_value_persisted": False,
                "model_endpoint_or_velocity_persisted": False,
                "self_generated_rgb_or_latent_copied_to_output": False,
                "ephemeral_posterior_broadcast_inside_frozen_extractor_only": True,
                "broadcast_posterior_payload_persisted": False,
                "ephemeral_absolute_hidden_zero_reference_released_before_publication": True,
            },
            "training_authority": {
                "optimizer_created": False,
                "optimization_steps": 0,
                "generator_parameters_updated": False,
                "cache_is_not_a_flow_matching_target": True,
            },
            "model_identity": {"checkpoint_tree_sha256": _digest("checkpoint"), "base_frozen": True},
            "runtime_identity": {"world_size": 4},
            "method_source_sha256": _digest("middle-extractor-source"),
        }
        receipt["receipt_digest"] = subject._sha256_bytes(subject._canonical_json_bytes(receipt))
        receipt_path = cache_root / "receipt.json"
        receipt_path.write_bytes(subject._canonical_json_bytes(receipt, pretty=True))
        return cache, receipt_path

    def _materialize(self, output: str = "cohort", anchor_kind: str = "target"):
        return subject.materialize_cohort(
            correct_cache=self.correct[0],
            correct_receipt=self.correct[1],
            temporal_shuffle_cache=self.shuffle[0],
            temporal_shuffle_receipt=self.shuffle[1],
            reverse_cache=self.reverse[0],
            reverse_receipt=self.reverse[1],
            wrong_action_cache=self.wrong[0],
            wrong_action_receipt=self.wrong[1],
            output_dir=self.root / output,
            case_id="case-a",
            anchor_kind=anchor_kind,
            action_family="head_turn",
            wrong_case_id="case-b",
            wrong_action_family="barbell_lift",
            incomplete_action_phases=8,
        )

    def test_middle_zero_incomplete_and_wrong_energy_match_replay(self) -> None:
        assert torch is not None and load_file is not None
        receipt = self._materialize()
        root = self.root / "cohort"
        self.assertEqual(receipt["contracts"]["abi_component"], "delta_h_middle")
        self.assertTrue(receipt["contracts"]["weighted_compensation_forbidden"])
        zero = load_file(str(root / "zero_or_noop.safetensors"), device="cpu")
        self.assertTrue(all(int(torch.count_nonzero(value).item()) == 0 for value in zero.values()))
        original = load_file(str(self.correct[0]), device="cpu")
        incomplete = load_file(str(root / "incomplete.safetensors"), device="cpu")
        for key in subject.REQUIRED_TENSORS:
            self.assertTrue(torch.equal(incomplete[key][:, :9], original[key][:, :9]))
            self.assertEqual(int(torch.count_nonzero(incomplete[key][:, 9:]).item()), 0)
        wrong = load_file(str(root / "wrong_action_energy_matched.safetensors"), device="cpu")
        self.assertLessEqual(
            abs(subject._energy(wrong) - subject._energy(original)) / subject._energy(original),
            subject.ENERGY_MATCH_RTOL,
        )
        self.assertEqual(subject.verify_cohort_receipt(root / "cohort_receipt.json"), receipt)

    def test_fp16_quantized_energy_boundary_is_calibrated_in_output_dtype(
        self,
    ) -> None:
        """Replay the observed ~2.454/2.457 RMS FP16 boundary failure."""

        assert torch is not None
        generator = torch.Generator(device="cpu").manual_seed(20260824)
        base = torch.randn(
            len(subject.REQUIRED_TENSORS), 2, 21, 3, 5,
            generator=generator,
            dtype=torch.float64,
        )
        base[:, :, 0].zero_()
        donor_values = (base * 2.51434).to(dtype=torch.float16)
        correct_values = (
            donor_values.double() * 0.999026
        ).to(dtype=torch.float16)

        def loaded(name: str, values):
            tensors = {
                key: values[index].contiguous()
                for index, key in enumerate(subject.REQUIRED_TENSORS)
            }
            return subject.LoadedMiddle(
                path=self.root / f"{name}.safetensors",
                sha256=_digest(f"{name}-cache"),
                receipt_path=self.root / f"{name}.json",
                receipt_sha256=_digest(f"{name}-receipt"),
                receipt={},
                metadata={},
                tensors=tensors,
            )

        correct = loaded("fp16-correct", correct_values)
        donor = loaded("fp16-donor", donor_values)
        correct_energy = subject._energy(correct.tensors)
        donor_energy = subject._energy(donor.tensors)
        analytic_scale = correct_energy / donor_energy
        naive = {
            key: (donor.tensors[key] * analytic_scale).contiguous()
            for key in subject.REQUIRED_TENSORS
        }
        naive_relative_error = (
            abs(subject._energy(naive) - correct_energy) / correct_energy
        )
        self.assertAlmostEqual(correct_energy, 2.4542673840, delta=5.0e-5)
        self.assertAlmostEqual(donor_energy, 2.4565841030, delta=5.0e-5)
        self.assertAlmostEqual(analytic_scale, 0.9990569348, delta=2.0e-6)
        self.assertGreater(naive_relative_error, subject.ENERGY_MATCH_RTOL)

        generated, diagnostics = subject._build_transforms(
            correct,
            donor,
            incomplete_action_phases=8,
        )
        matched = generated["wrong_action_energy_matched"]
        self.assertTrue(
            all(
                value.dtype == torch.float16
                for value in matched.values()
            )
        )
        self.assertGreater(
            diagnostics["wrong_action_initial_relative_energy_error"],
            subject.ENERGY_MATCH_RTOL,
        )
        self.assertLessEqual(
            diagnostics["wrong_action_relative_energy_error"],
            subject.ENERGY_MATCH_RTOL,
        )
        self.assertNotEqual(
            diagnostics["wrong_action_initial_scale"],
            diagnostics["wrong_action_final_scale"],
        )
        self.assertEqual(
            diagnostics["wrong_action_scale"],
            diagnostics["wrong_action_final_scale"],
        )
        self.assertTrue(diagnostics["wrong_action_quantization_calibrated"])
        self.assertEqual(
            diagnostics["wrong_action_scale_calibration_iterations"], 32
        )
        self.assertEqual(
            diagnostics["wrong_action_scale_calibration_compute_dtype"],
            "torch.float64",
        )
        self.assertEqual(
            diagnostics["wrong_action_output_dtype"], "torch.float16"
        )

    def test_real_extractor_receipt_shape_and_target_role_map_round_trip(self) -> None:
        assert torch is not None
        extractor = _load_extractor()
        inputs: dict[str, tuple[Path, Path]] = {}
        roles = subject.TARGET_EXTERNAL_ROLE_MAP
        cases = {
            "correct": "case-real-a",
            "temporal_shuffle": "case-real-a",
            "reverse": "case-real-a",
            "wrong_action_donor": "case-real-b",
        }
        instructions = {
            "correct": "turn",
            "temporal_shuffle": "turn",
            "reverse": "turn",
            "wrong_action_donor": "lift",
        }
        for offset, slot in enumerate(subject.EXTERNAL_ROLES, start=1):
            cache_root = self.root / f"real-{slot}"
            cache_root.mkdir()
            cache = cache_root / "middle_repr.safetensors"
            receipt_path = cache_root / "receipt.json"
            tensors = {}
            for block in subject.BLOCK_INDICES:
                value = torch.arange(
                    2 * 21 * 3 * 5, dtype=torch.float32
                ).reshape(2, 21, 3, 5)
                value = value.mul(float(offset)).add(float(block + offset))
                value[:, 0].zero_()
                tensors[f"middle_block_{block:02d}"] = value.contiguous()
            tensor_rows = extractor.validate_cache_tensors(
                tensors, sigma_count=2, projection_width=5
            )
            extractor._atomic_safetensors(
                cache,
                tensors,
                metadata={
                    "schema_version": extractor.CACHE_SCHEMA,
                    "method": extractor.METHOD,
                    "representation_origin": "decoded_video_reencode",
                    "anchor_source_role": roles[slot],
                    "blocks": ",".join(map(str, extractor.BLOCK_INDICES)),
                    "sigmas": "0.85,0.55",
                    "projection_width": "5",
                    "contains_detached_projected_residuals_only": "true",
                    "contains_rgb_latent_absolute_hidden_qkv_or_endpoint": "false",
                },
            )
            gaussian_match = _explicit_gaussian_match(cases[slot])
            receipt = extractor.build_receipt(
                case_id=cases[slot],
                input_role=roles[slot],
                input_video_sha256=_digest(f"real-video-{slot}"),
                instruction_sha256=_digest(instructions[slot]),
                cache_path=cache,
                cache_sha256=extractor.file_sha256(cache),
                cache_tensors=tensor_rows,
                sigmas=(0.85, 0.55),
                projection_width=5,
                projection_seed=2026082401,
                projection_sha256=_digest("real-projection"),
                patch_grid=(21, 1, 3),
                noise_max_abs_error=0.0,
                noise_max_abs_forward_error_bound=0.0,
                noise_max_error_to_bound_ratio=0.0,
                noise_original_dtype="torch.float32",
                noise_dtype_epsilon=float(torch.finfo(torch.float32).eps),
                canonical_gaussian_sha256=str(
                    gaussian_match["canonical_gaussian_sha256"]
                ),
                gaussian_authority=gaussian_match["authority"],
                deterministic_vae_authority=(
                    _deterministic_vae_authority(cases[slot])
                ),
                phase0_clean_max_abs_error=0.0,
                block_metrics={},
                model_identity={"base_frozen": True},
                runtime_identity={"world_size": 4},
                method_source_sha256=_digest("real-extractor-source"),
            )
            extractor.validate_receipt(receipt)
            extractor._atomic_json(receipt_path, receipt)
            inputs[slot] = (cache, receipt_path)
        receipt = subject.materialize_cohort(
            correct_cache=inputs["correct"][0],
            correct_receipt=inputs["correct"][1],
            temporal_shuffle_cache=inputs["temporal_shuffle"][0],
            temporal_shuffle_receipt=inputs["temporal_shuffle"][1],
            reverse_cache=inputs["reverse"][0],
            reverse_receipt=inputs["reverse"][1],
            wrong_action_cache=inputs["wrong_action_donor"][0],
            wrong_action_receipt=inputs["wrong_action_donor"][1],
            output_dir=self.root / "real-extractor-cohort",
            case_id="case-real-a",
            anchor_kind="target",
            action_family="head_turn",
            wrong_case_id="case-real-b",
            wrong_action_family="barbell_lift",
            incomplete_action_phases=8,
        )
        self.assertEqual(
            receipt["contracts"]["external_role_contract"],
            subject.TARGET_EXTERNAL_ROLE_MAP,
        )
        self.assertEqual(receipt["correct_role"], "real_forward")
        self.assertEqual(
            subject.verify_cohort_receipt(
                self.root / "real-extractor-cohort" / "cohort_receipt.json"
            ),
            receipt,
        )

    def test_selfgen_role_is_separate(self) -> None:
        self.correct = self._middle("sg-correct", case_id="case-a", role="self_generated", instruction="turn", factor=1.0)
        self.shuffle = self._middle("sg-shuffle", case_id="case-a", role="self_generated_temporal_shuffle", instruction="turn", factor=0.8, phase_shift=3)
        self.reverse = self._middle("sg-reverse", case_id="case-a", role="self_generated_reverse", instruction="turn", factor=-0.7, reverse=True)
        self.wrong = self._middle("sg-wrong", case_id="case-b", role="self_generated", instruction="lift", factor=2.5, phase_shift=5)
        receipt = self._materialize("selfgen-cohort", "selfgen")
        self.assertEqual(receipt["correct_role"], "self_generated")
        self.assertTrue(receipt["contracts"]["target_and_selfgen_judged_separately"])
        self.assertEqual(
            receipt["contracts"]["external_role_contract"],
            subject.SELFGEN_EXTERNAL_ROLE_MAP,
        )

    def test_ambiguous_selfgen_control_role_fails_closed(self) -> None:
        self.correct = self._middle("sg2-correct", case_id="case-a", role="self_generated", instruction="turn", factor=1.0)
        self.shuffle = self._middle("sg2-shuffle", case_id="case-a", role="self_generated", instruction="turn", factor=0.8, phase_shift=3)
        self.reverse = self._middle("sg2-reverse", case_id="case-a", role="self_generated_reverse", instruction="turn", factor=-0.7, reverse=True)
        self.wrong = self._middle("sg2-wrong", case_id="case-b", role="self_generated", instruction="lift", factor=2.5, phase_shift=5)
        with self.assertRaisesRegex(subject.G1MiddleControlError, "provenance"):
            self._materialize("ambiguous-selfgen", "selfgen")
        self.assertFalse((self.root / "ambiguous-selfgen").exists())

    def test_target_temporal_shuffle_requires_exact_extractor_role(self) -> None:
        self.shuffle = self._middle(
            "bad-target-shuffle",
            case_id="case-a",
            role="real_forward",
            instruction="turn",
            factor=0.8,
            phase_shift=3,
        )
        with self.assertRaisesRegex(subject.G1MiddleControlError, "provenance"):
            self._materialize("bad-target-role")
        self.assertFalse((self.root / "bad-target-role").exists())

    def test_missing_legacy_or_weakened_gaussian_authority_fails_closed(self) -> None:
        def remove_match(receipt: dict[str, object]) -> None:
            representation = receipt["representation"]
            assert isinstance(representation, dict)
            representation.pop("gaussian_match")

        def mutate_authority(
            receipt: dict[str, object], key: str, value: object
        ) -> None:
            representation = receipt["representation"]
            assert isinstance(representation, dict)
            gaussian_match = representation["gaussian_match"]
            assert isinstance(gaussian_match, dict)
            authority = gaussian_match["authority"]
            assert isinstance(authority, dict)
            authority[key] = value

        mutations = {
            "missing_match": remove_match,
            "wrong_comparison_stage": lambda value: value["representation"][
                "gaussian_match"
            ].__setitem__("comparison_stage", "after_fm_interpolation"),
            "wrong_criterion": lambda value: value["representation"][
                "gaussian_match"
            ].__setitem__("criterion", "inverse_recovery_with_tolerance"),
            "legacy_recovered": lambda value: (
                mutate_authority(
                    value,
                    "authority_kind",
                    "legacy_bit_identical_source_self_pair_inverse",
                ),
                mutate_authority(value, "recovered_from_x_or_velocity", True),
            ),
            "duplicate_action_injection": lambda value: mutate_authority(
                value, "action_injection_count", 2
            ),
            "branch_sha_mismatch": lambda value: mutate_authority(
                value, "noop_gaussian_sha256", _digest("different-noop-gaussian")
            ),
            "no_world4_consensus": lambda value: mutate_authority(
                value, "world4_raw_sha256_consensus", False
            ),
            "vendor_hash_mismatch": lambda value: mutate_authority(
                value, "vendor_data_file_sha256", _digest("other-vendor-data")
            ),
            "vendor_module_mutated": lambda value: mutate_authority(
                value, "vendor_module_mutated", True
            ),
            "function_globals_mutated": lambda value: mutate_authority(
                value, "original_function_globals_mutated", True
            ),
            "trainer_received_authority": lambda value: mutate_authority(
                value, "trainer_received_authority", True
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                receipt = json.loads(self.correct[1].read_text(encoding="ascii"))
                mutate(receipt)
                receipt.pop("receipt_digest")
                receipt["receipt_digest"] = subject._sha256_bytes(
                    subject._canonical_json_bytes(receipt)
                )
                with self.assertRaisesRegex(
                    subject.G1MiddleControlError, "Gaussian"
                ):
                    subject._validate_upstream_receipt(receipt, label=name)

    def test_deterministic_vae_authority_tamper_and_legacy_fail_closed(
        self,
    ) -> None:
        def authority(receipt: dict[str, object]) -> dict[str, object]:
            representation = receipt["representation"]
            assert isinstance(representation, dict)
            value = representation["deterministic_vae_authority"]
            assert isinstance(value, dict)
            return value

        def representation(receipt: dict[str, object]) -> dict[str, object]:
            value = receipt["representation"]
            assert isinstance(value, dict)
            return value

        mutations = {
            "legacy_missing_authority": lambda value: representation(value).pop(
                "deterministic_vae_authority"
            ),
            "legacy_authority_kind": lambda value: authority(value).__setitem__(
                "authority_kind", "torch_save_posterior_identity"
            ),
            "missing_nested_key": lambda value: authority(value).pop("policy"),
            "unexpected_nested_key": lambda value: authority(value).__setitem__(
                "repair_method", "copy_phase0"
            ),
            "wrong_policy": lambda value: authority(value).__setitem__(
                "policy", "best_effort_determinism"
            ),
            "not_rank0": lambda value: authority(value).__setitem__(
                "producer_rank", 1
            ),
            "single_encode": lambda value: authority(value).__setitem__(
                "encode_call_count", 1
            ),
            "wrong_scope": lambda value: authority(value).__setitem__(
                "scope", "global_process_scope"
            ),
            "non_boolean_before_flag": lambda value: authority(value)[
                "before_flags"
            ].__setitem__("cudnn_benchmark", 0),
            "deterministic_algorithms_disabled_during": lambda value: authority(
                value
            )["during_flags"].__setitem__(
                "deterministic_algorithms_enabled", False
            ),
            "warn_only_during": lambda value: authority(value)[
                "during_flags"
            ].__setitem__("deterministic_algorithms_warn_only", True),
            "cudnn_not_deterministic_during": lambda value: authority(value)[
                "during_flags"
            ].__setitem__("cudnn_deterministic", False),
            "cudnn_benchmark_during": lambda value: authority(value)[
                "during_flags"
            ].__setitem__("cudnn_benchmark", True),
            "restored_flags_differ": lambda value: authority(value)[
                "restored_flags"
            ].__setitem__("cudnn_benchmark", True),
            "restoration_not_exact": lambda value: authority(value).__setitem__(
                "flags_restored_exact", False
            ),
            "posterior_phase0_nonzero": lambda value: authority(value).__setitem__(
                "posterior_phase0_max_abs_error", 1.0e-7
            ),
            "posterior_phase0_not_bit_exact": lambda value: authority(
                value
            ).__setitem__("posterior_phase0_bit_exact", False),
            "posterior_phase0_sha_differs": lambda value: authority(
                value
            ).__setitem__(
                "noop_phase0_posterior_sha256", _digest("different-phase0")
            ),
            "posterior_modified": lambda value: authority(value).__setitem__(
                "posterior_modified_after_encode", True
            ),
            "posterior_copy_or_splice": lambda value: authority(
                value
            ).__setitem__("posterior_copy_or_splice_used", True),
            "trainer_received_posterior": lambda value: authority(
                value
            ).__setitem__("trainer_received_posterior", True),
            "nonzero_clean_phase0": lambda value: representation(
                value
            ).__setitem__("phase0_clean_max_abs_error", 1.0e-7),
            "nonzero_phase0_atol": lambda value: representation(
                value
            ).__setitem__("phase0_match_atol", 2.0e-5),
            "boolean_clean_phase0": lambda value: representation(
                value
            ).__setitem__("phase0_clean_max_abs_error", False),
            "boolean_phase0_atol": lambda value: representation(
                value
            ).__setitem__("phase0_match_atol", False),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                receipt = json.loads(self.correct[1].read_text(encoding="ascii"))
                mutate(receipt)
                receipt.pop("receipt_digest")
                receipt["receipt_digest"] = subject._sha256_bytes(
                    subject._canonical_json_bytes(receipt)
                )
                with self.assertRaisesRegex(
                    subject.G1MiddleControlError, "deterministic VAE"
                ):
                    subject._validate_upstream_receipt(receipt, label=name)

    def test_same_case_temporal_controls_require_one_gaussian_authority(
        self,
    ) -> None:
        receipt = json.loads(self.shuffle[1].read_text(encoding="ascii"))
        authority = receipt["representation"]["gaussian_match"]["authority"]
        authority["derived_seed"] += 1
        receipt.pop("receipt_digest")
        receipt["receipt_digest"] = subject._sha256_bytes(
            subject._canonical_json_bytes(receipt)
        )
        self.shuffle[1].write_bytes(
            subject._canonical_json_bytes(receipt, pretty=True)
        )
        with self.assertRaisesRegex(
            subject.G1MiddleControlError, "Gaussian authority"
        ):
            self._materialize("mismatched-gaussian-authority")
        self.assertFalse((self.root / "mismatched-gaussian-authority").exists())

    def test_projection_mismatch_fails_before_publication(self) -> None:
        mismatched = self._middle(
            "mismatched",
            case_id="case-b",
            role="real_forward",
            instruction="lift",
            factor=2.5,
            projection_seed=7,
        )
        self.wrong = mismatched
        with self.assertRaisesRegex(subject.G1MiddleControlError, "projection geometry"):
            self._materialize("blocked")
        self.assertFalse((self.root / "blocked").exists())

    def test_tampered_cache_fails_receipt_replay(self) -> None:
        self._materialize()
        path = self.root / "cohort" / "incomplete.safetensors"
        path.write_bytes(path.read_bytes() + b"tamper")
        with self.assertRaises(subject.G1MiddleControlError):
            subject.verify_cohort_receipt(self.root / "cohort" / "cohort_receipt.json")


if __name__ == "__main__":
    unittest.main()
