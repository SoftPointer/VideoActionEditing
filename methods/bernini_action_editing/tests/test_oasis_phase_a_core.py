from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = METHOD_ROOT / "oasis_phase_a_core.py"
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import oasis_phase_a_core as oasis


def _sha(character: str) -> str:
    return character * 64


class OASISSourceSetCoreTests(unittest.TestCase):
    sample_id = "dog-fit"
    sample_digest = _sha("1")
    source_video_sha = _sha("2")
    instruction_sha = _sha("3")
    source_conditioning_sha = _sha("4")
    frame_set_sha = _sha("5")
    parent_sha = _sha("6")
    descriptor_sha = _sha("7")
    carrier_sha = _sha("8")
    seed = 20260808
    shape = [1, 16, 21, 8, 8]

    @property
    def source_instruction_sha(self) -> str:
        return oasis.object_sha256(
            {
                "source_video_sha256": self.source_video_sha,
                "edit_instruction_sha256": self.instruction_sha,
            }
        )

    @property
    def carrier_seed(self) -> int:
        return oasis.carrier_seed_for(
            sample_digest=self.sample_digest, seed=self.seed
        )

    def artifact(self, raw_sha: str, file_character: str) -> dict:
        return {
            "path": f"/sealed/{file_character}.safetensors",
            "file_sha256": _sha(file_character),
            "raw_value_sha256": raw_sha,
            "shape": list(self.shape),
        }

    def operator_receipt(self, arm: str) -> dict:
        rho = oasis.NOISE_RHO_BY_ARM[arm]
        active = rho > 0.0
        return {
            "schema_version": "bernini-motion-null-appearance-noise-v2",
            "ablation_only": True,
            "trainer_integration_executed": False,
            "operator_self_registers_sampler_hook": False,
            "operator_self_registers_launcher": False,
            "scientific_claim_authorized": False,
            "semantic_old_action_absence_claimed": False,
            "forbidden_api_inputs": sorted(oasis.FORBIDDEN_OPERATOR_INPUTS),
            "diagnostics": {
                "rho": rho,
                "carrier_seed": self.carrier_seed,
                "gaussian_shape": list(self.shape),
                "independent_frame_count": 4,
                "source_temporal_indices_consumed": False,
                "source_temporal_phase_consumed": False,
                "source_spatial_phase_consumed": False,
                "source_low_frequency_layout_consumed": False,
                "carrier_strict_temporal_dc": True,
                "numerical_audit_passed": True,
                "descriptor_sha256": self.descriptor_sha if active else None,
                "carrier_sha256": self.carrier_sha if active else None,
                "rho_zero_exact_object_alias": not active,
                "source_conditioned_non_gaussian": active,
                "carrier_constructed": active,
            },
        }

    def row(self, arm: str) -> dict:
        rho = oasis.NOISE_RHO_BY_ARM[arm]
        active = rho > 0.0
        injected_sha = _sha("9" if arm.endswith("005") else "a") if active else self.parent_sha
        operator = self.operator_receipt(arm)
        unsigned = {
            "schema_version": oasis.ROLLOUT_SCHEMA,
            "candidate_id": oasis.candidate_id_for(
                sample_id=self.sample_id, seed=self.seed, noise_arm=arm
            ),
            "sample_id": self.sample_id,
            "sample_digest": self.sample_digest,
            "source_video_sha256": self.source_video_sha,
            "edit_instruction_sha256": self.instruction_sha,
            "source_instruction_binding_digest": self.source_instruction_sha,
            "source_conditioning_digest": self.source_conditioning_sha,
            "family": "dog_sit_hold",
            "analysis_split": "fit",
            "seed": self.seed,
            "noise_arm": arm,
            "source_carrier_rho": rho,
            "carrier_seed": self.carrier_seed,
            "source_frame_set_digest": self.frame_set_sha,
            "source_frame_order_consumed": False,
            "full_video_latent_consumed_by_carrier": False,
            "operator_receipt": operator,
            "operator_receipt_digest": oasis.object_sha256(operator),
            "operator_runtime_binding": {
                "callable": oasis.NOISE_OPERATOR_CALLABLE,
                "integration_owner": (
                    "infer_oasis_phase_a_noise_bank._sample_with_oasis_noise_arm"
                ),
                "official_randn_called_first": True,
                "inference_integration_executed": True,
                "operator_self_registered_sampler_hook": False,
            },
            "parent_official_gaussian_raw_value_sha256": self.parent_sha,
            "baseline_artifact": self.artifact(self.parent_sha, "b"),
            "sampler_initial_noise_artifact": self.artifact(injected_sha, "c"),
            "external_initial_noise_injection": active,
            "rho_zero_exact_native_object_forwarded": not active,
            "active_noise_parent_matches_official_control": True,
            "native_sampling": {
                "num_frames": 81,
                "num_inference_steps": 40,
                "guidance_mode": "rv2v",
                "seed": self.seed,
                "condition_mode": "rv2v4",
                "guidance_policy": "fixed_native_rv2v_no_ablation",
                "guidance_implementation_replaced": False,
                "sample_one_step_replaced": False,
                "scheduler_replaced": False,
                "exact81": True,
                "exact40": True,
            },
            "endpoint": {
                "path": "/sealed/output.mp4",
                "sha256": _sha("d"),
                "frame_count": 81,
                "fps": 25.0,
                "normalized_clean_latent": {
                    "path": "/sealed/clean.safetensors",
                    "sha256": _sha("e"),
                    "shape": list(self.shape),
                    "stored_dtype": "torch.float32",
                    "tensor_key": "normalized_clean_latent",
                    "native_sampler_before_vae_decode": True,
                    "mp4_decode_reencode_used": False,
                    "roundtrip_byte_exact_fp32": True,
                },
            },
            "endpoint_candidate_only": True,
            "legacy_pair_v5_native_rollout_schema_compatible": False,
            "external_action_scorer_consumed": False,
            "action_source_scoring_performed": False,
            "endpoint_selection_performed": False,
            "optimizer_or_training_authorized": False,
            "training_performed": False,
            "scientific_action_editing_success_claim": False,
        }
        return {**unsigned, "rollout_digest": oasis.object_sha256(unsigned)}

    def rows(self) -> list[dict]:
        return [self.row(arm) for arm in oasis.NOISE_ARM_ORDER]

    @staticmethod
    def reseal(row: dict) -> None:
        unsigned = dict(row)
        unsigned.pop("rollout_digest", None)
        row["rollout_digest"] = oasis.object_sha256(unsigned)

    def validate(self, rows):
        return oasis.validate_matched_rollout_triplet(
            rows,
            sample_id=self.sample_id,
            sample_digest=self.sample_digest,
            source_video_sha256=self.source_video_sha,
            edit_instruction_sha256=self.instruction_sha,
            source_conditioning_digest=self.source_conditioning_sha,
            source_frame_set_digest=self.frame_set_sha,
            family="dog_sit_hold",
            analysis_split="fit",
            seed=self.seed,
        )

    def test_core_is_dependency_light_and_contains_no_scorer_or_optimizer(self) -> None:
        source = CORE_PATH.read_text(encoding="utf-8")
        for forbidden in (
            "import torch",
            "mace_candidate_action_energy",
            "optimizer.step",
            "backward()",
            "denoising_energy",
            "run_candidate_own_one_step_search",
        ):
            self.assertNotIn(forbidden, source)
        contract = oasis.static_contract()
        self.assertTrue(contract["candidate_generation_only"])
        self.assertFalse(contract["external_action_scorer_dependency"])
        self.assertFalse(contract["endpoint_selection_performed"])
        self.assertFalse(contract["optimizer_authorized"])

    def test_seed_and_candidate_ids_are_domain_separated_and_deterministic(self) -> None:
        first = oasis.carrier_seed_for(
            sample_digest=self.sample_digest, seed=self.seed
        )
        second = oasis.carrier_seed_for(
            sample_digest=self.sample_digest, seed=self.seed
        )
        other = oasis.carrier_seed_for(
            sample_digest=self.sample_digest, seed=self.seed + 1
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertEqual(
            oasis.candidate_id_for(
                sample_id=self.sample_id,
                seed=self.seed,
                noise_arm="official_gaussian",
            ),
            f"{self.sample_id}-s{self.seed}-official_gaussian",
        )

    def test_valid_matched_triplet_is_accepted(self) -> None:
        audit = self.validate(self.rows())
        self.assertEqual(audit.sample_id, self.sample_id)
        self.assertEqual(audit.source_conditioning_digest, self.source_conditioning_sha)
        self.assertEqual(audit.parent_official_gaussian_raw_value_sha256, self.parent_sha)
        self.assertEqual(len(audit.candidate_ids), 3)
        self.assertEqual(len(audit.audit_digest), 64)

    def test_cross_source_or_instruction_substitution_fails(self) -> None:
        for field in (
            "source_video_sha256",
            "edit_instruction_sha256",
            "source_conditioning_digest",
            "source_frame_set_digest",
        ):
            rows = self.rows()
            rows[1][field] = _sha("f")
            self.reseal(rows[1])
            with self.subTest(field=field), self.assertRaisesRegex(
                oasis.OASISPhaseAError, "provenance/authority"
            ):
                self.validate(rows)

    def test_all_arms_must_share_the_official_parent_gaussian(self) -> None:
        rows = self.rows()
        rows[2]["parent_official_gaussian_raw_value_sha256"] = _sha("f")
        rows[2]["baseline_artifact"]["raw_value_sha256"] = _sha("f")
        self.reseal(rows[2])
        with self.assertRaisesRegex(oasis.OASISPhaseAError, "share one official"):
            self.validate(rows)

    def test_active_rhos_must_share_descriptor_and_carrier(self) -> None:
        rows = self.rows()
        active = rows[2]
        active["operator_receipt"]["diagnostics"]["descriptor_sha256"] = _sha("f")
        active["operator_receipt_digest"] = oasis.object_sha256(
            active["operator_receipt"]
        )
        self.reseal(active)
        with self.assertRaisesRegex(oasis.OASISPhaseAError, "share one source"):
            self.validate(rows)

    def test_rho_zero_must_forward_the_exact_native_object(self) -> None:
        rows = self.rows()
        rows[0]["external_initial_noise_injection"] = True
        self.reseal(rows[0])
        with self.assertRaisesRegex(oasis.OASISPhaseAError, "rho0"):
            self.validate(rows)

    def test_operator_forbidden_input_closure_is_enforced(self) -> None:
        rows = self.rows()
        active = rows[1]
        active["operator_receipt"]["forbidden_api_inputs"].remove("mask")
        active["operator_receipt_digest"] = oasis.object_sha256(
            active["operator_receipt"]
        )
        self.reseal(active)
        with self.assertRaisesRegex(oasis.OASISPhaseAError, "forbidden inputs"):
            self.validate(rows)

    def test_scoring_selection_and_training_authority_are_all_rejected(self) -> None:
        for field in (
            "external_action_scorer_consumed",
            "action_source_scoring_performed",
            "endpoint_selection_performed",
            "optimizer_or_training_authorized",
            "training_performed",
            "scientific_action_editing_success_claim",
        ):
            rows = self.rows()
            rows[0][field] = True
            self.reseal(rows[0])
            with self.subTest(field=field), self.assertRaisesRegex(
                oasis.OASISPhaseAError, "provenance/authority"
            ):
                self.validate(rows)

    def test_rollout_field_addition_is_rejected_even_when_resealed(self) -> None:
        rows = self.rows()
        rows[0]["target_video_path"] = "/forbidden.mp4"
        self.reseal(rows[0])
        with self.assertRaisesRegex(oasis.OASISPhaseAError, "field closure"):
            self.validate(rows)


if __name__ == "__main__":
    unittest.main()
