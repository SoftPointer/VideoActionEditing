#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_native_action_scaffold_identity_rebinding_canary as scaffold
import infer_native_role_reversed_owner_identity_rebinding_canary as subject


class RoleReversedOwnerContractTests(unittest.TestCase):
    def test_fixed_matched_three_arm_design(self) -> None:
        self.assertEqual(
            subject.ARM_ORDER,
            ("owner-only", "owner-source-refs", "source-source-refs"),
        )
        plan = subject.arm_plan()
        self.assertEqual(len(plan), 3)
        self.assertFalse(plan[0].use_source_references)
        self.assertTrue(plan[1].use_source_references)
        self.assertTrue(plan[2].use_source_references)
        self.assertEqual(plan[0].video_role, plan[1].video_role)
        self.assertNotEqual(plan[1].video_role, plan[2].video_role)

    def test_condition_marginal_and_source_ids_are_exact(self) -> None:
        owner_only, owner_refs, source_refs = subject.arm_plan()
        self.assertEqual(
            subject.condition_plan(owner_only),
            {
                "video_role": "pure_t2v_owner_predecode_clean_latent",
                "reference_video_role": None,
                "source_reference_indices_in_order": [],
                "first_video_alone_enters_v": True,
                "video_and_references_enter_vi": True,
                "target_source_id": 0,
                "video_source_id": 1.0,
                "vi_reference_source_ids": [],
                "image_only_reference_source_ids": [],
                "native_source_id_extrapolation_used": False,
            },
        )
        self.assertEqual(
            subject.condition_plan(owner_refs)["vi_reference_source_ids"],
            [2.0, 3.0, 4.0, 5.0],
        )
        self.assertEqual(
            subject.condition_plan(owner_refs)["image_only_reference_source_ids"],
            [1.0, 2.0, 3.0, 4.0],
        )
        self.assertEqual(
            subject.condition_plan(source_refs)["source_reference_indices_in_order"],
            [0, 27, 53, 80],
        )
        for spec in subject.arm_plan():
            native_spec = subject.scaffold_spec(spec)
            contract = scaffold.condition_source_id_contract(native_spec)
            self.assertTrue(contract["all_patch_source_ids_within_trained_interval_0_through_5"])
            self.assertFalse(contract["conditioning_source_id_extrapolation_used"])

    def test_cell_geometry_is_dynamic_and_exact(self) -> None:
        self.assertEqual(
            subject.bucket_and_patch_geometry((1, 16, 21, 60, 62)),
            ((480, 496), 19_530, 930),
        )
        self.assertEqual(
            subject.bucket_and_patch_geometry((1, 16, 21, 64, 58)),
            ((512, 464), 19_488, 928),
        )
        for invalid in (
            (1, 16, 20, 60, 62),
            (1, 16, 21, 61, 62),
            (2, 16, 21, 60, 62),
        ):
            with self.assertRaises(subject.RoleReversedOwnerCanaryError):
                subject.bucket_and_patch_geometry(invalid)

    def test_role_clause_is_conditional_and_shared(self) -> None:
        value = subject.renderer_body("A dog sits.")
        self.assertTrue(value.startswith("A dog sits."))
        self.assertIn("When source reference images are present", value)
        self.assertIn("full-video condition primarily as temporal action evidence", value)
        with self.assertRaises(subject.RoleReversedOwnerCanaryError):
            subject.renderer_body("  ")

    def test_dynamic_native_observer_contract_always_restores(self) -> None:
        before = (
            scaffold.PATCH_TOKENS,
            scaffold.REFERENCE_PATCH_TOKENS,
            scaffold.TARGET_SEED,
        )
        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with subject.dynamic_scaffold_audit_contract(
                target_patch_tokens=19_488,
                reference_patch_tokens=928,
                seed=17,
            ):
                self.assertEqual(scaffold.PATCH_TOKENS, 19_488)
                self.assertEqual(scaffold.REFERENCE_PATCH_TOKENS, 928)
                self.assertEqual(scaffold.TARGET_SEED, 17)
                raise RuntimeError("sentinel")
        self.assertEqual(
            (
                scaffold.PATCH_TOKENS,
                scaffold.REFERENCE_PATCH_TOKENS,
                scaffold.TARGET_SEED,
            ),
            before,
        )

    @staticmethod
    def valid_args() -> argparse.Namespace:
        return argparse.Namespace(
            num_inference_steps=40,
            experimental_owner_primal_ack=subject.EXPERIMENTAL_ACK,
            expected_registry_sha256="a" * 64,
            expected_owner_master_receipt_sha256="b" * 64,
            expected_audit_sidecar_sha256="f" * 64,
            expected_audit_public_key_sha256="1" * 64,
            runtime_source_archive_sha256="c" * 64,
            launcher_source_sha256="d" * 64,
            expected_checkpoint_tree_sha256=(
                subject.native.legacy.trainer.CHECKPOINT_TREE_SHA256
            ),
            runtime_source_revision="e" * 40,
            expected_bernini_commit=subject.native.legacy.trainer.BERNINI_OFFICIAL_COMMIT,
            expected_veomni_commit=subject.native.legacy.trainer.VEOMNI_TESTED_COMMIT,
        )

    def test_cli_requires_exact40_and_explicit_primal_ack(self) -> None:
        args = self.valid_args()
        subject._validate_cli(args)
        args.num_inference_steps = 1
        with self.assertRaisesRegex(subject.RoleReversedOwnerCanaryError, "exact40"):
            subject._validate_cli(args)
        args = self.valid_args()
        args.experimental_owner_primal_ack = "yes"
        with self.assertRaisesRegex(subject.RoleReversedOwnerCanaryError, "acknowledgement"):
            subject._validate_cli(args)

    def test_runtime_declares_no_training_or_pseudo_target(self) -> None:
        text = Path(subject.__file__).read_text(encoding="utf-8")
        self.assertNotIn("torch.optim", text)
        self.assertNotIn(".backward(", text)
        self.assertIn('"training_performed": False', text)
        self.assertIn('"pseudo_target_distillation_performed": False', text)
        self.assertIn('"q_mosaic_allowed_owner_channel": False', text)
        self.assertIn("load_pending_owner_generation_inputs", text)

    def test_source_vae_condition_is_encoded_once_then_broadcast(self) -> None:
        text = Path(subject.__file__).read_text(encoding="utf-8")
        rank_zero = text.index("if distributed.rank == 0:", text.index("vae.to(device)"))
        encode = text.index("source_latent = _vae_encode", rank_zero)
        broadcast = text.index("dist.broadcast(source_latent, src=0)", encode)
        identity = text.index("source_identity = native._all_rank_tensor_identity", broadcast)
        self.assertLess(rank_zero, encode)
        self.assertLess(encode, broadcast)
        self.assertLess(broadcast, identity)
        self.assertIn("dist.broadcast(source_references[index], src=0)", text)


if __name__ == "__main__":
    unittest.main()
