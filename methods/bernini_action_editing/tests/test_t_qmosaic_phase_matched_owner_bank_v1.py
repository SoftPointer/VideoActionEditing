from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import sys
from typing import Any
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import torch

import self_imagined_motion_cotangent_v1 as motion
import t_qmosaic_phase_matched_owner_bank_v1 as bank
import t_qmosaic_trajectory_intervention_v1 as trajectory


CELL_ID = "dog"
LATENT_SHAPES = {
    "dog": (1, 16, 21, 60, 62),
    "human": (1, 16, 21, 64, 58),
}


def _clean_latent(cell_id: str = CELL_ID) -> torch.Tensor:
    return torch.zeros(LATENT_SHAPES[cell_id], dtype=torch.float32).detach()


def _authority(clean: torch.Tensor) -> dict[str, Any]:
    clean_digest = motion.tensor_value_digest(clean, label="test owner clean latent")
    return {
        "registry_file_sha256": bank.PINNED_REGISTRY_FILE_SHA256,
        "owner_master_receipt_digest": "1" * 64,
        "owner_child_receipt_digest": "2" * 64,
        "external_full81_audit_sidecar_receipt_digest": "3" * 64,
        "owner_clean_latent_file_sha256": "4" * 64,
        "owner_clean_latent_tensor_sha256": clean_digest,
        "checkpoint_content_receipt_digest": "5" * 64,
        "bernini_revision": bank.PINNED_BERNINI_REVISION,
        "veomni_revision": bank.PINNED_VEOMNI_REVISION,
    }


def _hidden_triplets(cell_id: str = CELL_ID) -> dict[int, dict[int, dict[str, torch.Tensor]]]:
    seeds = bank.CELL_SPECS[cell_id]["query_seeds"]
    spatial = torch.tensor((0.7, 1.0, 1.4), dtype=torch.float32).reshape(1, 3)
    reverse_spatial = torch.flip(spatial, dims=(1,)).contiguous()
    temporal = torch.linspace(-1.0, 1.0, 21, dtype=torch.float32).reshape(21, 1)
    curved = (temporal.square() - temporal.square().mean()).contiguous()
    result: dict[int, dict[int, dict[str, torch.Tensor]]] = {}
    for phase_position, phase_index in enumerate(bank.PHASE_INDICES):
        phase: dict[int, dict[str, torch.Tensor]] = {}
        for seed_position, query_seed in enumerate(seeds):
            code = 1 + phase_position * 7 + seed_position
            base = torch.zeros((1, 21, 3, 1536), dtype=torch.float32)
            base[0, :, :, 0] = (
                (1.0 + 0.08 * phase_position + 0.01 * seed_position)
                * temporal
                * spatial
            )
            base[0, :, :, 1] = (
                (0.08 + 0.01 * code) * curved * reverse_spatial
            )
            noop = torch.full_like(base, float(code) * 1.0e-4).contiguous()
            noop[0, :, :, 2] = float(code) * 2.0e-4
            action = (noop + base).float().contiguous().detach()
            reverse = (noop - base).float().contiguous().detach()
            phase[query_seed] = {
                "action": action,
                "reverse_wrong_family": reverse,
                "common_scene_noop": noop.detach(),
            }
        result[phase_index] = phase
    return result


def _proofs_from_state_rows(
    rows: tuple[dict[str, Any], ...],
) -> dict[int, dict[int, dict[str, Any]]]:
    result: dict[int, dict[int, dict[str, Any]]] = {
        phase: {} for phase in bank.PHASE_INDICES
    }
    for row in rows:
        phase_index = row["native_schedule_index"]
        query_seed = row["query_seed"]
        invocation = hashlib.sha256(
            f"owner-forward:{phase_index}:{query_seed}".encode("ascii")
        ).hexdigest()
        result[phase_index][query_seed] = {
            **row,
            "prompt_order": list(bank.PROMPT_ORDER),
            "same_x_sigma_object_for_all_three_prompts": True,
            "shared_tensor_bytes_unchanged": True,
            "hook_coordinate": motion.HOOK_COORDINATE,
            "target_suffix_only": True,
            "source_condition_consumed": False,
            "mask_flow_pose_track_trajectory_consumed": False,
            "spatial_orderless_sketch": True,
            "full_hidden_persisted": False,
            "transformer_frozen": True,
            "adapter_loaded": False,
            "phase_forward_invocation_digest": invocation,
        }
    return result


def _recursive_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(_recursive_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_recursive_keys(child))
    return keys


class PhaseMatchedOwnerBankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.clean = _clean_latent()
        cls.state_bundle = bank.derive_fixed_owner_state_bindings_v1(
            cell_id=CELL_ID,
            owner_clean_latent=cls.clean,
        )
        cls.state_rows = cls.state_bundle.rows_in_fixed_order()

    def _inputs(self) -> tuple[
        torch.Tensor,
        dict[int, dict[int, dict[str, torch.Tensor]]],
        dict[int, dict[int, dict[str, Any]]],
        dict[str, Any],
    ]:
        clean = self.clean.clone().detach()
        return (
            clean,
            _hidden_triplets(),
            _proofs_from_state_rows(self.state_rows),
            _authority(clean),
        )

    def _build(
        self,
        *,
        clean: torch.Tensor | None = None,
        hidden: dict[int, dict[int, dict[str, torch.Tensor]]] | None = None,
        proofs: dict[int, dict[int, dict[str, Any]]] | None = None,
        authority: dict[str, Any] | None = None,
    ) -> bank.PhaseMatchedOwnerQuotientBankV1:
        default_clean, default_hidden, default_proofs, default_authority = self._inputs()
        actual_clean = default_clean if clean is None else clean
        return bank.build_phase_matched_owner_bank_v1(
            cell_id=CELL_ID,
            owner_clean_latent=actual_clean,
            hidden_triplets=default_hidden if hidden is None else hidden,
            forward_proofs=default_proofs if proofs is None else proofs,
            owner_authority_binding=(
                default_authority if authority is None else authority
            ),
        )

    def test_fixed_registry_schedule_and_canary_are_owner_bank_only(self) -> None:
        self.assertEqual(bank.PHASE_INDICES, (20, 28, 33))
        self.assertEqual(bank.PHASE_TIMESTEPS, (833, 682, 516))
        self.assertEqual(
            bank.PHASE_SIGMA_FLOAT32_HEX,
            ("3f556787", "3f2ebaf8", "3f042120"),
        )
        self.assertEqual(bank.PACKED_CHANNELS, 64)
        self.assertNotIn("latent_shape", bank.CELL_SPECS["dog"])
        self.assertEqual(
            bank.CELL_SPECS["dog"]["source_iid"], "7b88a1ca1f804f41"
        )
        self.assertEqual(
            bank.CELL_SPECS["human"]["source_iid"], "a35b590961d24694"
        )
        self.assertFalse(any("vjp" in name.lower() for name in bank.__all__))

        receipt = bank.preregistered_canary_receipt_v1()
        self.assertEqual(receipt["schedule_sha256"], trajectory.PINNED_SCHEDULE_SHA256)
        self.assertEqual(receipt["guidance_mode"], "t2v_apg_owner_hidden_queries")
        self.assertEqual(receipt["owner_hidden_forward_count_per_cell"], 18)
        self.assertEqual(receipt["editor_hidden_forward_count_per_cell"], 0)
        self.assertFalse(receipt["packed_state_vjp_materialized"])
        self.assertFalse(receipt["trajectory_replay_performed"])
        self.assertFalse(receipt["seed_selection"])
        self.assertFalse(receipt["dose_input_authorized"])
        self.assertFalse(receipt["sign_input_authorized"])
        self.assertFalse(receipt["optimizer_constructed"])
        self.assertFalse(receipt["training_update_authorized"])
        self.assertNotIn("latent_shape", _recursive_keys(receipt))
        self.assertNotIn("packed_layout", _recursive_keys(receipt))

    def test_state_bundle_binds_every_phase_without_retaining_primal(self) -> None:
        rows = self.state_bundle.rows_in_fixed_order()
        expected_pairs = tuple(
            (phase, seed)
            for phase in bank.PHASE_INDICES
            for seed in bank.CELL_SPECS[CELL_ID]["query_seeds"]
        )
        self.assertEqual(
            tuple((row["native_schedule_index"], row["query_seed"]) for row in rows),
            expected_pairs,
        )
        for position, phase_index in enumerate(bank.PHASE_INDICES):
            for row in rows[position * 2 : position * 2 + 2]:
                self.assertEqual(row["native_schedule_index"], phase_index)
                self.assertEqual(row["native_timestep"], bank.PHASE_TIMESTEPS[position])
                self.assertEqual(
                    row["sigma_float32_hex"],
                    bank.PHASE_SIGMA_FLOAT32_HEX[position],
                )
        for seed in bank.CELL_SPECS[CELL_ID]["query_seeds"]:
            seed_rows = [row for row in rows if row["query_seed"] == seed]
            self.assertEqual(
                len({row["official_gaussian_tensor_sha256"] for row in seed_rows}),
                1,
            )
            self.assertEqual(
                len({row["same_x_sigma_tensor_sha256"] for row in seed_rows}),
                3,
            )
        receipt = self.state_bundle.receipt()
        self.assertEqual(receipt["schedule_sha256"], trajectory.PINNED_SCHEDULE_SHA256)
        self.assertTrue(receipt["all_x_sigma_digests_distinct_per_query_seed"])
        self.assertFalse(receipt["owner_clean_latent_tensor_persisted"])
        self.assertFalse(receipt["owner_gaussian_or_noise_tensor_persisted"])
        self.assertFalse(receipt["owner_x_sigma_tensor_persisted"])
        self.assertFalse(
            receipt["owner_patch_layout_or_spatial_coordinate_count_persisted"]
        )
        self.assertNotIn("latent_shape", _recursive_keys(receipt))

        exported = self.state_bundle.rows_in_fixed_order()
        exported[0]["native_timestep"] = -1
        self.assertEqual(
            self.state_bundle.rows_in_fixed_order()[0]["native_timestep"], 833
        )

    def test_builds_six_phase_specific_rows_and_closed_receipt(self) -> None:
        result = self._build()
        rows = result.rows_in_fixed_order()
        self.assertEqual(len(rows), 6)
        self.assertEqual(
            tuple((row.phase_index, row.query_seed) for row in rows),
            tuple(
                (phase, seed)
                for phase in bank.PHASE_INDICES
                for seed in bank.CELL_SPECS[CELL_ID]["query_seeds"]
            ),
        )
        payload = result.tensor_payload_in_fixed_order()
        self.assertEqual(
            tuple(payload),
            tuple(
                f"phase_{phase}_query_seed_{seed}"
                for phase in bank.PHASE_INDICES
                for seed in bank.CELL_SPECS[CELL_ID]["query_seeds"]
            ),
        )
        for value in payload.values():
            self.assertEqual(value.dtype, torch.float32)
            self.assertFalse(value.requires_grad)
            self.assertTrue(bool(torch.isfinite(value).all().item()))

        receipt = result.receipt()
        binding = receipt["owner_source_seed_prompt_binding"]
        self.assertEqual(binding["source_iid"], bank.CELL_SPECS[CELL_ID]["source_iid"])
        self.assertEqual(
            binding["source_video_sha256"],
            bank.CELL_SPECS[CELL_ID]["source_video_sha256"],
        )
        self.assertEqual(
            binding["action_caption_utf8_sha256"],
            bank.CELL_SPECS[CELL_ID]["action_caption_utf8_sha256"],
        )
        self.assertEqual(receipt["schedule_sha256"], trajectory.PINNED_SCHEDULE_SHA256)
        self.assertEqual(
            len({row["state_binding_digest"] for row in receipt["rows"]}), 6
        )
        self.assertEqual(
            len({row["hidden_triplet_digest"] for row in receipt["rows"]}), 6
        )
        self.assertEqual(
            len(
                {
                    row["phase_forward_invocation_digest"]
                    for row in receipt["rows"]
                }
            ),
            6,
        )
        for row in receipt["rows"]:
            specificity = row["prompt_specificity"]
            self.assertGreaterEqual(
                specificity["reverse_wrong_family_margin"],
                bank.MINIMUM_SPECIFICITY_MARGIN,
            )
            self.assertGreaterEqual(
                specificity["common_scene_null_margin"],
                bank.MINIMUM_SPECIFICITY_MARGIN,
            )
            self.assertTrue(specificity["all_margins_pass_without_compensation"])
        self.assertFalse(receipt["index33_tensor_reuse_or_broadcast"])
        self.assertTrue(receipt["all_phase_hidden_tensor_objects_and_storages_distinct"])
        self.assertTrue(receipt["all_phase_hidden_tensor_value_digests_distinct"])
        self.assertTrue(receipt["all_phase_forward_invocation_digests_distinct"])
        self.assertFalse(receipt["owner_rgb_persisted"])
        self.assertFalse(receipt["owner_clean_latent_tensor_or_shape_persisted"])
        self.assertFalse(receipt["owner_gaussian_or_noise_tensor_persisted"])
        self.assertFalse(receipt["owner_x_sigma_tensor_persisted"])
        self.assertFalse(receipt["owner_full_hidden_tensor_persisted"])
        self.assertFalse(
            receipt["owner_patch_layout_or_spatial_coordinate_count_persisted"]
        )
        self.assertFalse(receipt["external_callback_consumed"])
        self.assertFalse(receipt["seed_selection"])
        self.assertFalse(receipt["dose_input_authorized"])
        self.assertFalse(receipt["sign_input_authorized"])
        self.assertFalse(receipt["optimizer_constructed"])
        self.assertFalse(receipt["training_update_authorized"])
        self.assertNotIn("latent_shape", _recursive_keys(receipt))
        self.assertNotIn("owner_spatial_coordinates", _recursive_keys(receipt))
        self.assertNotIn("packed_layout", _recursive_keys(receipt))

    def test_rejects_reordered_axes_wrong_schedule_and_wrong_authority(self) -> None:
        clean, hidden, proofs, authority = self._inputs()
        reordered = {
            bank.PHASE_INDICES[1]: hidden[bank.PHASE_INDICES[1]],
            bank.PHASE_INDICES[0]: hidden[bank.PHASE_INDICES[0]],
            bank.PHASE_INDICES[2]: hidden[bank.PHASE_INDICES[2]],
        }
        with self.assertRaisesRegex(bank.TQMosaicPhaseOwnerBankError, "phase order"):
            self._build(clean=clean, hidden=reordered, proofs=proofs, authority=authority)

        bad_proofs = copy.deepcopy(proofs)
        seed = bank.CELL_SPECS[CELL_ID]["query_seeds"][0]
        bad_proofs[28][seed]["sigma_float32_hex"] = "00000000"
        with self.assertRaisesRegex(bank.TQMosaicPhaseOwnerBankError, "proof differs"):
            self._build(clean=clean, hidden=hidden, proofs=bad_proofs, authority=authority)

        bad_authority = dict(authority)
        bad_authority["owner_clean_latent_tensor_sha256"] = "0" * 64
        with self.assertRaisesRegex(bank.TQMosaicPhaseOwnerBankError, "authority"):
            self._build(
                clean=clean,
                hidden=hidden,
                proofs=proofs,
                authority=bad_authority,
            )

    def test_rejects_same_object_and_exact_clone_across_phases(self) -> None:
        clean, hidden, proofs, authority = self._inputs()
        seed = bank.CELL_SPECS[CELL_ID]["query_seeds"][0]
        hidden[20][seed]["action"] = hidden[33][seed]["action"]
        with self.assertRaisesRegex(bank.TQMosaicPhaseOwnerBankError, "reused"):
            self._build(clean=clean, hidden=hidden, proofs=proofs, authority=authority)

        clean, hidden, proofs, authority = self._inputs()
        hidden[20][seed]["action"] = hidden[33][seed]["action"].clone().detach()
        with self.assertRaisesRegex(bank.TQMosaicPhaseOwnerBankError, "cloned"):
            self._build(clean=clean, hidden=hidden, proofs=proofs, authority=authority)

    def test_rejects_distinct_views_of_one_storage_across_phases(self) -> None:
        clean, hidden, proofs, authority = self._inputs()
        seed = bank.CELL_SPECS[CELL_ID]["query_seeds"][0]
        left = hidden[20][seed]["action"]
        right = hidden[33][seed]["action"]
        backing = torch.empty(
            (*left.shape[:-1], left.shape[-1] * 2), dtype=torch.float32
        )
        backing[..., : left.shape[-1]].copy_(left)
        backing[..., left.shape[-1] :].copy_(right)
        hidden[20][seed]["action"] = backing[..., : left.shape[-1]].detach()
        hidden[33][seed]["action"] = backing[..., left.shape[-1] :].detach()
        with self.assertRaisesRegex(bank.TQMosaicPhaseOwnerBankError, "aliased"):
            self._build(clean=clean, hidden=hidden, proofs=proofs, authority=authority)

    def test_rejects_reused_phase_forward_invocation(self) -> None:
        clean, hidden, proofs, authority = self._inputs()
        seeds = bank.CELL_SPECS[CELL_ID]["query_seeds"]
        proofs[20][seeds[0]]["phase_forward_invocation_digest"] = proofs[33][
            seeds[1]
        ]["phase_forward_invocation_digest"]
        with self.assertRaisesRegex(bank.TQMosaicPhaseOwnerBankError, "invocation"):
            self._build(clean=clean, hidden=hidden, proofs=proofs, authority=authority)

    def test_action_reverse_specificity_is_noncompensating_per_row(self) -> None:
        clean, hidden, proofs, authority = self._inputs()
        seed = bank.CELL_SPECS[CELL_ID]["query_seeds"][0]
        action = hidden[28][seed]["action"]
        # A temporally static offset changes bytes/storage but is removed by
        # Phi, making the wrong-family quotient numerically equal to action.
        hidden[28][seed]["reverse_wrong_family"] = (
            action + 0.012345
        ).float().contiguous().detach()
        with self.assertRaisesRegex(bank.TQMosaicPhaseOwnerBankError, "specificity"):
            self._build(clean=clean, hidden=hidden, proofs=proofs, authority=authority)

    def test_spatial_permutation_leaves_quotient_values_unchanged(self) -> None:
        clean, hidden, proofs, authority = self._inputs()
        original = self._build(
            clean=clean,
            hidden=hidden,
            proofs=proofs,
            authority=authority,
        ).tensor_payload_in_fixed_order()
        permuted = {
            phase: {
                seed: {
                    role: value.flip(dims=(2,)).contiguous().detach()
                    for role, value in triplet.items()
                }
                for seed, triplet in phase_rows.items()
            }
            for phase, phase_rows in hidden.items()
        }
        changed = self._build(
            clean=clean,
            hidden=permuted,
            proofs=proofs,
            authority=authority,
        ).tensor_payload_in_fixed_order()
        self.assertEqual(tuple(original), tuple(changed))
        for key in original:
            torch.testing.assert_close(
                original[key], changed[key], rtol=2.0e-6, atol=2.0e-6
            )

    def test_payload_is_cloned_and_internal_tensor_mutation_is_detected(self) -> None:
        result = self._build()
        payload = result.tensor_payload_in_fixed_order()
        key = next(iter(payload))
        payload[key].zero_()
        self.assertNotEqual(
            int(torch.count_nonzero(result.tensor_payload_in_fixed_order()[key]).item()),
            0,
        )
        result.rows_in_fixed_order()[0].unit_feature.add_(1.0)
        with self.assertRaisesRegex(bank.TQMosaicPhaseOwnerBankError, "bytes changed"):
            result.receipt()

    def test_public_build_surface_has_no_control_or_update_knobs(self) -> None:
        parameters = set(inspect.signature(bank.build_phase_matched_owner_bank_v1).parameters)
        self.assertEqual(
            parameters,
            {
                "cell_id",
                "owner_clean_latent",
                "hidden_triplets",
                "forward_proofs",
                "owner_authority_binding",
            },
        )
        forbidden = {
            "callback",
            "seed",
            "dose",
            "sign",
            "candidate",
            "optimizer",
            "update",
        }
        self.assertTrue(parameters.isdisjoint(forbidden))
        source = inspect.getsource(bank.build_phase_matched_owner_bank_v1)
        self.assertNotIn("torch.autograd", source)
        self.assertNotIn("optimizer.step", source)


class PinnedBerniniLayoutTests(unittest.TestCase):
    def test_pack_unpack_roundtrip_exact_for_dog_and_human_geometry(self) -> None:
        for shape in LATENT_SHAPES.values():
            count = 1
            for extent in shape:
                count *= extent
            spatial = torch.arange(count, dtype=torch.float32).reshape(shape)
            packed = bank.pack_pinned_bernini_state_v1(
                spatial, latent_shape=shape
            )
            self.assertEqual(packed.shape, bank.expected_packed_shape(shape))
            restored = bank.unpack_pinned_bernini_state_v1(
                packed, latent_shape=shape
            )
            self.assertTrue(torch.equal(restored, spatial))

    def test_pack_unpack_preserves_graph_and_rejects_bad_inputs(self) -> None:
        shape = (1, 16, 21, 4, 6)
        spatial = torch.linspace(-1.0, 1.0, 1 * 16 * 21 * 4 * 6).reshape(shape)
        spatial.requires_grad_(True)
        packed = bank.pack_pinned_bernini_state_v1(spatial, latent_shape=shape)
        restored = bank.unpack_pinned_bernini_state_v1(
            packed, latent_shape=shape
        )
        self.assertTrue(torch.equal(restored, spatial))
        self.assertTrue(restored.requires_grad)
        restored.square().sum().backward()
        self.assertIsNotNone(spatial.grad)
        self.assertTrue(bool(torch.isfinite(spatial.grad).all().item()))

        with self.assertRaises(bank.TQMosaicPhaseOwnerBankError):
            bank.pack_pinned_bernini_state_v1(
                torch.zeros(shape, dtype=torch.float64), latent_shape=shape
            )
        nonfinite = torch.zeros(shape, dtype=torch.float32)
        nonfinite.reshape(-1)[0] = float("nan")
        with self.assertRaises(bank.TQMosaicPhaseOwnerBankError):
            bank.pack_pinned_bernini_state_v1(nonfinite, latent_shape=shape)
        with self.assertRaises(bank.TQMosaicPhaseOwnerBankError):
            bank.expected_packed_shape((1, 16, 21, 5, 6))
        with self.assertRaises(bank.TQMosaicPhaseOwnerBankError):
            bank.unpack_pinned_bernini_state_v1(
                torch.zeros((1, 3, 64), dtype=torch.float32),
                latent_shape=shape,
            )

    @unittest.skipUnless(
        os.environ.get("BERNINI_TQ_REAL_PACK_TEST") == "1"
        and os.environ.get("BERNINI_TQ_BERNINI_ROOT"),
        "set BERNINI_TQ_REAL_PACK_TEST=1 and BERNINI_TQ_BERNINI_ROOT",
    )
    def test_helpers_match_pinned_vendor_source_and_einops(self) -> None:
        vendor_root = Path(os.environ["BERNINI_TQ_BERNINI_ROOT"])
        source_path = vendor_root / "bernini" / "models" / "wan_diffusion.py"
        source_bytes = source_path.read_bytes()
        self.assertEqual(
            hashlib.sha256(source_bytes).hexdigest(),
            bank.PINNED_WAN_DIFFUSION_SHA256,
        )
        source = source_bytes.decode("utf-8")
        pack_match = re.search(r'^_PACK = "([^"]+)"$', source, re.MULTILINE)
        unpack_match = re.search(r'^_UNPACK = "([^"]+)"$', source, re.MULTILINE)
        self.assertIsNotNone(pack_match)
        self.assertIsNotNone(unpack_match)
        self.assertIn(
            "def _to_spatial(x, shape):\n"
            "    return rearrange(x, _PACK, t=shape[2], h=shape[3] // 2, "
            "w=shape[4] // 2, pt=1, ph=2, pw=2)",
            source,
        )
        self.assertIn(
            "def _to_packed(x, shape):\n"
            "    return rearrange(x, _UNPACK, t=shape[2], h=shape[3] // 2, "
            "w=shape[4] // 2, pt=1, ph=2, pw=2)",
            source,
        )

        from einops import rearrange

        shape = (1, 16, 21, 4, 6)
        spatial = torch.arange(1 * 16 * 21 * 4 * 6, dtype=torch.float32).reshape(shape)
        packed = bank.pack_pinned_bernini_state_v1(spatial, latent_shape=shape)
        vendor_packed = rearrange(
            spatial,
            unpack_match.group(1),
            t=shape[2],
            h=shape[3] // 2,
            w=shape[4] // 2,
            pt=1,
            ph=2,
            pw=2,
        )
        self.assertTrue(torch.equal(packed, vendor_packed))
        vendor_spatial = rearrange(
            packed,
            pack_match.group(1),
            t=shape[2],
            h=shape[3] // 2,
            w=shape[4] // 2,
            pt=1,
            ph=2,
            pw=2,
        )
        self.assertTrue(
            torch.equal(
                vendor_spatial,
                bank.unpack_pinned_bernini_state_v1(packed, latent_shape=shape),
            )
        )


if __name__ == "__main__":
    unittest.main()
