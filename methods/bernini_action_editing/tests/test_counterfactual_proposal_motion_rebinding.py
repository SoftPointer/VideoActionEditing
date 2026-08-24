from __future__ import annotations

import ast
from dataclasses import fields, replace
import hashlib
import inspect
import json
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import counterfactual_proposal_motion_rebinding as cpmr  # noqa: E402


class CPMRPureContractTests(unittest.TestCase):
    def test_exact_dog_contract_is_closed(self) -> None:
        config = cpmr.CPMRConfig()
        self.assertEqual(config.rgb_frames, 81)
        self.assertEqual(config.source_patch_shape, (21, 31, 30, 1536))
        self.assertEqual(config.carrier_shape, (21, 8, 8, 1536))
        self.assertEqual(config.patch_tokens, 21 * 31 * 30)
        self.assertEqual(config.carrier_tokens, 1344)
        self.assertEqual(config.epsilon, 1.0e-6)
        self.assertEqual(config.token_rms_cap, 4.0)
        self.assertEqual(config.coordinate_scale, 0.02)
        self.assertEqual(config.coordinate_axis_width, 512)
        self.assertEqual(config.coordinate_frequencies, 256)
        self.assertEqual(config.contract_sha256(), cpmr.CONFIG_CONTRACT_SHA256)
        self.assertEqual(
            {item.name for item in fields(cpmr.CPMRConfig)},
            {
                "rgb_frames",
                "latent_phases",
                "patch_height",
                "patch_width",
                "hidden_size",
                "pool_height",
                "pool_width",
                "epsilon",
                "token_rms_cap",
                "coordinate_scale",
                "coordinate_axis_width",
                "coordinate_frequencies",
                "coordinate_base",
                "shuffle_source_phases",
            },
        )

    def test_every_nominal_configuration_knob_fails_closed(self) -> None:
        invalid = (
            {"rgb_frames": 41},
            {"latent_phases": 11},
            {"patch_height": 30},
            {"patch_width": 31},
            {"hidden_size": 768},
            {"pool_height": 7},
            {"pool_width": 7},
            {"epsilon": 1.0e-5},
            {"token_rms_cap": 3.99},
            {"coordinate_scale": 0.01},
            {"coordinate_axis_width": 256},
            {"coordinate_frequencies": 128},
            {"coordinate_base": 1000.0},
            {"shuffle_source_phases": tuple(range(1, 21))},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(cpmr.CPMRTensorContractError):
                    cpmr.CPMRConfig(**kwargs)

    def test_reverse_and_shuffle_mappings_are_exact_and_fixed(self) -> None:
        self.assertEqual(
            cpmr.REVERSE_SOURCE_PHASES,
            (0, *tuple(range(20, 0, -1))),
        )
        expected_shuffle = [
            17,
            18,
            1,
            6,
            16,
            4,
            12,
            11,
            7,
            13,
            19,
            2,
            15,
            8,
            3,
            9,
            20,
            5,
            10,
            14,
        ]
        canonical = json.dumps(expected_shuffle, separators=(",", ":"))
        self.assertEqual(cpmr.SHUFFLE_CANONICAL_JSON, canonical)
        self.assertEqual(
            hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "399dbccd424e30cbb4a129c7e5535bdb6475cf6573d024c757eb8491803eb02f",
        )
        self.assertEqual(cpmr.SHUFFLE_SOURCE_PHASES, (0, *expected_shuffle))
        self.assertEqual(sorted(cpmr.SHUFFLE_SOURCE_PHASES), list(range(21)))

    def test_coordinate_spec_and_digests_are_pinned(self) -> None:
        self.assertEqual(cpmr.COORDINATE_SPEC["axis_order"], ["time", "y", "x"])
        self.assertEqual(cpmr.COORDINATE_SPEC["axis_width"], 512)
        self.assertEqual(cpmr.COORDINATE_SPEC["frequencies_per_axis"], 256)
        self.assertEqual(
            cpmr.canonical_json_sha256(cpmr.COORDINATE_SPEC),
            "f3712bd5420a313f3b98f0e06081998f3c14f735ddcc4d34ff42f592f2ecfb03",
        )
        for digest in (
            cpmr.CONFIG_CONTRACT_SHA256,
            cpmr.COORDINATE_SPEC_SHA256,
            cpmr.CANONICAL_COORDINATE_TENSOR_SHA256,
            cpmr.SHUFFLE_CANONICAL_SHA256,
        ):
            self.assertEqual(len(digest), 64)
            int(digest, 16)

    def test_module_has_no_eager_torch_or_out_of_scope_runtime_layer(self) -> None:
        tree = ast.parse(Path(cpmr.__file__).read_text(encoding="utf-8"))
        eager_torch = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                eager_torch.extend(
                    alias.name for alias in node.names if alias.name == "torch"
                )
            elif isinstance(node, ast.ImportFrom) and node.module == "torch":
                eager_torch.append(node.module)
        self.assertEqual(eager_torch, [])
        public_parameters = set()
        for function in (
            cpmr.build_motion_carrier,
            cpmr.build_reverse_control,
            cpmr.build_shuffle_control,
            cpmr.build_negative_control,
            cpmr.build_nn_control,
        ):
            public_parameters.update(inspect.signature(function).parameters)
        self.assertTrue(
            {"processor", "transformer", "ulysses", "lora", "runner"}.isdisjoint(
                public_parameters
            )
        )


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is unavailable")
class CPMRTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # A spatially broadcast field keeps test input storage small while the
        # production builder still materializes and processes the exact dog grid.
        phase = torch.arange(21, dtype=torch.float32).reshape(1, 21, 1, 1, 1)
        channel = torch.arange(1536, dtype=torch.float32).reshape(1, 1, 1, 1, 1536)
        compact = (
            torch.sin((phase + 1.0) * (channel + 1.0) * 0.0017)
            + 0.3 * torch.cos((phase + 2.0) * (channel + 1.0) * 0.0009)
        )
        cls.action_compact = compact
        cls.action = compact.expand(1, 21, 31, 30, 1536)
        cls.noop = torch.zeros(1, 1, 1, 1, 1, dtype=torch.float32).expand(
            1, 21, 31, 30, 1536
        )
        cls.base = cpmr.build_motion_carrier(cls.action, cls.noop)

    def test_fixed_3d_coordinate_encoding_and_tensor_digest(self) -> None:
        coordinate = cpmr.fixed_3d_coordinate_encoding()
        self.assertEqual(tuple(coordinate.shape), (1, 21, 8, 8, 1536))
        self.assertEqual(coordinate.dtype, torch.float32)
        self.assertEqual(
            cpmr.tensor_sha256(coordinate),
            cpmr.CANONICAL_COORDINATE_TENSOR_SHA256,
        )
        self.assertEqual(
            cpmr.coordinate_tensor_sha256(),
            "de31a70bc0c62d4764bc5025f508805d030bd25207c05986a2b4372d9eb52861",
        )
        # Every axis begins with interleaved sin(0), cos(0).
        self.assertEqual(coordinate[0, 0, 0, 0, 0].item(), 0.0)
        self.assertEqual(coordinate[0, 0, 0, 0, 1].item(), 1.0)
        self.assertEqual(coordinate[0, 0, 0, 0, 512].item(), 0.0)
        self.assertEqual(coordinate[0, 0, 0, 0, 513].item(), 1.0)
        self.assertEqual(coordinate[0, 0, 0, 0, 1024].item(), 0.0)
        self.assertEqual(coordinate[0, 0, 0, 0, 1025].item(), 1.0)
        mutated = coordinate.clone()
        mutated[0, 0, 0, 0, 0] = 9.0
        self.assertEqual(cpmr.fixed_3d_coordinate_encoding()[0, 0, 0, 0, 0], 0.0)

    def test_proposal_patch_reshape_is_exact_and_never_interpolates(self) -> None:
        packed = torch.arange(1536, dtype=torch.float32).to(torch.float16)
        packed = packed.reshape(1, 1, 1536)
        packed = packed.expand(1, 21 * 31 * 30, 1536)
        field = cpmr.reshape_proposal_patch_tokens(packed)
        self.assertEqual(tuple(field.shape), (1, 21, 31, 30, 1536))
        self.assertEqual(field.dtype, torch.float16)
        self.assertTrue(torch.equal(field[0, 7, 12, 19], packed[0, 0]))
        channel_first = field.permute(0, 4, 1, 2, 3)
        round_trip = cpmr.reshape_proposal_patch_tokens(channel_first)
        self.assertTrue(torch.equal(round_trip, field))
        with self.assertRaisesRegex(cpmr.CPMRTensorContractError, "contract"):
            cpmr.reshape_proposal_patch_tokens(torch.zeros(1, 41, 8, 8, 1536))

    def test_builder_matches_fp32_difference_pool_normalize_equations(self) -> None:
        result = self.base
        self.assertEqual(tuple(result.carrier_fp32.shape), (1, 21, 8, 8, 1536))
        self.assertEqual(tuple(result.flattened().shape), (1, 1344, 1536))
        self.assertEqual(result.carrier_fp32.dtype, torch.float32)
        self.assertEqual(result.bfloat16().dtype, torch.bfloat16)
        self.assertEqual(
            result.activity.tolist(), [[False] + [True] * 20]
        )
        self.assertEqual(torch.count_nonzero(result.carrier_fp32[:, 0]).item(), 0)
        self.assertEqual(torch.count_nonzero(result.clipped_content[:, 0]).item(), 0)

        phase_index = 7
        increment = (
            self.action_compact[0, phase_index, 0, 0]
            - self.action_compact[0, phase_index - 1, 0, 0]
        ).float()
        expected_rms = increment.square().mean().sqrt()
        expected_content = increment / max(expected_rms.item(), 1.0e-6)
        self.assertTrue(
            torch.allclose(
                result.phase_rms[0, phase_index], expected_rms, atol=1.0e-6, rtol=1.0e-6
            )
        )
        self.assertTrue(
            torch.allclose(
                result.clipped_content[0, phase_index, 3, 5],
                expected_content,
                atol=2.0e-6,
                rtol=2.0e-6,
            )
        )
        expected_carrier = expected_content + 0.02 * result.coordinate_encoding[
            0, phase_index, 3, 5
        ]
        self.assertTrue(
            torch.allclose(
                result.carrier_fp32[0, phase_index, 3, 5],
                expected_carrier,
                atol=2.0e-6,
                rtol=2.0e-6,
            )
        )

    def test_activity_is_exact_nonzero_not_an_epsilon_threshold(self) -> None:
        pooled = torch.zeros(1, 21, 8, 8, 1536, dtype=torch.float32)
        pooled[0, 3, 0, 0, 0] = torch.finfo(torch.float32).tiny
        normalized = cpmr.normalize_and_clip_pooled_increments(pooled)
        self.assertTrue(normalized.activity[0, 3].item())
        self.assertFalse(normalized.activity[0, 2].item())
        self.assertEqual(torch.count_nonzero(normalized.clipped_content[0, 2]).item(), 0)
        self.assertEqual(normalized.phase_rms[0, 2].item(), 0.0)

    def test_token_rms_is_clipped_at_four_before_coordinates(self) -> None:
        pooled = torch.zeros(1, 21, 8, 8, 1536, dtype=torch.float32)
        pooled[:, 1].fill_(1.0)
        pooled[:, 1, 0, 0].fill_(100.0)
        normalized = cpmr.normalize_and_clip_pooled_increments(pooled)
        self.assertGreater(normalized.token_rms_before_clip[0, 1, 0, 0].item(), 4.0)
        clipped_rms = normalized.clipped_content.square().mean(dim=-1).sqrt()
        self.assertLessEqual(clipped_rms.max().item(), 4.0 + 1.0e-5)
        self.assertAlmostEqual(clipped_rms[0, 1, 0, 0].item(), 4.0, places=5)
        self.assertGreater(normalized.clip_fraction[0, 1].item(), 0.0)
        self.assertEqual(normalized.clip_fraction[0, 0].item(), 0.0)

    def test_controls_reject_forged_or_tampered_base_dataclasses(self) -> None:
        forged_mapping = replace(
            self.base, source_phase_indices=cpmr.REVERSE_SOURCE_PHASES
        )
        with self.assertRaisesRegex(cpmr.CPMRTensorContractError, "identity"):
            cpmr.build_reverse_control(forged_mapping)

        forged_activity = replace(
            self.base, activity=self.base.activity.to(dtype=torch.float32)
        )
        with self.assertRaisesRegex(cpmr.CPMRTensorContractError, "activity dtype"):
            cpmr.build_reverse_control(forged_activity)

        forged_content = self.base.clipped_content.clone()
        forged_content[:, 0, 0, 0, 0] = 1.0
        with self.assertRaisesRegex(
            cpmr.CPMRTensorContractError, "activity|exact"
        ):
            cpmr.build_reverse_control(
                replace(self.base, clipped_content=forged_content)
            )

        forged_metric = replace(self.base, phase_rms=self.base.phase_rms.double())
        with self.assertRaisesRegex(cpmr.CPMRTensorContractError, "phase_rms dtype"):
            cpmr.build_reverse_control(forged_metric)

        forged_scale = self.base.clip_scale.clone()
        forged_scale[:, 7, 0, 0] += 0.25
        with self.assertRaisesRegex(cpmr.CPMRTensorContractError, "clipping rule"):
            cpmr.build_reverse_control(
                replace(self.base, clip_scale=forged_scale)
            )

        forged_coordinate = self.base.coordinate_encoding.clone()
        forged_coordinate[:, 7, 0, 0, 0] += 1.0
        with self.assertRaisesRegex(cpmr.CPMRTensorContractError, "coordinate"):
            cpmr.build_reverse_control(
                replace(self.base, coordinate_encoding=forged_coordinate)
            )

        forged_carrier = self.base.carrier_fp32.clone()
        forged_carrier[:, 7, 0, 0, 0] += 1.0
        with self.assertRaisesRegex(cpmr.CPMRTensorContractError, "byte-exact"):
            cpmr.build_reverse_control(
                replace(self.base, carrier_fp32=forged_carrier)
            )

        # Numeric equality is insufficient: signed zero changes the receipt bytes.
        signed_zero_carrier = self.base.carrier_fp32.clone()
        signed_zero_carrier[:, 0].fill_(-0.0)
        self.assertTrue(torch.equal(signed_zero_carrier, self.base.carrier_fp32))
        self.assertNotEqual(
            cpmr.tensor_sha256(signed_zero_carrier),
            cpmr.tensor_sha256(self.base.carrier_fp32),
        )
        with self.assertRaisesRegex(cpmr.CPMRTensorContractError, "byte-exact"):
            cpmr.build_reverse_control(
                replace(self.base, carrier_fp32=signed_zero_carrier)
            )

    def test_reverse_moves_activity_and_content_but_not_coordinates(self) -> None:
        reverse = cpmr.build_reverse_control(self.base)
        self.assertEqual(reverse.source_phase_indices, cpmr.REVERSE_SOURCE_PHASES)
        self.assertTrue(torch.equal(reverse.coordinate_encoding, self.base.coordinate_encoding))
        for destination, source in enumerate(cpmr.REVERSE_SOURCE_PHASES):
            self.assertTrue(
                torch.equal(
                    reverse.clipped_content[:, destination],
                    self.base.clipped_content[:, source],
                )
            )
            self.assertTrue(
                torch.equal(reverse.activity[:, destination], self.base.activity[:, source])
            )
        destination, source = 1, 20
        expected = self.base.clipped_content[:, source] + 0.02 * self.base.coordinate_encoding[
            :, destination
        ]
        self.assertTrue(torch.equal(reverse.carrier_fp32[:, destination], expected))
        self.assertFalse(
            torch.equal(reverse.carrier_fp32[:, destination], self.base.carrier_fp32[:, source])
        )

    def test_shuffle_uses_frozen_permutation_and_fixed_destination_coordinates(self) -> None:
        shuffled = cpmr.build_shuffle_control(self.base)
        self.assertEqual(shuffled.source_phase_indices, cpmr.SHUFFLE_SOURCE_PHASES)
        self.assertEqual(
            shuffled.digest_manifest()["phase_mapping_sha256"],
            cpmr.SHUFFLE_CANONICAL_SHA256,
        )
        self.assertTrue(torch.equal(shuffled.coordinate_encoding, self.base.coordinate_encoding))
        for destination, source in enumerate(cpmr.SHUFFLE_SOURCE_PHASES):
            self.assertTrue(
                torch.equal(
                    shuffled.clipped_content[:, destination],
                    self.base.clipped_content[:, source],
                )
            )

    def test_negative_negates_only_content_and_preserves_activity_coordinates(self) -> None:
        negative = cpmr.build_negative_control(self.base)
        self.assertTrue(torch.equal(negative.activity, self.base.activity))
        self.assertTrue(torch.equal(negative.coordinate_encoding, self.base.coordinate_encoding))
        self.assertTrue(torch.equal(negative.clipped_content, -self.base.clipped_content))
        expected = torch.where(
            self.base.activity[:, :, None, None, None],
            -self.base.clipped_content + 0.02 * self.base.coordinate_encoding,
            torch.zeros_like(self.base.carrier_fp32),
        )
        self.assertTrue(torch.equal(negative.carrier_fp32, expected))

    def test_nn_is_bit_exact_zero_through_every_gated_stage(self) -> None:
        nn = cpmr.build_nn_control(self.noop, self.noop)
        self.assertEqual(nn.control, "nn")
        self.assertFalse(nn.activity.any().item())
        self.assertEqual(torch.count_nonzero(nn.clipped_content).item(), 0)
        self.assertEqual(torch.count_nonzero(nn.carrier_fp32).item(), 0)
        self.assertEqual(torch.count_nonzero(nn.phase_rms).item(), 0)
        self.assertEqual(torch.count_nonzero(nn.clip_fraction).item(), 0)
        self.assertEqual(nn.activity.tolist(), [[False] * 21])
        nn_bfloat16 = nn.bfloat16()
        self.assertEqual(torch.count_nonzero(nn_bfloat16).item(), 0)
        self.assertEqual(
            torch.count_nonzero(nn_bfloat16.contiguous().view(torch.uint8)).item(),
            0,
        )

        base_bfloat16_phase_zero = self.base.bfloat16()[:, 0].contiguous()
        self.assertEqual(torch.count_nonzero(base_bfloat16_phase_zero).item(), 0)
        self.assertEqual(
            torch.count_nonzero(base_bfloat16_phase_zero.view(torch.uint8)).item(),
            0,
        )

    def test_receipt_records_required_fp32_then_bfloat16_audit(self) -> None:
        receipt = self.base.audit_receipt()
        self.assertEqual(receipt["shape"], [1, 21, 8, 8, 1536])
        self.assertEqual(receipt["flattened_shape"], [1, 1344, 1536])
        self.assertEqual(receipt["activity_bitset"], ["0" + "1" * 20])
        self.assertEqual(len(receipt["phase_rms_fp32"][0]), 21)
        self.assertEqual(
            receipt["phase_rms_fp32_semantics"],
            "pooled_increment_pre_normalization",
        )
        self.assertEqual(
            receipt["phase_rms_fp32"],
            receipt["pooled_increment_phase_rms_fp32"],
        )
        for name in (
            "pooled_increment_phase_rms_fp32",
            "clipped_content_phase_rms_fp32",
            "final_carrier_phase_rms_fp32",
        ):
            self.assertEqual(len(receipt[name]), 1)
            self.assertEqual(len(receipt[name][0]), 21)
        self.assertEqual(len(receipt["clip_fraction"][0]), 21)
        digests = receipt["digests"]
        self.assertEqual(
            digests["coordinate_encoding_fp32_sha256"],
            cpmr.CANONICAL_COORDINATE_TENSOR_SHA256,
        )
        self.assertNotEqual(
            digests["carrier_fp32_sha256"], digests["carrier_bfloat16_sha256"]
        )
        self.assertEqual(
            digests["phase_rms_fp32_sha256"],
            cpmr.tensor_sha256(self.base.phase_rms),
        )
        self.assertEqual(
            digests["pooled_increment_phase_rms_fp32_sha256"],
            digests["phase_rms_fp32_sha256"],
        )
        self.assertEqual(
            digests["token_rms_before_clip_fp32_sha256"],
            cpmr.tensor_sha256(self.base.token_rms_before_clip),
        )
        self.assertEqual(
            digests["clip_scale_fp32_sha256"],
            cpmr.tensor_sha256(self.base.clip_scale),
        )
        self.assertEqual(
            digests["clip_fraction_fp32_sha256"],
            cpmr.tensor_sha256(self.base.clip_fraction),
        )

        clipped_phase_rms = self.base.clipped_content.square().mean(
            dim=(2, 3, 4)
        ).sqrt()
        final_phase_rms = self.base.carrier_fp32.square().mean(
            dim=(2, 3, 4)
        ).sqrt()
        self.assertEqual(
            digests["clipped_content_phase_rms_fp32_sha256"],
            cpmr.tensor_sha256(clipped_phase_rms),
        )
        self.assertEqual(
            digests["final_carrier_phase_rms_fp32_sha256"],
            cpmr.tensor_sha256(final_phase_rms),
        )
        self.assertTrue(
            torch.equal(
                torch.tensor(receipt["clipped_content_phase_rms_fp32"]),
                clipped_phase_rms.cpu(),
            )
        )
        self.assertTrue(
            torch.equal(
                torch.tensor(receipt["final_carrier_phase_rms_fp32"]),
                final_phase_rms.cpu(),
            )
        )
        for digest in digests.values():
            self.assertEqual(len(digest), 64)
            int(digest, 16)

    def test_wrong_shapes_dtypes_phase_zero_and_nonfinite_fail_closed(self) -> None:
        with self.assertRaisesRegex(cpmr.CPMRTensorContractError, "float32"):
            cpmr.normalize_and_clip_pooled_increments(
                torch.zeros(1, 21, 8, 8, 1536, dtype=torch.float16)
            )
        wrong = torch.zeros(1, 20, 8, 8, 1536, dtype=torch.float32)
        with self.assertRaisesRegex(cpmr.CPMRTensorContractError, "must be"):
            cpmr.normalize_and_clip_pooled_increments(wrong)
        phase_zero = torch.zeros(1, 21, 8, 8, 1536, dtype=torch.float32)
        phase_zero[:, 0, 0, 0, 0] = 1.0
        with self.assertRaisesRegex(cpmr.CPMRTensorContractError, r"phase 0|P\[0\]"):
            cpmr.normalize_and_clip_pooled_increments(phase_zero)
        nonfinite = torch.zeros(1, 21, 8, 8, 1536, dtype=torch.float32)
        nonfinite[:, 1, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(cpmr.CPMRTensorContractError, "NaN"):
            cpmr.normalize_and_clip_pooled_increments(nonfinite)


if __name__ == "__main__":
    unittest.main()
