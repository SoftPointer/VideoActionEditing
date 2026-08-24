#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
from pathlib import Path
import sys
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

if _TORCH_AVAILABLE:
    # Once torch exists, a broken field14/dependency import is a test error,
    # never a reason to silently skip the suite.
    import graft_phase_a_field14_exact40_v1 as field14
    import run_graft_phase_a_field14_exact40_gpu_v1 as gpu_runner
else:
    field14 = None  # type: ignore[assignment]
    gpu_runner = None  # type: ignore[assignment]


@unittest.skipUnless(_TORCH_AVAILABLE, "torch runtime is required")
class Field14Exact40CoreTests(unittest.TestCase):
    family = "dog"
    confirmation_iid = "841b5e0080a1441d"
    confirmation_source_sha256 = "1" * 64
    wrong_owner_iid = "7b88a1ca1f804f41"
    wrong_owner_source_sha256 = "2" * 64

    def _packet(self, index: int):
        def identity(digest, *, shape, dtype, device):
            element_bytes = {
                "torch.bfloat16": 2,
                "torch.float32": 4,
                "torch.int64": 8,
                "torch.complex128": 16,
            }[dtype]
            elements = 1
            for item in shape:
                elements *= item
            return {
                "shape": list(shape),
                "dtype": dtype,
                "device_type_at_observation": device,
                "finite": True,
                "byte_count": elements * element_bytes,
                "raw_sha256": digest,
                "content_sha256": digest,
            }

        fields = {
            name: (
                torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 2, 2)
                + float(index * 10 + ordinal)
            ).contiguous()
            for ordinal, name in enumerate(field14.FIELD_ROLES)
        }
        gate = field14.expected_route_gate(index)
        state = {
            "confirmation_source_zs": identity(
                "a" * 64,
                shape=[1, 16, 6, 4, 4],
                dtype="torch.float32",
                device="cuda",
            ),
            "epsilon": identity(
                "b" * 64,
                shape=[1, 16, 6, 4, 4],
                dtype="torch.float32",
                device="cuda",
            ),
            "noisy_target_x_sigma": identity(
                "c" * 64,
                shape=[1, 16, 6, 4, 4],
                dtype="torch.float32",
                device="cuda",
            ),
            "native_visual_pack": identity(
                "d" * 64,
                shape=[1, 16, 1536],
                dtype="torch.bfloat16",
                device="cuda",
            ),
            "native_rotary_pack": identity(
                "e" * 64,
                shape=[1, 1, 16, 64],
                dtype="torch.complex128",
                device="cuda",
            ),
            "sigma": identity(
                "f" * 64, shape=[], dtype="torch.float32", device="cpu"
            ),
            "timestep": identity(
                "0" * 64, shape=[1], dtype="torch.int64", device="cuda"
            ),
            "negative_condition": identity(
                "3" * 64,
                shape=[1, 512, 4096],
                dtype="torch.bfloat16",
                device="cuda",
            ),
            "noop_positive_condition": identity(
                "4" * 64,
                shape=[1, 512, 4096],
                dtype="torch.bfloat16",
                device="cuda",
            ),
            "action_positive_condition": identity(
                "5" * 64,
                shape=[1, 512, 4096],
                dtype="torch.bfloat16",
                device="cuda",
            ),
        }
        atlas = {
            "correct_confirmation_atlas": identity(
                "6" * 64,
                shape=[1, 32, 1536],
                dtype="torch.float32",
                device="cuda",
            ),
            "wrong_same_family_fit_atlas": identity(
                "7" * 64,
                shape=[1, 32, 1536],
                dtype="torch.float32",
                device="cuda",
            ),
        }

        def route(name):
            dropped = name.startswith("drop_")
            wrong = name.startswith("wrong_")
            return field14.seal_mapping(
                {
                    "branch_name": "V",
                    "total_tokens": 8,
                    "condition_tokens": 4,
                    "target_tokens": 4,
                    "sequence_parallel_rank": 0,
                    "sequence_parallel_size": 4,
                    "sigma_hex": field14.sigma_strata.PINNED_POSITIVE_SIGMAS[
                        index
                    ].hex(),
                    "enabled": not dropped,
                    "gate_hex": (0.0 if dropped else gate).hex(),
                    "atlas_receipt_digest": (
                        None if dropped else ("7" * 64 if wrong else "6" * 64)
                    ),
                    "source_memory_owned_by_V_VI_only": True,
                }
            )

        raw_hashes = {
            "correct_negative": "8" * 64,
            "correct_noop": "9" * 64,
            "correct_action": "a" * 64,
            "wrong_negative": "8" * 64 if index < 26 else "b" * 64,
            "wrong_noop": "9" * 64 if index < 26 else "c" * 64,
            "drop_negative": "8" * 64 if index < 26 else "d" * 64,
            "drop_noop": "9" * 64 if index < 26 else "e" * 64,
            "drop_action": "a" * 64 if index < 26 else "f" * 64,
        }
        raw_identities = {
            name: {
                "shape": [1, 2],
                "dtype": "torch.bfloat16",
                "device_type_at_observation": "cuda",
                "finite": True,
                "byte_count": 4,
                "raw_sha256": digest,
                "content_sha256": digest,
            }
            for name, digest in raw_hashes.items()
        }
        inactive_parity = None
        if index in field14.INACTIVE_INDICES:
            inactive_parity = dict(
                field14.seal_mapping(
                    {
                        "schedule_index": index,
                        "route_gate_float64_hex": 0.0.hex(),
                        "correct_wrong_drop_negative_raw_byte_exact": True,
                        "correct_wrong_drop_noop_raw_byte_exact": True,
                        "correct_drop_action_raw_byte_exact": True,
                        "all_same_condition_raw_equal_preinstall": True,
                        "source_noise_xsigma_vpack_rotary_timestep_conditions_equal_preinstall": True,
                        "preinstall_row_sha256": {
                            "negative": raw_hashes["correct_negative"],
                            "noop_positive": raw_hashes["correct_noop"],
                            "action_positive": raw_hashes["correct_action"],
                        },
                    }
                )
            )
        provenance = field14.build_field14_provenance(
            schedule_index=index,
            family=self.family,
            confirmation_iid=self.confirmation_iid,
            confirmation_source_sha256=self.confirmation_source_sha256,
            wrong_owner_iid=self.wrong_owner_iid,
            wrong_owner_source_sha256=self.wrong_owner_source_sha256,
            fields=fields,
            runtime_evidence={
                "confirmation_source_state_receipt_digest": "d" * 64,
                "wrong_fit_source_state_receipt_digest": "e" * 64,
                "epsilon_receipt_digest": "f" * 64,
                "noisy_target_receipt_digest": "0" * 64,
                "raw_tensor_identities": raw_identities,
                "raw_call_order": list(field14.RAW_ROLES),
                "route_receipts": {
                    name: dict(route(name)) for name in field14.RAW_ROLES
                },
                "coordinate": dict(
                    field14.seal_mapping(
                        {
                            "schedule_index": index,
                            "timestep": field14.sigma_strata.PINNED_TIMESTEPS[index],
                            "sigma_float32_be_hex": (
                                field14.sigma_strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[
                                    index
                                ]
                            ),
                            "schedule_sha256": field14.sigma_strata.SCHEDULE_SHA256,
                            "scheduler_step_called": False,
                        }
                    )
                ),
                "same_state_identities_before_model_fields": state,
                "same_state_identities_after_all_model_fields": dict(state),
                "atlas_identities_before_model_fields": atlas,
                "atlas_identities_after_all_model_fields": dict(atlas),
                "same_state_tensor_identities_recomputed_byte_equal": True,
                "wrong_route_receipts_differ_only_in_atlas_memory": True,
                "drop_route_receipts_retain_v_branch_disable_only_rebinder": True,
                "action_noop_route_receipts_equal_with_negative_raw_reuse": True,
                "expected_enabled_route_gate_float64_hex_recomputed": gate.hex(),
                "inactive_raw_parity": inactive_parity,
                "ambient_torch_no_grad": True,
            },
        )
        return field14.Field14TensorSet(**fields, provenance=provenance)

    def _release(self, index: int):
        return field14.build_release_receipt(index, cuda_cache_requested=True)

    def test_exact40_order_regimes_hash_and_release(self) -> None:
        seen = []

        def measure(index):
            seen.append(index)
            return self._packet(index)

        with torch.no_grad():
            result = field14.execute_exact40_sweep(
                family=self.family,
                confirmation_iid=self.confirmation_iid,
                confirmation_source_sha256=self.confirmation_source_sha256,
                wrong_owner_iid=self.wrong_owner_iid,
                wrong_owner_source_sha256=self.wrong_owner_source_sha256,
                short_result_digest="3" * 64,
                preinstall_baseline_digest="4" * 64,
                measure_index=measure,
                release_index=self._release,
            )
        self.assertEqual(seen, list(range(40)))
        self.assertEqual(result["schedule_indices"], list(range(40)))
        self.assertEqual(len(result["rows"]), 40)
        self.assertTrue(result["one_index_admitted_hashed_and_released_before_next"])
        self.assertFalse(result["cross_index_tensor_retention"])
        self.assertFalse(result["checkpoint_written"])
        self.assertFalse(result["checkpoint_payload_returned"])
        self.assertFalse(result["publication_performed"])
        for row in result["rows"]:
            self.assertEqual(len(row["field_tensor_sha256"]), 6)
            self.assertTrue(row["all_field_tensor_objects_released"])

    def test_exact_gate_partition_is_index_0_to_25_then_26_to_39(self) -> None:
        for index in range(26):
            self.assertGreaterEqual(
                field14.sigma_strata.PINNED_POSITIVE_SIGMAS[index], 0.75
            )
            self.assertEqual(field14.expected_route_gate(index), 0.0)
        for index in range(26, 40):
            gate = field14.expected_route_gate(index)
            self.assertLess(field14.sigma_strata.PINNED_POSITIVE_SIGMAS[index], 0.75)
            self.assertGreater(gate, 0.0)

    def test_alias_and_provenance_regime_attack_fail_closed(self) -> None:
        packet = self._packet(26)
        alias = replace(
            packet,
            wrong_atlas_noop_velocity=packet.correct_atlas_noop_velocity,
        )
        with self.assertRaisesRegex(field14.Field14Exact40Error, "alias"):
            field14.admit_field14_tensor_set(
                alias,
                schedule_index=26,
                family=self.family,
                confirmation_iid=self.confirmation_iid,
                confirmation_source_sha256=self.confirmation_source_sha256,
                wrong_owner_iid=self.wrong_owner_iid,
                wrong_owner_source_sha256=self.wrong_owner_source_sha256,
            )

        attacked = dict(packet.provenance)
        attacked.pop("digest")
        attacked["active_finite_nonzero_gate_provenance_verified"] = False
        attacked = field14.seal_mapping(attacked)
        with self.assertRaisesRegex(field14.Field14Exact40Error, "regime"):
            field14.admit_field14_tensor_set(
                replace(packet, provenance=attacked),
                schedule_index=26,
                family=self.family,
                confirmation_iid=self.confirmation_iid,
                confirmation_source_sha256=self.confirmation_source_sha256,
                wrong_owner_iid=self.wrong_owner_iid,
                wrong_owner_source_sha256=self.wrong_owner_source_sha256,
            )

    def test_retained_tensor_fails_before_next_index(self) -> None:
        retained = []

        def measure(index):
            packet = self._packet(index)
            retained.append(packet.correct_atlas_noop_velocity)
            return packet

        with torch.no_grad(), self.assertRaisesRegex(
            field14.Field14Exact40Error, "retained"
        ):
            field14.execute_exact40_sweep(
                family=self.family,
                confirmation_iid=self.confirmation_iid,
                confirmation_source_sha256=self.confirmation_source_sha256,
                wrong_owner_iid=self.wrong_owner_iid,
                wrong_owner_source_sha256=self.wrong_owner_source_sha256,
                short_result_digest="3" * 64,
                preinstall_baseline_digest="4" * 64,
                measure_index=measure,
                release_index=self._release,
            )

    def test_grad_enabled_and_authority_fail_closed(self) -> None:
        with self.assertRaisesRegex(field14.Field14Exact40Error, "no_grad"):
            field14.execute_exact40_sweep(
                family=self.family,
                confirmation_iid=self.confirmation_iid,
                confirmation_source_sha256=self.confirmation_source_sha256,
                wrong_owner_iid=self.wrong_owner_iid,
                wrong_owner_source_sha256=self.wrong_owner_source_sha256,
                short_result_digest="3" * 64,
                preinstall_baseline_digest="4" * 64,
                measure_index=self._packet,
                release_index=self._release,
            )
        packet = self._packet(0)
        attacked = dict(packet.provenance)
        attacked.pop("digest")
        attacked["scientific_success_claimed"] = True
        attacked = field14.seal_mapping(attacked)
        with self.assertRaisesRegex(field14.Field14Exact40Error, "authority"):
            field14.admit_field14_tensor_set(
                replace(packet, provenance=attacked),
                schedule_index=0,
                family=self.family,
                confirmation_iid=self.confirmation_iid,
                confirmation_source_sha256=self.confirmation_source_sha256,
                wrong_owner_iid=self.wrong_owner_iid,
                wrong_owner_source_sha256=self.wrong_owner_source_sha256,
            )

    def test_unknown_claim_distinct_owner_and_raw_schema_fail_closed(self) -> None:
        packet = self._packet(0)

        unknown = dict(packet.provenance)
        unknown.pop("digest")
        unknown["checkpoint_created"] = True
        with self.assertRaisesRegex(field14.Field14Exact40Error, "key schema"):
            field14.admit_field14_tensor_set(
                replace(packet, provenance=field14.seal_mapping(unknown)),
                schedule_index=0,
                family=self.family,
                confirmation_iid=self.confirmation_iid,
                confirmation_source_sha256=self.confirmation_source_sha256,
                wrong_owner_iid=self.wrong_owner_iid,
                wrong_owner_source_sha256=self.wrong_owner_source_sha256,
            )

        authority = dict(packet.provenance)
        authority.pop("digest")
        authority["scientific_success_claimed"] = 1
        with self.assertRaisesRegex(field14.Field14Exact40Error, "authority"):
            field14.admit_field14_tensor_set(
                replace(packet, provenance=field14.seal_mapping(authority)),
                schedule_index=0,
                family=self.family,
                confirmation_iid=self.confirmation_iid,
                confirmation_source_sha256=self.confirmation_source_sha256,
                wrong_owner_iid=self.wrong_owner_iid,
                wrong_owner_source_sha256=self.wrong_owner_source_sha256,
            )

        with self.assertRaisesRegex(field14.Field14Exact40Error, "wrong-owner"):
            field14.admit_field14_tensor_set(
                packet,
                schedule_index=0,
                family=self.family,
                confirmation_iid=self.confirmation_iid,
                confirmation_source_sha256=self.confirmation_source_sha256,
                wrong_owner_iid=self.confirmation_iid,
                wrong_owner_source_sha256=self.confirmation_source_sha256,
            )

        malformed = dict(packet.provenance)
        malformed.pop("digest")
        raw = {name: dict(value) for name, value in malformed["raw_tensor_identities"].items()}
        raw["correct_negative"].pop("finite")
        malformed["raw_tensor_identities"] = raw
        with self.assertRaisesRegex(field14.Field14Exact40Error, "tensor identity"):
            field14.admit_field14_tensor_set(
                replace(packet, provenance=field14.seal_mapping(malformed)),
                schedule_index=0,
                family=self.family,
                confirmation_iid=self.confirmation_iid,
                confirmation_source_sha256=self.confirmation_source_sha256,
                wrong_owner_iid=self.wrong_owner_iid,
                wrong_owner_source_sha256=self.wrong_owner_source_sha256,
            )

        malformed_state = dict(packet.provenance)
        malformed_state.pop("digest")
        before = {
            name: dict(value)
            for name, value in malformed_state[
                "same_state_identities_before_model_fields"
            ].items()
        }
        before["sigma"]["device_type_at_observation"] = "cuda"
        malformed_state["same_state_identities_before_model_fields"] = before
        malformed_state["same_state_identities_after_all_model_fields"] = {
            name: dict(value) for name, value in before.items()
        }
        with self.assertRaisesRegex(field14.Field14Exact40Error, "tensor identity"):
            field14.admit_field14_tensor_set(
                replace(
                    packet,
                    provenance=field14.seal_mapping(malformed_state),
                ),
                schedule_index=0,
                family=self.family,
                confirmation_iid=self.confirmation_iid,
                confirmation_source_sha256=self.confirmation_source_sha256,
                wrong_owner_iid=self.wrong_owner_iid,
                wrong_owner_source_sha256=self.wrong_owner_source_sha256,
            )

        same_atlas = dict(packet.provenance)
        same_atlas.pop("digest")
        atlas_before = {
            name: dict(value)
            for name, value in same_atlas[
                "atlas_identities_before_model_fields"
            ].items()
        }
        atlas_before["wrong_same_family_fit_atlas"] = dict(
            atlas_before["correct_confirmation_atlas"]
        )
        same_atlas["atlas_identities_before_model_fields"] = atlas_before
        same_atlas["atlas_identities_after_all_model_fields"] = {
            name: dict(value) for name, value in atlas_before.items()
        }
        with self.assertRaisesRegex(field14.Field14Exact40Error, "tensor bytes"):
            field14.admit_field14_tensor_set(
                replace(packet, provenance=field14.seal_mapping(same_atlas)),
                schedule_index=0,
                family=self.family,
                confirmation_iid=self.confirmation_iid,
                confirmation_source_sha256=self.confirmation_source_sha256,
                wrong_owner_iid=self.wrong_owner_iid,
                wrong_owner_source_sha256=self.wrong_owner_source_sha256,
            )

    def test_recursive_canonical_json_owns_sealed_mappings(self) -> None:
        nested = field14.seal_mapping(
            {
                "row": field14.seal_mapping({"value": 7}),
                "items": (field14.seal_mapping({"value": 9}),),
            }
        )
        payload = field14.canonical_json_bytes(nested)
        self.assertEqual(payload, field14.canonical_json_bytes(dict(nested)))
        self.assertIn(b'"items":[{"digest":', payload)

    def test_core_contains_no_optimizer_or_artifact_write_path(self) -> None:
        source = inspect.getsource(field14)
        self.assertNotIn("torch.optim", source)
        self.assertNotIn("torch.save", source)
        self.assertNotIn("open(", source)
        self.assertNotIn("write_text", source)
        self.assertNotIn("write_bytes", source)

    def _world8_packets(self, *, attack=None):
        packets = []
        for rank in range(8):
            arm = rank // 4
            short = gpu_runner.short_runner.seal_mapping(
                {"schema_version": "cpu-fake-short", "rank": rank}
            )
            rows = []
            for index in range(40):
                field_hashes = {
                    name: hashlib.sha256(
                        f"arm{arm}:{index}:{name}".encode("ascii")
                    ).hexdigest()
                    for name in field14.FIELD_ROLES
                }
                if attack == "sp_mismatch" and rank == 1 and index == 7:
                    field_hashes[field14.FIELD_ROLES[0]] = "f" * 64
                row = {
                    "schedule_index": index,
                    "admission_digest": hashlib.sha256(
                        f"admission:{arm}:{index}".encode("ascii")
                    ).hexdigest(),
                    "field_tensor_sha256": field_hashes,
                    "semantic_metrics_digest": hashlib.sha256(
                        f"metrics:{arm}:{index}".encode("ascii")
                    ).hexdigest(),
                    "provenance_digest": hashlib.sha256(
                        f"provenance:{rank}:{index}".encode("ascii")
                    ).hexdigest(),
                    "release_digest": hashlib.sha256(
                        f"release:{index}".encode("ascii")
                    ).hexdigest(),
                    "all_field_tensor_objects_released": True,
                }
                if attack == "bad_digest" and rank == 0 and index == 0:
                    row["admission_digest"] = "z" * 64
                if attack == "extra_key" and rank == 0 and index == 0:
                    row["unexpected"] = False
                rows.append(row)
            sweep_plain = {
                "schema_version": field14.SCHEMA_VERSION,
                "status": "completed_in_memory_exact40_no_grad_no_checkpoint",
                "family": gpu_runner.short_runner.FAMILY_BY_DP_ARM[arm],
                "confirmation_iid": gpu_runner.short_runner.CONFIRMATION_IID_BY_DP_ARM[
                    arm
                ],
                "confirmation_source_sha256": ("1" if arm == 0 else "3") * 64,
                "wrong_owner_iid": gpu_runner.short_runner.FIT_IID_BY_DP_ARM[arm],
                "wrong_owner_source_sha256": ("2" if arm == 0 else "4") * 64,
                "short_result_digest": short["digest"],
                "preinstall_baseline_digest": ("5" if arm == 0 else "6") * 64,
                "schedule_indices": list(range(40)),
                "inactive_indices": list(range(26)),
                "active_indices": list(range(26, 40)),
                "field_roles": list(field14.FIELD_ROLES),
                "rows": rows,
                "exact40_official_order": True,
                "ambient_torch_no_grad": True,
                "one_index_admitted_hashed_and_released_before_next": True,
                "cross_index_tensor_retention": False,
                "cross_index_compensation_used": (
                    attack == "missing_false" and rank == 0
                ),
                "cross_index_selection_used": False,
                "semantic_metrics_are_diagnostic_only": True,
                "checkpoint_written": False,
                "checkpoint_payload_returned": False,
                "publication_performed": False,
                **{name: False for name in field14.AUTHORITY_FIELDS},
            }
            sweep = field14.seal_mapping(sweep_plain)
            packets.append(
                {
                    "global_rank": rank,
                    "short_result_digest": short["digest"],
                    "short_result": dict(short),
                    "field14_result_digest": sweep["digest"],
                    "field14_result": dict(sweep),
                    "trainable_sha256_before_sweep": "7" * 64,
                    "trainable_sha256_after_sweep": "7" * 64,
                    "base_sha256_before": "8" * 64,
                    "base_sha256_after": "8" * 64,
                    "checkpoint_written": False,
                    "checkpoint_payload_returned": False,
                    "publication_performed": False,
                    **{name: False for name in field14.AUTHORITY_FIELDS},
                }
            )
        return packets

    def test_world8_assembler_deep_schema_and_sp4_consensus(self) -> None:
        fake_short_world8 = gpu_runner.short_runner.seal_mapping(
            {"schema_version": "cpu-fake-short-world8"}
        )
        with mock.patch.object(
            gpu_runner.short_runner,
            "assemble_world8_local_results",
            return_value=fake_short_world8,
        ), mock.patch.object(
            gpu_runner.short_runner,
            "_assert_no_elevated_authority_or_checkpoint",
            return_value=None,
        ):
            result = gpu_runner._assemble_world8_packets(self._world8_packets())
            self.assertEqual(len(result["all_eight_field14_results"]), 8)
            self.assertEqual(len(result["arm_representatives"]), 2)
            self.assertTrue(result["both_sp4_arms_exact_field_hash_and_metric_consensus"])
            for attack in ("bad_digest", "extra_key", "missing_false", "sp_mismatch"):
                with self.subTest(attack=attack), self.assertRaises(
                    (gpu_runner.Field14GPUError, TypeError)
                ):
                    gpu_runner._assemble_world8_packets(
                        self._world8_packets(attack=attack)
                    )


if __name__ == "__main__":
    unittest.main()
