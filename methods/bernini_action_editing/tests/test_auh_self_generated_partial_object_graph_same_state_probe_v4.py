from __future__ import annotations

import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from methods.bernini_action_editing import (
    auh_self_generated_partial_object_graph_same_state_probe_v4 as probe,
)


class AUHPartialObjectGraphSameStateProbeV4StaticTests(unittest.TestCase):
    def test_contract_has_explicit_frozen_base_and_exact_native_geometry(self) -> None:
        value = probe.probe_contract()
        self.assertEqual(
            value["frozen_base_arm"],
            "B0_FROZEN_BASE_OBSERVER_ABSENT",
        )
        self.assertEqual(value["frozen_base_cell_count"], 9)
        self.assertTrue(value["frozen_base_per_appearance_sigma"])
        self.assertFalse(value["frozen_base_graph_observation_supplied"])
        self.assertEqual(value["capture_count"], 144)
        self.assertEqual(value["native_raw_qk_shape"], [1, 21, 925, 12, 128])
        self.assertEqual(value["native_role_proxy_shape"], [1, 21, 5, 925])

    def test_roles_edges_and_per_arm_residency_are_explicit(self) -> None:
        value = probe.probe_contract()
        self.assertEqual(
            [row["role_id"] for row in value["roles_explicit"]],
            [
                "agent",
                "moving_object",
                "start_support",
                "end_support",
                "null_context",
            ],
        )
        self.assertEqual(len(value["edges_explicit"]), 4)
        self.assertTrue(value["per_arm_immediate_reduce_and_zeroize_required"])
        self.assertEqual(
            value["maximum_simultaneously_resident_raw_prompt_arms"], 1
        )
        self.assertFalse(value["four_arm_raw_qk_bundle_permitted"])

    def test_missing_control_cannot_admit_representation(self) -> None:
        value = probe.probe_contract()
        self.assertFalse(value["shuffled_prompt_control_executed"])
        self.assertFalse(value["shuffled_prompt_robustness_claimed"])
        self.assertTrue(
            value["missing_shuffled_prompt_gate_counts_as_representation_failure"]
        )
        self.assertFalse(
            value["component_admission_can_imply_representation_admission"]
        )
        self.assertTrue(value["representation_admission_hard_false"])
        self.assertFalse(value["scientific_claim_authorized"])
        self.assertFalse(value["stable_transferable_action_representation_claimed"])

    def test_absolute_and_shared_frame_gates_are_launch_blocking_contract(self) -> None:
        value = probe.probe_contract()
        self.assertTrue(value["native_proxy_simplex_required"])
        self.assertTrue(value["absolute_evidence_before_zscore_or_topk"])
        self.assertTrue(value["failed_absolute_evidence_kernel_exact_zero"])
        self.assertEqual(value["shared_frame_sources"], ["noop", "static"])
        self.assertFalse(value["action_or_reverse_defines_shared_frame"])
        self.assertTrue(value["failed_shared_frame_phase_abstains_all_four_arms"])
        self.assertTrue(value["four_arm_common_edge_domain_required"])
        self.assertTrue(value["reverse_endpoint_topology_gate_required"])
        self.assertTrue(value["observer_only_diagnostic_launch_authorized"])
        self.assertFalse(value["launch_blocked_pending_failure_path_audit"])
        self.assertFalse(value["representation_or_renderer_launch_authorized"])
        self.assertTrue(
            value["explicit_capture_ownership_boundary_exception_scrub_required"]
        )
        self.assertTrue(
            value["uncovered_exception_requires_nonzero_exit_without_receipt"]
        )
        self.assertFalse(value["all_allocation_failure_zeroization_claimed"])

    def test_failure_path_finally_scrubs_native_capture_group(self) -> None:
        graph_registry = probe.make_graph_registry()
        partition = probe.attention_hook.ExhaustiveTextRolePartition(
            graph_registry.role_ids, (0, 1, 2, 3) + (4,) * 508
        )

        class Capture:
            query = torch.ones((1, 1))
            key = torch.ones((1, 1))
            derived_qk_role_responsibility_proxy = torch.ones((1, 1))

        class Bank:
            calls = 0

            def zeroize(self, rows):
                self.calls += 1
                for row in rows:
                    row.query.zero_()
                    row.key.zero_()
                    row.derived_qk_role_responsibility_proxy.zero_()

        capture = Capture()
        bank = Bank()
        with self.assertRaises(probe.AUHPartialObjectGraphProbeV4Error):
            probe._reduce_native_capture_group_finally_scrubbed(
                captures=(capture,),
                native_bank=bank,
                invocation=None,
                authority=None,
                arm="action",
                partition=partition,
                assemblers={},
                graph_registry=graph_registry,
            )
        self.assertEqual(bank.calls, 1)
        self.assertEqual(int(torch.count_nonzero(capture.query)), 0)

    def test_partial_native_commit_fault_scrubs_bank_and_pending_compacts(self):
        graph_registry = probe.make_graph_registry()
        partition = probe.attention_hook.ExhaustiveTextRolePartition(
            graph_registry.role_ids, (0, 1, 2, 3) + (4,) * 508
        )
        sigma = probe.native.SigmaCell("mid", 18, 0.55)
        runtime = SimpleNamespace(
            source_geometry=SimpleNamespace(height=37, width=25)
        )
        authority = SimpleNamespace(
            appearance_id="appearance_0",
            sigma_cell=sigma,
            state_tensor_sha256={
                "noisy_latents": "1" * 64,
                "timesteps": "2" * 64,
                "rotary_embs": "3" * 64,
            },
            call=lambda _arm: torch.ones((1,), dtype=torch.float32),
        )

        class RankBank:
            @contextmanager
            def observe(self, _invocation):
                yield

            def take_rank(self, _invocation):
                return ()

        class Capture:
            def __init__(self):
                self.block_index = 6
                self.query = torch.ones((2, 2))
                self.key = torch.ones((2, 2))
                self.derived_qk_role_responsibility_proxy = torch.ones((2, 2))

            def zeroize(self):
                self.query.zero_()
                self.key.zero_()
                self.derived_qk_role_responsibility_proxy.zero_()

        class NativeBank:
            def __init__(self):
                self._captures = {}
                self.consumed_count = 0
                self.zeroized_count = 0

            def zeroize(self, rows):
                for row in rows:
                    row.zeroize()
                    self.zeroized_count += 1

        class Assembler:
            def __init__(self):
                self.abort_calls = 0

            def abort(self):
                self.abort_calls += 1

        native_bank = NativeBank()
        raw = Capture()
        assemblers = {block: Assembler() for block in probe.native.BLOCKS}

        def partial_commit(**kwargs):
            invocation = kwargs["invocation"]
            native_bank._captures[invocation.key] = {6: raw}
            raise RuntimeError("fault after partial native commit")

        with mock.patch.object(
            probe.attention_hook,
            "commit_world4_shards_to_native_bank",
            side_effect=partial_commit,
        ):
            with self.assertRaisesRegex(RuntimeError, "partial native commit"):
                probe._capture_one_arm(
                    runtime=runtime,
                    authority=authority,
                    arm="action",
                    partition=partition,
                    rank_bank=RankBank(),
                    native_bank=native_bank,
                    assemblers=assemblers,
                    graph_registry=graph_registry,
                    rank=0,
                )
        self.assertFalse(native_bank._captures)
        self.assertEqual(native_bank.consumed_count, 1)
        self.assertEqual(native_bank.zeroized_count, 1)
        self.assertEqual(int(torch.count_nonzero(raw.query)), 0)
        self.assertTrue(all(row.abort_calls == 1 for row in assemblers.values()))

    def test_no_target_training_renderer_decode_or_route(self) -> None:
        value = probe.probe_contract()
        self.assertFalse(value["target_inputs_consumed"])
        self.assertFalse(value["entrypoint_exercised_call_path_decoder_called"])
        self.assertTrue(value["claims_apply_to_exercised_entrypoint_call_path_only"])
        self.assertFalse(value["renderer_called"])
        self.assertFalse(value["optimizer_created"])
        self.assertEqual(value["parameter_updates"], 0)
        self.assertFalse(value["route_or_injection_called"])
        source = Path(probe.__file__).read_text(encoding="utf-8")
        calls = [
            node.func.attr
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        ]
        self.assertNotIn("decode", calls)
        self.assertIn('output.open("x"', source)
        self.assertNotIn("target_video=", source)

    def test_forward_counts_and_source_manifest_close(self) -> None:
        self.assertEqual(probe.EXPECTED_TRAJECTORY_FORWARDS, 240)
        self.assertEqual(probe.EXPECTED_TRAJECTORY_STEPS, 120)
        self.assertEqual(probe.EXPECTED_FROZEN_BASE_CELLS, 9)
        self.assertEqual(probe.EXPECTED_OBSERVER_FORWARDS, 36)
        self.assertEqual(probe.EXPECTED_TOTAL_FORWARDS, 285)
        manifest = probe.source_manifest()
        self.assertEqual(manifest["file_count"], 12)
        self.assertEqual(manifest["file_count"], len(manifest["files"]))
        self.assertTrue(manifest["all_plain_nonsymlink_files"])
        self.assertTrue(
            all(
                len(row["sha256"]) == 64 and row["bytes"] > 0
                for row in manifest["files"]
            )
        )


if __name__ == "__main__":
    unittest.main()
