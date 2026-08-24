from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = METHOD_ROOT / "elal3_c0_v1.py"
CLI_PATH = METHOD_ROOT / "run_elal3_c0_v1.py"
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch
    import elal3_c0_v1 as elal3
    import run_elal3_c0_v1 as runner

    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    elal3 = None  # type: ignore[assignment]
    runner = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


class ELAL3C0StaticTests(unittest.TestCase):
    def test_exact3_sources_parse_and_contract_markers_exist(self) -> None:
        module_source = MODULE_PATH.read_text(encoding="utf-8")
        cli_source = CLI_PATH.read_text(encoding="utf-8")
        ast.parse(module_source)
        ast.parse(cli_source)
        for fragment in (
            "class ELAL3LatentV1",
            "q_camera: torch.Tensor",
            "MEMORY_TOKENS = (ENTITY_SLOTS + RELATION_SLOTS + 1) * LATENT_PHASES",
            "class ELAL3RouteV1",
            "def elal3_checkpoint_context_fn_v1",
            "class ELAL3BlockInjectionV1",
            "for _ in range(BERNINI_BLOCKS)",
            "result[:, selector, :] = delta.to(query_states.dtype)",
        ):
            self.assertIn(fragment, module_source)
        for fragment in (
            "no_relation-w64, full-w64, full-w128",
            "bernini-elal3-c0-paired-initialization-v1",
            '"paired_master_plan_rows"',
            '"paired_active_parameter_mapping"',
            '"frozen_output_encoder_receipt"',
            '"complete_elal3_c0"',
            '"production_elal3_c0_authority": False',
            '"peak_allocated_bytes"',
            '"renderer_block_grad_norms"',
            "os.O_EXCL",
        ):
            self.assertIn(fragment, cli_source)


@unittest.skipUnless(_TORCH_AVAILABLE, "torch runtime is required")
class ELAL3C0FunctionalTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)

    def test_fixed_abi_memory_and_no_relation_mask(self) -> None:
        latent = runner.synthetic_latent(torch.device("cpu"))
        full = elal3.ELAL3ActionMemoryBuilderV1(variant="full")(latent)
        ablated = elal3.ELAL3ActionMemoryBuilderV1(variant="no_relation")(latent)
        self.assertEqual(tuple(full.tokens.shape), (1, 210, 256))
        self.assertEqual(tuple(full.valid.shape), (1, 210))
        self.assertTrue(full.valid[:, 63:189].all())
        self.assertFalse(ablated.valid[:, 63:189].any())
        self.assertEqual(tuple(full.local_tokens.shape), (1, 21, 64))
        self.assertEqual(tuple(latent.q_camera.shape), (1, 21, 128))

    def test_interventions_preserve_rank_and_remap_relations(self) -> None:
        latent = runner.synthetic_latent(torch.device("cpu"))
        reverse = elal3.intervene_elal3_v1(latent, "phase_reverse")
        swapped = elal3.intervene_elal3_v1(latent, "role_slot_swap")
        zero = elal3.intervene_elal3_v1(latent, "zero")
        for value in (reverse, swapped, zero):
            value.validate()
            self.assertEqual(value.q_local.shape, latent.q_local.shape)
            self.assertEqual(value.q_relation.shape, latent.q_relation.shape)
            self.assertTrue(torch.equal(value.q_camera, latent.q_camera))
        self.assertTrue(torch.equal(reverse.q_local[:, 0], latent.q_local[:, -1]))
        self.assertTrue(torch.equal(swapped.q_entity[:, 0], latent.q_entity[:, 1]))
        self.assertEqual(float(zero.q_local.abs().sum()), 0.0)

    def test_present_entity_cannot_hide_behind_all_false_temporal_validity(self) -> None:
        latent = runner.synthetic_latent(torch.device("cpu"))
        invalid_temporal = latent.temporal_valid.clone()
        invalid_temporal[:, 0] = False
        invalid = elal3.ELAL3LatentV1(
            q_local=latent.q_local,
            q_entity=latent.q_entity,
            q_relation=latent.q_relation,
            q_phase=latent.q_phase,
            q_terminal=latent.q_terminal,
            q_camera=latent.q_camera,
            entity_presence=latent.entity_presence,
            temporal_valid=invalid_temporal.contiguous(),
            relation_valid=torch.zeros_like(latent.relation_valid),
            phase_valid=latent.phase_valid,
        )
        with self.assertRaisesRegex(elal3.ELAL3C0Error, "present entity"):
            invalid.validate()

    def test_sp4_selector_has_target_alignment_and_append_padding(self) -> None:
        latent = runner.synthetic_latent(torch.device("cpu"))
        memory = elal3.ELAL3ActionMemoryBuilderV1(variant="full")(latent)
        route = elal3.ELAL3RouteV1(
            total_tokens=42,
            condition_tokens=21,
            sequence_parallel_rank=3,
            sequence_parallel_size=4,
            memory=memory,
            route_identity="selector-test",
        )
        self.assertEqual(route.local_length, 11)
        self.assertEqual(route.local_target_indices(device=torch.device("cpu")).tolist(), list(range(12, 21)) + [-1, -1])
        self.assertEqual(int(route.local_padding_selector(device=torch.device("cpu")).sum()), 2)
        self.assertTrue(runner._checkpoint_route_gate(memory))

    def test_source_and_padding_preserve_signed_zero_bits(self) -> None:
        class NegativeZeroBlock(torch.nn.Module):
            def forward(self, hidden):
                return torch.copysign(torch.zeros_like(hidden), -torch.ones_like(hidden))

        class NegativeZeroRenderer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList(NegativeZeroBlock() for _ in range(30))

            def forward(self, hidden):
                for block in self.blocks:
                    hidden = block(hidden)
                return hidden

        latent = runner.synthetic_latent(torch.device("cpu"))
        model = NegativeZeroRenderer()
        handle = elal3.install_elal3_c0_v1(
            model,
            variant="full",
            attention_width=64,
            hidden_size=32,
            test_only=True,
        )
        memory = handle.build_memory(latent)
        source_route = elal3.ELAL3RouteV1(
            total_tokens=42,
            condition_tokens=21,
            sequence_parallel_rank=0,
            sequence_parallel_size=1,
            memory=memory,
            route_identity="signed-zero-source",
        )
        with handle.route(source_route):
            source_output = model(torch.ones((1, 42, 32), dtype=torch.float32))
        self.assertTrue(torch.signbit(source_output[:, :21]).all())
        self.assertTrue(all(row["source_bit_exact"] for row in handle.audit_records))
        handle.clear_audit()
        padding_route = elal3.ELAL3RouteV1(
            total_tokens=42,
            condition_tokens=21,
            sequence_parallel_rank=3,
            sequence_parallel_size=4,
            memory=memory,
            route_identity="signed-zero-padding",
        )
        with handle.route(padding_route):
            padding_output = model(torch.ones((1, 11, 32), dtype=torch.float32))
        self.assertTrue(torch.signbit(padding_output[:, -2:]).all())
        self.assertTrue(all(row["padding_bit_exact"] for row in handle.audit_records))
        handle.restore()

    def test_three_registered_engineering_arms_close_truthfully(self) -> None:
        receipts = {
            (variant, width): runner.run_gate(
                variant=variant,
                attention_width=width,
                device=torch.device("cpu"),
                seed=20260817,
            )
            for variant, width in (
                ("no_relation", 64),
                ("full", 64),
                ("full", 128),
            )
        }
        for (variant, width), receipt in receipts.items():
            self.assertTrue(receipt["engineering_gate_pass"], receipt)
            self.assertFalse(receipt["production_elal3_c0_authority"])
            self.assertFalse(receipt["training_authorized"])
            self.assertEqual(receipt["registered_arm"], f"{variant}-w{width}")
            self.assertEqual(len(receipt["renderer_block_grad_norms"]), 30)
            self.assertTrue(receipt["gates"]["source_rows_bit_exact"])
            self.assertTrue(receipt["gates"]["padding_rows_bit_exact"])
            self.assertTrue(receipt["gates"]["q_camera_nuisance_is_not_injected_into_action_loss"])
            self.assertIsNone(receipt["action_latent_input_grad_norms"]["q_camera"])
            self.assertIsInstance(receipt["receipt_digest"], str)
            self.assertEqual(
                receipt["paired_initialization_schema"],
                runner.PAIRED_INITIALIZATION_SCHEMA_VERSION,
            )
            self.assertEqual(receipt["paired_master_plan_row_count"], 219)
            self.assertEqual(
                receipt["paired_master_plan_row_count"],
                len(receipt["paired_master_plan_rows"]),
            )
            self.assertEqual(
                receipt["paired_master_plan_digest"],
                runner.object_sha256(receipt["paired_master_plan_rows"]),
            )
            self.assertEqual(
                receipt["paired_active_parameter_row_count"],
                len(receipt["paired_active_parameter_mapping"]),
            )
            self.assertEqual(
                receipt["paired_active_parameter_mapping_digest"],
                runner.object_sha256(receipt["paired_active_parameter_mapping"]),
            )
            self.assertFalse(receipt["frozen_output_encoder_receipt"]["requires_grad"])
        self.assertFalse(receipts[("no_relation", 64)]["synthetic_full_structure_gate_pass"])
        self.assertTrue(receipts[("full", 64)]["synthetic_full_structure_gate_pass"])
        self.assertTrue(receipts[("full", 128)]["synthetic_full_structure_gate_pass"])
        self.assertTrue(all(not receipt["complete_elal3_c0"] for receipt in receipts.values()))
        self.assertEqual(
            receipts[("no_relation", 64)]["intervention_target_rms_deltas_from_correct"]["relation_zero"],
            0.0,
        )
        self.assertGreater(
            receipts[("full", 64)]["intervention_target_rms_deltas_from_correct"]["relation_zero"],
            runner.INTERVENTION_TOLERANCE,
        )
        self.assertEqual(
            {receipt["synthetic_input_digest"] for receipt in receipts.values()},
            {receipts[("full", 64)]["synthetic_input_digest"]},
        )
        self.assertTrue(
            all(
                receipt["synthetic_inputs_generated_before_arm_initialization"]
                for receipt in receipts.values()
            )
        )
        no_relation = receipts[("no_relation", 64)]
        full64 = receipts[("full", 64)]
        full128 = receipts[("full", 128)]
        self.assertEqual(
            {receipt["paired_master_plan_digest"] for receipt in receipts.values()},
            {full64["paired_master_plan_digest"]},
        )
        self.assertTrue(
            all(
                receipt["paired_master_plan_rows"] == full64["paired_master_plan_rows"]
                for receipt in receipts.values()
            )
        )
        self.assertEqual(
            {runner.object_sha256(receipt["frozen_output_encoder_receipt"])
             for receipt in receipts.values()},
            {runner.object_sha256(full64["frozen_output_encoder_receipt"])},
        )
        self.assertEqual(no_relation["paired_active_parameter_row_count"], 216)
        self.assertEqual(full64["paired_active_parameter_row_count"], 219)
        self.assertEqual(full128["paired_active_parameter_row_count"], 219)

        def action_rows(receipt):
            return {
                row["parameter"]: row
                for row in receipt["paired_active_parameter_mapping"]
                if row["component"] == "action"
            }

        no_relation_rows = action_rows(no_relation)
        full64_rows = action_rows(full64)
        full128_rows = action_rows(full128)
        for parameter_name in sorted(set(no_relation_rows) & set(full64_rows)):
            self.assertEqual(
                no_relation_rows[parameter_name],
                full64_rows[parameter_name],
                parameter_name,
            )
        self.assertNotIn("memory_builder.relation_projection.weight", no_relation_rows)
        self.assertIn("memory_builder.relation_projection.weight", full64_rows)

        masters, master_rows = runner._materialize_paired_master_plan(
            seed=20260817,
            hidden_size=32,
        )
        self.assertEqual(master_rows, full64["paired_master_plan_rows"])
        for parameter_name, row64 in full64_rows.items():
            row128 = full128_rows[parameter_name]
            self.assertEqual(row64["master_namespace"], row128["master_namespace"])
            self.assertEqual(row64["master_sha256"], row128["master_sha256"])
            master = masters[row64["master_namespace"]]
            expected64, expected64_slice = runner._active_master_slice(
                master,
                parameter_name=parameter_name,
                attention_width=64,
            )
            expected128, expected128_slice = runner._active_master_slice(
                master,
                parameter_name=parameter_name,
                attention_width=128,
            )
            self.assertEqual(row64["active_slice"], expected64_slice)
            self.assertEqual(row128["active_slice"], expected128_slice)
            self.assertEqual(row64["active_sha256"], runner._tensor_receipt(expected64)["sha256"])
            self.assertEqual(row128["active_sha256"], runner._tensor_receipt(expected128)["sha256"])
            if parameter_name.endswith((".query.weight", ".key.weight", ".value.weight")):
                self.assertTrue(torch.equal(expected64, expected128[:64, :]))
            elif parameter_name.endswith(".output.weight"):
                self.assertTrue(torch.equal(expected64, expected128[:, :64]))
            else:
                self.assertTrue(torch.equal(expected64, expected128))

    def test_namespace_seed_changes_inputs_master_plan_and_frozen_readout(self) -> None:
        seed_a = 20260817
        seed_b = 20260818
        receipt_a = runner.run_gate(
            variant="full",
            attention_width=64,
            device=torch.device("cpu"),
            seed=seed_a,
        )
        receipt_b = runner.run_gate(
            variant="full",
            attention_width=64,
            device=torch.device("cpu"),
            seed=seed_b,
        )
        self.assertNotEqual(
            receipt_a["synthetic_input_digest"],
            receipt_b["synthetic_input_digest"],
        )
        self.assertNotEqual(
            receipt_a["paired_master_plan_digest"],
            receipt_b["paired_master_plan_digest"],
        )
        self.assertNotEqual(
            receipt_a["paired_active_parameter_mapping_digest"],
            receipt_b["paired_active_parameter_mapping_digest"],
        )
        self.assertNotEqual(
            receipt_a["frozen_output_encoder_receipt"]["sha256"],
            receipt_b["frozen_output_encoder_receipt"]["sha256"],
        )

    def test_cli_writes_one_create_only_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve() / "C0_gradient_causality_receipt.json"
            code = runner.main(
                (
                    "--variant", "no_relation",
                    "--attention-width", "64",
                    "--device", "cpu",
                    "--seed", "20260817",
                    "--output", str(output),
                )
            )
            self.assertEqual(code, 0)
            receipt = json.loads(output.read_text(encoding="ascii"))
            self.assertEqual(receipt["status"], "SYNTHETIC_ABLATION_GO")
            self.assertFalse(receipt["complete_elal3_c0"])
            self.assertEqual(
                runner.object_sha256({key: value for key, value in receipt.items() if key != "receipt_digest"}),
                receipt["receipt_digest"],
            )
            self.assertEqual(runner.main((
                "--variant", "no_relation",
                "--attention-width", "64",
                "--device", "cpu",
                "--output", str(output),
            )), 3)


if __name__ == "__main__":
    unittest.main()
