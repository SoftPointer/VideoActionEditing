from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_fitq_official_runtime_scan as runtime  # noqa: E402


BRANCHES = (
    "frozen_t2v_action",
    "frozen_t2v_hard_negative[0]",
    "frozen_t2v_hard_negative[1]",
    "frozen_identity_noop_correct",
    "frozen_identity_noop_wrong_source[0]",
    "frozen_identity_action_correct",
    "frozen_identity_action_wrong_source[0]",
)


class FITQOfficialRuntimePureContractTests(unittest.TestCase):
    def test_raw_tensor_identity_comparison_includes_dtype_and_value_bits(self) -> None:
        base = {
            "shape": [1, 2],
            "dtype": "torch.float32",
            "numel": 2,
            "byte_count": 8,
            "raw_storage_sha256": "a" * 64,
            "label": "left",
        }
        relabelled = {**base, "label": "right"}
        self.assertTrue(runtime._raw_tensor_identities_equal(base, relabelled))
        for key, value in (
            ("dtype", "torch.float64"),
            ("shape", [2, 1]),
            ("raw_storage_sha256", "b" * 64),
        ):
            with self.subTest(key=key):
                changed = {**relabelled, key: value}
                self.assertFalse(runtime._raw_tensor_identities_equal(base, changed))
        incomplete = dict(base)
        incomplete.pop("raw_storage_sha256")
        with self.assertRaises(runtime.FITQOfficialRuntimeScanError):
            runtime._raw_tensor_identities_equal(base, incomplete)

    def test_branch_plan_binds_every_forward_to_explicit_mode(self) -> None:
        plan = runtime.build_explicit_branch_plan(BRANCHES)
        self.assertEqual(len(plan), 7)
        self.assertEqual([item["branch"] for item in plan], list(BRANCHES))
        self.assertEqual([item["mode"] for item in plan[:3]], ["t2v"] * 3)
        self.assertEqual([item["mode"] for item in plan[3:]], ["mv2v"] * 4)

        invalid = (
            BRANCHES[:-1],
            ("unknown",) + BRANCHES[1:],
            (BRANCHES[0],) * 7,
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(
                runtime.FITQOfficialRuntimeScanError
            ):
                runtime.build_explicit_branch_plan(value)

    def test_statistics_path_must_be_new_absolute_child(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve()
            wanted = parent / "statistics"
            self.assertEqual(
                runtime._require_absolute_new_directory(str(wanted)), wanted
            )
            wanted.mkdir()
            with self.assertRaisesRegex(
                runtime.FITQOfficialRuntimeScanError, "refusing to reuse"
            ):
                runtime._require_absolute_new_directory(str(wanted))
            with self.assertRaises(runtime.FITQOfficialRuntimeScanError):
                runtime._require_absolute_new_directory("relative/statistics")

    def test_truthful_evidence_replaces_legacy_no_callback_claims(self) -> None:
        cell = {
            "cell_digest": "a" * 64,
            "no_forward_callback": True,
            "direct_official_shared_step": True,
        }
        original = {
            "method": "bernini-iar-official-runtime-smoke-v1",
            "forwards_per_rank": 84,
            "cell_records": [cell],
            "iar_core": {
                "full_cell_grid_closure": {"cell_digests": ["a" * 64]},
                "full_cell_grid_digest": "b" * 64,
            },
            "forward_callback_present": False,
            "custom_core_present": False,
        }
        untouched = copy.deepcopy(original)
        evidence = runtime._truthful_field_evidence(
            original,
            underlying_schema_validation_shadow_digest="c" * 64,
        )
        self.assertEqual(original, untouched)
        self.assertEqual(evidence["field_grid_forwards_per_rank"], 84)
        self.assertEqual(evidence["hooked_action_duplicate_forwards_per_rank"], 1)
        self.assertEqual(evidence["hook_off_reference_forwards_per_rank"], 1)
        self.assertEqual(evidence["forwards_per_rank"], 86)
        self.assertTrue(evidence["forward_callback_present"])
        self.assertTrue(evidence["read_only_forward_hooks_present"])
        self.assertFalse(evidence["custom_forward_core_present"])
        self.assertTrue(evidence["custom_analysis_core_present"])
        self.assertTrue(
            evidence["underlying_iar_shadow_is_hypothetical_not_runtime_provenance"]
        )
        self.assertFalse(evidence["underlying_iar_shadow_emitted_as_receipt"])
        rewritten = evidence["cell_records"][0]
        self.assertFalse(rewritten["no_forward_callback"])
        self.assertTrue(rewritten["read_only_forward_hooks_present"])
        self.assertNotEqual(rewritten["cell_digest"], "a" * 64)

    def test_runtime_source_does_not_use_torch_equal_for_final_fields(self) -> None:
        source = Path(runtime.__file__).read_text(encoding="utf-8")
        self.assertNotIn("torch.equal(hook_off_reference, field)", source)
        self.assertNotIn("torch.equal(field, duplicate_field)", source)
        self.assertIn("shape_dtype_and_raw_storage_sha256", source)

    def test_common_evidence_is_single_proposal_non_authorizing(self) -> None:
        adapter = runtime.FITQOfficialRuntimeAdapter(
            iar_module=object(),
            statistics_dir=Path("/tmp/fitq-unit-statistics"),
            original_direct_prediction=None,
            original_run_cell=None,
            original_assemble_receipt=None,
        )
        adapter._observed_forward_count = 85
        adapter._reference_forward_count = 1
        adapter._artifact_records = [{} for _ in range(85)]
        adapter._context_records = [{} for _ in range(85)]
        adapter._cell_parity_records = [
            {"all_pairs_byte_exact": True} for _ in range(12)
        ]
        adapter._hook_output_parity = {"byte_exact_equal": True}
        adapter._action_duplicate_floor = {
            "block0_same_state_exact": {"all": True}
        }
        adapter._duplicate_local_fingerprint_pair_digest = "d" * 64
        evidence = adapter._fitq_common_evidence(
            {"forwards_per_rank": 84, "cell_records": [{} for _ in range(12)]}
        )
        self.assertEqual(evidence["field_grid_forwards_per_rank"], 84)
        self.assertEqual(evidence["observed_hooked_forwards_per_rank"], 85)
        self.assertEqual(evidence["total_official_forwards_per_rank"], 86)
        self.assertEqual(
            evidence["analysis_statistics"], "phase_head_mean_second_moment"
        )
        self.assertEqual(evidence["proposal_bank_status"], "insufficient_bank")
        self.assertEqual(
            evidence["decision_scope"], "engineering_N0_like_diagnostic_only"
        )
        self.assertFalse(evidence["fitq_go_authorized"])
        self.assertFalse(evidence["tokenwise_localization_available"])
        self.assertEqual(
            evidence["scientific_fitq_outcome"], "not_evaluated_single_proposal"
        )
        self.assertFalse(evidence["cross_mode_transport_gate_evaluated"])
        self.assertFalse(evidence["causal_intervention_gate_evaluated"])


if __name__ == "__main__":
    unittest.main()
