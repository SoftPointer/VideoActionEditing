from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import scan_cross_mode_teacher_gate as scan


SHA1 = "a" * 40
SHA256 = "b" * 64


class _Route:
    def __init__(self, ordinal: int):
        self.iid = f"iid-{ordinal:03d}"
        self.tier = "motion_only"
        self.full_target_weight = 0.0


def _routes():
    return [(1000 + ordinal, _Route(ordinal)) for ordinal in range(359)]


def _args(*extra: str):
    argv = [
        "--bernini-root",
        "/bernini",
        "--veomni-root",
        "/veomni",
        "--checkpoint",
        "/checkpoint",
        "--preprocessed-parquet-dir",
        "/data",
        "--dataset-summary",
        "/summary.json",
        "--routing-jsonl",
        "/route.jsonl",
        "--output",
        "/output",
        "--method-source-revision",
        SHA1,
        "--method-source-archive-sha256",
        SHA256,
        *extra,
    ]
    return scan.build_parser().parse_args(argv)


def _record(candidate, *, contract_sha=SHA256, passed=True):
    value = {
        "schema_version": scan.RECORD_SCHEMA,
        "scan_contract_sha256": contract_sha,
        "candidate": candidate.as_dict(),
        "candidate_sha256": scan._candidate_digest(candidate),
        "instruction_sha256": "c" * 64,
        "t2v_rope_parity": {"verified": True},
        "frozen_forward_order": list(scan.FROZEN_FORWARD_ORDER),
        "frozen_velocity_rms": {
            name: 1.0 + index
            for index, name in enumerate(scan.FROZEN_FORWARD_ORDER)
        },
        "gate": {
            "passed": bool(passed),
            "active_phase_count": 8,
            "mean_direction_cosine": 0.5,
            "log_amplitude_mae": 0.4,
            "covered_phase_fraction": 0.75,
            "normalized_rmse": 0.8,
            "frozen_prior_rms": 0.3,
            "target_motion_rms": 0.25,
        },
    }
    value["record_sha256"] = scan.trainer.legacy.object_sha256(value)
    return value


class OfflineGateScanPureContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grid = scan.build_fixed_grid(_routes(), base_seed=20260807)
        cls.grid_sha = scan.fixed_grid_sha256(cls.grid)

    def test_module_is_lazy_and_has_no_training_surface(self) -> None:
        tree = ast.parse(Path(scan.__file__).read_text(encoding="utf-8"))
        eager_torch = []
        forbidden_calls = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                eager_torch.extend(
                    alias.name for alias in node.names if alias.name == "torch"
                )
            elif isinstance(node, ast.ImportFrom) and node.module == "torch":
                eager_torch.append(node.module)
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr == "backward":
                    forbidden_calls.append("backward")
                if isinstance(node.func, ast.Name) and node.func.id == "get_peft_model":
                    forbidden_calls.append("get_peft_model")
        self.assertEqual(eager_torch, [])
        self.assertEqual(forbidden_calls, [])
        options = {
            option
            for action in scan.build_parser()._actions
            for option in action.option_strings
        }
        self.assertNotIn("--max-steps", options)
        self.assertNotIn("--learning-rate", options)
        self.assertNotIn("--disable-frozen-prior-gate", options)

    def test_cli_and_frozen_branch_identity_are_pinned(self) -> None:
        scan.validate_cli(_args())
        self.assertEqual(scan.STRICT_PAIR_COUNT, 359)
        self.assertEqual(scan.SIGMA_COUNT, 40)
        self.assertEqual(scan.GRID_SIZE, 14360)
        expected = tuple(
            value
            for value in scan.trainer.FORWARD_CELL_ORDER
            if value != "adapted_editor_action_full_source"
        )
        self.assertEqual(scan.FROZEN_FORWARD_ORDER, expected)
        with self.assertRaisesRegex(scan.OfflineGateScanError, "semantic no-op"):
            scan.validate_cli(_args("--noop-instruction", "do something"))

    def test_fixed_grid_is_pair_major_complete_and_seeded_per_cell(self) -> None:
        grid = self.grid
        self.assertEqual(len(grid), 359 * 40)
        self.assertEqual(grid[0].candidate_ordinal, 0)
        self.assertEqual(grid[0].pair_ordinal, 0)
        self.assertEqual(grid[0].sigma_schedule_index, 0)
        self.assertEqual(grid[39].pair_ordinal, 0)
        self.assertEqual(grid[39].sigma_schedule_index, 39)
        self.assertEqual(grid[40].pair_ordinal, 1)
        self.assertEqual(grid[40].sigma_schedule_index, 0)
        self.assertEqual(
            grid[40].seed,
            scan.trainer.legacy.step_seed(20260807, 40, 1001),
        )
        self.assertTrue(grid[30].teacher_active)
        self.assertFalse(grid[31].teacher_active)
        self.assertEqual(
            grid[17].sigma_float32_be_hex,
            scan.trainer.sigma_strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[17],
        )
        self.assertEqual(self.grid_sha, scan.fixed_grid_sha256(grid))
        changed = scan.build_fixed_grid(_routes(), base_seed=20260808)
        self.assertNotEqual(self.grid_sha, scan.fixed_grid_sha256(changed))

    def test_grid_rejects_non_strict_or_duplicate_routes(self) -> None:
        short = _routes()[:-1]
        with self.assertRaisesRegex(scan.OfflineGateScanError, "exactly 359"):
            scan.build_fixed_grid(short, base_seed=1)
        duplicate = _routes()
        duplicate[-1] = duplicate[0]
        with self.assertRaisesRegex(scan.OfflineGateScanError, "strict359"):
            scan.build_fixed_grid(duplicate, base_seed=1)

    def test_selection_is_hash_deterministic_and_preserves_sigma_grid(self) -> None:
        records = []
        for candidate in self.grid:
            # For one active sigma, make exactly one pair eligible.  For every
            # rho=0 sigma, make every gate fail: selection must still preserve
            # those strata because the trainer's late replay gate is inactive.
            if candidate.sigma_schedule_index == 7:
                passed = candidate.pair_ordinal == 23
            elif not candidate.teacher_active:
                passed = False
            else:
                passed = True
            records.append(_record(candidate, passed=passed))
        first = scan.build_selection_table(
            records,
            self.grid,
            grid_sha256=self.grid_sha,
            scan_contract_sha256=SHA256,
        )
        second = scan.build_selection_table(
            records,
            self.grid,
            grid_sha256=self.grid_sha,
            scan_contract_sha256=SHA256,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["entry_count"], 40)
        self.assertTrue(first["training_authorized"])
        self.assertEqual(first["missing_active_sigma_indices"], [])
        self.assertEqual(
            [entry["sigma_schedule_index"] for entry in first["entries"]],
            list(range(40)),
        )
        self.assertEqual(first["entries"][7]["candidate"]["pair_ordinal"], 23)
        self.assertEqual(first["entries"][7]["eligibility"], "gate_passed")
        self.assertFalse(first["entries"][31]["gate_required"])
        self.assertEqual(
            first["entries"][31]["eligibility"], "rho_zero_gate_inactive"
        )
        digest_candidate = dict(first)
        digest = digest_candidate.pop("selection_table_sha256")
        self.assertEqual(digest, scan.trainer.legacy.object_sha256(digest_candidate))

    def test_negative_result_is_hashed_but_cannot_authorize_training(self) -> None:
        records = [
            _record(
                candidate,
                passed=(candidate.sigma_schedule_index != 5),
            )
            for candidate in self.grid
        ]
        selection = scan.build_selection_table(
            records,
            self.grid,
            grid_sha256=self.grid_sha,
            scan_contract_sha256=SHA256,
        )
        self.assertFalse(selection["training_authorized"])
        self.assertEqual(selection["missing_active_sigma_indices"], [5])
        self.assertEqual(selection["selected_count"], 39)
        missing = selection["entries"][5]
        self.assertEqual(missing["eligibility"], "no_gate_passing_candidate")
        self.assertIsNone(missing["candidate"])
        self.assertIsNone(missing["record_sha256"])
        digest_candidate = dict(selection)
        digest = digest_candidate.pop("selection_table_sha256")
        self.assertEqual(digest, scan.trainer.legacy.object_sha256(digest_candidate))

    def test_record_prefix_and_hash_tampering_fail_closed(self) -> None:
        prefix = [_record(candidate) for candidate in self.grid[:3]]
        scan.validate_complete_records(
            prefix,
            self.grid,
            scan_contract_sha256=SHA256,
            allow_prefix=True,
        )
        with self.assertRaisesRegex(scan.OfflineGateScanError, "count"):
            scan.validate_complete_records(
                prefix, self.grid, scan_contract_sha256=SHA256
            )
        tampered = dict(prefix[1])
        tampered["gate"] = dict(tampered["gate"])
        tampered["gate"]["mean_direction_cosine"] = 0.9
        with self.assertRaisesRegex(scan.OfflineGateScanError, "hash differs"):
            scan.validate_complete_records(
                [prefix[0], tampered],
                self.grid,
                scan_contract_sha256=SHA256,
                allow_prefix=True,
            )

    def test_summary_binds_distribution_records_and_selection(self) -> None:
        records = [_record(candidate) for candidate in self.grid]
        selection = scan.build_selection_table(
            records,
            self.grid,
            grid_sha256=self.grid_sha,
            scan_contract_sha256=SHA256,
        )
        summary = scan.build_summary(
            records,
            self.grid,
            grid_sha256=self.grid_sha,
            scan_contract_sha256=SHA256,
            selection_table=selection,
            records_file_sha256="d" * 64,
        )
        self.assertTrue(summary["complete"])
        self.assertTrue(summary["read_only_frozen_scan"])
        self.assertEqual(summary["model_parameter_updates"], 0)
        self.assertEqual(summary["backward_calls"], 0)
        self.assertEqual(summary["record_count"], 14360)
        self.assertEqual(summary["gate_pass_rate"], 1.0)
        self.assertTrue(summary["training_authorized"])
        self.assertEqual(len(summary["per_sigma"]), 40)
        self.assertEqual(len(summary["per_pair"]), 359)
        self.assertEqual(
            summary["metric_distributions"]["mean_direction_cosine"]["p50"],
            0.5,
        )
        digest_candidate = dict(summary)
        digest = digest_candidate.pop("summary_sha256")
        self.assertEqual(digest, scan.trainer.legacy.object_sha256(digest_candidate))


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class OfflineGateScanTensorContractTests(unittest.TestCase):
    def test_five_frozen_forwards_use_exact_order_and_exact_gate(self) -> None:
        labels = []

        def velocity(_renderer, batch):
            labels.append(batch["label"])
            return torch.full((1, 21, 2), len(labels), dtype=torch.bfloat16)

        candidate = scan.trainer.MovedCandidate(
            editor_negative={"label": "editor_negative"},
            editor_noop={"label": "editor_noop"},
            editor_action={"label": "editor_action"},
            generator_action={"label": "generator_action"},
            generator_negative={"label": "generator_negative"},
            generator_action_text_fields={},
            generator_negative_text_fields={},
            auxiliary={
                "shared_noisy": torch.zeros(1, 21, 2, dtype=torch.float32),
                "sigma": torch.tensor(0.5, dtype=torch.float32),
                "source_clean": torch.zeros(1, 21, 2, dtype=torch.float32),
                "target_clean": torch.ones(1, 21, 2, dtype=torch.float32),
            },
            spatial_hw=(1, 1),
            instruction_sha256="e" * 64,
            t2v_rope_parity={"verified": True},
        )
        generator_uncond = torch.zeros(1, 21, 1, 2, dtype=torch.float32)
        generator_action = torch.ones(1, 21, 1, 2, dtype=torch.float32)
        gate = SimpleNamespace(passed=torch.tensor([True]))
        captured = {}

        def exact_gate(prior, target, *, config):
            captured["prior"] = prior
            captured["target"] = target
            captured["config"] = config
            return gate

        def phase(value):
            return value.reshape(1, 21, 1, 2)

        with mock.patch.object(
            scan.trainer.motion,
            "renderer_velocity_prediction",
            side_effect=velocity,
        ), mock.patch.object(
            scan.trainer,
            "_generator_plain_cfg_clean",
            return_value=(generator_uncond, generator_action),
        ), mock.patch.object(
            scan.trainer.v5, "_as_phase_grid", side_effect=phase
        ), mock.patch.object(
            scan.trainer.spectrum, "q0", side_effect=lambda value: value
        ), mock.patch.object(
            scan.trainer.core,
            "compute_frozen_prior_gate",
            side_effect=exact_gate,
        ):
            result = scan._run_five_frozen_forward_cell(
                renderer=object(), candidate=candidate
            )

        self.assertEqual(
            labels,
            [
                "editor_negative",
                "editor_noop",
                "editor_action",
                "generator_negative",
                "generator_action",
            ],
        )
        self.assertIs(result.gate, gate)
        self.assertFalse(captured["prior"].requires_grad)
        self.assertFalse(captured["target"].requires_grad)
        self.assertTrue(captured["config"].enforce_frozen_prior_gate)
        self.assertEqual(
            list(result.frozen_velocity_rms), list(scan.FROZEN_FORWARD_ORDER)
        )


if __name__ == "__main__":
    unittest.main()
