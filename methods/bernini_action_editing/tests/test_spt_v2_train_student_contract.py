from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
SPT_ROOT = METHOD_ROOT / "spt_v2"
for root in (METHOD_ROOT, SPT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import train_student as train
import phase_transport as spt
import phase_query_planner as phase_query
import audit_teacher_cohort as cohort


SHA1 = "1" * 40


def _args(**overrides):
    values = {
        "num_frames": 81,
        "max_steps": 10,
        "save_every": 2,
        "resume": None,
        "train_prefix_rows": None,
        "selected_membership": None,
        "learning_rate": 2e-4,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "seed": 7,
        "hidden_channels": 128,
        "attention_heads": 8,
        "match_channels": 32,
        "edit_slots": 8,
        "dense_query_chunk_size": 4096,
        "planner_architecture": phase_query.ARCHITECTURE_NAME,
        "gate_loss_weight": 1.0,
        "conditional_gate_loss_weight": 1.0,
        "gate_mass_loss_weight": 0.05,
        "offset_loss_weight": 0.25,
        "smooth_loss_weight": 0.01,
        "noop_loss_weight": 0.25,
        "change_tversky_weight": 0.5,
        "phase_change_mass_weight": 0.1,
        "phase_generate_mass_weight": 0.1,
        "mid_change_loss_weight": 0.25,
        "coarse_change_loss_weight": 0.125,
        "expected_offset_loss_weight": 0.1,
        "noop_generate_weight": 0.2,
        "noop_offset_weight": 0.25,
        "teacher_temperature": 0.08,
        "teacher_generate_threshold": 0.35,
        "teacher_feature_channels": 64,
        "max_generate_fraction_per_phase": 0.12,
        "noop_instruction": "keep unchanged",
        "expected_bernini_commit": SHA1,
        "expected_veomni_commit": SHA1,
        "expected_checkpoint_tree_sha256": train.legacy.CHECKPOINT_TREE_SHA256,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class StudentCliTests(unittest.TestCase):
    def test_defaults_are_planner_only_81f_with_resume_and_save(self) -> None:
        args = train.build_parser().parse_args(
            [
                "--bernini-root", "/b",
                "--veomni-root", "/v",
                "--checkpoint", "/c",
                "--preprocessed-parquet-dir", "/d",
                "--dataset-summary", "/s",
                "--output", "/o",
            ]
        )
        self.assertEqual(args.num_frames, 81)
        self.assertEqual(args.max_steps, 644)
        self.assertEqual(args.save_every, 64)
        self.assertIsNone(args.resume)
        self.assertIsNone(args.train_prefix_rows)
        self.assertIsNone(args.selected_membership)
        self.assertEqual(args.planner_architecture, phase_query.ARCHITECTURE_NAME)
        self.assertEqual(args.max_generate_fraction_per_phase, 0.12)
        self.assertEqual(args.conditional_gate_loss_weight, 1.0)
        self.assertEqual(args.gate_mass_loss_weight, 0.05)

    def test_student_api_has_no_target_parameter(self) -> None:
        self.assertEqual(
            list(inspect.signature(train.student_plan).parameters),
            ["planner", "source", "instruction_tokens"],
        )

    def test_torchrun_children_receive_disjoint_writable_runtime_caches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(
                os.environ,
                {
                    "BERNINI_SPT_RANK_CACHE_ROOT": directory,
                    "LOCAL_RANK": "3",
                },
                clear=False,
            ):
                self.assertTrue(train.configure_rank_local_runtime_cache())
                expected = Path(directory).resolve() / "rank-3"
                for name, suffix in (
                    ("MIOPEN_USER_DB_PATH", "miopen-user"),
                    ("MIOPEN_CUSTOM_CACHE_DIR", "miopen-custom"),
                    ("TORCH_EXTENSIONS_DIR", "torch-extensions"),
                    ("TRITON_CACHE_DIR", "triton"),
                ):
                    self.assertEqual(Path(os.environ[name]), expected / suffix)
                    self.assertTrue(Path(os.environ[name]).is_dir())

    def test_invalid_or_zero_supervision_fails_closed(self) -> None:
        invalid = (
            {"num_frames": 41},
            {"max_steps": 0},
            {"learning_rate": 0.0},
            {"noop_loss_weight": -1.0},
            {"conditional_gate_loss_weight": -1.0},
            {"gate_mass_loss_weight": -1.0},
            {"train_prefix_rows": 0},
            {"selected_membership": ""},
            {"selected_membership": "/membership.json", "train_prefix_rows": 8},
            {"attention_heads": 7},
            {"teacher_feature_channels": 8},
            {"max_generate_fraction_per_phase": 1.0},
            {
                "gate_loss_weight": 0.0,
                "offset_loss_weight": 0.0,
                "noop_loss_weight": 0.0,
            },
        )
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(train.StudentTrainingError):
                    train.validate_cli(_args(**values))

    def test_receipt_declares_teacher_only_target_and_explicit_allreduce(self) -> None:
        class Dataset:
            root = Path("/data")
            signature = "signature"
            def __len__(self):
                return 8

        class Planner:
            __name__ = "Planner"

        class Distributed:
            world_size = 4

        receipt = train._receipt(
            args=_args(),
            global_step=1,
            metrics={"total": 1.0},
            immutable={"value": {}, "digest": "digest"},
            dataset=Dataset(),
            dataset_summary={},
            training_membership={
                "selection": "ordered_prefix",
                "full_dataset_rows": 8,
                "training_rows": 1,
                "diagnostic_subset": True,
                "members": [{"row_index": 0, "iid": "x", "identity_sha256": "hash"}],
                "membership_sha256": "membership",
            },
            planner=Planner(),
            named=[("weight", type("P", (), {"numel": lambda self: 3})())],
            initialization_digest="init",
            distributed=Distributed(),
            backend="nccl/rccl",
            resumed_from=None,
        )
        self.assertIs(receipt["supervision"]["student_target_argument_exists"], False)
        self.assertIs(receipt["supervision"]["target_used_by_oracle_teacher_only"], True)
        self.assertIs(receipt["distributed"]["explicit_planner_gradient_all_reduce"], True)
        self.assertIs(receipt["distributed"]["same_pair_all_ranks"], False)
        self.assertEqual(receipt["distributed"]["samples_per_optimizer_step"], 4)
        self.assertEqual(receipt["global_samples_seen"], 4)
        self.assertEqual(receipt["planner"]["architecture"], phase_query.ARCHITECTURE_NAME)
        self.assertIs(receipt["dataset"]["diagnostic_subset"], True)
        self.assertEqual(receipt["dataset"]["training_membership_sha256"], "membership")
        self.assertEqual(
            receipt["supervision"]["action_gate_loss"],
            "hierarchical_sparse_change_then_conditional_tg_v1",
        )
        self.assertEqual(
            receipt["supervision"]["action_change_loss"],
            "ordinary_cellwise_bce_on_one_minus_preserve",
        )
        self.assertEqual(
            receipt["supervision"]["offset_loss"],
            "transport_cell_and_three_axis_mean_smooth_l1",
        )
        candidate = dict(receipt)
        declared = candidate.pop("receipt_digest")
        self.assertEqual(train.legacy.object_sha256(candidate), declared)

    def test_immutable_contract_is_exact_after_json_roundtrip(self) -> None:
        class Dataset:
            signature = "dataset"

        immutable = train._immutable(
            args=_args(),
            dataset=Dataset(),
            dataset_summary={"sha256": "summary", "index_sha256": "index"},
            planner_config=phase_query.PhaseQueryPlannerConfig(),
            teacher_config=spt.PhaseTransportConfig(),
            training_membership={
                "selection": "full_dataset",
                "full_dataset_rows": 1,
                "training_rows": 1,
                "diagnostic_subset": False,
                "members": [{"row_index": 0, "iid": "x", "identity_sha256": "hash"}],
                "membership_sha256": "membership",
            },
            world_size=4,
        )
        self.assertEqual(json.loads(json.dumps(immutable)), immutable)
        self.assertEqual(
            immutable["value"]["planner_architecture"], phase_query.ARCHITECTURE_NAME
        )
        self.assertIsNone(immutable["value"]["instruction_pooling"])
        self.assertEqual(immutable["value"]["data_parallel_world_size"], 4)
        self.assertEqual(immutable["value"]["ulysses_size"], 1)

    def test_four_way_data_parallel_schedule_covers_four_distinct_rows(self) -> None:
        self.assertEqual(
            [train.data_parallel_row_index(0, 4, rank, 8) for rank in range(4)],
            [0, 1, 2, 3],
        )
        self.assertEqual(
            [train.data_parallel_row_index(1, 4, rank, 8) for rank in range(4)],
            [4, 5, 6, 7],
        )
        self.assertEqual(
            [train.data_parallel_row_index(2, 4, rank, 8) for rank in range(4)],
            [0, 1, 2, 3],
        )
        seeds = [train.data_parallel_seed(7, 0, 4, rank, rank) for rank in range(4)]
        self.assertEqual(len(set(seeds)), 4)

    def test_four_way_schedule_maps_ordinals_to_sparse_selected_rows(self) -> None:
        selected_rows = [2, 7, 11, 19, 23, 29, 31, 37]
        membership = {
            "training_rows": len(selected_rows),
            "members": [
                {
                    "row_index": row,
                    "iid": f"iid-{row}",
                    "identity_sha256": f"hash-{row}",
                }
                for row in selected_rows
            ],
        }
        self.assertEqual(
            [
                train.data_parallel_dataset_row_index(0, 4, rank, membership)
                for rank in range(4)
            ],
            selected_rows[:4],
        )
        self.assertEqual(
            [
                train.data_parallel_dataset_row_index(1, 4, rank, membership)
                for rank in range(4)
            ],
            selected_rows[4:],
        )
        cohort = [
            {
                "rank": rank,
                "row_index": selected_rows[rank],
                "iid": f"iid-{selected_rows[rank]}",
                "identity_sha256": f"hash-{selected_rows[rank]}",
            }
            for rank in range(4)
        ]
        train.validate_data_parallel_cohort(
            cohort,
            global_step=0,
            world_size=4,
            training_membership=membership,
        )

    def test_training_membership_strictly_loads_teacher_trust_rows(self) -> None:
        class Dataset:
            signature = "dataset-signature"

            def __init__(self):
                self.rows = [
                    {"iid": f"iid-{index}", "inputs": {"instruction": str(index)}}
                    for index in range(6)
                ]

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, index):
                return self.rows[index]

        dataset = Dataset()
        summary = {
            "sha256": "a" * 64,
            "summary_digest": "b" * 64,
            "index_sha256": "c" * 64,
        }
        selected = (1, 4)
        reports = []
        for row_index in selected:
            row = dataset[row_index]
            reports.append(
                {
                    "row_index": row_index,
                    "iid": row["iid"],
                    "identity_sha256": train.legacy.dataset_identity(row, row_index),
                    "selection": {"selected": True},
                }
            )
        value = cohort.build_selected_membership(
            reports=reports,
            dataset=dataset,
            dataset_summary=summary,
            scan=cohort.ScanWindow("ordered_prefix", 0, 6),
            thresholds=cohort.SelectorThresholds(),
            minimum_selected=2,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "membership.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            actual = train._training_membership(
                dataset,
                None,
                selected_membership=str(path),
                dataset_summary=summary,
            )
        self.assertEqual(actual["selection"], "teacher_trust_membership")
        self.assertEqual(
            [member["row_index"] for member in actual["members"]],
            list(selected),
        )
        self.assertEqual(
            actual["selected_membership_digest"], value["membership_digest"]
        )
        self.assertTrue(actual["implicit_dataset_fallback_forbidden"])

    def test_data_parallel_cohort_binds_each_rank_to_membership_iid_and_hash(self) -> None:
        membership = {
            "training_rows": 8,
            "members": [
                {"row_index": row, "iid": f"iid-{row}", "identity_sha256": f"hash-{row}"}
                for row in range(8)
            ],
        }
        cohort = [
            {
                "rank": rank,
                "row_index": rank,
                "iid": f"iid-{rank}",
                "identity_sha256": f"hash-{rank}",
            }
            for rank in range(4)
        ]
        actual = train.validate_data_parallel_cohort(
            cohort,
            global_step=0,
            world_size=4,
            training_membership=membership,
        )
        self.assertEqual([item["row_index"] for item in actual], [0, 1, 2, 3])
        corrupted = [dict(item) for item in cohort]
        corrupted[2]["identity_sha256"] = "wrong"
        with self.assertRaisesRegex(train.StudentTrainingError, "identity hash"):
            train.validate_data_parallel_cohort(
                corrupted,
                global_step=0,
                world_size=4,
                training_membership=membership,
            )

    def test_legacy_global_checkpoint_fails_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy_receipt = {
                "schema_version": "bernini-spt-v2-student-receipt-v1",
                "planner": {"class": "PhaseTransportAdapter"},
            }
            candidate = dict(legacy_receipt)
            legacy_receipt["receipt_digest"] = train.legacy.object_sha256(candidate)
            (root / "receipt.json").write_text(json.dumps(legacy_receipt), encoding="utf-8")
            (root / "planner_config.json").write_text(
                json.dumps({"latent_channels": 64}), encoding="utf-8"
            )
            (root / "planner.safetensors").touch()
            (root / "optimizer.pt").touch()
            with self.assertRaisesRegex(train.StudentTrainingError, "global-pooled"):
                train._load_resume(root)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is unavailable")
class StudentTokenTensorTests(unittest.TestCase):
    def test_unpadding_keeps_every_real_token_without_mean_pooling(self) -> None:
        tokens = torch.randn(1, 7, 12)
        tokens[:, 5:].zero_()
        actual = train._unpadded_text_tokens(torch.tensor([7]), tokens)
        self.assertEqual(tuple(actual.shape), (1, 5, 12))
        self.assertTrue(torch.equal(actual, tokens[:, :5].float()))

    def test_internal_zero_token_fails_closed(self) -> None:
        tokens = torch.randn(1, 5, 12)
        tokens[:, 2].zero_()
        with self.assertRaisesRegex(train.StudentTrainingError, "internal zero"):
            train._unpadded_text_tokens(torch.tensor([5]), tokens)

    def test_uniform_gate_gradient_restores_sparse_preserve_prior(self) -> None:
        logits = torch.zeros(1, 3, 1, 1, 20, requires_grad=True)
        student = logits.softmax(dim=1)
        teacher = torch.zeros_like(student)
        teacher[:, spt.GATE_PRESERVE, ..., :18] = 1.0
        teacher[:, spt.GATE_TRANSPORT, ..., 18] = 1.0
        teacher[:, spt.GATE_GENERATE, ..., 19] = 1.0
        parts = train.hierarchical_sparse_gate_loss(student, teacher)
        loss = (
            parts["change_bce"]
            + parts["conditional_tg_ce"]
            + 0.05 * parts["gate_mass_l1"]
        )
        loss.backward()
        preserve_gradient = float(logits.grad[:, spt.GATE_PRESERVE].mean())
        self.assertLess(preserve_gradient, 0.0)
        self.assertGreater(abs(preserve_gradient), 0.01)

    def test_hierarchical_change_bce_is_safe_under_autocast(self) -> None:
        logits = torch.zeros(1, 3, 1, 1, 4, requires_grad=True)
        teacher = torch.zeros_like(logits)
        teacher[:, spt.GATE_PRESERVE, ..., :3] = 1.0
        teacher[:, spt.GATE_TRANSPORT, ..., 3] = 1.0
        device_type = logits.device.type
        autocast_dtype = torch.bfloat16 if device_type == "cpu" else torch.float16
        with torch.autocast(
            device_type=device_type,
            dtype=autocast_dtype,
            enabled=True,
        ):
            student = logits.softmax(dim=1)
            parts = train.hierarchical_sparse_gate_loss(student, teacher)
            loss = parts["change_bce"] + parts["conditional_tg_ce"]
        loss.backward()
        self.assertEqual(parts["change_bce"].dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))

    def test_perfect_hierarchical_plan_loss_is_near_zero(self) -> None:
        teacher = torch.zeros(1, 3, 1, 1, 5)
        teacher[:, spt.GATE_PRESERVE, ..., :3] = 1.0
        teacher[:, spt.GATE_TRANSPORT, ..., 3] = 1.0
        teacher[:, spt.GATE_GENERATE, ..., 4] = 1.0
        parts = train.hierarchical_sparse_gate_loss(teacher.clone(), teacher)
        total = (
            parts["change_bce"]
            + parts["conditional_tg_ce"]
            + 0.05 * parts["gate_mass_l1"]
        )
        self.assertLess(float(total), 2e-6)

    def test_rare_generate_cell_has_finite_conditional_gradient(self) -> None:
        logits = torch.zeros(1, 3, 1, 1, 10, requires_grad=True)
        logits.data[:, spt.GATE_TRANSPORT] = 2.0
        logits.data[:, spt.GATE_GENERATE] = -2.0
        student = logits.softmax(dim=1)
        teacher = torch.zeros_like(student)
        teacher[:, spt.GATE_PRESERVE, ..., :5] = 1.0
        teacher[:, spt.GATE_TRANSPORT, ..., 5:9] = 1.0
        teacher[:, spt.GATE_GENERATE, ..., 9] = 1.0
        conditional = train.hierarchical_sparse_gate_loss(student, teacher)[
            "conditional_tg_ce"
        ]
        conditional.backward()
        rare_generate_gradient = logits.grad[0, spt.GATE_GENERATE, 0, 0, 9]
        self.assertTrue(bool(torch.isfinite(conditional)))
        self.assertTrue(bool(torch.isfinite(rare_generate_gradient)))
        self.assertLess(float(rare_generate_gradient), 0.0)

    def test_transport_offset_huber_means_cells_and_three_axes(self) -> None:
        student_offsets = torch.full((1, 3, 1, 1, 2), 2.0)
        teacher_offsets = torch.zeros_like(student_offsets)
        teacher_gates = torch.zeros(1, 3, 1, 1, 2)
        teacher_gates[:, spt.GATE_TRANSPORT, ..., 0] = 1.0
        teacher_gates[:, spt.GATE_PRESERVE, ..., 1] = 1.0
        actual = train.transport_offset_huber_loss(
            student_offsets,
            teacher_offsets,
            teacher_gates,
        )
        mae = train.transport_cell_offset_mae(
            student_offsets,
            teacher_offsets,
            teacher_gates,
        )
        # SmoothL1(beta=1) at error=2 is 1.5.  Averaging the three axes must
        # remain 1.5 rather than the legacy sum-over-axis value 4.5.
        self.assertAlmostEqual(float(actual), 1.5, places=6)
        self.assertAlmostEqual(float(mae), 2.0, places=6)

    def test_hard_change_metrics_handle_empty_and_missed_sets(self) -> None:
        teacher = torch.zeros(1, 3, 1, 1, 4)
        teacher[:, spt.GATE_PRESERVE] = 1.0
        perfect = train.hard_gate_spatial_metrics(teacher, teacher)
        for name in (
            "hard_gate_argmax_accuracy",
            "change_iou",
            "change_precision",
            "change_recall",
        ):
            self.assertEqual(float(perfect[name]), 1.0)

        changed_teacher = teacher.clone()
        changed_teacher[:, spt.GATE_PRESERVE, ..., -1] = 0.0
        changed_teacher[:, spt.GATE_GENERATE, ..., -1] = 1.0
        missed = train.hard_gate_spatial_metrics(teacher, changed_teacher)
        self.assertEqual(float(missed["change_iou"]), 0.0)
        self.assertEqual(float(missed["change_precision"]), 0.0)
        self.assertEqual(float(missed["change_recall"]), 0.0)
        self.assertAlmostEqual(float(missed["hard_gate_argmax_accuracy"]), 0.75)

    def test_planner_loss_reports_spatial_and_per_phase_generate_metrics(self) -> None:
        source = torch.zeros(1, 21, 1, 10, 4)
        offsets = torch.zeros(1, 3, 21, 1, 10)
        gates = torch.zeros(1, 3, 21, 1, 10)
        gates[:, spt.GATE_PRESERVE, ..., :8] = 1.0
        gates[:, spt.GATE_TRANSPORT, ..., 8] = 1.0
        gates[:, spt.GATE_GENERATE, ..., 9] = 1.0
        diagnostics = {
            "prebudget_generate_fraction": 0.5,
            "postbudget_generate_fraction": 0.1,
            "budget_reject_fraction": 0.4,
            "max_generate_fraction_per_phase": 0.12,
            "observed_max_postbudget_generate_fraction_per_phase": 0.1,
        }
        teacher = spt.PhasePlan(
            offsets=offsets.clone(),
            gate_probs=gates.clone(),
            provenance="oracle_pair_proxy",
            diagnostics=diagnostics,
        )
        action = spt.PhasePlan(
            offsets=offsets.clone(),
            gate_probs=gates.clone(),
            provenance="student",
        )
        noop = spt.exact_identity_plan(source)
        loss, parts = train._planner_loss(action, teacher, noop, source, _args())
        self.assertLess(float(loss), 2e-6)
        self.assertAlmostEqual(
            float(parts["student_observed_max_generate_fraction_per_phase"]),
            0.1,
            places=6,
        )
        self.assertEqual(float(parts["hard_gate_argmax_accuracy"]), 1.0)
        self.assertEqual(float(parts["change_iou"]), 1.0)
        self.assertEqual(float(parts["change_precision"]), 1.0)
        self.assertEqual(float(parts["change_recall"]), 1.0)
        self.assertEqual(float(parts["transport_cell_offset_mae"]), 0.0)


if __name__ == "__main__":
    unittest.main()
