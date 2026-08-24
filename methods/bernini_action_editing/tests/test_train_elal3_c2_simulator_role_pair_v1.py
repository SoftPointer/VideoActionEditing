from __future__ import annotations

import ast
import copy
import gc
import hashlib
import io
import json
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
import tempfile
from typing import Mapping
import unittest
from unittest import mock
import weakref


METHOD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = METHOD_ROOT.parents[1]
MODULE_PATH = METHOD_ROOT / "train_elal3_c2_simulator_role_pair_v1.py"
C1_PATH = METHOD_ROOT / "train_elal3_c1_simulator_overfit_v1.py"
CORE_PATH = METHOD_ROOT / "elal3_c0_v1.py"
CONTRACT_PATH = (
    REPO_ROOT
    / "md/action_editing/20260817_box/evidence/"
    "elal3_c2_role_binding_experiment_contract_v1.json"
)
EXTERNAL_AUTHORITY_PATH = (
    REPO_ROOT
    / "md/action_editing/20260817_box/evidence/"
    "elal3_c2_simulator_optimizer_diagnostic_authority_v1.json"
)
PACKET_ROOT = REPO_ROOT / "md/action_editing/20260817_box/simulator_gt_canary_v1"
EXACT16_FIXTURE_ROOT = (
    REPO_ROOT
    / "md/action_editing/20260817_box/evidence/"
    "elal3_c2_exact16_materialization_r3_node226"
)

import sys

if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import train_elal3_c2_simulator_role_pair_v1 as trainer

try:
    import torch
    import elal3_simulator_c2_label_v1 as c2_label

    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    c2_label = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

try:
    import safetensors  # noqa: F401
    import materialize_elal3_simulator_c2_vae_v1 as c2_materializer

    _SAFETENSORS_AVAILABLE = _TORCH_AVAILABLE
except ModuleNotFoundError:
    c2_materializer = None  # type: ignore[assignment]
    _SAFETENSORS_AVAILABLE = False


class ELAL3C2TrainerStaticTests(unittest.TestCase):
    @staticmethod
    def _digest(label: str) -> str:
        return hashlib.sha256(label.encode("utf-8")).hexdigest()

    @classmethod
    def _portable_checkpoint_tree(cls, steps=(0, 1)):
        parameter_order = [
            *[f"block.{index}.lora_A" for index in range(480)],
            *[
                f"block.{index}.elal3_c0_v1.weight"
                for index in range(188)
            ],
        ]
        inventory = [
            {
                "name": name,
                "shape": [1],
                "dtype": "torch.float32",
                "numel": 1,
            }
            for name in parameter_order
        ]
        rows = []
        parameters = []
        for step in steps:
            parameter_sha = cls._digest(f"parameters:{step}")
            parameters.append(parameter_sha)
            file_order = ["adapter-and-elal3.pt"]
            if step:
                file_order.append("optimizer.pt")
            file_order.append("CHECKPOINT_RECEIPT.json")
            files = [
                {
                    "name": name,
                    "sha256": cls._digest(f"{step}:{name}"),
                    "size": 1,
                    "mode": 0o444,
                    "nlink": 1,
                    "held_fd_double_hash_verified": True,
                    "named_identity_replayed": True,
                }
                for name in file_order
            ]
            optimizer_digest = cls._digest(f"optimizer:{step}") if step else None
            optimizer_inventory = (
                {
                    "state_entry_count": 668,
                    "param_group_count": 1,
                    "parameter_count": 668,
                    "parameter_inventory_digest": trainer.object_sha256(inventory),
                    "optimizer_step": step,
                    "exp_avg_nonzero_parameter_count": 1,
                    "exp_avg_sq_nonzero_parameter_count": 1,
                    "state_keys_by_parameter": [
                        {
                            "parameter_id": index,
                            "state_keys": ["exp_avg", "exp_avg_sq", "step"],
                        }
                        for index in range(668)
                    ],
                    "tree_digest": optimizer_digest,
                }
                if step
                else None
            )
            unsigned = {
                "schema_version": trainer.CHECKPOINT_SCHEMA,
                "step": step,
                "file_order": file_order,
                "directory_entries": file_order,
                "directory_mode": 0o500,
                "files": files,
                "adapter_payload_tree_digest": cls._digest(f"adapter:{step}"),
                "parameter_order": parameter_order,
                "parameter_inventory": inventory,
                "optimizer_payload_tree_digest": optimizer_digest,
                "optimizer_state_inventory": optimizer_inventory,
                "checkpoint_receipt_digest": cls._digest(f"receipt:{step}"),
                "trainable_parameter_sha256": parameter_sha,
                "strict_reload_pass": True,
            }
            rows.append(
                {
                    **unsigned,
                    "portable_record_digest": trainer.object_sha256(unsigned),
                }
            )
        tree = {
            "schema_version": "bernini-elal3-c2-sealed-checkpoint-tree-v1",
            "expected_steps": list(steps),
            "directory_entries": [
                f"checkpoint-{step:08d}" for step in steps
            ],
            "directory_mode": 0o500,
            "portable_checkpoint_records": rows,
            "portable_checkpoint_tree_digest": trainer.object_sha256(rows),
            "physical_origin_replay_passed": True,
        }
        return tree, tuple(parameters)

    def test_source_parses_has_closed_markers_and_c1_is_untouched(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        ast.parse(source)
        for marker in (
            'ARM_DUPLICATE = "A_duplicate_control"',
            'ARM_ROLE_PAIR = "B_paired_role"',
            'ARM_ROLE_REPLICA = "B_paired_role_replica"',
            "MAX_STEPS = 10",
            'CONTROLLED_GAIN_FLOAT32_HEX = "3a5ef53f"',
            "def exact_two_branch_objective_v1",
            "def sequential_two_branch_objective_receipt_v1",
            "strict_sequential_forward_backward_release_then_next",
            "first_prediction_weakref_released_before_second_forward",
            "second_prediction_weakref_released_before_post_branch_work",
            "def full_q_route_matrix_v1",
            "def role_only_cell_v1",
            "def role_only_swap_invariants_v1",
            "def text_lens_runtime_list_abi_v1",
            "def prediction_hash_projection_v1",
            'PREDICTION_HASH_PROJECTION_PRODUCTION_DTYPE = "torch.bfloat16"',
            "PREDICTION_HASH_PROJECTION_PRODUCTION_STRIDE = (",
            'expected_original_device_index=int(row["world_rank"])',
            "controlled_target_prediction_projection = prediction_hash_projection_v1(",
            "prediction10_projection = prediction_hash_projection_v1(",
            '"canonical_typed_list_digest": object_sha256(typed)',
            "batch_text_seqlen=text_lens",
            "def validate_fresh1_acceptance_gate_v1",
            "def aggregate_preoptimizer_evidence_v1",
            '"participant_role_binding_claim_forbidden": True',
            '"frozen_teacher_used": False',
            '"frozen_velocity_reference_used": False',
            '"reward_used": False',
        ):
            self.assertIn(marker, source)
        for forbidden in (
            "frozen_source_action_velocity",
            "teacher_unit",
            "self_distillation_loss",
            "saic_event_reward",
        ):
            self.assertNotIn(forbidden, source)
        self.assertNotIn("text_lens.detach()", source)
        self.assertEqual(
            hashlib.sha256(C1_PATH.read_bytes()).hexdigest(),
            trainer.C1_TRAINER_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(CORE_PATH.read_bytes()).hexdigest(),
            trainer.C1_CORE_SHA256,
        )

    def test_runtime_source_orders_graph_release_before_next_branch_and_collectives(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        training = source[
            source.index("for step_zero in range(args.max_steps):") : source.index(
                "    final_digest = c1.trainable_digest_v1(named)"
            )
        ]
        markers = (
            "optimizer.zero_grad(set_to_none=True)",
            "first_backward_loss.backward()",
            "del first, first_branch_loss, first_weighted_loss, first_backward_loss",
            'fail("C2 first training graph survived before second forward")',
            "second = renderer_branch_forward_v1(",
            "second_backward_loss.backward()",
            "del second, second_branch_loss, second_weighted_loss, second_backward_loss",
            'fail("C2 second training graph survived before gradient reduction")',
            "c1.synchronize_gradients_v1(named, parallel)",
            "torch.nn.utils.clip_grad_norm_(",
            "optimizer.step()",
        )
        positions = [training.index(marker) for marker in markers]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(training.count("c1.all_trainable_graph_zero_v1("), 1)
        self.assertNotIn("exact_two_branch_objective_v1(", training)

        preflight = source[
            source.index("# Every stage replays the same actual-shape") : source.index(
                "    evaluation0, controlled_target_prediction = evaluate_local_row_v1("
            )
        ]
        preflight_markers = (
            "training_first = renderer_branch_forward_v1(",
            "del training_first, preoptimizer_first_loss",
            'fail("C2 preoptimizer first branch graph survived inter-branch release")',
            "training_second = renderer_branch_forward_v1(",
            "del training_second, preoptimizer_second_loss",
            'fail("C2 preoptimizer second branch graph survived post-branch release")',
        )
        positions = [preflight.index(marker) for marker in preflight_markers]
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn(".backward()", preflight)
        self.assertNotIn("torch.no_grad", preflight)

    def test_real_preoptimizer_runtime_row_id_and_all8_hostiles(self) -> None:
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        dictionary_keys = {}
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Dict)
            ):
                dictionary_keys[node.targets[0].id] = {
                    key.value
                    for key in node.value.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                }
        placement = {"world_rank", "row_index", "row_id", "sp_rank"}
        self.assertTrue(
            placement.issubset(dictionary_keys["preoptimizer_runtime"])
        )
        self.assertTrue(placement.issubset(dictionary_keys["local_step"]))

        rows = []
        for world_rank in range(trainer.WORLD_SIZE):
            row_index = world_rank // trainer.SP_SIZE
            first = {
                "coordinate": dict(trainer.TRAINING_SIGMA_EXACT10[0]),
                "coordinate_kind": "training_sigma_stratum",
                "actual_input_digest": self._digest(f"first:{world_rank}"),
            }
            second = {
                "coordinate": dict(trainer.TRAINING_SIGMA_EXACT10[0]),
                "coordinate_kind": "training_sigma_stratum",
                "actual_input_digest": self._digest(f"second:{world_rank}"),
            }
            rows.append(
                {
                    "world_rank": world_rank,
                    "row_index": row_index,
                    "row_id": trainer.ROW_IDS[row_index],
                    "sp_rank": world_rank % trainer.SP_SIZE,
                    "first_hook": {},
                    "second_hook": {},
                    "first_partition": {},
                    "second_partition": {},
                    "first_actual_input_receipt": first,
                    "second_actual_input_receipt": second,
                    "objective": {
                        "first_actual_input_digest": first["actual_input_digest"],
                        "second_actual_input_digest": second["actual_input_digest"],
                        "actual_branch_inputs_closed_and_verified": True,
                        "execution_mode": "preflight_forward_only",
                    },
                    "branch_lifecycle": {},
                    "memory": {},
                    "optimizer_constructed": False,
                }
            )
        with mock.patch.multiple(
            trainer,
            _validate_hook_receipt_closed_v1=mock.DEFAULT,
            _validate_partition_closed_v1=mock.DEFAULT,
            _validate_objective_receipt_closed_v1=mock.DEFAULT,
            validate_branch_lifecycle_receipt_v1=mock.DEFAULT,
            _validate_actual_branch_pair_closed_v1=mock.DEFAULT,
            _validate_memory_row_closed_v1=mock.DEFAULT,
        ) as patched:
            patched["validate_branch_lifecycle_receipt_v1"].return_value = {
                "execution_mode": "preflight_forward_only"
            }
            trainer._validate_all8_graph_rows_closed_v1(
                rows,
                arm_id=trainer.ARM_DUPLICATE,
                completed_step=None,
                label="real preoptimizer fixture",
            )
            omitted = copy.deepcopy(rows)
            omitted[3].pop("row_id")
            with self.assertRaisesRegex(
                trainer.ELAL3C2TrainingError, "rank/DP2xSP4 placement"
            ):
                trainer._validate_all8_graph_rows_closed_v1(
                    omitted,
                    arm_id=trainer.ARM_DUPLICATE,
                    completed_step=None,
                    label="omitted row_id",
                )
            resigned_wrong = copy.deepcopy(rows)
            resigned_wrong[4]["row_id"] = trainer.ROW_IDS[0]
            with self.assertRaisesRegex(
                trainer.ELAL3C2TrainingError, "rank/DP2xSP4 placement"
            ):
                trainer._validate_all8_graph_rows_closed_v1(
                    resigned_wrong,
                    arm_id=trainer.ARM_DUPLICATE,
                    completed_step=None,
                    label="resigned wrong row_id",
                )

    def test_final_experiment_contract_literal_and_closed_recipe(self) -> None:
        value = trainer.validate_experiment_contract_v1(
            CONTRACT_PATH,
            expected_sha256=trainer.EXPERIMENT_CONTRACT_SHA256,
        )
        self.assertEqual(
            value["selection_rule"]["primary_metric"],
            "minimum_of_four_role_only_matched_vs_mismatch_margins",
        )
        self.assertTrue(
            value["comparison_contract"]["arms_are_independent_world8_runs"]
        )
        with self.assertRaisesRegex(
            trainer.ELAL3C2TrainingError, "literal SHA"
        ):
            trainer.validate_experiment_contract_v1(
                CONTRACT_PATH, expected_sha256="0" * 64
            )

    def test_external_authority_is_exact2_max10_and_no_claim(self) -> None:
        value = trainer.validate_external_authority_v1(
            EXTERNAL_AUTHORITY_PATH,
            expected_sha256=trainer.EXTERNAL_AUTHORITY_SHA256,
        )
        self.assertEqual(tuple(value["authorized_row_ids"]), trainer.ROW_IDS)
        self.assertEqual(value["max_optimizer_updates_per_arm"], 10)
        self.assertTrue(
            value["training_objective_restrictions"]
            ["target_grounded_event_and_context_flow_only"]
        )

    def test_parser_closes_arm_seed_steps_pins_and_five_acks(self) -> None:
        digest = "1" * 64
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory) / "elal3_c2_A_fresh")
            argv = [
                "--arm-id", trainer.ARM_DUPLICATE,
                "--bernini-root", "/b",
                "--veomni-root", "/v",
                "--checkpoint", "/c",
                "--checkpoint-exact23-manifest", "/checkpoint.sha256",
                "--expected-checkpoint-exact23-manifest-sha256",
                trainer.CHECKPOINT_EXACT23_MANIFEST_SHA256,
                "--packet-root", "/p",
                "--latent-bundle", "/l",
                "--expected-latent-bundle-sha256", digest,
                "--latent-bundle-receipt", "/r",
                "--expected-latent-bundle-receipt-sha256", digest,
                "--materializer-run-complete", "/RUN_COMPLETE.json",
                "--expected-materializer-run-complete-sha256",
                trainer.MATERIALIZER_RUN_COMPLETE_SHA256,
                "--external-authority", "/a",
                "--expected-external-authority-sha256",
                trainer.EXTERNAL_AUTHORITY_SHA256,
                "--model-authority", "/m",
                "--expected-model-authority-sha256",
                trainer.MODEL_AUTHORITY_SHA256,
                "--experiment-contract", "/e",
                "--expected-experiment-contract-sha256",
                trainer.EXPERIMENT_CONTRACT_SHA256,
                "--output", output,
                "--max-steps", "10",
                "--seed", "20260821",
                "--expected-runner-source-sha256", digest,
                "--expected-c1-trainer-source-sha256", trainer.C1_TRAINER_SHA256,
                "--expected-elal3-core-source-sha256", trainer.C1_CORE_SHA256,
                "--expected-c2-label-source-sha256", trainer.C2_LABEL_SHA256,
                "--expected-c2-materializer-source-sha256", trainer.C2_MATERIALIZER_SHA256,
                "--expected-train-lora-source-sha256", trainer.TRAIN_LORA_SHA256,
                "--expected-packed-lora-source-sha256", trainer.PACKED_LORA_SHA256,
                "--expected-runtime-source-sha256", trainer.RUNTIME_SHA256,
                "--expected-sigma-source-sha256", trainer.SIGMA_SHA256,
                "--preflight-only",
                "--ack-simulator-oracle-q-diagnostic-only",
                "--ack-not-source-instruction-inference",
                "--ack-not-formal-c2",
                "--ack-not-exact160",
                "--ack-no-real-video-or-scientific-claim",
            ]
            args = trainer.parser().parse_args(argv)
            trainer.validate_args_static_v1(args)
            args.seed = 20260822
            with self.assertRaisesRegex(
                trainer.ELAL3C2TrainingError, "seed differs"
            ):
                trainer.validate_args_static_v1(args)
            args.seed = 20260821
            args.ack_not_exact160 = False
            with self.assertRaisesRegex(trainer.ELAL3C2TrainingError, "five"):
                trainer.validate_args_static_v1(args)
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                trainer.parser().parse_args(
                    [*argv[: argv.index("10")], "20", *argv[argv.index("10") + 1 :]]
                )

    def test_evaluation_seed_abi_is_exact(self) -> None:
        coordinate = trainer.EvaluationCoordinateV1()
        self.assertEqual(coordinate.as_dict()["timestep"], 999)
        self.assertEqual(coordinate.as_dict()["renderer_timestep_dtype"], "torch.int64")
        self.assertEqual(coordinate.as_dict()["sigma_float32_be_hex"], "3f800000")
        self.assertEqual(trainer.evaluation_noise_seed_v1(20260821, 0), 2026082100)
        self.assertEqual(trainer.evaluation_noise_seed_v1(20260821, 1), 2026082101)
        with self.assertRaises(trainer.ELAL3C2TrainingError):
            trainer.evaluation_noise_seed_v1(20260821, 2)

    def test_bundle_relocation_is_allowed_but_live_binding_tamper_is_not(self) -> None:
        digest = "a" * 64
        original = {
            "path": "/materializer/output/c2-exact16-latents.safetensors",
            "sha256": digest,
            "size": 123,
            "mode": 0o444,
            "nlink": 1,
            "held_fd_double_read_verified": True,
            "held_openat_parent_chain_replayed": True,
        }
        live = {
            **original,
            "path": "/release/input/c2-exact16-latents.safetensors",
        }
        receipt = trainer.validate_bundle_relocation_binding_v1(
            receipt_bundle_binding=original,
            live_bundle_binding=live,
            runtime_path=Path(live["path"]),
            expected_sha256=digest,
            expected_size=123,
        )
        self.assertFalse(receipt["same_path_required"])
        self.assertNotEqual(
            receipt["original_materialization_path"], receipt["runtime_path"]
        )
        with self.assertRaisesRegex(
            trainer.ELAL3C2TrainingError, "relocation binding"
        ):
            trainer.validate_bundle_relocation_binding_v1(
                receipt_bundle_binding=original,
                live_bundle_binding={**live, "size": 124},
                runtime_path=Path(live["path"]),
                expected_sha256=digest,
                expected_size=123,
            )

    def test_main_is_linked_but_bundle_release_none_fails_before_model(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("Runtime body is linked after", source)
        self.assertIn("require_bundle_release_literals_v1", source)
        names = (
            "LATENT_BUNDLE_SHA256",
            "LATENT_BUNDLE_SIZE",
            "LATENT_BUNDLE_RECEIPT_SHA256",
            "LATENT_BUNDLE_RECEIPT_SIZE",
            "LATENT_BUNDLE_RECEIPT_DIGEST",
        )
        saved = {name: getattr(trainer, name) for name in names}
        try:
            for name in names:
                setattr(trainer, name, None)
            with self.assertRaisesRegex(
                trainer.ELAL3C2TrainingError, "bundle release literals"
            ):
                trainer.require_bundle_release_literals_v1(
                    expected_bundle_sha256="a" * 64,
                    expected_receipt_sha256="b" * 64,
                    expected_run_complete_sha256="c" * 64,
                )
        finally:
            for name, value in saved.items():
                setattr(trainer, name, value)

    def test_portable_checkpoint_replays_exact668_and_optimizer_state(self) -> None:
        tree, parameters = self._portable_checkpoint_tree()
        trainer._validate_portable_checkpoint_tree_v1(
            tree, expected_steps=(0, 1), expected_parameters=parameters
        )
        tampered = __import__("copy").deepcopy(tree)
        optimizer = tampered["portable_checkpoint_records"][1][
            "optimizer_state_inventory"
        ]
        optimizer["state_keys_by_parameter"] = optimizer[
            "state_keys_by_parameter"
        ][:-1]
        row = tampered["portable_checkpoint_records"][1]
        unsigned_row = dict(row)
        unsigned_row.pop("portable_record_digest")
        row["portable_record_digest"] = trainer.object_sha256(unsigned_row)
        tampered["portable_checkpoint_tree_digest"] = trainer.object_sha256(
            tampered["portable_checkpoint_records"]
        )
        with self.assertRaisesRegex(
            trainer.ELAL3C2TrainingError, "optimizer exact668"
        ):
            trainer._validate_portable_checkpoint_tree_v1(
                tampered, expected_steps=(0, 1), expected_parameters=parameters
            )

    def test_exact10_attestation_is_bound_to_sealed_predecessor_gates(self) -> None:
        tree, parameters = self._portable_checkpoint_tree((0, 10))
        cross = {
            "gate_sha256": self._digest("cross-file"),
            "gate_digest": self._digest("cross-value"),
            "recipe_version_digest": self._digest("cross-recipe"),
        }
        fresh = {
            "gate_sha256": self._digest("fresh-file"),
            "gate_digest": self._digest("fresh-value"),
            "cross_arm_gate_sha256": cross["gate_sha256"],
            "cross_arm_gate_digest": cross["gate_digest"],
            "cross_arm_recipe_version_digest": cross["recipe_version_digest"],
        }
        origin = {
            "name": "origin-verifier.py",
            "sha256": self._digest("origin-verifier"),
            "size": 1,
            "mode": 0o444,
            "nlink": 1,
        }
        controller = {
            "name": "gate-controller.py",
            "sha256": self._digest("gate-controller"),
            "size": 1,
            "mode": 0o444,
            "nlink": 1,
        }
        unsigned = {
            "schema_version": trainer.EXACT10_ORIGIN_ATTESTATION_SCHEMA,
            "status": "EXACT10_ORIGIN_PHYSICAL_REPLAY_PASS",
            "stage": "exact10",
            "arm_id": trainer.ARM_DUPLICATE,
            "holder_job_id": "141620",
            "node": "auh7-1b-gpu-226",
            "seed": 20260821,
            "receipt_sha256": self._digest("exact10-receipt-file"),
            "receipt_size": 1,
            "receipt_digest": self._digest("exact10-receipt-value"),
            "receipt_status": "EXACT10_LATENT_GATES_PASS_DECODED_REVIEW_PENDING",
            "initial_trainable_sha256": parameters[0],
            "final_trainable_sha256": parameters[1],
            "common_comparison_payload_digest": self._digest("common"),
            "row_input_noise_schedule_digest": self._digest("schedule"),
            "history_digest": self._digest("history"),
            "portable_checkpoint_tree": tree,
            "portable_checkpoint_tree_digest": tree[
                "portable_checkpoint_tree_digest"
            ],
            "cross_arm_gate_sha256": cross["gate_sha256"],
            "cross_arm_gate_digest": cross["gate_digest"],
            "fresh1_acceptance_gate_sha256": fresh["gate_sha256"],
            "fresh1_acceptance_gate_digest": fresh["gate_digest"],
            "latent_hard_gates_pass": True,
            "decoded_track_effect_gate_pending": True,
            "runner_source_sha256": self._digest("runner"),
            "latent_bundle_sha256": self._digest("bundle"),
            "source_pins": {},
            "experiment_contract_sha256": trainer.EXPERIMENT_CONTRACT_SHA256,
            "external_authority_sha256": trainer.EXTERNAL_AUTHORITY_SHA256,
            "model_authority_sha256": trainer.MODEL_AUTHORITY_SHA256,
            "materializer_run_complete_sha256": trainer.MATERIALIZER_RUN_COMPLETE_SHA256,
            "materializer_run_complete_digest": trainer.MATERIALIZER_RUN_COMPLETE_DIGEST,
            "checkpoint_exact23_binding_digest": self._digest("exact23"),
            "bernini_execution_source_binding_digest": self._digest("bernini"),
            "origin_verifier_binding": origin,
            "gate_controller_binding": controller,
            "physical_origin_replay_passed": True,
            "closed_validator_passed": True,
        }
        attestation = {
            **unsigned,
            "attestation_digest": trainer.object_sha256(unsigned),
        }
        kwargs = {
            "arm_id": trainer.ARM_DUPLICATE,
            "expected_runner_sha256": unsigned["runner_source_sha256"],
            "expected_bundle_sha256": unsigned["latent_bundle_sha256"],
            "expected_source_pins": {},
            "expected_cross_gate_binding": cross,
            "expected_fresh1_gate_binding": fresh,
            "expected_origin_verifier_binding": origin,
            "expected_gate_controller_binding": controller,
        }
        trainer._validate_exact10_origin_attestation_value_v1(
            attestation, **kwargs
        )
        tampered = dict(attestation)
        tampered["cross_arm_gate_sha256"] = self._digest("attacker-cross")
        unsigned_tampered = dict(tampered)
        unsigned_tampered.pop("attestation_digest")
        tampered["attestation_digest"] = trainer.object_sha256(unsigned_tampered)
        with self.assertRaisesRegex(
            trainer.ELAL3C2TrainingError, "attestation envelope"
        ):
            trainer._validate_exact10_origin_attestation_value_v1(
                tampered, **kwargs
            )


@unittest.skipUnless(_TORCH_AVAILABLE, "torch runtime is required")
class ELAL3C2TrainerTensorTests(unittest.TestCase):
    @staticmethod
    def _partition():
        event = torch.tensor([[[True, True, False, False]]])
        return event, ~event

    @staticmethod
    def _sha(label: str) -> str:
        return hashlib.sha256(label.encode("utf-8")).hexdigest()

    def test_torch27_adamw_decoupled_weight_decay_exact_roundtrip(self) -> None:
        named = [
            (f"parameter.{index}", torch.nn.Parameter(torch.ones(1)))
            for index in range(668)
        ]
        optimizer = torch.optim.AdamW(
            [parameter for _, parameter in named],
            lr=trainer.DEFAULT_LR,
            betas=(0.9, 0.95),
            eps=1.0e-8,
            weight_decay=0.0,
        )
        # Torch 2.7.1 emits this fixed AdamW field.  The lab image predates it,
        # so inject the exact observed live ABI before exercising state_dict.
        optimizer.param_groups[0]["decoupled_weight_decay"] = True
        sum(parameter.sum() for _, parameter in named).backward()
        optimizer.step()
        inventory = trainer.c1.trainable_inventory_v1(named)
        live = optimizer.state_dict()
        live_receipt = trainer._optimizer_state_inventory_v1(
            live,
            expected_parameter_inventory=inventory,
            expected_step=1,
        )
        stream = io.BytesIO()
        torch.save(live, stream)
        stream.seek(0)
        loaded = torch.load(stream, map_location="cpu", weights_only=True)
        loaded_receipt = trainer._optimizer_state_inventory_v1(
            loaded,
            expected_parameter_inventory=inventory,
            expected_step=1,
        )
        self.assertEqual(live_receipt, loaded_receipt)
        self.assertIs(loaded["param_groups"][0]["decoupled_weight_decay"], True)

        hostiles = []
        false_value = copy.deepcopy(loaded)
        false_value["param_groups"][0]["decoupled_weight_decay"] = False
        hostiles.append(false_value)
        integer_decoy = copy.deepcopy(loaded)
        integer_decoy["param_groups"][0]["decoupled_weight_decay"] = 1
        hostiles.append(integer_decoy)
        missing = copy.deepcopy(loaded)
        missing["param_groups"][0].pop("decoupled_weight_decay")
        hostiles.append(missing)
        extra = copy.deepcopy(loaded)
        extra["param_groups"][0]["resigned_decoy"] = True
        hostiles.append(extra)
        for hostile in hostiles:
            with self.subTest(keys=sorted(hostile["param_groups"][0])):
                with self.assertRaisesRegex(
                    trainer.ELAL3C2TrainingError,
                    "fixed param-group/ID closure",
                ):
                    trainer._optimizer_state_inventory_v1(
                        hostile,
                        expected_parameter_inventory=inventory,
                        expected_step=1,
                    )

    def _actual_branch_receipt(
        self,
        *,
        variant: str,
        donor: str,
        route: str,
        prediction,
        velocity,
        event,
        context,
        epsilon_donor: str = "shared-epsilon",
    ):
        q_rows = {
            name: {
                "shape": list(shape),
                "dtype": dtype,
                "sha256": self._sha(f"{donor}:{name}"),
            }
            for name, (shape, dtype) in trainer.ACTUAL_Q_ABI.items()
        }
        coordinate = dict(trainer.TRAINING_SIGMA_EXACT10[0])
        text_lens_abi = trainer.text_lens_runtime_list_abi_v1([512])
        unsigned = {
            "row_id": trainer.ROW_IDS[0],
            "input_variant": variant,
            "label_binding_digest": self._sha(f"{donor}:label"),
            "actual_q_tensor_rows": q_rows,
            "actual_q_tensor_rows_digest": trainer.object_sha256(q_rows),
            "source_sha256": self._sha("shared-source"),
            "clean_target_sha256": self._sha(f"{donor}:clean"),
            "epsilon_sha256": self._sha(epsilon_donor),
            "noisy_target_sha256": self._sha(f"{donor}:noisy:{epsilon_donor}"),
            "target_velocity_sha256": self._sha(f"{donor}:vae-velocity"),
            "event_mask_vae_sha256": self._sha(f"{donor}:vae-event"),
            "context_mask_vae_sha256": self._sha(f"{donor}:vae-context"),
            "text_lens_runtime_abi": text_lens_abi,
            "text_lens_sha256": text_lens_abi[
                "audit_only_cpu_int64_tensor_sha256"
            ],
            "text_embs_sha256": self._sha("shared-text-embs"),
            "coordinate": coordinate,
            "coordinate_kind": "training_sigma_stratum",
            "renderer_timestep_receipt": {
                "timestep_cpu_origin": True,
                "timestep_dtype": "torch.int64",
                "timestep_value": coordinate["timestep"],
                "sigma_float32_be_hex": coordinate["sigma_float32_be_hex"],
            },
            "route_identity": route,
            "registered_sp4_partition": {"sp_rank": 0},
            "all30_hooks_used": True,
            "prediction_sha256": trainer.c1.tensor_sha256_v1(
                prediction.detach().contiguous().cpu()
            ),
            "packed_target_velocity_sha256": trainer.c1.tensor_sha256_v1(
                velocity.detach().contiguous().cpu()
            ),
            "packed_event_mask_sha256": trainer.c1.tensor_sha256_v1(
                event.detach().contiguous().cpu()
            ),
            "packed_context_mask_sha256": trainer.c1.tensor_sha256_v1(
                context.detach().contiguous().cpu()
            ),
        }
        return {**unsigned, "actual_input_digest": trainer.object_sha256(unsigned)}

    @staticmethod
    def _resign_actual(receipt, **changes):
        unsigned = dict(receipt)
        unsigned.pop("actual_input_digest")
        unsigned.update(changes)
        return {**unsigned, "actual_input_digest": trainer.object_sha256(unsigned)}

    def test_renderer_receives_cpu_origin_int64_exact999(self) -> None:
        captured = {}

        class Decoder:
            def shared_step(self, **kwargs):
                captured.update(kwargs)
                return torch.zeros((1, 5, trainer.c1.PATCH_VALUES))

        packed = {
            "embedded": torch.zeros((1, 5, trainer.HIDDEN)),
            "rotary": torch.zeros((2, 5, 3)),
            "total_tokens": 5,
            "source_tokens": 2,
            "target_tokens": 3,
        }
        text_lens = [512]
        prediction, receipt, text_lens_abi = trainer.predict_target_c2_v1(
            renderer=SimpleNamespace(diff_dec=Decoder()),
            packed=packed,
            coordinate=trainer.EvaluationCoordinateV1(),
            text_lens=text_lens,
            text_embs=torch.zeros((1, 1, 4)),
        )
        self.assertEqual(tuple(prediction.shape), (1, 3, trainer.c1.PATCH_VALUES))
        self.assertEqual(captured["timesteps"].device.type, "cpu")
        self.assertEqual(captured["timesteps"].dtype, torch.int64)
        self.assertEqual(captured["timesteps"].tolist(), [999])
        self.assertIs(captured["batch_text_seqlen"], text_lens)
        self.assertEqual(captured["batch_text_seqlen"], [512])
        self.assertEqual(
            text_lens_abi,
            trainer.text_lens_runtime_list_abi_v1([512]),
        )
        self.assertEqual(
            text_lens_abi["canonical_typed_list_digest"],
            trainer.object_sha256(
                {
                    key: text_lens_abi[key]
                    for key in (
                        "schema_version",
                        "container",
                        "length",
                        "element_type",
                        "values",
                        "allowed_value_range_inclusive",
                        "exact_runtime_value_required",
                    )
                }
            ),
        )
        self.assertTrue(receipt["timestep_cpu_origin"])

    def test_text_lens_runtime_list_abi_rejects_tensor_decoys_and_malformed(self) -> None:
        expected = trainer.text_lens_runtime_list_abi_v1([512])
        self.assertEqual(expected["container"], "python_list")
        self.assertFalse(expected["tensor_substitution_into_shared_step"])
        for hostile in (
            torch.tensor([512]),
            (512,),
            [],
            [512, 512],
            [True],
            [0],
            [513],
            [511],
            [512.0],
        ):
            with self.subTest(hostile=repr(hostile)):
                with self.assertRaisesRegex(
                    trainer.ELAL3C2TrainingError, "exact built-in Python list"
                ):
                    trainer.text_lens_runtime_list_abi_v1(hostile)

    def test_predict_rejects_tensor_before_shared_step_and_list_mutation_after(self) -> None:
        calls = []

        class Decoder:
            def shared_step(self, **kwargs):
                calls.append(kwargs)
                kwargs["batch_text_seqlen"][0] = 511
                return torch.zeros((1, 5, trainer.c1.PATCH_VALUES))

        packed = {
            "embedded": torch.zeros((1, 5, trainer.HIDDEN)),
            "rotary": torch.zeros((2, 5, 3)),
            "total_tokens": 5,
            "source_tokens": 2,
            "target_tokens": 3,
        }
        with self.assertRaisesRegex(
            trainer.ELAL3C2TrainingError, "exact built-in Python list"
        ):
            trainer.predict_target_c2_v1(
                renderer=SimpleNamespace(diff_dec=Decoder()),
                packed=packed,
                coordinate=trainer.EvaluationCoordinateV1(),
                text_lens=torch.tensor([512]),
                text_embs=torch.zeros((1, 1, 4)),
            )
        self.assertEqual(calls, [])
        with self.assertRaisesRegex(
            trainer.ELAL3C2TrainingError, "exact built-in Python list"
        ):
            trainer.predict_target_c2_v1(
                renderer=SimpleNamespace(diff_dec=Decoder()),
                packed=packed,
                coordinate=trainer.EvaluationCoordinateV1(),
                text_lens=[512],
                text_embs=torch.zeros((1, 1, 4)),
            )
        self.assertEqual(len(calls), 1)

    def test_actual_receipt_rejects_resigned_text_lens_decoy(self) -> None:
        event, context = self._partition()
        prediction = torch.zeros((1, 1, 4))
        velocity = torch.ones((1, 1, 4))
        receipt = self._actual_branch_receipt(
            variant="target",
            donor="target",
            route="text-lens-real",
            prediction=prediction,
            velocity=velocity,
            event=event,
            context=context,
        )
        trainer._validate_actual_branch_receipt_closed_v1(
            receipt, label="positive text_lens ABI"
        )
        fake_abi = dict(receipt["text_lens_runtime_abi"])
        fake_abi["values"] = [511]
        fake_abi["canonical_typed_list_digest"] = trainer.object_sha256(
            {
                key: fake_abi[key]
                for key in (
                    "schema_version",
                    "container",
                    "length",
                    "element_type",
                    "values",
                    "allowed_value_range_inclusive",
                    "exact_runtime_value_required",
                )
            }
        )
        hostile = self._resign_actual(receipt, text_lens_runtime_abi=fake_abi)
        with self.assertRaisesRegex(
            trainer.ELAL3C2TrainingError, "semantic closure"
        ):
            trainer._validate_actual_branch_receipt_closed_v1(
                hostile, label="resigned text_lens decoy"
            )
        hostile_sha = self._resign_actual(
            receipt, text_lens_sha256=self._sha("decoy-text-lens")
        )
        with self.assertRaisesRegex(
            trainer.ELAL3C2TrainingError, "semantic closure"
        ):
            trainer._validate_actual_branch_receipt_closed_v1(
                hostile_sha, label="resigned text_lens SHA decoy"
            )

    def test_held_live_bundle_bytes_reject_tamper_under_same_expected_sha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = root / "relocated.safetensors"
            original = b"sealed-exact16-fixture"
            path.write_bytes(original)
            path.chmod(0o444)
            digest = hashlib.sha256(original).hexdigest()
            payload, binding = c2_label.stable_read_path(
                path,
                label="relocated bundle fixture",
                expected_sha256=digest,
                expected_mode=0o444,
                allowed_root=root,
            )
            self.assertEqual(payload, original)
            self.assertEqual(binding["sha256"], digest)
            path.chmod(0o644)
            path.write_bytes(b"tampered-exact16-fixture")
            path.chmod(0o444)
            with self.assertRaisesRegex(Exception, "SHA replay differs"):
                c2_label.stable_read_path(
                    path,
                    label="relocated bundle fixture",
                    expected_sha256=digest,
                    expected_mode=0o444,
                    allowed_root=root,
                )

    @unittest.skipUnless(
        _SAFETENSORS_AVAILABLE and EXACT16_FIXTURE_ROOT.is_dir(),
        "safetensors and frozen retry2 exact16 fixture are required",
    )
    def test_real_retry2_exact16_receipt_and_relocated_bundle_load(self) -> None:
        # The local ``vd`` environment has a NumPy-2/old-Torch ABI mismatch;
        # hash the same canonical CPU bytes without NumPy for this fixture test.
        import ctypes

        original_hash = trainer.c1.tensor_sha256_v1

        def tensor_sha_without_numpy(value):
            tensor = value.detach().cpu().contiguous()
            header = trainer.c1.canonical_json_bytes(
                {
                    "dtype": str(tensor.dtype),
                    "shape": [int(item) for item in tensor.shape],
                }
            )
            raw = ctypes.string_at(
                tensor.data_ptr(), tensor.numel() * tensor.element_size()
            )
            return hashlib.sha256(header + b"\0" + raw).hexdigest()

        try:
            trainer.c1.tensor_sha256_v1 = tensor_sha_without_numpy
            run = trainer.validate_materializer_run_complete_v1(
                EXACT16_FIXTURE_ROOT / "RUN_COMPLETE.json",
                expected_sha256=trainer.MATERIALIZER_RUN_COMPLETE_SHA256,
                label_module=c2_label,
            )
            bundle = trainer.load_c2_latent_bundle_v1(
                bundle_path=(
                    EXACT16_FIXTURE_ROOT / "c2-exact16-latents.safetensors"
                ),
                expected_bundle_sha256=trainer.LATENT_BUNDLE_SHA256,
                receipt_path=EXACT16_FIXTURE_ROOT / "latent-bundle-receipt.json",
                expected_receipt_sha256=trainer.LATENT_BUNDLE_RECEIPT_SHA256,
                packet_root=PACKET_ROOT,
                local_row_index=0,
                label_module=c2_label,
                materializer_module=c2_materializer,
            )
        finally:
            trainer.c1.tensor_sha256_v1 = original_hash
        self.assertEqual(run["status"], "COMPLETE_SIMULATOR_C2_EXACT16_ONLY")
        self.assertEqual(bundle.local_row_id, trainer.ROW_IDS[0])
        self.assertEqual(set(bundle.local_tensors), {"source", "target", "role_swap"})
        self.assertTrue(
            all(
                tuple(tensor.shape) == trainer.LATENT_SHAPE
                for tensor in bundle.local_tensors.values()
            )
        )

    def test_common_noise_schedule_hashes_noisy_input_and_velocity(self) -> None:
        class Coordinate:
            def __init__(self, step):
                self.optimizer_step = step
                self.sigma = float(torch.tensor((step + 1) / 20, dtype=torch.float32))

            def as_dict(self):
                return {
                    "optimizer_step": self.optimizer_step,
                    "sigma": self.sigma,
                    "sigma_float32_be_hex": __import__("struct").pack(
                        ">f", self.sigma
                    ).hex(),
                }

        sigma = SimpleNamespace(select_sigma_stratum=lambda step: Coordinate(step))
        original_shape = trainer.LATENT_SHAPE
        try:
            trainer.LATENT_SHAPE = (1, 1, 1, 1, 2)
            schedules = trainer.build_noise_schedule_receipt_v1(
                arm_seed=20260821,
                targets_by_row={
                    0: torch.zeros(trainer.LATENT_SHAPE, dtype=torch.float32),
                    1: torch.ones(trainer.LATENT_SHAPE, dtype=torch.float32),
                },
                sigma_module=sigma,
            )
        finally:
            trainer.LATENT_SHAPE = original_shape
        self.assertEqual(
            len(schedules["training_exact10_common_target_branch"]), 20
        )
        row = schedules["training_exact10_common_target_branch"][0]
        for field in (
            "epsilon_sha256",
            "target_sha256",
            "noisy_target_sha256",
            "target_velocity_sha256",
            "common_target_input_digest",
        ):
            self.assertRegex(row[field], r"^[0-9a-f]{64}$")

    def test_controlled_gate_is_analytic_small_nonzero_and_restored(self) -> None:
        class Injection(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.residual_gain = torch.nn.Parameter(torch.ones(()))

        handle = SimpleNamespace(
            components=SimpleNamespace(
                injections=torch.nn.ModuleList(
                    [Injection() for _ in range(trainer.BLOCKS)]
                )
            )
        )
        receipt = trainer.install_controlled_nonzero_gates_v1(handle)
        self.assertEqual(receipt["float32_be_hex"], "3a5ef53f")
        before = [item.residual_gain.detach().clone() for item in handle.components.injections]
        with trainer.temporary_gate_zero_probe_v1(handle):
            self.assertTrue(
                all(float(item.residual_gain.item()) == 0.0 for item in handle.components.injections)
            )
        self.assertTrue(
            all(torch.equal(item.residual_gain.detach(), value) for item, value in zip(handle.components.injections, before))
        )
        safety = trainer.step0_gain_safety_receipt_v1(
            torch.ones((1, 8)),
            torch.ones((1, 8)) * 1.01,
            parameter_digest_before="a" * 64,
            parameter_digest_after="a" * 64,
        )
        self.assertTrue(safety["finite_nonzero_bounded"])
        self.assertFalse(safety["semantic_diagonal_dominance_required"])

    def test_noncontiguous_eval_prediction_hash_uses_canonical_audit_copy_only(self) -> None:
        base = torch.arange(
            19_110 * trainer.c1.PATCH_VALUES,
            dtype=torch.float32,
        ).reshape(1, trainer.c1.PATCH_VALUES, 19_110)
        prediction = base.transpose(1, 2)
        self.assertEqual(tuple(prediction.shape), (1, 19_110, trainer.c1.PATCH_VALUES))
        self.assertFalse(prediction.is_contiguous())
        identity = id(prediction)
        pointer = prediction.untyped_storage().data_ptr()
        version = prediction._version
        stride = prediction.stride()
        with self.assertRaisesRegex(
            Exception, "contiguous CPU strided tensor"
        ):
            trainer.c1.tensor_sha256_v1(prediction)
        receipt = trainer.prediction_hash_projection_v1(
            prediction,
            label="unit noncontiguous eval prediction",
        )
        expected = trainer.c1.tensor_sha256_v1(
            prediction.detach().to(device="cpu").contiguous()
        )
        self.assertEqual(receipt["prediction_sha256"], expected)
        self.assertFalse(receipt["original_is_contiguous"])
        self.assertTrue(receipt["projection_is_contiguous"])
        self.assertEqual(receipt["projection_device_type"], "cpu")
        self.assertEqual(
            receipt["projection_stride"],
            [19_110 * trainer.c1.PATCH_VALUES, trainer.c1.PATCH_VALUES, 1],
        )
        self.assertTrue(receipt["audit_projection_only"])
        self.assertEqual(id(prediction), identity)
        self.assertEqual(prediction.untyped_storage().data_ptr(), pointer)
        self.assertEqual(prediction._version, version)
        self.assertEqual(prediction.stride(), stride)
        trainer.validate_prediction_hash_projection_receipt_v1(
            receipt,
            expected_prediction_sha256=expected,
            expected_original_device_type="cpu",
            expected_original_dtype="torch.float32",
            expected_original_stride=stride,
            expected_original_storage_offset=0,
            expected_original_requires_grad=False,
            expected_original_is_contiguous=False,
            label="unit noncontiguous eval prediction",
        )

    def test_eval_prediction_hash_projection_rejects_resigned_layout_and_sha(self) -> None:
        prediction = torch.empty_strided(
            (1, 19_110, trainer.c1.PATCH_VALUES),
            trainer.PREDICTION_HASH_PROJECTION_PRODUCTION_STRIDE,
            dtype=torch.bfloat16,
        )
        prediction.zero_()
        self.assertTrue(prediction.is_contiguous())
        receipt = trainer.prediction_hash_projection_v1(
            prediction,
            label="projection hostile fixture",
        )

        def resign(**changes):
            unsigned = {**receipt, **changes}
            unsigned.pop("projection_receipt_digest", None)
            return {
                **unsigned,
                "projection_receipt_digest": trainer.object_sha256(unsigned),
            }

        self.assertEqual(
            receipt["original_stride"],
            list(trainer.PREDICTION_HASH_PROJECTION_PRODUCTION_STRIDE),
        )
        self.assertTrue(receipt["original_is_contiguous"])
        self.assertEqual(
            receipt["projection_stride"],
            [19_110 * trainer.c1.PATCH_VALUES, trainer.c1.PATCH_VALUES, 1],
        )
        self.assertEqual(
            receipt["projection_operation"],
            "detach_then_to_cpu_then_explicit_dense_copy_audit_only",
        )
        canonical = torch.empty(
            tuple(prediction.shape), dtype=prediction.dtype, memory_format=torch.contiguous_format
        )
        canonical.copy_(prediction)
        self.assertEqual(
            receipt["prediction_sha256"], trainer.c1.tensor_sha256_v1(canonical)
        )
        self.assertEqual(receipt["original_storage_offset"], 0)

        hostile_layout = resign(
            original_stride=[
                trainer.PREDICTION_HASH_PROJECTION_PRODUCTION_STRIDE[0] + 1,
                trainer.c1.PATCH_VALUES,
                1,
            ]
        )
        with self.assertRaisesRegex(
            trainer.ELAL3C2TrainingError, "semantic closure"
        ):
            trainer.validate_prediction_hash_projection_receipt_v1(
                hostile_layout,
                expected_prediction_sha256=receipt["prediction_sha256"],
                expected_original_device_type="cpu",
                expected_original_dtype="torch.bfloat16",
                expected_original_stride=trainer.PREDICTION_HASH_PROJECTION_PRODUCTION_STRIDE,
                expected_original_storage_offset=0,
                expected_original_requires_grad=False,
                expected_original_is_contiguous=True,
                label="resigned alternate noncontiguous original stride",
            )
        hostile_offset = resign(original_storage_offset=999)
        with self.assertRaisesRegex(
            trainer.ELAL3C2TrainingError, "semantic closure"
        ):
            trainer.validate_prediction_hash_projection_receipt_v1(
                hostile_offset,
                expected_prediction_sha256=receipt["prediction_sha256"],
                expected_original_device_type="cpu",
                expected_original_dtype="torch.bfloat16",
                expected_original_stride=trainer.PREDICTION_HASH_PROJECTION_PRODUCTION_STRIDE,
                expected_original_storage_offset=0,
                expected_original_requires_grad=False,
                expected_original_is_contiguous=True,
                label="resigned original storage offset",
            )
        hostile_requires_grad = resign(original_requires_grad=True)
        with self.assertRaisesRegex(
            trainer.ELAL3C2TrainingError, "semantic closure"
        ):
            trainer.validate_prediction_hash_projection_receipt_v1(
                hostile_requires_grad,
                expected_prediction_sha256=receipt["prediction_sha256"],
                expected_original_device_type="cpu",
                expected_original_dtype="torch.bfloat16",
                expected_original_stride=trainer.PREDICTION_HASH_PROJECTION_PRODUCTION_STRIDE,
                expected_original_storage_offset=0,
                expected_original_requires_grad=False,
                expected_original_is_contiguous=True,
                label="resigned original requires-grad",
            )
        hostile_device_index = resign(
            original_device_type="cuda",
            original_device_index=999,
        )
        with self.assertRaisesRegex(
            trainer.ELAL3C2TrainingError, "semantic closure"
        ):
            trainer.prediction_hash_projection_consensus_view_v1(
                hostile_device_index,
                expected_prediction_sha256=receipt["prediction_sha256"],
                expected_original_device_index=0,
                label="resigned CUDA device index",
            )
        hostile_projection = resign(projection_stride=[1, 1, 1])
        with self.assertRaisesRegex(
            trainer.ELAL3C2TrainingError, "semantic closure"
        ):
            trainer.validate_prediction_hash_projection_receipt_v1(
                hostile_projection,
                expected_prediction_sha256=receipt["prediction_sha256"],
                expected_original_device_type="cpu",
                expected_original_dtype="torch.bfloat16",
                expected_original_is_contiguous=True,
                label="resigned projection layout",
            )
        hostile_sha = resign(
            prediction_sha256=hashlib.sha256(b"wrong-prediction").hexdigest()
        )
        with self.assertRaisesRegex(
            trainer.ELAL3C2TrainingError, "SHA join"
        ):
            trainer.validate_prediction_hash_projection_receipt_v1(
                hostile_sha,
                expected_prediction_sha256=receipt["prediction_sha256"],
                expected_original_device_type="cpu",
                expected_original_dtype="torch.bfloat16",
                expected_original_is_contiguous=True,
                label="resigned prediction SHA",
            )
        hostile_unknown_dtype = resign(
            original_dtype="evil",
            projection_dtype="evil",
        )
        with self.assertRaisesRegex(
            trainer.ELAL3C2TrainingError, "semantic closure"
        ):
            trainer.validate_prediction_hash_projection_receipt_v1(
                hostile_unknown_dtype,
                expected_prediction_sha256=receipt["prediction_sha256"],
                expected_original_device_type="cpu",
                label="resigned unknown dtype",
            )
        hostile_production_fp32 = resign(
            original_dtype="torch.float32",
            projection_dtype="torch.float32",
        )
        with self.assertRaisesRegex(
            trainer.ELAL3C2TrainingError, "semantic closure"
        ):
            trainer.validate_prediction_hash_projection_receipt_v1(
                hostile_production_fp32,
                expected_prediction_sha256=receipt["prediction_sha256"],
                expected_original_device_type="cpu",
                expected_original_dtype=trainer.PREDICTION_HASH_PROJECTION_PRODUCTION_DTYPE,
                label="resigned production fp32 dtype",
            )

    def test_projection_semantic_failure_emits_bounded_metadata_only(self) -> None:
        prediction = torch.empty_strided(
            (1, 19_110, trainer.c1.PATCH_VALUES),
            (7, trainer.c1.PATCH_VALUES, 1),
            dtype=torch.float32,
        )
        prediction.fill_(12_345.5)
        self.assertTrue(prediction.is_contiguous())
        pointer = prediction.untyped_storage().data_ptr()
        receipt = trainer.prediction_hash_projection_v1(
            prediction, label="diagnostic singleton-stride fixture"
        )
        self.assertEqual(
            receipt["projection_stride"],
            [19_110 * trainer.c1.PATCH_VALUES, trainer.c1.PATCH_VALUES, 1],
        )
        unsigned = {**receipt, "projection_stride": [7, trainer.c1.PATCH_VALUES, 1]}
        unsigned.pop("projection_receipt_digest")
        hostile = {
            **unsigned,
            "projection_receipt_digest": trainer.object_sha256(unsigned),
        }
        with self.assertRaisesRegex(
            trainer.ELAL3C2TrainingError,
            "semantic closure differs; bounded_metadata=",
        ) as caught:
            trainer.validate_prediction_hash_projection_receipt_v1(
                hostile,
                expected_prediction_sha256=receipt["prediction_sha256"],
                expected_original_device_type="cpu",
                expected_original_device_index=None,
                expected_original_dtype="torch.float32",
                expected_original_stride=[7, trainer.c1.PATCH_VALUES, 1],
                expected_original_storage_offset=0,
                expected_original_requires_grad=False,
                expected_original_is_contiguous=True,
                label="diagnostic singleton-stride fixture",
            )
        message = str(caught.exception)
        metadata = json.loads(message.partition("bounded_metadata=")[2])
        self.assertEqual(
            set(metadata), set(trainer.PREDICTION_HASH_PROJECTION_DIAGNOSTIC_FIELDS)
        )
        self.assertEqual(len(metadata), 18)
        self.assertEqual(metadata["original_stride"], [7, trainer.c1.PATCH_VALUES, 1])
        self.assertEqual(metadata["projection_stride"], [7, trainer.c1.PATCH_VALUES, 1])
        self.assertEqual(metadata["original_shape"], [1, 19_110, trainer.c1.PATCH_VALUES])
        self.assertEqual(metadata["projection_shape"], [1, 19_110, trainer.c1.PATCH_VALUES])
        self.assertNotIn("12345.5", message)
        self.assertNotIn(str(pointer), message)
        self.assertNotIn("data_ptr", message)
        self.assertNotIn("prediction_sha256", message)
        self.assertNotIn("tensor_values", message)
        self.assertNotIn("identity", message)

    def test_duplicate_and_role_pair_are_exact2_equal_means(self) -> None:
        event, context = self._partition()
        prediction = torch.zeros((1, 1, 4), requires_grad=True)
        velocity_target = torch.ones((1, 1, 4))
        duplicate_prediction = prediction.clone()
        target_actual = self._actual_branch_receipt(
            variant="target",
            donor="target",
            route="branch-0",
            prediction=prediction,
            velocity=velocity_target,
            event=event,
            context=context,
        )
        duplicate_actual = self._actual_branch_receipt(
            variant="target",
            donor="target",
            route="branch-1",
            prediction=duplicate_prediction,
            velocity=velocity_target,
            event=event,
            context=context,
        )
        duplicate, duplicate_receipt = trainer.exact_two_branch_objective_v1(
            arm_id=trainer.ARM_DUPLICATE,
            prediction_target=prediction,
            velocity_target=velocity_target,
            event_target=event,
            context_target=context,
            prediction_second=duplicate_prediction,
            velocity_second=velocity_target,
            event_second=event,
            context_second=context,
            first_actual_input_receipt=target_actual,
            second_actual_input_receipt=duplicate_actual,
        )
        self.assertEqual(float(duplicate.item()), 1.0)
        self.assertTrue(duplicate_receipt["duplicate_control"])
        role_prediction = prediction.clone()
        role_velocity = torch.ones((1, 1, 4)) * 2
        role_event = torch.tensor([[[True, False, True, False]]])
        role_context = ~role_event
        role_actual = self._actual_branch_receipt(
            variant="role_swap",
            donor="role-swap",
            route="branch-1",
            prediction=role_prediction,
            velocity=role_velocity,
            event=role_event,
            context=role_context,
        )
        role, role_receipt = trainer.exact_two_branch_objective_v1(
            arm_id=trainer.ARM_ROLE_PAIR,
            prediction_target=prediction,
            velocity_target=velocity_target,
            event_target=event,
            context_target=context,
            prediction_second=role_prediction,
            velocity_second=role_velocity,
            event_second=role_event,
            context_second=role_context,
            first_actual_input_receipt=target_actual,
            second_actual_input_receipt=role_actual,
        )
        self.assertEqual(float(role.item()), 2.5)
        self.assertTrue(role_receipt["paired_role_supervision"])
        self.assertFalse(role_receipt["tunable_loss_weights"])
        wrong_epsilon_actual = self._resign_actual(
            duplicate_actual,
            epsilon_sha256=self._sha("different-actual-second-epsilon"),
        )
        with self.assertRaisesRegex(trainer.ELAL3C2TrainingError, "noise"):
            trainer.exact_two_branch_objective_v1(
                arm_id=trainer.ARM_DUPLICATE,
                prediction_target=prediction,
                velocity_target=velocity_target,
                event_target=event,
                context_target=context,
                prediction_second=duplicate_prediction,
                velocity_second=velocity_target,
                event_second=event,
                context_second=context,
                first_actual_input_receipt=target_actual,
                second_actual_input_receipt=wrong_epsilon_actual,
            )
        with self.assertRaisesRegex(trainer.ELAL3C2TrainingError, "duplicate control"):
            trainer.exact_two_branch_objective_v1(
                arm_id=trainer.ARM_DUPLICATE,
                prediction_target=prediction,
                velocity_target=velocity_target,
                event_target=event,
                context_target=context,
                prediction_second=prediction,
                velocity_second=velocity_target,
                event_second=event,
                context_second=context,
                first_actual_input_receipt=target_actual,
                second_actual_input_receipt=duplicate_actual,
            )

        wrong_q_actual = self._resign_actual(
            duplicate_actual,
            input_variant="role_swap",
        )
        with self.assertRaisesRegex(
            trainer.ELAL3C2TrainingError, "paired-role actual q"
        ):
            trainer.exact_two_branch_objective_v1(
                arm_id=trainer.ARM_ROLE_PAIR,
                prediction_target=prediction,
                velocity_target=velocity_target,
                event_target=event,
                context_target=context,
                prediction_second=duplicate_prediction,
                velocity_second=velocity_target,
                event_second=event,
                context_second=context,
                first_actual_input_receipt=target_actual,
                second_actual_input_receipt=wrong_q_actual,
            )

    def test_sequential_backward_is_gradient_equivalent_and_first_graph_dies(self) -> None:
        event_target, context_target = self._partition()
        event_role = torch.tensor([[[True, False, True, False]]])
        context_role = ~event_role
        velocity_target = torch.tensor([[[0.3, -0.2, 0.9, -0.7]]])
        velocity_role = torch.tensor([[[-0.8, 0.6, 0.1, 1.2]]])
        x_target = torch.tensor([[[0.2, 0.4, -0.5, 0.7]]])
        x_role = torch.tensor([[[-0.3, 0.8, 0.6, -0.1]]])

        reference_weight = torch.nn.Parameter(torch.tensor(0.25))
        reference_target = reference_weight * x_target
        reference_role = reference_weight * x_role
        reference_target_receipt = self._actual_branch_receipt(
            variant="target",
            donor="target",
            route="reference-target",
            prediction=reference_target,
            velocity=velocity_target,
            event=event_target,
            context=context_target,
        )
        reference_role_receipt = self._actual_branch_receipt(
            variant="role_swap",
            donor="role-swap",
            route="reference-role",
            prediction=reference_role,
            velocity=velocity_role,
            event=event_role,
            context=context_role,
        )
        reference_loss, _ = trainer.exact_two_branch_objective_v1(
            arm_id=trainer.ARM_ROLE_PAIR,
            prediction_target=reference_target,
            velocity_target=velocity_target,
            event_target=event_target,
            context_target=context_target,
            prediction_second=reference_role,
            velocity_second=velocity_role,
            event_second=event_role,
            context_second=context_role,
            first_actual_input_receipt=reference_target_receipt,
            second_actual_input_receipt=reference_role_receipt,
        )
        reference_loss.backward()
        reference_gradient = reference_weight.grad.detach().clone()

        sequential_weight = torch.nn.Parameter(torch.tensor(0.25))
        sequential_target = sequential_weight * x_target
        target_receipt = self._actual_branch_receipt(
            variant="target",
            donor="target",
            route="sequential-target",
            prediction=sequential_target,
            velocity=velocity_target,
            event=event_target,
            context=context_target,
        )
        first_branch = {
            "prediction": sequential_target,
            "target_velocity": velocity_target,
            "event_mask": event_target,
            "context_mask": context_target,
            "hook_receipt": {"all30_used": True, "calls_by_block": {}},
            "registered_sp4_partition": {"sp_rank": 0},
            "actual_input_receipt": target_receipt,
        }
        first_loss, first_evidence = trainer.detach_branch_loss_evidence_v1(
            first_branch, label="fake first graph"
        )
        first_ref = weakref.ref(sequential_target)
        (first_loss * 0.5).backward()
        del first_branch, first_loss, sequential_target
        gc.collect()
        self.assertIsNone(first_ref())

        sequential_role = sequential_weight * x_role
        role_receipt = self._actual_branch_receipt(
            variant="role_swap",
            donor="role-swap",
            route="sequential-role",
            prediction=sequential_role,
            velocity=velocity_role,
            event=event_role,
            context=context_role,
        )
        second_branch = {
            "prediction": sequential_role,
            "target_velocity": velocity_role,
            "event_mask": event_role,
            "context_mask": context_role,
            "hook_receipt": {"all30_used": True, "calls_by_block": {}},
            "registered_sp4_partition": {"sp_rank": 0},
            "actual_input_receipt": role_receipt,
        }
        second_loss, second_evidence = trainer.detach_branch_loss_evidence_v1(
            second_branch, label="fake second graph"
        )
        (second_loss * 0.5).backward()
        # The tiny fake graph exercises calculus only; resign its portable
        # receipts with the production packed-element closure expected by the
        # runtime objective validator.
        packed_elements = 21 * 26 * 35 * trainer.c1.PATCH_VALUES
        for evidence in (first_evidence, second_evidence):
            branch_loss = dict(evidence["branch_loss"])
            branch_loss["event_elements"] = 1
            branch_loss["context_elements"] = packed_elements - 1
            unsigned = dict(evidence)
            unsigned.pop("branch_evidence_digest")
            unsigned["branch_loss"] = branch_loss
            evidence.clear()
            evidence.update(
                {
                    **unsigned,
                    "branch_evidence_digest": trainer.object_sha256(unsigned),
                }
            )
        receipt = trainer.sequential_two_branch_objective_receipt_v1(
            arm_id=trainer.ARM_ROLE_PAIR,
            first_evidence=first_evidence,
            second_evidence=second_evidence,
            execution_mode="training_forward_backward",
        )
        self.assertTrue(
            torch.allclose(
                sequential_weight.grad,
                reference_gradient,
                rtol=1.0e-6,
                atol=1.0e-7,
            )
        )
        self.assertEqual(receipt["fixed_branch_coefficients"], [0.5, 0.5])
        self.assertEqual(receipt["simultaneous_live_autograd_branch_graphs_maximum"], 1)
        self.assertTrue(receipt["first_graph_released_before_second_forward"])
        trainer.canonical_json_bytes(first_evidence)

    def test_detached_branch_evidence_rejects_tensor_smuggling(self) -> None:
        event, context = self._partition()
        prediction = torch.zeros((1, 1, 4), requires_grad=True)
        velocity = torch.ones((1, 1, 4))
        actual = self._actual_branch_receipt(
            variant="target",
            donor="target",
            route="tensor-smuggling",
            prediction=prediction,
            velocity=velocity,
            event=event,
            context=context,
        )
        branch = {
            "prediction": prediction,
            "target_velocity": velocity,
            "event_mask": event,
            "context_mask": context,
            "hook_receipt": {"hostile_tensor": torch.ones(())},
            "registered_sp4_partition": {"sp_rank": 0},
            "actual_input_receipt": actual,
        }
        with self.assertRaisesRegex(
            trainer.ELAL3C2TrainingError, "canonical finite ASCII JSON"
        ):
            trainer.detach_branch_loss_evidence_v1(
                branch, label="hostile detached evidence"
            )

    def test_lifecycle_and_gradient_guards_reject_false_release_claims(self) -> None:
        lifecycle = {
            "execution_mode": "training_forward_backward",
            "activation_checkpoint_profile": trainer.ACTIVATION_CHECKPOINT_PROFILE,
            "activation_checkpointed_blocks": list(trainer.ACTIVATION_CHECKPOINT_BLOCKS),
            "activation_uncheckpointed_blocks": list(
                trainer.ACTIVATION_UNCHECKPOINTED_BLOCKS
            ),
            "activation_checkpoint_nonreentrant": True,
            "activation_checkpoint_elal_route_context_replay": True,
            "first_backward_completed": True,
            "second_backward_completed": True,
            "first_prediction_weakref_released_before_second_forward": True,
            "second_prediction_weakref_released_before_post_branch_work": True,
            "first_graph_deleted_before_second_forward": True,
            "second_graph_deleted_before_post_branch_work": True,
            "inter_branch_gc_collect_called": True,
            "inter_branch_cuda_empty_cache_called": True,
            "second_forward_started_after_first_release": True,
            "simultaneous_live_autograd_branch_graphs_maximum": 1,
            "first_gradient_tensors_preserved_across_graph_release": True,
            "gradient_reduce_clip_optimizer_after_both_branches": True,
            "preflight_grad_enabled_training_graph": None,
            "preflight_backward_executed": False,
            "peak_semantics": "maximum_of_sequential_true_branch_graphs_with_retained_parameter_gradients",
            "first_branch_peak_allocated_bytes": 100,
            "post_first_release_allocated_bytes": 80,
            "second_branch_peak_allocated_bytes": 120,
            "post_second_release_allocated_bytes": 90,
            "dummy_or_padding_allocations": False,
        }
        trainer.validate_branch_lifecycle_receipt_v1(
            lifecycle,
            execution_mode="training_forward_backward",
            label="positive lifecycle",
        )
        hostile = dict(lifecycle)
        hostile["first_prediction_weakref_released_before_second_forward"] = False
        with self.assertRaisesRegex(
            trainer.ELAL3C2TrainingError, "lifecycle"
        ):
            trainer.validate_branch_lifecycle_receipt_v1(
                hostile,
                execution_mode="training_forward_backward",
                label="hostile lifecycle",
            )
        preflight = dict(lifecycle)
        preflight.update(
            {
                "execution_mode": "preflight_forward_only",
                "first_backward_completed": False,
                "second_backward_completed": False,
                "first_gradient_tensors_preserved_across_graph_release": None,
                "gradient_reduce_clip_optimizer_after_both_branches": False,
                "preflight_grad_enabled_training_graph": True,
                "peak_semantics": "maximum_of_sequential_true_grad_enabled_branch_graphs_without_backward",
            }
        )
        trainer.validate_branch_lifecycle_receipt_v1(
            preflight,
            execution_mode="preflight_forward_only",
            label="positive preflight lifecycle",
        )

        parameters = [
            (f"parameter-{index:03d}", torch.nn.Parameter(torch.ones(())))
            for index in range(668)
        ]
        for _, parameter in parameters:
            parameter.grad = torch.ones_like(parameter)
        guard = trainer.gradient_accumulation_guard_v1(
            parameters, label="fake accumulated first branch"
        )
        trainer.validate_gradient_accumulation_guard_v1(
            parameters, guard, label="fake graph release"
        )
        parameters[0][1].grad.add_(1.0)
        with self.assertRaisesRegex(
            trainer.ELAL3C2TrainingError, "changed during graph release"
        ):
            trainer.validate_gradient_accumulation_guard_v1(
                parameters, guard, label="hostile fake graph release"
            )

    def test_evaluation_scalars_are_bound_to_actual_forward_hashes(self) -> None:
        event_target, context_target = self._partition()
        event_role = torch.tensor([[[True, False, True, False]]])
        context_role = ~event_role
        velocity_target = torch.ones((1, 1, 4))
        velocity_role = torch.ones((1, 1, 4)) * 2
        prediction_target = torch.zeros((1, 1, 4))
        prediction_role = torch.ones((1, 1, 4)) * 2
        mismatch_target = torch.ones((1, 1, 4)) * 0.25
        mismatch_role = torch.ones((1, 1, 4)) * 1.75
        full_q = trainer.full_q_route_matrix_v1(
            prediction_target_q=prediction_target,
            prediction_role_q=prediction_role,
            velocity_target=velocity_target,
            velocity_role=velocity_role,
            event_target=event_target,
            context_target=context_target,
            event_role=event_role,
            context_role=context_role,
        )
        union = torch.ones_like(event_target)
        role_cells = [
            trainer.role_only_cell_v1(
                row_id=trainer.ROW_IDS[0],
                clean_variant="target",
                prediction_matched=prediction_target,
                prediction_mismatch=mismatch_target,
                velocity_clean=velocity_target,
                velocity_opposite=velocity_role,
                event_clean=event_target,
                context_clean=context_target,
                contrast_union_mask=union,
                require_positive_contrast=False,
            ),
            trainer.role_only_cell_v1(
                row_id=trainer.ROW_IDS[0],
                clean_variant="role_swap",
                prediction_matched=prediction_role,
                prediction_mismatch=mismatch_role,
                velocity_clean=velocity_role,
                velocity_opposite=velocity_target,
                event_clean=event_role,
                context_clean=context_role,
                contrast_union_mask=union,
                require_positive_contrast=False,
            ),
        ]
        evidence = {
            "full_target": {
                "actual_input_receipt": self._actual_branch_receipt(
                    variant="target",
                    donor="target",
                    route="eval-full-target",
                    prediction=prediction_target,
                    velocity=velocity_target,
                    event=event_target,
                    context=context_target,
                )
            },
            "full_role_swap": {
                "actual_input_receipt": self._actual_branch_receipt(
                    variant="role_swap",
                    donor="role-swap",
                    route="eval-full-role",
                    prediction=prediction_role,
                    velocity=velocity_role,
                    event=event_role,
                    context=context_role,
                )
            },
            "mismatch_target": {
                "actual_input_receipt": self._actual_branch_receipt(
                    variant="target_role_mismatch",
                    donor="target",
                    route="eval-mismatch-target",
                    prediction=mismatch_target,
                    velocity=velocity_target,
                    event=event_target,
                    context=context_target,
                )
            },
            "mismatch_role_swap": {
                "actual_input_receipt": self._actual_branch_receipt(
                    variant="role_swap_role_mismatch",
                    donor="role-swap",
                    route="eval-mismatch-role",
                    prediction=mismatch_role,
                    velocity=velocity_role,
                    event=event_role,
                    context=context_role,
                )
            },
        }
        # Production receipts are canonical JSON (``sort_keys=True``), so all
        # nested mapping insertion orders are intentionally destroyed here.
        persisted = json.loads(
            trainer.canonical_json_bytes(
                {
                    "full_q_route": full_q,
                    "role_only_cells": role_cells,
                    "actual_forward_evidence": evidence,
                }
            )
        )
        self.assertNotEqual(
            list(persisted["full_q_route"]["energies"]),
            [
                "target_q__target",
                "target_q__role_swap",
                "role_swap_q__target",
                "role_swap_q__role_swap",
            ],
        )
        result = trainer.validate_evaluation_observation_binding_v1(
            full_q_route=persisted["full_q_route"],
            role_only_cells=persisted["role_only_cells"],
            actual_forward_evidence=persisted["actual_forward_evidence"],
            row_id=trainer.ROW_IDS[0],
            stage="step0",
        )
        self.assertTrue(
            result["full_q_and_role_only_scalars_bound_to_actual_forward_hashes"]
        )
        tampered = __import__("copy").deepcopy(full_q)
        tampered["energy_input_bindings"]["target_q__target"][
            "prediction_sha256"
        ] = self._sha("forged-prediction")
        with self.assertRaisesRegex(
            trainer.ELAL3C2TrainingError, "scalar/input binding"
        ):
            trainer.validate_evaluation_observation_binding_v1(
                full_q_route=tampered,
                role_only_cells=role_cells,
                actual_forward_evidence=evidence,
                row_id=trainer.ROW_IDS[0],
                stage="step0",
            )

        for nested_name in ("energies", "energy_input_bindings"):
            for mutation in ("missing", "extra", "renamed"):
                with self.subTest(nested_name=nested_name, mutation=mutation):
                    hostile = copy.deepcopy(persisted["full_q_route"])
                    nested = hostile[nested_name]
                    if mutation == "missing":
                        nested.pop("target_q__target")
                    elif mutation == "extra":
                        nested["attacker_extra"] = copy.deepcopy(
                            next(iter(nested.values()))
                        )
                    else:
                        nested["attacker_renamed"] = nested.pop(
                            "target_q__target"
                        )
                    with self.assertRaisesRegex(
                        trainer.ELAL3C2TrainingError,
                        "full-q scalar observation closure",
                    ):
                        trainer.validate_evaluation_observation_binding_v1(
                            full_q_route=hostile,
                            role_only_cells=persisted["role_only_cells"],
                            actual_forward_evidence=persisted[
                                "actual_forward_evidence"
                            ],
                            row_id=trainer.ROW_IDS[0],
                            stage="step0",
                        )

    def test_evaluation_exact4_evidence_canonical_roundtrip_and_member_hostiles(
        self,
    ) -> None:
        order = (
            "full_target",
            "full_role_swap",
            "mismatch_target",
            "mismatch_role_swap",
        )
        coordinate = trainer.EvaluationCoordinateV1().as_dict()
        timestep = {
            "timestep_cpu_origin": True,
            "timestep_dtype": "torch.int64",
            "timestep_value": coordinate["timestep"],
            "sigma_float32_be_hex": coordinate["sigma_float32_be_hex"],
        }
        partition = {
            "sp_rank": 0,
            "local_start": 0,
            "local_stop": trainer.c1.LOCAL_SP_TOKENS,
            "local_tokens": trainer.c1.LOCAL_SP_TOKENS,
            "source_only": True,
            "target_only": False,
        }
        hook = {
            "all30_used": True,
            "calls_by_block": {
                str(index): 1 for index in range(trainer.BLOCKS)
            },
        }
        fixed_q = (
            "q_local",
            "q_phase",
            "q_terminal",
            "q_camera",
            "entity_presence",
            "temporal_valid",
            "relation_valid",
            "phase_valid",
        )
        q_target = {name: f"target:{name}" for name in fixed_q}
        q_target.update({"q_entity": "target:q_entity", "q_relation": "target:q_relation"})
        q_role = {name: f"role:{name}" for name in fixed_q}
        q_role.update({"q_entity": "role:q_entity", "q_relation": "role:q_relation"})

        def actual(
            *, variant: str, clean: str, label: str, q_rows: Mapping[str, str]
        ) -> Mapping[str, object]:
            return {
                "row_id": trainer.ROW_IDS[0],
                "input_variant": variant,
                "label_binding_digest": label,
                "actual_q_tensor_rows": dict(q_rows),
                "actual_q_tensor_rows_digest": trainer.object_sha256(q_rows),
                "source_sha256": "source",
                "clean_target_sha256": f"{clean}:clean",
                "epsilon_sha256": "epsilon",
                "noisy_target_sha256": "sigma1-noisy",
                "target_velocity_sha256": f"{clean}:velocity",
                "event_mask_vae_sha256": f"{clean}:event",
                "context_mask_vae_sha256": f"{clean}:context",
                "text_lens_runtime_abi": {"values": [512]},
                "text_lens_sha256": "text-lens",
                "text_embs_sha256": "text-embs",
                "coordinate": coordinate,
                "coordinate_kind": "evaluation_sigma1",
                "renderer_timestep_receipt": timestep,
                "registered_sp4_partition": partition,
                "packed_target_velocity_sha256": f"{clean}:packed-velocity",
                "packed_event_mask_sha256": f"{clean}:packed-event",
                "packed_context_mask_sha256": f"{clean}:packed-context",
            }

        actuals = {
            "full_target": actual(
                variant="target", clean="target", label="target-label", q_rows=q_target
            ),
            "full_role_swap": actual(
                variant="role_swap", clean="role", label="role-label", q_rows=q_role
            ),
            "mismatch_target": actual(
                variant="target_role_mismatch",
                clean="target",
                label="target-mismatch-label",
                q_rows={**q_target, "q_entity": q_role["q_entity"], "q_relation": q_role["q_relation"]},
            ),
            "mismatch_role_swap": actual(
                variant="role_swap_role_mismatch",
                clean="role",
                label="role-mismatch-label",
                q_rows={**q_role, "q_entity": q_target["q_entity"], "q_relation": q_target["q_relation"]},
            ),
        }
        evidence = {
            name: {
                "actual_input_receipt": actuals[name],
                "hook_receipt": hook,
                "registered_sp4_partition": partition,
                "timestep_receipt": timestep,
            }
            for name in order
        }
        input_payload = {
            "source_sha256": "source",
            "epsilon_sha256": "epsilon",
            "target_sha256": "target:clean",
            "role_swap_sha256": "role:clean",
            "target_label_digest": "target-label",
            "role_swap_label_digest": "role-label",
            "target_q_digest": actuals["full_target"]["actual_q_tensor_rows_digest"],
            "role_swap_q_digest": actuals["full_role_swap"]["actual_q_tensor_rows_digest"],
            "target_mismatch_digest": "target-mismatch-label",
            "role_swap_mismatch_digest": "role-mismatch-label",
        }
        persisted = json.loads(trainer.canonical_json_bytes(evidence))
        self.assertNotEqual(tuple(persisted), order)
        with mock.patch.object(
            trainer,
            "_validate_actual_branch_receipt_closed_v1",
            side_effect=lambda value, *, label: value,
        ), mock.patch.object(
            trainer, "_validate_hook_receipt_closed_v1"
        ), mock.patch.object(
            trainer, "_validate_partition_closed_v1"
        ):
            receipt = trainer.validate_evaluation_forward_evidence_v1(
                persisted,
                row_id=trainer.ROW_IDS[0],
                sp_rank=0,
                input_payload=input_payload,
            )
            self.assertTrue(receipt["exact4_actual_renderer_forwards_closed"])
            for mutation in ("missing", "extra", "renamed"):
                with self.subTest(mutation=mutation):
                    hostile = copy.deepcopy(persisted)
                    if mutation == "missing":
                        hostile.pop("mismatch_role_swap")
                    elif mutation == "extra":
                        hostile["attacker_extra"] = copy.deepcopy(
                            hostile["full_target"]
                        )
                    else:
                        hostile["attacker_renamed"] = hostile.pop(
                            "mismatch_role_swap"
                        )
                    with self.assertRaisesRegex(
                        trainer.ELAL3C2TrainingError,
                        "actual forward member closure",
                    ):
                        trainer.validate_evaluation_forward_evidence_v1(
                            hostile,
                            row_id=trainer.ROW_IDS[0],
                            sp_rank=0,
                            input_payload=input_payload,
                        )

    def test_role_only_invariant_proves_fixed_masks_and_opposite_donor(self) -> None:
        role_code_order = (
            "agent",
            "wrong_agent",
            "patient",
            "co_agent",
            "receiver",
            "inactive",
            "instrument",
            "patient_object",
        )

        def latent(entity_motion, relation_motion, active_roles):
            q_entity = torch.zeros((3, 27), dtype=torch.float32)
            q_entity[:, 0] = torch.tensor(entity_motion, dtype=torch.float32)
            for slot, role in enumerate(active_roles):
                q_entity[slot, 19 + role_code_order.index(role)] = 1.0
            q_relation = torch.zeros((6, 11), dtype=torch.float32)
            q_relation[:, 0] = torch.tensor(relation_motion, dtype=torch.float32)
            q_relation[:, 9:11] = torch.tensor(
                [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]],
                dtype=torch.float32,
            )
            return SimpleNamespace(
                q_local=torch.tensor([1.0]),
                q_entity=q_entity,
                q_relation=q_relation,
                q_phase=torch.tensor([2.0]),
                q_terminal=torch.tensor([3.0]),
                q_camera=torch.tensor([4.0]),
                entity_presence=torch.tensor([True, True, True]),
                temporal_valid=torch.tensor([True, True, True]),
                relation_valid=torch.tensor([True] * 6),
                phase_valid=torch.tensor([True]),
            )

        masks = {
            name: torch.tensor([True, False])
            for name in (
                "event_mask_patch",
                "context_mask_patch",
                "event_mask_vae",
                "context_mask_vae",
                "role_amodal_mask_patch",
                "role_visible_mask_patch",
                "role_event_mask_patch",
                "role_event_mask_vae",
            )
        }
        source = torch.arange(8, dtype=torch.float32)
        epsilon = torch.arange(8, dtype=torch.float32) + 10
        cases = (
            (
                "blocking",
                ("agent", "patient", "instrument"),
                ("agent", "patient", "instrument"),
            ),
            (
                "handover",
                ("agent", "co_agent", "patient_object"),
                ("agent", "receiver", "patient_object"),
            ),
        )
        for _, matched_roles, opposite_roles in cases:
            matched = latent(
                [0, 1, 2], [0, 1, 2, 3, 4, 5], matched_roles
            )
            opposite = latent(
                [1, 0, 2], [2, 3, 0, 1, 5, 4], opposite_roles
            )
            mismatch = latent(
                [1, 0, 2], [2, 3, 0, 1, 5, 4], opposite_roles
            )
            receipt = trainer.role_only_swap_invariants_v1(
                matched,
                opposite,
                mismatch,
                matched_masks=masks,
                mismatch_masks={name: value.clone() for name, value in masks.items()},
                matched_source=source,
                mismatch_source=source.clone(),
                matched_instruction="the first dog yields to the second dog",
                mismatch_instruction="the first dog yields to the second dog",
                matched_sigma=1.0,
                mismatch_sigma=1.0,
                matched_x_sigma=epsilon,
                mismatch_x_sigma=epsilon.clone(),
                matched_epsilon=epsilon,
                mismatch_epsilon=epsilon.clone(),
                matched_slot_entity_ids=("dog-a", "dog-b", "barrier"),
                opposite_slot_entity_ids=("dog-b", "dog-a", "barrier"),
                mismatch_slot_entity_ids=("dog-b", "dog-a", "barrier"),
                matched_slot_roles=matched_roles,
                opposite_slot_roles=opposite_roles,
                mismatch_slot_roles=opposite_roles,
                matched_role_code_order=role_code_order,
                opposite_role_code_order=role_code_order,
                mismatch_role_code_order=role_code_order,
            )
            self.assertTrue(all(receipt["spatial_masks_fixed_bit_exact"].values()))
            self.assertTrue(
                all(receipt["swapped_fields_copied_bit_exact_from_opposite"].values())
            )
            self.assertTrue(
                receipt["semantic_role_code_order_receipt"]
                ["role_code_order_fixed_across_variants"]
            )
            self.assertTrue(
                receipt["physical_entity_to_semantic_slot_mapping_different"]
            )
        hostile_masks = {name: value.clone() for name, value in masks.items()}
        hostile_masks["event_mask_vae"] = torch.tensor([False, True])
        with self.assertRaisesRegex(trainer.ELAL3C2TrainingError, "closure differs"):
            trainer.role_only_swap_invariants_v1(
                matched,
                opposite,
                mismatch,
                matched_masks=masks,
                mismatch_masks=hostile_masks,
                matched_source=source,
                mismatch_source=source.clone(),
                matched_instruction="the first dog yields to the second dog",
                mismatch_instruction="the first dog yields to the second dog",
                matched_sigma=1.0,
                mismatch_sigma=1.0,
                matched_x_sigma=epsilon,
                mismatch_x_sigma=epsilon.clone(),
                matched_epsilon=epsilon,
                mismatch_epsilon=epsilon.clone(),
                matched_slot_entity_ids=("dog-a", "dog-b", "barrier"),
                opposite_slot_entity_ids=("dog-b", "dog-a", "barrier"),
                mismatch_slot_entity_ids=("dog-b", "dog-a", "barrier"),
                matched_slot_roles=matched_roles,
                opposite_slot_roles=opposite_roles,
                mismatch_slot_roles=opposite_roles,
                matched_role_code_order=role_code_order,
                opposite_role_code_order=role_code_order,
                mismatch_role_code_order=role_code_order,
            )
        with self.assertRaisesRegex(trainer.ELAL3C2TrainingError, "closure differs"):
            trainer.role_only_swap_invariants_v1(
                matched,
                opposite,
                mismatch,
                matched_masks=masks,
                mismatch_masks={name: value.clone() for name, value in masks.items()},
                matched_source=source,
                mismatch_source=source.clone(),
                matched_instruction="the first dog yields to the second dog",
                mismatch_instruction="the first dog yields to the second dog",
                matched_sigma=1.0,
                mismatch_sigma=1.0,
                matched_x_sigma=epsilon,
                mismatch_x_sigma=epsilon.clone(),
                matched_epsilon=epsilon,
                mismatch_epsilon=epsilon.clone(),
                matched_slot_entity_ids=("dog-a", "dog-b", "barrier"),
                opposite_slot_entity_ids=("dog-b", "dog-a", "barrier"),
                mismatch_slot_entity_ids=("dog-b", "dog-a", "barrier"),
                matched_slot_roles=matched_roles,
                opposite_slot_roles=opposite_roles,
                mismatch_slot_roles=opposite_roles,
                matched_role_code_order=role_code_order,
                opposite_role_code_order=role_code_order,
                mismatch_role_code_order=(*role_code_order[:-1], "bad-role"),
            )

    def test_real_packet_blocking_and_handover_role_only_hybrids(self) -> None:
        source = torch.arange(8, dtype=torch.float32)
        epsilon = torch.arange(8, dtype=torch.float32) + 10
        mask_names = (
            "event_mask_patch",
            "context_mask_patch",
            "event_mask_vae",
            "context_mask_vae",
            "role_amodal_mask_patch",
            "role_visible_mask_patch",
            "role_event_mask_patch",
            "role_event_mask_vae",
        )
        for row_id in trainer.ROW_IDS:
            labels = {
                variant: c2_label.load_oracle_q_label_v1(
                    PACKET_ROOT,
                    row_id=row_id,
                    media_variant=variant,
                    patch_grid=(21, 26, 35),
                    external_authority_path=EXTERNAL_AUTHORITY_PATH,
                    experiment_contract_path=CONTRACT_PATH,
                )
                for variant in trainer.TRAIN_VARIANTS
            }
            if row_id == "c2-three-entity-handover-occlusion":
                self.assertEqual(
                    labels["target"].receipt["slot_roles"],
                    ["agent", "co_agent", "patient_object"],
                )
                self.assertEqual(
                    labels["role_swap"].receipt["slot_roles"],
                    ["agent", "receiver", "patient_object"],
                )
            for clean_variant in trainer.TRAIN_VARIANTS:
                opposite_variant = (
                    "role_swap" if clean_variant == "target" else "target"
                )
                matched = labels[clean_variant]
                opposite = labels[opposite_variant]
                mismatch = c2_label.build_role_only_hybrid_v1(matched, opposite)
                receipt = trainer.role_only_swap_invariants_v1(
                    matched.latent,
                    opposite.latent,
                    mismatch.latent,
                    matched_masks={name: getattr(matched, name) for name in mask_names},
                    mismatch_masks={name: getattr(mismatch, name) for name in mask_names},
                    matched_source=source,
                    mismatch_source=source.clone(),
                    matched_instruction="registered row instruction",
                    mismatch_instruction="registered row instruction",
                    matched_sigma=1.0,
                    mismatch_sigma=1.0,
                    matched_x_sigma=epsilon,
                    mismatch_x_sigma=epsilon.clone(),
                    matched_epsilon=epsilon,
                    mismatch_epsilon=epsilon.clone(),
                    matched_slot_entity_ids=matched.receipt["slot_entity_ids"],
                    opposite_slot_entity_ids=opposite.receipt["slot_entity_ids"],
                    mismatch_slot_entity_ids=mismatch.receipt[
                        "opposite_slot_entity_ids"
                    ],
                    matched_slot_roles=matched.receipt["slot_roles"],
                    opposite_slot_roles=opposite.receipt["slot_roles"],
                    mismatch_slot_roles=mismatch.receipt["opposite_slot_roles"],
                    matched_role_code_order=matched.receipt["role_code_order"],
                    opposite_role_code_order=opposite.receipt["role_code_order"],
                    mismatch_role_code_order=opposite.receipt["role_code_order"],
                )
                self.assertTrue(
                    receipt["physical_entity_to_semantic_slot_mapping_different"]
                )

    def test_full_q_is_route_only_and_step10_role_only_improves_all_four(self) -> None:
        event, context = self._partition()
        target = torch.zeros((1, 1, 4))
        role = torch.ones((1, 1, 4)) * 2
        matrix = trainer.full_q_route_matrix_v1(
            prediction_target_q=target,
            prediction_role_q=role,
            velocity_target=target,
            velocity_role=role,
            event_target=event,
            context_target=context,
            event_role=event,
            context_role=context,
        )
        self.assertTrue(matrix["strict_diagonal_dominance"])
        self.assertTrue(matrix["participant_role_binding_claim_forbidden"])
        before_matrix = {
            "signed_diagonal_margin": -1.0,
            "strict_diagonal_dominance": False,
        }
        before_cells = []
        after_cells = []
        union = torch.ones_like(event, dtype=torch.bool)
        for row_id, clean_variant in trainer.ROLE_ONLY_CELL_ORDER:
            if clean_variant == "target":
                clean, opposite, matched, mismatch = target, role, target, torch.ones_like(target)
            else:
                clean, opposite, matched, mismatch = role, target, role, torch.ones_like(role)
            after = trainer.role_only_cell_v1(
                row_id=row_id,
                clean_variant=clean_variant,
                prediction_matched=matched,
                prediction_mismatch=mismatch,
                velocity_clean=clean,
                velocity_opposite=opposite,
                event_clean=event,
                context_clean=context,
                contrast_union_mask=union,
            )
            before_cells.append(
                {
                    "row_id": row_id,
                    "clean_variant": clean_variant,
                    "margin_mismatch_minus_matched": 0.0,
                    "normalized_predicted_vs_clean_role_contrast": -1.0,
                }
            )
            after_cells.append(after)
        gate = trainer.validate_step10_gates_v1(
            step0_full_q={row: before_matrix for row in trainer.ROW_IDS},
            step10_full_q={row: matrix for row in trainer.ROW_IDS},
            step0_role_only=before_cells,
            step10_role_only=after_cells,
        )
        self.assertTrue(gate["all_hard_gates_pass"])
        self.assertGreater(gate["primary_metric_value"], 0.0)
        self.assertEqual(
            gate["primary_metric"],
            "minimum_of_four_role_only_matched_vs_mismatch_margins",
        )

    def test_step0_role_contrast_may_be_zero_but_step10_may_not(self) -> None:
        event, context = self._partition()
        prediction = torch.zeros((1, 1, 4))
        clean = torch.zeros((1, 1, 4))
        opposite = torch.ones((1, 1, 4))
        row = trainer.role_only_cell_v1(
            row_id=trainer.ROW_IDS[0],
            clean_variant="target",
            prediction_matched=prediction,
            prediction_mismatch=prediction.clone(),
            velocity_clean=clean,
            velocity_opposite=opposite,
            event_clean=event,
            context_clean=context,
            contrast_union_mask=torch.ones_like(event),
            require_positive_contrast=False,
        )
        self.assertIsNone(row["normalized_predicted_vs_clean_role_contrast"])
        self.assertFalse(row["positive_contrast_required_at_this_stage"])
        with self.assertRaisesRegex(
            trainer.ELAL3C2TrainingError, "predicted contrast norm"
        ):
            trainer.role_only_cell_v1(
                row_id=trainer.ROW_IDS[0],
                clean_variant="target",
                prediction_matched=prediction,
                prediction_mismatch=prediction.clone(),
                velocity_clean=clean,
                velocity_opposite=opposite,
                event_clean=event,
                context_clean=context,
                contrast_union_mask=torch.ones_like(event),
                require_positive_contrast=True,
            )


if __name__ == "__main__":
    unittest.main()
