from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = METHOD_ROOT / "train_clean_source_visual_context_stage_b_v1.py"
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import clean_source_visual_context_stage_b_contract_v1 as contract

try:
    import torch
    from torch import nn

    import clean_source_visual_context_adapter_v1 as visual
    import clean_source_visual_context_training_v1 as training
    import train_clean_source_visual_context_stage_b_v1 as stage_b_runner

    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    visual = None  # type: ignore[assignment]
    training = None  # type: ignore[assignment]
    stage_b_runner = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


def _sealed(value: dict) -> dict:
    return {**value, "receipt_digest": contract.object_sha256(value)}


def _write_receipt(path: Path, value: dict) -> None:
    path.write_bytes(contract.canonical_json_bytes(value) + b"\n")


def _stage_a_family_runtime(root: Path, family: str) -> dict:
    """Materialize one minimal, real-file-backed A1-v2 56-output receipt."""

    plan = [dict(row) for row in contract._expected_stage_a_plan()]
    media_root = root / f"{family}_media"
    media_root.mkdir()
    seed = 20260814
    candidates = []
    outputs = {}
    for index, expected in enumerate(plan):
        candidate_unsigned = {
            **expected,
            "seed": seed,
            "score": None,
            "rank": None,
            "selected": False,
            "trace_gate": {"passed": True, "hook": expected["hook"]},
        }
        candidates.append(
            {
                **candidate_unsigned,
                "candidate_digest": contract.object_sha256(candidate_unsigned),
            }
        )
        media_path = media_root / f"{index:02d}_{expected['key']}.mp4"
        media_path.write_bytes(
            f"test-only decoded media\0{family}\0{expected['key']}\n".encode("ascii")
        )
        outputs[expected["key"]] = {
            "path": str(media_path),
            "sha256": contract.file_sha256(media_path),
            "frame_count": 81,
            "fps": 25.0,
        }

    unsigned = {
        "schema_version": contract.STAGE_A_RUNTIME_SCHEMA,
        "method": contract.EXPECTED_STAGE_A_METHOD,
        "stage": "preservation_stage_A_decoded_causal_localization",
        "registered_schedule_block_policy": {
            "receipt_digest": contract.EXPECTED_STAGE_A_POLICY_DIGEST,
            "schedule_indices": list(contract.STAGE_A_SCHEDULE_INDICES),
            "block_bands": {
                name: list(blocks)
                for name, blocks in contract.STAGE_A_BLOCKS_BY_BAND.items()
            },
            "optimizer_authorized": False,
            "parameter_update_authorized": False,
        },
        "intervention_contract": {
            "digest": contract.EXPECTED_STAGE_A_INTERVENTION_DIGEST,
            "optimizer": False,
            "parameter_update": False,
            "reward": False,
            "feature_scalar": False,
            "ranking": False,
            "selection": False,
        },
        "full_grid_contract": {
            "digest": contract.EXPECTED_STAGE_A_FULL_GRID_DIGEST,
        },
        "shard": {
            "family": family,
            "schedule_indices": list(contract.STAGE_A_SCHEDULE_INDICES),
            "block_bands": list(contract.STAGE_A_REQUIRED_BLOCK_BANDS),
            "full_registered_grid": False,
            "candidate_count": contract.STAGE_A_OUTPUTS_PER_REQUIRED_FAMILY,
            "plan": plan,
        },
        "authority": {},
        "runtime_source": {},
        "pinned_sources": {
            "bernini_commit": contract.EXPECTED_BERNINI_COMMIT,
            "veomni_commit": contract.EXPECTED_VEOMNI_COMMIT,
        },
        "checkpoint": {
            "tree_sha256": contract.EXPECTED_CHECKPOINT_TREE_SHA256,
            "opened_read_only": True,
            "content_identity": {
                "manifest_sha256_computed": (
                    contract.EXPECTED_CHECKPOINT_MANIFEST_SHA256
                ),
                "manifest_sha256_expected": (
                    contract.EXPECTED_CHECKPOINT_MANIFEST_SHA256
                ),
                "every_file_sha256_verified": True,
            },
        },
        "source": {"wrong_owner_same_action_family": True},
        "prompts": {},
        "sampling": {
            "seed": seed,
            "exact40": True,
            "exact81": True,
            "same_initial_gaussian_all_candidates": True,
            "source_on_native_parity_bit_exact": True,
        },
        "candidates": candidates,
        "traces": {},
        "generated_identities": {},
        "outputs": outputs,
        "frozen_model": {"unchanged": True},
        "resource_lifetime": {},
        "runtime_versions": {},
        "interpretation": {
            "decoded_complete_video_required": True,
            "score_computed": False,
            "reward_computed": False,
            "ranking_performed": False,
            "selection_performed": False,
            "training_performed": False,
            "optimizer_present": False,
            "backward_performed": False,
            "parameter_update": False,
            "stage_B_authorized_by_runtime_alone": False,
        },
    }
    return _sealed(unsigned)


class StageBStaticContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = RUNNER_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_admission_is_loaded_before_optimizer_construction(self) -> None:
        main_source = self.source[self.source.index("def main(") :]
        self.assertLess(
            main_source.index(
                "admission, method_release_identity = validate_cli_and_admission(args)"
            ),
            main_source.index("optimizer = torch.optim.AdamW("),
        )
        self.assertIn("STAGE_A_ADMISSION_SCHEMA", contract.__dict__)
        self.assertEqual(
            contract.PREREGISTERED_SPARSE_BLOCK_INDICES,
            (8, 12, 16, 20),
        )
        self.assertLess(
            main_source.index("step0_exact_base_parity_unsigned ="),
            main_source.index("optimizer = torch.optim.AdamW("),
        )
        self.assertIn(
            'fail("step-0 zero-init adapter lost bit-exact frozen-base parity")',
            main_source,
        )

    def test_runner_uses_only_scalar_safe_stage_b_parameter_attestation(self) -> None:
        self.assertNotIn("runtime.synchronize_initial_parameters(", self.source)
        self.assertNotIn("runtime.parameter_consensus(", self.source)
        self.assertEqual(
            self.source.count("stage_b_synchronize_initial_parameters("), 2
        )
        self.assertEqual(self.source.count("stage_b_parameter_consensus("), 7)
        self.assertIn("tensor.reshape(-1).view(torch.uint8)", self.source)

    def test_step0_parity_gate_does_not_equate_sp_local_evidence(self) -> None:
        main_source = self.source[self.source.index("def main(") :]
        self.assertNotIn("parity_projection =", main_source)
        self.assertNotIn(
            'label="step-0 exact-base parity DP arm"', main_source
        )
        self.assertIn(
            "parity_world8_gate = validate_step0_exact_base_parity_world8(",
            main_source,
        )
        self.assertIn('"world_rank": distributed.rank', main_source)
        self.assertIn(
            '"cross_sp_prediction_sha_equality_required": False',
            main_source,
        )

    def test_structural_preflight_is_world8_but_cannot_train_or_checkpoint(self) -> None:
        main_source = self.source[self.source.index("def main(") :]
        branch = main_source.index(
            'if args.execution_scope == "structural-parity-preflight":'
        )
        optimizer = main_source.index("optimizer = torch.optim.AdamW(")
        self.assertLess(branch, optimizer)
        self.assertIn('"optimizer_constructed": False', main_source)
        self.assertIn('"backward_executed": False', main_source)
        self.assertIn('"optimizer_step_count": 0', main_source)
        self.assertIn('"checkpoint_written": False', main_source)
        self.assertIn('"checkpoint_root_created": False', main_source)
        self.assertIn(
            'fail("formal exact80 reached optimizer boundary without Stage-A admission")',
            main_source,
        )
        self.assertIn(
            '"backward-feasibility-preflight",',
            self.source,
        )

    def test_runner_is_plain_target_only_flow_matching(self) -> None:
        for fragment in (
            "visual.no_op_flow_matching_loss(",
            '"target_rows_only": True',
            '"synthetic_target_posterior_accessed": False',
            '"frozen_feature_reward": False',
            '"vlm_reward": False',
            '"rl": False',
            '"native_self_attention": True',
            '"native_text_cross_attention": True',
            '"vae_loaded_in_training_process": False',
        ):
            self.assertIn(fragment, self.source)
        self.assertNotIn("posterior_list[1]", self.source)
        self.assertNotIn("reward_model", self.source)

    def test_continuous_exact80_accum4_cadence_and_two_arms_are_fixed(self) -> None:
        self.assertEqual(contract.CHECKPOINT_STEPS, (0, 20, 40, 60, 80))
        self.assertEqual(
            contract.MEMORY_INPUT_KINDS,
            ("clean_source", "same_noise_forward_noised_source"),
        )
        coordinates = contract.exact80_coordinates()
        self.assertEqual(contract.PHYSICAL_DP_SIZE, 2)
        self.assertEqual(contract.GRADIENT_ACCUMULATION_STEPS, 4)
        self.assertEqual(contract.MICROBATCHES_PER_DP_ARM, 320)
        self.assertEqual(contract.LOGICAL_RECORDS, 640)
        self.assertEqual(len(coordinates), 320)
        for step in range(80):
            step_coordinates = contract.coordinates_for_optimizer_step(step)
            self.assertEqual(len(step_coordinates), 4)
            self.assertEqual(
                [item.microbatch_index for item in step_coordinates], [0, 1, 2, 3]
            )
        for interval in range(4):
            interval_coordinates = [
                item for item in coordinates if item.checkpoint_interval == interval
            ]
            self.assertEqual(len(interval_coordinates), 80)
            self.assertTrue(
                all(
                    sum(item.schedule_index == index for item in interval_coordinates)
                    == 2
                    for index in range(40)
                )
            )
        for fragment in (
            "coordinates = contract.coordinates_for_optimizer_step(step_zero_based)",
            "for coordinate in coordinates:",
            "scaled_loss = raw_loss / float(",
            "scaled_loss.backward()",
            "preclip_norm = runtime.synchronize_gradients(trainable, parallel)",
            "optimizer.step()",
        ):
            self.assertIn(fragment, self.source)
        loop = self.source[self.source.index("for step_zero_based in range(") :]
        self.assertLess(loop.index("scaled_loss.backward()"), loop.index("optimizer.step()"))
        self.assertLess(
            loop.index("preclip_norm = runtime.synchronize_gradients"),
            loop.index("optimizer.step()"),
        )

    def test_effective_batch_covers_all_64_train_rows(self) -> None:
        receipt = contract.sample_coverage_receipt()
        self.assertEqual(receipt["physical_dp_size"], 2)
        self.assertEqual(receipt["gradient_accumulation_steps"], 4)
        self.assertEqual(receipt["effective_global_batch"], 8)
        self.assertEqual(receipt["sample_exposures"], 640)
        self.assertTrue(receipt["all_train_rows_seen"])
        self.assertEqual(receipt["minimum_exposures_per_row"], 10)
        self.assertEqual(receipt["maximum_exposures_per_row"], 10)
        self.assertEqual(
            receipt["logical_records_at_checkpoint_steps"],
            [0, 160, 320, 480, 640],
        )

    def test_admission_binds_dog_and_human_a1_v2_runtime_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            runtime_refs = []
            runtimes = {}
            for family in contract.STAGE_A_REQUIRED_FAMILIES:
                runtime = _stage_a_family_runtime(root, family)
                runtime_path = root / f"{family}_runtime.json"
                _write_receipt(runtime_path, runtime)
                runtimes[family] = runtime
                runtime_refs.append(
                    {
                        "family": family,
                        "path": str(runtime_path),
                        "file_sha256": contract.file_sha256(runtime_path),
                        "receipt_digest": runtime["receipt_digest"],
                    }
                )
            admission_unsigned = {
                "schema_version": contract.STAGE_A_ADMISSION_SCHEMA,
                "complete": True,
                "decision": "admit",
                "optimizer_authorized": True,
                "passed_block_bands": list(
                    contract.STAGE_A_REQUIRED_BLOCK_BANDS
                ),
                "preregistered_sparse_representatives_by_band": {
                    name: list(indices)
                    for name, indices in (
                        contract.PREREGISTERED_SPARSE_BLOCKS_BY_BAND.items()
                    )
                },
                "installed_sparse_block_indices": list(
                    contract.PREREGISTERED_SPARSE_BLOCK_INDICES
                ),
                "per_block_causal_localization_claimed": False,
                "runtime_receipts": runtime_refs,
                "manual_conjunctive_review": {
                    "decoded_media_reviewed_families": list(
                        contract.STAGE_A_REQUIRED_FAMILIES
                    ),
                    "reviewed_schedule_indices": list(
                        contract.STAGE_A_SCHEDULE_INDICES
                    ),
                    "reviewed_block_bands": list(
                        contract.STAGE_A_REQUIRED_BLOCK_BANDS
                    ),
                    "per_band_manual_pass": {
                        name: True
                        for name in contract.STAGE_A_REQUIRED_BLOCK_BANDS
                    },
                    "forward_action_prior_acceptable": True,
                    "forward_differs_from_source_off": True,
                    "reverse_and_incomplete_differ_from_forward": True,
                    "owner_specific_difference_interpretable": True,
                    "preservation_not_worse_than_adapter_off": True,
                    "scalar_threshold_used": False,
                    "feature_reward_used": False,
                    "vlm_used": False,
                },
                "training_constraints": {
                    "source_only_train_rows": 64,
                    "optimizer_steps": 80,
                    "gradient_accumulation_steps": 4,
                    "effective_global_batch": 8,
                    "logical_training_records": 640,
                    "checkpoint_steps": [0, 20, 40, 60, 80],
                    "synthetic_target_accessed": False,
                    "objective": "standard_target_only_noop_flow_matching",
                    "arms": list(contract.MEMORY_INPUT_KINDS),
                },
            }
            admission = _sealed(admission_unsigned)
            admission_path = root / "admission.json"
            _write_receipt(admission_path, admission)
            loaded = contract.load_stage_a_admission(
                admission_path,
                expected_sha256=contract.file_sha256(admission_path),
            )
            self.assertTrue(loaded.receipt()["optimizer_authorized"])
            self.assertEqual(
                loaded.runtime_families, contract.STAGE_A_REQUIRED_FAMILIES
            )
            self.assertEqual(
                loaded.runtime_receipt_digests,
                tuple(
                    runtimes[family]["receipt_digest"]
                    for family in contract.STAGE_A_REQUIRED_FAMILIES
                ),
            )

            rejected = dict(admission_unsigned)
            rejected["decision"] = "reject"
            rejected = _sealed(rejected)
            rejected_path = root / "rejected.json"
            rejected_path.write_bytes(
                contract.canonical_json_bytes(rejected) + b"\n"
            )
            with self.assertRaisesRegex(
                contract.CleanSourceVisualStageBContractError,
                "decision/constraints",
            ):
                contract.load_stage_a_admission(
                    rejected_path,
                    expected_sha256=contract.file_sha256(rejected_path),
                )

    def test_early_only_fourteen_output_runtime_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            runtime = _stage_a_family_runtime(root, "dog")
            early_unsigned = dict(runtime)
            early_unsigned.pop("receipt_digest")
            early_plan = list(runtime["shard"]["plan"][:14])
            early_keys = [row["key"] for row in early_plan]
            early_unsigned["shard"] = {
                **runtime["shard"],
                "block_bands": ["early"],
                "candidate_count": 14,
                "plan": early_plan,
            }
            early_unsigned["candidates"] = list(runtime["candidates"][:14])
            early_unsigned["outputs"] = {
                key: runtime["outputs"][key] for key in early_keys
            }
            early = _sealed(early_unsigned)
            with self.assertRaisesRegex(
                contract.CleanSourceVisualStageBContractError,
                "decoded source-edge runtime evidence differs",
            ):
                contract.validate_stage_a_runtime_receipt(early)


@unittest.skipUnless(_TORCH_AVAILABLE, "torch runtime is required")
class StageBDynamicContractTests(unittest.TestCase):
    def test_fixed_3d_position_breaks_reverse_phase_token_permutation(self) -> None:
        torch.manual_seed(19)
        encoder = visual.CleanSourceVisualEncoder(
            hidden_size=12,
            encoder_width=8,
            patch_size=(1, 2, 2),
            memory_token_cap=48,
        )
        latent = torch.randn((1, 16, 3, 8, 8), dtype=torch.float32).contiguous()
        forward = encoder.build_memory(
            latent,
            source_video_sha256="a" * 64,
            memory_input_latent_sha256="b" * 64,
        )
        reverse = encoder.build_memory(
            latent.flip(2).contiguous(),
            source_video_sha256="a" * 64,
            memory_input_latent_sha256="c" * 64,
        )
        phases, height, width = forward.pooled_grid
        bare_permutation = (
            forward.tokens.reshape(1, phases, height * width, -1)
            .flip(1)
            .reshape_as(forward.tokens)
        )
        self.assertFalse(torch.equal(reverse.tokens, bare_permutation))
        self.assertEqual(phases, 3)
        self.assertEqual(
            encoder.architecture_receipt()["position_representation"],
            "fixed_absolute_3d_fourier_phase_y_x_v1",
        )

    def test_stage_b_parameter_digest_supports_scalar_and_preserves_vector_hash(self) -> None:
        scalar = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        vector = nn.Parameter(torch.tensor([1.0, -2.0, 3.5], dtype=torch.float32))
        named = (("block.residual_gain", scalar), ("block.output.weight", vector))

        digest = stage_b_runner.stage_b_trainable_parameters_digest(named)
        self.assertRegex(digest, r"\A[0-9a-f]{64}\Z")

        expected = hashlib.sha256()
        for name, parameter in named:
            tensor = parameter.detach().contiguous()
            metadata = stage_b_runner.canonical_json_bytes(
                {
                    "name": name,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                }
            )
            expected.update(len(metadata).to_bytes(8, "big"))
            expected.update(metadata)
            expected.update(
                tensor.reshape(-1)
                .view(torch.uint8)
                .cpu()
                .numpy()
                .tobytes(order="C")
            )
        self.assertEqual(digest, expected.hexdigest())

        legacy_vector = hashlib.sha256()
        vector_tensor = vector.detach().contiguous()
        vector_metadata = stage_b_runner.canonical_json_bytes(
            {
                "name": "block.output.weight",
                "shape": list(vector_tensor.shape),
                "dtype": str(vector_tensor.dtype),
            }
        )
        legacy_vector.update(len(vector_metadata).to_bytes(8, "big"))
        legacy_vector.update(vector_metadata)
        legacy_vector.update(
            vector_tensor.view(torch.uint8).cpu().numpy().tobytes(order="C")
        )
        self.assertEqual(
            stage_b_runner.stage_b_trainable_parameters_digest(
                (("block.output.weight", vector),)
            ),
            legacy_vector.hexdigest(),
        )

    def test_step0_world8_gate_accepts_sp_local_outputs_and_rejects_local_mismatch(self) -> None:
        def route(sp_rank: int, *, enabled: bool, memory_digest: str | None) -> dict:
            unsigned = {
                "total_tokens": 10,
                "condition_tokens": 6,
                "target_tokens": 4,
                "sequence_parallel_rank": sp_rank,
                "sequence_parallel_size": 4,
                "enabled": enabled,
                "memory_digest": memory_digest,
                "query_rows": "local_target_suffix_only",
                "key_value_rows": (
                    "independent_registered_source_visual_memory_only"
                ),
            }
            return {**unsigned, "digest": stage_b_runner.object_sha256(unsigned)}

        selector_counts = (0, 0, 3, 1)
        records = []
        for dp_arm in range(2):
            for sp_rank in range(4):
                local_sha = hashlib.sha256(
                    f"legal-sp-local-output-{dp_arm}-{sp_rank}".encode("ascii")
                ).hexdigest()
                records.append(
                    {
                        "world_rank": dp_arm * 4 + sp_rank,
                        "dp_arm": dp_arm,
                        "sp_rank": sp_rank,
                        "iid": f"source-{dp_arm}",
                        "row_position": dp_arm,
                        "manifest_index": 10 + dp_arm,
                        "noise_seed": 20260814 + dp_arm,
                        "optimizer_step": 1,
                        "checkpoint_interval": 0,
                        "step_in_checkpoint_interval": 0,
                        "microbatch_index": 0,
                        "interval_micro_ordinal": 0,
                        "interval_schedule_cycle": 0,
                        "schedule_index": 0,
                        "timestep_int64": 999,
                        "sigma": 1.0,
                        "sigma_float32_be_hex": "3f800000",
                        "memory_input_kind": "clean_source",
                        "input_patch_shape": [20, 64],
                        "prediction_shape": [1, 4, 64],
                        "disabled_route_receipt": route(
                            sp_rank, enabled=False, memory_digest=None
                        ),
                        "enabled_route_receipt": route(
                            sp_rank,
                            enabled=True,
                            memory_digest=("a" if dp_arm == 0 else "b") * 64,
                        ),
                        "local_target_selector_count": selector_counts[sp_rank],
                        "disabled_route_prediction_sha256": local_sha,
                        "enabled_zero_init_route_prediction_sha256": local_sha,
                        "bit_exact_equal": True,
                        "zero_output_projection_names": ["blocks.8.output.weight"],
                        "zero_output_projections_exact": True,
                        "optimizer_constructed": False,
                        "checkpoint_written": False,
                    }
                )

        gate = stage_b_runner.validate_step0_exact_base_parity_world8(
            tuple(reversed(records))
        )
        self.assertTrue(gate["bit_exact_base_vs_zero_init_on_every_rank"])
        self.assertFalse(gate["cross_sp_prediction_sha_equality_required"])
        self.assertEqual(
            [arm["sp_ranks"] for arm in gate["dp_arm_records"]],
            [[0, 1, 2, 3], [0, 1, 2, 3]],
        )
        self.assertEqual(
            len(
                {
                    row["disabled_route_prediction_sha256"]
                    for arm in gate["dp_arm_records"]
                    for row in arm["sp_local_records"]
                }
            ),
            8,
        )

        mismatched = [dict(record) for record in records]
        mismatched[2]["enabled_zero_init_route_prediction_sha256"] = "f" * 64
        with self.assertRaisesRegex(
            stage_b_runner.CleanSourceVisualStageBTrainingError,
            "per-rank base/zero-init parity differs",
        ):
            stage_b_runner.validate_step0_exact_base_parity_world8(mismatched)

        shared_drift = [dict(record) for record in records]
        shared_drift[1]["iid"] = "different-source-within-one-dp-arm"
        with self.assertRaisesRegex(
            stage_b_runner.CleanSourceVisualStageBTrainingError,
            "shared semantic fields differ",
        ):
            stage_b_runner.validate_step0_exact_base_parity_world8(shared_drift)

        route_drift = [dict(record) for record in records]
        changed_route = dict(route_drift[1]["enabled_route_receipt"])
        changed_route["memory_digest"] = "c" * 64
        changed_route_unsigned = dict(changed_route)
        changed_route_unsigned.pop("digest")
        changed_route["digest"] = stage_b_runner.object_sha256(
            changed_route_unsigned
        )
        route_drift[1]["enabled_route_receipt"] = changed_route
        with self.assertRaisesRegex(
            stage_b_runner.CleanSourceVisualStageBTrainingError,
            "shared route semantic fields differ",
        ):
            stage_b_runner.validate_step0_exact_base_parity_world8(route_drift)

    def test_checkpoint_loader_accepts_all_registered_steps_and_rejects_sha(self) -> None:
        class Handle:
            def __init__(self) -> None:
                self.components = nn.Sequential(nn.Linear(3, 2, bias=False))

            def trainable_named_parameters(self):
                return tuple(self.components.named_parameters())

        for step in contract.CHECKPOINT_STEPS:
            with self.subTest(step=step), tempfile.TemporaryDirectory() as directory:
                handle = Handle()
                state = {
                    name: parameter.detach().float().cpu().contiguous().clone()
                    for name, parameter in handle.trainable_named_parameters()
                }
                metadata = {
                    "schema_version": contract.CHECKPOINT_SCHEMA,
                    "global_step": step,
                    "logical_records_seen": step * 8,
                    "gradient_accumulation_steps": 4,
                    "effective_global_batch": 8,
                    "checkpoint_cadence": list(contract.CHECKPOINT_STEPS),
                    "manifest_digest": "d" * 64,
                    "split_counts": {"train": 64, "confirmation": 16, "heldout": 8},
                    "authorization": {
                        "stage_a_admission_digest": "e" * 64,
                        "memory_input_kind": "clean_source",
                        "optimizer_authorized": True,
                    },
                    "adapter_receipt": {},
                    "adapter_parameter_digest": training._state_digest(state),
                    "base_frozen": True,
                    "native_kv_untouched": True,
                    "source_posterior_only": True,
                    "synthetic_target_posterior_accessed": False,
                    "objective": "same_real_source_noop_flow_matching",
                    "feature_or_vlm_reward": False,
                }
                path = Path(directory) / f"checkpoint_step_{step:08d}.pt"
                torch.save(
                    {
                        "metadata": metadata,
                        "adapter_state_dict": state,
                        "optimizer_state_dict": {},
                    },
                    path,
                )
                loaded = contract.load_visual_context_checkpoint(
                    path,
                    expected_file_sha256=contract.file_sha256(path),
                    expected_step=step,
                    expected_manifest_digest="d" * 64,
                    expected_admission_digest="e" * 64,
                    expected_memory_input_kind="clean_source",
                    handle=handle,
                )
                self.assertEqual(loaded["global_step"], step)
                with self.assertRaisesRegex(
                    contract.CleanSourceVisualStageBContractError, "SHA-256 differs"
                ):
                    contract.load_visual_context_checkpoint(
                        path,
                        expected_file_sha256="f" * 64,
                        expected_step=step,
                        expected_manifest_digest="d" * 64,
                        expected_admission_digest="e" * 64,
                        expected_memory_input_kind="clean_source",
                        handle=handle,
                    )


if __name__ == "__main__":
    unittest.main()
