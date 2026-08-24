from __future__ import annotations

import argparse
from dataclasses import replace
import inspect
import math
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import tri_branch_unipc as tri  # noqa: E402
from spt_v2 import generator_native_sparse_router as sparse_router  # noqa: E402
from spt_v2 import infer_c2fr as inference  # noqa: E402


SHA1 = "1" * 40
SHA256 = "2" * 64


def _args(**overrides) -> argparse.Namespace:
    values = {
        "instruction": "Make the actor crouch.",
        "num_inference_steps": 40,
        "seed": 42,
        "alpha": 1.0,
        "max_generate_fraction": 0.12,
        "energy_coverage": 0.85,
        "expected_bernini_commit": inference.trainer.BERNINI_OFFICIAL_COMMIT,
        "expected_veomni_commit": inference.trainer.VEOMNI_TESTED_COMMIT,
        "expected_checkpoint_tree_sha256": inference.trainer.CHECKPOINT_TREE_SHA256,
        "method_source_revision": SHA1,
        "method_source_archive_sha256": SHA256,
        "checkpoint": "/checkpoint/Bernini-R-1.3B-Diffusers",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _valid_traces() -> tuple[tri.TriBranchTrace, inference.RouterExecutionTrace]:
    tri_records = []
    router_records = []
    cells = 100
    counts = (1,) * 21
    for index in range(40):
        sigma = 1.0 - index / 41.0
        timestep = 1000.0 - 20.0 * index
        tri_records.append(
            tri.TriBranchStepRecord(
                step_index=index,
                timestep=timestep,
                sigma=sigma,
                model_id="transformer_1",
                transformer_forwards=3,
                shared_negative_forwards=1,
                action_forwards=1,
                noop_forwards=1,
                original_scheduler_calls=1,
                callback_correction_rms=0.2,
                raw_action_noop_delta_rms=0.1,
                guided_action_noop_delta_rms=0.15,
                guided_action_noop_delta_l2=3.0,
                action_noop_exact_parity=False,
                effective_guidance_scale=4.0,
                official_action_parity_rms_error=0.0,
                official_action_parity_max_abs_error=0.0,
                official_action_exact_parity=True,
                sample_dtype="torch.float32",
                branch_velocity_dtype="torch.bfloat16",
                official_model_output_dtype="torch.float32",
            )
        )
        router_records.append(
            inference.RouterStepRecord(
                step_index=index,
                timestep=timestep,
                sigma=sigma,
                selected_cell_count=sum(counts),
                total_cell_count=21 * cells,
                generate_fraction=sum(counts) / (21 * cells),
                max_phase_generate_fraction=1 / cells,
                active_phase_count=21,
                cells_per_phase=cells,
                integer_capacity_per_phase=12,
                per_phase_selected_counts=counts,
                saliency_mean=0.1,
                saliency_max=1.0,
                phase_activity_mean=0.1,
                phase_activity_max=0.2,
                support_sha256="3" * 64,
            )
        )
    return (
        tri.TriBranchTrace(records=tri_records, sample_calls=1),
        inference.RouterExecutionTrace(
            alpha=1.0,
            config=sparse_router.GeneratorNativeSparseRouterConfig(),
            records=router_records,
        ),
    )


class PureC2FRInferenceContractTests(unittest.TestCase):
    def test_cli_is_frozen_base_source_instruction_only(self) -> None:
        parser = inference.build_parser()
        destinations = {action.dest for action in parser._actions}
        self.assertTrue(
            {
                "source_video",
                "instruction",
                "alpha",
                "max_generate_fraction",
                "energy_coverage",
            }
            <= destinations
        )
        self.assertTrue(
            destinations.isdisjoint(
                {
                    "planner_checkpoint",
                    "adapter_checkpoint",
                    "lora",
                    "noop_instruction",
                    "target_video",
                    "oracle_plan",
                    "mask",
                    "track",
                    "pose",
                    "flow",
                    "trajectory",
                    "first_frame_anchor",
                }
            )
        )
        args = parser.parse_args(
            [
                "--bernini-root",
                "/b",
                "--veomni-root",
                "/v",
                "--checkpoint",
                "/c",
                "--source-video",
                "/source.mp4",
                "--instruction",
                "move",
                "--output",
                "/out.mp4",
                "--method-source-revision",
                SHA1,
                "--method-source-archive-sha256",
                SHA256,
            ]
        )
        self.assertEqual(args.num_inference_steps, 40)
        self.assertEqual(args.alpha, 1.0)
        self.assertEqual(args.max_generate_fraction, 0.12)
        self.assertEqual(args.energy_coverage, 0.85)

    def test_cli_rejects_solver_or_router_drift(self) -> None:
        inference.validate_cli(_args())
        invalid = (
            {"num_inference_steps": 41},
            {"seed": -1},
            {"alpha": -0.1},
            {"alpha": math.nan},
            {"max_generate_fraction": 0.120001},
            {"max_generate_fraction": 0.0},
            {"energy_coverage": 0.0},
            {"energy_coverage": 1.1},
            {"instruction": "\x00"},
        )
        for changed in invalid:
            with self.subTest(changed=changed), self.assertRaises(
                inference.C2FRInferenceError
            ):
                inference.validate_cli(_args(**changed))

    def test_sampler_is_exact_81f_forty_step_official_apg(self) -> None:
        contract = inference.exact_sampler_contract(seed=7)
        self.assertEqual(contract["num_frames"], 81)
        self.assertEqual(contract["num_inference_steps"], 40)
        self.assertEqual(contract["guidance_mode"], "v2v_apg")
        self.assertEqual(contract["flow_shift"], 5.0)
        self.assertEqual(contract["omega_txt"], 4.0)
        self.assertEqual(contract["eta"], 0.5)
        self.assertEqual(inference.base.ULYSSES_SIZE, 4)

    def test_noop_is_fixed_semantic_control_and_not_external_cli(self) -> None:
        self.assertIn("exactly unchanged", inference.motion.DEFAULT_NOOP_INSTRUCTION)
        self.assertIn("action", inference.motion.DEFAULT_NOOP_INSTRUCTION)
        self.assertNotIn(
            "noop_instruction",
            {action.dest for action in inference.build_parser()._actions},
        )

    def test_runtime_helpers_have_no_privileged_arguments(self) -> None:
        forbidden = {
            "target",
            "target_video",
            "paired_target",
            "mask",
            "track",
            "pose",
            "flow",
            "trajectory",
            "anchor",
            "planner",
            "adapter",
        }
        for function in (
            inference.encode_semantic_noop_prompt,
            inference.TracedGeneratorNativeSparseCallback.__init__,
            inference.TracedGeneratorNativeSparseCallback.__call__,
            inference.validate_execution_trace,
        ):
            with self.subTest(function=function.__qualname__):
                self.assertTrue(
                    forbidden.isdisjoint(inspect.signature(function).parameters)
                )

    def test_trace_requires_exact_apg_and_one_original_unipc_per_step(self) -> None:
        tri_trace, router_trace = _valid_traces()
        payload = inference.validate_execution_trace(tri_trace, router_trace)
        self.assertEqual(payload["certificate"]["step_count"], 40)
        self.assertEqual(payload["certificate"]["official_action_apg_exact_steps"], 40)
        self.assertEqual(payload["certificate"]["original_unipc_calls"], 40)
        self.assertEqual(payload["certificate"]["transformer_forwards"], 120)
        self.assertFalse(payload["certificate"]["custom_integrator"])
        self.assertRegex(payload["trace_digest"], r"^[0-9a-f]{64}$")

        tri_trace.records[7] = replace(
            tri_trace.records[7], official_action_exact_parity=False
        )
        with self.assertRaisesRegex(inference.C2FRInferenceError, "exact certificate"):
            inference.validate_execution_trace(tri_trace, router_trace)

    def test_trace_rejects_scheduler_duplication_and_support_over_cap(self) -> None:
        tri_trace, router_trace = _valid_traces()
        tri_trace.records[3] = replace(
            tri_trace.records[3], original_scheduler_calls=2
        )
        with self.assertRaisesRegex(inference.C2FRInferenceError, "one original UniPC"):
            inference.validate_execution_trace(tri_trace, router_trace)

        tri_trace, router_trace = _valid_traces()
        bad_counts = (13,) + (1,) * 20
        router_trace.records[3] = replace(
            router_trace.records[3],
            selected_cell_count=sum(bad_counts),
            generate_fraction=sum(bad_counts) / 2100,
            max_phase_generate_fraction=0.13,
            per_phase_selected_counts=bad_counts,
        )
        with self.assertRaisesRegex(inference.C2FRInferenceError, "cardinality"):
            inference.validate_execution_trace(tri_trace, router_trace)

    def test_receipt_is_frozen_base_and_has_no_train_test_gap(self) -> None:
        tri_trace, router_trace = _valid_traces()
        execution = inference.validate_execution_trace(tri_trace, router_trace)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt = inference.build_inference_receipt(
                args=_args(),
                source_path=root / "source.mp4",
                source_sha256="4" * 64,
                source_metadata={"source_derived_bucket_hw": [480, 496]},
                output_path=root / "out.mp4",
                output_sha256="5" * 64,
                noop_identity={
                    "token_shape": [1, 512],
                    "embedding_shape": [1, 512, 4096],
                    "frozen_t5": True,
                },
                execution_trace=execution,
                bernini_revision=inference.trainer.BERNINI_OFFICIAL_COMMIT,
                veomni_revision=inference.trainer.VEOMNI_TESTED_COMMIT,
                inference_file_hashes={},
                wan_diffusion_path=root / "bernini/models/wan_diffusion.py",
                wan_diffusion_sha256=tri.PINNED_WAN_DIFFUSION_SHA256,
                runtime_versions={},
            )
        self.assertTrue(receipt["base_model"]["frozen"])
        self.assertFalse(receipt["base_model"]["planner_checkpoint_loaded"])
        self.assertFalse(receipt["base_model"]["lora_or_peft_loaded"])
        self.assertEqual(
            receipt["input"]["accepted_external_conditions"],
            ["source_video", "edit_instruction"],
        )
        self.assertFalse(receipt["input"]["target_accessed_by_inference"])
        self.assertFalse(receipt["input"]["external_mask_track_pose_flow_trajectory"])
        self.assertTrue(
            receipt["prompt_contract"]["action_noop_negative_use_frozen_t5"]
        )
        self.assertEqual(receipt["sampling"]["tri_branch_contract"], tri.sampler_contract())
        candidate = dict(receipt)
        declared = candidate.pop("receipt_digest")
        self.assertEqual(inference.base.object_sha256(candidate), declared)

    def test_main_installs_audited_tri_branch_router_at_official_boundary(self) -> None:
        source = inspect.getsource(inference.main)
        self.assertIn("tri.tri_branch_unipc_hook", source)
        self.assertIn("wan_diffusion_path=wan_diffusion_path", source)
        self.assertIn("TracedGeneratorNativeSparseCallback", source)
        self.assertIn("model.sample", source)
        self.assertNotIn("planner_checkpoint", source)
        self.assertNotIn("adapter_checkpoint", source)

    def test_rank_local_cache_paths_are_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = inference.configure_rank_local_caches(
                {
                    "BERNINI_C2FR_RANK_CACHE_ROOT": str(Path(directory).resolve()),
                    "LOCAL_RANK": "2",
                }
            )
        self.assertEqual(set(paths), {
            "MIOPEN_USER_DB_PATH",
            "MIOPEN_CUSTOM_CACHE_DIR",
            "TORCH_EXTENSIONS_DIR",
            "TRITON_CACHE_DIR",
        })
        self.assertTrue(all("rank-2" in value for value in paths.values()))

    def test_auh_launcher_is_four_gpu_frozen_base_and_hash_bound(self) -> None:
        launcher = (
            METHOD_ROOT / "spt_v2/scripts/auh_infer_c2fr.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("--nproc_per_node=4", launcher)
        self.assertIn("--gres=gpu:mi210:4", launcher)
        self.assertIn("--alpha", launcher)
        self.assertIn("--max-generate-fraction", launcher)
        self.assertIn("--energy-coverage", launcher)
        self.assertIn("BERNINI_C2FR_RANK_CACHE_ROOT", launcher)
        self.assertIn("wan_diffusion.py", launcher)
        self.assertNotIn("planner_checkpoint", launcher)
        self.assertNotIn("adapter_checkpoint", launcher)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is unavailable")
class TensorC2FRInferenceContractTests(unittest.TestCase):
    def test_traced_callback_records_support_and_returns_exact_source_outside(self) -> None:
        layout = tri.PackedLatentLayout.from_spatial_shape((1, 16, 21, 10, 10))
        phase_shape = (1, 21, 5, 5, 64)
        source_phase = torch.randn(phase_shape)
        noop_phase = torch.randn(phase_shape)
        delta_phase = torch.zeros(phase_shape)
        for phase, column in ((8, 1), (9, 2), (10, 3)):
            delta_phase[:, phase, 2, column, :] = 3.0
        action_phase = noop_phase + delta_phase
        action = sparse_router.phase_video_to_spatial(action_phase, layout=layout)
        noop = sparse_router.phase_video_to_spatial(noop_phase, layout=layout)
        delta = sparse_router.phase_video_to_spatial(
            action_phase - noop_phase, layout=layout
        )
        zeros = torch.zeros_like(action)
        fields = tri.CleanFieldStep(
            step_index=0,
            timestep=900.0,
            sigma=0.9,
            model_id="transformer_1",
            noisy=zeros,
            negative_velocity=zeros,
            action_velocity=zeros,
            noop_velocity=zeros,
            negative_clean=zeros,
            action_condition_clean=action,
            noop_condition_clean=noop,
            action_guided_clean=action,
            noop_guided_clean=noop,
            action_delta_clean=delta,
        )
        callback = inference.TracedGeneratorNativeSparseCallback(
            source_clean=source_phase,
            layout=layout,
            config=sparse_router.GeneratorNativeSparseRouterConfig(
                activity_energy_floor=0.0,
                relative_phase_activity_floor=0.0,
                energy_coverage=0.8,
            ),
            alpha=0.5,
        )
        result = callback(fields)
        self.assertEqual(len(callback.trace.records), 1)
        record = callback.trace.records[0]
        self.assertLessEqual(record.max_phase_generate_fraction, 0.12)
        self.assertRegex(record.support_sha256, r"^[0-9a-f]{64}$")
        result_phase = sparse_router.spatial_to_phase_video(result, layout=layout)
        support = callback.inner.last_execution.plan.gate_probs[:, 2].bool().unsqueeze(-1)
        outside = (~support).expand_as(source_phase)
        self.assertTrue(torch.equal(result_phase[outside], source_phase[outside]))


if __name__ == "__main__":
    unittest.main()
