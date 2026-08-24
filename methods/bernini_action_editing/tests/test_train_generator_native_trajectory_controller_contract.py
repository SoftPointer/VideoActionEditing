from __future__ import annotations

import argparse
import ast
import inspect
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import generator_native_trajectory_controller as egntc  # noqa: E402
import train_generator_native_trajectory_controller as trainer  # noqa: E402

try:
    import torch
except ImportError:  # pragma: no cover - dependency-light orchestration hosts
    torch = None


def _valid_args(**updates):
    values = dict(
        bernini_root="/bernini",
        veomni_root="/veomni",
        checkpoint="/checkpoint",
        k2_config="/episode.json",
        expected_k2_config_sha256="1" * 64,
        preview_manifest="/preview.jsonl",
        vae_index="/vae.jsonl",
        output="/absolute/output",
        num_frames=81,
        k_shot=2,
        num_inference_steps=40,
        optimizer_steps_per_round=25,
        learning_rate=0.02,
        max_grad_norm=1.0,
        seed=20260809,
        rollout_seed=2029,
        teacher_sigma_index=20,
        engineering_smoke=False,
        ack_preview_experimental_only=True,
        expected_bernini_commit=trainer.episode_train.legacy.BERNINI_OFFICIAL_COMMIT,
        expected_veomni_commit=trainer.episode_train.legacy.VEOMNI_TESTED_COMMIT,
        expected_checkpoint_tree_sha256=(
            trainer.episode_train.legacy.CHECKPOINT_TREE_SHA256
        ),
        method_source_revision="2" * 40,
        method_source_archive_sha256="3" * 64,
    )
    values.update(updates)
    return argparse.Namespace(**values)


def _record(index: int, **updates):
    values = dict(
        step_index=index,
        timestep=float(egntc.sigma_strata.PINNED_TIMESTEPS[index]),
        sigma=float(egntc.sigma_strata.PINNED_POSITIVE_SIGMAS[index]),
        model_id="transformer_1",
        transformer_forwards=3,
        shared_negative_forwards=1,
        action_forwards=1,
        noop_forwards=1,
        original_scheduler_calls=1,
        official_action_exact_parity=True,
    )
    values.update(updates)
    return types.SimpleNamespace(**values)


class StaticWorld8RunnerContractTests(unittest.TestCase):
    def test_exact_geometry_parameter_and_world_contract(self):
        self.assertEqual(trainer.NUM_FRAMES, 81)
        self.assertEqual(trainer.LATENT_PHASES, 21)
        self.assertEqual(trainer.NUM_INFERENCE_STEPS, 40)
        self.assertEqual(trainer.FORWARDS_PER_STEP, 3)
        self.assertEqual(trainer.FORWARDS_PER_SUPPORT_ROUND, 120)
        self.assertEqual(trainer.TRAINABLE_DIMENSION, 36)
        self.assertEqual(trainer.WORLD_SIZE, 8)
        self.assertEqual(trainer.ULYSSES_SIZE, 4)
        self.assertEqual(trainer.DATA_PARALLEL_SIZE, 2)
        self.assertEqual(trainer.CACHE_SPATIAL_HW, (60, 62))
        self.assertEqual(trainer.SUPERVISED_STEP_INDICES, (0, 8, 16, 24, 32, 39))

    def test_smoke_and_formal_round_plans_keep_full_capture(self):
        smoke = trainer.round_plan(engineering_smoke=True, optimizer_steps=25)
        self.assertEqual(len(smoke), 1)
        self.assertEqual(smoke[0]["execution_policy"], "official_action")
        self.assertEqual(smoke[0]["optimizer_steps"], 1)
        self.assertEqual(smoke[0]["full_unipc_steps"], 40)
        self.assertEqual(smoke[0]["transformer_forwards"], 120)
        formal = trainer.round_plan(engineering_smoke=False, optimizer_steps=25)
        self.assertEqual(len(formal), 2)
        self.assertEqual(
            [item["execution_policy"] for item in formal],
            ["official_action", "learned_controller"],
        )
        self.assertTrue(all(item["full_unipc_steps"] == 40 for item in formal))
        self.assertTrue(all(item["transformer_forwards"] == 120 for item in formal))

    def test_cli_locks_81f_k2_40step_and_preview_ack(self):
        trainer.validate_cli(_valid_args())
        for update in (
            {"num_frames": 41},
            {"k_shot": 1},
            {"num_inference_steps": 39},
            {"ack_preview_experimental_only": False},
            {"optimizer_steps_per_round": 0},
            {"output": "relative/output"},
        ):
            with self.subTest(update=update):
                with self.assertRaises(trainer.EGNTCTrainingError):
                    trainer.validate_cli(_valid_args(**update))

    def test_capture_api_cannot_receive_a_target_or_oracle(self):
        parameters = inspect.signature(trainer._capture_rollout).parameters
        self.assertEqual(
            set(parameters),
            {
                "renderer",
                "tokenizer",
                "support",
                "parameters",
                "execution_policy",
                "round_index",
                "rollout_seed",
                "device",
                "prompt_cleaner",
                "bernini_revision",
                "wan_diffusion_path",
            },
        )
        self.assertFalse(any("target" in name for name in parameters))
        callback_parameters = inspect.signature(
            trainer._RolloutCaptureCallback.__init__
        ).parameters
        self.assertFalse(any("target" in name for name in callback_parameters))
        self.assertEqual(
            list(trainer.INFERENCE_CONDITIONS), ["source_video", "action_instruction"]
        )
        forbidden = set(trainer.FORBIDDEN_INFERENCE_CONDITIONS)
        self.assertTrue(
            {
                "target_video",
                "mask",
                "flow",
                "pose",
                "track",
                "trajectory",
                "edited_first_frame",
            }.issubset(forbidden)
        )

    def test_target_object_is_named_training_only_and_offline(self):
        source = inspect.getsource(trainer)
        self.assertIn("class TrainingOnlyMotionTeacher", source)
        self.assertIn("The only function that accepts the paired target object", source)
        capture_source = inspect.getsource(trainer._capture_rollout)
        self.assertNotIn("target_clean", capture_source)
        self.assertNotIn("TrainingOnlyMotionTeacher", capture_source)
        self.assertIn("target_clean.detach()", inspect.getsource(trainer._source_relative_objective))
        self.assertIn('"full_target_reconstruction_weight": 0.0', source)

    def test_runner_reuses_episode_world8_teacher_and_tri_branch_implementations(self):
        source = inspect.getsource(trainer)
        required = (
            "episode_train.load_audited_episode(args)",
            "episode_train.epmc_distributed_contract()",
            "episode_train.initialise_epmc_distributed(contract)",
            "episode_train.validate_epmc_parallel_state",
            "episode_train._prepare_teacher_cell",
            "episode_train._read_episode_parquet",
            "tri.tri_branch_unipc_hook",
            "init_parallel_state(ulysses_size=ULYSSES_SIZE)",
            "support = episode.supports[parallel.support_index]",
        )
        # The last semantic is expressed by assigned_row in the executable.
        for expected in required[:-1]:
            self.assertIn(expected, source)
        self.assertIn("assigned_row = episode.supports[parallel.support_index]", source)
        self.assertIn("episode_train._all_reduce_code_gradients", source)
        self.assertIn("episode_train._exchange_k2_objects", source)

    def test_every_collective_has_an_explicit_group(self):
        tree = ast.parse(inspect.getsource(trainer))
        collectives = {"all_reduce", "all_gather", "all_gather_object", "barrier"}
        calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in collectives:
                continue
            owner = node.func.value
            if not isinstance(owner, ast.Name) or owner.id not in {"dist", "dist_module"}:
                continue
            calls.append((node.func.attr, node.lineno, {item.arg for item in node.keywords}))
        self.assertTrue(calls)
        self.assertEqual(
            [],
            [(name, line) for name, line, keywords in calls if "group" not in keywords],
        )

    def test_no_forbidden_inference_cli_or_custom_integrator(self):
        source = inspect.getsource(trainer.build_parser)
        for forbidden in (
            "--target-video",
            "--mask",
            "--flow",
            "--pose",
            "--track",
            "--trajectory",
            "--first-frame",
        ):
            self.assertNotIn(forbidden, source)
        module_source = inspect.getsource(trainer)
        self.assertNotIn("EulerDiscreteScheduler", module_source)
        self.assertNotIn("scheduler.step(", module_source)
        self.assertIn("one original UniPC", trainer.__doc__)

    def test_source_uses_full_resolution_fp32_cache_before_nonlinear_rmsclip(self):
        source = inspect.getsource(trainer._cache_clean_cpu)
        self.assertIn("(60, 62)", repr(trainer.CACHE_SPATIAL_HW))
        self.assertIn("value.detach().float().contiguous().cpu()", source)
        self.assertNotIn("adaptive_avg_pool", source)
        self.assertIn("RMSClip is nonlinear", source)
        hashing_source = inspect.getsource(trainer._tensor_content_bytes)
        self.assertIn('memoryview(tensor.numpy()).cast("B")', hashing_source)
        self.assertIn("except RuntimeError", hashing_source)


class TraceAndLineageTests(unittest.TestCase):
    def test_trace_requires_exact_40_steps_120_forwards_and_40_unipc_calls(self):
        trace = types.SimpleNamespace(
            sample_calls=1, records=[_record(index) for index in range(40)]
        )
        receipt = trainer.validate_full_rollout_trace(trace)
        self.assertEqual(receipt["step_count"], 40)
        self.assertEqual(receipt["transformer_forwards"], 120)
        self.assertEqual(receipt["original_unipc_scheduler_calls"], 40)
        for bad in (
            types.SimpleNamespace(sample_calls=2, records=trace.records),
            types.SimpleNamespace(sample_calls=1, records=trace.records[:-1]),
            types.SimpleNamespace(
                sample_calls=1,
                records=[
                    _record(index, transformer_forwards=2 if index == 9 else 3)
                    for index in range(40)
                ],
            ),
            types.SimpleNamespace(
                sample_calls=1,
                records=[
                    _record(index, original_scheduler_calls=0 if index == 9 else 1)
                    for index in range(40)
                ],
            ),
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(trainer.EGNTCTrainingError):
                    trainer.validate_full_rollout_trace(bad)

    def test_lineage_rejects_old_cache_relabelled_as_round_two(self):
        first = {
            "round_index": 1,
            "execution_policy": "official_action",
            "input_controller_sha256": None,
            "output_controller_sha256": "3" * 64,
            "rollout_id": "1" * 64,
            "cache_sha256": "2" * 64,
            "state_fields_sha256": "8" * 64,
        }
        second = {
            "round_index": 2,
            "execution_policy": "learned_controller",
            "input_controller_sha256": "3" * 64,
            "output_controller_sha256": "6" * 64,
            "rollout_id": "4" * 64,
            "cache_sha256": "5" * 64,
            "state_fields_sha256": "9" * 64,
        }
        trainer.validate_round_lineage([first, second])
        for changed in (
            {**second, "rollout_id": first["rollout_id"]},
            {**second, "execution_policy": "official_action"},
            {**second, "input_controller_sha256": "7" * 64},
            {**second, "round_index": 1},
            {**second, "state_fields_sha256": first["state_fields_sha256"]},
        ):
            with self.subTest(changed=changed):
                with self.assertRaises(trainer.EGNTCTrainingError):
                    trainer.validate_round_lineage([first, changed])

    def test_prototype_gate_is_per_support_conjunction_and_non_deployable(self):
        good = [
            {
                "iid": iid,
                "round_postfit_objectives": [1.0, 0.95],
                "losses": {
                    "noop": 1.0,
                    "raw_action": 0.80,
                    "prototype": 0.75,
                    "prototype_phase_reverse": 0.82,
                    "prototype_sigma_shuffle": 0.81,
                },
            }
            for iid in ("support-1", "support-2")
        ]
        decision = trainer.prototype_gate(
            good, support_parameter_cosine=0.8, engineering_smoke=False
        )
        self.assertEqual(decision["representability_gate"], "GO")
        self.assertEqual(decision["tensor_representability_gate"], "GO")
        self.assertFalse(decision["deployable"])
        self.assertTrue(decision["diagnostic_only"])
        self.assertEqual(
            decision["aggregation"],
            "per_support_conjunction_and_worst_support_ratios",
        )
        failed = [dict(item) for item in good]
        failed[1] = {
            **failed[1],
            "losses": {**failed[1]["losses"], "prototype_sigma_shuffle": 0.76},
        }
        no_go = trainer.prototype_gate(
            failed, support_parameter_cosine=0.8, engineering_smoke=False
        )
        self.assertEqual(no_go["representability_gate"], "NO_GO")
        self.assertIn(
            "every_support_beats_sigma_shuffle_by_5pct", no_go["failed_checks"]
        )

    def test_smoke_gate_is_explicitly_not_evaluated(self):
        decision = trainer.prototype_gate(
            [], support_parameter_cosine=0.0, engineering_smoke=True
        )
        self.assertEqual(
            decision["representability_gate"],
            "NOT_EVALUATED_ENGINEERING_SMOKE",
        )
        self.assertFalse(decision["deployable"])


@unittest.skipIf(torch is None, "torch is unavailable")
class TensorAndArtifactTests(unittest.TestCase):
    def test_rollout_tensor_hash_bytes_do_not_require_numpy(self):
        first = torch.tensor([1.0, -2.5, 3.25], dtype=torch.float32)
        payload = trainer._tensor_content_bytes(first)
        self.assertEqual(len(payload), first.numel() * first.element_size())
        self.assertEqual(payload, trainer._tensor_content_bytes(first.clone()))
        changed = first.clone()
        changed[1] = -2.0
        self.assertNotEqual(payload, trainer._tensor_content_bytes(changed))

    def test_cache_arm_reads_authenticated_nested_controller_step_count(self):
        field = torch.zeros(1, dtype=torch.float32)
        statistics = {
            "total": 0.25,
            "feature_match": 0.10,
            "phase_rms_match": 0.05,
            "temporal_dc": 0.02,
            "phase0_parity": 0.01,
            "cross_sigma_consistency": 0.07,
            "parameter_l2": 0.0,
            "schedule_smoothness": 0.0,
        }
        nested_receipt = {"state": {"step_count": trainer.NUM_INFERENCE_STEPS}}
        teacher = types.SimpleNamespace(
            source_clean_cpu=field,
            target_clean_cpu=field,
        )
        with (
            mock.patch.object(trainer, "_cache_clean_cpu", side_effect=lambda value: value),
            mock.patch.object(
                trainer,
                "_execute_cached_controller",
                return_value=([field] * len(trainer.SUPERVISED_STEP_INDICES), nested_receipt),
            ) as execution,
            mock.patch.object(
                trainer,
                "_source_relative_objective",
                return_value=(field, statistics),
            ),
            mock.patch.object(egntc, "validate_controller_receipt"),
        ):
            result = trainer._evaluate_cache_arm(
                object(),
                arm="prototype",
                parameters=object(),
                controls=None,
                teacher=teacher,
                device=torch.device("cpu"),
            )
            execution.return_value = (
                [field] * len(trainer.SUPERVISED_STEP_INDICES),
                {"step_count": trainer.NUM_INFERENCE_STEPS},
            )
            with self.assertRaisesRegex(
                trainer.EGNTCTrainingError, "nested 40-step state"
            ):
                trainer._evaluate_cache_arm(
                    object(),
                    arm="prototype",
                    parameters=object(),
                    controls=None,
                    teacher=teacher,
                    device=torch.device("cpu"),
                )
        self.assertEqual(result["controller_trace_steps"], 40.0)

    def test_official_action_callback_returns_exact_object_and_has_no_target(self):
        cache = trainer.DetachedTrajectoryCache(
            rollout_id="1" * 64,
            round_index=1,
            execution_policy="official_action",
            input_controller_sha256=None,
        )
        source = torch.zeros(1, 16, 21, 60, 62, dtype=torch.float32)
        action = torch.ones_like(source)
        noop = torch.zeros_like(source)
        callback = trainer._RolloutCaptureCallback(
            cache=cache,
            execution_policy="official_action",
            parameters=None,
            source_clean=source,
        )
        fields = types.SimpleNamespace(
            step_index=0,
            timestep=float(egntc.sigma_strata.PINNED_TIMESTEPS[0]),
            sigma=float(egntc.sigma_strata.PINNED_POSITIVE_SIGMAS[0]),
            model_id="transformer_1",
            action_guided_clean=action,
            noop_guided_clean=noop,
            action_delta_clean=action - noop,
        )
        self.assertIs(callback(fields), action)
        self.assertEqual(len(cache.steps), 1)
        self.assertFalse(cache.steps[0].action_clean_cpu.requires_grad)
        self.assertEqual(cache.steps[0].action_clean_cpu.dtype, torch.float32)
        self.assertEqual(tuple(cache.steps[0].action_clean_cpu.shape[-2:]), (60, 62))

    def test_source_relative_objective_is_differentiable_only_through_controller(self):
        source = torch.zeros(1, 16, 21, 8, 8, dtype=torch.float32)
        target = source.clone()
        target[:, :, 1:] = torch.linspace(0.0, 0.2, 20).reshape(1, 1, 20, 1, 1)
        parameters = egntc.EGNTCParameters()
        scale = parameters.flat_tensor(detach=False).mean()
        predicted = [source + (0.01 * (index + 1) * scale) for index in range(6)]
        loss, statistics = trainer._source_relative_objective(
            predicted,
            source_clean=source,
            target_clean=target,
            parameters=parameters,
        )
        self.assertTrue(torch.isfinite(loss).item())
        self.assertEqual(set(statistics), {
            "total",
            "feature_match",
            "phase_rms_match",
            "temporal_dc",
            "phase0_parity",
            "cross_sigma_consistency",
            "parameter_l2",
            "schedule_smoothness",
        })
        loss.backward()
        self.assertTrue(
            all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in parameters.parameters()
            )
        )
        self.assertIsNone(target.grad)

    def test_controller_checkpoint_is_one_cpu_fp32_36d_tensor(self):
        try:
            from safetensors.torch import load_file
        except ImportError as error:  # pragma: no cover
            self.skipTest(str(error))
        vector = torch.linspace(-1.0, 1.0, 36, dtype=torch.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controller.safetensors"
            receipt = trainer._atomic_save_controller(path, vector)
            values = load_file(str(path), device="cpu")
        self.assertEqual(set(values), {"controller_raw_36d"})
        self.assertTrue(torch.equal(values["controller_raw_36d"], vector))
        self.assertEqual(receipt["shape"], [36])
        self.assertEqual(receipt["dtype"], "torch.float32")

    def test_k2_midpoint_uses_canonical_flat_parameter_order(self):
        first = torch.arange(36, dtype=torch.float32)
        second = first + 2.0
        midpoint = torch.stack((first, second)).mean(dim=0)
        parameters = egntc.EGNTCParameters.from_flat_tensor(midpoint)
        self.assertTrue(torch.equal(parameters.flat_tensor(detach=True), first + 1.0))
        self.assertEqual(tuple(parameters.alpha_logits.shape), (6, 4))
        self.assertEqual(tuple(parameters.kappa_raw.shape), (6,))
        self.assertEqual(tuple(parameters.rho_raw.shape), (6,))

    def test_companion_receipt_reads_nested_training_provenance(self):
        source = inspect.getsource(trainer.main)
        self.assertIn('provenance = companion.get("training_provenance")', source)
        self.assertNotIn('companion.get("deployable")', source)
        self.assertIn("build_controller_training_receipt", inspect.getsource(
            trainer._write_controller_companion_receipt
        ))


if __name__ == "__main__":
    unittest.main()
