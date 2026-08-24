from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_native_v_axis_exact81_probe_v1 as runner  # noqa: E402
import native_v_axis_guidance_v1 as core  # noqa: E402


try:
    import torch

    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


def _valid_trace(arm: str) -> dict:
    digest = "a" * 64
    omega = float(core.arm_contract(arm)["omega_video"])
    steps = []
    for index in range(40):
        native = digest
        executed = "b" * 64 if arm == "V-off" else native
        steps.append(
            {
                "step_index": index,
                "omega_video_hex": omega.hex(),
                "standalone_v_axis_active": omega != 0.0,
                "branch_call_counts": {
                    "none_uncond": 1,
                    "V_uncond": 1,
                    "VI_uncond": 1,
                    "VI_cond": 1,
                },
                "branch_target_raw_sha256": {
                    "none_uncond": digest,
                    "V_uncond": digest,
                    "VI_uncond": digest,
                    "VI_cond": digest,
                },
                "transformer_forward_count": 4,
                "original_scheduler_call_count": 1,
                "native_formula_exact_parity": True,
                "v_vi_u_minus_v_v_term_retained": True,
                "native_velocity_raw_sha256": native,
                "executed_velocity_raw_sha256": executed,
                "scheduler_received_original_model_output_object": arm != "V-off",
                "target_tokens": 7560,
            }
        )
    return {
        "step_count": 40,
        "expected_transformer_forwards": 160,
        "observed_transformer_forwards": 160,
        "steps": steps,
    }


class NativeVAxisCoreTests(unittest.TestCase):
    def test_registered_formula_changes_only_standalone_v_axis(self) -> None:
        values = (2.0, 5.0, 11.0, 17.0)
        native = core.v_axis_velocity(*values, omega_video=1.25)
        off = core.v_axis_velocity(*values, omega_video=0.0)
        self.assertEqual(native, 2.0 + 1.25 * 3.0 + 4.5 * 6.0 + 4.0 * 6.0)
        self.assertEqual(off, 2.0 + 4.5 * 6.0 + 4.0 * 6.0)
        self.assertEqual(native - off, 1.25 * (values[1] - values[0]))
        with self.assertRaises(core.NativeVAxisGuidanceError):
            core.v_axis_velocity(*values, omega_video=0.5)

    def test_three_arms_change_only_registered_coordinates(self) -> None:
        on = core.arm_contract("V-on")
        off = core.arm_contract("V-off")
        wrong = core.arm_contract("wrong-V")
        self.assertEqual(tuple(core.ARM_ORDER), ("V-on", "V-off", "wrong-V"))
        self.assertEqual(on["omega_video"], 1.25)
        self.assertEqual(off["omega_video"], 0.0)
        self.assertEqual(wrong["omega_video"], 1.25)
        self.assertEqual(on["full_video_condition_role"], "correct")
        self.assertEqual(off["full_video_condition_role"], "correct")
        self.assertEqual(wrong["full_video_condition_role"], "wrong")
        for row in (on, off, wrong):
            self.assertTrue(row["correct_image_references"])
            self.assertTrue(row["same_instruction"])
            self.assertTrue(row["same_scheduler"])
            self.assertTrue(row["v_vi_u_minus_v_v_term_retained"])

    def test_hook_contract_is_not_i_axis_131497(self) -> None:
        contract = core.hook_contract()
        self.assertEqual(contract["arm_order"], ["V-on", "V-off", "wrong-V"])
        self.assertIn("0.0*(vV-v0)", contract["v_off_formula"])
        self.assertEqual(contract["transformer_forwards_per_step"], 4)
        self.assertFalse(contract["feature_scorer"])
        self.assertFalse(contract["selection"])
        self.assertNotIn("I_axis", contract["method"])


def _load_i_axis_fake_module():
    path = Path(__file__).with_name("test_native_i_axis_guidance.py")
    spec = importlib.util.spec_from_file_location("_native_i_axis_fake_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load native-I fake runtime")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_fake_v_arm(arm: str):
    fake = _load_i_axis_fake_module()
    diffusion = fake._FakeDiffusion()
    hook = core.NativeVAxisGuidanceHook(
        diffusion,
        arm=arm,
        expected_steps=40,
        expected_bernini_commit=core.PINNED_BERNINI_COMMIT,
        observed_wan_diffusion_sha256=core.PINNED_WAN_DIFFUSION_SHA256,
    )
    refs = [torch.full((1, 16, 1, 1, 1), float(i + 1)) for i in range(4)]
    hook.install()
    try:
        result = diffusion.sample(
            prompt_embeds=torch.ones((1, 2, 3)),
            uncond_prompt_embeds=torch.zeros((1, 2, 3)),
            image_vae_latents=None,
            multi_video_vae_latents=[torch.ones((1, 16, 21, 1, 1))],
            multi_image_vae_latents=refs,
            width=16,
            height=16,
            num_frames=81,
            num_inference_steps=40,
            guidance_mode="rv2v",
            omega_vid=1.25,
            omega_img=4.5,
            omega_txt=4.0,
            omega_scale=0.8,
            flow_shift=5.0,
            seed=7,
            eta=0.5,
            norm_threshold=(50.0, 50.0),
            momentum=0.0,
            device="cpu",
        )
    finally:
        hook.restore()
    return diffusion, hook, result


@unittest.skipUnless(_TORCH_AVAILABLE, "PyTorch runtime is required")
class NativeVAxisHookTests(unittest.TestCase):
    def test_v_on_and_wrong_v_are_native_pointer_parity(self) -> None:
        for arm in ("V-on", "wrong-V"):
            diffusion, hook, result = _run_fake_v_arm(arm)
            self.assertEqual(tuple(result.shape), (1, 4, 4))
            self.assertEqual(diffusion.shared_call_count, 160)
            self.assertEqual(hook.trace["observed_transformer_forwards"], 160)
            self.assertTrue(
                all(
                    row["scheduler_received_original_model_output_object"]
                    for row in hook.trace["steps"]
                )
            )
            self.assertTrue(
                all(
                    row["native_velocity_raw_sha256"]
                    == row["executed_velocity_raw_sha256"]
                    for row in hook.trace["steps"]
                )
            )

    def test_v_off_changes_every_scheduler_input_without_extra_forward(self) -> None:
        diffusion, hook, _ = _run_fake_v_arm("V-off")
        self.assertEqual(diffusion.shared_call_count, 160)
        self.assertEqual(hook.trace["observed_transformer_forwards"], 160)
        self.assertTrue(
            all(
                not row["scheduler_received_original_model_output_object"]
                for row in hook.trace["steps"]
            )
        )
        self.assertTrue(
            all(row["v_axis_correction_rms"] > 0 for row in hook.trace["steps"])
        )
        self.assertNotIn("sample", vars(diffusion))
        self.assertNotIn("step", vars(diffusion.scheduler))


class NativeVAxisRunnerTests(unittest.TestCase):
    def test_sealed_spec_matches_runtime_contract(self) -> None:
        path = METHOD_ROOT / "assets/native_v_axis_exact81_core2_v1.json"
        raw = path.read_bytes()
        root, dog, loaded, observed = runner.load_cell_spec(
            path,
            expected_file_sha256=hashlib.sha256(raw).hexdigest(),
            cell_id="dog",
        )
        self.assertEqual(loaded, path)
        self.assertEqual(observed, hashlib.sha256(raw).hexdigest())
        self.assertEqual(root["contract"], runner._expected_spec_contract())
        self.assertEqual(dog["cell_id"], "dog")
        self.assertFalse(dog["wrong_source_pure_identity_control"])

    def test_spec_loader_rejects_post_registration_change(self) -> None:
        source = METHOD_ROOT / "assets/native_v_axis_exact81_core2_v1.json"
        root = json.loads(source.read_text(encoding="utf-8"))
        root["contract"]["arm_order"] = ["V-off", "V-on", "wrong-V"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.json"
            path.write_text(json.dumps(root), encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(runner.NativeVAxisProbeError):
                runner.load_cell_spec(
                    path, expected_file_sha256=digest, cell_id="dog"
                )

    def test_exact40_receipt_gate_accepts_all_three_arms(self) -> None:
        for arm in core.ARM_ORDER:
            gate = runner.validate_exact40_trace(_valid_trace(arm), arm=arm)
            self.assertTrue(gate["passed"])
            self.assertEqual(gate["step_count"], 40)
            self.assertEqual(len(gate["digest"]), 64)

    def test_exact40_gate_rejects_coefficient_or_scheduler_pointer_drift(self) -> None:
        bad_coefficient = _valid_trace("V-off")
        bad_coefficient["steps"][7]["omega_video_hex"] = float(1.25).hex()
        with self.assertRaises(runner.NativeVAxisProbeError):
            runner.validate_exact40_trace(bad_coefficient, arm="V-off")
        bad_native = _valid_trace("wrong-V")
        bad_native["steps"][3][
            "scheduler_received_original_model_output_object"
        ] = False
        with self.assertRaises(runner.NativeVAxisProbeError):
            runner.validate_exact40_trace(bad_native, arm="wrong-V")

    def test_cli_exposes_no_target_reward_or_selection_surface(self) -> None:
        options = {
            option
            for action in runner.build_parser()._actions
            for option in action.option_strings
        }
        forbidden = {
            "--target-video", "--reward", "--scorer", "--metric",
            "--rank", "--select", "--trainer", "--optimizer", "--lora",
            "--mask", "--pose", "--flow", "--track", "--trajectory",
        }
        self.assertTrue(forbidden.isdisjoint(options))

    def test_runtime_writes_exact81_outputs_and_never_scores(self) -> None:
        source = Path(runner.__file__).read_text(encoding="utf-8")
        self.assertIn("native._save_outputs(", source)
        self.assertIn("prior._output_staging_directory(output_dir)", source)
        self.assertIn("source-correct.mp4", source)
        self.assertIn("source-wrong-V.mp4", source)
        self.assertIn('"feature_scorer_consumed": False', source)
        self.assertIn('"ranking_performed": False', source)
        self.assertIn('"best_arm_selected": False', source)
        self.assertNotIn("normalized_guidance(", source)

    def test_60g_resource_contract_is_sealed_and_receipted(self) -> None:
        contract = runner.resource_lifetime_contract()
        self.assertEqual(contract["slurm_child_memory_bytes"], 60 * 1024**3)
        self.assertEqual(contract["world_size"], 4)
        self.assertEqual(contract["sequence_parallel_size"], 4)
        for key in (
            "rank_serialized_checkpoint_deserialize",
            "model_moved_to_rank_device_before_load_lock_release",
            "host_allocator_trim_after_each_rank_load",
            "prompt_embeddings_encoded_once_per_process",
            "text_encoder_retired_before_vae_and_sampling",
            "vae_instantiated_on_rank_zero_only",
            "condition_latents_broadcast_rank_zero_to_all_ranks",
            "sampling_model_destroyed_without_cpu_offload_before_rank_zero_decode",
            "dog_human_process_trees_serial",
        ):
            self.assertTrue(contract[key], key)
        spec = runner._expected_spec_contract()
        self.assertEqual(spec["resource_lifetime_contract"], contract)
        source = Path(runner.__file__).read_text(encoding="utf-8")
        self.assertIn('"resource_lifetime": {', source)
        self.assertIn('"freeze_certificate": {', source)

    def test_checkpoint_load_is_serialized_and_moved_before_unlock(self) -> None:
        source = Path(runner.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        load_context = None
        for node in ast.walk(tree):
            if not isinstance(node, (ast.With, ast.AsyncWith)):
                continue
            text = ast.get_source_segment(source, node) or ""
            if "_serialized_host_checkpoint_load()" in text:
                load_context = text
                break
        self.assertIsNotNone(load_context)
        assert load_context is not None
        self.assertIn("BerniniRendererModel(config)", load_context)
        self.assertIn("model.to(device)", load_context)
        self.assertIn("_trim_host_allocator()", load_context)
        self.assertIn("fcntl.LOCK_EX", source)
        self.assertIn("NATIVE_V_AXIS_LOAD_LOCK", source)

    def test_t5_is_one_shot_then_retired_and_sampler_uses_embeddings(self) -> None:
        source = Path(runner.__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count("positive_embeds = model.encode_prompt("), 1)
        self.assertEqual(source.count("negative_embeds = model.encode_prompt("), 1)
        self.assertIn("model.t5_text_encoder = None", source)
        self.assertIn("del retired_text_encoder, tokenizer", source)
        self.assertIn("diffusion.sample(", source)
        self.assertNotIn("lambda video=selected_video, selected_seed=seed: model.sample(", source)
        self.assertLess(
            source.index("model.t5_text_encoder = None"),
            source.index("vae = AutoencoderKLWan.from_pretrained("),
        )

    def test_vae_is_rank_zero_only_and_conditions_are_broadcast(self) -> None:
        source = Path(runner.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        vae_if = None
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            body = "\n".join(
                ast.get_source_segment(source, item) or "" for item in node.body
            )
            if "AutoencoderKLWan.from_pretrained" in body:
                vae_if = node
                break
        self.assertIsNotNone(vae_if)
        assert vae_if is not None
        test_text = ast.get_source_segment(source, vae_if.test) or ""
        self.assertEqual(test_text, "distributed.rank == 0")
        nonzero = "\n".join(
            ast.get_source_segment(source, item) or "" for item in vae_if.orelse
        )
        self.assertNotIn("AutoencoderKLWan.from_pretrained", nonzero)
        self.assertIn("if distributed.rank != 0:", source)
        self.assertIn("full_correct_latent = torch.empty(", source)
        self.assertIn("native._broadcast_condition_from_rank_zero(", source)

    def test_sampling_model_is_destroyed_without_cpu_offload_before_decode(self) -> None:
        source = Path(runner.__file__).read_text(encoding="utf-8")
        destroy = source.index("del diffusion, model, positive_embeds, negative_embeds")
        decode = source.index("outputs = native._save_outputs(")
        self.assertLess(destroy, decode)
        self.assertNotIn('model.to("cpu")', source)


if __name__ == "__main__":
    unittest.main()
