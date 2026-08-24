from pathlib import Path
import ast
import unittest


ROOT = Path(__file__).resolve().parents[1]
HOLDER = ROOT / "scripts/auh_infer_preservation_residual_single_holder_v1.sh"
EXEC = ROOT / "scripts/auh_infer_preservation_residual_exec_v1.sh"
RUNTIME = ROOT / "infer_preservation_residual_action_canary_v1.py"


class PreservationInferenceHolderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.holder = HOLDER.read_text(encoding="utf-8")
        cls.exec = EXEC.read_text(encoding="utf-8")
        cls.runtime = RUNTIME.read_text(encoding="utf-8")

    def test_only_authorized_parents_and_parent_is_retained(self) -> None:
        self.assertIn("135407:auh7-1b-gpu-260", self.holder)
        self.assertIn("135411:auh7-1b-gpu-214", self.holder)
        self.assertNotIn("scancel", self.holder.lower())
        self.assertIn("parent_not_released=true", self.holder)

    def test_all8_holder_serially_reuses_one_sp4_group(self) -> None:
        self.assertIn("--gres=gpu:mi210:8", self.holder)
        self.assertIn("launch_group dog 0,1,2,3", self.exec)
        self.assertIn("launch_group human 0,1,2,3", self.exec)
        dog_launch = self.exec.index("launch_group dog 0,1,2,3")
        human_launch = self.exec.index("launch_group human 0,1,2,3")
        self.assertLess(dog_launch, human_launch)
        self.assertNotIn('logs/dog.log" 2>&1 &', self.exec)
        self.assertNotIn('logs/human.log" 2>&1 &', self.exec)
        self.assertIn("--nproc_per_node=4", self.exec)

    def test_inference_compares_native_and_unit_gain_preservation(self) -> None:
        self.assertIn("native-rv2v.mp4", self.exec)
        self.assertIn("preservation-residual.mp4", self.exec)
        self.assertIn("NativeRV2VPreservationResidualPatch", self.runtime)
        self.assertNotIn("ARM_SCALES", self.runtime)
        self.assertIn('"feature_reward_consumed": False', self.runtime)
        self.assertIn('"synthetic_target_consumed": False', self.runtime)

    def test_host_checkpoint_load_is_serialized_across_both_groups(self) -> None:
        self.assertIn("PRESERVATION_INFER_LOAD_LOCK", self.exec)
        self.assertIn("fcntl.LOCK_EX", self.runtime)
        self.assertIn("model.to(device)", self.runtime)

    def test_host_residency_is_bounded_after_prompt_encoding(self) -> None:
        self.assertIn("if distributed.rank == 0:", self.runtime)
        self.assertIn("rank_zero_only_vae", self.runtime)
        self.assertIn("model.t5_text_encoder = None", self.runtime)
        self.assertIn("full_guard_after_prompt != full_guard_before_prompt", self.runtime)
        self.assertIn("del adapter, diffusion, model, result", self.runtime)
        self.assertNotIn('model.to("cpu")', self.runtime)
        self.assertIn("_rank_zero_strong_model_freeze_certificate", self.runtime)
        self.assertIn("_model_mutation_guard", self.runtime)
        self.assertIn("_trim_host_allocator", self.runtime)

    def test_mutation_guards_compare_matching_model_topologies(self) -> None:
        native_before = self.runtime.index("native_sampling_guard_before =")
        install = self.runtime.index("preservation_load.strict_load(")
        adapted_before = self.runtime.index("adapted_sampling_guard_before =")
        adapted_after = self.runtime.index("adapted_sampling_guard_after =")
        restore = self.runtime.index("adapter.restore()", adapted_after)
        native_after = self.runtime.index("native_sampling_guard_after =", restore)
        self.assertLess(native_before, install)
        self.assertLess(install, adapted_before)
        self.assertLess(adapted_before, adapted_after)
        self.assertLess(adapted_after, restore)
        self.assertLess(restore, native_after)
        self.assertNotIn("sampling_guard_after != sampling_guard_before", self.runtime)

    def test_vae_cpu_offload_is_rank_zero_only(self) -> None:
        tree = ast.parse(self.runtime)
        rank_zero_vae_if = None
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            body_source = "\n".join(
                ast.get_source_segment(self.runtime, item) or "" for item in node.body
            )
            if "AutoencoderKLWan.from_pretrained" in body_source:
                rank_zero_vae_if = node
                break
        self.assertIsNotNone(rank_zero_vae_if)
        assert rank_zero_vae_if is not None
        rank_zero_body = "\n".join(
            ast.get_source_segment(self.runtime, item) or ""
            for item in rank_zero_vae_if.body
        )
        nonzero_body = "\n".join(
            ast.get_source_segment(self.runtime, item) or ""
            for item in rank_zero_vae_if.orelse
        )
        self.assertIn('vae.to("cpu")', rank_zero_body)
        self.assertNotIn('vae.to("cpu")', nonzero_body)


if __name__ == "__main__":
    unittest.main()
