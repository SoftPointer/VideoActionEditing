from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import counterfactual_proposal_motion_branch_ulysses_smoke as smoke


SMOKE_PATH = METHOD_ROOT / "counterfactual_proposal_motion_branch_ulysses_smoke.py"
BRANCH_PATH = METHOD_ROOT / "counterfactual_proposal_motion_branch.py"
SBATCH_PATH = (
    METHOD_ROOT
    / "scripts"
    / "auh_counterfactual_proposal_motion_branch_ulysses_smoke.sbatch"
)


class CPMRUlyssesSmokeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.smoke_source = SMOKE_PATH.read_text(encoding="utf-8")
        cls.branch_source = BRANCH_PATH.read_text(encoding="utf-8")
        cls.sbatch_source = SBATCH_PATH.read_text(encoding="utf-8")
        cls.smoke_tree = ast.parse(cls.smoke_source)

    def test_exact_dog_81_frame_production_geometry(self):
        self.assertEqual(smoke.GLOBAL_Q, 39_060)
        self.assertEqual(smoke.LOCAL_Q, 9_765)
        self.assertEqual(smoke.SOURCE_Q, 19_530)
        self.assertEqual(smoke.TARGET_Q, 19_530)
        self.assertEqual(smoke.CARRIER_KV, 1_344)
        self.assertEqual(smoke.HEADS, 12)
        self.assertEqual(smoke.HEAD_DIM, 128)
        self.assertEqual(smoke.HIDDEN_SIZE, 1_536)
        self.assertEqual(smoke.DOG_SOURCE_SHA256, (
            "5ed911f66fea3ed2000f507412da75adecb8099b26b71089d0fd2c0ac2982b18"
        ))
        self.assertIn('"frames": 81', self.smoke_source)
        self.assertIn('"latent_phases": 21', self.smoke_source)
        self.assertNotIn('"frames": 41', self.smoke_source)
        self.assertNotIn("--num-frames 41", self.sbatch_source)

    def test_video_probe_is_in_process_and_decodes_exact_frames(self):
        self.assertIn('import av', self.smoke_source)
        self.assertIn('with av.open(str(path), mode="r") as container:', self.smoke_source)
        self.assertIn('sum(1 for _ in container.decode(stream))', self.smoke_source)
        self.assertNotIn('"ffprobe"', self.smoke_source)

    def test_pins_exact_official_revisions_and_processor(self):
        self.assertEqual(
            smoke.BERNINI_COMMIT,
            "2d2b4591ac053ec25c6371b01a5a6746679e5793",
        )
        self.assertEqual(
            smoke.VEOMNI_COMMIT,
            "f90b3dc6fbb0ce693745223cc7a94064123dbf4d",
        )
        self.assertIn("from bernini.models.transformer_wan import WanAttnProcessor2_0", self.smoke_source)
        self.assertGreaterEqual(self.smoke_source.count("WanAttnProcessor2_0()"), 3)

    def test_real_diffusers_attn2_call_exercises_kwarg_filter(self):
        self.assertIn("single_block.attn2.set_processor(wrapper)", self.smoke_source)
        self.assertEqual(
            self.smoke_source.count(
                "single_block.attn2(q_local, **text_kwargs)"
            ),
            2,
        )
        self.assertIn(
            "single_block.add_module(branch.MOTION_MODULE_NAME, motion)",
            self.smoke_source,
        )
        self.assertNotIn("zero_output = wrapper(", self.smoke_source)
        self.assertNotIn("active_output = wrapper(", self.smoke_source)
        signature = ast.parse(self.branch_source)
        processor_calls = [
            node
            for node in ast.walk(signature)
            if isinstance(node, ast.FunctionDef)
            and node.name == "__call__"
            and any(
                isinstance(parent, ast.ClassDef)
                and parent.name == "CPMRTextAttnProcessor"
                and node in parent.body
                for parent in ast.walk(signature)
            )
        ]
        self.assertEqual(len(processor_calls), 1)
        parameter_names = [item.arg for item in processor_calls[0].args.args]
        self.assertIn("batch_image_vae_seqlen", parameter_names)
        self.assertIn("origin_hidden_states_seq_len", parameter_names)
        self.assertIn("cu_seqlens_k_cross_cache", parameter_names)

    def test_batch_lengths_use_real_gpu_tensor_path(self):
        self.assertIn("batch_image_vae_seqlen = torch.tensor(", self.smoke_source)
        self.assertIn("[GLOBAL_Q], dtype=torch.int64, device=device", self.smoke_source)
        self.assertGreaterEqual(
            self.smoke_source.count(
                "batch_image_vae_seqlen=batch_image_vae_seqlen"
            ),
            2,
        )
        self.assertIn(
            '"batch_image_vae_seqlen": batch_image_vae_seqlen',
            self.smoke_source,
        )

    def test_official_a2a_proxies_are_instrumented_and_fail_closed(self):
        fake = SimpleNamespace(
            gather_seq_scatter_heads=lambda value: value + 1,
            gather_heads_scatter_seq=lambda value: value + 2,
        )
        original_sequence = fake.gather_seq_scatter_heads
        original_heads = fake.gather_heads_scatter_seq
        with smoke._instrument_official_a2a(fake) as counts:
            self.assertEqual(fake.gather_seq_scatter_heads(3), 4)
            self.assertEqual(fake.gather_heads_scatter_seq(3), 5)
            self.assertEqual(
                counts,
                {
                    "gather_seq_scatter_heads": 1,
                    "gather_heads_scatter_seq": 1,
                },
            )
        self.assertIs(fake.gather_seq_scatter_heads, original_sequence)
        self.assertIs(fake.gather_heads_scatter_seq, original_heads)
        self.assertIn("motion branch entered A2A", self.smoke_source)
        self.assertIn(
            '"reference_only_dist_all_gather_calls_per_rank"',
            self.smoke_source,
        )

    def test_cross_sp_shapes_and_cu_contract_are_runtime_asserted(self):
        for fragment in (
            "(1, LOCAL_Q, HIDDEN_SIZE)",
            "(1, GLOBAL_Q, HIDDEN_SIZE)",
            'event["k_shape"] != [CARRIER_KV, HEADS, HEAD_DIM]',
            'event["v_shape"] != [CARRIER_KV, HEADS, HEAD_DIM]',
            '"cu_q": [int(item) for item in result[1].tolist()]',
            '"cu_k": [int(item) for item in result[0].tolist()]',
        ):
            self.assertIn(fragment, self.smoke_source)
        self.assertIn("[int(item) for item in text_cu_q.tolist()] != [0, LOCAL_Q]", self.smoke_source)
        self.assertIn("[int(item) for item in text_cu_k.tolist()] != [0, TEXT_TOKENS]", self.smoke_source)

    def test_mask_is_global_pad_slice_and_applied_after_output_bias(self):
        global_mask = self.branch_source.index(
            "global_mask = hidden_states.new_zeros((1, GLOBAL_VISUAL_TOKENS, 1))"
        )
        pad_slice = self.branch_source.index(
            "local_mask = slice_fn(pad_fn(global_mask, dim=1), dim=1)"
        )
        output_bias = self.branch_source.index("output = self.to_out[0](output)")
        dropout = self.branch_source.index("output = self.to_out[1](output)")
        mask_apply = self.branch_source.index(
            "output = torch.where(local_mask.bool(), output, torch.zeros_like(output))"
        )
        self.assertLess(global_mask, pad_slice)
        self.assertLess(output_bias, dropout)
        self.assertLess(dropout, mask_apply)
        self.assertIn("source residual leaked after to_out bias", self.smoke_source)
        self.assertIn("expected_mask_sums = [0, 0, LOCAL_Q, LOCAL_Q]", self.smoke_source)

    def test_zero_and_active_gate_certificates_are_explicit(self):
        self.assertEqual(smoke.ACTIVE_GATE, 0.10)
        self.assertIn("if zero_output is not recorder.last_output", self.smoke_source)
        self.assertIn("zero_output.data_ptr() != recorder.last_output.data_ptr()", self.smoke_source)
        self.assertIn("rank 0/1 active output is not byte-exact", self.smoke_source)
        self.assertIn("def _tensor_byte_equal", self.smoke_source)
        self.assertIn(
            "source_active_byte_exact = _tensor_byte_equal(",
            self.smoke_source,
        )
        self.assertIn(
            "rank >= 2 or _tensor_byte_equal(torch, active_output, active_base)",
            self.smoke_source,
        )
        self.assertIn("active_target_delta_l2", self.smoke_source)
        self.assertIn('"zero_gate_delegations": 1', self.smoke_source)
        self.assertIn('"motion_calls": 1', self.smoke_source)
        self.assertIn(
            "zero_binding_receipt != expected_binding_receipt",
            self.smoke_source,
        )
        self.assertIn(
            "active_binding_receipt != expected_binding_receipt",
            self.smoke_source,
        )
        self.assertIn(
            "wrapper_stats != expected_wrapper_stats",
            self.smoke_source,
        )

    def test_phase_zero_and_real_encoder_identity_are_fail_closed(self):
        self.assertIn(
            "carrier[:, : branch.CARRIER_TOKENS_PER_PHASE].zero_()",
            self.smoke_source,
        )
        self.assertIn("activity[:, 1:] = True", self.smoke_source)
        self.assertIn("raw_prompt = text.clone()", self.smoke_source)
        self.assertIn("prompt_object=raw_prompt", self.smoke_source)
        self.assertIn("encoder_hidden_states=raw_prompt", self.smoke_source)
        self.assertIn(
            "branch._conditioned_encoder_binding_for_processors((wrapper,))",
            self.smoke_source,
        )
        self.assertIn(
            '"phase_zero_positive_zero_and_inactive": True',
            self.smoke_source,
        )
        self.assertIn(
            '"post_transform_attn2_encoder_identity_bound": True',
            self.smoke_source,
        )
        self.assertIn(
            '"outer_raw_and_attn2_encoder_objects_distinct": True',
            self.smoke_source,
        )

    def test_collective_claim_separates_declared_from_measured(self):
        self.assertIn(
            '"explicit_custom_collective_calls": 0',
            self.smoke_source,
        )
        self.assertIn(
            '"measured_custom_collective_calls": None',
            self.smoke_source,
        )
        self.assertIn("motion_stats != expected_motion_stats", self.smoke_source)
        self.assertIn('"a2a_proxy_calls": dict(a2a_counts)', self.smoke_source)

    def test_gathered_branch_matches_same_weight_full_reference(self):
        self.assertIn("_single_rank_bernini_state", self.smoke_source)
        self.assertIn("full_reference = motion(", self.smoke_source)
        self.assertIn("reference_parity = _parity", self.smoke_source)
        self.assertIn("gathered branch/full reference parity failed", self.smoke_source)
        self.assertIn("source residual is not exact zero", self.smoke_source)
        self.assertIn("target residual is identically zero", self.smoke_source)

    def test_receipt_forbids_video_science_training_and_lora_claims(self):
        for field in (
            '"scientific_claim": False',
            '"video_claim": False',
            '"video_quality_claim": False',
            '"training_claim": False',
            '"lora_claim": False',
            '"full_transformer_forward_claim": False',
            '"gradient_checkpoint_training_claim": False',
            '"frozen_single_block_engineering_only": True',
            '"synthetic_hidden_content": True',
        ):
            self.assertIn(field, self.smoke_source)

    def test_sbatch_is_four_mi210_git_archive_and_does_not_submit(self):
        self.assertIn("#SBATCH --gres=gpu:mi210:4", self.sbatch_source)
        self.assertIn("--nproc_per_node=4", self.sbatch_source)
        self.assertIn('git -C "${source_repository}" archive --format=tar', self.sbatch_source)
        self.assertIn("counterfactual_proposal_motion_branch.py", self.sbatch_source)
        self.assertIn("counterfactual_proposal_motion_branch_ulysses_smoke.py", self.sbatch_source)
        self.assertNotIn("\nsbatch ", self.sbatch_source)
        result = subprocess.run(
            ["bash", "-n", str(SBATCH_PATH)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_sbatch_receipt_is_create_only_and_validated(self):
        self.assertIn("os.O_EXCL", self.sbatch_source)
        self.assertNotIn("os.replace", self.sbatch_source)
        self.assertIn("refusing to overwrite receipt", self.sbatch_source)
        self.assertIn("receipt digest differs", self.sbatch_source)
        self.assertIn("verified and all(item is True", self.sbatch_source)
        self.assertIn(
            'receipt.get("full_transformer_forward_claim") is False',
            self.sbatch_source,
        )
        self.assertIn(
            'receipt.get("gradient_checkpoint_training_claim") is False',
            self.sbatch_source,
        )
        self.assertIn('runtime.get("reference_only_dist_all_gather_calls_per_rank") == 2', self.sbatch_source)
        self.assertIn(
            'runtime.get("zero_gate_conditioned_encoder_binding") == expected_binding',
            self.sbatch_source,
        )
        self.assertIn(
            'runtime.get("active_conditioned_encoder_binding") == expected_binding',
            self.sbatch_source,
        )
        self.assertIn(
            'runtime.get("wrapper_statistics") == expected_wrapper_stats',
            self.sbatch_source,
        )
        self.assertIn(
            'runtime.get("motion_statistics") == expected_motion_stats',
            self.sbatch_source,
        )
        self.assertIn('os.chmod(output, 0o400)', self.sbatch_source)


if __name__ == "__main__":
    unittest.main()
