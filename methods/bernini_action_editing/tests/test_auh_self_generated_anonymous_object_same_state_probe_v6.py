from pathlib import Path
import hashlib
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import auh_self_generated_anonymous_object_same_state_probe_v6 as runner
import self_generated_anonymous_object_registry_v6 as registry


class AUHAnonymousObjectSameStateProbeV6Test(unittest.TestCase):
    @staticmethod
    def _legacy_contiguous_tensor_sha256(value):
        logical = value.detach().contiguous()
        header = registry.canonical_json_bytes(
            {"dtype": str(logical.dtype), "shape": list(logical.shape)}
        )
        raw = logical.view(torch.uint8).cpu().numpy().tobytes(order="C")
        digest = hashlib.sha256()
        digest.update(header)
        digest.update(b"\x00")
        digest.update(raw)
        return digest.hexdigest()

    def test_tensor_digest_preserves_legacy_contiguous_bytes(self):
        fixtures = (
            torch.arange(6, dtype=torch.int64).reshape(2, 3),
            torch.tensor([[1.25, -0.0], [3.5, -7.0]], dtype=torch.float32),
            torch.tensor([[1.5, -2.25], [0.0, 9.0]], dtype=torch.bfloat16),
            torch.empty((0, 3), dtype=torch.bfloat16),
        )
        for value in fixtures:
            with self.subTest(dtype=str(value.dtype)):
                self.assertTrue(value.is_contiguous())
                self.assertEqual(
                    runner.locator.tensor_sha256(value),
                    self._legacy_contiguous_tensor_sha256(value),
                )

    def test_tensor_digest_canonicalizes_live_stride_zero_and_other_layouts(self):
        timestep = torch.tensor(999, dtype=torch.int64)
        live_timestep_object = timestep.expand(1)
        self.assertEqual(live_timestep_object.stride(), (0,))
        self.assertEqual(
            runner.locator.tensor_sha256(live_timestep_object),
            runner.locator.tensor_sha256(torch.tensor([999], dtype=torch.int64)),
        )

        expanded = torch.tensor(17, dtype=torch.int64).expand(4, 3)
        self.assertEqual(expanded.stride(), (0, 0))
        self.assertEqual(
            runner.locator.tensor_sha256(expanded),
            runner.locator.tensor_sha256(torch.full((4, 3), 17, dtype=torch.int64)),
        )

        transposed = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
        self.assertFalse(transposed.is_contiguous())
        self.assertEqual(
            runner.locator.tensor_sha256(transposed),
            runner.locator.tensor_sha256(transposed.contiguous()),
        )

        bf16_transposed = torch.tensor(
            [[1.5, -2.25, 0.0], [9.0, 3.0, -4.0]],
            dtype=torch.bfloat16,
        ).t()
        self.assertEqual(
            runner.locator.tensor_sha256(bf16_transposed),
            runner.locator.tensor_sha256(bf16_transposed.contiguous()),
        )

        scalar = torch.tensor(-31, dtype=torch.int64)
        scalar_digest = runner.locator.tensor_sha256(scalar)
        self.assertRegex(scalar_digest, r"^[0-9a-f]{64}$")
        self.assertEqual(scalar_digest, runner.locator.tensor_sha256(scalar.clone()))
        self.assertNotEqual(
            scalar_digest,
            runner.locator.tensor_sha256(scalar.reshape(1)),
        )

    def test_forward_multiplicity_and_B0_matrix(self):
        self.assertEqual(runner.EXPECTED_TRAJECTORY_STEPS, 120)
        self.assertEqual(runner.EXPECTED_TRAJECTORY_FORWARDS, 240)
        self.assertEqual(runner.EXPECTED_B0_CELLS, 9)
        self.assertEqual(runner.EXPECTED_OBSERVER_FORWARDS, 72)
        self.assertEqual(runner.EXPECTED_PROJECTED_BLOCK_CAPTURES, 288)
        self.assertEqual(runner.EXPECTED_TOTAL_FROZEN_FORWARDS, 321)
        contract = runner.probe_contract()
        self.assertEqual(contract["trajectory_guidance_transformer_forwards_per_step"], 2)
        self.assertEqual(contract["total_frozen_transformer_forward_count"], 321)
        self.assertEqual(contract["site_source_geometry"], [21, 37, 25])
        self.assertEqual(contract["registry_patch_geometry"], [21, 37, 25])
        self.assertEqual(
            tuple(runner.v2.site.SOURCE_GEOMETRY),
            (registry.PHASES, registry.PATCH_HEIGHT, registry.PATCH_WIDTH),
        )

    def test_launch_authority_is_three_way_consistent_and_authorized(self):
        prereg = registry.load_preregistration()["claims"]
        contract = runner.probe_contract()
        template = runner.remote_launch_template()
        self.assertTrue(runner.GPU_LAUNCH_AUTHORIZED)
        self.assertFalse(runner.LAUNCH_BLOCKED_PENDING_INDEPENDENT_AUDIT)
        for row in (prereg, contract, template):
            self.assertTrue(row["gpu_launch_authorized"])
            self.assertFalse(row["launch_blocked_pending_independent_audit"])
        completion = registry.load_preregistration()["receipt_completion_authority"]
        self.assertEqual(contract["receipt_completion_authority"], completion)
        self.assertEqual(template["receipt_completion_authority"], completion)
        self.assertFalse(
            completion["candidate_file_presence_is_completion_authority"]
        )
        self.assertTrue(completion["candidate_may_exist_after_nonzero_exit"])
        self.assertTrue(completion["external_completion_seal_required"])
        self.assertFalse(completion["external_completion_seal_written_by_probe"])

    def test_run_mode_reaches_world4_boundary_without_running_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "absent.json"
            with mock.patch.object(runner, "_initialize_world4") as initialize, mock.patch.object(
                runner, "run_real_world4_probe", return_value={}
            ) as run:
                self.assertEqual(
                    runner.main(["--run", "--output", str(output.resolve())]),
                    0,
                )
                initialize.assert_called_once_with()
                run.assert_called_once_with(output.resolve())

    def test_contract_forbids_role_locator_and_representation_claim(self):
        row = runner.probe_contract()
        self.assertFalse(row["caption_token_offsets_computed"])
        self.assertFalse(row["caption_role_partition_computed"])
        self.assertFalse(row["text_key_or_value_observed"])
        self.assertFalse(row["fixed_semantic_role_inventory_used"])
        self.assertTrue(row["representation_admission_hard_false"])
        self.assertFalse(row["stable_transferable_action_representation_claimed"])

    def test_contract_discloses_source_bootstrap_then_scrub(self):
        row = runner.probe_contract()
        self.assertTrue(row["site_source_bootstrap_tensors_created"])
        self.assertTrue(row["site_source_bootstrap_tensors_scrubbed_before_trajectory"])
        self.assertFalse(row["source_bootstrap_tensor_consumed_by_probe_forward"])

    def test_source_manifest_is_closed_and_hashes_match(self):
        manifest = runner.source_manifest()
        self.assertEqual(manifest["file_count"], 26)
        self.assertEqual(len(manifest["files"]), 26)
        self.assertEqual(
            manifest["digest"],
            registry.object_sha256(
                {
                    "files": manifest["files"],
                    "file_count": manifest["file_count"],
                    "all_plain_nonsymlink_files": True,
                }
            ),
        )
        for row in manifest["files"]:
            if "canonical_path" in row:
                candidates = [Path(row["canonical_path"])]
            else:
                candidates = list(METHOD_ROOT.rglob(row["file"]))
                candidates.extend(
                    Path(root) / row["file"]
                    for root in sys.path
                    if root and (Path(root) / row["file"]).is_file()
                )
            self.assertTrue(candidates, row["file"])
            payload = candidates[0].read_bytes()
            import hashlib

            self.assertEqual(hashlib.sha256(payload).hexdigest(), row["sha256"])
        expected_python = {
            "auh_self_generated_anonymous_object_same_state_probe_v6.py",
            "self_generated_anonymous_object_observer_v6.py",
            "self_generated_anonymous_object_registry_v6.py",
            "anonymous_visual_projection_hook_v6.py",
            "auh_self_generated_relational_t2v_trajectory_probe_v2.py",
            "anchor_sga_anc_controller.py",
            "auh_native_relational_attention_parity_smoke_v1.py",
            "differential_sampler.py",
            "auh_source_owned_role_locator_v15_adapter.py",
            "source_owned_role_locator_v15.py",
            "anchor_cross_attention_transport.py",
            "anchor_qk_transport.py",
            "guided_source_aligned_controller.py",
            "infer_native_self_generated_intermediate_anchor_canary_v1.py",
            "infer_native_self_generated_relational_graph_observer_v1.py",
            "native_relational_attention_hook_v1.py",
            "self_generated_intermediate_action_anchor_v1.py",
            "self_generated_relational_action_graph_observer_v1.py",
            "self_generated_relational_t2v_probe_registry_v2.py",
            "source_aligned_controller.py",
            "source_kv_replay.py",
            "source_owned_role_locator_v15b_e00_asset.py",
            "tri_branch_unipc.py",
        }
        self.assertEqual(
            {row["file"] for row in manifest["files"] if row["file"].endswith(".py")}
            - {"transformer_wan.py"},
            expected_python,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.py"
            target.write_text("pass\n", encoding="utf-8")
            link = root / "link.py"
            link.symlink_to(target)
            with self.assertRaisesRegex(
                runner.AUHAnonymousObjectProbeV6Error, "plain file"
            ):
                runner._source_row("symlink_negative", link)
        tests = runner.test_source_manifest()
        self.assertEqual(tests["file_count"], 4)
        self.assertEqual(len(tests["files"]), 4)
        self.assertEqual(tests["expected_unittest_case_count"], 48)
        self.assertFalse(tests["execution_claimed_by_gpu_receipt"])
        self.assertEqual(
            tests["digest"],
            registry.object_sha256(
                {
                    "files": tests["files"],
                    "file_count": 4,
                    "expected_unittest_case_count": 48,
                    "all_plain_nonsymlink_files": True,
                    "execution_claimed_by_gpu_receipt": False,
                }
            ),
        )
        self.assertEqual(
            {row["file"] for row in tests["files"]},
            {
                "test_self_generated_anonymous_object_registry_v6.py",
                "test_anonymous_visual_projection_hook_v6.py",
                "test_self_generated_anonymous_object_observer_v6.py",
                "test_auh_self_generated_anonymous_object_same_state_probe_v6.py",
            },
        )
        import hashlib

        for row in tests["files"]:
            path = METHOD_ROOT / "tests" / row["file"]
            self.assertTrue(path.is_file())
            self.assertFalse(path.is_symlink())
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row["sha256"])
        self.assertEqual(runner.probe_contract()["test_source_manifest"], tests)

    def test_runner_uses_validated_v4_trajectory_branch_and_sampler(self):
        source = Path(runner.__file__).read_text(encoding="utf-8")
        self.assertIn('branch="anchor_action_trajectory"', source)
        self.assertNotIn('branch="anonymous_v6_action_trajectory"', source)
        self.assertIn("import differential_sampler as cdf", source)
        self.assertNotIn("differential_cdf_sampler", source)

    def test_embedding_control_validator_accepts_exact_cycle_and_neutral(self):
        rows = {}
        for index, appearance in enumerate(registry.APPEARANCES):
            rows[f"{appearance.appearance_id}:action"] = f"action-{index}"
            rows[f"{appearance.appearance_id}:neutral"] = "neutral-one"
        for index, appearance in enumerate(registry.APPEARANCES):
            rows[f"{appearance.appearance_id}:source_swap"] = f"action-{(index + 1) % 3}"
        checks, neutral = runner._validate_anonymous_embedding_controls(rows)
        self.assertEqual(len(checks), 3)
        self.assertTrue(all(row["equal"] for row in checks))
        self.assertEqual(neutral, "neutral-one")

    def test_embedding_control_validator_rejects_mismatch(self):
        rows = {}
        for index, appearance in enumerate(registry.APPEARANCES):
            rows[f"{appearance.appearance_id}:action"] = f"action-{index}"
            rows[f"{appearance.appearance_id}:source_swap"] = "wrong"
            rows[f"{appearance.appearance_id}:neutral"] = "neutral-one"
        with self.assertRaises(runner.AUHAnonymousObjectProbeV6Error):
            runner._validate_anonymous_embedding_controls(rows)

    def test_fake_tokenizer_encoder_binds_source_swap_and_neutral_embeddings(self):
        class Legacy:
            @staticmethod
            def _tokenize_training_prompt(_tokenizer, text):
                digest = sum(text.encode("utf-8")) % 251
                ids = torch.tensor([[digest, 1, 0, 0]], dtype=torch.long)
                mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)
                return ids, mask

            @staticmethod
            def _tokenize_renderer_negative(_tokenizer, text):
                return Legacy._tokenize_training_prompt(_tokenizer, text)

        class Encoder:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

            def to(self, _device):
                return self

        class Model:
            t5_text_encoder = None

            @staticmethod
            def encode_prompt(ids, mask):
                value = torch.zeros((1, 4, 8), dtype=torch.bfloat16)
                value[0, 0, 0] = ids[0, 0].to(torch.bfloat16)
                value[0, 0, 1] = mask.sum().to(torch.bfloat16)
                return value

        class EventRuntime:
            @staticmethod
            def _retire_t5_text_encoder(model, torch_module):
                model.t5_text_encoder = None

        runtime = types.SimpleNamespace(
            _legacy=Legacy(),
            _tokenizer=object(),
            model=Model(),
            device=torch.device("cpu"),
            _event_runtime=EventRuntime(),
        )
        bernini_cli = types.ModuleType("bernini.cli")
        bernini_cli.DEFAULT_NEG_PROMPT = "negative"
        transformers = types.ModuleType("transformers")
        transformers.UMT5EncoderModel = Encoder
        with (
            mock.patch.dict(
                sys.modules,
                {"bernini.cli": bernini_cli, "transformers": transformers},
            ),
            mock.patch.object(runner, "TEXT_LENGTH", 4),
            mock.patch.object(runner, "TEXT_WIDTH", 8),
            mock.patch.object(runner, "_model_prompt", side_effect=lambda text: text),
            mock.patch.object(runner.dist, "broadcast_object_list", side_effect=lambda value, src: None),
            mock.patch.object(runner.dist, "barrier", side_effect=lambda: None),
            mock.patch.object(runner.dist, "broadcast", side_effect=lambda value, src: None),
            mock.patch.object(runner, "_require_all_rank_equal", side_effect=lambda *a, **k: None),
        ):
            bank, negative, receipt = runner._encode_anonymous_prompt_bank(runtime, rank=0)
        self.assertEqual(tuple(negative.shape), (1, 4, 8))
        self.assertTrue(receipt["source_swap_embedding_equals_next_action_embedding"])
        self.assertEqual(receipt["identical_neutral_caption_embedding_unique_count"], 1)
        for index, appearance in enumerate(registry.APPEARANCES):
            next_appearance = registry.APPEARANCES[(index + 1) % 3]
            self.assertTrue(
                torch.equal(
                    bank[appearance.appearance_id]["source_swap"],
                    bank[next_appearance.appearance_id]["action"],
                )
            )


if __name__ == "__main__":
    unittest.main()
