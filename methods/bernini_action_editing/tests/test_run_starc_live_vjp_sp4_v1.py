from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(METHOD_ROOT))

import run_starc_live_vjp_sp4_v1 as runtime  # noqa: E402
import starch_live_vjp_bridge_v1 as bridge  # noqa: E402


class STARCLiveVJPSP4StaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = METHOD_ROOT / "run_starc_live_vjp_sp4_v1.py"
        cls.source = cls.path.read_text(encoding="utf-8")

    def test_loader_archive_member_and_runtime_pins_match_bridge(self) -> None:
        self.assertEqual(
            runtime.LOADER_SOURCE_ARCHIVE_MEMBER,
            "methods/bernini_action_editing/run_starc_live_vjp_sp4_v1.py",
        )
        self.assertEqual(runtime.EXPECTED_WORLD_SIZE, 4)
        self.assertIn("live_bridge.BERNINI_OFFICIAL_COMMIT", self.source)
        self.assertIn("live_bridge.VEOMNI_TESTED_COMMIT", self.source)
        self.assertIn("live_bridge.BERNINI_CHECKPOINT_TREE_SHA256", self.source)
        self.assertIn(
            "live_bridge.BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256", self.source
        )
        self.assertIn("expected_loader_source_sha256", self.source)
        self.assertIn("file_sha256(loader)", self.source)

    def test_real_live_proof_is_written_in_process_not_composed_from_json(self) -> None:
        body = inspect.getsource(runtime.run_one_sp4)
        prove = body.index("prove_current_clean_latent_vjp(")
        write = body.index("write_authenticated_composite_receipt(")
        self.assertLess(prove, write)
        self.assertIn("args.output,\n                    proof,", body)
        self.assertNotIn("mechanism_receipt", body)
        self.assertNotIn("json.load", body)

    def test_cpu_authoring_entry_seals_the_exact_runtime_candidate_schema(self) -> None:
        choices = runtime.build_parser()._subparsers._group_actions[0].choices
        self.assertIn("author-candidate", choices)
        body = inspect.getsource(runtime.author_candidate_manifest)
        for token in (
            "live_bridge.CANDIDATE_BINDING_SCHEMA",
            '"source_video_sha256": file_sha256(source)',
            '"instruction_sha256": instruction_digest',
            '"current_clean_latent_tensor_sha256": clean_digest',
            '"external_inference_inputs": list(live_bridge.EXTERNAL_INFERENCE_INPUTS)',
            '"auxiliary_spatial_inputs": []',
            "live_bridge.authenticate_current_candidate_manifest(",
            "os.chmod(output, 0o400)",
        ):
            self.assertIn(token, body)

    def test_official_frozen_bernini_veomni_checkpoint_and_critic_are_loaded(self) -> None:
        for token in (
            "legacy.trainer.validate_source_trees(",
            "legacy.trainer.validate_checkpoint(",
            "BerniniRendererConfig.from_pretrained(",
            "BerniniRendererModel(config).requires_grad_(False).eval().to(device)",
            "AutoTokenizer.from_pretrained(",
            "materializer._encode_prompt_pair(",
            "FrozenHiddenTemporalEventCritic(",
            "verify_frozen_starc_critic_artifact(",
            "authenticate_frozen_bernini_checkpoint_content(",
        ):
            self.assertIn(token, self.source)

    def test_real_world4_ulysses_and_rank0_only_create_only_write(self) -> None:
        for token in (
            'backend="nccl"',
            "init_parallel_state(ulysses_size=EXPECTED_WORLD_SIZE)",
            "distributed.world_size != EXPECTED_WORLD_SIZE",
            "if distributed.rank == 0:",
            "dist.broadcast_object_list(write_rows, src=0)",
            "dist.destroy_process_group()",
        ):
            self.assertIn(token, self.source)

    def test_only_public_source_instruction_and_internal_noop_noise_are_accepted(self) -> None:
        parser_actions = {
            action.dest for action in runtime.build_parser()._subparsers._group_actions[0].choices["run"]._actions
        }
        for required in (
            "source_video",
            "instruction_file",
            "noop_caption_file",
            "current_clean_latent",
            "native_noise",
            "candidate_manifest",
        ):
            self.assertIn(required, parser_actions)
        for forbidden in bridge.FORBIDDEN_AUXILIARY_INPUTS:
            self.assertNotIn(forbidden, parser_actions)
        self.assertNotIn("target_video", parser_actions)
        self.assertNotIn("adapter", parser_actions)

    def test_no_editor_update_or_claim_authority_surface(self) -> None:
        self.assertNotIn("optimizer.step", self.source)
        self.assertNotIn("loss.backward", self.source)
        self.assertNotIn("load_adapter", self.source)
        for token in (
            "--ack-mechanism-probe-only",
            "--ack-no-editor-parameter-or-update",
            "--ack-no-scientific-or-action-editing-claim",
            '"editor_parameter_or_update_authorized": False',
            '"scientific_or_action_editing_claim_authorized": False',
        ):
            self.assertIn(token, self.source)

    def test_canonical_text_loader_hashes_exact_bytes_and_rejects_outer_space(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root).resolve()
            good = root / "instruction.txt"
            good.write_text("the dog turns its head right", encoding="utf-8")
            digest = hashlib.sha256(good.read_bytes()).hexdigest()
            text, path, text_digest = runtime.load_canonical_text_file(
                good, digest, label="instruction"
            )
            self.assertEqual(text, "the dog turns its head right")
            self.assertEqual(path, good)
            self.assertEqual(text_digest, digest)

            bad = root / "bad.txt"
            bad.write_text(" trailing space ", encoding="utf-8")
            with self.assertRaisesRegex(
                runtime.STARCLiveVJPRuntimeError, "canonical text"
            ):
                runtime.load_canonical_text_file(
                    bad,
                    hashlib.sha256(bad.read_bytes()).hexdigest(),
                    label="instruction",
                )

    def test_output_is_fresh_and_plain(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root).resolve()
            output = root / "receipt.json"
            self.assertEqual(runtime._fresh_output(output), output)
            output.write_text("{}", encoding="ascii")
            with self.assertRaisesRegex(
                runtime.STARCLiveVJPRuntimeError, "fresh absolute"
            ):
                runtime._fresh_output(output)


if __name__ == "__main__":
    unittest.main()
