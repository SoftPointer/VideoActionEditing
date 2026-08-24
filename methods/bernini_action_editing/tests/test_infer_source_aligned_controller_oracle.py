from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_source_aligned_controller_oracle as oracle  # noqa: E402


LOCKED_ARM_TABLE_SHA256 = (
    "f1d7f6f8c9c33e8e8f2eacc9f9d99e03deb7cc0bf2ad749edaabf79c723d5851"
)


def _args(arm: str = "K1", **overrides):
    values = {
        "bernini_root": "/bernini",
        "veomni_root": "/veomni",
        "checkpoint": "/checkpoint",
        "checkpoint_content_manifest": "/manifest",
        "source_video": "/scratch/source.mp4",
        "original_source_path": oracle.EXPECTED_ORIGINAL_SOURCE_PATH,
        "expected_source_sha256": oracle.EXPECTED_SOURCE_SHA256,
        "instruction": oracle.EXPECTED_INSTRUCTION,
        "output": "/output/result.mp4",
        "arm": arm,
        "expected_bernini_commit": oracle.BERNINI_COMMIT,
        "expected_veomni_commit": oracle.VEOMNI_COMMIT,
        "expected_checkpoint_tree_sha256": oracle.CHECKPOINT_TREE_SHA256,
        "method_source_revision": "1" * 40,
        "method_source_archive": "/scratch/method.tar",
        "method_source_archive_sha256": "2" * 64,
        "method_source_tree_sha256": "3" * 64,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _trace(arm: str):
    if arm == "C0":
        return {
            "identity_bypassed": True,
            "sigmas": [],
            "candidate_counts": [],
            "anc_retained_variance": [],
            "anc_nominal_correlation": [],
            "sga_scores": [],
            "sga_weights": [],
            "delta_rms": [],
            "update_rms": [],
            "noise_state_change_rms": [],
            "fresh_noise_draws": 0,
        }
    counts = [5, 5, 5] + [1] * 37 if arm == "SGA5" else [1] * 40
    scores = [[0.1] * 5 if count == 5 else [] for count in counts]
    weights = [[0.2] * 5 if count == 5 else [1.0] for count in counts]
    sigmas = [1.0 - index / 40.0 for index in range(41)]
    retention = [
        0.0
        if sigma >= 1.0
        else 1.0
        if sigma <= oracle.ANC_LOCK_SIGMA
        else (1.0 - sigma) / (1.0 - oracle.ANC_LOCK_SIGMA)
        for sigma in sigmas[:-1]
    ]
    return {
        "identity_bypassed": False,
        "sigmas": sigmas,
        "candidate_counts": counts,
        "anc_retained_variance": retention,
        "anc_nominal_correlation": [item**0.5 for item in retention],
        "sga_scores": scores,
        "sga_weights": weights,
        "delta_rms": [0.5] * 40,
        "update_rms": [0.5 * (sigmas[index] - sigmas[index + 1]) for index in range(40)],
        "noise_state_change_rms": [0.2] * 40,
        "fresh_noise_draws": 52 if arm == "SGA5" else 40,
    }


def _identity(content: str, label: str):
    return {
        "shape": [1, 16, 21, 62, 60],
        "dtype": "torch.float32",
        "numel": 1_249_920,
        "byte_count": 4_999_680,
        "content_sha256": content,
        "raw_storage_sha256": content,
        "finite": True,
        "label": label,
    }


def _rows(arm: str):
    spec = oracle.arm_spec(arm)
    trace = _trace(arm)
    validation = oracle.validate_trace(
        trace, spec=spec, shared_step_calls=spec.expected_shared_step_calls
    )
    source = _identity("a" * 64, "source latent")
    generated = _identity(
        "a" * 64 if arm == "C0" else "b" * 64, "generated latent"
    )
    freeze = {
        "base_frozen": True,
        "trainable_parameter_tensors": 0,
        "trainable_parameter_elements": 0,
        "lora_module_count": 0,
        "adapter_modules_absent": True,
    }
    return [
        {
            "rank": rank,
            "local_rank": rank,
            "ulysses_size": 4,
            "arm": arm,
            "source_video_sha256": oracle.EXPECTED_SOURCE_SHA256,
            "source_latent": source,
            "action_prompt_embeddings": _identity("c" * 64, "action"),
            "noop_prompt_embeddings": _identity("d" * 64, "noop"),
            "generated_latent": generated,
            "identity_object_reused": arm == "C0",
            "trace": trace,
            "trace_validation": validation,
            "freeze_before": freeze,
            "freeze_after": freeze,
            "shared_step_audit_restored": True,
            "method_manifest_digest": "e" * 64,
        }
        for rank in range(4)
    ]


class ArmAndCliContractTests(unittest.TestCase):
    def test_exact_three_arm_registry_and_locked_digest(self):
        self.assertEqual(oracle.ARM_NAMES, ("C0", "K1", "SGA5"))
        self.assertEqual(oracle.ARM_TABLE_SHA256, LOCKED_ARM_TABLE_SHA256)
        c0, k1, sga5 = (oracle.arm_spec(name) for name in oracle.ARM_NAMES)
        self.assertEqual((c0.motion_scale, c0.sga_steps), (0.0, 0))
        self.assertEqual((k1.motion_scale, k1.sga_steps), (1.0, 0))
        self.assertEqual((sga5.motion_scale, sga5.sga_steps), (1.0, 3))
        self.assertEqual(k1.configured_sga_candidates, 5)
        self.assertIn("one_anc_candidate", k1.effective_candidate_policy)
        self.assertEqual((k1.expected_shared_step_calls, k1.expected_fresh_noise_draws), (80, 40))
        self.assertEqual((sga5.expected_shared_step_calls, sga5.expected_fresh_noise_draws), (104, 52))

    def test_cli_exposes_paths_provenance_and_arm_but_no_hyperparameters(self):
        parser = oracle.build_parser()
        options = {
            option
            for action in parser._actions
            for option in action.option_strings
        }
        self.assertIn("--arm", options)
        for forbidden in (
            "--motion-scale",
            "--motion-strength",
            "--sga-steps",
            "--sga-candidates",
            "--sga-temperature",
            "--anc-lock-sigma",
            "--num-inference-steps",
            "--seed",
            "--target-video",
            "--mask",
            "--flow",
            "--pose",
            "--track",
            "--trajectory",
            "--first-frame-anchor",
            "--adapter",
            "--adapter-checkpoint",
        ):
            self.assertNotIn(forbidden, options)

    def test_cli_is_exact_canonical_dog_contract(self):
        for arm in oracle.ARM_NAMES:
            self.assertEqual(oracle.validate_cli(_args(arm)).arm, arm)
        for name, value in (
            ("instruction", "Make the cat jump."),
            ("original_source_path", "/other/source.mp4"),
            ("expected_source_sha256", "0" * 64),
            ("expected_bernini_commit", "0" * 40),
            ("expected_veomni_commit", "0" * 40),
            ("expected_checkpoint_tree_sha256", "0" * 64),
            ("method_source_revision", "short"),
            ("method_source_archive_sha256", "bad"),
            ("method_source_tree_sha256", "bad"),
        ):
            with self.subTest(name=name), self.assertRaises(
                oracle.SourceAlignedInferenceError
            ):
                oracle.validate_cli(_args(**{name: value}))


class TraceAndRankCertificateTests(unittest.TestCase):
    def test_each_arm_trace_is_exact(self):
        for arm in oracle.ARM_NAMES:
            spec = oracle.arm_spec(arm)
            result = oracle.validate_trace(
                _trace(arm),
                spec=spec,
                shared_step_calls=spec.expected_shared_step_calls,
            )
            self.assertTrue(result["validated"])
            expected = [] if arm == "C0" else (
                [5, 5, 5] + [1] * 37 if arm == "SGA5" else [1] * 40
            )
            self.assertEqual(result["effective_candidate_counts"], expected)

    def test_trace_rejects_call_draw_candidate_and_weight_drift(self):
        spec = oracle.arm_spec("SGA5")
        cases = []
        changed = _trace("SGA5")
        changed["candidate_counts"][0] = 4
        cases.append((changed, spec.expected_shared_step_calls))
        changed = _trace("SGA5")
        changed["fresh_noise_draws"] = 51
        cases.append((changed, spec.expected_shared_step_calls))
        changed = _trace("SGA5")
        changed["sga_weights"][0] = [0.1] * 5
        cases.append((changed, spec.expected_shared_step_calls))
        cases.append((_trace("SGA5"), 103))
        for trace, calls in cases:
            with self.subTest(calls=calls), self.assertRaises(
                oracle.SourceAlignedInferenceError
            ):
                oracle.validate_trace(trace, spec=spec, shared_step_calls=calls)

    def test_four_rank_latent_prompt_and_trace_agreement(self):
        for arm in oracle.ARM_NAMES:
            result = oracle.validate_four_rank_runtime(
                _rows(arm), spec=oracle.arm_spec(arm)
            )
            self.assertTrue(result["all_four_ranks_exact"])
            self.assertTrue(result["all_rank_generated_latent_exact"])
            self.assertEqual(result["c0_source_latent_byte_exact"], arm == "C0")

    def test_four_rank_disagreement_fails_closed(self):
        rows = _rows("K1")
        rows[2] = dict(rows[2])
        rows[2]["trace"] = dict(rows[2]["trace"])
        rows[2]["trace"]["update_rms"] = [0.2] * 40
        with self.assertRaises(oracle.SourceAlignedInferenceError):
            oracle.validate_four_rank_runtime(rows, spec=oracle.arm_spec("K1"))


class AtomicArtifactTests(unittest.TestCase):
    def test_video_publish_is_no_overwrite_and_cleans_temporary(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(
            os.environ, {"BERNINI_OUTPUT_TRANSACTION_ID": "unit-video"}
        ):
            output = Path(root) / "result.mp4"

            def writer(decoded, path, fps):
                self.assertEqual(decoded, "decoded")
                self.assertEqual(fps, 25)
                Path(path).write_bytes(b"mp4")

            oracle.save_video_atomically(
                "decoded", output, save_output_fn=writer
            )
            self.assertEqual(output.read_bytes(), b"mp4")
            self.assertFalse((Path(root) / ".result.tmp-unit-video.mp4").exists())
            with self.assertRaises(FileExistsError):
                oracle.save_video_atomically(
                    "decoded", output, save_output_fn=writer
                )
            self.assertEqual(output.read_bytes(), b"mp4")

    def test_encoder_failure_removes_partial_temporary(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(
            os.environ, {"BERNINI_OUTPUT_TRANSACTION_ID": "unit-fail"}
        ):
            output = Path(root) / "result.mp4"

            def broken(decoded, path, fps):
                Path(path).write_bytes(b"partial")
                raise RuntimeError("encoder failed")

            with self.assertRaises(RuntimeError):
                oracle.save_video_atomically(
                    object(), output, save_output_fn=broken
                )
            self.assertFalse(output.exists())
            self.assertFalse((Path(root) / ".result.tmp-unit-fail.mp4").exists())

    def test_receipt_is_canonical_no_overwrite_and_temp_clean(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(
            os.environ, {"BERNINI_OUTPUT_TRANSACTION_ID": "unit-receipt"}
        ):
            path = Path(root) / "result.mp4.receipt.json"
            oracle.write_receipt_atomically(path, {"b": 2, "a": 1})
            self.assertEqual(path.read_bytes(), b'{"a":1,"b":2}\n')
            self.assertFalse(
                (Path(root) / ".result.mp4.receipt.json.tmp-unit-receipt").exists()
            )
            with self.assertRaises(FileExistsError):
                oracle.write_receipt_atomically(path, {"new": True})
            self.assertEqual(path.read_bytes(), b'{"a":1,"b":2}\n')


class ProvenanceAndHookTests(unittest.TestCase):
    def test_method_tree_manifest_rejects_writable_and_bytecode(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "source.py"
            path.write_text("value = 1\n")
            with self.assertRaises(oracle.SourceAlignedInferenceError):
                oracle.method_tree_manifest(Path(root))
            path.chmod(0o444)
            value = oracle.method_tree_manifest(Path(root))
            self.assertEqual(value["file_count"], 1)
            self.assertTrue(value["all_plain_read_only"])
            cache = Path(root) / "__pycache__"
            cache.mkdir()
            with self.assertRaises(oracle.SourceAlignedInferenceError):
                oracle.method_tree_manifest(Path(root))

    def test_checkpoint_manifest_hashes_every_plain_file(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint = Path(root) / "checkpoint"
            checkpoint.mkdir()
            first = checkpoint / "a.bin"
            second = checkpoint / "nested" / "b.json"
            second.parent.mkdir()
            first.write_bytes(b"a")
            second.write_bytes(b"b")
            lines = [
                f"{hashlib.sha256(first.read_bytes()).hexdigest()}  ./a.bin",
                f"{hashlib.sha256(second.read_bytes()).hexdigest()}  ./nested/b.json",
            ]
            manifest = Path(root) / "manifest.sha256"
            manifest.write_text("\n".join(lines) + "\n")
            digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
            with mock.patch.object(
                oracle, "CHECKPOINT_CONTENT_MANIFEST_SHA256", digest
            ), mock.patch.object(oracle, "CHECKPOINT_CONTENT_FILE_COUNT", 2):
                identity = oracle.validate_checkpoint_content(checkpoint, manifest)
            self.assertTrue(identity["every_file_sha256_verified"])
            self.assertEqual(identity["verified_file_count"], 2)

    def test_shared_step_audit_counts_and_restores_exact_method(self):
        class Diffusion:
            def shared_step(self, **kwargs):
                return kwargs["value"]

        diffusion = Diffusion()
        fake_controller = SimpleNamespace(
            cdf=SimpleNamespace(resolve_diffusion_core=lambda renderer: diffusion)
        )
        with mock.patch.object(oracle, "controller", fake_controller):
            audit = oracle.SharedStepAudit(object())
            with audit:
                self.assertEqual(
                    diffusion.shared_step(model_id="transformer_1", value=7), 7
                )
            self.assertEqual(audit.calls, 1)
            self.assertTrue(audit.restored)
            self.assertNotIn("shared_step", vars(diffusion))

    def test_receipt_binds_frozen_source_only_runtime_and_self_hash(self):
        spec = oracle.arm_spec("K1")
        method = {
            "revision": "1" * 40,
            "archive_sha256": "2" * 64,
            "runtime_tree": {
                "tree_sha256": "3" * 64,
                "all_plain_read_only": True,
                "bytecode_absent": True,
            },
            "bytecode_policy": {"dont_write_bytecode": True},
        }
        runtime = oracle.validate_four_rank_runtime(
            _rows("K1"), spec=spec
        )
        fake_controller = SimpleNamespace(
            controller_contract=lambda: {
                "status": "dynaedit_inspired_bernini_adaptation_not_official_reproduction",
                "user_inputs": ["source_video", "edit_instruction"],
            }
        )
        with mock.patch.object(oracle, "controller", fake_controller):
            receipt = oracle.build_receipt(
                args=_args("K1"),
                spec=spec,
                source_path=Path("/scratch/source.mp4"),
                source_sha256=oracle.EXPECTED_SOURCE_SHA256,
                source_metadata={"source_derived_bucket_hw": [496, 480]},
                checkpoint_identity={"every_file_sha256_verified": True},
                method_pre=method,
                method_post=method,
                bernini_revision=oracle.BERNINI_COMMIT,
                veomni_revision=oracle.VEOMNI_COMMIT,
                bernini_training_files={"training.py": "4" * 64},
                bernini_inference_files={"inference.py": "5" * 64},
                action_prompt_sha256="7" * 64,
                noop_prompt_sha256="8" * 64,
                runtime=runtime,
                runtime_versions={"torch": "test"},
                output_path=Path("/output/result.mp4"),
                output_sha256="6" * 64,
            )
        candidate = dict(receipt)
        stored = candidate.pop("receipt_digest")
        self.assertEqual(stored, oracle.object_sha256(candidate))
        self.assertNotIn("adapter", receipt)
        self.assertEqual(
            receipt["input"]["accepted_external_conditions"],
            ["source_video", "edit_instruction"],
        )
        self.assertTrue(
            receipt["sampling"]["runtime_execution_certificate"][
                "all_rank_controller_trace_exact"
            ]
        )


class StaticEntrypointAndLauncherTests(unittest.TestCase):
    def test_runner_ast_calls_controller_without_forbidden_conditions(self):
        path = METHOD_ROOT / "infer_source_aligned_controller_oracle.py"
        source = path.read_text()
        tree = ast.parse(source)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        controller_calls = [
            node
            for node in calls
            if isinstance(node.func, ast.Attribute)
            and node.func.attr == "sample_source_aligned_controller"
        ]
        self.assertEqual(len(controller_calls), 1)
        keywords = {item.arg for item in controller_calls[0].keywords}
        self.assertEqual(
            keywords,
            {
                "source_latent",
                "source_rgb_frames",
                "action_prompt_embeds",
                "noop_prompt_embeds",
                "config",
                "return_trace",
            },
        )
        self.assertNotIn("model.sample(", source)
        self.assertNotIn("import peft", source)
        self.assertIn("os.link(temporary, output_path)", source)
        self.assertIn("os.link(temporary, path)", source)

    def test_launcher_is_fixed_four_gpu_fail_closed_contract(self):
        path = (
            METHOD_ROOT
            / "scripts"
            / "auh_infer_source_aligned_controller_oracle.sbatch"
        )
        source = path.read_text()
        completed = subprocess.run(
            ["bash", "-n", str(path)], capture_output=True, text=True, check=False
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for required in (
            "#SBATCH --gres=gpu:mi210:4",
            "--nproc_per_node=4",
            'source_video="/vast/users/guangyi.chen/dataset/goku/',
            'instruction="Make the dog pick up the bone and hold it in its mouth."',
            "expected_source_sha256=\"5ed911f6",
            "--checkpoint-content-manifest",
            "--method-source-archive",
            "--method-source-tree-sha256",
            "git -C \"${source_repository}\" archive --format=tar",
            "validate_checkpoint_content \"${checkpoint_manifest}\"",
            "method_tree_digest_post",
            "pinned_source_identity_post",
            "PYTHONDONTWRITEBYTECODE=1",
            "PYTHONPYCACHEPREFIX",
            "trap cleanup EXIT",
            "rm -f -- \"${output_temporary}\" \"${receipt_temporary}\"",
            "artifact_committed=1",
            "frames=81 fps=25 steps=40 seed=2027 shift=5 Ulysses=4",
            "target=false mask=false flow=false pose=false track=false trajectory=false first_frame_anchor=false adapter=false",
        ):
            self.assertIn(required, source)
        for forbidden in (
            "BERNINI_ACTION_INFERENCE_STEPS",
            "BERNINI_ACTION_INFERENCE_SEED",
            "BERNINI_CDF_MOTION_STRENGTH",
            "BERNINI_SGA_STEPS",
            "BERNINI_SGA_CANDIDATES",
            "--target-video",
            "--adapter-checkpoint",
        ):
            self.assertNotIn(forbidden, source)
        self.assertRegex(source, r"case \"\$\{arm\}\" in[\s\S]*C0\)[\s\S]*K1\)[\s\S]*SGA5\)")


if __name__ == "__main__":
    unittest.main()
