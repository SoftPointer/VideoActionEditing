from __future__ import annotations

import ast
import hashlib
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(METHOD_ROOT))

import guided_source_aligned_controller as guided  # noqa: E402
import infer_guided_source_aligned_controller_oracle as oracle  # noqa: E402


def _args(arm: str = "FANC1G", **overrides):
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
        "durable_method_source_archive": "/durable/method.tar",
        "method_source_archive_sha256": "2" * 64,
        "method_source_tree_sha256": "3" * 64,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _schedule():
    start = guided.PINNED_UNIPC_START_SIGMA
    sigmas = [start * (1.0 - index / 40.0) for index in range(41)]
    timesteps = [value * 1000.0 for value in sigmas[:-1]]
    payload = {
        "timesteps": timesteps,
        "sigmas": sigmas,
        "flow_shift": 5.0,
        "steps": 40,
    }
    return {
        **payload,
        "digest": guided.PINNED_UNIPC_SCHEDULE_DIGEST,
        "pinned_start_sigma": start,
        "first_anc_retained_variance": (1.0 - start) / 0.75,
        "first_anc_predecessor_policy": "zero_initialized_per_dynaedit_pseudocode",
        "start_sigma_claimed_exact_one": False,
        "scheduler_sigma_fp32_digest": guided.PINNED_UNIPC_SIGMA_FP32_DIGEST,
        "scheduler_sigma_dtype": "torch.float32",
        "scheduler_sigma_device": "cpu",
        "scheduler_sigma_direct_views": True,
    }


def _trace(arm: str):
    spec = oracle.arm_spec(arm)
    schedule = _schedule()
    if arm == "C0":
        return {
            "mode": "vae_identity",
            "shared_step_calls": 0,
            "fresh_noise_draws": 0,
        }, None
    if arm == "O0":
        return {
            "mode": "official_v2v_apg_unipc",
            "branch_order": ["negative", "action"],
            "branch_counts": [40, 40],
            "shared_step_calls": 80,
            "fresh_noise_draws": 0,
            "schedule_digest": schedule["digest"],
            "scheduler_sigma_fp32_digest": schedule[
                "scheduler_sigma_fp32_digest"
            ],
            "scheduler_sigma_dtype": "torch.float32",
            "scheduler_sigma_device": "cpu",
            "scheduler_sigma_direct_views": True,
        }, schedule
    counts = [5, 5, 5] + [1] * 37 if "5G" in arm else [1] * 40
    weights = [[0.2] * 5 if count == 5 else [1.0] for count in counts]
    entropy = [math.log(5.0) if count == 5 else 0.0 for count in counts]
    margins = [0.0 if count == 5 else 1.0 for count in counts]
    bank = guided.noise_bank_pairing_contract(seed=2027)
    retained = [
        0.0
        if arm == "FIID1G"
        else guided.raw_controller.anc_retained_variance(sigma)
        for sigma in schedule["sigmas"][:-1]
    ]
    trace = {
        "arm": arm,
        "sigmas": schedule["sigmas"],
        "timesteps": schedule["timesteps"],
        "schedule_digest": schedule["digest"],
        "scheduler_sigma_fp32_digest": schedule["scheduler_sigma_fp32_digest"],
        "scheduler_sigma_dtype": "torch.float32",
        "scheduler_sigma_device": "cpu",
        "scheduler_sigma_direct_views": True,
        "candidate_counts": counts,
        "anc_retained_variance": retained,
        "anc_nominal_correlation": [value**0.5 for value in retained],
        "sga_scores": [[0.0] * count if count > 1 else [] for count in counts],
        "sga_weights": weights,
        "sga_entropy": entropy,
        "sga_top1_margin": margins,
        "delta_rms": [0.1] * 40,
        "update_rms": [0.01] * 40,
        "noise_state_change_rms": [1.0] * 40,
        "fresh_noise_draws": spec.expected_fresh_noise_draws,
        "used_noise_key_digest": guided.used_noise_key_digest(
            guided.GuidedSourceAlignedConfig(arm=arm)
        ),
        "used_fresh_noise_content_digest": "6" * 64,
        "candidate0_fresh_noise_content_digest": "7" * 64,
        "full_noise_bank_digest": bank["full_bank_digest"],
        "candidate0_noise_bank_digest": bank["candidate0_bank_digest"],
        "branch_order": list(guided.BRANCH_ORDER),
        "branch_counts": [spec.expected_candidate_evaluations] * 4,
        "total_shared_step_calls": spec.expected_shared_step_calls,
        "apg_parameters": [
            ["guidance_mode", "v2v_apg"],
            ["guidance_scale", 4.0],
            ["eta", 0.5],
            ["norm_threshold", 50.0],
            ["momentum", 0.0],
        ],
        "target_branch_query_parity": True,
        "source_branch_query_parity": True,
        "raw_velocity_dtype": "torch.bfloat16",
        "guided_velocity_dtype": "torch.float32",
        "apg_clean_dtype": "torch.float32",
        "delta_dtype": "torch.float32",
        "edit_state_dtype": "torch.float32",
        "candidate_continuation": "candidate_0",
        "weighted_noise_collapse_used": False,
        "anc_initial_predecessor_policy": "zero_initialized_per_dynaedit_pseudocode",
    }
    return trace, schedule


def _identity(digest: str, label: str):
    return {
        "shape": [1, 16, 21, 62, 60],
        "dtype": "torch.float32",
        "numel": 1_249_920,
        "byte_count": 4_999_680,
        "content_sha256": digest,
        "raw_storage_sha256": digest,
        "finite": True,
        "label": label,
    }


def _rows(arm: str):
    spec = oracle.arm_spec(arm)
    trace, schedule = _trace(arm)
    validation = oracle.validate_trace(
        trace,
        spec=spec,
        shared_step_calls=spec.expected_shared_step_calls,
        schedule=schedule,
    )
    source = _identity("a" * 64, "source latent")
    generated = _identity(
        "a" * 64 if arm == "C0" else "b" * 64, "generated latent"
    )
    source_consensus = {
        "policy": "pinned_three_of_four_then_broadcast",
        "all_rank_post_broadcast_exact": True,
    }
    freeze = {"base_frozen": True, "lora_module_count": 0}
    return [
        {
            "rank": rank,
            "local_rank": rank,
            "ulysses_size": 4,
            "arm": arm,
            "source_video_sha256": oracle.EXPECTED_SOURCE_SHA256,
            "source_latent": source,
            "source_latent_consensus": source_consensus,
            "action_prompt_embeddings": _identity("c" * 64, "action"),
            "noop_prompt_embeddings": _identity("d" * 64, "noop"),
            "negative_prompt_embeddings": _identity("e" * 64, "negative"),
            "generated_latent": generated,
            "identity_object_reused": arm == "C0",
            "trace": trace,
            "trace_validation": validation,
            "schedule_identity": schedule,
            "freeze_before": freeze,
            "freeze_after": freeze,
            "shared_step_audit_restored": True,
            "method_manifest_digest": "f" * 64,
        }
        for rank in range(4)
    ]


class RegistryAndTraceTests(unittest.TestCase):
    def test_source_latent_consensus_requires_pinned_three_of_four(self):
        def row(rank, content, raw):
            identity = _identity(content, "source latent before consensus")
            identity["raw_storage_sha256"] = raw
            return {"rank": rank, "identity": identity}

        pinned_content = oracle.EXPECTED_SOURCE_LATENT_CONTENT_SHA256
        pinned_raw = oracle.EXPECTED_SOURCE_LATENT_RAW_STORAGE_SHA256
        rows = [
            row(0, "0" * 64, "1" * 64),
            row(1, pinned_content, pinned_raw),
            row(2, pinned_content, pinned_raw),
            row(3, pinned_content, pinned_raw),
        ]
        owner, certificate = oracle.select_pinned_source_latent_consensus(rows)
        self.assertEqual(owner, 1)
        self.assertEqual(certificate["matching_ranks"], [1, 2, 3])
        self.assertEqual(
            certificate["policy"], "pinned_three_of_four_then_broadcast"
        )
        rows[2] = row(2, "2" * 64, "3" * 64)
        with self.assertRaises(oracle.GuidedInferenceError):
            oracle.select_pinned_source_latent_consensus(rows)

    def test_exact_six_arm_registry_and_forward_counts(self):
        self.assertEqual(
            oracle.ARM_NAMES,
            ("C0", "O0", "FIID1G", "FANC1G", "FAVG5G", "FSGA5G"),
        )
        self.assertEqual(
            {name: oracle.arm_spec(name).expected_shared_step_calls for name in oracle.ARM_NAMES},
            {"C0": 0, "O0": 80, "FIID1G": 160, "FANC1G": 160, "FAVG5G": 208, "FSGA5G": 208},
        )

    def test_cli_has_durable_archive_and_no_privileged_or_tunable_inputs(self):
        options = {
            option
            for action in oracle.build_parser()._actions
            for option in action.option_strings
        }
        self.assertIn("--durable-method-source-archive", options)
        for forbidden in (
            "--target-video", "--mask", "--flow", "--pose", "--track",
            "--trajectory", "--first-frame-anchor", "--adapter",
            "--num-inference-steps", "--seed", "--temperature", "--tau",
            "--apg-eta", "--guidance-scale",
        ):
            self.assertNotIn(forbidden, options)
        for arm in oracle.ARM_NAMES:
            self.assertEqual(oracle.validate_cli(_args(arm)).arm, arm)

    def test_all_trace_contracts_include_actual_sigma_and_apg_evidence(self):
        for arm in oracle.ARM_NAMES:
            spec = oracle.arm_spec(arm)
            trace, schedule = _trace(arm)
            result = oracle.validate_trace(
                trace,
                spec=spec,
                shared_step_calls=spec.expected_shared_step_calls,
                schedule=schedule,
            )
            self.assertTrue(result["validated"])
        trace, schedule = _trace("FANC1G")
        self.assertGreater(trace["anc_retained_variance"][0], 0.0)
        self.assertEqual(
            trace["anc_initial_predecessor_policy"],
            "zero_initialized_per_dynaedit_pseudocode",
        )
        changed = dict(trace, guided_velocity_dtype="torch.bfloat16")
        with self.assertRaises(oracle.GuidedInferenceError):
            oracle.validate_trace(
                changed,
                spec=oracle.arm_spec("FANC1G"),
                shared_step_calls=160,
                schedule=schedule,
            )

    def test_four_rank_certificate_binds_negative_embedding(self):
        rows = _rows("FSGA5G")
        result = oracle.validate_four_rank_runtime(
            rows, spec=oracle.arm_spec("FSGA5G")
        )
        self.assertTrue(result["all_rank_negative_embedding_exact"])
        rows[2] = dict(rows[2])
        rows[2]["negative_prompt_embeddings"] = _identity("0" * 64, "negative")
        with self.assertRaises(oracle.GuidedInferenceError) as captured:
            oracle.validate_four_rank_runtime(
                rows, spec=oracle.arm_spec("FSGA5G")
            )
        self.assertIn(
            "negative_prompt_embeddings.content_sha256", str(captured.exception)
        )


class OwnedArtifactTests(unittest.TestCase):
    def test_video_and_receipt_are_no_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            video = Path(root) / "result.mp4"

            def writer(decoded, path, fps):
                self.assertEqual((decoded, fps), ("decoded", 25))
                Path(path).write_bytes(b"ours")

            identity = oracle.publish_video_owned(
                "decoded", video, save_output_fn=writer, transaction_token="unit"
            )
            self.assertEqual(video.read_bytes(), b"ours")
            self.assertEqual(identity, oracle.artifact_identity(video))
            with self.assertRaises(FileExistsError):
                oracle.publish_video_owned(
                    "decoded", video, save_output_fn=writer, transaction_token="unit2"
                )
            self.assertEqual(video.read_bytes(), b"ours")
            receipt = Path(root) / "result.mp4.receipt.json"
            receipt_identity = oracle.publish_receipt_owned(
                receipt, {"b": 2, "a": 1}, transaction_token="receipt"
            )
            self.assertEqual(receipt.read_bytes(), b'{"a":1,"b":2}\n')
            self.assertEqual(receipt_identity, oracle.artifact_identity(receipt))

    def test_concurrent_final_substitution_is_never_deleted(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "result.mp4"

            def writer(decoded, path, fps):
                Path(path).write_bytes(b"ours")

            def substitute(parent):
                output.unlink()
                output.write_bytes(b"theirs")
                raise OSError("fsync fault after substitution")

            with mock.patch.object(oracle, "_fsync_directory", side_effect=substitute):
                with self.assertRaises(OSError):
                    oracle.publish_video_owned(
                        object(), output, save_output_fn=writer,
                        transaction_token="substitution",
                    )
            self.assertEqual(output.read_bytes(), b"theirs")

    def test_receipt_substitution_is_never_deleted(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "result.receipt.json"

            def substitute(parent):
                path.unlink()
                path.write_bytes(b"theirs")
                raise OSError("fsync fault after substitution")

            with mock.patch.object(oracle, "_fsync_directory", side_effect=substitute):
                with self.assertRaises(OSError):
                    oracle.publish_receipt_owned(
                        path, {"ours": True}, transaction_token="substitution"
                    )
            self.assertEqual(path.read_bytes(), b"theirs")


class StaticAndReceiptTests(unittest.TestCase):
    def test_runner_uses_guided_source_only_api_and_pinned_negative_import(self):
        path = METHOD_ROOT / "infer_guided_source_aligned_controller_oracle.py"
        source = path.read_text()
        tree = ast.parse(source)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        guided_calls = [
            node for node in calls
            if isinstance(node.func, ast.Attribute)
            and node.func.attr == "sample_guided_source_aligned_controller"
        ]
        self.assertEqual(len(guided_calls), 1)
        self.assertEqual(
            {keyword.arg for keyword in guided_calls[0].keywords},
            {
                "source_latent", "source_rgb_frames", "action_prompt_embeds",
                "noop_prompt_embeds", "negative_prompt_embeds", "config",
                "return_trace",
            },
        )
        self.assertIn("from bernini.cli import DEFAULT_NEG_PROMPT", source)
        self.assertNotIn("bernini.models.wan.wan_diffusion import DEFAULT_NEG_PROMPT", source)
        self.assertNotIn("v1.save_video_atomically", source)
        self.assertNotIn("v1.write_receipt_atomically", source)
        self.assertLess(
            source.index("else preflight_schedule_identity("),
            source.index("with shared_step_audit:"),
        )
        self.assertIn(
            'if digest != guided.PINNED_UNIPC_SCHEDULE_DIGEST:', source
        )

    def test_receipt_is_distinct_source_only_and_durable(self):
        spec = oracle.arm_spec("FANC1G")
        method = {
            "revision": "1" * 40,
            "durable_archive_path": "/durable/method.tar",
            "archive_sha256": "2" * 64,
        }
        runtime = oracle.validate_four_rank_runtime(
            _rows("FANC1G"), spec=spec
        )
        output = {
            "path": "/output/result.mp4", "sha256": "6" * 64,
            "device": 1, "inode": 2, "size": 3,
        }
        receipt = oracle.build_receipt(
            args=_args("FANC1G"), spec=spec,
            source_path=Path("/scratch/source.mp4"),
            source_sha256=oracle.EXPECTED_SOURCE_SHA256,
            source_metadata={"source_derived_bucket_hw": [496, 480]},
            checkpoint_identity={"every_file_sha256_verified": True},
            method_pre=method, method_post=method,
            bernini_revision=oracle.BERNINI_COMMIT,
            veomni_revision=oracle.VEOMNI_COMMIT,
            bernini_training_files={"train": "4" * 64},
            bernini_inference_files={"infer": "5" * 64},
            prompt_hashes={"action": "7" * 64, "noop": "8" * 64, "negative": "9" * 64},
            runtime=runtime, runtime_versions={"torch": "test"},
            output_identity=output, transaction_token="unit",
        )
        self.assertEqual(receipt["schema_version"], oracle.RECEIPT_SCHEMA)
        self.assertEqual(
            receipt["input"]["accepted_external_conditions"],
            ["source_video", "edit_instruction"],
        )
        self.assertEqual(
            receipt["method_provenance"]["durable_archive_path"],
            "/durable/method.tar",
        )
        self.assertTrue(receipt["prompt_contract"]["negative_prompt_cleaner_applied"] is False)
        self.assertEqual(receipt["artifact_transaction"]["output"], output)
        candidate = dict(receipt)
        digest = candidate.pop("receipt_digest")
        self.assertEqual(digest, oracle.v1.object_sha256(candidate))


if __name__ == "__main__":
    unittest.main()
