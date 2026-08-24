from __future__ import annotations

from contextlib import contextmanager
import inspect
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_source_value_residual_oracle as oracle  # noqa: E402


ARM_TABLE_SHA256 = "8664401d6372632e41ecaf8623dc38f4245b60e7f3350ad2f6adb5ed3cdbd202"


def _args(arm: str = "F25", **overrides):
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
        "expected_bernini_commit": oracle.legacy.trainer.BERNINI_OFFICIAL_COMMIT,
        "expected_veomni_commit": oracle.legacy.trainer.VEOMNI_TESTED_COMMIT,
        "expected_checkpoint_tree_sha256": oracle.legacy.trainer.CHECKPOINT_TREE_SHA256,
        "method_source_revision": "1" * 40,
        "method_source_archive_sha256": "2" * 64,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _trace(rank: int = 0):
    steps = [
        {
            "generation": 0,
            "step_index": step,
            "timestep_token": f"step-{step}:float64-0x1.0p+0",
            "rank": rank,
            "ulysses_size": 4,
            "model_id": "transformer_1",
            "source_tokens_runtime": oracle.EXPECTED_SOURCE_TOKENS,
            "pair_tokens_runtime": oracle.EXPECTED_PAIR_TOKENS,
            "carrier_forwards": 1,
            "negative_replay_forwards": 1,
            "action_replay_forwards": 1,
            "cleared_after_both_replays": True,
        }
        for step in range(40)
    ]
    return {
        "sample_calls": 1,
        "step_count": 40,
        "unique_identity_count": 40,
        "steps": steps,
    }


def _metrics(gate: float):
    if gate == 0.0:
        return {
            "calls": 0,
            "accumulated_elements_including_recompute": 0,
            "all_finite": None,
            "combined_attention_output_all_finite": None,
            "projected_output_all_finite": True,
            "base_target_rms": None,
            "source_value_delta_rms": None,
            "delta_memory_rms": None,
            "gated_delta_rms": None,
            "gated_to_base_rms_ratio": None,
        }
    return {
        "calls": 80,
        "accumulated_elements_including_recompute": 1234,
        "all_finite": True,
        "combined_attention_output_all_finite": True,
        "projected_output_all_finite": True,
        "base_target_rms": 1.0,
        "source_value_delta_rms": 0.5,
        "delta_memory_rms": 0.25,
        "gated_delta_rms": 0.25 * gate,
        "gated_to_base_rms_ratio": 0.25 * gate,
    }


def _core_receipt(spec: oracle.ArmSpec):
    indices = list(spec.block_indices)
    contract = oracle.value_core.source_value_residual_contract(
        selection=spec.block_selection,
        operator=spec.operator,
        gate=spec.gate,
    )
    lookup = 0 if spec.gate == 0.0 else len(indices) * 80
    branches = (
        {}
        if spec.gate == 0.0
        else {
            "frozen_action": len(indices) * 40,
            "frozen_negative": len(indices) * 40,
        }
    )
    residual = 80 * spec.residual_varlen_calls_per_layer
    per_block = [
        {
            "block_index": index,
            "operator": spec.operator,
            "fixed_gate": spec.gate,
            "capture_calls": 40,
            "replay_calls": 80,
            "zero_gate_delegations": 80 if spec.gate == 0.0 else 0,
            "residual_varlen_calls": residual,
            "branch_counts": {
                "frozen_action": 40,
                "frozen_negative": 40,
                "frozen_noop_carrier": 40,
            },
            "execution_phase_counts": {
                "eager": 120,
                "checkpoint_forward": 0,
                "checkpoint_recompute": 0,
            },
            "last_source_tokens": oracle.EXPECTED_SOURCE_TOKENS,
            "ulysses_observed": True,
            "metrics": _metrics(spec.gate),
        }
        for index in indices
    ]
    contract["runtime"] = {
        "installed_block_count": len(indices),
        "restored": True,
        "cache": {
            "identity": None,
            "captured_blocks": [],
            "entries": [],
            "capture_calls": len(indices) * 40,
            "replay_lookups": lookup,
            "replay_branch_counts": branches,
            "retired_identity_count": 40,
            "checkpoint_context_counts": {
                "checkpoint_forward": 0,
                "checkpoint_recompute": 0,
            },
        },
        "per_block": per_block,
    }
    contract["runtime_digest"] = oracle.legacy.object_sha256(contract["runtime"])
    return contract


def _latent_identity(content: str = "a" * 64):
    return {
        "shape": [1, 16, 21, 62, 60],
        "dtype": "torch.bfloat16",
        "numel": 1249920,
        "byte_count": 2499840,
        "content_sha256": content,
        "raw_storage_sha256": "b" * 64,
        "finite": True,
        "label": "full generated latent",
    }


def _local_row(spec: oracle.ArmSpec, rank: int = 0):
    row = oracle.validate_value_runtime_certificate(
        _core_receipt(spec),
        _trace(rank),
        spec=spec,
        expected_source_tokens=oracle.EXPECTED_SOURCE_TOKENS,
        rank=rank,
        hook_restored=True,
    )
    row["generated_latent"] = _latent_identity()
    row["z0_control"] = None
    if spec.arm == "Z0":
        row["z0_control"] = {
            "byte_exact": True,
            "official_o0_latent": _latent_identity(),
            "official_o0_runtime": {
                "validated": True,
                "replay": "off",
                "rank": rank,
                "ulysses_size": 4,
            },
            "historical_o0_mp4_sha256": oracle.PINNED_HISTORICAL_O0_MP4_SHA256,
        }
    return row


def _built_receipt(spec: oracle.ArmSpec):
    rows = [_local_row(spec, rank) for rank in range(4)]
    runtime = oracle.validate_four_rank_runtime(rows, spec=spec)
    pairing = {
        "source_video_sha256": oracle.EXPECTED_SOURCE_SHA256,
    }
    pairing["causal_pairing_digest"] = oracle.legacy.object_sha256(pairing)
    return oracle.build_receipt(
        args=_args(spec.arm),
        spec=spec,
        source_path=Path("/scratch/source.mp4"),
        source_sha256=oracle.EXPECTED_SOURCE_SHA256,
        source_metadata={"source_derived_bucket_hw": [496, 480]},
        source_tokens=oracle.EXPECTED_SOURCE_TOKENS,
        output_path=Path("/output/result.mp4"),
        output_sha256="4" * 64,
        bernini_revision=oracle.legacy.trainer.BERNINI_OFFICIAL_COMMIT,
        veomni_revision=oracle.legacy.trainer.VEOMNI_TESTED_COMMIT,
        inference_file_hashes={"wan_diffusion.py": "5" * 64},
        runtime_versions={"torch": "test"},
        freeze_certificate={
            "base_frozen": True,
            "trainable_parameter_tensors": 0,
            "trainable_parameter_elements": 0,
            "lora_module_count": 0,
        },
        four_rank_runtime=runtime,
        rank0_core_receipt=_core_receipt(spec),
        pairing=pairing,
        checkpoint_content_identity={
            "every_file_sha256_verified": True,
            "verified_file_count": 23,
            "manifest_sha256_computed": "6" * 64,
            "manifest_sha256_expected": "6" * 64,
        },
    )


class ArmContractTests(unittest.TestCase):
    def test_exact_seven_arm_registry_and_locked_digest(self):
        self.assertEqual(
            oracle.ARM_NAMES,
            ("Z0", "F25", "F50", "F100", "FA25", "SN10", "CK10"),
        )
        self.assertEqual(oracle.ARM_TABLE_SHA256, ARM_TABLE_SHA256)
        expected = {
            "Z0": ("full_k_value", "late", 0.0),
            "F25": ("full_k_value", "late", 0.25),
            "F50": ("full_k_value", "late", 0.50),
            "F100": ("full_k_value", "late", 1.0),
            "FA25": ("full_k_value", "all", 0.25),
            "SN10": ("source_normalized_value", "late", 0.10),
            "CK10": ("centered_cached_kv", "late", 0.10),
        }
        self.assertEqual(
            {
                arm: (spec.operator, spec.block_selection, spec.gate)
                for arm, spec in oracle.ARM_SPECS.items()
            },
            expected,
        )

    def test_only_arm_controls_method_operator_scope_and_gate(self):
        actions = {action.dest for action in oracle.build_parser()._actions}
        self.assertIn("arm", actions)
        for forbidden in (
            "operator",
            "gate",
            "block_selection",
            "replay",
            "adapter",
            "target_video",
            "mask",
            "flow",
            "pose",
        ):
            self.assertNotIn(forbidden, actions)

    def test_cli_is_locked_to_canonical_dog(self):
        for arm in oracle.ARM_NAMES:
            self.assertEqual(oracle.validate_cli(_args(arm)).arm, arm)
        invalid = (
            _args(instruction="another action"),
            _args(original_source_path="/tmp/dog.mp4"),
            _args(expected_source_sha256="0" * 64),
        )
        for args in invalid:
            with self.assertRaises(oracle.SourceValueResidualOracleError):
                oracle.validate_cli(args)

    def test_unknown_arm_fails_closed(self):
        with self.assertRaises(oracle.SourceValueResidualOracleError):
            oracle.arm_spec("F75")


class RuntimeCertificateTests(unittest.TestCase):
    def test_every_arm_has_exact_counts(self):
        for arm in oracle.ARM_NAMES:
            with self.subTest(arm=arm):
                spec = oracle.arm_spec(arm)
                result = oracle.validate_value_runtime_certificate(
                    _core_receipt(spec),
                    _trace(),
                    spec=spec,
                    expected_source_tokens=oracle.EXPECTED_SOURCE_TOKENS,
                    rank=0,
                    hook_restored=True,
                )
                layers = len(spec.block_indices)
                self.assertEqual(result["rank_local_bank_capture_calls"], layers * 40)
                self.assertEqual(
                    result["rank_local_bank_replay_lookups"],
                    0 if arm == "Z0" else layers * 80,
                )
                self.assertEqual(
                    result["per_layer_residual_varlen_calls"],
                    160 if arm == "CK10" else 0 if arm == "Z0" else 80,
                )

    def test_zero_gate_requires_official_delegation_and_no_lookup(self):
        spec = oracle.arm_spec("Z0")
        core = _core_receipt(spec)
        core["runtime"]["per_block"][0]["zero_gate_delegations"] = 79
        with self.assertRaises(oracle.SourceValueResidualOracleError):
            oracle.validate_value_runtime_certificate(
                core,
                _trace(),
                spec=spec,
                expected_source_tokens=oracle.EXPECTED_SOURCE_TOKENS,
                rank=0,
                hook_restored=True,
            )

    def test_operator_or_runtime_tamper_fails(self):
        spec = oracle.arm_spec("F25")
        for mutate in (
            lambda core: core.__setitem__("operator", "centered_cached_kv"),
            lambda core: core["runtime"]["cache"].__setitem__("replay_lookups", 0),
            lambda core: core["runtime"]["per_block"][0]["metrics"].__setitem__("all_finite", False),
        ):
            core = _core_receipt(spec)
            mutate(core)
            with self.assertRaises(oracle.SourceValueResidualOracleError):
                oracle.validate_value_runtime_certificate(
                    core,
                    _trace(),
                    spec=spec,
                    expected_source_tokens=oracle.EXPECTED_SOURCE_TOKENS,
                    rank=0,
                    hook_restored=True,
                )

    def test_four_rank_digest_and_full_latent_equality(self):
        spec = oracle.arm_spec("F50")
        rows = [_local_row(spec, rank) for rank in range(4)]
        result = oracle.validate_four_rank_runtime(rows, spec=spec)
        self.assertTrue(result["all_rank_generated_latent_exact"])
        self.assertEqual(result["full_generated_latent_sha256"], "a" * 64)
        self.assertEqual(len(result["all_rank_certificate_digest"]), 64)
        rows[3]["generated_latent"] = _latent_identity("c" * 64)
        with self.assertRaises(oracle.SourceValueResidualOracleError):
            oracle.validate_four_rank_runtime(rows, spec=spec)

    def test_z0_requires_same_job_byte_exact_official_control(self):
        spec = oracle.arm_spec("Z0")
        rows = [_local_row(spec, rank) for rank in range(4)]
        result = oracle.validate_four_rank_runtime(rows, spec=spec)
        self.assertTrue(result["z0_same_job_official_byte_exact"])
        rows[2]["z0_control"]["byte_exact"] = False
        with self.assertRaises(oracle.SourceValueResidualOracleError):
            oracle.validate_four_rank_runtime(rows, spec=spec)

    def test_non_z0_rejects_injected_o0_control(self):
        spec = oracle.arm_spec("F25")
        rows = [_local_row(spec, rank) for rank in range(4)]
        rows[0]["z0_control"] = {"byte_exact": True}
        with self.assertRaises(oracle.SourceValueResidualOracleError):
            oracle.validate_four_rank_runtime(rows, spec=spec)


class ReuseAndReceiptTests(unittest.TestCase):
    def test_value_sample_reuses_carrier_hook_without_global_monkeypatch(self):
        spec = oracle.arm_spec("F25")
        fake_patch = SimpleNamespace(
            cache_bank=object(), receipt=lambda: {"core": True}
        )
        fake_hook = SimpleNamespace(
            trace=SimpleNamespace(as_dict=lambda: {"trace": True}), restored=True
        )
        model = SimpleNamespace(sample=mock.Mock(return_value="generated"))

        @contextmanager
        def patch_context(*args, **kwargs):
            del args, kwargs
            yield fake_patch

        @contextmanager
        def hook_context(*args, **kwargs):
            del args, kwargs
            yield fake_hook

        with mock.patch.object(
            oracle.value_core, "source_value_residual", patch_context
        ):
            with mock.patch.object(
                oracle.carrier_oracle, "source_kv_carrier_hook", hook_context
            ):
                with mock.patch.object(
                    oracle,
                    "validate_value_runtime_certificate",
                    return_value={"validated": True},
                ) as validate:
                    generated, core, certificate = oracle._sample_value_arm(
                        model,
                        spec=spec,
                        noop_prompt_embeds=object(),
                        rank=0,
                        source_tokens=oracle.EXPECTED_SOURCE_TOKENS,
                        sample_kwargs={"seed": 2027},
                    )
        self.assertEqual(generated, "generated")
        self.assertEqual(core, {"core": True})
        self.assertTrue(certificate["validated"])
        validate.assert_called_once()
        self.assertNotIn("monkeypatch", inspect.getsource(oracle._sample_value_arm).lower())

    def test_pairing_contract_does_not_accept_arm(self):
        parameters = inspect.signature(
            oracle.carrier_oracle.causal_pairing_contract
        ).parameters
        for excluded in ("arm", "operator", "gate", "block_selection"):
            self.assertNotIn(excluded, parameters)

    def test_receipt_binds_arm_core_allrank_and_source_instruction_only(self):
        spec = oracle.arm_spec("F25")
        receipt = _built_receipt(spec)
        self.assertEqual(receipt["schema_version"], oracle.RECEIPT_SCHEMA)
        self.assertEqual(receipt["arm_registry"]["selected"]["arm"], "F25")
        self.assertEqual(receipt["causal_control"]["operator"], "full_k_value")
        self.assertTrue(receipt["causal_control"]["arm_excluded_from_pairing_digest"])
        self.assertEqual(
            receipt["input"]["accepted_model_conditions"],
            ["source_video", "edit_instruction"],
        )
        self.assertFalse(receipt["input"]["target_video_argument"])
        self.assertFalse(receipt["weights"]["adapter_argument_supported"])
        self.assertNotIn("adapter", receipt)
        stored = receipt.pop("receipt_digest")
        self.assertEqual(stored, oracle.legacy.object_sha256(receipt))


class TensorIdentityTests(unittest.TestCase):
    def test_tensor_digest_is_byte_sensitive(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        first = oracle.tensor_identity(
            torch.tensor([1.0, 2.0], dtype=torch.float32), label="x"
        )
        same = oracle.tensor_identity(
            torch.tensor([1.0, 2.0], dtype=torch.float32), label="x"
        )
        changed = oracle.tensor_identity(
            torch.tensor([1.0, 3.0], dtype=torch.float32), label="x"
        )
        self.assertEqual(first, same)
        self.assertNotEqual(first["content_sha256"], changed["content_sha256"])


class AtomicVideoSaveTests(unittest.TestCase):
    def test_encoder_failure_unlinks_only_fresh_hidden_temporary(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "result.mp4"
            hidden = Path(root) / ".result.tmp-pid-314.mp4"

            def failing_writer(decoded, path, *, fps):
                self.assertEqual(decoded, "frames")
                self.assertEqual(fps, 16)
                Path(path).write_bytes(b"partial")
                raise RuntimeError("encoder failed")

            with mock.patch.dict(oracle.os.environ, {}, clear=False):
                oracle.os.environ.pop("BERNINI_OUTPUT_TRANSACTION_ID", None)
                with mock.patch.object(oracle.os, "getpid", return_value=314):
                    with self.assertRaisesRegex(RuntimeError, "encoder failed"):
                        oracle.save_video_atomically(
                            "frames",
                            output,
                            fps=16,
                            save_output_fn=failing_writer,
                        )
            self.assertFalse(hidden.exists())
            self.assertFalse(output.exists())

    def test_success_replaces_output_and_stale_temp_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "result.mp4"
            hidden = Path(root) / ".result.tmp-pid-2718.mp4"

            def writer(decoded, path, *, fps):
                del decoded, fps
                Path(path).write_bytes(b"video")

            with mock.patch.dict(oracle.os.environ, {}, clear=False):
                oracle.os.environ.pop("BERNINI_OUTPUT_TRANSACTION_ID", None)
                with mock.patch.object(oracle.os, "getpid", return_value=2718):
                    oracle.save_video_atomically(
                        object(), output, fps=16, save_output_fn=writer
                    )
            self.assertEqual(output.read_bytes(), b"video")
            self.assertFalse(hidden.exists())

            output.unlink()
            hidden.write_bytes(b"preexisting")
            with mock.patch.dict(oracle.os.environ, {}, clear=False):
                oracle.os.environ.pop("BERNINI_OUTPUT_TRANSACTION_ID", None)
                with mock.patch.object(oracle.os, "getpid", return_value=2718):
                    with self.assertRaisesRegex(
                        oracle.SourceValueResidualOracleError, "stale temporary"
                    ):
                        oracle.save_video_atomically(
                            object(), output, fps=16, save_output_fn=writer
                        )
            self.assertEqual(hidden.read_bytes(), b"preexisting")

    def test_receipt_atomic_write_uses_launcher_visible_token_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as root:
            receipt_path = Path(root) / "result.mp4.receipt.json"
            hidden = Path(root) / ".result.mp4.receipt.json.tmp-slurm-9"
            with mock.patch.dict(
                oracle.os.environ,
                {"BERNINI_OUTPUT_TRANSACTION_ID": "slurm-9"},
            ):
                oracle.write_receipt_atomically(receipt_path, {"ok": True})
            self.assertEqual(
                receipt_path.read_bytes(),
                oracle.legacy.canonical_json_bytes({"ok": True}) + b"\n",
            )
            self.assertFalse(hidden.exists())

            receipt_path.unlink()
            hidden.write_bytes(b"preexisting")
            with mock.patch.dict(
                oracle.os.environ,
                {"BERNINI_OUTPUT_TRANSACTION_ID": "slurm-9"},
            ):
                with self.assertRaisesRegex(
                    oracle.SourceValueResidualOracleError,
                    "stale temporary receipt",
                ):
                    oracle.write_receipt_atomically(receipt_path, {"ok": True})
            self.assertEqual(hidden.read_bytes(), b"preexisting")

            hidden.unlink()
            with mock.patch.dict(
                oracle.os.environ,
                {"BERNINI_OUTPUT_TRANSACTION_ID": "slurm-9"},
            ):
                with mock.patch.object(
                    oracle.os, "replace", side_effect=OSError("replace failed")
                ):
                    with self.assertRaisesRegex(OSError, "replace failed"):
                        oracle.write_receipt_atomically(
                            receipt_path, {"ok": True}
                        )
            self.assertFalse(hidden.exists())
            self.assertFalse(receipt_path.exists())

    def test_transaction_token_rejects_path_injection(self):
        with mock.patch.dict(
            oracle.os.environ,
            {"BERNINI_OUTPUT_TRANSACTION_ID": "../escape"},
        ):
            with self.assertRaises(oracle.SourceValueResidualOracleError):
                oracle.output_transaction_token()


class AUHLauncherContractTests(unittest.TestCase):
    def setUp(self):
        self.path = (
            METHOD_ROOT
            / "scripts"
            / "auh_infer_source_value_residual_oracle.sbatch"
        )
        self.text = self.path.read_text(encoding="utf-8")

    def test_launcher_is_exact_four_gpu_81_frame_40_step_dog_oracle(self):
        self.assertIn("#SBATCH --gres=gpu:mi210:4", self.text)
        self.assertIn("num_steps=40", self.text)
        self.assertIn("num_frames=81", self.text)
        self.assertIn("seed=2027", self.text)
        self.assertIn("expected_source_tokens=19530", self.text)
        self.assertIn("expected_pair_tokens=39060", self.text)
        self.assertEqual(oracle.EXPECTED_BUCKET_HW, (496, 480))
        self.assertIn(oracle.EXPECTED_SOURCE_SHA256, self.text)
        self.assertIn(oracle.EXPECTED_ORIGINAL_SOURCE_PATH, self.text)
        self.assertIn(oracle.EXPECTED_INSTRUCTION, self.text)

    def test_launcher_exposes_only_arm_as_method_control(self):
        self.assertIn('arm="${BERNINI_SOURCE_VALUE_ARM:', self.text)
        for arm in oracle.ARM_NAMES:
            self.assertIn(f"  {arm})", self.text)
        invocation = self.text.split('"${python_bin}" -m torch.distributed.run', 1)[1]
        invocation = invocation.split("receipt_path=", 1)[0]
        self.assertIn("infer_source_value_residual_oracle.py", invocation)
        self.assertIn('--arm "${arm}"', invocation)
        for forbidden in ("--operator", "--gate", "--block-selection", "--adapter", "--target"):
            self.assertNotIn(forbidden, invocation)

    def test_launcher_is_commit_bound_and_runs_required_tests(self):
        self.assertIn('git -C "${source_repository}" archive --format=tar', self.text)
        self.assertIn('"${source_revision}" methods/bernini_action_editing', self.text)
        self.assertIn('"${archive_from_revision}"', self.text)
        self.assertIn("infer_source_value_residual_oracle.py", self.text)
        self.assertIn("source_value_residual.py", self.text)
        self.assertIn("test_infer_source_value_residual_oracle.py", self.text)
        self.assertIn("test_source_value_residual.py", self.text)
        self.assertIn("checkpoint_content_manifest_sha256=", self.text)

    def test_launcher_has_secondary_output_receipt_hash_closure_and_cleanup(self):
        self.assertIn("output_sha_round1=", self.text)
        self.assertIn("receipt_sha_round1=", self.text)
        self.assertIn("output_sha_round2=", self.text)
        self.assertIn("receipt_sha_round2=", self.text)
        self.assertEqual(self.text.count("audit_receipt || fail"), 2)
        self.assertIn('trap cleanup EXIT', self.text)
        self.assertIn("trap 'exit 143' TERM", self.text)
        self.assertIn("trap 'exit 130' INT", self.text)
        self.assertIn('export TMPDIR="${task_scratch}"', self.text)
        self.assertIn('export BERNINI_OUTPUT_TRANSACTION_ID="${transaction_id}"', self.text)
        self.assertIn('rm -rf -- "${task_scratch}"', self.text)
        self.assertIn('rm -f -- "${output_video}" "${receipt_path}"', self.text)
        self.assertIn("artifact_committed=1", self.text)
        self.assertIn("PASS cleanup_verified=true", self.text)
        self.assertIn('"${task_scratch}" == "${prefix}"*', self.text)
        self.assertEqual(self.text.count('validate_checkpoint_content "${checkpoint_manifest}"'), 2)
        self.assertGreaterEqual(
            self.text.count('git -C "${veomni_root}" rev-parse HEAD'), 2
        )

    def test_launcher_closes_method_and_bernini_sources_after_long_run(self):
        self.assertIn("method_tree_digest_pre=", self.text)
        self.assertIn("method_tree_digest_post=", self.text)
        self.assertIn("pinned_source_identity_pre=", self.text)
        self.assertIn("pinned_source_identity_post=", self.text)
        self.assertIn("original method archive changed during oracle", self.text)
        self.assertIn("private method archive changed during oracle", self.text)
        self.assertIn("method-after-revision.tar", self.text)
        self.assertIn("Bernini source archive changed during oracle", self.text)
        self.assertIn("executed method source tree changed during oracle", self.text)
        self.assertIn("Bernini/VeOmni pinned source identity changed", self.text)

    def test_launcher_exit_trap_failure_injection_removes_only_job_artifacts(self):
        body = self.text.split("cleanup() {", 1)[1].split(
            "\n}\ntrap cleanup EXIT", 1
        )[0]
        cleanup_function = "cleanup() {" + body + "\n}\n"
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            scratch = root_path / "bernini-source-value-77.abcd"
            scratch.mkdir()
            (scratch / "private").write_bytes(b"scratch")
            output = root_path / "F25.mp4"
            receipt = root_path / "F25.mp4.receipt.json"
            output_temp = root_path / ".F25.tmp-slurm-77.mp4"
            receipt_temp = root_path / ".F25.mp4.receipt.json.tmp-slurm-77"
            for path in (output, receipt, output_temp, receipt_temp):
                path.write_bytes(b"job")
            sentinel = root_path / "unrelated.keep"
            sentinel.write_bytes(b"user")
            script = cleanup_function + f'''
scratch_parent={root!r}
SLURM_JOB_ID=77
task_scratch={str(scratch)!r}
artifact_paths_validated=1
artifact_committed=0
output_temporary={str(output_temp)!r}
receipt_temporary={str(receipt_temp)!r}
output_video={str(output)!r}
receipt_path={str(receipt)!r}
false
cleanup
'''
            completed = subprocess.run(
                ["bash", "-c", script],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(completed.returncode, 1, completed.stderr)
            for path in (scratch, output, receipt, output_temp, receipt_temp):
                self.assertFalse(path.exists(), str(path))
            self.assertEqual(sentinel.read_bytes(), b"user")

            scratch.mkdir()
            output.write_bytes(b"video")
            receipt.write_bytes(b"receipt")
            output_temp.write_bytes(b"temporary")
            receipt_temp.write_bytes(b"temporary")
            success_script = cleanup_function + f'''
scratch_parent={root!r}
SLURM_JOB_ID=77
task_scratch={str(scratch)!r}
artifact_paths_validated=1
artifact_committed=1
output_temporary={str(output_temp)!r}
receipt_temporary={str(receipt_temp)!r}
output_video={str(output)!r}
receipt_path={str(receipt)!r}
output_sha_round2=abc
receipt_sha_round2=def
true
cleanup
'''
            completed = subprocess.run(
                ["bash", "-c", success_script],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("PASS cleanup_verified=true", completed.stdout)
            self.assertEqual(output.read_bytes(), b"video")
            self.assertEqual(receipt.read_bytes(), b"receipt")
            self.assertFalse(output_temp.exists())
            self.assertFalse(receipt_temp.exists())
            self.assertFalse(scratch.exists())

    def test_embedded_receipt_auditor_shell_python_arity_matches(self):
        call = self.text.split("audit_receipt() {", 1)[1].split("<<'PY'", 1)[0]
        shell_arguments = re.findall(
            r'^\s{4}"\$\{[A-Za-z0-9_]+\}"', call, flags=re.MULTILINE
        )
        assignment = self.text.split("(\n    path,", 1)[1].split(
            ") = sys.argv[1:]", 1
        )[0]
        python_targets = [
            line.strip()[:-1]
            for line in ("path," + assignment).splitlines()
            if line.strip().endswith(",")
        ]
        self.assertEqual(len(shell_arguments), 17)
        self.assertEqual(len(python_targets), len(shell_arguments))
        self.assertEqual(python_targets.count("gate_text"), 1)

    def test_embedded_receipt_auditor_accepts_every_runner_arm_receipt(self):
        code = self.text.split("audit_receipt() {", 1)[1].split(
            "<<'PY'\n", 1
        )[1].split("\nPY\n}", 1)[0]
        compile(code, "embedded-v10-receipt-audit", "exec")
        with tempfile.TemporaryDirectory() as root:
            for arm in oracle.ARM_NAMES:
                with self.subTest(arm=arm):
                    spec = oracle.arm_spec(arm)
                    receipt = _built_receipt(spec)
                    path = Path(root) / f"{arm}.json"
                    path.write_text(
                        json.dumps(receipt, sort_keys=True), encoding="utf-8"
                    )
                    argv = [
                        "embedded-v10-receipt-audit",
                        str(path),
                        arm,
                        spec.operator,
                        spec.block_selection,
                        str(spec.gate),
                        str(len(spec.block_indices)),
                        str(80 * spec.residual_varlen_calls_per_layer),
                        str(0 if spec.gate == 0.0 else 80),
                        "1" * 40,
                        "2" * 64,
                        oracle.EXPECTED_ORIGINAL_SOURCE_PATH,
                        "/scratch/source.mp4",
                        oracle.EXPECTED_SOURCE_SHA256,
                        "6" * 64,
                        "4" * 64,
                        oracle.PINNED_HISTORICAL_O0_MP4_SHA256,
                        oracle.ARM_TABLE_SHA256,
                    ]
                    with mock.patch.object(sys, "argv", argv):
                        with self.assertRaises(SystemExit) as exited:
                            exec(code, {"__name__": "__main__"})
                    self.assertEqual(exited.exception.code, 0)

    def test_launcher_and_runner_have_no_adapter_or_target_loader(self):
        runner = inspect.getsource(oracle)
        for forbidden in (
            "from peft",
            "import peft",
            "PeftModel",
            "load_adapter",
            "merge_and_unload",
            "--target-video",
            "--mask",
            "--flow",
        ):
            self.assertNotIn(forbidden, runner)
        self.assertIn("adapter=false", self.text)
        self.assertIn("target=false", self.text)
        self.assertNotIn("legacy._atomic_write_json", runner)
        self.assertIn("write_receipt_atomically(receipt_path, receipt)", runner)
        self.assertIn("unlink_fresh_artifact(output_path)", runner)


if __name__ == "__main__":
    unittest.main()
