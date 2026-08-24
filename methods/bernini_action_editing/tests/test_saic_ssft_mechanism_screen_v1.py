#!/usr/bin/env python3
"""AUH-only hostile/contract tests for the pre-registered SSFT screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(METHOD_ROOT))

import infer_saic_ssft_mechanism_screen_v1 as runner  # noqa: E402
import release_saic_ssft_mechanism_screen_v1 as release  # noqa: E402


PLAN_PATH = METHOD_ROOT / "assets" / "saic_ssft_mechanism_screen_v1.json"
LAUNCHER_PATH = (
    METHOD_ROOT
    / "scripts"
    / "auh_run_saic_ssft_mechanism_screen_all8_v1.sbatch"
)
SUBMITTER_PATH = (
    METHOD_ROOT
    / "scripts"
    / "auh_submit_saic_ssft_mechanism_screen_v1.sh"
)


class ArmRegistryTests(unittest.TestCase):
    def test_only_four_preregistered_arms_in_fixed_order(self) -> None:
        self.assertEqual(runner.ARM_NAMES, ("T1", "IAVG", "I1", "I1A"))
        self.assertEqual(tuple(runner.ARM_SPECS), runner.ARM_NAMES)
        for forbidden in ("T0", "I0", "R00", "R11", "V-old", "I2"):
            with self.assertRaises(runner.SAICInferenceError):
                runner.arm_spec(forbidden)

    def test_arm_contract_is_exact_preregistration(self) -> None:
        expected = {
            "T1": ("t2v_apg", "t2v", "t2v_apg", "source_similarity_softmax", 0.01, False, 208),
            "IAVG": ("r2v_apg_source_i0", "r2v", "r2v_apg", "uniform", None, False, 312),
            "I1": ("r2v_apg_source_i0", "r2v", "r2v_apg", "source_similarity_softmax", 0.01, False, 312),
            "I1A": ("r2v_apg_source_i0", "r2v", "r2v_apg", "source_similarity_softmax", 0.01, True, 312),
        }
        for name, values in expected.items():
            spec = runner.arm_spec(name)
            self.assertEqual(
                (
                    spec.field_regime,
                    spec.task_name,
                    spec.guidance_mode,
                    spec.aggregation_mode,
                    spec.temperature,
                    spec.anchor_latent_phase_zero,
                    spec.expected_raw_forwards,
                ),
                values,
            )
            self.assertEqual(spec.candidate_schedule, (5, 5, 5) + (1,) * 37)
            self.assertTrue(spec.anc_enabled)
            self.assertEqual(spec.expected_guided_queries, 104)

    def test_delegate_is_bound_to_new_schema_registry_and_runtime_file(self) -> None:
        self.assertEqual(runner._impl.SCHEMA_VERSION, runner.SCHEMA_VERSION)
        self.assertEqual(runner._impl.METHOD, runner.METHOD)
        self.assertIs(runner._impl.arm_spec, runner.arm_spec)
        self.assertEqual(tuple(runner._impl.ARM_NAMES), runner.ARM_NAMES)
        self.assertEqual(
            runner._impl.RUNTIME_METHOD_FILES.count(
                "infer_saic_ssft_mechanism_screen_v1.py"
            ),
            1,
        )

    def test_parser_rejects_unregistered_arm(self) -> None:
        parser = runner.build_parser()
        arm_action = next(action for action in parser._actions if action.dest == "arm")
        self.assertEqual(tuple(arm_action.choices), runner.ARM_NAMES)


class PlanAndLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        cls.launcher = LAUNCHER_PATH.read_text(encoding="utf-8")
        cls.submitter = SUBMITTER_PATH.read_text(encoding="utf-8")

    def test_plan_has_exact_runtime_and_false_authority(self) -> None:
        plan = self.plan
        self.assertEqual(
            plan["preregistration"]["fixed_arm_order"], list(runner.ARM_NAMES)
        )
        self.assertTrue(plan["preregistration"]["registered_before_job132387"])
        self.assertFalse(plan["preregistration"]["scientific_treatment_extension"])
        self.assertEqual(plan["runtime"]["candidate_schedule"], [5, 5, 5] + [1] * 37)
        self.assertEqual(
            [plan["runtime"][key] for key in ("frames", "fps", "latent_frames", "inference_steps", "flow_shift")],
            [81, 25, 21, 40, 5.0],
        )
        self.assertEqual(plan["runtime"]["optimizer_steps"], 0)
        self.assertEqual(plan["runtime"]["training_updates"], 0)
        self.assertTrue(all(value is False for value in plan["authority"].values()))
        self.assertEqual(
            plan["preregistration"]["document"],
            "md/action_editing/bernini_saic_source_state_flow_transport_v3.md",
        )
        self.assertRegex(
            plan["preregistration"]["document_sha256"], r"^[0-9a-f]{64}$"
        )

    def test_plan_dependencies_are_exact_file_hashes(self) -> None:
        dependencies = self.plan["implementation_dependencies"]
        self.assertEqual(
            set(dependencies),
            {
                "infer_saic_source_state_flow_transport_v2.py",
                "materialize_saic_frame0_latent_v1.py",
            },
        )
        for name, expected in dependencies.items():
            self.assertRegex(expected, r"^[0-9a-f]{64}$")
            self.assertEqual(
                hashlib.sha256((METHOD_ROOT / name).read_bytes()).hexdigest(),
                expected,
            )

    def test_source_coordinates_are_exact_job132387_artifacts(self) -> None:
        sources = self.plan["sealed_inputs"]["sources"]
        self.assertEqual([item["group"] for item in sources], ["dog", "human"])
        self.assertEqual(self.plan["sealed_inputs"]["upstream_job_id"], "132387")
        old_root = (
            "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
            "VideoEdit_experiments/bernini_saic_v1_20260809/runs/"
            "ssft-t0-i0-ec4bfb6-j6"
        )
        self.assertEqual(
            self.plan["sealed_inputs"]["source_manifest"]["path"],
            f"{old_root}/sealed-inputs/saic_reversible_source_set_v1.json",
        )
        self.assertEqual(
            self.plan["sealed_inputs"]["event_bank"]["path"],
            f"{old_root}/sealed-inputs/saic_pure_t2v_event_bank_v1.json",
        )
        expected = {
            "dog": {
                "row_id": "fit-dog-00-7b88a1ca1f804f41",
                "rollout_seed": 2026082101,
                "source_clean_latent_path": f"{old_root}/groups/dog/source.clean-latent.safetensors",
                "source_clean_latent_sha256": "e3560e77546d3936f7e7231d5aceb78b8f29ea379b98748c26e4b37a5f277d7a",
                "source_clean_receipt_sha256": "bf1a817b6ab2f415311c19b0111f90c7a5537b3243df135d5fbcba06f63c37f1",
                "source_clean_tensor_raw_sha256": "86c6b10800cfee13a1c3ca95b388a6ce8f310af6cbb84eea462728d8cd046343",
            },
            "human": {
                "row_id": "fit-human-00-a35b590961d24694",
                "rollout_seed": 2026082121,
                "source_clean_latent_path": f"{old_root}/groups/human/source.clean-latent.safetensors",
                "source_clean_latent_sha256": "9cab2ac419833a3d8451b24c578c3ca3b466341f7bc17ea40e4067c8ebf6b7e9",
                "source_clean_receipt_sha256": "ffdc8e4bce3181e0328f0714289b9d03d3c25ff5ac4ae57357d984101ddae0ba",
                "source_clean_tensor_raw_sha256": "2f7333434783bf681005c62200877aca1f95a395f21f228a2e83689ea0e1b263",
            },
        }
        for source in sources:
            self.assertIn("/ssft-t0-i0-ec4bfb6-j6/", source["source_clean_latent_path"])
            for key, value in expected[source["group"]].items():
                self.assertEqual(source[key], value)
            for key in (
                "source_video_sha256",
                "source_clean_latent_sha256",
                "source_clean_receipt_sha256",
                "source_clean_tensor_raw_sha256",
                "job132387_ephemeral_frame0_tensor_raw_sha256",
            ):
                self.assertRegex(source[key], r"^[0-9a-f]{64}$")
            self.assertFalse(source["fresh_frame0_must_match_job132387_ephemeral"])

    def test_launcher_uses_all_eight_gpus_and_fixed_waves(self) -> None:
        text = self.launcher
        self.assertIn("#SBATCH --gres=gpu:mi210:8", text)
        self.assertIn("readonly arms=(T1 IAVG I1 I1A)", text)
        self.assertIn('"0,1,2,3"', text)
        self.assertIn('"4,5,6,7"', text)
        self.assertIn("--nproc_per_node=4", text)
        self.assertIn('for arm in "${arms[@]}"', text)
        self.assertNotIn("--dependency", text)
        self.assertNotIn("--hold", text)

    def test_launcher_materializes_only_two_frame0_coordinates(self) -> None:
        text = self.launcher
        self.assertEqual(text.count("materialize_frame0 dog "), 1)
        self.assertEqual(text.count("materialize_frame0 human "), 1)
        self.assertIn(
            '"fresh_frame0_materialization_count": 2',
            (METHOD_ROOT / "release_saic_ssft_mechanism_screen_v1.py").read_text(
                encoding="utf-8"
            ),
        )
        self.assertIn("--reference-frame0-latent", text)
        self.assertIn("--source-clean-latent", text)
        self.assertNotIn("materialize_saic_source_clean_latent_v1.py", text)

    def test_launcher_has_no_training_or_selection_entrypoint(self) -> None:
        text = self.launcher
        for forbidden in (
            "torch.optim",
            "optimizer.step",
            "train_saic",
            "selected_for_training=true",
            "semantic_action_success=true",
        ):
            self.assertNotIn(forbidden, text)

    def test_release_uses_explicit_pinned_ffprobe_without_ambient_path(self) -> None:
        text = self.launcher
        release_text = (
            METHOD_ROOT / "release_saic_ssft_mechanism_screen_v1.py"
        ).read_text(encoding="utf-8")
        self.assertIn("readonly pinned_ffprobe=/vast/", text)
        self.assertIn(
            "pinned_ffprobe_sha256=356754aa8e327b139dd54dda6846af6425673b73572ab6c1c182ed970f1107f5",
            text,
        )
        self.assertIn(
            "pinned_ffprobe_version_stdout_sha256=2271b81138bdaf07532b801ac7abd5b48d9e84dd66a6287a82fb44bc04c84f6b",
            text,
        )
        self.assertIn('--ffprobe-bin "${pinned_ffprobe}"', text)
        self.assertIn('str(ffprobe_bin)', release_text)
        self.assertNotIn('[\n        "ffprobe",', release_text)

    def test_submitter_creates_schedulable_job_without_hold_or_dependency(self) -> None:
        text = self.submitter
        self.assertIn("sbatch --parsable --export=ALL", text)
        self.assertNotIn("sbatch --hold", text)
        self.assertNotIn("--dependency=", text)
        self.assertIn('[[ "${state}" == PENDING || "${state}" == RUNNING ]]', text)
        self.assertIn('[[ "${reason}" != JobHeldUser', text)


class ReleaseHostileTests(unittest.TestCase):
    def _write_receipt(self, path: Path, unsigned: dict) -> None:
        value = {**unsigned, "receipt_digest": release.object_sha256(unsigned)}
        path.write_bytes(release.canonical_json_bytes(value) + b"\n")
        path.chmod(0o444)

    def test_canonical_receipt_accepts_exact_and_rejects_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            self._write_receipt(path, {"schema_version": "x", "value": False})
            receipt, digest = release.load_canonical_receipt(path)
            self.assertEqual(receipt["schema_version"], "x")
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
            path.chmod(0o644)
            path.write_bytes(path.read_bytes().replace(b"false", b"true ", 1))
            path.chmod(0o444)
            with self.assertRaises(release.MechanismScreenReleaseError):
                release.load_canonical_receipt(path)

    def test_plan_loader_rejects_unpinned_dependency(self) -> None:
        value = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        value["implementation_dependencies"][
            "infer_saic_source_state_flow_transport_v2.py"
        ] = "PIN_LATER"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            path.chmod(0o444)
            with self.assertRaises(release.MechanismScreenReleaseError):
                release.load_plan(path)

    def test_authority_rejects_any_positive_claim(self) -> None:
        base = {
            "authority": {
                "frozen_inference_execution_receipt": True,
                "semantic_action_success": False,
                "training_authority": False,
            }
        }
        release._false_authority(base, label="ok")
        bad = json.loads(json.dumps(base))
        bad["authority"]["semantic_action_success"] = True
        with self.assertRaises(release.MechanismScreenReleaseError):
            release._false_authority(bad, label="bad")

    def test_ffprobe_rejects_wrong_frame_count(self) -> None:
        result = mock.Mock(stdout=json.dumps({
            "streams": [{
                "codec_name": "h264",
                "width": 496,
                "height": 480,
                "avg_frame_rate": "25/1",
                "nb_read_frames": "80",
            }]
        }))
        with mock.patch.object(release.subprocess, "run", return_value=result):
            with self.assertRaises(release.MechanismScreenReleaseError):
                release._ffprobe_exact81(
                    Path("/does/not/matter.mp4"), Path("/pinned/ffprobe")
                )

    def test_ffprobe_command_starts_with_explicit_absolute_pin(self) -> None:
        result = mock.Mock(stdout=json.dumps({
            "streams": [{
                "codec_name": "h264",
                "width": 496,
                "height": 480,
                "avg_frame_rate": "25/1",
                "nb_read_frames": "81",
            }]
        }))
        with mock.patch.object(release.subprocess, "run", return_value=result) as run:
            value = release._ffprobe_exact81(
                Path("/does/not/matter.mp4"), Path("/pinned/ffprobe")
            )
        self.assertEqual(run.call_args.args[0][0], "/pinned/ffprobe")
        self.assertEqual(value["frames"], 81)

    def test_ffprobe_identity_rejects_wrong_bytes_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            probe = Path(directory).resolve() / "ffprobe"
            probe.write_bytes(b"#!/bin/sh\nexit 0\n")
            probe.chmod(0o555)
            with mock.patch.object(release.subprocess, "run") as run:
                with self.assertRaises(release.MechanismScreenReleaseError):
                    release._validate_ffprobe(
                        probe,
                        expected_sha256="0" * 64,
                        expected_version_stdout_sha256="1" * 64,
                        expected_version_first_line="never",
                    )
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
