from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = METHOD_ROOT / "scripts" / "auh_infer_source_noised_carrier_stage_b_two_holder_v4.sh"
V1_CONTROLLER = METHOD_ROOT / "scripts" / "auh_infer_source_noised_carrier_stage_b_two_holder_v1.sh"
V2_CONTROLLER = METHOD_ROOT / "scripts" / "auh_infer_source_noised_carrier_stage_b_two_holder_v2.sh"
V3_CONTROLLER = METHOD_ROOT / "scripts" / "auh_infer_source_noised_carrier_stage_b_two_holder_v3.sh"
R1_RELEASE_ROOT = METHOD_ROOT / "releases" / "source_noised_carrier_stage_b_inference_r1"
RELEASE_ROOT = METHOD_ROOT / "releases" / "source_noised_carrier_stage_b_inference_r2"


class StageBInferenceSelectableTwoHolderControllerV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = CONTROLLER.read_text(encoding="utf-8")

    def test_bash_syntax_and_usage_fail_closed(self) -> None:
        subprocess.run(["bash", "-n", str(CONTROLLER)], check=True)
        environment = dict(os.environ)
        environment.update(
            BERNINI_SNC_STAGE_B_INFER_WORK_JOB0="135412",
            BERNINI_SNC_STAGE_B_INFER_WORK_JOB1="135407",
        )
        result = subprocess.run(
            [str(CONTROLLER)], env=environment, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("usage:", result.stderr)

    def test_exact_three_holder_choose_two_closure_and_child_propagation(self) -> None:
        ordered_pairs = (
            ("135407", "135411"), ("135411", "135407"),
            ("135407", "135412"), ("135412", "135407"),
            ("135411", "135412"), ("135412", "135411"),
        )
        for first, second in ordered_pairs:
            environment = dict(os.environ)
            environment.update(
                BERNINI_SNC_STAGE_B_INFER_WORK_JOB0=first,
                BERNINI_SNC_STAGE_B_INFER_WORK_JOB1=second,
            )
            result = subprocess.run(
                [str(CONTROLLER)], env=environment, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("usage:", result.stderr)
        for first, second, message in (
            ("135412", "135412", "must be distinct"),
            ("135412", "999999", "outside exact allowlist"),
        ):
            environment = dict(os.environ)
            environment.update(
                BERNINI_SNC_STAGE_B_INFER_WORK_JOB0=first,
                BERNINI_SNC_STAGE_B_INFER_WORK_JOB1=second,
            )
            result = subprocess.run(
                [str(CONTROLLER)], env=environment, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn(message, result.stderr)
        self.assertIn('135407) printf \'%s\\n\' auh7-1b-gpu-260', self.source)
        self.assertIn('135411) printf \'%s\\n\' auh7-1b-gpu-214', self.source)
        self.assertIn('135412) printf \'%s\\n\' auh7-1b-gpu-293', self.source)
        self.assertIn('BERNINI_SNC_STAGE_B_INFER_WORK_JOB0="${work_job0}"', self.source)
        self.assertIn('BERNINI_SNC_STAGE_B_INFER_WORK_JOB1="${work_job1}"', self.source)
        self.assertNotIn("readonly work_job0=135407", self.source)

    def test_unique_release_authority_and_all_fourteen_members_are_pinned(self) -> None:
        archive = RELEASE_ROOT / "source.tar"
        manifest_path = RELEASE_ROOT / "source.manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
        archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
        manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        revision = manifest["content_closure_sha1"]
        self.assertEqual(archive_sha, "298a0a63dda44843a828b70b87c9b41441c84a004b7cfec9f4340e2497ba54c1")
        self.assertEqual(manifest_sha, "72a787ce5e645e7cb44d3331777899e93d35d07e7155c9bff97682367fe38766")
        self.assertEqual(revision, "65440005f770fa2fb0691d2d6dfe1fcbc3bce357")
        self.assertEqual(manifest["file_count"], 14)
        for value in (archive_sha, manifest_sha, revision):
            self.assertIn(value, self.source)
        self.assertIn(
            '[[ "${source_archive_sha}" == "${expected_release_archive_sha}" && "${source_manifest_sha}" == "${expected_release_manifest_sha}" && "${source_revision}" == "${expected_release_revision}" ]]',
            self.source,
        )
        for row in manifest["files"]:
            self.assertIn(row["sha256"], self.source)

    def test_prior_controllers_and_runtime_release_remain_byte_identical(self) -> None:
        self.assertEqual(
            hashlib.sha256(V1_CONTROLLER.read_bytes()).hexdigest(),
            "de4a9c8e4955194c4a6fbf2a11a69afdc77b1598df3925b3e5480d74ffa56f0f",
        )
        self.assertEqual(
            hashlib.sha256(V2_CONTROLLER.read_bytes()).hexdigest(),
            "0dd2918e5adfa553edc6900a5b400daab1df46e0912ea796ede3179da780c7cd",
        )
        self.assertEqual(
            hashlib.sha256(V3_CONTROLLER.read_bytes()).hexdigest(),
            "f0c93036f383591754d97702dbe36ecaa4feda91389c46631a09851c97f91444",
        )
        self.assertEqual(
            hashlib.sha256((R1_RELEASE_ROOT / "source.tar").read_bytes()).hexdigest(),
            "646d5a9c73364db689b2592d3b1a4a486c2e9e032031f09e5be3a003845f3698",
        )
        self.assertEqual(
            hashlib.sha256((R1_RELEASE_ROOT / "source.manifest.json").read_bytes()).hexdigest(),
            "50f87120b08e4a576acd3ff44efdda699848db9c2d9d13336f5006431f418639",
        )
        self.assertNotEqual(V3_CONTROLLER.read_bytes(), CONTROLLER.read_bytes())

    def test_real_run_branch_reaches_first_gate_under_nounset(self) -> None:
        start = self.source.index("run_branch() {")
        end = self.source.index("\nverify_pair() {", start)
        real_function = self.source[start:end]
        harness = "\n".join(
            (
                "set -u",
                "run_root=/private/tmp/stage-b-v4-run-branch-nounset-$$",
                "fail() { printf 'FAIL:%s\\n' \"$*\" >&2; exit 2; }",
                "assert_idle_twice() { printf 'REACHED_RUN_BRANCH_GATE:%s\\n' \"$1\"; exit 73; }",
                real_function,
                "run_branch registered_probes_frozen_base registered-probes frozen_base 29861",
            )
        )
        result = subprocess.run(
            ["bash", "-c", harness], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self.assertEqual(result.returncode, 73, result.stderr)
        self.assertEqual(
            result.stdout.strip(),
            "REACHED_RUN_BRANCH_GATE:pre-registered_probes_frozen_base",
        )
        self.assertNotIn("unbound variable", result.stderr)

    def test_no_local_builtin_references_a_variable_assigned_earlier_in_same_builtin(self) -> None:
        unsafe: list[tuple[int, str, str]] = []
        assignment = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)=")
        for line_number, line in enumerate(self.source.splitlines(), 1):
            command = line.split(";", 1)[0]
            if not command.lstrip().startswith("local "):
                continue
            declarations = list(assignment.finditer(command))
            for item in declarations:
                if f"${{{item.group(1)}}}" in command[item.end():]:
                    unsafe.append((line_number, item.group(1), command.strip()))
        self.assertEqual(unsafe, [])

    def test_real_numpy_decode_smoke_is_no_gpu_create_only_and_precedes_launch(self) -> None:
        start = self.source.index('readonly decode_smoke_receipt="${run_root}/real-numpy-decode-smoke.json"')
        end = self.source.index('\n"${python_bin}" -I -B - "${dataset_root}"', start)
        smoke = self.source[start:end]
        self.assertIn("CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=", smoke)
        self.assertIn("import numpy as np", smoke)
        self.assertIn("save_validated_vae_decoded_clip", smoke)
        self.assertIn("writer_calls[0][0] is decoded", smoke)
        self.assertIn("os.O_WRONLY | os.O_CREAT | os.O_EXCL", smoke)
        self.assertIn('"gpu_visibility_disabled": True', smoke)
        self.assertNotIn("torch.cuda", smoke)
        self.assertLess(start, self.source.index("run_branch() {"))
        self.assertLess(start, self.source.index("run_branch registered_probes_frozen_base"))

    def test_four_world4_branches_are_serial_and_pair_verification_is_in_order(self) -> None:
        statements = [
            "run_branch registered_probes_frozen_base registered-probes frozen_base",
            "run_branch registered_probes_trained registered-probes trained",
            'verify_pair "${run_root}/registered_probes_frozen_base"',
            "run_branch full40_evolved_target_frozen_base full40-evolved-target-all40-route-extrapolation frozen_base",
            "run_branch full40_evolved_target_trained full40-evolved-target-all40-route-extrapolation trained",
            'verify_pair "${run_root}/full40_evolved_target_frozen_base"',
        ]
        positions = [self.source.index(value) for value in statements]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(self.source.count("run_branch registered_probes_"), 2)
        self.assertEqual(self.source.count("run_branch full40_evolved_target_"), 2)
        self.assertIn("--nnodes=2 --nproc_per_node=2", self.source)
        self.assertIn('--seed "${seed}"', self.source)
        self.assertNotIn("--anchor-action-video", self.source)
        self.assertNotIn("--expected-anchor-action-sha256", self.source)

    def test_child_signals_are_pid_identity_bound_and_parent_actions_forbidden(self) -> None:
        self.assertIn('ppid="$(proc_field "${pid}" 4)"', self.source)
        self.assertIn('[[ "${ppid}" == "$$"', self.source)
        self.assertIn('"$(basename -- "${exe}")" == srun', self.source)
        self.assertIn("pid_cmd_sha", self.source)
        self.assertIn("signal_owned_pid", self.source)
        forbidden = ("s" + "cancel", "scontrol " + "release", "scontrol " + "requeue", "p" + "kill", "kill" + "all")
        for value in forbidden:
            self.assertNotIn(value, self.source)
        direct_signals = [
            line.strip()
            for line in self.source.splitlines()
            if "kill -" in line and "kill -0" not in line
        ]
        self.assertEqual(
            direct_signals,
            ['signal_owned_pid() { if pid_identity_matches "$1"; then kill -"$2" "$1" 2>/dev/null || true; elif [[ -e "/proc/$1" ]]; then echo "REFUSE_SIGNAL pid=$1" >&2; fi; }'],
        )

    def test_resources_retained_parent_and_final_state_are_hard_gated(self) -> None:
        self.assertIn("readonly memory_peak_limit_bytes=55834574848", self.source)
        self.assertIn("--ntasks=1 --cpus-per-task=16 --mem=56G --gres=gpu:mi210:2", self.source)
        self.assertIn('sampled<int(limit)', self.source)
        self.assertIn('sacct<int(limit)', self.source)
        self.assertIn('assert_parent_running "${work_job0}"', self.source)
        self.assertIn('assert_parent_running "${work_job1}"', self.source)
        self.assertIn('assert_parent_running "${retained_job}"', self.source)
        self.assertNotIn('launch_child "${retained_job}"', self.source)
        self.assertNotIn('assert_remote_idle_once "${retained_job}"', self.source)
        self.assertLess(self.source.rindex("assert_idle_twice final"), self.source.rindex("assert_all_parents_running"))
        self.assertIn("parents_retained=135407,135411,135412", self.source)


if __name__ == "__main__":
    unittest.main()
