from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = (
    METHOD_ROOT
    / "scripts/auh_infer_schedule_block_causal_localization_two_holder_v3.sh"
)
V1_CONTROLLER = (
    METHOD_ROOT
    / "scripts/auh_infer_schedule_block_causal_localization_two_holder_v1.sh"
)
V1_CONTROLLER_TEST = (
    METHOD_ROOT
    / "tests/test_auh_infer_schedule_block_causal_localization_two_holder_v1.py"
)
V2_CONTROLLER = (
    METHOD_ROOT
    / "scripts/auh_infer_schedule_block_causal_localization_two_holder_v2.sh"
)
V2_CONTROLLER_TEST = (
    METHOD_ROOT
    / "tests/test_auh_infer_schedule_block_causal_localization_two_holder_v2.py"
)
STAGE_B_CONTROLLER = (
    METHOD_ROOT / "scripts/auh_infer_source_noised_carrier_stage_b_two_holder_v5.sh"
)
STAGE_B_RUNTIME = METHOD_ROOT / "infer_source_noised_carrier_stage_b_v1.py"
STAGE_B_RELEASE = METHOD_ROOT / "releases/source_noised_carrier_stage_b_inference_r3"


class ScheduleBlockCausalLocalizationTwoHolderControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = CONTROLLER.read_text(encoding="utf-8")

    def test_v1_v2_controller_and_test_bytes_are_preserved(self) -> None:
        self.assertEqual(
            hashlib.sha256(V1_CONTROLLER.read_bytes()).hexdigest(),
            "0a820c0d88718f94e019ec7188b3a2482a4330f7e4e1b507404aa2a717cc669a",
        )
        self.assertEqual(
            hashlib.sha256(V1_CONTROLLER_TEST.read_bytes()).hexdigest(),
            "5750d39ca82a46416f8072ea16ebad66750fa66b64528988a62843de05afd080",
        )
        self.assertEqual(
            hashlib.sha256(V2_CONTROLLER.read_bytes()).hexdigest(),
            "dcf7a39cb0171f4606cbdbc63d71eacf4083c128f247520809703a2263ae892e",
        )
        self.assertEqual(
            hashlib.sha256(V2_CONTROLLER_TEST.read_bytes()).hexdigest(),
            "8fb1a334933450297610b29e85811c833b70e08828e3d25edd7135dbf80be793",
        )

    def test_v3_safety_and_cli_logic_are_byte_equivalent_after_pin_normalization(self) -> None:
        normalized = (
            self.source.replace(
                "05b62e8575a2421b535f533530f5e075a12f34814394408fa03f2f51f891c9da",
                "31a32125c11a36104b233a6ab271026add82478cdcb3144331fef6ad1e5f3b05",
            )
            .replace(
                "cfcb491e1059eac2745780c43b799eb5d771e28769c008f6e8e36f97cff46a2f",
                "1d9d4eb37aedffc13d0e1aaf0663561ae989aeb15a42e52437ea4a0dd9287a9f",
            )
            .replace(
                "7e2fb5f91b8ec98106e8f891dbb6058ee6251a81e8bc97407aeaf61778206636",
                "c5226bd3b77630352938451ada7c68b8a5dbf51d95b6c1182b2f47c2aaee237a",
            )
            .replace(
                "aee01060b5661b94be9551b406fdc3c41ea3cc34",
                "7ced6fc99f00c728af477e07cdd58a9e239e973c",
            )
            .replace(
                "auh_infer_schedule_block_causal_localization_two_holder_v3.sh",
                "auh_infer_schedule_block_causal_localization_two_holder_v2.sh",
            )
            .replace(
                "bernini-schedule-block-causal-localization-release-v3",
                "bernini-schedule-block-causal-localization-release-v2",
            )
            .replace(
                'value["release_generation"]=="r3"',
                'value["release_generation"]=="r2"',
            )
        )
        self.assertEqual(normalized, V2_CONTROLLER.read_text(encoding="utf-8"))

    def test_bash_syntax_and_usage_fail_closed(self) -> None:
        subprocess.run(["bash", "-n", str(CONTROLLER)], check=True)
        python_heredocs = re.findall(
            r"<<'PY'\n(.*?)\nPY(?:\n|$)", self.source, flags=re.DOTALL
        )
        self.assertEqual(len(python_heredocs), 7)
        for index, source in enumerate(python_heredocs):
            ast.parse(source, filename=f"controller-heredoc-{index}")
        env = dict(os.environ)
        env.update(
            BERNINI_SBCL_STAGE_A_WORK_JOB0="135412",
            BERNINI_SBCL_STAGE_A_WORK_JOB1="135407",
        )
        result = subprocess.run(
            [str(CONTROLLER)],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("usage:", result.stderr)

    def test_exact_selectable_holder_pair_and_retained_third(self) -> None:
        for first, second in (
            ("135407", "135411"),
            ("135411", "135407"),
            ("135407", "135412"),
            ("135412", "135407"),
            ("135411", "135412"),
            ("135412", "135411"),
        ):
            env = dict(os.environ)
            env.update(
                BERNINI_SBCL_STAGE_A_WORK_JOB0=first,
                BERNINI_SBCL_STAGE_A_WORK_JOB1=second,
            )
            result = subprocess.run(
                [str(CONTROLLER)],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("usage:", result.stderr)
        self.assertIn("135407) printf '%s\\n' auh7-1b-gpu-260", self.source)
        self.assertIn("135411) printf '%s\\n' auh7-1b-gpu-214", self.source)
        self.assertIn("135412) printf '%s\\n' auh7-1b-gpu-293", self.source)
        self.assertIn('assert_parent_running "${retained_job}"', self.source)
        self.assertNotIn('launch_node "${retained_job}"', self.source)
        self.assertNotIn('assert_remote_idle_once "${retained_job}"', self.source)

    def test_formal_profile_is_one_load_smoke_then_fixed_full_grid(self) -> None:
        self.assertIn(
            'readonly profile="${BERNINI_SBCL_STAGE_A_PROFILE:-smoke-then-full-fixed}"',
            self.source,
        )
        self.assertIn(
            '[[ "${profile}" == smoke-then-full-fixed || "${profile}" == smoke-only ]]',
            self.source,
        )
        self.assertIn('"${runtime_entry}" run --profile "${profile}"', self.source)
        self.assertEqual(
            len(re.findall(r"^run_localization$", self.source, flags=re.MULTILINE)),
            1,
        )
        self.assertNotIn("run_smoke", self.source)
        self.assertNotIn("run_full_grid", self.source)
        self.assertIn('"single_distributed_invocation":True', self.source)
        self.assertIn('"single_model_load_required":True', self.source)
        self.assertIn('"automatic_scientific_cell_selection":False', self.source)
        self.assertIn(
            'value["formal_full_continuation_automatic_after_c0_pass"] is True',
            self.source,
        )
        self.assertIn(
            'value["c0_failure_forbids_full_grid"] is True', self.source
        )
        self.assertIn(
            "d11dbd0cfca34f26ea5f72bdd2f5ed8b21c512387410b659ade9f217d866c923",
            self.source,
        )
        self.assertIn(
            "6fd3299a1af84968bebe12cd6f1b2a84feb0fb28a07d29619fbcfac66bf4d2e8",
            self.source,
        )

    def test_world4_two_by_two_and_strict_memory_contract(self) -> None:
        self.assertIn("--nnodes=2 --nproc_per_node=2", self.source)
        self.assertIn('--node_rank="${rank}"', self.source)
        self.assertIn(
            "--ntasks=1 --cpus-per-task=16 --mem=56G --gres=gpu:mi210:2",
            self.source,
        )
        self.assertIn("readonly memory_peak_limit_bytes=55834574848", self.source)
        self.assertIn("sampled<int(limit)", self.source)
        self.assertIn("sacct<int(limit)", self.source)
        self.assertIn("torch.cuda.device_count() == 2", self.source)
        self.assertIn("assert_idle_twice pre-stage-a", self.source)
        self.assertIn("assert_idle_twice final", self.source)

    def test_only_identity_bound_direct_child_srun_can_be_signaled(self) -> None:
        self.assertIn('ppid="$(proc_field "${pid}" 4)"', self.source)
        self.assertIn('[[ "${ppid}" == "$$"', self.source)
        self.assertIn('"$(basename -- "${exe}")" == srun', self.source)
        self.assertIn("pid_cmd_sha", self.source)
        forbidden = (
            "s" + "cancel",
            "scontrol " + "release",
            "scontrol " + "requeue",
            "p" + "kill",
            "kill" + "all",
        )
        for item in forbidden:
            self.assertNotIn(item, self.source)
        direct = [
            line.strip()
            for line in self.source.splitlines()
            if "kill -" in line and "kill -0" not in line
        ]
        self.assertEqual(
            direct,
            [
                'signal_owned_pid() { if pid_identity_matches "$1"; then kill -"$2" "$1" 2>/dev/null || true; elif [[ -e "/proc/$1" ]]; then echo "REFUSE_SIGNAL pid=$1" >&2; fi; }'
            ],
        )

    def test_nfs_visibility_is_two_node_bounded_and_receipted(self) -> None:
        self.assertIn('readonly visibility_spec="${run_root}/nfs-visibility.spec"', self.source)
        self.assertIn('"${source_manifest}" "${method_root}"', self.source)
        self.assertIn(
            'release_pairs=[(str(method_root/row["path"]),row["sha256"]) for row in manifest["files"]]',
            self.source,
        )
        self.assertIn(
            'len({path for path,_ in pairs})==len(pairs)', self.source
        )
        self.assertIn("for attempt in {1..30}", self.source)
        self.assertIn('assert_remote_visibility "${work_node0}" 0', self.source)
        self.assertIn('assert_remote_visibility "${work_node1}" 1', self.source)
        self.assertIn("bernini-schedule-block-stage-a-nfs-visibility-v1", self.source)
        self.assertIn("bounded_retry_attempts", self.source)

    def test_both_dataset_authorities_and_wrong_owner_variant_are_pinned(self) -> None:
        self.assertIn(
            "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_counterfactual_identity_orbit_v5_20260808_c099c6f/datasets/portrait2_rv2v4_exact81_v1",
            self.source,
        )
        pins = (
            "de2f92f314da538f8af322a8f1db23cbdf1feab4b28d2da66248d25309a25595",
            "77d89b3ec2e563f624bab62451b49b616ffa7f7890db6105c4458617aac0d106",
            "6ed77cf7d98391c2074e5938ab50d0688d457bddfd688f9a5825d455447a20bb",
            "12ede44ebab03215e19574967a9afec3c634f246f2cfd2634a48ce0e3dea8738",
            "72c0f104b123a1b7ad69f32697a0b7f7e8c2fdf766c951f3c0bed7518f0f564f",
            "25522068a18893afbc21f54a7851dbf641bc10ea7229653cdfe0c772be1f934e",
            "181e93b1620cafce7de3806b334b6bfdd8e24aa633119cbd6506f3761175a269",
            "845727b8e9c461b9cf1f8bb98c0e27519599ffffcd6619bd8895250a2e075baf",
            "c088eb0128c3c807941f60eb3e763d0e71f4c8dbb190c60b9c0dad6caeca0230",
            "9000dd9dace16501587196ac8459b620529301508ee6c98662f266b3b29b8982",
        )
        for pin in pins:
            self.assertIn(pin, self.source)
        self.assertIn(
            '"${source_spec_sha}" "${source_spec_digest}" "${source_parquet_sha}"',
            self.source,
        )
        self.assertIn(
            '"${wrong_owner_spec_sha}" "${wrong_owner_spec_digest}"',
            self.source,
        )
        self.assertIn('"${wrong_owner_reference_encoding_digest}"', self.source)
        self.assertIn('value["spec"][spec_sha_key] == spec_sha', self.source)
        self.assertIn('value["spec"]["digest"] == spec_digest', self.source)
        self.assertIn(
            'value["spec"]["reference_encoding_contract_digest"] == reference_digest',
            self.source,
        )
        self.assertIn(
            'value["dataset"]["iids"].count(expected_iid) == 1', self.source
        )
        self.assertIn(
            'schema="bernini-appearance-counterfactual-identity-orbit-dataset-receipt-v3"',
            self.source,
        )
        self.assertIn("readonly wrong_owner_variant=variant_a", self.source)
        self.assertIn("00435ad621c44fac", self.source)

    def test_diagnostic_only_and_no_optimizer_authority(self) -> None:
        self.assertNotIn("optimizer.step", self.source)
        self.assertNotIn("backward(", self.source)
        self.assertNotIn("adapter-checkpoint", self.source)
        self.assertIn('"diagnostic_only":True', self.source)
        self.assertIn('"optimizer_authorized":False', self.source)
        self.assertIn('"parameter_update_authorized":False', self.source)
        self.assertIn('"route_update_authorized":False', self.source)
        self.assertIn('"prompt_calibration_action_reverse_direction_passed":True', self.source)
        self.assertIn('"prompt_calibration_noop_incomplete_semantics_passed":False', self.source)
        self.assertIn('"negative_cluster_semantically_validated":False', self.source)
        self.assertIn('"full_grid_cells_retained_without_deletion":True', self.source)
        self.assertIn('"negative_cluster_scientific_veto_authorized":False', self.source)
        self.assertIn('"method_success_claimed":False', self.source)
        self.assertIn('"scientific_claim_authorized":False', self.source)

    def test_real_run_localization_reaches_idle_gate_under_nounset(self) -> None:
        start = self.source.index("run_localization() {")
        end = self.source.index("\n\nrun_localization\n", start)
        real_function = self.source[start:end]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "topology").mkdir()
            harness = "\n".join(
                (
                    "set -u",
                    f"run_root={root}",
                    "master_port=30031",
                    "fail() { printf 'FAIL:%s\\n' \"$*\" >&2; exit 2; }",
                    "assert_idle_twice() { printf 'REACHED_STAGE_A_GATE:%s\\n' \"$1\"; exit 73; }",
                    real_function,
                    "run_localization",
                )
            )
            result = subprocess.run(
                ["bash", "-c", harness],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        self.assertEqual(result.returncode, 73, result.stderr)
        self.assertEqual(result.stdout.strip(), "REACHED_STAGE_A_GATE:pre-stage-a")
        self.assertNotIn("unbound variable", result.stderr)

    def test_no_local_builtin_self_reference(self) -> None:
        unsafe = []
        assignment = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)=")
        for line_number, line in enumerate(self.source.splitlines(), 1):
            command = line.split(";", 1)[0]
            if not command.lstrip().startswith("local "):
                continue
            declarations = list(assignment.finditer(command))
            for item in declarations:
                if f"${{{item.group(1)}}}" in command[item.end() :]:
                    unsafe.append((line_number, item.group(1), command.strip()))
        self.assertEqual(unsafe, [])

    def test_frozen_release_runtime_and_core_pins_are_exact_nonzero(self) -> None:
        self.assertNotIn("0" * 64, self.source)
        self.assertNotIn("readonly expected_release_revision=" + "0" * 40, self.source)
        for line in (
            "readonly expected_runtime_sha=05b62e8575a2421b535f533530f5e075a12f34814394408fa03f2f51f891c9da",
            "readonly expected_core_sha=385cc2321da888f75d5aff5017175b85acf06174969aaa39210b802cc14695c5",
            "readonly expected_release_archive_sha=cfcb491e1059eac2745780c43b799eb5d771e28769c008f6e8e36f97cff46a2f",
            "readonly expected_release_manifest_sha=7e2fb5f91b8ec98106e8f891dbb6058ee6251a81e8bc97407aeaf61778206636",
            "readonly expected_release_revision=aee01060b5661b94be9551b406fdc3c41ea3cc34",
        ):
            self.assertIn(line, self.source)

    def test_frozen_runtime_verifier_and_formal_receipt_closure_are_wired(self) -> None:
        self.assertIn('"${runtime_entry}" verify \\', self.source)
        self.assertIn(
            '--output-dir "${run_root}/localization" --profile "${profile}"',
            self.source,
        )
        self.assertIn('--seed "${seed}"', self.source)
        self.assertIn('len(outputs)==(118 if formal else 6)', self.source)
        self.assertIn('len(artifacts)==(120 if formal else 7)', self.source)
        self.assertIn('len(full["processor_audits"])==136', self.source)
        self.assertIn('len(full["cache_audits"])==24', self.source)
        self.assertIn('actual_objects_rehashed_at_each_phase', self.source)
        self.assertIn('all_actual_input_bytes_unchanged', self.source)
        self.assertIn('native_shared_step_internal_rccl_failure_recoverable":False', self.source)
        self.assertNotIn("RUNTIME_CLI_WIRING", self.source)
        self.assertNotIn("RUNTIME_RECEIPT_WIRING", self.source)

    def test_formal_118_output_postflight_heredoc_executes(self) -> None:
        from methods.bernini_action_editing import (
            infer_schedule_block_causal_localization_v1 as runtime,
        )
        from methods.bernini_action_editing.tests.test_infer_schedule_block_causal_localization_v1 import (
            formal_receipt_fixture,
            resign,
        )

        script = re.findall(
            r"<<'PY'\n(.*?)\nPY(?:\n|$)", self.source, flags=re.DOTALL
        )[-1]
        with tempfile.TemporaryDirectory(prefix="stagea-postflight-", dir="/tmp") as directory:
            root = Path(directory).resolve()
            (root / "localization").mkdir()
            (root / "logs").mkdir()
            (root / "topology/stage-a").mkdir(parents=True)
            receipt = formal_receipt_fixture()
            receipt.pop("receipt_digest")
            receipt["method_source"] = {
                "revision": "aee01060b5661b94be9551b406fdc3c41ea3cc34",
                "archive_sha256": "cfcb491e1059eac2745780c43b799eb5d771e28769c008f6e8e36f97cff46a2f",
            }
            distributed = receipt["distributed"]
            topology = distributed["topology_admission"]
            topology["path"] = str(root / "topology/stage-a")
            resign(topology)
            resign(distributed)
            receipt = runtime.finalize_receipt(receipt)
            (root / "localization/receipt.json").write_bytes(
                runtime.canonical_json_bytes(receipt) + b"\n"
            )
            spec_raw = b"visibility fixture\n"
            (root / "nfs-visibility.spec").write_bytes(spec_raw)
            nfs = {
                "schema_version": "bernini-schedule-block-stage-a-nfs-visibility-v1",
                "complete": True,
                "spec_path": str(root / "nfs-visibility.spec"),
                "spec_sha256": hashlib.sha256(spec_raw).hexdigest(),
                "visibility_digest": "a" * 64,
                "nodes": [
                    {"node": "host-a", "attempt": 1},
                    {"node": "host-b", "attempt": 2},
                ],
                "bounded_retry_attempts": 30,
            }
            nfs["receipt_digest"] = runtime.object_sha256(nfs)
            (root / "nfs-visibility-receipt.json").write_bytes(
                runtime.canonical_json_bytes(nfs) + b"\n"
            )
            for rank, job in ((0, "135407"), (1, "135411")):
                memory = {
                    "schema_version": "bernini-schedule-block-stage-a-memory-crosscheck-v1",
                    "job_id": job,
                    "step_id": str(rank + 1),
                    "sampled_memory_current_peak_bytes": 1234,
                    "sacct_max_rss_raw": "2G",
                    "sacct_max_rss_bytes": 2 * 1024**3,
                    "limit_bytes": 55834574848,
                    "both_below_limit": True,
                }
                (root / "logs" / f"stage-a-node{rank}-memory.json").write_bytes(
                    runtime.canonical_json_bytes(memory) + b"\n"
                )
            args = [
                str(root), "smoke-then-full-fixed", "f" * 64,
                "cfcb491e1059eac2745780c43b799eb5d771e28769c008f6e8e36f97cff46a2f",
                "7e2fb5f91b8ec98106e8f891dbb6058ee6251a81e8bc97407aeaf61778206636",
                "aee01060b5661b94be9551b406fdc3c41ea3cc34",
                "05b62e8575a2421b535f533530f5e075a12f34814394408fa03f2f51f891c9da",
                "385cc2321da888f75d5aff5017175b85acf06174969aaa39210b802cc14695c5",
                "1be281b0419a23254d51556a41eda0d014ecd75cb044caaf5e3ceb96f7c54998",
                "/sealed/source", "/sealed/orbit", "/sealed/source.mp4",
                "/sealed/bernini", "/sealed/veomni", "/sealed/checkpoint",
                "/sealed/checkpoint-manifest.json", "135407", "host-a",
                "135411", "host-b", "135412", str(METHOD_ROOT), "variant_a",
            ]
            result = subprocess.run(
                [sys.executable, "-I", "-B", "-", *args], input=script,
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            final = json.loads((root / "controller-receipt.json").read_text())
            self.assertEqual(
                final["runtime_verified_output_closure"]["decoded_output_count"],
                118,
            )
            self.assertEqual(final["runtime_verified_output_closure"]["artifact_count"], 120)

    def test_frozen_stage_b_bytes_remain_unchanged(self) -> None:
        self.assertEqual(
            hashlib.sha256(STAGE_B_CONTROLLER.read_bytes()).hexdigest(),
            "5aa68c97c52cba9f2a2171b9ff98f6fc865c67ab641c11a07799369715e71f02",
        )
        self.assertEqual(
            hashlib.sha256(STAGE_B_RUNTIME.read_bytes()).hexdigest(),
            "7e6cdba95c62d2ae9bbe81cfa123ac208c2ca890f134cfe6d0538cefea68db50",
        )
        self.assertEqual(
            hashlib.sha256((STAGE_B_RELEASE / "source.tar").read_bytes()).hexdigest(),
            "e3880934c3e6cfcb0dfe56aa34a03f3ffbb2cb192a262fdb8ae1734a02f183ca",
        )
        self.assertEqual(
            hashlib.sha256(
                (STAGE_B_RELEASE / "source.manifest.json").read_bytes()
            ).hexdigest(),
            "6849ed11ad214e4c49f72731e4beb88948f2abf26e79f0ff5cf8c4e2814e62a3",
        )


if __name__ == "__main__":
    unittest.main()
