from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


MODULE_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = MODULE_ROOT / "tools"
for entry in (MODULE_ROOT, TOOLS_ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import full644_exploratory_matched_r5d_root_bootstrap_probe_runner_v1 as bootstrap
import full644_exploratory_matched_r5d_static_nomodel_probe_v1 as static
import full644_exploratory_matched_r5d_cpu_consumption_probe_v1 as consumption
import full644_exploratory_matched_spooled_launcher_auh_r5d as launcher
import materialize_full644_exploratory_matched_r5d_case00_package_v1 as materializer


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class R5DPackageContractTests(unittest.TestCase):
    def test_frozen_production_tuple_and_physical_release(self) -> None:
        expected = {
            "full644_exploratory_matched_infer_adapter_auh_r5d.py":
                "5794e1f0e5ecb84ffdb37f618fe63696ee4f87176952ac083c8c91792a9d192a",
            "full644_exploratory_matched_spooled_launcher_auh_r5d.py":
                "85ccc17b30d97a7bf048702cd8a8ed10c3421e01721902fea7db6242eac45753",
            "full644_exploratory_matched_runner_auh_r5.py":
                "847b91a267fe55cfbfa793027548f82beb5ec9630efab329878576ae6c5a9223",
            "full644_exploratory_matched_torchrun_fd_bridge_v2.py":
                "c91de7eb821a05c61f66349c02f9232ede27c49e54659f351f72930fb071d136",
            "full644_exploratory_matched_infer_adapter_v2.py":
                "53b75aea4897a0ec5ad70c8ea2b2dd314b93d1331cf5e41d65c3b51339f4d4ca",
            "action_preservation_decoded_eval_model_authority_v2.py":
                "b9457e434b8000e5368056c925edd0227b4dd3d8a439090494af088817d51ecf",
        }
        for name, digest in expected.items():
            self.assertEqual(sha(MODULE_ROOT / name), digest)
        self.assertEqual(
            hashlib.sha256(launcher.ROOT_BOOTSTRAP.encode("utf-8")).hexdigest(),
            static.ROOT_BOOTSTRAP_SHA256,
        )
        self.assertEqual(len(materializer.RELEASE_FILES), 17)
        self.assertEqual(materializer.RELEASE_FILES, static.RELEASE_FILES)
        self.assertEqual(len(static.EXTERNAL_IDENTITY_PINS), 7)
        self.assertIn(
            "methods/bernini_action_editing/full644_exploratory_matched_spooled_launcher_auh_r5.py",
            materializer.RELEASE_FILES,
        )
        self.assertIn(
            "methods/bernini_action_editing/full644_exploratory_matched_spooled_launcher_auh_r5d.py",
            materializer.RELEASE_FILES,
        )

    def test_profiles_are_exact_and_launch_is_case00(self) -> None:
        self.assertEqual(
            {key: str(value) for key, value in materializer.TARGETS.items()},
            {
                ("143811", "auh7-1b-gpu-306"): (
                    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
                    "VideoEditing/VideoEdit_experiments/"
                    "bernini_full644_exploratory_matched_eval_auh_r5d_"
                    "job143811_node306_case00_847b91a2_c91de7eb_85ccc17b_r1"
                ),
                ("143812", "auh7-1b-gpu-293"): (
                    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
                    "VideoEditing/VideoEdit_experiments/"
                    "bernini_full644_exploratory_matched_eval_auh_r5d_"
                    "job143812_node293_case00_847b91a2_c91de7eb_85ccc17b_r1"
                ),
                ("143808", "auh7-1b-gpu-315"): (
                    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
                    "VideoEditing/VideoEdit_experiments/"
                    "bernini_full644_exploratory_matched_eval_auh_r5d_"
                    "job143808_node315_case00_847b91a2_c91de7eb_85ccc17b_r1"
                ),
            },
        )
        root = Path("/vast/fresh-r5d")
        plan = root / "plan/full644_exploratory_matched_plan_auh_r5d.json"
        value = materializer.launch_input(root, "143812", "auh7-1b-gpu-293", plan)
        self.assertEqual(value["schema_version"], "full644-exploratory-matched-root-launch-input-auh-r5d")
        self.assertEqual(value["campaign_mode"], "case00-pair-canary")
        self.assertTrue(value["adapter"].endswith("_infer_adapter_auh_r5d.py"))
        self.assertTrue(value["base_adapter"].endswith("_infer_adapter_v2.py"))
        self.assertNotEqual(value["adapter"], value["base_adapter"])

    def test_diagnostic_pins_and_consumption_contract_are_exact(self) -> None:
        for relative in (
            materializer.ROOT_BOOTSTRAP_PROBE,
            materializer.STATIC_NOMODEL_PROBE,
            materializer.CPU_CONSUMPTION_PROBE,
        ):
            self.assertEqual(
                sha(MODULE_ROOT.parents[1] / relative),
                materializer.DIAGNOSTIC_SOURCE_PINS[relative],
            )
        materializer.ensure_ready_pins()
        self.assertNotIn(
            materializer.CPU_CONSUMPTION_PROBE, materializer.RELEASE_FILES
        )
        self.assertEqual(
            consumption.SCHEMA,
            "full644-exploratory-matched-r5d-cpu-consumption-probe-v1",
        )
        required_options = {
            action.dest
            for action in consumption.build_parser()._actions
            if action.required
        }
        self.assertEqual(
            required_options,
            {
                "methods_root", "site_packages_root", "work_root", "receipt",
                "probe_sha256",
            },
        )
        for relative, digest, size in consumption.SOURCE_SPECS.values():
            release_relative = "methods/bernini_action_editing/" + relative
            source = MODULE_ROOT / relative
            self.assertEqual(materializer.RELEASE_FILES[release_relative], digest)
            self.assertEqual(sha(source), digest)
            self.assertEqual(source.stat().st_size, size)

    def test_unfrozen_diagnostic_pin_blocks_before_root_creation(self) -> None:
        arguments = types.SimpleNamespace(
            job_id="143812",
            node="auh7-1b-gpu-293",
            source_root=str(MODULE_ROOT.parents[1].resolve()),
        )
        with mock.patch.dict(
            materializer.DIAGNOSTIC_SOURCE_PINS,
            {materializer.CPU_CONSUMPTION_PROBE: "PENDING"},
        ), mock.patch.object(
            materializer, "mkdir_fresh"
        ) as mkdir, self.assertRaisesRegex(
            materializer.R5DMaterializationError, "not frozen"
        ):
            materializer._materialize(arguments)
        mkdir.assert_not_called()

    def test_static_payload_is_captured_source_and_syntax_valid(self) -> None:
        root = Path("/vast/fresh-r5d")
        raw = materializer.build_static_payload(
            root=root,
            job_id="143812",
            node="auh7-1b-gpu-293",
            source=root / "diagnostics/static.py",
            source_sha256="a" * 64,
            plan=root / "plan/full644_exploratory_matched_plan_auh_r5d.json",
            plan_sha256="b" * 64,
            launch_input_path=root / "launch/root_launch_input_auh_r5d.json",
            launch_input_sha256="c" * 64,
            launch_receipt_path=root / "launch/root_launch_receipt_auh_r5d.json",
            launch_receipt_sha256="d" * 64,
        )
        self.assertEqual(raw.count(b"exec -c"), 1)
        self.assertEqual(raw.count(b"-I -S -B -c"), 1)
        self.assertNotIn(b"python3.12 -", raw)
        self.assertIn(b'"$R5D_SOURCE_FD"', raw)
        self.assertNotIn("source_raw=read(srcfd", materializer.CAPTURED_PROBE_BOOTSTRAP)
        self.assertEqual(
            materializer.CAPTURED_PROBE_BOOTSTRAP.count("source_raw=raw"), 1
        )
        self.assertLess(
            materializer.CAPTURED_PROBE_BOOTSTRAP.index(
                'hashlib.sha256(raw).hexdigest()!=pin'
            ),
            materializer.CAPTURED_PROBE_BOOTSTRAP.index(
                'if not executable: source_raw=raw'
            ),
        )
        self.assertIn(
            "compile(source_raw.decode", materializer.CAPTURED_PROBE_BOOTSTRAP
        )
        compile(materializer.CAPTURED_PROBE_BOOTSTRAP, "<bootstrap>", "exec")
        with tempfile.NamedTemporaryFile(suffix=".sh", delete=False) as handle:
            handle.write(raw)
            path = Path(handle.name)
        self.addCleanup(path.unlink, missing_ok=True)
        completed = subprocess.run(
            ["/bin/bash", "-n", str(path)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr.decode())

    def test_bootstrap_rejects_wrong_campaign_and_extra_fourteen(self) -> None:
        bootstrap.validate_campaign_contract(
            "case00-pair-canary",
            ("shared8-00-base", "shared8-00-full644"),
        )
        with self.assertRaises(bootstrap.R5DRootBootstrapProbeError):
            bootstrap.validate_campaign_contract(
                "full16-production",
                ("shared8-00-base", "shared8-00-full644"),
            )
        with self.assertRaises(bootstrap.R5DRootBootstrapProbeError):
            bootstrap.validate_campaign_contract("case00-pair-canary", static.ALL_TASKS)

    def _plan(self, root: Path) -> dict[str, object]:
        tasks: list[dict[str, object]] = []
        for index in range(8):
            iid, source_sha, instruction_sha = static.EXPECTED_CASES[index]
            for arm in ("base", "full644"):
                video = root / f"outputs/media/case{index:02d}-{arm}.mp4"
                adapter = None if arm == "base" else {
                    "adapter_model_sha256": static.CHECKPOINT_METADATA["adapter_model_sha256"],
                    "checkpoint_manifest": static.CHECKPOINT_METADATA,
                    "checkpoint_root": str(Path(static.CHECKPOINT_METADATA["path"]).parent),
                    "profile": "full644-r64-reference-dpo-preservation-one-pass-v1",
                }
                tasks.append(
                    {
                        "task_id": f"shared8-{index:02d}-{arm}",
                        "case_index": index,
                        "arm": arm,
                        "adapter": adapter,
                        "iid": iid,
                        "instruction": static.EXPECTED_INSTRUCTIONS[index],
                        "instruction_sha256": instruction_sha,
                        "num_inference_steps": 40,
                        "output": {
                            "create_only": True,
                            "video_path": str(video),
                            "receipt_path": str(Path(str(video) + ".receipt.json")),
                        },
                        "seed": 2026 + index,
                        "source_onset_policy": "none",
                        "source_video": (
                            "/vast/users/guangyi.chen/dataset/goku/subject_movement/"
                            f"extracted/videos/{iid}/source.mp4"
                        ),
                        "source_video_sha256": source_sha,
                    }
                )
        release = root / "release"
        value: dict[str, object] = {
            "authority": {
                "exposure_audit": {
                    "path": str(release / "methods/action_editing_baselines/manifests/goku_legacy_shared8_exposure.json"),
                    "sha256": static.RELEASE_FILES["methods/action_editing_baselines/manifests/goku_legacy_shared8_exposure.json"],
                },
                "input_manifest": {
                    "path": str(release / "methods/action_editing_baselines/manifests/goku_legacy_heldout8_inputs.jsonl"),
                    "sha256": static.RELEASE_FILES["methods/action_editing_baselines/manifests/goku_legacy_heldout8_inputs.jsonl"],
                },
                "source_bytes_verified": True,
            },
            "checkpoint_manifest": static.CHECKPOINT_METADATA,
            "claim_limits": {
                "content_disjoint_split": False,
                "evaluation_role": "engineering_diagnostic_only",
                "formal_claim_authorized": False,
                "historical_shared8_exposed": True,
                "human_reviewed_labels": False,
                "iid_heldout_diagnostic": True,
                "iid_overlap_with_full644": 0,
                "scientific_generalization_claim_authorized": False,
            },
            "execution": {
                "all_16_tasks_required_no_cherry_pick": True,
                "external_frozen_runner_attestation_required": True,
                "local_contract_only": True,
                "receipt_contract_alone_cannot_prove_process_execution": True,
                "runner_included": False,
                "training_or_inference_launched": False,
            },
            "producer": {
                "ffprobe_path": "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/runtime/ffprobe_conda_cf_ffmpeg9_hdabad70_r2/runtime/ffprobe",
                "ffprobe_sha256": "356754aa8e327b139dd54dda6846af6425673b73572ab6c1c182ed970f1107f5",
                "infer_lora_path": str(release / "methods/bernini_action_editing/infer_lora.py"),
                "infer_lora_sha256": static.RELEASE_FILES["methods/bernini_action_editing/infer_lora.py"],
                "inference_receipt_schema": "bernini-r-1p3b-action-lora-inference-receipt-v5",
                "method_source_archive_sha256": "12a28ddec99704963af42f1a82b09dff31828e3af8e53e5d0bbd0d43db272828",
                "method_source_revision": "ce4cffc1e8a144448c92252d9fb63087f03bbd8c",
            },
            "schema_version": static.PLAN_SCHEMA,
            "task_count": 16,
            "pair_count": 8,
            "production_ready": True,
            "tasks": tasks,
        }
        value["plan_digest"] = static.object_sha256(value)
        return value

    def test_static_plan_validates_physical16_without_opening_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "outputs/media").mkdir(parents=True)
            value = self._plan(root)
            static.validate_plan(value, root)
            hostile = json.loads(json.dumps(value))
            hostile["tasks"][2]["task_id"] = "shared8-00-base"
            unsigned = dict(hostile)
            unsigned.pop("plan_digest")
            hostile["plan_digest"] = static.object_sha256(unsigned)
            with self.assertRaises(static.R5DStaticProbeError):
                static.validate_plan(hostile, root)
            for mutate in (
                lambda row: row.__setitem__("task_count", 16.0),
                lambda row: row["tasks"][0].__setitem__("seed", 2026.0),
                lambda row: row["checkpoint_manifest"].__setitem__(
                    "global_step", 644.0
                ),
            ):
                hostile = json.loads(json.dumps(value))
                mutate(hostile)
                unsigned = dict(hostile)
                unsigned.pop("plan_digest")
                hostile["plan_digest"] = static.object_sha256(unsigned)
                with self.assertRaises(static.R5DStaticProbeError):
                    static.validate_plan(hostile, root)

    def test_static_release_tree_rejects_extra_and_special_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            release = Path(temporary).resolve() / "release"
            for relative in static.RELEASE_FILES:
                target = release / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"fixture\n")
            static.validate_release_tree(release)
            extra = release / "methods/bernini_action_editing/extra.py"
            extra.write_bytes(b"extra\n")
            with self.assertRaises(static.R5DStaticProbeError):
                static.validate_release_tree(release)
            extra.unlink()
            fifo = release / "methods/bernini_action_editing/hostile.fifo"
            os.mkfifo(fifo)
            with self.assertRaises(static.R5DStaticProbeError):
                static.validate_release_tree(release)
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            real = parent / "real"
            for relative in static.RELEASE_FILES:
                target = real / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"fixture\n")
            linked = parent / "release"
            linked.symlink_to(real, target_is_directory=True)
            with self.assertRaises(static.R5DStaticProbeError):
                static.validate_release_tree(linked)

    def _launch_receipt(self, *, selected: list[str], campaign: str) -> tuple[dict[str, object], dict[str, object], dict[str, int]]:
        input_value = {
            "holder_job_id": "143812",
            "expected_node": "auh7-1b-gpu-293",
            "plan": "/release/plan",
            "output_report": "/fresh/report",
            "runner_attestation": "/fresh/attestation",
            "model_root": "/external/model",
            "bernini_root": "/external/bernini",
            "veomni_root": "/external/veomni",
            "authority_root": "/fresh/authority",
            "rank_cache_root": "/fresh/rank-cache",
        }
        identity = {
            "device": 1, "inode": 2, "uid": 3, "gid": 4, "mode": 0o100444,
            "nlink": 1, "rdev": 0, "size": 10, "blocks": 8,
            "mtime_ns": 11, "ctime_ns": 12,
        }
        local_roles = {
            "runner": "methods/bernini_action_editing/full644_exploratory_matched_runner_auh_r5.py",
            "bridge": "methods/bernini_action_editing/full644_exploratory_matched_torchrun_fd_bridge_v2.py",
            "adapter": "methods/bernini_action_editing/full644_exploratory_matched_infer_adapter_auh_r5d.py",
            "base_adapter": "methods/bernini_action_editing/full644_exploratory_matched_infer_adapter_v2.py",
            "eval_v1": "methods/bernini_action_editing/full644_exploratory_matched_eval_v1.py",
            "eval_v2": "methods/bernini_action_editing/full644_exploratory_matched_eval_v2.py",
            "model_authority": "methods/bernini_action_editing/action_preservation_decoded_eval_model_authority_v2.py",
            "model_manifest": "methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256",
        }
        roles = {}
        for role, relative in local_roles.items():
            input_value[role] = f"/release/{role}"
            roles[role] = {
                "path": input_value[role], "sha256": static.RELEASE_FILES[relative],
                "identity": identity,
            }
        for role, digest in static.EXTERNAL_IDENTITY_PINS.items():
            input_value[role] = f"/external/{role}"
            roles[role] = {
                "path": input_value[role], "sha256": digest, "identity": identity,
            }
        roles["plan"] = {
            "path": input_value["plan"], "sha256": "b" * 64, "identity": identity,
        }
        release = {
            "schema_version": static.RELEASE_SCHEMA,
            "entry_mode": "trusted_stdin",
            "external_root_of_trust": "trusted-controller-streamed-stdin-bytes",
            "bash_path": "/bin/bash",
            "bash_privileged_mode": True,
            "slurm_export_none": True,
            "python_is_executed_from_held_fd": True,
            "runner_is_compiled_from_captured_fd_bytes": True,
            "named_payload_execution_forbidden": True,
            "expected_allocation_gpu_count": 8,
            "campaign_mode": campaign,
            "selected_task_ids": selected,
            "formal_full16_report": campaign == "full16-production",
            "canary_stops_after_pair_for_manual_visual_review": campaign == static.CAMPAIGN,
            "holder_job_id": "143812",
            "expected_node": "auh7-1b-gpu-293",
            "identities": roles,
            "slurm_environment_contract": {
                "required_source_names": [
                    "SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_GPUS_ON_NODE",
                    "SLURM_GPUS_PER_NODE", "SLURM_STEP_GPUS", "SLURM_NNODES",
                    "SLURM_STEP_NUM_NODES", "SLURM_JOB_NODELIST", "SLURM_STEP_NODELIST",
                ],
                "required_absent_names": ["SLURM_JOB_GPUS", "SLURM_JOB_NUM_NODES"],
                "caller_synthesized_slurm_facts_forbidden": True,
            },
            "runner_arguments": [],
        }
        release["runner_arguments"] = static.expected_runner_arguments(
            input_value, roles
        )
        if campaign != static.CAMPAIGN:
            release["runner_arguments"][:2] = ["--campaign-mode", campaign]
        receipt: dict[str, object] = {
            "schema_version": static.RECEIPT_SCHEMA,
            "status": "MATERIALIZED_NOT_SUBMITTED",
            "launch_input": {"identity": identity},
            "release": release,
            "release_digest": static.object_sha256(release),
            "root_bootstrap_sha256": static.ROOT_BOOTSTRAP_SHA256,
            "payload_path": "/release/payload",
            "payload_sha256": "d" * 64,
            "payload_size": 123,
            "payload_mode": 0o444,
            "receipt_path": "/release/receipt",
            "required_entry": (
                "trusted controller: srun --export=NONE /bin/bash -p -s < <payload>"
            ),
            "named_payload_execution_forbidden": True,
            "submission_or_execution_performed": False,
            "remote_execution_authorized_by_this_receipt": False,
        }
        receipt["receipt_digest"] = static.object_sha256(receipt)
        return receipt, input_value, identity

    def test_static_receipt_rejects_full16_and_selection_expansion(self) -> None:
        good, value, identity = self._launch_receipt(
            selected=list(static.SELECTED), campaign=static.CAMPAIGN
        )
        static.validate_launch_receipt(good, value, identity, "b" * 64)
        truncated = json.loads(json.dumps(good))
        truncated["release"]["runner_arguments"].pop()
        truncated["release_digest"] = static.object_sha256(truncated["release"])
        body = dict(truncated)
        body.pop("receipt_digest")
        truncated["receipt_digest"] = static.object_sha256(body)
        with self.assertRaises(static.R5DStaticProbeError):
            static.validate_launch_receipt(truncated, value, identity, "b" * 64)
        bool_alias = json.loads(json.dumps(good))
        bool_alias["release"]["slurm_environment_contract"][
            "caller_synthesized_slurm_facts_forbidden"
        ] = 1
        bool_alias["release_digest"] = static.object_sha256(bool_alias["release"])
        body = dict(bool_alias)
        body.pop("receipt_digest")
        bool_alias["receipt_digest"] = static.object_sha256(body)
        with self.assertRaises(static.R5DStaticProbeError):
            static.validate_launch_receipt(bool_alias, value, identity, "b" * 64)
        for campaign, selected in (
            ("full16-production", list(static.ALL_TASKS)),
            (static.CAMPAIGN, list(static.ALL_TASKS)),
        ):
            hostile, value, identity = self._launch_receipt(
                selected=selected, campaign=campaign
            )
            with self.assertRaises(static.R5DStaticProbeError):
                static.validate_launch_receipt(hostile, value, identity, "b" * 64)

    def test_diagnostics_do_not_import_torch(self) -> None:
        for path in (
            MODULE_ROOT / "full644_exploratory_matched_r5d_root_bootstrap_probe_runner_v1.py",
            MODULE_ROOT / "full644_exploratory_matched_r5d_static_nomodel_probe_v1.py",
        ):
            raw = path.read_text(encoding="utf-8")
            self.assertNotIn("import torch", raw)
            self.assertNotIn("from torch", raw)


if __name__ == "__main__":
    unittest.main()
