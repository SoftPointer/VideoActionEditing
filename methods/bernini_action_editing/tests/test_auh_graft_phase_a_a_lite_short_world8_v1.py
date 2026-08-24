from __future__ import annotations

import ast
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tarfile
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
PLAN = METHOD_ROOT / "assets/graft_phase_a_a_lite_short_world8_plan_v1.json"
LAUNCHER = METHOD_ROOT / "scripts/auh_run_graft_phase_a_a_lite_short_world8_v1.sbatch"
SUBMIT = METHOD_ROOT / "scripts/auh_submit_graft_phase_a_a_lite_short_world8_v1.sh"
RUNNER = METHOD_ROOT / "run_graft_phase_a_a_lite_short_gpu_v1.py"
CHECKPOINT_MANIFEST = (
    METHOD_ROOT / "audits/bernini_r13_ff4c5d4_checkpoint.sha256"
)

PLAN_SHA = "ab2eb2e7b93341b47498184821761eb8e5c924f9dd8460284087e23a27ba34d8"
RUNNER_SHA = "4b98bc520c7b90f71a3fe1d58e5e2e2f96d05465611f4c4bb4143e6cc51a62c4"
CHECKPOINT_MANIFEST_SHA = (
    "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
)
MATERIALIZER_SHA = (
    "4686463642a38e771c6858d1c10fc6aacb815a56e4f3eae951a336018d186cf4"
)
REAL_RELEASE_STEM = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_graft_v1_20260810/a_lite_source_release/"
    "10d9798_0c0aff6_c4_r1/outputs/canary4/graft_a_lite_source_canary4"
)
EXPORT_NAMES = (
    "GRAFT_SHORT_SOURCE_ARCHIVE",
    "GRAFT_SHORT_SOURCE_ARCHIVE_SHA256",
    "GRAFT_SHORT_RUNTIME_CLOSURE_MANIFEST",
    "GRAFT_SHORT_RUNTIME_CLOSURE_MANIFEST_SHA256",
    "GRAFT_SHORT_PYTHON_BIN",
    "GRAFT_SHORT_PYTHON_SHA256",
    "BERNINI_OFFICIAL_ROOT",
    "BERNINI_VEOMNI_ROOT",
    "BERNINI_ACTION_CHECKPOINT",
    "BERNINI_CHECKPOINT_CONTENT_MANIFEST",
    "BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256",
    "GRAFT_SHORT_PLAN",
    "GRAFT_SHORT_PLAN_SHA256",
    "GRAFT_SHORT_TERMINAL_ADMISSION",
    "GRAFT_SHORT_TERMINAL_ADMISSION_SHA256",
    "GRAFT_SHORT_TERMINAL_MATERIALIZER_RUNTIME_SHA256",
    "GRAFT_SHORT_OUTPUT_ROOT",
    "GRAFT_SHORT_LAUNCHER_SOURCE",
    "GRAFT_SHORT_LAUNCHER_SHA256",
)
AUTHORITY_FIELDS = (
    "action_authority",
    "identity_authority",
    "cross_clip_identity_authority",
    "quality_authority",
    "training_authority",
    "checkpoint_authority",
    "publication_authority",
    "production_authority",
    "data_governance_authority",
    "data_license_authority",
    "scientific_success_claimed",
    "semantic_action_editing_success_claimed",
)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def seal(value: dict[str, object], *, field: str = "digest") -> dict[str, object]:
    output = dict(value)
    output[field] = sha(canonical(output))
    return output


def false_authority() -> dict[str, bool]:
    return {name: False for name in AUTHORITY_FIELDS}


def marked_python_source(start_marker: str, end_marker: str) -> str:
    source = LAUNCHER.read_text(encoding="utf-8")
    start = source.index(start_marker)
    end = source.index(end_marker, start) + len(end_marker)
    return source[start:end]


def embedded_closure() -> dict[str, str]:
    source = LAUNCHER.read_text(encoding="utf-8")
    anchor = source.index('commit = "a884d357a6c0742f751be48d226ba72c952bae76"')
    tree = ast.parse(source[source.index("expected = {", anchor) : source.index("\n\ndef digest", anchor)])
    assignment = tree.body[0]
    assert isinstance(assignment, ast.Assign)
    value = ast.literal_eval(assignment.value)
    assert isinstance(value, dict)
    return value


def local_recursive_closure() -> set[str]:
    files = {
        path.relative_to(METHOD_ROOT).as_posix(): path
        for path in METHOD_ROOT.rglob("*.py")
    }
    queue = [RUNNER.name]
    seen: set[str] = set()
    while queue:
        relative = queue.pop(0)
        if relative in seen:
            continue
        seen.add(relative)
        tree = ast.parse(files[relative].read_bytes(), filename=relative)
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                for candidate in (
                    name.replace(".", "/") + ".py",
                    name.split(".")[0] + ".py",
                ):
                    if candidate in files and candidate not in seen:
                        queue.append(candidate)
    return seen


def run_archive_preflight(
    test: unittest.TestCase,
    *,
    omit: str | None = None,
    mutate: str | None = None,
    extra_kind: str | None = None,
) -> subprocess.CompletedProcess[str]:
    temporary = tempfile.TemporaryDirectory()
    test.addCleanup(temporary.cleanup)
    root = Path(temporary.name)
    archive = root / "runtime.tar"
    manifest = root / "closure.json"
    destination = root / "source-tree"
    destination.mkdir()
    expected = embedded_closure()
    value = {
        "files": expected,
        "root": "methods/bernini_action_editing",
        "schema_version": "bernini-graft-phase-a-short-runtime-python-closure-v2",
        "selection": "commit-a884d35-recursive-runtime-import-closure-v2",
        "source_git_commit": "a884d357a6c0742f751be48d226ba72c952bae76",
    }
    manifest.write_bytes(canonical(value) + b"\n")
    with tarfile.open(archive, "w") as handle:
        for relative in sorted(expected):
            if relative == omit:
                continue
            payload = (METHOD_ROOT / relative).read_bytes()
            if relative == mutate:
                payload += b"\n# hostile mutation\n"
            member = tarfile.TarInfo(
                "methods/bernini_action_editing/" + relative
            )
            member.mode = 0o444
            member.size = len(payload)
            handle.addfile(member, io.BytesIO(payload))
        if extra_kind == "directory":
            member = tarfile.TarInfo("methods/bernini_action_editing/__pycache__/")
            member.type = tarfile.DIRTYPE
            handle.addfile(member)
        elif extra_kind == "pycache":
            payload = b"shadow"
            member = tarfile.TarInfo(
                "methods/bernini_action_editing/__pycache__/shadow.pyc"
            )
            member.size = len(payload)
            handle.addfile(member, io.BytesIO(payload))
        elif extra_kind == "symlink":
            # Replace one regular member to preserve the exact member count.
            pass
    if extra_kind == "symlink":
        replacement = root / "symlink.tar"
        victim = sorted(expected)[0]
        with tarfile.open(archive, "r") as old, tarfile.open(replacement, "w") as new:
            for member in old.getmembers()[1:]:
                source = old.extractfile(member)
                assert source is not None
                payload = source.read()
                clone = tarfile.TarInfo(member.name)
                clone.mode = member.mode
                clone.size = len(payload)
                new.addfile(clone, io.BytesIO(payload))
            link = tarfile.TarInfo("methods/bernini_action_editing/" + victim)
            link.type = tarfile.SYMTYPE
            link.linkname = "/tmp/attacker"
            new.addfile(link)
        replacement.replace(archive)
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-",
            str(archive),
            str(manifest),
            str(destination),
            sha(archive.read_bytes()),
            sha(manifest.read_bytes()),
        ],
        input=marked_python_source(
            "# BEGIN GRAFT_PHASE_A_SHORT_ARCHIVE_PREFLIGHT_V1",
            "# END GRAFT_PHASE_A_SHORT_ARCHIVE_PREFLIGHT_V1",
        ),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )


class PlanAndLauncherStaticTests(unittest.TestCase):
    def test_shell_syntax_privileged_boundary_and_fixed_scheduler(self) -> None:
        for path in (LAUNCHER, SUBMIT):
            result = subprocess.run(
                ["/bin/bash", "-n", str(path)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(path.read_bytes().startswith(b"#!/bin/bash -p\n"))
        launcher = LAUNCHER.read_text(encoding="utf-8")
        submit = SUBMIT.read_text(encoding="utf-8")
        launcher_sha = sha(LAUNCHER.read_bytes())
        for fragment in (
            "#SBATCH --gres=gpu:mi210:8",
            "--nproc-per-node=8",
            "WORLD8=DP2xSP4",
            "--max-restarts=0",
            "--redirects=3",
            '[[ "$#" -eq 0 ]]',
            "/usr/bin/env -i",
            "pass_fds=(launcher_fd,sbatch_fd)",
            'exec 8<"${output_root}"',
            'exec 9<"${rank_log_root}"',
            "8<&- 9<&- &",
            "root_fd=os.dup(inherited_root_fd)",
            "log_fd=os.dup(inherited_log_fd)",
            "completed_0_0_observed_before_sbatch",
            "trust_anchor_computed_from_receipt\":False",
        ):
            self.assertIn(fragment, launcher + submit)
        self.assertNotIn("--export=ALL", submit)
        self.assertIn(f"readonly required_launcher_sha256={launcher_sha}", submit)
        self.assertEqual(submit.count(launcher_sha), 2)
        self.assertNotIn("PASS WORLD8=DP2xSP4", launcher)
        self.assertIn(
            "provisional_non_success_inode_may_survive_cleanup_failure",
            launcher + submit,
        )
        self.assertIn(
            "failure_absence_of_any_receipt_inode_guaranteed", launcher + submit
        )
        self.assertIn(
            "readonly export_names_csv=" + ",".join(EXPORT_NAMES), submit
        )
        self.assertIn("exactly twenty short-run interface variables", submit)
        self.assertIn("GRAFT_SHORT_SUBMIT_WRAPPER_SHA256", submit)
        self.assertNotIn(
            "readonly export_names_csv="
            + ",".join((*EXPORT_NAMES, "GRAFT_SHORT_SUBMIT_WRAPPER_SHA256")),
            submit,
        )

    def test_plan_is_canonical_live_and_matches_runner_source_constants(self) -> None:
        raw = PLAN.read_bytes()
        value = json.loads(raw)
        self.assertEqual(raw, canonical(value) + b"\n")
        self.assertEqual(sha(raw), PLAN_SHA)
        self.assertEqual(value["topology"], {
            "allocation": "single-node-8xMI210",
            "dp_size": 2,
            "sp_size": 4,
            "world_size": 8,
        })
        self.assertEqual(value["update_indices"], [29, 38])
        self.assertEqual(value["confirmation_indices"], [29, 38])
        self.assertEqual(value["adapter_off_parity_indices"], [0, 25])
        self.assertEqual(
            value["families"],
            [
                {
                    "confirmation_iid": "841b5e0080a1441d",
                    "dp_arm": 0,
                    "family": "dog",
                    "fit_iid": "7b88a1ca1f804f41",
                    "same_family_wrong_iid": "7b88a1ca1f804f41",
                },
                {
                    "confirmation_iid": "a66e6818e4144928",
                    "dp_arm": 1,
                    "family": "human",
                    "fit_iid": "a35b590961d24694",
                    "same_family_wrong_iid": "a35b590961d24694",
                },
            ],
        )
        runtime = value["runtime"]
        self.assertEqual(runtime["runner_sha256"], RUNNER_SHA)
        self.assertEqual(
            runtime["source_commit"],
            "a884d357a6c0742f751be48d226ba72c952bae76",
        )
        self.assertEqual(
            runtime["world8_result_schema"],
            "bernini-graft-phase-a-short-world8-result-set-v2",
        )
        self.assertEqual(
            runtime["world8_full_results_schema"],
            "bernini-graft-phase-a-short-world8-full-results-v1",
        )
        self.assertEqual(value["authority"], false_authority())
        self.assertTrue(value["no_checkpoint"])
        release = value["release"]
        self.assertEqual(release["logical_output_stem"], REAL_RELEASE_STEM)
        self.assertEqual(
            release["artifacts"]["manifest"]["path"],
            REAL_RELEASE_STEM + ".manifest.jsonl",
        )
        self.assertEqual(
            release["artifacts"]["producer"]["path"],
            REAL_RELEASE_STEM + ".receipt.json",
        )
        self.assertEqual(
            release["artifacts"]["execution"]["path"],
            REAL_RELEASE_STEM + ".execution.receipt.json",
        )
        self.assertEqual(
            release["artifacts"]["submission"]["path"],
            REAL_RELEASE_STEM + ".submission.receipt.json",
        )
        self.assertEqual(
            release["terminal_admission"]["path"],
            REAL_RELEASE_STEM + ".terminal.admission.receipt.json",
        )
        launcher = LAUNCHER.read_text(encoding="utf-8")
        for fragment in (
            "plan does not match runner constants",
            "runner.ACTION_INSTRUCTION_BY_DP_ARM",
            "consumer.NOOP_INSTRUCTION",
            "runner.UPDATE_INDICES",
            "runner.CONFIRMATION_INDICES",
            "runner.ADAPTER_OFF_PARITY_INDICES",
            "runner.assert_pinned_dependencies()",
            PLAN_SHA,
            REAL_RELEASE_STEM,
            'terminal_path!=stem+".terminal.admission.receipt.json"',
        ):
            self.assertIn(fragment, launcher)

    def test_no_stale_source_pins_or_v1_result_schema_residue(self) -> None:
        combined = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (PLAN, LAUNCHER, SUBMIT, Path(__file__))
        )
        for forbidden in (
            "2a" + "72af5258923d4648c4dd2d3ff69015d158460c",
            "commit-" + "2a72af5-recursive-runtime-import-closure-v2",
            "8a" + "a9d83b1a25576fb56274bfb25bf8607731a584863782c49c92a81e30440f3c",
            "91" + "f8808bc0d29b0624b623a5e4d1f4b9717a2d294d80379d10fc69425d4ee330",
            "d3" + "236fbfa72d851ef2eb096a88e66c96208b3219bd779edfb677107c359bea5e",
            "b8" + "1327b7bab615e2c539241a65e5a2e5d3a9c423",
            "98" + "37f2175b58ced0b454e14a77684c73c861d78db5e89d0b9b6bff027fff1f67",
            "26" + "17f62a9b206960cd09e87cfd52b6855486fcb47777214a9f31878d977dc98c",
            "98" + "4dc8e9c7865e72a0e3d3d254cf244b1ba23e610bbee6eac11d9560d162551d",
            "f9" + "1328925b99c6741c45b42739c1e3e64f486bf7546c7c024a85dd3470fe20fc",
            "00" + "fe1ae910a947a3660ae73b7203e4495623965c2192eff7dca5ebbf0f098a0f",
            "be" + "6987ca1d39a9450f4f08259a925adae4f7767f",
            "commit-" + "be6987c-recursive-runtime-import-closure-v2",
            "d7" + "78a56a078468fcd4554df9e9e77c74bf3330a428c10a81b94c7bf8b111ea94",
            "f4" + "6a9d79d982b43c7aca9f05f70af1a4e804252f1315a581f5f0ed1e1ddf11b0",
            "21" + "c978fe8c0292735d300106921aba3645c6b916b6a8f0e0d54134f9b091f37b",
            "aa" + "850c8d357f9a055aca175e292003ef097f900d77281827d493880f72c388ef",
            "6f" + "bfb5efc816543ea15fa0ed6fe6bac6f5c1279f9eb9b289ec1a73bc7b0050b8",
            "9c" + "64d3179a689ca17994b6fcdd4c2466fcca16a810e6cf7d591b51b3502d2ae9",
            "44" + "c8560",
            "253" + "152",
            "d7" + "fab4a04d83796b407cf6b659d4c41db52e1fe2",
            "d1" + "d2e6af019acb4a9a6c8fd7ce0d6ecf49bb7bc3ca13a2e04aa76f611e791c11",
            "ea" + "1b5d16b74f8eae8c11884bf957484ed1b25d03635ce63e194af1475b2e20fd",
            "d6" + "e2695645f9043cd02387a3e6154639c5a416a2726df930f7da6ce5a28b1584",
            "93" + "901a111ed1099c6ef3decc3a446c6cb46f1752",
            "21" + "c27af968b4dd4503845d97cc5da9ad02770da9ab8ec2b6ab057ae78abdea84",
            "51" + "fbdf03afc4e336f3163de75ad55aa670e03ffb0ff6bf443ca301212b1a388f",
            "d2" + "9561947de707c31e0af0bf818147342b368b823dbd8df5aa48a7c9bf7e6245",
            "369" + "f65efb12ba1e2e12af01f1e9dea25016d6ec3",
            "commit-" + "369f65e-recursive-runtime-import-closure-v2",
            "9da" + "60769b0717707427f50ae58125f16b9dc497ae278a8c3cd81566e701fa10e",
            "19c" + "4b32778afedc586b1ba63d9376ce134afef9b62eab8d13b0cbe1d83d0a764",
            "af6" + "7ab21ca05985053e66d7cbab00f6498b41b2925afefcef6553e01dd4d0e7d",
            "27d" + "d5c43ee9d77bd3e271157e32aebc324d5504cb187de0d5251e53cb1257520",
            "source-" + "369f65e-launcher-59e3126-r1",
            "40d" + "23c9b1a63a3f09bd4fa559b083d90b9526fba",
            "commit-" + "40d23c9-recursive-runtime-import-closure-v2",
            "012" + "ba28ee2292f6218b42a2cd287800affdcab83edd0cdf9d3e4ad7bfdafc443",
            "e54" + "d064fc59858c39bc43341a1281b6f8746aeb74cbb7985ffce7890257244b0",
            "bd1" + "87e9508fc49111f0ba6d502124dc269ac95c2268e41593d23abbe15918f82",
            "23d" + "1d064358696cb662fd71ddea4f4455538c1017df1127d3fbc3894230528e1",
            "e09" + "29465f3c0367e9c5f2ba8988d9c0473e641d616efde4f983d44e499987f82",
            "7c1" + "e909e96e2b35fe3227b880667d026d024ac48de9be94843ab25db8d53e88f",
            "90d" + "4f58ae42aba5febc06c33a5ffd65e8be92f56438deb1e90448137e2ba6d05",
            "755" + "25802806d06b4e518582f1b1b560c8e4732a4",
            "source-" + "40d23c9-launcher-7552580-r2",
            "bernini-graft-phase-a-short-world8-result-set-" + "v1",
            "bernini-graft-phase-a-a-lite-short-world8-parent-" + "v1",
        ):
            self.assertNotIn(forbidden, combined)
        self.assertIn(
            "export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True", combined
        )
        self.assertIn(
            "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", combined
        )

    def test_complete_recursive_closure_is_23_exact_regular_members(self) -> None:
        expected = embedded_closure()
        self.assertEqual(len(expected), 23)
        static = local_recursive_closure()
        self.assertEqual(
            static,
            set(expected) - {
                "tools/materialize_vae.py",
                "tools/build_renderer_dataset.py",
            },
        )
        for relative, wanted in expected.items():
            self.assertEqual(sha((METHOD_ROOT / relative).read_bytes()), wanted)
        self.assertEqual(expected[RUNNER.name], RUNNER_SHA)
        launcher = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("exactly 23 safe regular members and no directories", launcher)
        self.assertIn("complete recursive static closure differs", launcher)
        self.assertIn("dynamic source closure edge differs", launcher)

    def test_archive_preflight_accepts_exact23_and_rejects_hostile_members(self) -> None:
        result = run_archive_preflight(self)
        self.assertEqual(result.returncode, 0, result.stderr)
        for kwargs in (
            {"omit": RUNNER.name},
            {"mutate": RUNNER.name},
            {"extra_kind": "directory"},
            {"extra_kind": "pycache"},
            {"extra_kind": "symlink"},
        ):
            with self.subTest(**kwargs):
                result = run_archive_preflight(self, **kwargs)
                self.assertNotEqual(result.returncode, 0)

    def test_terminal_is_external_pre_submission_pin_not_derived(self) -> None:
        plan = json.loads(PLAN.read_bytes())
        terminal = plan["release"]["terminal_admission"]
        materializer = plan["release"]["terminal_materializer"]
        self.assertEqual(terminal["sha256_source"], "external-pre-submission-pin")
        self.assertEqual(materializer["runtime_sha256_source"], "external-pre-submission-pin")
        self.assertEqual(materializer["implementation_sha256"], MATERIALIZER_SHA)
        submit = SUBMIT.read_text(encoding="utf-8")
        self.assertIn("external terminal trust anchor", submit)
        self.assertIn("external materializer runtime trust anchor", submit)
        self.assertNotIn("terminal_sha=digest(terminal_raw)", submit)


FIELD_ROLES = (
    "source_noop_target_velocity",
    "correct_atlas_noop_velocity",
    "wrong_atlas_noop_velocity",
    "dropped_atlas_noop_velocity",
    "correct_atlas_action_velocity",
    "dropped_atlas_action_velocity",
)
PROVENANCE_TRUE = (
    "exactly_six_fields",
    "all_fields_detached_fp32_finite_contiguous",
    "all_field_storages_pairwise_disjoint",
    "same_confirmation_source_zs_bytes_all_fields",
    "same_native_full_source_v_pack_bytes_all_model_fields",
    "same_noisy_target_object_and_bytes_all_fields",
    "same_epsilon_bytes_all_fields",
    "same_sigma_timestep_coordinate_all_fields",
    "same_negative_condition_bytes_all_model_fields",
    "correct_atlas_from_confirmation_row",
    "wrong_atlas_from_same_family_fit_row",
    "wrong_intervention_changes_only_identity_atlas_memory",
    "drop_intervention_disables_only_identity_rebinder_residual_and_atlas_route",
    "drop_retains_native_full_source_v_pack",
    "noop_positive_condition_shared_across_correct_wrong_drop",
    "action_positive_condition_shared_across_correct_drop",
    "action_noop_pair_differs_only_in_positive_text_embedding",
    "source_noop_target_velocity_recomputed_from_same_x_sigma_and_source_zs",
    "native_v2v_apg_field_formula_used",
    "same_state_tensor_identities_recomputed_byte_equal",
    "wrong_route_receipts_differ_only_in_atlas_memory",
    "drop_route_receipts_retain_v_branch_disable_only_rebinder",
    "action_noop_route_receipts_equal_with_negative_raw_reuse",
)
PROVENANCE_FALSE = (
    "confirmation_row_consumed_by_optimizer",
    "wrong_atlas_is_cross_family",
    "native_source_v_pack_dropped",
    "negative_condition_changed_by_intervention",
    "noise_or_coordinate_changed_by_intervention",
    "target_video_used",
    "generated_proposal_used",
    "t2v_branch_used",
    "source_retelling_used",
    "selector_used",
    "mask_pose_track_flow_or_motion_donor_used",
)
FIT_IIDS = ("7b88a1ca1f804f41", "a35b590961d24694")
CONFIRMATION_IIDS = ("841b5e0080a1441d", "a66e6818e4144928")
FAMILIES = ("dog", "human")


def identity(
    label: str,
    *,
    dtype: str = "torch.float32",
    shape: list[int] | None = None,
    device: str = "cuda",
) -> dict[str, object]:
    observed_shape = [1, 2] if shape is None else shape
    element_bytes = {
        "torch.bfloat16": 2,
        "torch.float32": 4,
        "torch.int64": 8,
    }[dtype]
    elements = 1
    for dimension in observed_shape:
        elements *= dimension
    return {
        "shape": observed_shape,
        "dtype": dtype,
        "device_type_at_observation": device,
        "finite": True,
        "byte_count": elements * element_bytes,
        "raw_sha256": sha((label + ":raw").encode()),
        "content_sha256": sha((label + ":content").encode()),
    }


def metric(arm: int, index: int, *, hard_gate: bool = True) -> dict[str, object]:
    correct_loss = 0.75
    wrong_loss = 1.0 if hard_gate else 0.75
    dropped_loss = 1.0
    wrong_gain = (wrong_loss - correct_loss) / wrong_loss
    dropped_gain = (dropped_loss - correct_loss) / dropped_loss
    return seal(
        {
            "schema_version": "bernini-graft-phase-a-confirmation-metrics-v1",
            "schedule_index": index,
            "field_roles": list(FIELD_ROLES),
            "field_shape": [1, 2],
            "field_dtype": "torch.float32",
            "field_device_type": "cuda",
            "field_tensor_sha256": {
                role: identity(f"field:{arm}:{index}:{role}")["raw_sha256"]
                for role in FIELD_ROLES
            },
            "noop_fm_loss_float64_hex": {
                "correct_atlas": correct_loss.hex(),
                "wrong_atlas": wrong_loss.hex(),
                "dropped_atlas": dropped_loss.hex(),
            },
            "relative_gain_formula": (
                "(L_control-L_correct)/max(L_control,float64_tiny)"
            ),
            "relative_gain_float64_hex": {
                "correct_vs_wrong": wrong_gain.hex(),
                "correct_vs_drop": dropped_gain.hex(),
            },
            "minimum_relative_gain_float64_hex": float(1.0e-4).hex(),
            "action_delta_formula": "v_action-v_noop",
            "action_delta_norm_float64_hex": {
                "correct_atlas": float(0.75).hex(),
                "dropped_atlas": float(1.0).hex(),
            },
            "action_delta_correct_drop_norm_ratio_formula": (
                "norm(delta_correct)/max(norm(delta_drop),float64_tiny)"
            ),
            "action_delta_correct_drop_norm_ratio_float64_hex": float(0.75).hex(),
            "minimum_action_delta_correct_drop_norm_ratio_float64_hex": float(0.5).hex(),
            "action_delta_correct_drop_cosine_float64_hex": float(0.25).hex(),
            "minimum_action_delta_cosine_float64_hex": float(0.0).hex(),
            "float64_tiny_hex": float(2.2250738585072014e-308).hex(),
            "noncompensating_gates": {
                "correct_vs_wrong_noop_relative_gain": hard_gate,
                "correct_vs_drop_noop_relative_gain": True,
                "action_delta_correct_drop_norm_ratio": True,
                "action_delta_correct_drop_cosine": True,
            },
            "noncompensating_all_pass": hard_gate,
            "metrics_computed_from_six_detached_fields_by_this_core": True,
            "field_origin_same_noise_state_coordinate_verified_by_this_core": False,
            **false_authority(),
        }
    )


def route_receipts(rank: int, index: int) -> dict[str, dict[str, object]]:
    arm = rank // 4
    sigma_hex, gate_hex = {
        29: ("0x1.4fa88c0000000p-1", "0x1.7f027f7ae326bp-4"),
        38: ("0x1.b166a20000000p-3", "0x1.0000000000000p+0"),
    }[index]
    base = {
        "branch_name": "V",
        "total_tokens": 42,
        "condition_tokens": 21,
        "target_tokens": 21,
        "sequence_parallel_rank": rank % 4,
        "sequence_parallel_size": 4,
        "sigma_hex": sigma_hex,
        "source_memory_owned_by_V_VI_only": True,
    }
    correct = seal(
        {
            **base,
            "enabled": True,
            "gate_hex": gate_hex,
            "atlas_receipt_digest": sha(f"correct:{arm}:{index}".encode()),
        }
    )
    wrong = seal(
        {
            **base,
            "enabled": True,
            "gate_hex": gate_hex,
            "atlas_receipt_digest": sha(f"wrong:{arm}:{index}".encode()),
        }
    )
    drop = seal(
        {
            **base,
            "enabled": False,
            "gate_hex": float(0.0).hex(),
            "atlas_receipt_digest": None,
        }
    )
    return {
        "correct_negative": dict(correct),
        "correct_noop": dict(correct),
        "correct_action": dict(correct),
        "wrong_negative": dict(wrong),
        "wrong_noop": dict(wrong),
        "drop_negative": dict(drop),
        "drop_noop": dict(drop),
        "drop_action": dict(drop),
    }


def provenance(rank: int, index: int) -> dict[str, object]:
    arm = rank // 4
    same_state = {
        name: identity(f"same:{arm}:{index}:{name}")
        for name in (
            "confirmation_source_zs",
            "epsilon",
            "noisy_target_x_sigma",
            "native_visual_pack",
            "native_rotary_pack",
            "sigma",
            "timestep",
            "negative_condition",
            "noop_positive_condition",
            "action_positive_condition",
        )
    }
    same_state["sigma"] = identity(
        f"same:{arm}:{index}:sigma", shape=[], device="cpu"
    )
    same_state["timestep"] = identity(
        f"same:{arm}:{index}:timestep", dtype="torch.int64", shape=[1]
    )
    correct_atlas = identity(f"correct-atlas:{arm}:{index}", dtype="torch.float32")
    wrong_atlas = identity(f"wrong-atlas:{arm}:{index}", dtype="torch.float32")
    routes = route_receipts(rank, index)
    return seal(
        {
            "schema_version": "bernini-graft-phase-a-short-six-field-provenance-v1",
            "schedule_index": index,
            "confirmation_iid": CONFIRMATION_IIDS[arm],
            "confirmation_source_sha256": sha(
                f"confirmation-source:{arm}".encode()
            ),
            "wrong_owner_iid": FIT_IIDS[arm],
            "wrong_owner_source_sha256": sha(f"fit-source:{arm}".encode()),
            "field_roles": list(FIELD_ROLES),
            "field_tensor_identities": {
                role: identity(f"field:{arm}:{index}:{role}")
                for role in FIELD_ROLES
            },
            "same_state_identities_before_model_fields": same_state,
            "same_state_identities_after_all_fields": dict(same_state),
            "confirmation_source_state_receipt_digest": sha(
                f"confirmation-state:{arm}".encode()
            ),
            "wrong_fit_source_state_receipt_digest": sha(
                f"fit-state:{arm}".encode()
            ),
            "coordinate": seal(
                {
                    "schema_version": "bernini-graft-phase-a-short-coordinate-v1",
                    "schedule_index": index,
                    "timestep": {29: 655, 38: 211}[index],
                    "sigma_float32_be_hex": {29: "3f27d446", 38: "3e58b351"}[
                        index
                    ],
                    "schedule_sha256": "3e5ad4473d133318026cc9e8f32399782bf06313691b58870c89d9c4c87c3d03",
                    "scheduler_step_called": False,
                }
            ),
            "epsilon": same_state["epsilon"],
            "epsilon_receipt_digest": sha(f"epsilon:{arm}:{index}".encode()),
            "noisy_target": same_state["noisy_target_x_sigma"],
            "noisy_target_receipt_digest": sha(f"noisy:{arm}:{index}".encode()),
            "confirmation_source_zs": same_state["confirmation_source_zs"],
            "native_visual_pack": same_state["native_visual_pack"],
            "native_rotary_pack": same_state["native_rotary_pack"],
            "negative_condition": same_state["negative_condition"],
            "noop_positive_condition": same_state["noop_positive_condition"],
            "action_positive_condition": same_state["action_positive_condition"],
            "correct_atlas": correct_atlas,
            "wrong_atlas": wrong_atlas,
            "atlas_identities_before_model_fields": {
                "correct_confirmation_atlas": correct_atlas,
                "wrong_same_family_fit_atlas": wrong_atlas,
            },
            "atlas_identities_after_all_fields": {
                "correct_confirmation_atlas": correct_atlas,
                "wrong_same_family_fit_atlas": wrong_atlas,
            },
            "native_raw_call_order": [
                "correct_negative",
                "correct_noop_positive",
                "correct_action_positive",
                "wrong_negative",
                "wrong_noop_positive",
                "drop_negative",
                "drop_noop_positive",
                "drop_action_positive",
            ],
            "negative_raw_reused_for_correct_noop_and_action": True,
            "negative_raw_reused_for_drop_noop_and_action": True,
            "raw_tensor_identities": {
                name: identity(f"raw:{rank}:{index}:{name}", dtype="torch.bfloat16")
                for name in (
                    "correct_negative",
                    "correct_noop",
                    "correct_action",
                    "wrong_negative",
                    "wrong_noop",
                    "drop_negative",
                    "drop_noop",
                    "drop_action",
                )
            },
            "ambient_torch_no_grad": True,
            "route_receipts": routes,
            **{name: True for name in PROVENANCE_TRUE},
            **{name: False for name in PROVENANCE_FALSE},
            **false_authority(),
        }
    )


def admission(arm: int, index: int, observed_metric: dict[str, object]) -> dict[str, object]:
    record = {
        "row_iid": CONFIRMATION_IIDS[arm],
        "wrong_owner_iid": FIT_IIDS[arm],
        "schedule_index": index,
        "metrics_digest": observed_metric["digest"],
        "parameter_digest": "7" * 64,
        "base_digest": "8" * 64,
        "optimizer_digest": "9" * 64,
    }
    return seal(
        {
            "schema_version": "bernini-graft-phase-a-confirmation-field-admission-v1",
            **record,
            "metrics": dict(observed_metric),
            "sp4_consensus_digest": sha(canonical(record)),
            "no_grad": True,
            "optimizer_update_performed": False,
            "checkpoint_written": False,
            **false_authority(),
        }
    )


def parity(rank: int) -> dict[str, object]:
    rows = []
    for index in (0, 25):
        for role in ("negative", "noop_positive", "action_positive"):
            raw = sha(f"parity:{rank}:{index}:{role}".encode())
            rows.append(
                {
                    "schedule_index": index,
                    "branch_role": role,
                    "adapter_route_gate_float64_hex": float(0.0).hex(),
                    "adapter_off_raw_sha256": raw,
                    "installed_zero_gate_raw_sha256": raw,
                    "raw_storage_byte_exact": True,
                    "native_full_source_v_pack_bytes_unchanged": True,
                    "noisy_target_bytes_unchanged": True,
                    "epsilon_bytes_unchanged": True,
                    "sigma_timestep_unchanged": True,
                    "condition_bytes_unchanged": True,
                    "target_video_used": False,
                }
            )
    return seal(
        {
            "schema_version": "bernini-graft-phase-a-short-adapter-off-bf16-parity-v1",
            "schedule_indices": [0, 25],
            "branch_roles": ["negative", "noop_positive", "action_positive"],
            "baseline_captured_before_adapter_install": True,
            "comparison_executed_after_two_updates_and_confirmation": True,
            "all_installed_zero_gate_raw_bytes_equal_adapter_off": True,
            "raw_dtype": "torch.bfloat16",
            "rows": rows,
            "scheduler_unchanged": True,
            "checkpoint_written": False,
            **false_authority(),
        }
    )


def update_route(rank: int, ordinal: int) -> dict[str, object]:
    index = (29, 38)[ordinal - 1]
    gate_hex = {
        29: "0x1.7f027f7ae326bp-4",
        38: "0x1.0000000000000p+0",
    }[index]
    phases = (
        ("measurement", "negative", False),
        ("measurement", "positive", False),
        ("replay", "negative", True),
        ("replay", "positive", True),
    )
    return seal(
        {
            "schema_version": "bernini-graft-phase-a-short-update-route-v1",
            "update_number": ordinal,
            "schedule_index": index,
            "row_iid": FIT_IIDS[rank // 4],
            "row_source_sha256": sha(f"fit-source:{rank // 4}".encode()),
            "fit_row_only": True,
            "exact_four_native_forwards": True,
            "forward_order": [[phase, role] for phase, role, _ in phases],
            "fresh_atlas_per_forward": True,
            "measurement_atlas_detached": True,
            "replay_atlas_graph_bearing_only_on_target_owner": True,
            "rows": [
                {
                    "ordinal": position,
                    "phase": phase,
                    "role": role,
                    "graph_expected": graph,
                    "schedule_index": index,
                    "route_gate_float64_hex": gate_hex,
                    "local_target_rows": 2,
                    "adapter_graph_bearing": graph,
                    "fresh_atlas_object": True,
                    "atlas_tokens": identity(
                        f"route-atlas:{rank}:{ordinal}",
                        dtype="torch.float32",
                    ),
                    "atlas_receipt_digest": sha(
                        f"route-atlas-receipt:{rank}:{ordinal}".encode()
                    ),
                }
                for position, (phase, role, graph) in enumerate(phases)
            ],
            "checkpoint_written": False,
            **false_authority(),
        }
    )


def trainer_update(rank: int, ordinal: int) -> dict[str, object]:
    arm = rank // 4
    regime = ("bootstrap", "post_bootstrap")[ordinal - 1]
    schedule_index = (29, 38)[ordinal - 1]
    row_source_sha = sha(f"fit-source:{arm}".encode())
    cell_plan = {
        "schema_version": "bernini-graft-phase-a-short-cell-plan-v1",
        "update_number": ordinal,
        "schedule_index": schedule_index,
        "expected_regime": regime,
        "dp_arm": arm,
        "row_iid": FIT_IIDS[arm],
        "row_source_sha256": row_source_sha,
        "routing_digest": sha(b"routing"),
        "source_owned_bytes_only": True,
        "source_path_input_accepted": False,
    }
    categories = (
        "atlas_encoder",
        "query_projection",
        "key_projection",
        "value_projection",
        "output_projection",
    )
    category_norms = {
        name: (float(0.0) if ordinal == 1 and name != "output_projection" else 0.25).hex()
        for name in categories
    }
    preclip = 0.25 if ordinal == 1 else (5.0 * 0.25 * 0.25) ** 0.5
    sync = seal(
        {
            "schema_version": "bernini-graft-dp2-sp4-gradient-sync-v1",
            "update_number": ordinal,
            "training_regime": regime,
            "none_materialized_as_true_zero_count": 0,
            "none_materialized_parameter_names": [],
            "collective_order": [
                "SP4_SUM",
                "divide_by_4",
                "DP2_SUM",
                "divide_by_2",
            ],
            "category_l2_float64_hex": category_norms,
            "preclip_l2_float64_hex": preclip.hex(),
            "gate": (
                "world8_bootstrap_output_projection_only_nonzero"
                if ordinal == 1
                else "world8_post_bootstrap_all_five_categories_nonzero"
            ),
            "finite": True,
        }
    )
    return seal(
        {
            "schema_version": "bernini-graft-phase-a-short-update-v1",
            "update_number": ordinal,
            "schedule_index": schedule_index,
            "training_regime": regime,
            "dp_arm": arm,
            "sp_rank": rank % 4,
            "row_iid": FIT_IIDS[arm],
            "row_source_sha256": row_source_sha,
            "plan_digest": sha(canonical(cell_plan)),
            "native_admission_digest": "4" * 64,
            "gradient_synchronization_digest": sync["digest"],
            "gradient_sync": sync,
            "preclip_norm_float64_hex": preclip.hex(),
            "postclip_norm_float64_hex": preclip.hex(),
            "max_grad_norm": 1.0,
            "optimizer": {
                "kind": "torch.optim.AdamW",
                "learning_rate": 1.0e-3,
                "weight_decay": 0.0,
                "betas": [0.9, 0.999],
                "eps": 1.0e-8,
                "foreach": False,
            },
            "parameter_digest_before": ("5" if ordinal == 1 else "6") * 64,
            "parameter_digest_after": ("6" if ordinal == 1 else "7") * 64,
            "frozen_base_digest": "8" * 64,
            "parameter_world_consensus": True,
            "frozen_base_world_consensus": True,
            "gradients_reset_to_none_after_step": True,
            "checkpoint_written": False,
            **false_authority(),
        }
    )


def local_result(rank: int, *, human_hard_gate: bool = True) -> dict[str, object]:
    arm = rank // 4
    metrics = {
        str(index): metric(
            arm,
            index,
            hard_gate=(human_hard_gate if arm == 1 and index == 29 else True),
        )
        for index in (29, 38)
    }
    provenances = [provenance(rank, index) for index in (29, 38)]
    admissions = [admission(arm, index, metrics[str(index)]) for index in (29, 38)]
    consensus = {
        str(index): admissions[position]["sp4_consensus_digest"]
        for position, index in enumerate((29, 38))
    }
    updates = [trainer_update(rank, ordinal) for ordinal in (1, 2)]
    trainer = seal(
        {
            "schema_version": "bernini-graft-phase-a-a-lite-short-training-v1",
            "status": "completed_in_memory_orchestration",
            "topology": {
                "world_size": 8,
                "data_parallel_size": 2,
                "sequence_parallel_size": 4,
                "rank": rank,
                "dp_arm": arm,
                "sp_rank": rank % 4,
            },
            "dependency_pins": {
                "consumer_commit": "6bae78c40b531310ad89a9b47418a8ff4a81ce05",
                "consumer_source_sha256": "13ecb082ab3cff6f809b056c35715123be302a5c8d82a6760a7367861920ee75",
                "native_v2_source_sha256": "bf6a1d438183de5aa0460e729a39382e4597b3e43a4b9f1b3cdff5457439f20f",
                "native_v2_schema": "bernini-graft-phase-a-native-training-closure-v2",
                "dependency_runtime_surface_sha256": "1bdf3747b598c577d73132b8bc278d8a3f911da35d19236d12d706b4d476cac1",
                "trainer_execution_runtime_sha256": "f35f621938e7a6b8bd3b9b5a6b0fb782f5ebf483939f585f561e3908f993af3c",
            },
            "source_routing": {
                "routing_digest": sha(b"routing"),
                "source_release_result_digest": sha(b"source-release"),
                "pinset_digest": sha(b"pinset"),
                "path_free_owned_bytes_only": True,
                "update_rows": 2,
                "confirmation_rows": 2,
                "local_update_iid": FIT_IIDS[arm],
                "local_confirmation_iid": CONFIRMATION_IIDS[arm],
                "confirmation_rows_consumed_by_optimizer": False,
                "authority": {
                    name: False
                    for name in (
                        "action_authority",
                        "identity_authority",
                        "cross_clip_identity_authority",
                        "quality_authority",
                        "training_authority",
                        "production_authority",
                        "data_governance_authority",
                        "data_license_authority",
                        "scientific_success_claimed",
                    )
                },
            },
            "optimizer_contract": {
                "kind": "torch.optim.AdamW",
                "learning_rate": 1.0e-3,
                "weight_decay": 0.0,
                "betas": [0.9, 0.999],
                "eps": 1.0e-8,
                "max_grad_norm": 1.0,
                "steps": 2,
                "schedule_indices": [29, 38],
                "regimes": ["bootstrap", "post_bootstrap"],
                "gradient_collective_order": ["SP4_SUM_div_4", "DP2_SUM_div_2"],
            },
            "updates": updates,
            "confirmation": {
                "plan": (
                    "per_row_per_index_noop_fm_relative_gain_plus_"
                    "action_delta_geometry"
                ),
                "row_iid": CONFIRMATION_IIDS[arm],
                "wrong_owner_iid": FIT_IIDS[arm],
                "wrong_owner_is_same_family_fit_row": True,
                "schedule_indices": [29, 38],
                "field_roles": list(FIELD_ROLES),
                "thresholds": {
                    "minimum_noop_fm_relative_gain": 1.0e-4,
                    "minimum_action_delta_correct_drop_norm_ratio": 0.5,
                    "minimum_action_delta_cosine": 0.0,
                },
                "per_index_metrics": metrics,
                "sp4_consensus_digest": consensus,
                "all_indices_noncompensating_hard_gate_passed": all(
                    row["noncompensating_all_pass"] for row in metrics.values()
                ),
                "evaluated_under_no_grad": True,
                "optimizer_state_unchanged": True,
                "parameters_unchanged": True,
                "frozen_base_unchanged": True,
                "six_field_tensor_contract_and_metrics_authenticated_by_this_core": True,
                "same_noise_x_sigma_coordinate_authenticated_by_this_core": False,
                "field_origin_runtime_authenticated_by_this_core": False,
                "runner_adapter_off_parity_indices": [0, 25],
                "runner_adapter_off_parity_verified_by_this_core": False,
                "runner_must_block_checkpoint_without_adapter_off_parity": True,
            },
            "initial_parameter_digest": "5" * 64,
            "final_parameter_digest": "7" * 64,
            "initial_frozen_base_digest": "8" * 64,
            "final_frozen_base_digest": "8" * 64,
            "optimizer_state_digest_after_training_and_confirmation": "9" * 64,
            "parameter_world_consensus": True,
            "frozen_base_world_consensus": True,
            "test_only": False,
            "real_model_loaded_by_this_core": False,
            "source_bytes_to_latent_binding_verified_by_this_core": False,
            "trainer_execution_runtime_live_verified": True,
            "same_process_execution_integrity_formally_proven_by_this_core": False,
            "same_process_formal_security_proven_by_this_core": False,
            "full_sampler_used": False,
            "decoded_media_used": False,
            "checkpoint_written": False,
            "checkpoint_payload_returned": False,
            "publication_performed": False,
            **false_authority(),
        }
    )
    observed_parity = parity(rank)
    routes = [update_route(rank, ordinal) for ordinal in (1, 2)]
    trace_operations = (
        "open_short_training",
        "next_update_plan",
        "make_native_v2_cell",
        "run_update",
        "admit_update_route_evidence",
        "next_update_plan",
        "make_native_v2_cell",
        "run_update",
        "admit_update_route_evidence",
        "confirmation_plan",
        "measure_six_confirmation_fields",
        "admit_confirmation_fields",
        "measure_six_confirmation_fields",
        "admit_confirmation_fields",
        "admit_adapter_off_bf16_raw_parity",
        "finish_in_memory_short_core",
    )
    trace = [
        {"ordinal": ordinal, "operation": operation}
        for ordinal, operation in enumerate(trace_operations)
    ]
    for offset, ordinal in ((1, 1), (5, 2)):
        index = (29, 38)[ordinal - 1]
        trace[offset].update(
            update_number=ordinal,
            schedule_index=index,
            row_iid=FIT_IIDS[arm],
        )
        trace[offset + 1].update(update_number=ordinal, schedule_index=index)
        trace[offset + 2].update(
            update_number=ordinal,
            schedule_index=index,
            update_receipt_digest=updates[ordinal - 1]["digest"],
        )
        trace[offset + 3].update(
            update_number=ordinal,
            schedule_index=index,
            route_receipt_digest=routes[ordinal - 1]["digest"],
        )
    trace[9].update(
        row_iid=CONFIRMATION_IIDS[arm], wrong_owner_iid=FIT_IIDS[arm]
    )
    trace[10].update(
        schedule_index=29,
        row_iid=CONFIRMATION_IIDS[arm],
        wrong_owner_iid=FIT_IIDS[arm],
    )
    trace[11].update(
        schedule_index=29,
        provenance_digest=provenances[0]["digest"],
        admission_digest=admissions[0]["digest"],
    )
    trace[12].update(
        schedule_index=38,
        row_iid=CONFIRMATION_IIDS[arm],
        wrong_owner_iid=FIT_IIDS[arm],
    )
    trace[13].update(
        schedule_index=38,
        provenance_digest=provenances[1]["digest"],
        admission_digest=admissions[1]["digest"],
    )
    trace[14].update(
        schedule_indices=[0, 25], parity_digest=observed_parity["digest"]
    )
    trace[15]["trainer_receipt_digest"] = trainer["digest"]
    return seal(
        {
            "schema_version": "bernini-graft-phase-a-a-lite-short-gpu-runner-v1",
            "status": "completed_in_memory_diagnostic_no_checkpoint",
            "complete": True,
            "topology": {
                "world_size": 8,
                "data_parallel_size": 2,
                "sequence_parallel_size": 4,
                "rank": rank,
                "dp_arm": arm,
                "sp_rank": rank % 4,
                "family": FAMILIES[arm],
            },
            "dependency_source_sha256": {
                "a_lite_consumer": "13ecb082ab3cff6f809b056c35715123be302a5c8d82a6760a7367861920ee75",
                "native_v2_closure": "bf6a1d438183de5aa0460e729a39382e4597b3e43a4b9f1b3cdff5457439f20f",
                "short_trainer": "73e39048bb8836fef33516eb1aae4cbc3f9fa4ecefcfb5d2695925bcb150f7bb",
                "native_v1_gpu_runner_reuse": "e0b69442be284e091bad8d36a205bffe8bd314082188bfa55da72f4c2640945a",
            },
            "source_routing": {
                "routing_digest": sha(b"routing"),
                "fit_iid": FIT_IIDS[arm],
                "confirmation_iid": CONFIRMATION_IIDS[arm],
                "wrong_atlas_iid": FIT_IIDS[arm],
                "fit_row_consumed_by_optimizer": True,
                "confirmation_row_consumed_by_optimizer": False,
                "wrong_atlas_is_same_family_fit_row": True,
                "owned_source_bytes_only": True,
                "source_path_reopened_by_runner": False,
            },
            "confirmation": {
                "schedule_indices": [29, 38],
                "field_roles": list(FIELD_ROLES),
                "provenance": provenances,
                "admissions": admissions,
                "exact_six_fields_per_index": True,
                "same_state_interventions_verified": True,
                "wrong_atlas_same_family_fit_verified": True,
                "drop_disables_only_identity_rebinder_memory_verified": True,
            },
            "adapter_off_parity": observed_parity,
            "short_trainer_receipt": trainer,
            "update_route_receipts": routes,
            "execution_trace": trace,
            "training_updates_executed_for_diagnostic": 2,
            "full_sampler_used": False,
            "decoded_media_output_created": False,
            "checkpoint_payload_returned": False,
            "target_video_used": False,
            "generated_proposal_used": False,
            "t2v_branch_used": False,
            "source_retelling_used": False,
            "selector_used": False,
            "mask_pose_track_flow_or_motion_donor_used": False,
            "checkpoint_written": False,
            "result_staged_in_memory_only": True,
            "publication_performed": False,
            **false_authority(),
        }
    )


def make_world8_result(*, human_hard_gate: bool = True) -> dict[str, object]:
    receipts = [
        local_result(rank, human_hard_gate=human_hard_gate) for rank in range(8)
    ]
    rows = [
        {
            "global_rank": rank,
            "dp_arm": rank // 4,
            "sp_rank": rank % 4,
            "family": FAMILIES[rank // 4],
            "result_digest": receipt["digest"],
            "trainer_result_digest": receipt["short_trainer_receipt"]["digest"],
            "adapter_off_parity_digest": receipt["adapter_off_parity"]["digest"],
        }
        for rank, receipt in enumerate(receipts)
    ]
    arms = [
        {
            "dp_arm": arm,
            "family": FAMILIES[arm],
            "global_ranks": list(range(arm * 4, (arm + 1) * 4)),
            "representative_global_rank": arm * 4,
            "representative_full_receipt": receipts[arm * 4],
            "per_rank_result_digests": [
                receipt["digest"] for receipt in receipts[arm * 4 : (arm + 1) * 4]
            ],
            "all_four_confirmation_hard_gates_passed": human_hard_gate or arm == 0,
        }
        for arm in range(2)
    ]
    full = seal(
        {
            "schema_version": "bernini-graft-phase-a-short-world8-full-results-v1",
            "rank_order": list(range(8)),
            "dp2_family_order": list(FAMILIES),
            "all_eight_full_local_receipts": receipts,
            "arm_representatives": arms,
            "dog_and_human_exact_coverage": True,
            "all_eight_confirmation_hard_gates_passed": human_hard_gate,
            "checkpoint_written": False,
            "publication_performed": False,
            **false_authority(),
        }
    )
    source_binding = seal(
        {
            "schema_version": "bernini-graft-phase-a-short-source-binding-v1",
            "runner_sha256": RUNNER_SHA,
            "consumer_sha256": "13ecb082ab3cff6f809b056c35715123be302a5c8d82a6760a7367861920ee75",
            "native_v2_sha256": "bf6a1d438183de5aa0460e729a39382e4597b3e43a4b9f1b3cdff5457439f20f",
            "short_trainer_sha256": "73e39048bb8836fef33516eb1aae4cbc3f9fa4ecefcfb5d2695925bcb150f7bb",
            "native_runner_v1_sha256": "e0b69442be284e091bad8d36a205bffe8bd314082188bfa55da72f4c2640945a",
            "identity_rebinder_sha256": "1f954f7446bdd4c9e465d44b045c8fe40d34dfa9c2aad6585256f5b4fb29dde0",
            "bernini_commit": "2d2b4591ac053ec25c6371b01a5a6746679e5793",
            "veomni_commit": "f90b3dc6fbb0ce693745223cc7a94064123dbf4d",
            "bernini_inference_files": {
                "bernini/pipeline.py": "c6acf05c01a637d9bce69e8160eb6eb4260ff4ec798fd990de8e5aa73999ab40",
                "bernini/cli.py": "26949fbf246003403ed0cca1ec1bbb62c2099fc9740bb17ba5a1e7c86fbc0edf",
                "bernini/io_utils.py": "233541373746f5d97e1cb3680d3c2a41d5d212b797eefb97693afa6e3ab5f30a",
            },
        }
    )
    return seal(
        {
            "schema_version": "bernini-graft-phase-a-short-world8-result-set-v2",
            "status": "completed_in_memory_diagnostic_no_checkpoint",
            "world8_rows": rows,
            "world8_full_results": full,
            "local_result": receipts[0],
            "source_binding": source_binding,
            "checkpoint_identity": seal(
                {
                    "schema_version": "bernini-graft-phase-a-short-checkpoint-v1",
                    "identity": {
                        "manifest_path": "/checkpoint/manifest.sha256",
                        "manifest_sha256_computed": CHECKPOINT_MANIFEST_SHA,
                        "manifest_sha256_expected": CHECKPOINT_MANIFEST_SHA,
                        "verified_file_count": 23,
                        "every_file_sha256_verified": True,
                        "verified_entries_digest": "e" * 64,
                    },
                }
            ),
            "initialization": seal(
                {
                    "schema_version": "bernini-graft-phase-a-short-initial-registry-v1",
                    "rank0_broadcast_before_any_adapter_forward": True,
                    "parameter_count": 5,
                    "parameter_sha256": "5" * 64,
                    "zero_output_projection_count": 1,
                    "zero_output_projection_exact": True,
                }
            ),
            "topology_receipt": seal(
                {
                    "schema_version": "bernini-graft-phase-a-live-world8-dp2sp4-v1",
                    "world_size": 8,
                    "dp_size": 2,
                    "sp_size": 4,
                    "global_rank": 0,
                    "local_rank": 0,
                    "dp_arm": 0,
                    "sp_rank": 0,
                    "sp_members": [0, 1, 2, 3],
                    "dp_members": [0, 4],
                    "backend": "nccl",
                }
            ),
            "base_sha256_before": "a" * 64,
            "base_sha256_after": "a" * 64,
            "base_bytes_unchanged": True,
            "base_gradients_all_none": True,
            "wan_diffusion_sha256": "59e860ba3490a83f06bd4be75697490f49a118ee5ca969e85eea4dd7fa122512",
            "transformer_wan_sha256": "9fb579611e79e0f534d5d6ccdcd956c35e57b4513c15267e8533ff3832a1f223",
            "runtime_versions": {
                "torch": "2.7.0",
                "torch_hip": "6.3",
                "diffusers": "0.33.0",
                "transformers": "4.51.0",
            },
            "checkpoint_written": False,
            "publication_performed": False,
            **false_authority(),
        }
    )


def reseal(value: dict[str, object]) -> dict[str, object]:
    unsigned = dict(value)
    unsigned.pop("digest", None)
    return seal(unsigned)


def replace_full_result(
    outer: dict[str, object], full: dict[str, object]
) -> dict[str, object]:
    unsigned = dict(outer)
    unsigned.pop("digest")
    unsigned["world8_full_results"] = reseal(full)
    return seal(unsigned)


def replace_local_receipt(
    outer: dict[str, object], rank: int, local: dict[str, object]
) -> dict[str, object]:
    unsigned = dict(outer)
    unsigned.pop("digest")
    full = dict(unsigned["world8_full_results"])
    full.pop("digest")
    receipts = list(full["all_eight_full_local_receipts"])
    receipts[rank] = reseal(local)
    full["all_eight_full_local_receipts"] = receipts
    rows = [dict(row) for row in unsigned["world8_rows"]]
    rows[rank]["result_digest"] = receipts[rank]["digest"]
    rows[rank]["trainer_result_digest"] = receipts[rank]["short_trainer_receipt"][
        "digest"
    ]
    rows[rank]["adapter_off_parity_digest"] = receipts[rank][
        "adapter_off_parity"
    ]["digest"]
    unsigned["world8_rows"] = rows
    arms = [dict(arm) for arm in full["arm_representatives"]]
    arm = rank // 4
    arm_digests = list(arms[arm]["per_rank_result_digests"])
    arm_digests[rank % 4] = receipts[rank]["digest"]
    arms[arm]["per_rank_result_digests"] = arm_digests
    if rank in (0, 4):
        arms[arm]["representative_full_receipt"] = receipts[rank]
    full["arm_representatives"] = arms
    unsigned["world8_full_results"] = seal(full)
    if rank == 0:
        unsigned["local_result"] = receipts[0]
    return seal(unsigned)


def replace_trainer_receipt(
    local: dict[str, object], trainer: dict[str, object]
) -> dict[str, object]:
    """Reseal trainer -> finish-trace -> local, preserving producer aliases."""

    unsigned = dict(local)
    unsigned.pop("digest", None)
    owned_trainer = reseal(trainer)
    unsigned["short_trainer_receipt"] = owned_trainer
    trace = [dict(row) for row in unsigned["execution_trace"]]
    trace[15]["trainer_receipt_digest"] = owned_trainer["digest"]
    unsigned["execution_trace"] = trace
    return seal(unsigned)


def replace_trainer_update(
    local: dict[str, object], position: int, update: dict[str, object]
) -> dict[str, object]:
    """Reseal update -> trainer -> both digest-bearing trace aliases -> local."""

    trainer = dict(local["short_trainer_receipt"])
    trainer.pop("digest", None)
    updates = list(trainer["updates"])
    updates[position] = reseal(update)
    trainer["updates"] = updates
    updated = replace_trainer_receipt(local, trainer)
    unsigned = dict(updated)
    unsigned.pop("digest")
    trace = [dict(row) for row in unsigned["execution_trace"]]
    trace[(3, 7)[position]]["update_receipt_digest"] = updates[position]["digest"]
    unsigned["execution_trace"] = trace
    return seal(unsigned)


def replace_provenance(
    local: dict[str, object], position: int, observed: dict[str, object]
) -> dict[str, object]:
    """Reseal provenance -> admission-trace alias -> local."""

    unsigned = dict(local)
    unsigned.pop("digest", None)
    confirmation = dict(unsigned["confirmation"])
    rows = list(confirmation["provenance"])
    rows[position] = reseal(observed)
    confirmation["provenance"] = rows
    unsigned["confirmation"] = confirmation
    trace = [dict(row) for row in unsigned["execution_trace"]]
    trace[(11, 13)[position]]["provenance_digest"] = rows[position]["digest"]
    unsigned["execution_trace"] = trace
    return seal(unsigned)


def replace_metric_and_admission(
    local: dict[str, object],
    position: int,
    observed: dict[str, object],
    *,
    force_trainer_aggregate: bool | None = None,
) -> dict[str, object]:
    """Reseal metric -> admission/consensus -> trainer/trace -> local."""

    index = (29, 38)[position]
    metric_row = reseal(observed)
    trainer = dict(local["short_trainer_receipt"])
    trainer.pop("digest", None)
    trainer_confirmation = dict(trainer["confirmation"])
    metrics = dict(trainer_confirmation["per_index_metrics"])
    metrics[str(index)] = metric_row
    trainer_confirmation["per_index_metrics"] = metrics

    unsigned = dict(local)
    unsigned.pop("digest", None)
    runner_confirmation = dict(unsigned["confirmation"])
    admissions = list(runner_confirmation["admissions"])
    admission_row = dict(admissions[position])
    admission_row.pop("digest", None)
    admission_row["metrics"] = metric_row
    admission_row["metrics_digest"] = metric_row["digest"]
    record = {
        name: admission_row[name]
        for name in (
            "row_iid",
            "wrong_owner_iid",
            "schedule_index",
            "metrics_digest",
            "parameter_digest",
            "base_digest",
            "optimizer_digest",
        )
    }
    admission_row["sp4_consensus_digest"] = sha(canonical(record))
    admissions[position] = seal(admission_row)
    runner_confirmation["admissions"] = admissions
    unsigned["confirmation"] = runner_confirmation

    consensus = dict(trainer_confirmation["sp4_consensus_digest"])
    consensus[str(index)] = admission_row["sp4_consensus_digest"]
    trainer_confirmation["sp4_consensus_digest"] = consensus
    if force_trainer_aggregate is not None:
        trainer_confirmation[
            "all_indices_noncompensating_hard_gate_passed"
        ] = force_trainer_aggregate
    trainer["confirmation"] = trainer_confirmation
    unsigned = replace_trainer_receipt(unsigned, trainer)
    unsigned.pop("digest")
    trace = [dict(row) for row in unsigned["execution_trace"]]
    trace[(11, 13)[position]]["admission_digest"] = admissions[position]["digest"]
    unsigned["execution_trace"] = trace
    return seal(unsigned)


def replace_admission(
    local: dict[str, object], position: int, observed: dict[str, object]
) -> dict[str, object]:
    """Reseal admission -> trainer consensus -> both trace aliases -> local."""

    index = (29, 38)[position]
    admission_row = dict(observed)
    admission_row.pop("digest", None)
    record = {
        name: admission_row[name]
        for name in (
            "row_iid",
            "wrong_owner_iid",
            "schedule_index",
            "metrics_digest",
            "parameter_digest",
            "base_digest",
            "optimizer_digest",
        )
    }
    admission_row["sp4_consensus_digest"] = sha(canonical(record))
    admission_row = seal(admission_row)

    unsigned = dict(local)
    unsigned.pop("digest", None)
    runner_confirmation = dict(unsigned["confirmation"])
    admissions = list(runner_confirmation["admissions"])
    admissions[position] = admission_row
    runner_confirmation["admissions"] = admissions
    unsigned["confirmation"] = runner_confirmation

    trainer = dict(unsigned["short_trainer_receipt"])
    trainer.pop("digest", None)
    trainer_confirmation = dict(trainer["confirmation"])
    consensus = dict(trainer_confirmation["sp4_consensus_digest"])
    consensus[str(index)] = admission_row["sp4_consensus_digest"]
    trainer_confirmation["sp4_consensus_digest"] = consensus
    trainer["confirmation"] = trainer_confirmation
    unsigned = replace_trainer_receipt(unsigned, trainer)
    unsigned.pop("digest")
    trace = [dict(row) for row in unsigned["execution_trace"]]
    trace[(11, 13)[position]]["admission_digest"] = admission_row["digest"]
    unsigned["execution_trace"] = trace
    return seal(unsigned)


def replace_update_route(
    local: dict[str, object], position: int, route: dict[str, object]
) -> dict[str, object]:
    """Reseal route -> route-trace alias -> local."""

    unsigned = dict(local)
    unsigned.pop("digest", None)
    routes = list(unsigned["update_route_receipts"])
    routes[position] = reseal(route)
    unsigned["update_route_receipts"] = routes
    trace = [dict(row) for row in unsigned["execution_trace"]]
    trace[(4, 8)[position]]["route_receipt_digest"] = routes[position]["digest"]
    unsigned["execution_trace"] = trace
    return seal(unsigned)


class ParentFixture:
    def __init__(self, test: unittest.TestCase) -> None:
        temporary = tempfile.TemporaryDirectory()
        test.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "output"
        self.logs = self.output / "rank-logs" / "run" / "attempt_0"
        self.logs.mkdir(parents=True)
        self.result = make_world8_result()
        self.checkpoint_snapshot = sha(
            canonical(self.result["checkpoint_identity"]["identity"])
        )
        for rank in range(8):
            directory = self.logs / str(rank)
            directory.mkdir()
            ordinary = b"ordinary model log\n"
            if rank == 0:
                ordinary += canonical(self.result) + b"\n"
            (directory / "stdout.log").write_bytes(ordinary)
            (directory / "stderr.log").write_text("warning\n", encoding="utf-8")

    def run(self) -> subprocess.CompletedProcess[str]:
        identity = f"{self.output.stat().st_dev}:{self.output.stat().st_ino}"
        log_root = self.output / "rank-logs"
        log_identity = f"{log_root.stat().st_dev}:{log_root.stat().st_ino}"
        root_fd = os.open(self.output, os.O_RDONLY | os.O_DIRECTORY)
        log_fd = os.open(log_root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            return subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-B",
                    "-",
                    str(self.output),
                    identity,
                    str(root_fd),
                    str(log_root),
                    log_identity,
                    str(log_fd),
                    *("b" * 64 for _ in range(6)),
                    self.checkpoint_snapshot,
                    self.checkpoint_snapshot,
                ],
                input=marked_python_source(
                    "# BEGIN GRAFT_PHASE_A_SHORT_WORLD8_PARENT_VALIDATOR_V1",
                    "# END GRAFT_PHASE_A_SHORT_WORLD8_PARENT_VALIDATOR_V1",
                ),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=30,
                pass_fds=(root_fd, log_fd),
            )
        finally:
            os.close(log_fd)
            os.close(root_fd)

    def write_result(self, rank: int, value: dict[str, object], *, pretty: bool = False) -> None:
        path = self.logs / str(rank) / "stdout.log"
        payload = (
            json.dumps(value, indent=2, sort_keys=True).encode("ascii")
            if pretty
            else canonical(value)
        )
        path.write_bytes(b"ordinary\n" + payload + b"\n")


class ParentValidatorTests(unittest.TestCase):
    def test_admits_one_rank0_schema_ignores_logs_and_publishes_0444(self) -> None:
        fixture = ParentFixture(self)
        sigma = fixture.result["local_result"]["confirmation"]["provenance"][0][
            "same_state_identities_before_model_fields"
        ]["sigma"]
        self.assertEqual(sigma["shape"], [])
        self.assertEqual(sigma["byte_count"], 4)
        self.assertEqual(sigma["device_type_at_observation"], "cpu")
        result = fixture.run()
        self.assertEqual(result.returncode, 0, result.stderr)
        receipt_path = fixture.output / "receipt.json"
        raw = receipt_path.read_bytes()
        receipt = json.loads(raw)
        self.assertEqual(raw, canonical(receipt) + b"\n")
        self.assertEqual(stat.S_IMODE(receipt_path.stat().st_mode), 0o444)
        self.assertEqual(receipt["provenance"]["runtime_closure_exact_regular_members"], 23)
        self.assertTrue(receipt["validated"]["ordinary_logs_not_receipts"])
        self.assertTrue(
            receipt["validated"]["all_eight_full_local_receipts_deeply_validated"]
        )
        self.assertTrue(
            receipt["validated"]["dog_and_human_arm_representatives_deeply_validated"]
        )
        self.assertTrue(
            receipt["validated"][
                "rank_log_root_identity_retained_from_before_torchrun"
            ]
        )
        self.assertTrue(
            receipt["receipt_publication"][
                "output_root_identity_retained_from_creation"
            ]
        )
        for field in (
            "metric_values_recomputed_from_tensor_bytes",
            "confirmation_cosine_recomputed_from_tensor_bytes",
            "field_tensor_content_sha256_recomputed_from_bytes",
            "atlas_receipt_bodies_deeply_validated",
            "source_release_routing_body_deeply_validated",
            "native_admission_body_deeply_validated",
            "optimizer_world_consensus_proven_by_producer",
        ):
            self.assertFalse(receipt["validated"][field])
        self.assertEqual(
            receipt["schema_version"],
            "bernini-graft-phase-a-a-lite-short-world8-parent-v2",
        )
        self.assertEqual(
            receipt["provenance"]["allocator_environment"],
            {
                "requested_environment_only": {
                    "PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True",
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                },
                "effective_runtime_configuration_verified": False,
            },
        )
        self.assertFalse(receipt["checkpoint_written"])
        self.assertTrue(all(value is False for value in receipt["authority"].values()))

    def test_rejects_duplicate_nonzero_rank_result_schema(self) -> None:
        fixture = ParentFixture(self)
        fixture.write_result(7, fixture.result)
        result = fixture.run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exactly one rank-zero result", result.stderr)

    def test_rejects_noncanonical_result_line(self) -> None:
        fixture = ParentFixture(self)
        fixture.write_result(0, fixture.result, pretty=True)
        result = fixture.run()
        self.assertNotEqual(result.returncode, 0)

    def test_rejects_wrong_world8_group_geometry(self) -> None:
        fixture = ParentFixture(self)
        value = dict(fixture.result)
        value.pop("digest")
        rows = [dict(row) for row in value["world8_rows"]]
        rows[4]["dp_arm"] = 0
        value["world8_rows"] = rows
        fixture.result = seal(value)
        fixture.write_result(0, fixture.result)
        result = fixture.run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("rank4 WORLD8 summary geometry differs", result.stderr)

    def test_rejects_elevated_authority(self) -> None:
        fixture = ParentFixture(self)
        value = dict(fixture.result)
        value.pop("digest")
        value["training_authority"] = True
        fixture.write_result(0, seal(value))
        result = fixture.run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("elevated authority", result.stderr)

    def test_rejects_nested_adapter_parity_gate_even_when_aggregates_are_valid(self) -> None:
        fixture = ParentFixture(self)
        local = dict(fixture.result["world8_full_results"]["all_eight_full_local_receipts"][0])
        local.pop("digest")
        parity = dict(local["adapter_off_parity"])
        parity.pop("digest")
        parity["all_installed_zero_gate_raw_bytes_equal_adapter_off"] = False
        local["adapter_off_parity"] = seal(parity)
        fixture.result = replace_local_receipt(fixture.result, 0, local)
        fixture.write_result(0, fixture.result)
        result = fixture.run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("rank0 adapter parity contract differs", result.stderr)

    def test_rejects_human_false_gate_hidden_by_true_aggregate_flags(self) -> None:
        fixture = ParentFixture(self)
        local = fixture.result["world8_full_results"][
            "all_eight_full_local_receipts"
        ][4]
        false_metric = metric(1, 29, hard_gate=False)
        false_metric.pop("digest")
        local = replace_metric_and_admission(
            local, 0, false_metric, force_trainer_aggregate=True
        )
        fixture.result = replace_local_receipt(fixture.result, 4, local)
        fixture.write_result(0, fixture.result)
        result = fixture.run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("rank4 index29 metric gate differs", result.stderr)

    def test_rejects_zero_or_negative_tensor_dimensions_after_bottom_up_reseal(self) -> None:
        for dimension in (0, -1):
            with self.subTest(dimension=dimension):
                fixture = ParentFixture(self)
                local = fixture.result["world8_full_results"][
                    "all_eight_full_local_receipts"
                ][0]
                observed = dict(local["confirmation"]["provenance"][0])
                observed.pop("digest")
                before = {
                    name: dict(value)
                    for name, value in observed[
                        "same_state_identities_before_model_fields"
                    ].items()
                }
                sigma = dict(before["sigma"])
                sigma["shape"] = [dimension]
                sigma["byte_count"] = dimension * 4
                before["sigma"] = sigma
                observed["same_state_identities_before_model_fields"] = before
                observed["same_state_identities_after_all_fields"] = {
                    name: dict(value) for name, value in before.items()
                }
                local = replace_provenance(local, 0, observed)
                fixture.result = replace_local_receipt(fixture.result, 0, local)
                fixture.write_result(0, fixture.result)
                result = fixture.run()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("tensor identity differs", result.stderr)

    def test_rejects_resealed_field_byte_count_or_device_forgery(self) -> None:
        for case in ("byte_count", "device"):
            with self.subTest(case=case):
                fixture = ParentFixture(self)
                local = fixture.result["local_result"]
                observed = dict(local["confirmation"]["provenance"][0])
                observed.pop("digest")
                identities = {
                    name: dict(value)
                    for name, value in observed["field_tensor_identities"].items()
                }
                if case == "byte_count":
                    identities[FIELD_ROLES[0]]["byte_count"] += 1
                    expected = "byte count differs"
                else:
                    identities[FIELD_ROLES[0]][
                        "device_type_at_observation"
                    ] = "cpu"
                    expected = "device differs"
                observed["field_tensor_identities"] = identities
                local = replace_provenance(local, 0, observed)
                fixture.result = replace_local_receipt(fixture.result, 0, local)
                fixture.write_result(0, fixture.result)
                result = fixture.run()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)

    def test_rejects_resealed_update_state_chain_break(self) -> None:
        fixture = ParentFixture(self)
        local = fixture.result["local_result"]
        update = dict(local["short_trainer_receipt"]["updates"][0])
        update.pop("digest")
        update["parameter_digest_after"] = "c" * 64
        local = replace_trainer_update(local, 0, update)
        fixture.result = replace_local_receipt(fixture.result, 0, local)
        fixture.write_result(0, fixture.result)
        result = fixture.run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("rank0 trainer state chain differs", result.stderr)

    def test_rejects_resealed_metric_scalar_forgery(self) -> None:
        fixture = ParentFixture(self)
        local = fixture.result["local_result"]
        observed = dict(
            local["short_trainer_receipt"]["confirmation"]["per_index_metrics"][
                "29"
            ]
        )
        observed.pop("digest")
        gains = dict(observed["relative_gain_float64_hex"])
        gains["correct_vs_wrong"] = float(0.5).hex()
        observed["relative_gain_float64_hex"] = gains
        local = replace_metric_and_admission(local, 0, observed)
        fixture.result = replace_local_receipt(fixture.result, 0, local)
        fixture.write_result(0, fixture.result)
        result = fixture.run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("rank0 index29 numeric/formula gate differs", result.stderr)

    def test_rejects_resealed_cosine_gate_or_noncanonical_float(self) -> None:
        for case in ("cosine", "noncanonical"):
            with self.subTest(case=case):
                fixture = ParentFixture(self)
                local = fixture.result["local_result"]
                observed = dict(
                    local["short_trainer_receipt"]["confirmation"][
                        "per_index_metrics"
                    ]["29"]
                )
                observed.pop("digest")
                if case == "cosine":
                    observed[
                        "action_delta_correct_drop_cosine_float64_hex"
                    ] = float(-0.25).hex()
                    expected = "numeric/formula gate differs"
                else:
                    losses = dict(observed["noop_fm_loss_float64_hex"])
                    losses["correct_atlas"] = "0x1.8p-1"
                    observed["noop_fm_loss_float64_hex"] = losses
                    expected = "noncanonical or nonfinite"
                local = replace_metric_and_admission(local, 0, observed)
                fixture.result = replace_local_receipt(fixture.result, 0, local)
                fixture.write_result(0, fixture.result)
                result = fixture.run()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)

    def test_rejects_resealed_metric_provenance_raw_hash_mismatch(self) -> None:
        fixture = ParentFixture(self)
        local = fixture.result["local_result"]
        observed = dict(local["confirmation"]["provenance"][0])
        observed.pop("digest")
        identities = {
            name: dict(value)
            for name, value in observed["field_tensor_identities"].items()
        }
        identities[FIELD_ROLES[0]]["raw_sha256"] = "c" * 64
        observed["field_tensor_identities"] = identities
        local = replace_provenance(local, 0, observed)
        fixture.result = replace_local_receipt(fixture.result, 0, local)
        fixture.write_result(0, fixture.result)
        result = fixture.run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "rank0 index29 trainer/admission/provenance consensus differs",
            result.stderr,
        )

    def test_rejects_resealed_admission_not_linked_to_final_optimizer(self) -> None:
        fixture = ParentFixture(self)
        local = fixture.result["local_result"]
        observed = dict(local["confirmation"]["admissions"][0])
        observed.pop("digest")
        observed["optimizer_digest"] = "c" * 64
        local = replace_admission(local, 0, observed)
        fixture.result = replace_local_receipt(fixture.result, 0, local)
        fixture.write_result(0, fixture.result)
        result = fixture.run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "rank0 index29 trainer/admission/provenance consensus differs",
            result.stderr,
        )

    def test_rejects_valid_sha_that_is_not_recomputed_cell_plan_digest(self) -> None:
        fixture = ParentFixture(self)
        local = fixture.result["local_result"]
        update = dict(local["short_trainer_receipt"]["updates"][0])
        update.pop("digest")
        update["plan_digest"] = "c" * 64
        local = replace_trainer_update(local, 0, update)
        fixture.result = replace_local_receipt(fixture.result, 0, local)
        fixture.write_result(0, fixture.result)
        result = fixture.run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("rank0 update plan digest differs", result.stderr)

    def test_rejects_resealed_route_source_not_linked_to_update(self) -> None:
        fixture = ParentFixture(self)
        local = fixture.result["local_result"]
        route = dict(local["update_route_receipts"][0])
        route.pop("digest")
        route["row_source_sha256"] = "c" * 64
        local = replace_update_route(local, 0, route)
        fixture.result = replace_local_receipt(fixture.result, 0, local)
        fixture.write_result(0, fixture.result)
        result = fixture.run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("rank0 fit/confirmation source linkage differs", result.stderr)

    def test_rejects_resealed_trainer_test_only_or_unknown_metric_key(self) -> None:
        for case in ("test_only", "unknown_metric_key"):
            with self.subTest(case=case):
                fixture = ParentFixture(self)
                local = fixture.result["local_result"]
                if case == "test_only":
                    trainer = dict(local["short_trainer_receipt"])
                    trainer.pop("digest")
                    trainer["test_only"] = True
                    local = replace_trainer_receipt(local, trainer)
                    expected = "rank0 trainer contract differs"
                else:
                    observed = dict(
                        local["short_trainer_receipt"]["confirmation"][
                            "per_index_metrics"
                        ]["29"]
                    )
                    observed.pop("digest")
                    observed["unknown_forward_compatible_key"] = False
                    local = replace_metric_and_admission(local, 0, observed)
                    expected = "rank0 index29 metric gate differs"
                fixture.result = replace_local_receipt(fixture.result, 0, local)
                fixture.write_result(0, fixture.result)
                result = fixture.run()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)

    def test_rejects_resealed_checkpoint_identity_not_matching_external_snapshot(self) -> None:
        fixture = ParentFixture(self)
        unsigned = dict(fixture.result)
        unsigned.pop("digest")
        checkpoint = dict(unsigned["checkpoint_identity"])
        checkpoint.pop("digest")
        checkpoint_identity = dict(checkpoint["identity"])
        checkpoint_identity["verified_entries_digest"] = "c" * 64
        checkpoint["identity"] = checkpoint_identity
        unsigned["checkpoint_identity"] = seal(checkpoint)
        fixture.result = seal(unsigned)
        fixture.write_result(0, fixture.result)
        result = fixture.run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("checkpoint identity contract differs", result.stderr)

    def test_rejects_resealed_source_binding_initialization_or_topology(self) -> None:
        for case in ("source", "initialization", "topology", "topology_bool"):
            with self.subTest(case=case):
                fixture = ParentFixture(self)
                unsigned = dict(fixture.result)
                unsigned.pop("digest")
                if case == "source":
                    source = dict(unsigned["source_binding"])
                    source.pop("digest")
                    source["native_runner_v1_sha256"] = "c" * 64
                    unsigned["source_binding"] = seal(source)
                    expected = "WORLD8 source provenance differs"
                elif case == "initialization":
                    initialization = dict(unsigned["initialization"])
                    initialization.pop("digest")
                    initialization["parameter_sha256"] = "c" * 64
                    unsigned["initialization"] = seal(initialization)
                    expected = "initialization contract differs"
                else:
                    topology = dict(unsigned["topology_receipt"])
                    topology.pop("digest")
                    if case == "topology":
                        topology["sp_members"] = [0, 1, 2, 4]
                    else:
                        topology["global_rank"] = False
                    unsigned["topology_receipt"] = seal(topology)
                    expected = "rank0 topology receipt differs"
                fixture.result = seal(unsigned)
                fixture.write_result(0, fixture.result)
                result = fixture.run()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)

    def test_rank_stdout_links_and_parent_receipt_are_create_only(self) -> None:
        fixture = ParentFixture(self)
        os.link(
            fixture.logs / "0" / "stdout.log",
            fixture.logs / "0" / "stdout-hardlink.log",
        )
        result = fixture.run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("rank stdout is not bounded plain link-count-one", result.stderr)

        fixture = ParentFixture(self)
        receipt = fixture.output / "receipt.json"
        receipt.write_text("occupied", encoding="utf-8")
        result = fixture.run()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(receipt.read_text(encoding="utf-8"), "occupied")

    def test_rejects_missing_one_of_eight_full_receipts(self) -> None:
        fixture = ParentFixture(self)
        full = dict(fixture.result["world8_full_results"])
        full.pop("digest")
        full["all_eight_full_local_receipts"] = list(
            full["all_eight_full_local_receipts"]
        )[:-1]
        fixture.result = replace_full_result(fixture.result, full)
        fixture.write_result(0, fixture.result)
        result = fixture.run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("WORLD8 full result coverage differs", result.stderr)

    def test_rejects_world8_summary_digest_swap(self) -> None:
        fixture = ParentFixture(self)
        unsigned = dict(fixture.result)
        unsigned.pop("digest")
        rows = [dict(row) for row in unsigned["world8_rows"]]
        rows[4]["result_digest"] = rows[0]["result_digest"]
        unsigned["world8_rows"] = rows
        fixture.result = seal(unsigned)
        fixture.write_result(0, fixture.result)
        result = fixture.run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("rank4 summary/local digest differs", result.stderr)

    def test_rejects_swapped_dog_human_arm_representatives(self) -> None:
        fixture = ParentFixture(self)
        full = dict(fixture.result["world8_full_results"])
        full.pop("digest")
        arms = [dict(arm) for arm in full["arm_representatives"]]
        arms[0]["representative_full_receipt"], arms[1][
            "representative_full_receipt"
        ] = (
            arms[1]["representative_full_receipt"],
            arms[0]["representative_full_receipt"],
        )
        full["arm_representatives"] = arms
        fixture.result = replace_full_result(fixture.result, full)
        fixture.write_result(0, fixture.result)
        result = fixture.run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("arm0 representative differs", result.stderr)

    def test_rejects_old_dog_only_local_result_without_full_world8(self) -> None:
        fixture = ParentFixture(self)
        unsigned = dict(fixture.result)
        unsigned.pop("digest")
        unsigned.pop("world8_full_results")
        fixture.result = seal(unsigned)
        fixture.write_result(0, fixture.result)
        result = fixture.run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("runner diagnostic/base closure differs", result.stderr)


def terminal_receipt(runtime_sha: str, plan: dict[str, object]) -> bytes:
    artifacts = plan["release"]["artifacts"]
    core = {
        "schema_version": "bernini-graft-a-lite-source-independent-sacct-admission-v1",
        "status": "admitted",
        "materializer": {
            "schema_version": "bernini-graft-independent-sacct-admission-materializer-v1",
            "implementation_sha256": MATERIALIZER_SHA,
            "runtime_sha256": runtime_sha,
            "independent_of_submitted_job_process": True,
            "job_process_wrote_this_receipt": False,
            "observed_after_job_became_terminal": True,
        },
        "sacct_admission": {
            "source": "sacct",
            "queried_fields": ["JobIDRaw", "State", "ExitCode"],
            "job_id": "132549",
            "state": "COMPLETED",
            "exit_code": "0:0",
            "terminal_state_observed": True,
            "job_success": True,
            "raw_stdout_sha256": "a" * 64,
            "raw_stdout_size_bytes": 128,
            "selected_record_sha256": "b" * 64,
        },
        "artifact_bindings": {
            "manifest_file_sha256": artifacts["manifest"]["sha256"],
            "producer_receipt_file_sha256": artifacts["producer"]["sha256"],
            "producer_receipt_digest": "c" * 64,
            "execution_receipt_file_sha256": artifacts["execution"]["sha256"],
            "execution_receipt_digest": "d" * 64,
            "submission_receipt_file_sha256": artifacts["submission"]["sha256"],
            "submission_receipt_digest": "e" * 64,
        },
        "authority": {name: False for name in (
            "action_authority", "identity_authority", "cross_clip_identity_authority",
            "quality_authority", "training_authority", "production_authority",
            "data_governance_authority", "data_license_authority",
            "scientific_success_claimed",
        )},
    }
    return canonical(seal(core, field="receipt_digest")) + b"\n"


class SubmitFixture:
    def __init__(self, test: unittest.TestCase, *, behavior: str = "success") -> None:
        temporary = tempfile.TemporaryDirectory()
        test.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.inputs = self.root / "inputs"
        self.outputs = self.root / "outputs"
        self.inputs.mkdir()
        self.outputs.mkdir()
        self.observation = self.root / "observation.json"
        self.runtime_sha = "d" * 64

        self.launcher = self.inputs / "launcher.sbatch"
        self.launcher.write_bytes(LAUNCHER.read_bytes())
        self.launcher.chmod(0o555)
        self.launcher_sha = sha(self.launcher.read_bytes())

        plan = json.loads(PLAN.read_bytes())
        self.terminal = self.inputs / "terminal.json"
        plan["release"]["terminal_admission"]["path"] = str(self.terminal)
        self.plan = self.inputs / "plan.json"
        self.plan.write_bytes(canonical(plan) + b"\n")
        self.plan.chmod(0o444)
        self.plan_sha = sha(self.plan.read_bytes())
        self.terminal.write_bytes(terminal_receipt(self.runtime_sha, plan))
        self.terminal.chmod(0o444)

        self.archive = self.inputs / "runtime.tar"
        self.archive.write_bytes(b"archive")
        self.archive.chmod(0o444)
        self.closure = self.inputs / "closure.json"
        self.closure.write_bytes(b"closure\n")
        self.closure.chmod(0o444)
        self.checkpoint_manifest = self.inputs / "checkpoint.sha256"
        self.checkpoint_manifest.write_bytes(CHECKPOINT_MANIFEST.read_bytes())
        self.checkpoint_manifest.chmod(0o444)
        self.bernini = self.root / "bernini"
        self.veomni = self.root / "veomni"
        self.checkpoint = self.root / "checkpoint"
        for directory in (self.bernini, self.veomni, self.checkpoint):
            directory.mkdir()
        self.output = self.outputs / "short-run"

        python = Path(sys.executable).resolve(strict=True)
        fake = self.inputs / "sbatch"
        wrapper_path = self.inputs / "submit.sh"
        fake.write_text(
            f"#!{python}\n"
            "import hashlib,json,os\n"
            "from pathlib import Path\n"
            "import sys\n"
            f"behavior={behavior!r}\n"
            f"observation=Path({str(self.observation)!r})\n"
            f"launcher=Path({str(self.launcher)!r})\n"
            f"wrapper=Path({str(wrapper_path)!r})\n"
            "fd=Path(sys.argv[-1])\n"
            "observation.write_text(json.dumps({'argv':sys.argv[1:],'environment':dict(os.environ),'launcher_sha256':hashlib.sha256(fd.read_bytes()).hexdigest()},sort_keys=True,separators=(',',':')))\n"
            "if behavior=='bad_exit': raise SystemExit(7)\n"
            "if behavior=='replace_launcher':\n"
            " launcher.unlink(); launcher.write_bytes(b'attacker'); launcher.chmod(0o555)\n"
            "if behavior=='replace_wrapper':\n"
            " wrapper.unlink(); wrapper.write_bytes(b'attacker'); wrapper.chmod(0o444)\n"
            "if behavior=='bad_job': print('not-a-job')\n"
            "else: print('765432;testcluster')\n",
            encoding="utf-8",
        )
        fake.chmod(0o555)
        self.fake_sbatch = fake

        source = SUBMIT.read_text(encoding="utf-8")
        declared_launcher_sha = next(
            line.split("=", 1)[1]
            for line in source.splitlines()
            if line.startswith("readonly required_launcher_sha256=")
        )
        source = source.replace(declared_launcher_sha, self.launcher_sha)
        source = source.replace(
            "readonly required_sbatch_path=/usr/bin/sbatch",
            f"readonly required_sbatch_path={fake}",
        ).replace(
            "readonly required_fd_root=/proc/self/fd",
            "readonly required_fd_root=/dev/fd",
        ).replace(
            "readonly required_fd_stat_identity=true",
            "readonly required_fd_stat_identity=false",
        ).replace(
            "readonly required_execute_sbatch_from_fd=true",
            "readonly required_execute_sbatch_from_fd=false",
        ).replace(PLAN_SHA, self.plan_sha)
        self.wrapper = wrapper_path
        self.wrapper.write_text(source, encoding="utf-8")
        self.wrapper.chmod(0o444)

    def environment(self, **extra: str) -> dict[str, str]:
        python = Path(sys.executable).resolve(strict=True)
        values = {
            "GRAFT_SHORT_SOURCE_ARCHIVE": str(self.archive),
            "GRAFT_SHORT_SOURCE_ARCHIVE_SHA256": sha(self.archive.read_bytes()),
            "GRAFT_SHORT_RUNTIME_CLOSURE_MANIFEST": str(self.closure),
            "GRAFT_SHORT_RUNTIME_CLOSURE_MANIFEST_SHA256": sha(self.closure.read_bytes()),
            "GRAFT_SHORT_PYTHON_BIN": str(python),
            "GRAFT_SHORT_PYTHON_SHA256": sha(python.read_bytes()),
            "BERNINI_OFFICIAL_ROOT": str(self.bernini),
            "BERNINI_VEOMNI_ROOT": str(self.veomni),
            "BERNINI_ACTION_CHECKPOINT": str(self.checkpoint),
            "BERNINI_CHECKPOINT_CONTENT_MANIFEST": str(self.checkpoint_manifest),
            "BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256": CHECKPOINT_MANIFEST_SHA,
            "GRAFT_SHORT_PLAN": str(self.plan),
            "GRAFT_SHORT_PLAN_SHA256": self.plan_sha,
            "GRAFT_SHORT_TERMINAL_ADMISSION": str(self.terminal),
            "GRAFT_SHORT_TERMINAL_ADMISSION_SHA256": sha(self.terminal.read_bytes()),
            "GRAFT_SHORT_TERMINAL_MATERIALIZER_RUNTIME_SHA256": self.runtime_sha,
            "GRAFT_SHORT_OUTPUT_ROOT": str(self.output),
            "GRAFT_SHORT_LAUNCHER_SOURCE": str(self.launcher),
            "GRAFT_SHORT_LAUNCHER_SHA256": self.launcher_sha,
            "GRAFT_SHORT_SUBMIT_WRAPPER_SHA256": sha(self.wrapper.read_bytes()),
        }
        values.update(extra)
        return values

    def run(self, *args: str, environment: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["/bin/bash", "-p", str(self.wrapper), *args],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self.environment() if environment is None else environment,
            check=False,
            timeout=30,
        )

    @property
    def receipt(self) -> Path:
        return self.outputs / "short-run.submission.receipt.json"


class SubmissionBoundaryTests(unittest.TestCase):
    def test_success_exact_env_fd_launcher_and_submission_only_receipt(self) -> None:
        fixture = SubmitFixture(self)
        result = fixture.run()
        self.assertEqual(result.returncode, 0, result.stderr)
        observation = json.loads(fixture.observation.read_bytes())
        observed_environment = dict(observation["environment"])
        # Some macOS Python builds synthesize this CoreFoundation key in the
        # child after subprocess was given the exact env mapping.  It is not a
        # scheduler export and is absent on AUH/Linux.
        cf_hint = observed_environment.pop("__CF_USER_TEXT_ENCODING", None)
        if cf_hint is not None:
            self.assertRegex(cf_hint, r"^0x[0-9A-Fa-f]+:0x[0-9A-Fa-f]+:0x[0-9A-Fa-f]+$")
        self.assertEqual(
            set(observed_environment), {"PATH", "LC_ALL", "LANG", *EXPORT_NAMES}
        )
        self.assertEqual(observation["launcher_sha256"], fixture.launcher_sha)
        self.assertTrue(observation["argv"][-1].startswith("/dev/fd/"))
        raw = fixture.receipt.read_bytes()
        receipt = json.loads(raw)
        self.assertEqual(raw, canonical(receipt) + b"\n")
        unsigned = dict(receipt)
        claimed = unsigned.pop("receipt_digest")
        self.assertEqual(claimed, sha(canonical(unsigned)))
        self.assertTrue(receipt["submission_success"])
        self.assertIsNone(receipt["job_success"])
        self.assertFalse(receipt["job_terminal_state_observed"])
        terminal = receipt["submission_boundary"]["terminal_admission"]
        self.assertTrue(terminal["completed_0_0_observed_before_sbatch"])
        self.assertFalse(terminal["trust_anchor_computed_from_receipt"])
        bootstrap = receipt["submission_boundary"]["bootstrap_trust_boundary"]
        self.assertEqual(
            bootstrap["submit_wrapper_sha256_external_trust_anchor"],
            sha(fixture.wrapper.read_bytes()),
        )
        self.assertTrue(
            bootstrap["retained_submit_wrapper_bytes_matched_external_anchor"]
        )
        self.assertFalse(bootstrap["submit_wrapper_pre_exec_formal_security_proven"])
        self.assertFalse(bootstrap["python_pre_exec_formal_security_proven"])
        self.assertNotIn(
            "GRAFT_SHORT_SUBMIT_WRAPPER_SHA256",
            receipt["submission_boundary"]["exact_job_export_names"],
        )
        self.assertEqual(
            receipt["submission_boundary"]["exact_supervisor_interface_names"],
            [*EXPORT_NAMES, "GRAFT_SHORT_SUBMIT_WRAPPER_SHA256"],
        )
        self.assertFalse(
            receipt["failure_semantics"][
                "failure_absence_of_any_receipt_inode_guaranteed"
            ]
        )
        self.assertTrue(
            receipt["failure_semantics"][
                "provisional_non_success_inode_may_survive_cleanup_failure"
            ]
        )
        self.assertEqual(stat.S_IMODE(fixture.receipt.stat().st_mode), 0o444)

    def test_no_args_and_exact_interface(self) -> None:
        fixture = SubmitFixture(self)
        result = fixture.run("forbidden")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(fixture.observation.exists())

    def test_submit_wrapper_external_pin_is_required_and_fail_closed(self) -> None:
        for case in ("missing", "wrong"):
            with self.subTest(case=case):
                fixture = SubmitFixture(self)
                environment = fixture.environment()
                if case == "missing":
                    environment.pop("GRAFT_SHORT_SUBMIT_WRAPPER_SHA256")
                else:
                    environment["GRAFT_SHORT_SUBMIT_WRAPPER_SHA256"] = "e" * 64
                result = fixture.run(environment=environment)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(fixture.observation.exists())
                self.assertFalse(fixture.receipt.exists())

    def test_submit_wrapper_mode_and_link_count_are_fail_closed(self) -> None:
        for case in ("mode", "link"):
            with self.subTest(case=case):
                fixture = SubmitFixture(self)
                if case == "mode":
                    fixture.wrapper.chmod(0o644)
                else:
                    os.link(fixture.wrapper, fixture.inputs / "submit-hardlink")
                environment = fixture.environment()
                result = fixture.run(environment=environment)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(fixture.observation.exists())
                self.assertFalse(fixture.receipt.exists())
        environment = fixture.environment(GRAFT_SHORT_UNEXPECTED="poison")
        result = fixture.run(environment=environment)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(fixture.observation.exists())

    def test_path_ld_python_sbatch_bash_poison_not_forwarded(self) -> None:
        fixture = SubmitFixture(self)
        marker = fixture.root / "bash-env-ran"
        bash_env = fixture.root / "bash-env"
        bash_env.write_text(f"/usr/bin/touch {marker}\n", encoding="utf-8")
        environment = fixture.environment(
            PATH="/attacker",
            LD_PRELOAD="/attacker.so",
            PYTHONPATH="/attacker",
            SBATCH_EXPORT="ALL",
            BASH_ENV=str(bash_env),
            **{"BASH_FUNC_exec%%": "() { /usr/bin/false; }"},
        )
        result = fixture.run(environment=environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(marker.exists())
        observed = dict(json.loads(fixture.observation.read_bytes())["environment"])
        cf_hint = observed.pop("__CF_USER_TEXT_ENCODING", None)
        if cf_hint is not None:
            self.assertRegex(cf_hint, r"^0x[0-9A-Fa-f]+:0x[0-9A-Fa-f]+:0x[0-9A-Fa-f]+$")
        self.assertEqual(set(observed), {"PATH", "LC_ALL", "LANG", *EXPORT_NAMES})

    def test_terminal_sha_runtime_state_mode_and_link_count_fail_before_sbatch(self) -> None:
        cases = ("sha", "runtime", "state", "authority", "extra", "mode", "link")
        for case in cases:
            with self.subTest(case=case):
                fixture = SubmitFixture(self)
                environment = fixture.environment()
                if case == "sha":
                    environment["GRAFT_SHORT_TERMINAL_ADMISSION_SHA256"] = "e" * 64
                elif case == "runtime":
                    environment["GRAFT_SHORT_TERMINAL_MATERIALIZER_RUNTIME_SHA256"] = "e" * 64
                elif case == "state":
                    raw = json.loads(fixture.terminal.read_bytes())
                    raw.pop("receipt_digest")
                    raw["sacct_admission"]["state"] = "FAILED"
                    fixture.terminal.chmod(0o644)
                    fixture.terminal.write_bytes(canonical(seal(raw, field="receipt_digest")) + b"\n")
                    fixture.terminal.chmod(0o444)
                    environment["GRAFT_SHORT_TERMINAL_ADMISSION_SHA256"] = sha(fixture.terminal.read_bytes())
                elif case in {"authority", "extra"}:
                    raw = json.loads(fixture.terminal.read_bytes())
                    raw.pop("receipt_digest")
                    if case == "authority":
                        raw["authority"]["training_authority"] = True
                    else:
                        raw["unknown_terminal_claim"] = False
                    fixture.terminal.chmod(0o644)
                    fixture.terminal.write_bytes(
                        canonical(seal(raw, field="receipt_digest")) + b"\n"
                    )
                    fixture.terminal.chmod(0o444)
                    environment["GRAFT_SHORT_TERMINAL_ADMISSION_SHA256"] = sha(
                        fixture.terminal.read_bytes()
                    )
                elif case == "mode":
                    fixture.terminal.chmod(0o644)
                else:
                    os.link(fixture.terminal, fixture.inputs / "terminal-hardlink")
                result = fixture.run(environment=environment)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(fixture.observation.exists())
                self.assertFalse(fixture.receipt.exists())

    def test_sbatch_failure_or_bad_job_never_publishes_receipt(self) -> None:
        for behavior in ("bad_exit", "bad_job"):
            with self.subTest(behavior=behavior):
                fixture = SubmitFixture(self, behavior=behavior)
                result = fixture.run()
                self.assertNotEqual(result.returncode, 0)
                self.assertTrue(fixture.observation.exists())
                self.assertFalse(fixture.receipt.exists())

    def test_retained_launcher_or_wrapper_replacement_detected_after_submission(self) -> None:
        for behavior in ("replace_launcher", "replace_wrapper"):
            with self.subTest(behavior=behavior):
                fixture = SubmitFixture(self, behavior=behavior)
                result = fixture.run()
                self.assertNotEqual(result.returncode, 0)
                self.assertTrue(fixture.observation.exists())
                self.assertFalse(fixture.receipt.exists())

    def test_create_only_output_and_submission_receipt(self) -> None:
        fixture = SubmitFixture(self)
        fixture.output.mkdir()
        result = fixture.run()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(fixture.observation.exists())
        fixture = SubmitFixture(self)
        fixture.receipt.write_text("occupied", encoding="utf-8")
        result = fixture.run()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(fixture.observation.exists())


if __name__ == "__main__":
    unittest.main()
