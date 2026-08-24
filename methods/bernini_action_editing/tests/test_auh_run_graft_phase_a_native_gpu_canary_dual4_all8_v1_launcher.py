#!/usr/bin/env python3
"""Hostile CPU contracts for the sealed GRAFT Phase-A all8 launcher."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    METHOD_ROOT
    / "scripts/auh_run_graft_phase_a_native_gpu_canary_dual4_all8_v1.sbatch"
)
SOURCE_COMMIT = "00f7aba4bd58e67778542273cfca82fb785b648c"


def launcher_source() -> str:
    return LAUNCHER.read_text(encoding="utf-8")


def archive_preflight_source() -> str:
    source = launcher_source()
    marker = "# This is the complete local Python runtime closure"
    start = source.index(marker)
    heredoc = source.index("<<'PY'", start) + len("<<'PY'")
    end = source.index("\nPY\n", heredoc)
    return source[heredoc:end].lstrip("\n")


def vendor_audit_source() -> str:
    source = launcher_source()
    start = source.index("audit_vendor_tree() {")
    end = source.index('\naudit_vendor_tree "${bernini_root}"', start)
    return source[start:end]


def run_vendor_audit(root: Path, expected_commit: str) -> subprocess.CompletedProcess[str]:
    harness = f'''set -Eeuo pipefail
fail() {{ echo "$*" >&2; exit 2; }}
{vendor_audit_source()}
task_scratch="$3"
audit_vendor_tree "$1" "$2" TestVendor
'''
    return subprocess.run(
        [
            "/bin/bash",
            "-c",
            harness,
            "vendor-audit",
            str(root),
            expected_commit,
            str(root.parent),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=15,
    )


def initialize_vendor_repo(root: Path) -> str:
    subprocess.run(["/usr/bin/git", "init", "-q", str(root)], check=True)
    (root / "tracked.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / ".gitignore").write_text(
        "ignored-link\n__pycache__/\npipe.py\nnative.so/\n",
        encoding="utf-8",
    )
    subprocess.run(["/usr/bin/git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "/usr/bin/git",
            "-C",
            str(root),
            "-c",
            "user.name=GRAFT Test",
            "-c",
            "user.email=graft-test@example.invalid",
            "commit",
            "-q",
            "-m",
            "base",
        ],
        check=True,
    )
    return subprocess.run(
        ["/usr/bin/git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def expected_closure() -> dict[str, str]:
    tree = ast.parse(archive_preflight_source())
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "expected"
        ):
            value = ast.literal_eval(node.value)
            assert isinstance(value, dict)
            return value
    raise AssertionError("embedded committed closure map absent")


def canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def closure_manifest(files: dict[str, str], **updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "bernini-graft-phase-a-runtime-python-closure-v1",
        "root": "methods/bernini_action_editing",
        "selection": "commit-00f7aba-explicit-runtime-import-closure-v1",
        "source_git_commit": SOURCE_COMMIT,
        "files": files,
    }
    value.update(updates)
    return value


def build_archive(
    path: Path,
    files: dict[str, str],
    *,
    omit: str | None = None,
    extra_directory: bool = False,
) -> None:
    with tarfile.open(path, "w") as handle:
        if extra_directory:
            directory = tarfile.TarInfo("methods/bernini_action_editing/")
            directory.type = tarfile.DIRTYPE
            directory.mode = 0o555
            handle.addfile(directory)
        for relative, digest in sorted(files.items()):
            if relative == omit:
                continue
            payload = (METHOD_ROOT / relative).read_bytes()
            if hashlib.sha256(payload).hexdigest() != digest:
                raise AssertionError(f"workspace bytes differ from commit closure: {relative}")
            member = tarfile.TarInfo(
                f"methods/bernini_action_editing/{relative}"
            )
            member.size = len(payload)
            member.mode = 0o444
            handle.addfile(member, io.BytesIO(payload))


def run_archive_preflight(
    manifest_value: dict[str, object],
    *,
    omit: str | None = None,
    extra_directory: bool = False,
) -> subprocess.CompletedProcess[str]:
    files = expected_closure()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        archive = root / "runtime.tar"
        manifest = root / "closure.json"
        destination = root / "out"
        destination.mkdir()
        build_archive(
            archive,
            files,
            omit=omit,
            extra_directory=extra_directory,
        )
        manifest.write_bytes(canonical(manifest_value))
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
                hashlib.sha256(archive.read_bytes()).hexdigest(),
                hashlib.sha256(manifest.read_bytes()).hexdigest(),
            ],
            input=archive_preflight_source(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )


def parent_namespace() -> dict[str, object]:
    source = launcher_source()
    start_marker = "# BEGIN GRAFT_PHASE_A_PARENT_VALIDATOR_V1"
    end_marker = "# END GRAFT_PHASE_A_PARENT_VALIDATOR_V1"
    start = source.index(start_marker)
    end = source.index(end_marker, start) + len(end_marker)
    namespace: dict[str, object] = {}
    exec(compile(source[start:end], str(LAUNCHER), "exec"), namespace)
    return namespace


def seal_child(value: dict[str, object]) -> dict[str, object]:
    output = dict(value)
    output["receipt_digest"] = hashlib.sha256(
        json.dumps(
            output,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    return output


def tensor_identity(dtype: str) -> dict[str, object]:
    return {
        "shape": [],
        "dtype": dtype,
        "finite": True,
        "raw_sha256": "1" * 64,
        "content_sha256": "2" * 64,
    }


def child_receipt(cell: str) -> dict[str, object]:
    iid, digest, seed = {
        "dog": (
            "7b88a1ca1f804f41",
            "4d0c5cdfa9e0aae394af34a5bdda7de82ac770cd62cddbf3173ad2378458f3ed",
            2026080825,
        ),
        "human": (
            "a35b590961d24694",
            "6e9381d3889437f618e1ec6b694703b10598c4b42d8b361b0442db7780be97ed",
            2026080827,
        ),
    }[cell]
    core = {
        "guidance_mode": "v2v_apg",
        "raw_output_dtype": "torch.bfloat16",
        "raw_cotangent_dtype": "torch.bfloat16",
        "per_branch_raw_replay_exact": [True, True],
        "replay_visual_pack_detached_leaf": True,
        "replay_pack_gradient_cleared_after_each_branch": True,
    }
    authority = {
        "wiring_canary": True,
        "flow_matching_gradient_canary": True,
        "semantic_success": False,
        "action_success": False,
        "quality_success": False,
        "semantic_action_success": False,
        "visual_quality_success": False,
        "beneficial_training_evidence": False,
        "training_positive": False,
        "training_run": False,
        "optimizer_created": False,
        "optimizer_step": False,
        "parameters_updated": False,
        "scientific_claim_authorized": False,
        "production_claim_authorized": False,
        "full_sampler_executed": False,
        "full_sampler_parity": False,
    }
    return seal_child(
        {
            "schema_version": "bernini-graft-phase-a-native-gpu-canary-v1",
            "complete": True,
            "pass": True,
            "cell": {
                "cell_id": cell,
                "source_iid": iid,
                "source_video_sha256": digest,
                "noise_base_seed": seed,
                "frame_count": 81,
                "latent_phases": 21,
                "source_metadata": {"frame_count": 81, "fps": 25.0},
            },
            "source_only_phase_a": {
                "source_video_used": True,
                "target_video_used": False,
                "generated_proposal_used": False,
                "source_retelling_used": False,
                "proposal_selection_used": False,
                "phase_b_only": True,
                "canonical_noop_r2v": {
                    "source_only": True,
                    "action_instruction_used": False,
                    "guidance_mode": "v2v_apg",
                },
            },
            "schedule": {
                "selected_coordinate": {
                    "schedule_index": 33,
                    "schedule_steps": 40,
                    "timestep_dtype": "torch.int64",
                }
            },
            "native_phase_a": {
                "vendor_apg_leaf_vjp": True,
                "flow_matching_objective": "same_source_noop_velocity_mean_mse",
                "local_receipt": {
                    "phase_core_receipt": core,
                    "flow_matching_loss": tensor_identity("torch.float32"),
                },
                "world4_locality": {
                    "local_shard_rows_N": 7,
                    "observed_local_target_rows": [0, 0, 7, 7],
                    "replay_native_pack_leaf_all_ranks": True,
                    "zero_target_ranks_adapter_absent_or_zero": True,
                    "target_ranks_output_projection_local_grad_nonzero": True,
                    "target_ranks_qkv_and_atlas_local_grad_exact_zero": True,
                },
                "synchronized_gradients": {
                    "all_rank_exact_after_sync": True,
                    "reduction": "SUM_then_divide_by_WORLD4",
                    "total_l2_float64_hex": float(2.0).hex(),
                },
                "zero_initialized_gradient_gate": {
                    "gate": "output_projection_only_nonzero",
                    "output_projection_nonzero": True,
                    "query_key_value_exact_zero": True,
                    "external_atlas_encoder_exact_zero": True,
                },
            },
            "parameter_closure": {
                "trainable_bytes_unchanged": True,
                "frozen_base_bytes_unchanged": True,
                "frozen_base_gradients_all_none": True,
                "frozen_base_sha256_before": "3" * 64,
                "frozen_base_sha256_after": "3" * 64,
            },
            "output_directory_identity": {
                "st_dev": 1,
                "st_ino": 2,
                "created_fresh": True,
            },
            "provenance": {
                "runner_sha256": "a" * 64,
                "phase_a_closure_sha256": "b" * 64,
                "identity_rebinder_sha256": "c" * 64,
                "bernini_commit": "d" * 40,
                "veomni_commit": "e" * 40,
                "checkpoint_tree_sha256": "f" * 64,
                "checkpoint_content": {
                    "identity": {
                        "manifest_sha256_computed": "9" * 64,
                        "manifest_sha256_expected": "9" * 64,
                    }
                },
            },
            "authority": authority,
        }
    )


def parent_provenance() -> dict[str, str]:
    return {
        "runner_sha256": "a" * 64,
        "phase_a_closure_sha256": "b" * 64,
        "identity_rebinder_sha256": "c" * 64,
        "bernini_commit": "d" * 40,
        "veomni_commit": "e" * 40,
        "checkpoint_tree_sha256": "f" * 64,
        "checkpoint_content_manifest_sha256": "9" * 64,
        "checkpoint_content_pre_sha256": "8" * 64,
        "checkpoint_content_post_sha256": "8" * 64,
    }


class ShellAndTopologyTests(unittest.TestCase):
    def test_every_embedded_python_program_compiles(self) -> None:
        programs = re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", launcher_source(), re.DOTALL)
        self.assertEqual(len(programs), 6)
        for index, program in enumerate(programs):
            compile(program, f"{LAUNCHER}:heredoc-{index}", "exec")

    def test_shell_syntax_exact_all8_mapping_and_concurrency(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        source = launcher_source()
        self.assertIn("#SBATCH --gres=gpu:mi210:8", source)
        self.assertIn("--nproc_per_node=4", source)
        self.assertNotIn("--nproc_per_node=8", source)
        dog = source.index("run_group dog 0,1,2,3")
        human = source.index("run_group human 4,5,6,7")
        wait = source.index('wait "${dog_pid}"')
        self.assertLess(dog, wait)
        self.assertLess(human, wait)
        self.assertIn('export ROCR_VISIBLE_DEVICES="${visible_gpus}"', source)

    def test_preflight_is_before_world4_and_output_must_be_fresh(self) -> None:
        source = launcher_source()
        preflight = source.index("source video is not exact81/25")
        first_world4 = source.index("run_group dog 0,1,2,3")
        self.assertLess(preflight, first_world4)
        self.assertIn(
            '[[ ! -e "${output_root}" && ! -L "${output_root}" && "${output_root}" != / ]]',
            source,
        )
        self.assertLess(source.index("output root must be fresh"), first_world4)
        self.assertNotIn("/usr/bin/ffprobe", source)
        self.assertIn("materialize_vae._decode_exact_video(path)", source)

    def test_fresh_scratch_cwd_precedes_every_python_and_world4_isolated(self) -> None:
        source = launcher_source()
        scratch = source.index('task_scratch="$(/usr/bin/mktemp -d')
        cwd = source.index('builtin cd -- "${task_scratch}"', scratch)
        cwd_identity = source.index("fresh task scratch cwd identity differs", cwd)
        first_python = source.index('"${python_bin}" -I -S -B - "$1"')
        self.assertLess(scratch, cwd)
        self.assertLess(cwd, cwd_identity)
        self.assertLess(cwd_identity, first_python)
        self.assertIn('if ! "${python_bin}" -I -B - "${method_root}"', source)
        self.assertIn(
            'exec "${python_bin}" -I -B -m torch.distributed.run', source
        )
        self.assertNotIn('"${python_bin}" -B -', source)
        self.assertIn("scratch_cwd_stable=", source)

        with tempfile.TemporaryDirectory() as directory:
            probe = subprocess.run(
                [
                    "/bin/bash",
                    "-p",
                    "-c",
                    'target="$1"; expected="$(builtin cd -P -- "$target"; pwd -P)"; '
                    'builtin cd -- "$target"; [[ "$(pwd -P)" == "$expected" ]]',
                    "graft-cwd-probe",
                    directory,
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        self.assertEqual(probe.returncode, 0, probe.stderr)

    def test_privileged_shell_and_python_path_poison_are_closed_early(self) -> None:
        source = launcher_source()
        self.assertTrue(source.startswith("#!/bin/bash -p\n"))
        self.assertLess(source.index("Bash privileged mode is required"), source.index("source_archive="))
        self.assertLess(
            source.index("${!LD_*} ${!PYTHON*} ${!SBATCH_*}"),
            source.index("source_archive="),
        )
        self.assertLess(
            source.index("export PATH=/usr/bin:/bin"),
            source.index("source_archive="),
        )

    def test_failure_preserves_scratch_and_output_without_destructive_cleanup(self) -> None:
        source = launcher_source()
        self.assertIn("task_scratch_identity=", source)
        self.assertIn(
            '[[ "${observed}" == "${task_scratch_identity}" ]] && scratch_identity_stable=true',
            source,
        )
        self.assertNotIn('rm -rf -- "${task_scratch}"', source)
        self.assertNotIn('rm -rf -- "${output_root}"', source)
        self.assertIn("scratch_preserved=", source)
        self.assertIn("output_retained=true", source)

    def test_first_failed_arm_kills_and_waits_sibling(self) -> None:
        source = launcher_source()
        self.assertIn(
            'wait -n -p finished_pid "${dog_pid}" "${human_pid}"', source
        )
        failure = source.index("if (( first_status != 0 ))")
        kill = source.index('kill "${dog_pid}" "${human_pid}"', failure)
        wait_dog = source.index('wait "${dog_pid}"', kill)
        wait_human = source.index('wait "${human_pid}"', wait_dog)
        fail = source.index("sibling terminated", wait_human)
        self.assertLess(kill, wait_dog)
        self.assertLess(wait_dog, wait_human)
        self.assertLess(wait_human, fail)
        self.assertIn(
            'exec "${python_bin}" -I -B -m torch.distributed.run', source
        )

    def test_runtime_first_failed_child_terminates_long_lived_sibling(self) -> None:
        source = launcher_source()
        start = source.index('finished_pid=""')
        end = source.index(
            'audit_vendor_tree "${bernini_root}"', start
        )
        wait_block = source[start:end]
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "human-terminated"
            harness = f'''set -uo pipefail
fail() {{ echo "$*" >&2; exit 2; }}
output_root={temporary!r}
( /bin/sleep 0.10; exit 7 ) &
dog_pid=$!
( trap '/usr/bin/touch {str(marker)!r}; exit 143' TERM; while true; do /bin/sleep 1; done ) &
human_pid=$!
{wait_block}
'''
            before = time.monotonic()
            result = subprocess.run(
                ["bash", "-c", harness],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
                check=False,
            )
            elapsed = time.monotonic() - before
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("sibling terminated", result.stderr)
            self.assertTrue(marker.is_file())
            self.assertLess(elapsed, 4.0)

    def test_release_and_extracted_modes_are_readonly(self) -> None:
        source = launcher_source()
        self.assertIn('source archive must be released 0444', source)
        self.assertIn('closure manifest must be released 0444', source)
        self.assertIn('launcher source must be released 0555', source)
        self.assertIn('os.chmod(directory, 0o555)', source)
        self.assertIn('os.chmod(runtime_root, 0o555)', source)
        self.assertIn('sealed runner mode differs', source)

    def test_vendor_gate_accepts_clean_plain_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "vendor"
            root.mkdir()
            commit = initialize_vendor_repo(root)
            result = run_vendor_audit(root, commit)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_vendor_gate_rejects_every_symlink_outside_git(self) -> None:
        for case in (
            "file",
            "directory",
            "ignored",
            "pycache",
            "tracked",
        ):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "vendor"
                root.mkdir()
                commit = initialize_vendor_repo(root)
                if case == "file":
                    (root / "file-link").symlink_to("tracked.py")
                elif case == "directory":
                    (root / "real-directory").mkdir()
                    (root / "directory-link").symlink_to(
                        "real-directory", target_is_directory=True
                    )
                elif case == "ignored":
                    (root / "ignored-link").symlink_to("tracked.py")
                elif case == "pycache":
                    (root / "__pycache__").mkdir()
                    (root / "__pycache__/shadow.pyc").symlink_to("../tracked.py")
                else:
                    (root / "tracked-link").symlink_to("tracked.py")
                    subprocess.run(
                        ["/usr/bin/git", "-C", str(root), "add", "tracked-link"],
                        check=True,
                    )
                    subprocess.run(
                        [
                            "/usr/bin/git",
                            "-C",
                            str(root),
                            "-c",
                            "user.name=GRAFT Test",
                            "-c",
                            "user.email=graft-test@example.invalid",
                            "commit",
                            "-q",
                            "-m",
                            "tracked symlink",
                        ],
                        check=True,
                    )
                    commit = subprocess.run(
                        ["/usr/bin/git", "-C", str(root), "rev-parse", "HEAD"],
                        check=True,
                        text=True,
                        stdout=subprocess.PIPE,
                    ).stdout.strip()
                result = run_vendor_audit(root, commit)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("symlink outside .git", result.stderr)

    def test_vendor_gate_rejects_nonplain_import_capable_nodes(self) -> None:
        for case in ("fifo", "directory"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "vendor"
                root.mkdir()
                commit = initialize_vendor_repo(root)
                if case == "fifo":
                    os.mkfifo(root / "pipe.py")
                else:
                    (root / "native.so").mkdir()
                result = run_vendor_audit(root, commit)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("non-plain import-capable node", result.stderr)

    def test_parent_reads_children_via_retained_openat_descriptors(self) -> None:
        source = launcher_source()
        self.assertIn('os.open(cell, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)', source)
        self.assertIn('os.open("receipt.json", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=cell_fd)', source)
        self.assertIn('child receipt/output directory binding differs', source)


class SealedClosureHostileTests(unittest.TestCase):
    def test_exact_minimal_committed_closure_passes(self) -> None:
        files = expected_closure()
        self.assertEqual(len(files), 19)
        result = run_archive_preflight(closure_manifest(files))
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_lazy_dependency_fails_closed(self) -> None:
        files = expected_closure()
        result = run_archive_preflight(
            closure_manifest(files), omit="tools/build_renderer_dataset.py"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertRegex(result.stderr, r"lacks committed dependencies")

    def test_manifest_schema_drift_fails_closed(self) -> None:
        files = expected_closure()
        result = run_archive_preflight(
            closure_manifest(files, selection="unsealed-broader-tree")
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertRegex(result.stderr, r"schema/file map differs")

    def test_extra_directory_member_fails_closed(self) -> None:
        files = expected_closure()
        result = run_archive_preflight(
            closure_manifest(files), extra_directory=True
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertRegex(result.stderr, r"exactly 19 regular file members")


class ParentTruthHostileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ns = parent_namespace()
        self.validate_child = self.ns["validate_child"]
        self.build_parent = self.ns["build_parent_receipt"]

    def test_valid_children_close_minimal_parent_and_all_authority_false(self) -> None:
        children = {cell: child_receipt(cell) for cell in ("dog", "human")}
        parent = self.build_parent(children, parent_provenance(), (1, 2))
        self.assertTrue(parent["pass"])
        self.assertEqual(
            parent["cells"][0]["observed_local_target_rows"], [0, 0, 7, 7]
        )
        self.assertTrue(all(value is False for value in parent["authority"].values()))
        self.assertRegex(parent["parent_receipt_digest"], r"^[0-9a-f]{64}$")

    def test_mutated_sp_ownership_pack_leaf_and_base_each_fail(self) -> None:
        for mutator in (
            lambda row: row["native_phase_a"]["world4_locality"].__setitem__(
                "observed_local_target_rows", [0, 7, 0, 7]
            ),
            lambda row: row["native_phase_a"]["world4_locality"].__setitem__(
                "replay_native_pack_leaf_all_ranks", False
            ),
            lambda row: row["parameter_closure"].__setitem__(
                "frozen_base_sha256_after", "4" * 64
            ),
        ):
            row = child_receipt("dog")
            row.pop("receipt_digest")
            mutator(row)
            row = seal_child(row)
            with self.assertRaises(ValueError):
                self.validate_child(row, "dog")

    def test_semantic_authority_escalation_fails(self) -> None:
        row = child_receipt("human")
        row.pop("receipt_digest")
        row["authority"]["semantic_success"] = True
        row = seal_child(row)
        with self.assertRaisesRegex(ValueError, "forbidden authority"):
            self.validate_child(row, "human")

    def test_child_provenance_substitution_fails_parent(self) -> None:
        children = {cell: child_receipt(cell) for cell in ("dog", "human")}
        dog = children["dog"]
        dog.pop("receipt_digest")
        dog["provenance"]["runner_sha256"] = "0" * 64
        children["dog"] = seal_child(dog)
        with self.assertRaisesRegex(ValueError, "child provenance differs"):
            self.build_parent(children, parent_provenance(), (1, 2))


if __name__ == "__main__":
    unittest.main()
