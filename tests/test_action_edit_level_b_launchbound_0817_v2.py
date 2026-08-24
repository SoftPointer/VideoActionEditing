from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
METHOD = ROOT / "methods" / "bernini_action_editing"
BOOTSTRAP_PATH = METHOD / "action_edit_level_b_p2_00435_bootstrap_0817_v2.py"
CONTROLLER_PATH = (
    METHOD / "scripts" / "auh_launch_action_edit_level_b_p2_00435_job140846_v2.sh"
)
STEP_PATH = METHOD / "scripts" / "auh_action_edit_level_b_p2_00435_step_v2.sh"
RANK_PATH = METHOD / "scripts" / "auh_action_edit_level_b_p2_00435_rank_exec_v2.sh"
RELEASE_PATH = (
    METHOD / "audits" / "fresh_world8_level_b_p2_00435_v2_RELEASE_MANIFEST.json"
)
CORE_PATH = (
    METHOD
    / "audits"
    / "fresh_world8_level_b_p2_00435_v2_LAUNCH_AUTHORITY_CORE.json"
)
PINS_PATH = (
    METHOD / "audits" / "fresh_world8_level_b_p2_00435_v2_DEPLOYMENT_PINS.json"
)
RENDERER_PATH = METHOD / "infer_action_edit_level_b_renderer_0817_v1.py"
RENDERER_TEST_PATH = ROOT / "tests" / "test_infer_action_edit_level_b_renderer_0817_v1.py"
THIS_TEST_PATH = Path(__file__).resolve()

RENDERER_SHA = "b07ec9603e4820b4b9cb52fe7c8994591e3e5e30afb71b504086b0d259c06d68"
RENDERER_SIZE = 248964
RENDERER_TEST_SHA = "fe11570fdddc9a35f2c9faba9b5b7f6fee9def671a9416c029bc2a37ac810f26"
RENDERER_TEST_SIZE = 109773


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json_bytes(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def load_bootstrap():
    name = "action_edit_level_b_p2_00435_bootstrap_0817_v2_test"
    spec = importlib.util.spec_from_file_location(name, BOOTSTRAP_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(), filename=str(path))
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in statement.targets):
                return ast.literal_eval(statement.value)
    raise AssertionError(f"missing literal assignment: {name}")


bootstrap = load_bootstrap()


class LevelBLaunchboundV2FrozenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bootstrap_text = BOOTSTRAP_PATH.read_text()
        cls.controller = CONTROLLER_PATH.read_text()
        cls.step = STEP_PATH.read_text()
        cls.rank = RANK_PATH.read_text()
        cls.release = json.loads(RELEASE_PATH.read_text())
        cls.core = json.loads(CORE_PATH.read_text())
        cls.pins = json.loads(PINS_PATH.read_text())
        cls.launch_text = "\n".join(
            (cls.bootstrap_text, cls.controller, cls.step, cls.rank)
        )

    def shell_pin(self, text: str, name: str) -> str:
        match = re.search(rf"^readonly {re.escape(name)}=([0-9a-f]{{64}})$", text, re.M)
        self.assertIsNotNone(match, name)
        return match.group(1)

    def test_renderer_and_compatibility_qa_bytes_are_exact(self):
        self.assertEqual(RENDERER_PATH.stat().st_size, RENDERER_SIZE)
        self.assertEqual(sha256(RENDERER_PATH), RENDERER_SHA)
        self.assertEqual(RENDERER_TEST_PATH.stat().st_size, RENDERER_TEST_SIZE)
        self.assertEqual(sha256(RENDERER_TEST_PATH), RENDERER_TEST_SHA)

    def test_launch_chain_is_fully_frozen_without_pending_tokens(self):
        bootstrap._require_finalized()
        frozen = self.launch_text + RELEASE_PATH.read_text() + CORE_PATH.read_text()
        self.assertNotIn("PENDING_FINAL_", frozen)
        self.assertEqual(bootstrap.LEVEL_B_RENDERER_SHA256, RENDERER_SHA)
        self.assertEqual(bootstrap.LEVEL_B_RENDERER_SIZE, RENDERER_SIZE)

    def test_release_is_canonical_exact_five_with_valid_digest(self):
        self.assertEqual(RELEASE_PATH.read_bytes(), canonical_json_bytes(self.release) + b"\n")
        self.assertEqual(
            set(self.release),
            {"schema_version", "authority", "member_count", "members", "release_digest"},
        )
        self.assertEqual(self.release["member_count"], 5)
        self.assertEqual(
            self.release["release_digest"],
            hashlib.sha256(canonical_json_bytes(self.release["members"])).hexdigest(),
        )
        self.assertEqual(sha256(RELEASE_PATH), bootstrap.LEVEL_B_MANIFEST_SHA256)

    def test_all_exact_five_release_members_match_frozen_local_bytes(self):
        self.assertEqual(
            [row["path"] for row in self.release["members"]],
            list(bootstrap.LEVEL_B_MEMBER_PINS),
        )
        for row in self.release["members"]:
            relative = row["path"]
            expected_sha, expected_size = bootstrap.LEVEL_B_MEMBER_PINS[relative]
            path = METHOD / relative
            self.assertEqual(row, {
                "path": relative,
                "sha256": expected_sha,
                "size": expected_size,
                "mode": 0o444,
            })
            self.assertEqual(path.stat().st_size, expected_size)
            self.assertEqual(sha256(path), expected_sha)

    def test_qa_test_is_recorded_but_excluded_from_exact_five(self):
        member_paths = {row["path"] for row in self.release["members"]}
        self.assertNotIn("tests/test_infer_action_edit_level_b_renderer_0817_v1.py", member_paths)
        self.assertIs(self.core["release"]["compatibility_qa_test_is_release_member"], False)
        self.assertEqual(self.core["release"]["compatibility_qa_test_sha256"], RENDERER_TEST_SHA)
        self.assertEqual(self.core["release"]["compatibility_qa_test_size"], RENDERER_TEST_SIZE)

    def test_semantic_input_is_one_source_instruction_and_seed(self):
        self.assertEqual(
            bootstrap.SOURCE_VIDEO_SHA256,
            "b9218921597e43e2a3a6b223899ab84fb1b8d1a51692766bb2167e5941efbba1",
        )
        self.assertEqual(bootstrap.SOURCE_INPUT_HW, (1056, 704))
        self.assertEqual(bootstrap.SOURCE_BUCKET_HW, (592, 400))
        self.assertEqual(bootstrap.SOURCE_FRAMES, 81)
        self.assertEqual(bootstrap.SOURCE_FPS, 25)
        self.assertEqual(bootstrap.INFERENCE_SEED, 2026080821)
        self.assertEqual(
            hashlib.sha256(bootstrap.EDIT_INSTRUCTION.encode("utf-8")).hexdigest(),
            bootstrap.EDIT_INSTRUCTION_SHA256,
        )
        call = re.search(
            r"run_level_b_pre_d0_offline_inference\((.*?)\n    \)",
            self.bootstrap_text,
            re.S,
        ).group(1)
        for forbidden in ("target", "anchor", "teacher", "pose", "mask", "flow"):
            self.assertNotIn(forbidden, call.lower())

    def test_core_binds_explicit_canonical_input_geometry_and_instruction(self):
        value = self.core["input"]
        self.assertEqual(value["sample_id"], "00435ad621c44fac")
        self.assertEqual(value["source_video_path"], str(bootstrap.SOURCE_VIDEO))
        self.assertEqual(value["source_video_sha256"], bootstrap.SOURCE_VIDEO_SHA256)
        self.assertEqual(value["source_video_size"], 7364420)
        self.assertEqual(value["source_input_hw"], [1056, 704])
        self.assertEqual(value["bucket_hw"], [592, 400])
        self.assertEqual(value["latent_shape"], [1, 16, 21, 74, 50])
        self.assertEqual(value["edit_instruction"], bootstrap.EDIT_INSTRUCTION)
        self.assertEqual(value["inference_seed"], bootstrap.INFERENCE_SEED)
        self.assertIs(value["target_or_anchor_input_present"], False)

    def test_frozen_level_a_exact_eleven_member_pins_match_local_bytes(self):
        self.assertEqual(len(bootstrap.LEVEL_A_MEMBER_PINS), 11)
        for relative, (expected_sha, expected_size) in bootstrap.LEVEL_A_MEMBER_PINS.items():
            path = METHOD / relative
            self.assertEqual(path.stat().st_size, expected_size)
            self.assertEqual(sha256(path), expected_sha)

    def test_level_a_r2_p2_and_base_trust_roots_are_exact(self):
        self.assertEqual(
            bootstrap.LEVEL_A_MANIFEST_SHA256,
            "f9e9f8542ec701cc9890fed919695980b989fd6d731eb914a5588edb1de4eeaa",
        )
        self.assertEqual(
            bootstrap.R2_RELEASE_MANIFEST_SHA256,
            "671179995a64f20ee773273e84b5eb3f1f0bbd018fbfa3c0c6dc41d56c5555f5",
        )
        self.assertEqual(
            bootstrap.P2_PARAMETER_SHA256,
            "5f9c31e84ab9ec4330b07d86cb1a2fc79c7aa365f4bf88a9cdffc0c244dcaa3e",
        )
        self.assertEqual(
            bootstrap.BASE_CHECKPOINT_TREE_SHA256,
            "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca",
        )

    def test_bootstrap_authenticates_release_before_source_exec(self):
        authenticate = self.bootstrap_text.index("def _authenticate_level_b_module")
        stable_read = self.bootstrap_text.index("renderer_raw = _stable_bytes", authenticate)
        seeded_sha = self.bootstrap_text.index(
            '"_LEVEL_B_SEALED_LAUNCHER_EXPECTED_MANIFEST_SHA256"', stable_read
        )
        compile_source = self.bootstrap_text.index("module = _load_module", stable_read)
        self.assertLess(stable_read, seeded_sha)
        self.assertLess(seeded_sha, compile_source)

    def test_bootstrap_uses_opaque_runtime_and_fresh_level_a_bundle(self):
        run = self.bootstrap_text[self.bootstrap_text.index("def run_world8") :]
        self.assertIn("authenticate_level_b_runtime_release", run)
        self.assertIn("consume_frozen_r2_world8_checkpoint", run)
        self.assertIn("fresh_bundle=bundle", run)
        self.assertIn("verified_runtime=verified_runtime", run)

    def test_world8_terminal_requires_full_renderer_and_nonpromotion(self):
        run = self.bootstrap_text[self.bootstrap_text.index("def run_world8") :]
        for token in (
            'receipt.get("full40_denoise_executed") is not True',
            'receipt.get("full_bernini_renderer_denoise_verified") is not True',
            'receipt.get("mp4_emitted") is not True',
            'receipt.get("promotion_authorized") is not False',
            'receipt.get("counts_as_d0") is not False',
        ):
            self.assertIn(token, run)

    def test_postcommit_validator_rechecks_exact_inode_alias_and_claims(self):
        validate = self.bootstrap_text[self.bootstrap_text.index("def validate_product") :]
        for token in (
            "source_video_sha256",
            "instruction_utf8_sha256",
            "clean_target_present",
            "anchor_present",
            "full40_denoise_executed",
            "exact30_target_only_action_hooks_once_per_denoise_step",
            "ffprobe_exact81",
            "full_decode_frame_count",
            "ffprobe_geometry_hw",
            "receipt_info.st_dev != marker_info.st_dev",
            "receipt_info.st_ino != marker_info.st_ino",
            "receipt_inode_alias_marker_revalidated",
        ):
            self.assertIn(token, validate)

    def test_plain_file_requires_explicit_double_link_authority(self):
        with tempfile.TemporaryDirectory() as raw:
            source = Path(raw).resolve() / "receipt.json"
            marker = Path(raw).resolve() / "marker.json"
            source.write_bytes(b"{}")
            source.chmod(0o444)
            os.link(source, marker)
            self.assertEqual(bootstrap._plain_file(source, mode=0o444, nlink=2), source)
            with self.assertRaises(bootstrap.BootstrapError):
                bootstrap._plain_file(source, mode=0o444)

    def test_controller_contains_one_srun_and_no_parent_control_or_remote_copy(self):
        self.assertEqual(self.controller.count("/usr/bin/srun"), 1)
        for forbidden in (
            "/usr/bin/scancel",
            " scontrol ",
            " requeue ",
            " kill ",
            " ssh ",
            " scp ",
            " rsync ",
        ):
            self.assertNotIn(forbidden, self.controller)
        self.assertIn('--jobid="${job_id}"', self.controller)
        self.assertIn('--nodelist="${node}"', self.controller)

    def test_controller_has_exact_node279_64g_world8_request(self):
        self.assertIn("readonly node=auh7-1b-gpu-279", self.controller)
        self.assertIn("--nodes=1 --ntasks=1", self.controller)
        self.assertIn("--cpus-per-task=32 --mem=64G", self.controller)
        self.assertIn("--gres=gpu:mi210:8", self.controller)
        self.assertEqual(self.core["resources"]["world_size"], 8)
        self.assertEqual(self.core["resources"]["host_memory_gib"], 64)

    def test_parent_snapshot_uses_observed_slurm_gres_colon_format(self):
        expected = (
            "readonly expected_parent_state="
            "'RUNNING|auh7-1b-gpu-[246-248,279]|gres/gpu:mi210:8'"
        )
        self.assertIn(expected, self.controller)
        self.assertNotIn("gres/gpu:mi210=8", self.controller)

    def test_controller_has_persistent_atomic_started_claim_and_no_retry(self):
        self.assertIn('mkdir -m 0700 "${started}"', self.controller)
        self.assertNotIn('rmdir "${started}"', self.controller)
        self.assertNotIn('rm -rf "${started}"', self.controller)
        self.assertIn("no retry is authorized", self.controller)
        self.assertIn('"max_restarts":0', self.controller)

    def test_numeric_child_closure_is_scoped_to_parent_job(self):
        for text in (self.controller, self.step):
            self.assertIn('squeue --steps -w "${node}" -h -j "${job_id}"', text)
            self.assertIn("^[0-9]+$", text)
        self.assertIn('"${sibling_steps}" == "${current_step}"', self.step)

    def test_step_enforces_physical_host_and_gpu_admission(self):
        self.assertIn('"${SLURM_MEM_PER_NODE:-}" == 65536', self.step)
        self.assertIn("MemTotal", self.step)
        self.assertIn("120 * 1024**3", self.step)
        self.assertIn("MemAvailable", self.step)
        self.assertIn("memory.max", self.step)
        self.assertIn("MALLOC_ARENA_MAX=2", self.step)
        self.assertIn("torch.cuda.device_count() == 8", self.step)
        self.assertIn('"MI210" in name', self.step)
        self.assertIn("free * 100 >= total * 95", self.step)
        self.assertIn("allocates no dummy tensors", self.step)

    def test_torchelastic_has_zero_restart_authority(self):
        self.assertIn("--nproc_per_node=8", self.step)
        self.assertIn("--max-restarts=0", self.step)
        self.assertIn('"${TORCHELASTIC_MAX_RESTARTS:-}" == 0', self.rank)
        self.assertIn('"${TORCHELASTIC_RESTART_COUNT:-}" == 0', self.rank)

    def test_rank_and_step_hash_chain_matches_frozen_local_bytes(self):
        self.assertEqual(self.shell_pin(self.rank, "bootstrap_sha"), sha256(BOOTSTRAP_PATH))
        self.assertEqual(self.shell_pin(self.step, "bootstrap_sha"), sha256(BOOTSTRAP_PATH))
        self.assertEqual(self.shell_pin(self.step, "rank_exec_sha"), sha256(RANK_PATH))
        self.assertEqual(
            self.shell_pin(self.step, "release_manifest_sha"), sha256(RELEASE_PATH)
        )
        self.assertEqual(self.shell_pin(self.step, "renderer_sha"), sha256(RENDERER_PATH))

    def canonical_intent(self):
        roots = self.core["roots"]
        return {
            "schema_version": "bernini-action-edit-level-b-p2-attempt-intent-v2",
            "method": bootstrap.METHOD,
            "authority": bootstrap.AUTHORITY,
            "tag": bootstrap.TAG,
            "parent_job_id": 140846,
            "node": "auh7-1b-gpu-279",
            "job_name": "bernini0817-level-b-p2-00435-v2",
            "release_root": roots["release"],
            "launch_root": roots["launch"],
            "attempt_root": roots["attempt"],
            "run_root": roots["run"],
            "output_mp4": str(bootstrap.OUTPUT_MP4),
            "source_video_sha256": bootstrap.SOURCE_VIDEO_SHA256,
            "instruction_utf8_sha256": bootstrap.EDIT_INSTRUCTION_SHA256,
            "inference_seed": bootstrap.INFERENCE_SEED,
            "checkpoint_step": 2,
            "checkpoint_parameter_sha256": bootstrap.P2_PARAMETER_SHA256,
            "release_manifest_sha256": sha256(RELEASE_PATH),
            "renderer_sha256": sha256(RENDERER_PATH),
            "bootstrap_sha256": sha256(BOOTSTRAP_PATH),
            "step_payload_sha256": sha256(STEP_PATH),
            "rank_exec_sha256": sha256(RANK_PATH),
            "world_size": 8,
            "dp_size": 2,
            "sp_size": 4,
            "host_memory_gib": 64,
            "max_restarts": 0,
            "committed_marker_required": True,
            "automatic_relaunch_authorized": False,
            "parent_control_authorized": False,
            "formal_training_started": False,
            "counts_as_d0": False,
            "promotion_authorized": False,
        }

    def test_runtime_intent_generator_emits_the_exact_pinned_canonical_bytes(self):
        start = self.controller.index('"${python_bin}" -I -B -c \'\n', self.controller.index("readonly intent_tmp"))
        code_start = start + len('"${python_bin}" -I -B -c \'\n')
        code_end = self.controller.index("\n' \"${release_root}\"", code_start)
        code = self.controller[code_start:code_end]
        expected = self.canonical_intent()
        roots = self.core["roots"]
        argv = [
            sys.executable,
            "-I",
            "-B",
            "-c",
            code,
            roots["release"],
            roots["launch"],
            roots["attempt"],
            roots["run"],
            str(bootstrap.OUTPUT_MP4),
            sha256(RELEASE_PATH),
            sha256(RENDERER_PATH),
            sha256(BOOTSTRAP_PATH),
            sha256(STEP_PATH),
            sha256(RANK_PATH),
        ]
        completed = subprocess.run(argv, capture_output=True, check=True)
        self.assertEqual(completed.stdout, canonical_json_bytes(expected))
        intent_sha = hashlib.sha256(completed.stdout).hexdigest()
        self.assertEqual(intent_sha, self.core["launchers"]["intent_sha256"])
        self.assertEqual(intent_sha, self.shell_pin(self.controller, "attempt_intent_sha"))

    def test_core_and_controller_hash_chain_matches_frozen_bytes(self):
        self.assertEqual(CORE_PATH.read_bytes(), canonical_json_bytes(self.core) + b"\n")
        self.assertEqual(
            self.shell_pin(self.controller, "launch_authority_core_sha"), sha256(CORE_PATH)
        )
        self.assertEqual(self.shell_pin(self.controller, "step_payload_sha"), sha256(STEP_PATH))
        self.assertEqual(self.shell_pin(self.controller, "rank_exec_sha"), sha256(RANK_PATH))
        self.assertEqual(self.shell_pin(self.controller, "bootstrap_sha"), sha256(BOOTSTRAP_PATH))
        self.assertEqual(
            self.shell_pin(self.controller, "release_manifest_sha"), sha256(RELEASE_PATH)
        )
        self.assertEqual(self.shell_pin(self.controller, "renderer_sha"), sha256(RENDERER_PATH))

    def test_launch_root_contract_is_exactly_five_members(self):
        match = re.search(
            r"readonly expected_launch_entries=\$'(.*?)'\n", self.controller, re.S
        )
        self.assertIsNotNone(match)
        self.assertEqual(
            match.group(1).split(r"\n"),
            [
                "LAUNCH_AUTHORITY_CORE.json",
                BOOTSTRAP_PATH.name,
                RANK_PATH.name,
                STEP_PATH.name,
                CONTROLLER_PATH.name,
            ],
        )

    def test_output_contract_requires_exact_receipt_inode_alias_marker(self):
        self.assertIn("expected_run_entries", self.controller)
        self.assertIn('stat -c %h "${output_mp4}")" == 1', self.controller)
        self.assertIn('stat -c %h "${output_receipt}")" == 2', self.controller)
        self.assertIn('stat -c %h "${output_marker}")" == 2', self.controller)
        self.assertIn("COMMITTED marker is not the exact receipt inode alias", self.controller)
        self.assertIn("receipt_inode_alias_marker_verified", self.controller)
        self.assertEqual(self.controller.count('"${bootstrap}" validate-product'), 2)

    def test_success_cannot_precede_committed_double_validation(self):
        marker = self.controller.index("expected_run_entries")
        alias = self.controller.index("exact receipt inode alias", marker)
        validate = self.controller.index("validate-product", alias)
        terminal = self.controller.index("terminal.authority.json", validate)
        success = self.controller.index("SUCCESS.$$.tmp", terminal)
        self.assertLess(marker, alias)
        self.assertLess(alias, validate)
        self.assertLess(validate, terminal)
        self.assertLess(terminal, success)

    def test_failure_writes_status_but_never_success(self):
        status = self.controller.index('write_status "${child_exit}"')
        failure = self.controller.index("if (( child_exit != 0 ))", status)
        success = self.controller.index("SUCCESS.$$.tmp", failure)
        self.assertLess(status, failure)
        self.assertLess(failure, success)

    def test_hostile_real_bash_child_failure_seals_status_and_never_retries(self):
        """Exercise the exact controller failure boundary in a hostile shell.

        The fixture deliberately defines the same module-scope readonly names
        that broke v1.  Slurm is replaced by one deterministic child returning
        37; all filesystem topology and hash gates before that child remain
        live.  This is a real Bash execution, not a source-order assertion.
        """
        with tempfile.TemporaryDirectory() as raw:
            sandbox = Path(raw).resolve()
            experiment = sandbox / "experiment"
            tag = "fresh-world8-level-b-p2-00435-v2"
            release_root = experiment / "releases" / tag
            launch_root = experiment / "launchers" / tag
            attempt_root = experiment / "attempts" / tag
            run_root = experiment / "runs" / tag
            shim_root = sandbox / "shim"
            for path, mode in (
                (release_root, 0o755),
                (launch_root, 0o755),
                (attempt_root, 0o700),
                (run_root, 0o700),
                (shim_root, 0o755),
            ):
                path.mkdir(parents=True, exist_ok=False)
                path.chmod(mode)

            def executable(path: Path, source: str) -> None:
                path.write_text(source)
                path.chmod(0o755)

            # Portable GNU-contract shims let the Linux controller run under
            # both the Darwin QA host and its intended Linux deployment host.
            executable(
                shim_root / "stat",
                f"#!{sys.executable}\n"
                "import os,stat,sys\n"
                "assert sys.argv[1] == '-c' and len(sys.argv) == 4\n"
                "info=os.lstat(sys.argv[3]); fmt=sys.argv[2]\n"
                "rows={'%a':format(stat.S_IMODE(info.st_mode),'o'),"
                "'%h':str(info.st_nlink),'%d:%i':f'{info.st_dev}:{info.st_ino}'}\n"
                "print(rows[fmt])\n",
            )
            executable(
                shim_root / "sha256sum",
                f"#!{sys.executable}\n"
                "import hashlib,pathlib,sys\n"
                "assert len(sys.argv) == 2\n"
                "p=pathlib.Path(sys.argv[1]); print(hashlib.sha256(p.read_bytes()).hexdigest(), p)\n",
            )
            executable(
                shim_root / "find",
                f"#!{sys.executable}\n"
                "import pathlib,sys\n"
                "root=pathlib.Path(sys.argv[1]); rows=sorted(root.iterdir(),key=lambda p:p.name)\n"
                "if '-printf' in sys.argv:\n"
                "  [print(p.name) for p in rows]\n"
                "elif '-print' in sys.argv and '-quit' in sys.argv and rows:\n"
                "  print(rows[0])\n",
            )
            fake_squeue = shim_root / "squeue"
            executable(
                fake_squeue,
                f"#!{sys.executable}\n"
                "import sys\n"
                "if '--steps' not in sys.argv:\n"
                " print('RUNNING|auh7-1b-gpu-[246-248,279]|gres/gpu:mi210:8')\n",
            )
            child_counter = sandbox / "child-count"
            fake_srun = sandbox / "srun"
            executable(
                fake_srun,
                f"#!{sys.executable}\n"
                "import pathlib,sys\n"
                f"p=pathlib.Path({str(child_counter)!r})\n"
                "p.write_text(p.read_text()+'1\\n' if p.exists() else '1\\n')\n"
                "print('hostile child failed exactly once',file=sys.stderr)\n"
                "raise SystemExit(37)\n",
            )

            # Create an exact-five launch closure.  The three payloads are
            # inert because the fake Slurm child exits before executing argv.
            fixture_files = {
                BOOTSTRAP_PATH.name: b"fixture bootstrap v2\n",
                STEP_PATH.name: b"#!/usr/bin/env bash\nexit 99\n",
                RANK_PATH.name: b"#!/usr/bin/env bash\nexit 99\n",
                "LAUNCH_AUTHORITY_CORE.json": b"{\"fixture\":true}\n",
            }
            for name, payload in fixture_files.items():
                path = launch_root / name
                path.write_bytes(payload)
                path.chmod(0o555 if name.endswith(".sh") else 0o444)
            release_manifest = release_root / "RELEASE_MANIFEST.json"
            release_manifest.write_bytes(b"{\"fixture\":true}\n")
            release_manifest.chmod(0o444)

            transformed = self.controller
            replacements = {
                r"^readonly python_bin=.*$": f"readonly python_bin={sys.executable}",
                r"^readonly experiment_root=.*$": f"readonly experiment_root={experiment}",
                r"^readonly step_payload_sha=[0-9a-f]{64}$": (
                    "readonly step_payload_sha=" + sha256(launch_root / STEP_PATH.name)
                ),
                r"^readonly rank_exec_sha=[0-9a-f]{64}$": (
                    "readonly rank_exec_sha=" + sha256(launch_root / RANK_PATH.name)
                ),
                r"^readonly bootstrap_sha=[0-9a-f]{64}$": (
                    "readonly bootstrap_sha=" + sha256(launch_root / BOOTSTRAP_PATH.name)
                ),
                r"^readonly launch_authority_core_sha=[0-9a-f]{64}$": (
                    "readonly launch_authority_core_sha="
                    + sha256(launch_root / "LAUNCH_AUTHORITY_CORE.json")
                ),
                r"^readonly release_manifest_sha=[0-9a-f]{64}$": (
                    "readonly release_manifest_sha=" + sha256(release_manifest)
                ),
            }
            for pattern, replacement in replacements.items():
                transformed, count = re.subn(pattern, replacement, transformed, flags=re.M)
                self.assertEqual(count, 1, pattern)
            transformed = transformed.replace("/usr/bin/squeue", str(fake_squeue))
            transformed = transformed.replace("/usr/bin/srun", str(fake_srun))
            transformed = transformed.replace("sleep 2", "sleep 0")

            renderer_sha = self.shell_pin(transformed, "renderer_sha")
            intent = {
                "schema_version": "bernini-action-edit-level-b-p2-attempt-intent-v2",
                "method": "bernini-action-edit-level-b-p2-00435-bootstrap-0817-v2",
                "authority": "PRE_D0_ENGINEERING_ONLY",
                "tag": tag,
                "parent_job_id": 140846,
                "node": "auh7-1b-gpu-279",
                "job_name": "bernini0817-level-b-p2-00435-v2",
                "release_root": str(release_root),
                "launch_root": str(launch_root),
                "attempt_root": str(attempt_root),
                "run_root": str(run_root),
                "output_mp4": str(
                    run_root / "00435ad621c44fac_p2_seed2026080821_v2.mp4"
                ),
                "source_video_sha256": bootstrap.SOURCE_VIDEO_SHA256,
                "instruction_utf8_sha256": bootstrap.EDIT_INSTRUCTION_SHA256,
                "inference_seed": bootstrap.INFERENCE_SEED,
                "checkpoint_step": 2,
                "checkpoint_parameter_sha256": bootstrap.P2_PARAMETER_SHA256,
                "release_manifest_sha256": sha256(release_manifest),
                "renderer_sha256": renderer_sha,
                "bootstrap_sha256": sha256(launch_root / BOOTSTRAP_PATH.name),
                "step_payload_sha256": sha256(launch_root / STEP_PATH.name),
                "rank_exec_sha256": sha256(launch_root / RANK_PATH.name),
                "world_size": 8,
                "dp_size": 2,
                "sp_size": 4,
                "host_memory_gib": 64,
                "max_restarts": 0,
                "committed_marker_required": True,
                "automatic_relaunch_authorized": False,
                "parent_control_authorized": False,
                "formal_training_started": False,
                "counts_as_d0": False,
                "promotion_authorized": False,
            }
            intent_sha = hashlib.sha256(canonical_json_bytes(intent)).hexdigest()
            transformed, count = re.subn(
                r"^readonly attempt_intent_sha=[0-9a-f]{64}$",
                "readonly attempt_intent_sha=" + intent_sha,
                transformed,
                flags=re.M,
            )
            self.assertEqual(count, 1)

            controller = launch_root / CONTROLLER_PATH.name
            controller.write_text(transformed)
            controller.chmod(0o555)
            launch_root.chmod(0o555)
            release_root.chmod(0o555)

            env = dict(os.environ)
            env["PATH"] = str(shim_root) + os.pathsep + env.get("PATH", "")
            completed = subprocess.run(
                ["bash", str(controller)],
                cwd=str(sandbox),
                env=env,
                capture_output=True,
                text=True,
                timeout=20,
            )
            self.assertEqual(completed.returncode, 37, completed.stderr)
            self.assertNotIn("readonly variable", completed.stderr)
            self.assertIn("child failed rc=37; no retry is authorized", completed.stderr)
            self.assertEqual(child_counter.read_text(), "1\n")

            status_path = attempt_root / "controller.status.json"
            self.assertTrue(status_path.is_file())
            self.assertEqual(stat.S_IMODE(status_path.stat().st_mode), 0o444)
            status = json.loads(status_path.read_bytes())
            self.assertEqual(status["schema_version"], "bernini-action-edit-level-b-p2-controller-status-v2")
            self.assertEqual(status["tag"], tag)
            self.assertEqual(status["child_exit_code"], 37)
            self.assertEqual(status["parent_state_before"], status["parent_state_after"])
            self.assertEqual(status["intent_path"], str(attempt_root / "STARTED" / "intent.json"))
            self.assertEqual(status["run_log_path"], str(attempt_root / "run.log"))
            self.assertFalse((attempt_root / "SUCCESS").exists())
            self.assertFalse((attempt_root / "terminal.authority.json").exists())
            self.assertEqual(list(run_root.iterdir()), [])

    def test_parent_job_is_checked_before_after_and_terminal_without_control(self):
        self.assertGreaterEqual(self.controller.count("expected_parent_state"), 4)
        self.assertIn("parent_untouched=true", self.controller)
        self.assertIn('"parent_control_authorized":False', self.controller)
        self.assertEqual(self.core["parent_allocation"]["job_id"], 140846)
        self.assertIs(self.core["parent_allocation"]["control_authorized"], False)

    def test_frozen_claim_boundaries_remain_pre_d0_only(self):
        self.assertEqual(
            self.core["schema_version"],
            "bernini-action-edit-level-b-p2-launch-authority-core-v2",
        )
        self.assertEqual(
            self.pins["schema_version"],
            "bernini-action-edit-level-b-p2-deployment-pins-v2",
        )
        self.assertEqual(self.core["tag"], "fresh-world8-level-b-p2-00435-v2")
        self.assertEqual(self.pins["tag"], "fresh-world8-level-b-p2-00435-v2")
        self.assertEqual(
            self.core["status"],
            "FROZEN_ONE_SHOT_LAUNCH_AUTHORITY",
        )
        for payload in (self.core["claims"], self.pins["claims"]):
            self.assertIs(payload["formal_training_started"], False)
            self.assertIs(payload["counts_as_d0"], False)
            self.assertIs(payload["scientific_claim_authorized"], False)
            self.assertIs(payload["promotion_authorized"], False)
            self.assertIs(payload["automatic_relaunch_authorized"], False)

    def test_outer_pins_bind_controller_and_every_launch_chain_member(self):
        self.assertEqual(self.pins["status"], "LOCAL_FROZEN_NOT_DEPLOYED")
        chain = self.pins["launch_chain"]
        expected = {
            "bootstrap": BOOTSTRAP_PATH,
            "rank_exec": RANK_PATH,
            "step_payload": STEP_PATH,
            "launch_authority_core": CORE_PATH,
            "controller": CONTROLLER_PATH,
        }
        for name, path in expected.items():
            self.assertEqual(chain[name]["sha256"], sha256(path))
            self.assertEqual(chain[name]["size"], path.stat().st_size)
        self.assertEqual(chain["intent_sha256"], self.core["launchers"]["intent_sha256"])
        self.assertIs(self.pins["remote_writes_performed_by_builder"], False)

    def test_v2_supersedes_failed_v1_without_reuse_overwrite_or_retry(self):
        value = self.pins["supersession"]
        failed = value["superseded_attempt"]
        self.assertEqual(failed["tag"], "fresh-world8-level-b-p2-00435-v1")
        self.assertEqual(failed["child_step_id"], "140846.370")
        self.assertEqual(failed["state"], "FAILED")
        self.assertEqual(failed["exit_code"], "1:0")
        self.assertEqual(failed["elapsed_seconds"], 297)
        self.assertEqual(failed["max_rss"], "67002688K")
        self.assertEqual(
            failed["failure_reason"],
            "live Wan VAE load_config callable is not owned by its pinned source bytes",
        )
        self.assertIs(failed["failed_before_denoise"], True)
        self.assertIs(failed["failed_before_output"], True)
        self.assertEqual(failed["run_root_exact_member_count"], 0)
        self.assertIs(failed["mp4_present"], False)
        self.assertIs(failed["receipt_present"], False)
        self.assertIs(failed["committed_marker_present"], False)
        self.assertEqual(
            failed["run_log_sha256"],
            "4f9817b372abd957d512b642284352d27519b191be26b5d1b2151b135c9a9075",
        )
        self.assertEqual(
            failed["intent_sha256"],
            "e3757167323e2f228bc9f404344cb40d419c6077aa6b04b3be037cfd3ce98f80",
        )
        self.assertEqual(
            failed["screen_log_sha256"],
            "2b72505835c02f43355dfabbaaa8834b372db8c2462fc88ad8d8da54e7013bdf",
        )
        self.assertIs(failed["controller_status_present"], False)
        self.assertIs(failed["parent_untouched"], True)
        self.assertIs(failed["gpus_released"], True)
        replacement = value["replacement"]
        self.assertEqual(replacement["tag"], "fresh-world8-level-b-p2-00435-v2")
        self.assertIs(replacement["fresh_roots_required"], True)
        self.assertIs(replacement["reuse_v1_attempt_or_run"], False)
        self.assertIs(replacement["automatic_retry_of_v1"], False)
        self.assertEqual(replacement["controller_sha256"], sha256(CONTROLLER_PATH))
        self.assertIn("disjoint", replacement["controller_status_fix"])
        self.assertIn("callable ownership", replacement["renderer_fix"])
        for root in self.core["roots"].values():
            self.assertIn("fresh-world8-level-b-p2-00435-v2", root)
            self.assertNotIn("fresh-world8-level-b-p2-00435-v1", root)
        self.assertEqual(self.core["supersession"]["superseded_attempt"], failed)
        core_replacement = self.core["supersession"]["replacement"]
        self.assertNotIn("controller_sha256", core_replacement)
        self.assertEqual(
            core_replacement,
            {key: item for key, item in replacement.items() if key != "controller_sha256"},
        )

    def test_outer_pins_bind_exact_level_a_tools_base_and_vendor_sources(self):
        expected_level_a = {
            path: {"sha256": sha, "size": size, "mode": 0o444}
            for path, (sha, size) in bootstrap.LEVEL_A_MEMBER_PINS.items()
        }
        self.assertEqual(self.pins["frozen_level_a"]["members"], expected_level_a)
        renderer_tools = {
            "python": {
                "path": literal_assignment(RENDERER_PATH, "PINNED_PYTHON_PATH"),
                "sha256": literal_assignment(RENDERER_PATH, "PINNED_PYTHON_SHA256"),
            },
            "ffmpeg": {
                "path": literal_assignment(RENDERER_PATH, "PINNED_FFMPEG_PATH"),
                "sha256": literal_assignment(RENDERER_PATH, "PINNED_FFMPEG_SHA256"),
            },
            "ffprobe": {
                "path": literal_assignment(RENDERER_PATH, "PINNED_FFPROBE_PATH"),
                "sha256": literal_assignment(RENDERER_PATH, "PINNED_FFPROBE_SHA256"),
            },
        }
        self.assertEqual(self.pins["runtime_authority"]["tools"], renderer_tools)
        self.assertEqual(
            self.pins["runtime_authority"]["bernini_source_sha256"],
            literal_assignment(RENDERER_PATH, "PINNED_BERNINI_RUNTIME_FILE_HASHES"),
        )
        self.assertEqual(
            self.pins["runtime_authority"]["site_package_source_sha256"],
            literal_assignment(RENDERER_PATH, "PINNED_SITE_PACKAGE_SOURCE_HASHES"),
        )
        self.assertEqual(
            self.pins["base_checkpoint"]["tree_sha256"],
            bootstrap.BASE_CHECKPOINT_TREE_SHA256,
        )

    def test_outer_pins_record_both_nonrelease_tests(self):
        tests = self.pins["tests"]
        self.assertEqual(tests["renderer_compatibility_qa"]["sha256"], RENDERER_TEST_SHA)
        self.assertEqual(tests["renderer_compatibility_qa"]["size"], RENDERER_TEST_SIZE)
        self.assertIs(tests["renderer_compatibility_qa"]["is_release_member"], False)
        self.assertEqual(tests["launchbound"]["sha256"], sha256(THIS_TEST_PATH))
        self.assertEqual(tests["launchbound"]["size"], THIS_TEST_PATH.stat().st_size)
        self.assertIs(tests["launchbound"]["is_release_member"], False)

    def test_shell_and_embedded_python_syntax(self):
        for path in (CONTROLLER_PATH, STEP_PATH, RANK_PATH):
            completed = subprocess.run(
                ["bash", "-n", str(path)], capture_output=True, text=True
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
        for path in (CONTROLLER_PATH, STEP_PATH):
            snippets = re.findall(r'-[IB ]*-c \'\n(.*?)\n\'', path.read_text(), re.S)
            self.assertGreaterEqual(len(snippets), 1)
            for snippet in snippets:
                compile(snippet, f"{path}:embedded", "exec")

    def test_no_launcher_has_unbounded_retry_loop(self):
        for text in (self.controller, self.step, self.rank):
            self.assertNotIn("while true", text.lower())
            self.assertNotIn("until ", text.lower())
        self.assertEqual(self.controller.count("for poll in"), 1)

    def test_bootstrap_accepts_no_caller_semantic_arguments(self):
        source = self.bootstrap_text[self.bootstrap_text.index("def main") :]
        self.assertIn('values == ["run"]', source)
        self.assertIn('values == ["validate-product"]', source)
        self.assertNotIn("argparse", source)

    def test_output_name_and_unique_tag_are_fixed(self):
        self.assertEqual(
            bootstrap.OUTPUT_MP4.name,
            "00435ad621c44fac_p2_seed2026080821_v2.mp4",
        )
        self.assertEqual(self.pins["output"]["path"], str(bootstrap.OUTPUT_MP4))
        self.assertEqual(
            self.pins["output"]["receipt_path"],
            str(bootstrap.OUTPUT_MP4) + ".receipt.json",
        )
        self.assertEqual(
            self.pins["output"]["commit_marker_path"],
            str(bootstrap.OUTPUT_MP4) + ".COMMITTED.json",
        )
        for text in (self.bootstrap_text, self.controller, self.step, self.rank):
            self.assertIn("fresh-world8-level-b-p2-00435-v2", text)
        self.assertNotEqual(bootstrap.TAG, "fresh-world8-level-a-r2-p2-launchbound-v2")


if __name__ == "__main__":
    unittest.main()
