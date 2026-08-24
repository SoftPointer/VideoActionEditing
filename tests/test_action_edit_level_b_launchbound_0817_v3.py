from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
METHOD = ROOT / "methods" / "bernini_action_editing"
BOOTSTRAP_PATH = METHOD / "action_edit_level_b_p2_00435_bootstrap_0817_v3.py"
CONTROLLER_PATH = (
    METHOD / "scripts" / "auh_launch_action_edit_level_b_p2_00435_job140846_v3.sh"
)
STEP_PATH = METHOD / "scripts" / "auh_action_edit_level_b_p2_00435_step_v3.sh"
RANK_PATH = METHOD / "scripts" / "auh_action_edit_level_b_p2_00435_rank_exec_v3.sh"
RELEASE_PATH = (
    METHOD / "audits" / "fresh_world8_level_b_p2_00435_v3_RELEASE_MANIFEST.json"
)
CORE_PATH = (
    METHOD
    / "audits"
    / "fresh_world8_level_b_p2_00435_v3_LAUNCH_AUTHORITY_CORE.json"
)
PINS_PATH = (
    METHOD / "audits" / "fresh_world8_level_b_p2_00435_v3_DEPLOYMENT_PINS.json"
)
V2_CORE_PATH = (
    METHOD
    / "audits"
    / "fresh_world8_level_b_p2_00435_v2_LAUNCH_AUTHORITY_CORE.json"
)
V2_PINS_PATH = (
    METHOD / "audits" / "fresh_world8_level_b_p2_00435_v2_DEPLOYMENT_PINS.json"
)
RENDERER_PATH = METHOD / "infer_action_edit_level_b_renderer_0817_v1.py"
RENDERER_TEST_PATH = ROOT / "tests" / "test_infer_action_edit_level_b_renderer_0817_v1.py"
CALIBRATION_PATH = (
    METHOD
    / "audits"
    / "calibrate_fresh_world8_level_b_p2_00435_v3_static_preflight.py"
)
CALIBRATION_EVIDENCE_PATH = (
    METHOD
    / "audits"
    / "fresh_world8_level_b_p2_00435_v3_STATIC_PREFLIGHT_CALIBRATION_EVIDENCE.json"
)
THIS_TEST_PATH = Path(__file__).resolve()

RENDERER_SHA = "8e34d976481ed81e3b8b285253878f0c02bbfbe177ea608aa51b0f4b594bf1c6"
RENDERER_SIZE = 404000
RENDERER_TEST_SHA = "543a740b45c7a491b6c4e7d39406a82cb89b6cb9bafed6e9f9931787780d43e9"
RENDERER_TEST_SIZE = 155048


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
    name = "action_edit_level_b_p2_00435_bootstrap_0817_v3_test"
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


class LevelBLaunchboundV3FrozenTests(unittest.TestCase):
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
        for authority in (self.core, self.pins):
            for claim in (
                "validation_exit_codes_required_zero",
                "validation_fail_empty_cannot_publish",
                "validation_stderr_in_hashed_boundary",
                "validation_strict_canonical_json_and_exact_claims",
            ):
                self.assertIs(authority["postcommit"][claim], True)

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

    def test_static_stdlib_socket_source_requires_exact_thirteen_links(self):
        self.assertEqual(bootstrap.STATIC_PREFLIGHT_STDLIB_SOCKET_NLINK, 13)
        source_block = self.bootstrap_text[
            self.bootstrap_text.index("def stable_source(") :
            self.bootstrap_text.index("urllib3_source = stable_source(")
        ]
        self.assertIn("expected_nlink: int = 1", source_block)
        self.assertIn("expected_nlink=STATIC_PREFLIGHT_STDLIB_SOCKET_NLINK", source_block)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "socket.py"
            source.write_bytes(b"fixture")
            for index in range(12):
                os.link(source, root / f"socket-link-{index}")
            self.assertEqual(source.stat().st_nlink, 13)
            self.assertEqual(
                bootstrap._plain_file(source.resolve(), nlink=13),
                source.resolve(),
            )
            (root / "socket-link-0").unlink()
            with self.assertRaises(bootstrap.BootstrapError):
                bootstrap._plain_file(source.resolve(), nlink=13)

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

    def test_cpu_static_preflight_precedes_every_attempt_write_and_srun(self):
        parent_gate = self.controller.index('\nparent_before=\n')
        preflight = self.controller.index('"${bootstrap}" static-preflight')
        started = self.controller.index('readonly started="${attempt_root}/STARTED"')
        intent_write = self.controller.index('>"${intent_tmp}"', started)
        child = self.controller.index('/usr/bin/srun', intent_write)
        self.assertLess(parent_gate, preflight)
        self.assertLess(preflight, started)
        self.assertLess(started, intent_write)
        self.assertLess(intent_write, child)
        self.assertEqual(self.controller.count('"${bootstrap}" static-preflight'), 1)
        block = self.controller[parent_gate:started]
        for token in (
            "/usr/bin/env -i",
            "CUDA_VISIBLE_DEVICES=''",
            "ROCR_VISIBLE_DEVICES=''",
            "HIP_VISIBLE_DEVICES=''",
            "PYTHONDONTWRITEBYTECODE=1",
            "PYTHONNOUSERSITE=1",
            "HF_HUB_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1",
            "OPENBLAS_MAIN_FREE=1",
            "GOTOBLAS_MAIN_FREE=1",
            "VEOMNI_VERBOSITY=ERROR",
            '"${python_bin}" -I -B "${bootstrap}" static-preflight',
            '"${python_bin}" -I -B "${bootstrap}" static-preflight 2>&1',
            '"${base64_bin}" -w0',
            'static_preflight_pipeline_status=("${PIPESTATUS[@]}")',
            "__LEVEL_B_P2_00435_V3_STATIC_PREFLIGHT_PIPESTATUS_",
            '"${static_preflight_child_status}" == 000',
            '"${static_preflight_encoder_status}" == 000',
            '"${#static_preflight_base64}" == "${static_preflight_base64_size}"',
            '"${static_preflight_observed_base64_sha}" == "${static_preflight_base64_sha}"',
            "parent changed during CPU static preflight",
            "attempt root changed during CPU static preflight",
            "run root changed during CPU static preflight",
        ):
            self.assertIn(token, block)
        self.assertEqual(block.count("2>&1"), 1)
        self.assertNotIn(">", block.replace("2>&1", ""))
        self.assertIn(
            "readonly base64_sha=b10f8c059f50c0681c6497e7b09ebdba168e341498ae1733de9089dc8efa0898",
            self.controller,
        )
        self.assertNotIn("mkdir ", block)
        self.assertNotIn("/usr/bin/srun", block)

    def test_bootstrap_static_preflight_is_capability_only_and_fail_closed(self):
        source = self.bootstrap_text[
            self.bootstrap_text.index("def _install_static_preflight_read_only_guard") :
            self.bootstrap_text.index("def run_world8")
        ]
        for token in (
            "sys.addaudithook(audit)",
            'event == "open"',
            '"subprocess.Popen"',
            '"socket.connect"',
            '"os.putenv"',
            '"os.unsetenv"',
            "_authenticate_level_b_module()",
            "authenticate_level_b_runtime_release(",
            "LEVEL_B_MANIFEST",
            "run_level_b_cpu_static_runtime_preflight(",
            '"cuda_initialized_before"',
            '"cuda_initialized_after"',
            '"weights_loaded"',
            '"model_constructors_called"',
            '"product_output_writes"',
            '"persistent_filesystem_writes"',
            '"subprocesses_spawned"',
            '"network_accessed"',
            '"blas_import_environment_preseeded_before_vendor_imports"',
            '"blas_import_environment_mutations_allowed"',
            '"veomni_logging_environment_preseeded_before_vendor_imports"',
            '"pinned_bernini_and_veomni_roots_scoped_and_restored"',
            '"preexisting_bernini_or_veomni_modules_accepted"',
            "STATIC_PREFLIGHT_STDOUT_SHA256",
            "STATIC_PREFLIGHT_STDOUT_SIZE",
            'allowed_key == ("/dev/null", "r+", devnull_rplus_flags)',
            'allowed_key == ("/dev/null", "w", devnull_write_flags)',
            'allowed_key == ("/dev/null", None, devnull_rplus_flags)',
            'allowed_key == ("/dev/null", None, devnull_write_flags)',
            'event == "socket.__new__" and is_expected_blocked_socket_probe(args)',
            'arguments[1:] != (10, 1, 0)',
            'socket_state == (-1, 0, 0, 0)',
            'socket_frame.f_lineno == 233',
            'urllib3_frame.f_lineno == 126',
            'module_frame.f_lineno == 137',
            'process_guard.get("blocked_network_probe_count") != 1',
            'process_guard.get("socket_objects_created") is not False',
            '"blocked_network_capability_probe": blocked_socket_receipt',
            "_call_static_preflight_owner_without_stdout(",
            '"vendor_stdout_bytes_observed": 0',
            '"OPENBLAS_MAIN_FREE": "1"',
            '"GOTOBLAS_MAIN_FREE": "1"',
            'renderer_receipt.get("torch_jit_temporary_directory_suppression")',
            'renderer_receipt.get("torch_remote_module_template_suppression")',
            'renderer_receipt.get("numpy_blas_import_environment_seal")',
            'renderer_receipt.get("veomni_stdout_logging_suppression")',
            'renderer_receipt.get("scoped_module_source_closure")',
            'renderer_receipt.get("six_meta_path_importer_scope")',
            '"no_process_specific_repr_or_object_address_recorded": True',
            '"repr_or_object_address_in_receipt": False',
            '"exact_meta_path_finder_calls": 1',
            '"constructor_call_line": 21',
            '"atexit_registration_line": 23',
            '"module_call_line": 30',
            '"factory_calls_suppressed": 1',
            '"persistent_filesystem_writes": False',
        ):
            self.assertIn(token, source)
        for forbidden in (
            "from_pretrained(",
            "torch.cuda.is_available",
            "torch.cuda.device_count",
            "consume_frozen_r2_world8_checkpoint",
            "run_level_b_pre_d0_offline_inference",
            "validate_committed_level_b_product",
        ):
            self.assertNotIn(forbidden, source)

    def test_static_preflight_stdout_guard_accepts_silence_and_rejects_text_or_bytes(self):
        original = sys.stdout
        marker = object()
        self.assertIs(
            bootstrap._call_static_preflight_owner_without_stdout(lambda: marker),
            marker,
        )
        self.assertIs(sys.stdout, original)

        def text_writer():
            print("volatile INFO 2099-01-01 00:00:00")
            return marker

        with self.assertRaises(bootstrap.BootstrapError):
            bootstrap._call_static_preflight_owner_without_stdout(text_writer)
        self.assertIs(sys.stdout, original)

        def bytes_writer():
            sys.stdout.buffer.write(b"volatile vendor stdout\n")
            return marker

        with self.assertRaises(bootstrap.BootstrapError):
            bootstrap._call_static_preflight_owner_without_stdout(bytes_writer)
        self.assertIs(sys.stdout, original)

    def test_calibration_harness_proves_self_pin_dag_and_rejects_tamper(self):
        name = "level_b_v3_static_preflight_calibration_test"
        spec = importlib.util.spec_from_file_location(name, CALIBRATION_PATH)
        calibration = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(calibration)
        source = BOOTSTRAP_PATH.read_text()
        calibration.verify_static_dependency_dag(ast.parse(source))
        final_bytes = BOOTSTRAP_PATH.read_bytes()
        final_sha_assignment = (
            b'STATIC_PREFLIGHT_STDOUT_SHA256 = "'
            + bootstrap.STATIC_PREFLIGHT_STDOUT_SHA256.encode("ascii")
            + b'"\n'
        )
        final_size_assignment = (
            b"STATIC_PREFLIGHT_STDOUT_SIZE = "
            + str(bootstrap.STATIC_PREFLIGHT_STDOUT_SIZE).encode("ascii")
            + b"\n"
        )
        self.assertEqual(final_bytes.count(final_sha_assignment), 1)
        self.assertEqual(final_bytes.count(final_size_assignment), 1)
        pending = final_bytes.replace(
            final_sha_assignment, calibration.SHA_ASSIGNMENT
        ).replace(final_size_assignment, calibration.SIZE_ASSIGNMENT)
        self.assertEqual(
            hashlib.sha256(pending).hexdigest(),
            calibration.PENDING_BOOTSTRAP_SHA256,
        )
        self.assertEqual(len(pending), calibration.PENDING_BOOTSTRAP_SIZE)
        self.assertEqual(pending.count(calibration.SHA_ASSIGNMENT), 1)
        self.assertEqual(pending.count(calibration.SIZE_ASSIGNMENT), 1)
        transformed = pending.replace(
            calibration.SHA_ASSIGNMENT, calibration.SHA_CALIBRATION_ASSIGNMENT
        ).replace(
            calibration.SIZE_ASSIGNMENT, calibration.SIZE_CALIBRATION_ASSIGNMENT
        )
        self.assertEqual(len(transformed), calibration.CALIBRATION_SOURCE_SIZE)
        self.assertEqual(
            hashlib.sha256(transformed).hexdigest(),
            calibration.CALIBRATION_SOURCE_SHA256,
        )
        calibration.load_calibration_candidate(BOOTSTRAP_PATH.resolve(), pending)

        hostile_sources = (
            source.replace(
                "    unsigned = {\n",
                '    unsigned = {"hostile": STATIC_PREFLIGHT_STDOUT_SHA256,\n',
                1,
            ),
            source.replace(
                "    raw = canonical_json_bytes(result)\n",
                '    raw = b"hostile"\n',
                1,
            ),
            source.replace(
                '        fail("CPU static preflight canonical stdout differs from frozen authority")',
                '        fail("hostile earlier boundary")',
                1,
            ),
        )
        for hostile in hostile_sources:
            with self.assertRaises(calibration.CalibrationError):
                calibration.verify_static_dependency_dag(ast.parse(hostile))

    def test_controller_memory_boundary_rejects_valid_stdout_plus_stderr(self):
        block = self.controller[
            self.controller.index("static_preflight_frame=") :
            self.controller.index("readonly started=", self.controller.index("static_preflight_frame="))
        ]
        self.assertEqual(block.count("static-preflight 2>&1"), 1)
        self.assertIn('"${base64_bin}" -w0', block)
        self.assertIn('"${PIPESTATUS[@]}"', block)
        valid = canonical_json_bytes({
            "pass_token": bootstrap.STATIC_PREFLIGHT_PASS_TOKEN,
            "fixture": True,
        })
        encoded = __import__("base64").b64encode(valid)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            encoder = root / "encoder.py"
            encoder.write_text(
                "import base64,sys\n"
                "sys.stdout.buffer.write(base64.b64encode(sys.stdin.buffer.read()))\n"
            )

            def framed_status(stderr_payload: bytes, exit_code: int) -> int:
                producer = root / "producer.py"
                producer.write_text(
                    "import sys\n"
                    + "sys.stdout.buffer.write(" + repr(valid) + ")\n"
                    + "sys.stderr.buffer.write(" + repr(stderr_payload) + ")\n"
                    + f"raise SystemExit({exit_code})\n"
                )
                script = f'''set -Eeuo pipefail
sentinel=__LEVEL_B_P2_00435_V3_STATIC_PREFLIGHT_PIPESTATUS_
frame="$({sys.executable!s} -I -B {producer!s} 2>&1 | {sys.executable!s} -I -B {encoder!s}; status=("${{PIPESTATUS[@]}}"); printf '%s%03d_%03d__' "${{sentinel}}" "${{status[0]}}" "${{status[1]}}"; exit 0)"
[[ "${{frame}}" =~ ${{sentinel}}([0-9]{{3}})_([0-9]{{3}})__$ ]]
suffix="${{BASH_REMATCH[0]}}"; child="${{BASH_REMATCH[1]}}"; encoder_rc="${{BASH_REMATCH[2]}}"
payload="${{frame%"${{suffix}}"}}"
[[ "${{child}}" == 000 && "${{encoder_rc}}" == 000 ]]
[[ "${{#payload}}" == {len(encoded)} ]]
[[ "$(printf '%s' "${{payload}}" | shasum -a 256 | awk '{{print $1}}')" == {hashlib.sha256(encoded).hexdigest()} ]]
'''
                return subprocess.run(
                    ["bash", "-c", script], capture_output=True, text=True
                ).returncode

            self.assertEqual(framed_status(b"", 0), 0)
            for stderr_payload, exit_code in (
                (b"\n", 0),
                (b"\n\n", 0),
                (b"hostile timestamped stderr\n", 0),
                (b"\x00", 0),
                (b"", 17),
            ):
                with self.subTest(stderr=stderr_payload, exit_code=exit_code):
                    self.assertNotEqual(framed_status(stderr_payload, exit_code), 0)

    def test_terminal_parent_and_child_gate_precedes_terminal_and_success(self):
        final_child = self.controller.index(
            'require_no_node_children "numeric child appeared before terminal seal"'
        )
        final_parent = self.controller.index('terminal_parent_state=', final_child)
        terminal = self.controller.index('readonly terminal="${attempt_root}/terminal.authority.json"', final_parent)
        success = self.controller.index('readonly success_tmp=', terminal)
        self.assertLess(final_child, final_parent)
        self.assertLess(final_parent, terminal)
        self.assertLess(terminal, success)
        self.assertIn('"parent_state_at_terminal":sys.argv[9]', self.controller)
        self.assertIn('"parent_untouched":sys.argv[9]==', self.controller)

    def test_hostile_late_parent_drift_cannot_publish_terminal_or_success(self):
        require_start = self.controller.index("require_no_node_children() {")
        require_end = self.controller.index("\n}\n\nawait_child_teardown", require_start) + len("\n}")
        require_function = self.controller[require_start:require_end]
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            terminal = root / "terminal.authority.json"
            success = root / "SUCCESS"
            script = f'''set -Eeuo pipefail
expected_parent_state='RUNNING|auh7-1b-gpu-[246-248,279]|gres/gpu:mi210:8'
node_children() {{ return 0; }}
fail() {{ exit 95; }}
{require_function}
require_no_node_children hostile-terminal-closure
terminal_parent_state='COMPLETING|auh7-1b-gpu-[246-248,279]|gres/gpu:mi210:8'
[[ "${{terminal_parent_state}}" == "${{expected_parent_state}}" ]] || fail
printf terminal >{terminal!s}
printf success >{success!s}
'''
            completed = subprocess.run(["bash", "-c", script])
            self.assertEqual(completed.returncode, 95)
            self.assertFalse(terminal.exists())
            self.assertFalse(success.exists())

    def test_hostile_validation_fail_empty_nonjson_or_stderr_cannot_publish(self):
        start = self.controller.index("validated_product_probe() {")
        end = self.controller.index("\n}\n\nfor pending_sha", start) + len("\n}")
        function = self.controller[start:end]
        canonical = canonical_json_bytes({
            "schema_version": "bernini-action-edit-level-b-p2-product-validation-v3",
            "method": bootstrap.METHOD,
            "authority": bootstrap.AUTHORITY,
            "output_mp4": "/tmp/hostile-v3.mp4",
            "validation": {},
            "receipt_claims_revalidated": True,
            "receipt_inode_alias_marker_revalidated": True,
            "committed_marker_required": True,
            "formal_training_started": False,
            "counts_as_d0": False,
            "promotion_authorized": False,
        })
        producers = (
            "raise SystemExit(94)\n",
            "import sys\nsys.stdout.write('not-json')\n",
            "import sys\nsys.stdout.buffer.write(" + repr(canonical) + ")\nsys.stderr.write('\\n')\n",
        )
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            encoder = root / "base64"
            encoder.write_text(
                "#!/usr/bin/env python3\n"
                "import base64,sys\n"
                "assert sys.argv[1:] == ['-w0']\n"
                "sys.stdout.write(base64.b64encode(sys.stdin.buffer.read()).decode('ascii'))\n"
            )
            encoder.chmod(0o755)
            for index, producer_source in enumerate(producers):
                producer = root / f"producer-{index}.py"
                producer.write_text(producer_source)
                terminal = root / f"terminal-{index}"
                success = root / f"success-{index}"
                script = f'''set -Eeuo pipefail
python_bin={sys.executable!s}
base64_bin={encoder!s}
bootstrap={producer!s}
output_mp4=/tmp/hostile-v3.mp4
terminal={terminal!s}
success={success!s}
fail() {{ printf 'refused: %s\\n' "$*" >&2; exit 95; }}
{function}
probe_a=
if ! probe_a="$(validated_product_probe first)"; then fail "first committed-product validation failed"; fi
readonly probe_a
probe_b=
if ! probe_b="$(validated_product_probe second)"; then fail "second committed-product validation failed"; fi
readonly probe_b
[[ "${{probe_a}}" == "${{probe_b}}" ]] || fail differs
touch "${{terminal}}" "${{success}}"
'''
                completed = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
                self.assertEqual(completed.returncode, 95, (index, completed.stderr))
                self.assertFalse(terminal.exists())
                self.assertFalse(success.exists())

    def test_hostile_fail_empty_child_query_and_terminal_hash_cannot_publish(self):
        require_start = self.controller.index("require_no_node_children() {")
        require_end = self.controller.index("\n}\n\nawait_child_teardown", require_start) + len("\n}")
        require_function = self.controller[require_start:require_end]
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            terminal = root / "terminal"
            success = root / "SUCCESS"
            query_script = f'''set -Eeuo pipefail
terminal={terminal!s}
success={success!s}
fail() {{ exit 95; }}
node_children() {{ return 91; }}
{require_function}
require_no_node_children hostile-query
touch "${{terminal}}" "${{success}}"
'''
            completed = subprocess.run(["bash", "-c", query_script])
            self.assertEqual(completed.returncode, 95)
            self.assertFalse(terminal.exists())
            self.assertFalse(success.exists())

            terminal.write_text("sealed")
            shim = root / "sha256sum"
            shim.write_text("#!/usr/bin/env bash\nexit 93\n")
            shim.chmod(0o755)
            hash_script = f'''set -Eeuo pipefail
PATH={root!s}:/usr/bin:/bin
terminal={terminal!s}
success_tmp={root / '.SUCCESS.tmp'!s}
success={success!s}
fail() {{ exit 95; }}
terminal_sha=
if ! terminal_sha="$(sha256sum "${{terminal}}" | awk '{{print $1}}')"; then fail "terminal authority SHA query failed"; fi
readonly terminal_sha
[[ "${{terminal_sha}}" =~ ^[0-9a-f]{{64}}$ ]] || fail "terminal authority SHA differs"
printf '%s\\n' "${{terminal_sha}}" >"${{success_tmp}}"
mv "${{success_tmp}}" "${{success}}"
'''
            completed = subprocess.run(["bash", "-c", hash_script])
            self.assertEqual(completed.returncode, 95)
            self.assertFalse(success.exists())

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
            "schema_version": "bernini-action-edit-level-b-p2-attempt-intent-v3",
            "method": bootstrap.METHOD,
            "authority": bootstrap.AUTHORITY,
            "tag": bootstrap.TAG,
            "parent_job_id": 140846,
            "node": "auh7-1b-gpu-279",
            "job_name": "bernini0817-level-b-p2-00435-v3",
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
            "static_preflight_stdout_sha256": bootstrap.STATIC_PREFLIGHT_STDOUT_SHA256,
            "static_preflight_stdout_size": bootstrap.STATIC_PREFLIGHT_STDOUT_SIZE,
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
            bootstrap.STATIC_PREFLIGHT_STDOUT_SHA256,
            str(bootstrap.STATIC_PREFLIGHT_STDOUT_SIZE),
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
        self.assertEqual(
            self.shell_pin(self.controller, "static_preflight_sha"),
            bootstrap.STATIC_PREFLIGHT_STDOUT_SHA256,
        )
        match = re.search(r"^readonly static_preflight_size=([1-9][0-9]*)$", self.controller, re.M)
        self.assertIsNotNone(match)
        self.assertEqual(int(match.group(1)), bootstrap.STATIC_PREFLIGHT_STDOUT_SIZE)
        self.assertEqual(
            self.shell_pin(self.controller, "static_preflight_base64_sha"),
            self.core["static_preflight"]["combined_stdout_stderr_base64_sha256"],
        )
        base64_size = re.search(
            r"^readonly static_preflight_base64_size=([1-9][0-9]*)$",
            self.controller,
            re.M,
        )
        self.assertIsNotNone(base64_size)
        self.assertEqual(
            int(base64_size.group(1)),
            self.core["static_preflight"]["combined_stdout_stderr_base64_size"],
        )
        self.assertEqual(
            self.shell_pin(self.controller, "base64_sha"),
            self.core["static_preflight"]["base64_tool_sha256"],
        )

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
        self.assertIn('require_stat_value "${output_mp4}" %h 1', self.controller)
        self.assertIn('require_stat_value "${output_receipt}" %h 2', self.controller)
        self.assertIn('require_stat_value "${output_marker}" %h 2', self.controller)
        self.assertIn("COMMITTED marker is not the exact receipt inode alias", self.controller)
        self.assertIn("receipt_inode_alias_marker_verified", self.controller)
        self.assertEqual(self.controller.count('"${bootstrap}" validate-product'), 1)
        self.assertEqual(self.controller.count("validated_product_probe first"), 1)
        self.assertEqual(self.controller.count("validated_product_probe second"), 1)

    def test_success_cannot_precede_committed_double_validation(self):
        marker = self.controller.index("expected_run_entries")
        alias = self.controller.index("exact receipt inode alias", marker)
        validate = self.controller.index("validated_product_probe first", alias)
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
            tag = "fresh-world8-level-b-p2-00435-v3"
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
                "'%h':str(info.st_nlink),'%s':str(info.st_size),"
                "'%d:%i':f'{info.st_dev}:{info.st_ino}'}\n"
                "print(rows[fmt])\n",
            )
            executable(
                shim_root / "readlink",
                f"#!{sys.executable}\n"
                "import pathlib,sys\n"
                "assert sys.argv[1] == '-f' and len(sys.argv) == 3\n"
                "print(pathlib.Path(sys.argv[2]).resolve())\n",
            )
            executable(
                shim_root / "sha256sum",
                f"#!{sys.executable}\n"
                "import hashlib,pathlib,sys\n"
                "if len(sys.argv) == 1:\n"
                " data=sys.stdin.buffer.read(); print(hashlib.sha256(data).hexdigest(), '-')\n"
                "else:\n"
                " assert len(sys.argv) == 2\n"
                " p=pathlib.Path(sys.argv[1]); print(hashlib.sha256(p.read_bytes()).hexdigest(), p)\n",
            )
            executable(
                shim_root / "base64",
                f"#!{sys.executable}\n"
                "import base64,sys\n"
                "assert sys.argv[1:] == ['-w0']\n"
                "sys.stdout.buffer.write(base64.b64encode(sys.stdin.buffer.read()))\n",
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
            fixture_preflight = canonical_json_bytes({
                "pass_token": "LEVEL_B_P2_00435_V3_CPU_STATIC_PREFLIGHT_OK",
                "fixture": True,
            })
            fixture_preflight_base64 = __import__("base64").b64encode(
                fixture_preflight
            )
            fixture_files = {
                BOOTSTRAP_PATH.name: (
                    b"import sys\n"
                    b"assert sys.argv[1:] == ['static-preflight']\n"
                    b"sys.stdout.buffer.write(" + repr(fixture_preflight).encode() + b")\n"
                ),
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
                r"^readonly base64_bin=.*$": (
                    "readonly base64_bin=" + str(shim_root / "base64")
                ),
                r"^readonly base64_sha=[0-9a-f]{64}$": (
                    "readonly base64_sha=" + sha256(shim_root / "base64")
                ),
                r"^readonly base64_size=[0-9]+$": (
                    "readonly base64_size=" + str((shim_root / "base64").stat().st_size)
                ),
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
                r"^readonly static_preflight_sha=.*$": (
                    "readonly static_preflight_sha="
                    + hashlib.sha256(fixture_preflight).hexdigest()
                ),
                r"^readonly static_preflight_size=.*$": (
                    "readonly static_preflight_size=" + str(len(fixture_preflight))
                ),
                r"^readonly static_preflight_base64_sha=.*$": (
                    "readonly static_preflight_base64_sha="
                    + hashlib.sha256(fixture_preflight_base64).hexdigest()
                ),
                r"^readonly static_preflight_base64_size=.*$": (
                    "readonly static_preflight_base64_size="
                    + str(len(fixture_preflight_base64))
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
                "schema_version": "bernini-action-edit-level-b-p2-attempt-intent-v3",
                "method": "bernini-action-edit-level-b-p2-00435-bootstrap-0817-v3",
                "authority": "PRE_D0_ENGINEERING_ONLY",
                "tag": tag,
                "parent_job_id": 140846,
                "node": "auh7-1b-gpu-279",
                "job_name": "bernini0817-level-b-p2-00435-v3",
                "release_root": str(release_root),
                "launch_root": str(launch_root),
                "attempt_root": str(attempt_root),
                "run_root": str(run_root),
                "output_mp4": str(
                    run_root / "00435ad621c44fac_p2_seed2026080821_v3.mp4"
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
                "static_preflight_stdout_sha256": hashlib.sha256(
                    fixture_preflight
                ).hexdigest(),
                "static_preflight_stdout_size": len(fixture_preflight),
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
            self.assertEqual(status["schema_version"], "bernini-action-edit-level-b-p2-controller-status-v3")
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
            "bernini-action-edit-level-b-p2-launch-authority-core-v3",
        )
        self.assertEqual(
            self.pins["schema_version"],
            "bernini-action-edit-level-b-p2-deployment-pins-v3",
        )
        self.assertEqual(self.core["tag"], "fresh-world8-level-b-p2-00435-v3")
        self.assertEqual(self.pins["tag"], "fresh-world8-level-b-p2-00435-v3")
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
        self.assertEqual(
            self.pins["status"],
            "RELEASE_DEPLOYED_LAUNCH_NOT_DEPLOYED",
        )
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

    def test_copy_paste_deployment_inventory_is_exact_fresh_and_one_way(self):
        recipe = self.pins["deployment_recipe"]
        self.assertEqual(
            recipe["schema_version"],
            "bernini-action-edit-level-b-p2-v3-deployment-recipe-v1",
        )
        self.assertEqual(recipe["remote_host_alias"], "auh")
        self.assertEqual(recipe["local_repo_root"], str(ROOT))
        self.assertIs(recipe["remote_writes_performed"], True)
        self.assertIs(recipe["release_deployment_completed"], True)
        self.assertIs(recipe["release_independently_verified"], True)
        self.assertIs(recipe["remaining_deployment_authorized"], False)
        self.assertIs(recipe["launch_authorized"], False)
        self.assertIs(recipe["automatic_retry_authorized"], False)

        tag = bootstrap.TAG
        experiment = PurePosixPath(str(bootstrap.EXPERIMENT_ROOT))
        final_roots = {
            "release": str(experiment / "releases" / tag),
            "launch": str(experiment / "launchers" / tag),
            "attempt": str(experiment / "attempts" / tag),
            "run": str(experiment / "runs" / tag),
            "screen_log": str(experiment / "screen-logs" / tag),
        }
        self.assertEqual(recipe["final_roots"], final_roots)
        release_stage_leaf = f".{tag}.stage-release-{sha256(RELEASE_PATH)[:16]}"
        launch_stage_leaf = f".{tag}.stage-launch-{sha256(CONTROLLER_PATH)[:16]}"
        runtime_stage_leaf = f".{tag}.stage-runtime-{sha256(CONTROLLER_PATH)[:16]}"
        staging_roots = {
            "release": str(PurePosixPath(final_roots["release"]).parent / release_stage_leaf),
            "launch": str(PurePosixPath(final_roots["launch"]).parent / launch_stage_leaf),
            **{
                name: str(PurePosixPath(final_roots[name]).parent / runtime_stage_leaf)
                for name in ("attempt", "run", "screen_log")
            },
        }
        self.assertEqual(recipe["staging_roots"], staging_roots)
        for name, path in final_roots.items():
            self.assertIn(tag, path)
            self.assertNotIn("fresh-world8-level-b-p2-00435-v2", path)
            contract = recipe["root_contracts"][name]
            self.assertEqual(contract["stage_path"], staging_roots[name])
            self.assertEqual(contract["final_path"], path)
            self.assertEqual(contract["construction_mode"], 0o700)
            self.assertEqual(
                contract["sealed_mode_before_rename"], contract["final_mode"]
            )
            self.assertIs(contract["symlink_allowed"], False)
            self.assertIs(contract["directory_nlink_is_authority"], False)
        self.assertEqual(
            recipe["root_contracts"]["release"]["deployment_state"],
            "DEPLOYED_AND_INDEPENDENTLY_VERIFIED",
        )
        self.assertIs(
            recipe["root_contracts"]["release"]["stage_absent_after_rename"],
            True,
        )
        for name in ("launch", "attempt", "run", "screen_log"):
            self.assertEqual(
                recipe["root_contracts"][name]["deployment_state"],
                "NOT_DEPLOYED",
            )
        self.assertEqual(recipe["root_contracts"]["release"]["final_mode"], 0o555)
        self.assertEqual(recipe["root_contracts"]["launch"]["final_mode"], 0o555)
        for name in ("attempt", "run", "screen_log"):
            self.assertEqual(recipe["root_contracts"][name]["final_mode"], 0o700)
            self.assertEqual(recipe["root_contracts"][name]["exact_entries_before_launch"], [])
        self.assertEqual(
            recipe["root_contracts"]["release"]["exact_entries_before_launch"],
            [
                "RELEASE_MANIFEST.json",
                "action_preservation_decoded_eval_model_authority_v2.py",
                "infer_action_edit_level_b_renderer_0817_v1.py",
                "infer_lora.py",
                "tools",
            ],
        )
        self.assertEqual(
            recipe["root_contracts"]["launch"]["exact_entries_before_launch"],
            [
                "LAUNCH_AUTHORITY_CORE.json",
                BOOTSTRAP_PATH.name,
                RANK_PATH.name,
                STEP_PATH.name,
                CONTROLLER_PATH.name,
            ],
        )
        tools = recipe["nested_directory_contracts"]
        self.assertEqual(tools, [{
            "stage_path": staging_roots["release"] + "/tools",
            "final_path": final_roots["release"] + "/tools",
            "construction_mode": 0o700,
            "sealed_mode_before_rename": 0o555,
            "mode": 0o555,
            "symlink_allowed": False,
            "directory_nlink_is_authority": False,
            "exact_entries": ["build_renderer_dataset.py", "materialize_vae.py"],
        }])

        expected_local_to_final = {
            str(RELEASE_PATH.relative_to(ROOT)): final_roots["release"]
            + "/RELEASE_MANIFEST.json",
            str((METHOD / "action_preservation_decoded_eval_model_authority_v2.py").relative_to(ROOT)):
                final_roots["release"]
                + "/action_preservation_decoded_eval_model_authority_v2.py",
            str(RENDERER_PATH.relative_to(ROOT)): final_roots["release"]
            + "/infer_action_edit_level_b_renderer_0817_v1.py",
            str((METHOD / "infer_lora.py").relative_to(ROOT)): final_roots["release"]
            + "/infer_lora.py",
            str((METHOD / "tools" / "build_renderer_dataset.py").relative_to(ROOT)):
                final_roots["release"] + "/tools/build_renderer_dataset.py",
            str((METHOD / "tools" / "materialize_vae.py").relative_to(ROOT)):
                final_roots["release"] + "/tools/materialize_vae.py",
            str(CORE_PATH.relative_to(ROOT)): final_roots["launch"]
            + "/LAUNCH_AUTHORITY_CORE.json",
            str(BOOTSTRAP_PATH.relative_to(ROOT)): final_roots["launch"]
            + "/" + BOOTSTRAP_PATH.name,
            str(RANK_PATH.relative_to(ROOT)): final_roots["launch"]
            + "/" + RANK_PATH.name,
            str(STEP_PATH.relative_to(ROOT)): final_roots["launch"]
            + "/" + STEP_PATH.name,
            str(CONTROLLER_PATH.relative_to(ROOT)): final_roots["launch"]
            + "/" + CONTROLLER_PATH.name,
        }
        rows = recipe["file_inventory"]
        self.assertEqual(len(rows), len(expected_local_to_final))
        self.assertEqual(
            {row["local_path"]: row["final_path"] for row in rows},
            expected_local_to_final,
        )
        for row in rows:
            local = ROOT / row["local_path"]
            self.assertEqual(row["sha256"], sha256(local))
            self.assertEqual(row["size"], local.stat().st_size)
            self.assertEqual(row["mode"], 0o555 if local.suffix == ".sh" else 0o444)
            self.assertEqual(row["nlink"], 1)
            self.assertIs(row["symlink_allowed"], False)
            final = row["final_path"]
            if final.startswith(final_roots["release"] + "/"):
                expected_stage = staging_roots["release"] + final[len(final_roots["release"]):]
                self.assertIs(row["remote_deployed"], True)
                self.assertIs(row["remote_independently_verified"], True)
            else:
                expected_stage = staging_roots["launch"] + final[len(final_roots["launch"]):]
                self.assertIs(row["remote_deployed"], False)
                self.assertIs(row["remote_independently_verified"], False)
            self.assertEqual(row["stage_path"], expected_stage)

        completed_release = recipe["completed_release_atomic_rename"]
        self.assertEqual(completed_release["root"], "release")
        self.assertEqual(completed_release["stage_path"], staging_roots["release"])
        self.assertEqual(completed_release["final_path"], final_roots["release"])
        self.assertEqual(completed_release["operation"], "atomic-rename-no-replace")
        self.assertIs(completed_release["completed"], True)
        self.assertIs(completed_release["stage_absent_after"], True)
        self.assertIs(completed_release["independently_verified"], True)

        renames = recipe["pending_atomic_renames"]
        self.assertEqual(
            [row["root"] for row in renames],
            ["launch", "attempt", "run", "screen_log"],
        )
        for row in renames:
            name = row["root"]
            self.assertEqual(row["stage_path"], staging_roots[name])
            self.assertEqual(row["final_path"], final_roots[name])
            self.assertEqual(row["operation"], "atomic-rename-no-replace")
            self.assertIs(row["same_parent_filesystem_required"], True)
            self.assertIs(row["final_must_be_absent"], True)
            self.assertIs(row["stage_must_be_absent_after"], True)
            self.assertEqual(
                row["copy_paste_command"],
                "/usr/bin/mv -T --no-clobber -- "
                + staging_roots[name]
                + " "
                + final_roots[name]
                + " && /usr/bin/test ! -e "
                + staging_roots[name],
            )

        self.assertEqual(
            [phase["phase"] for phase in recipe["ordered_phases"]],
            [
                "release-already-deployed-and-verified",
                "deploy-sealed-launch-root",
                "foreground-static-preflight",
                "deploy-fresh-runtime-roots",
                "one-shot-screen-controller",
            ],
        )
        self.assertEqual(
            recipe["ordered_phases"][2]["must_complete_before"],
            ["attempt-root", "run-root", "screen-log-root", "screen", "controller"],
        )

        screen = recipe["screen"]
        self.assertEqual(screen["session_name"], "bernini-levelb-p2-00435-v3")
        self.assertEqual(
            screen["log_path"], final_roots["screen_log"] + "/controller.screen.log"
        )
        self.assertIs(screen["log_must_be_absent_before"], True)
        self.assertEqual(screen["created_log_mode"], 0o600)
        self.assertEqual(screen["created_log_nlink"], 1)
        self.assertIs(screen["log_outside_attempt_and_run_roots"], True)
        self.assertIs(screen["requires_foreground_static_preflight_pass"], True)
        command = screen["one_shot_command"]
        self.assertEqual(command.count("/usr/bin/screen"), 1)
        self.assertTrue(command.startswith("umask 077; exec /usr/bin/screen "))
        self.assertIn("-c /dev/null -dmS bernini-levelb-p2-00435-v3", command)
        self.assertIn("-L -Logfile " + screen["log_path"], command)
        self.assertTrue(command.endswith(final_roots["launch"] + "/" + CONTROLLER_PATH.name))
        self.assertNotIn("while ", command)
        self.assertNotIn("retry", command.lower())
        post_start = screen["post_start_read_only_log_verification"]
        self.assertEqual(post_start["expected_mode"], 0o600)
        self.assertEqual(post_start["expected_nlink"], 1)
        self.assertIs(post_start["regular_file_required"], True)
        self.assertIs(post_start["symlink_allowed"], False)
        for token in (
            "[[ -f \"${screen_log}\" && ! -L \"${screen_log}\" ]]",
            "stat -c %a",
            "stat -c %h",
            "== 600",
            "== 1",
        ):
            self.assertIn(token, post_start["copy_paste_command"])
        self.assertEqual(post_start["copy_paste_command"].count(screen["log_path"]), 1)
        self.assertIs(screen["created_only_after_foreground_gate"], True)
        self.assertIs(recipe["manual_screen_launch_is_separate_authority"], True)

    def test_static_preflight_authority_is_pinned_in_core_pins_and_handoff(self):
        expected = {
            "subcommand": "static-preflight",
            "stdout_sha256": bootstrap.STATIC_PREFLIGHT_STDOUT_SHA256,
            "stdout_size": bootstrap.STATIC_PREFLIGHT_STDOUT_SIZE,
            "pass_token": bootstrap.STATIC_PREFLIGHT_PASS_TOKEN,
            "combined_stdout_stderr_base64_sha256": (
                "594fabf3b03a53ea28e977c6fdc2c562e3d1e209701f01814a3d77baf0fa4417"
            ),
            "combined_stdout_stderr_base64_size": 38940,
            "combined_stdout_stderr_base64_size_formula": "4*ceil(stdout_size/3)",
            "base64_tool_path": "/usr/bin/base64",
            "base64_tool_sha256": (
                "b10f8c059f50c0681c6497e7b09ebdba168e341498ae1733de9089dc8efa0898"
            ),
            "base64_tool_size": 35336,
            "base64_tool_mode": 0o755,
            "base64_tool_nlink": 1,
            "internal_preflight_digest": (
                "bbebf0bddfd3bf36914fbf6122a1d44638c364ff05e7722e0c39ae792d5298a2"
            ),
            "stdout_has_trailing_lf": False,
            "cpu_only": True,
            "exact_five_release_authenticated": True,
            "before_started_latch": True,
            "before_srun": True,
            "cuda_visible_devices": "",
            "rocr_visible_devices": "",
            "hip_visible_devices": "",
            "cuda_initialized": False,
            "weights_loaded": False,
            "model_constructors_called": False,
            "output_files_written": False,
            "persistent_filesystem_writes": False,
            "ephemeral_devnull_rplus_open_count": 1,
            "ephemeral_devnull_write_open_count": 1,
            "ephemeral_devnull_os_open_count": 1,
            "ephemeral_devnull_os_write_open_count": 1,
            "ephemeral_devnull_st_rdev": 259,
            "ephemeral_devnull_major": 1,
            "ephemeral_devnull_minor": 3,
            "subprocesses_spawned_inside_preflight": False,
            "blocked_network_probe_attempt_count": 1,
            "outer_blocked_network_probe_delegation_count": 1,
            "socket_objects_created": False,
            "stdlib_socket_source_sha256": (
                bootstrap.STATIC_PREFLIGHT_STDLIB_SOCKET_SHA256
            ),
            "stdlib_socket_source_nlink": (
                bootstrap.STATIC_PREFLIGHT_STDLIB_SOCKET_NLINK
            ),
            "urllib3_connection_source_sha256": (
                bootstrap.STATIC_PREFLIGHT_URLLIB3_CONNECTION_SHA256
            ),
            "urllib3_connection_source_nlink": (
                bootstrap.STATIC_PREFLIGHT_URLLIB3_CONNECTION_NLINK
            ),
            "torch_jit_instantiator_source_sha256": (
                bootstrap.STATIC_PREFLIGHT_TORCH_JIT_INSTANTIATOR_SHA256
            ),
            "torch_jit_temporary_directory_calls_suppressed": 1,
            "torch_jit_atexit_registrations_suppressed": 1,
            "torch_jit_sys_path_append_removed": True,
            "torch_jit_meta_path_restored": True,
            "torch_jit_importer_cache_restored": True,
            "torch_remote_module_source_sha256": (
                bootstrap.STATIC_PREFLIGHT_TORCH_REMOTE_MODULE_SHA256
            ),
            "torch_remote_template_factory_calls_suppressed": 1,
            "torch_remote_template_source_written": False,
            "pinned_bernini_and_veomni_roots_scoped_and_restored": True,
            "scoped_module_source_closure_count": 52,
            "scoped_bernini_module_count": 13,
            "scoped_veomni_module_count": 39,
            "scoped_module_source_closure_digest": (
                "9db0b5160bd73caa229383119b79692a0d25dfcc1b03744d4cc293228bc93e96"
            ),
            "six_meta_path_importer_addition_count": 2,
            "six_meta_path_importers_restored": True,
            "six_meta_path_importer_scope_digest": (
                "8ff38cf49e2f585577f7f1c611f80b64a0d71ab05d9df9bb0a06ca7a86ae2224"
            ),
            "process_specific_repr_or_address_recorded": False,
            "openblas_main_free": "1",
            "gotoblas_main_free": "1",
            "veomni_verbosity": "ERROR",
            "veomni_logging_source_sha256": (
                bootstrap.STATIC_PREFLIGHT_VEOMNI_LOGGING_SHA256
            ),
            "veomni_timestamped_info_output_enabled": False,
            "numpy_core_init_source_sha256": (
                bootstrap.STATIC_PREFLIGHT_NUMPY_CORE_INIT_SHA256
            ),
            "numpy_environment_mutations_observed": 0,
            "network_accessed": False,
            "filesystem_mutation_audit_guard": True,
            "process_creation_audit_guard": True,
            "network_audit_guard": True,
            "canonical_stdout_held_in_shell_memory_only": True,
            "vendor_stdout_captured_in_memory": True,
            "vendor_stdout_bytes_observed": 0,
            "vendor_stdout_identity_restored": True,
            "foreground_gate_required_before_screen": True,
            "controller_gate_repeated_before_started": True,
            "stderr_merged_into_hashed_memory_boundary": True,
            "trailing_newline_only_stderr_preserved_by_base64_framing": True,
            "final_auh_fresh_process_count": 2,
            "final_auh_canonical_stdout_identical": True,
            "final_auh_stderr_empty": True,
            "final_auh_stderr_sha256": hashlib.sha256(b"").hexdigest(),
            "final_auh_stderr_size": 0,
            "final_auh_process_return_codes": [0, 0],
            "final_auh_exact_bootstrap_pre_post_identical": True,
            "final_auh_pre_post_inventory_sha256": (
                "8e08a4fafd10676b880f724191954e3177ecf16364fc5d71244e2fe5dc2c4b1e"
            ),
            "final_auh_evidence_root": (
                str(bootstrap.EXPERIMENT_ROOT)
                + "/launchers/.fresh-world8-level-b-p2-00435-v3.final-preflight-"
                + sha256(BOOTSTRAP_PATH)[:16]
            ),
            "final_auh_evidence_root_mode": 0o555,
            "final_auh_evidence_root_nlink": 2,
            "final_auh_evidence_root_dev_ino": "48:17966305388552742281",
            "final_auh_bootstrap_mode": 0o444,
            "final_auh_bootstrap_nlink": 1,
            "final_auh_bootstrap_dev_ino": "48:12278006435496194716",
            "final_auh_nonexistent_sandbox_remained_absent": True,
            "final_auh_numeric_parent_child_count": 0,
        }
        self.assertEqual(self.core["static_preflight"], expected)
        self.assertEqual(self.pins["static_preflight"], expected)
        self.assertEqual(
            expected["combined_stdout_stderr_base64_size"],
            4 * ((expected["stdout_size"] + 2) // 3),
        )
        handoff = self.pins["deployment_recipe"]["prelaunch_static_preflight"]
        self.assertEqual(handoff["stdout_sha256"], expected["stdout_sha256"])
        self.assertEqual(handoff["stdout_size"], expected["stdout_size"])
        self.assertEqual(
            handoff["combined_stdout_stderr_base64_sha256"],
            expected["combined_stdout_stderr_base64_sha256"],
        )
        self.assertEqual(
            handoff["combined_stdout_stderr_base64_size"],
            expected["combined_stdout_stderr_base64_size"],
        )
        self.assertEqual(
            handoff["base64_tool"],
            {
                "path": expected["base64_tool_path"],
                "sha256": expected["base64_tool_sha256"],
                "size": expected["base64_tool_size"],
                "mode": expected["base64_tool_mode"],
                "nlink": expected["base64_tool_nlink"],
            },
        )
        self.assertEqual(handoff["pass_token"], expected["pass_token"])
        self.assertIs(handoff["manual_run_required_after_deployment"], True)
        self.assertIs(handoff["foreground_gate_is_launch_prerequisite"], True)
        self.assertIs(handoff["manual_run_is_launch_authority"], False)
        self.assertIs(handoff["foreground_gate_is_required_safety_authority"], True)
        self.assertIs(handoff["foreground_gate_grants_launch_permission"], False)
        self.assertIs(handoff["controller_repeats_before_started"], True)
        self.assertIs(handoff["must_complete_before_runtime_root_creation"], True)
        self.assertIs(handoff["must_complete_before_screen_socket_or_log"], True)
        self.assertIs(handoff["stderr_merged_into_hashed_memory_boundary"], True)
        self.assertIs(handoff["remote_persistent_writes"], False)
        command = handoff["copy_paste_command"]
        for token in (
            "/usr/bin/env -i",
            "PATH=/usr/bin:/bin",
            "HOME=/nonexistent/bernini-level-b-p2-00435-v3",
            "TMPDIR=/nonexistent/bernini-level-b-p2-00435-v3/tmp",
            "CUDA_VISIBLE_DEVICES=''",
            "ROCR_VISIBLE_DEVICES=''",
            "HIP_VISIBLE_DEVICES=''",
            "PYTHONDONTWRITEBYTECODE=1",
            "PYTHONNOUSERSITE=1",
            "HF_HUB_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1",
            "OPENBLAS_MAIN_FREE=1",
            "GOTOBLAS_MAIN_FREE=1",
            "VEOMNI_VERBOSITY=ERROR",
            str(bootstrap.PYTHON_PATH),
            "-I -B",
            self.pins["roots"]["launch"] + "/" + BOOTSTRAP_PATH.name,
            "static-preflight",
            "2>&1",
            sha256(BOOTSTRAP_PATH),
            bootstrap.STATIC_PREFLIGHT_STDOUT_SHA256,
            str(bootstrap.STATIC_PREFLIGHT_STDOUT_SIZE),
            expected["combined_stdout_stderr_base64_sha256"],
            str(expected["combined_stdout_stderr_base64_size"]),
            expected["base64_tool_sha256"],
            "PIPESTATUS",
            "__LEVEL_B_P2_00435_V3_STATIC_PREFLIGHT_PIPESTATUS_",
        ):
            self.assertIn(token, command)
        self.assertEqual(command.count("2>&1"), 1)
        command_without_stderr_merge = command.replace("2>&1", "")
        for forbidden in ("srun", "torchrun", "screen", ">", "tee "):
            self.assertNotIn(forbidden, command_without_stderr_merge)

    def test_v3_supersedes_failed_v2_without_reuse_overwrite_or_retry(self):
        value = self.pins["supersession"]
        failed = value["superseded_attempt"]
        self.assertEqual(failed["tag"], "fresh-world8-level-b-p2-00435-v2")
        self.assertEqual(failed["child_step_id"], "140846.371")
        self.assertEqual(
            failed["sacct_exact"],
            "140846.371|bernini0817-level-b-p2-00435-v2||faculty-acc|FAILED|"
            "1:0|00:06:03|2026-08-17T08:16:46|2026-08-17T08:22:49|"
            "auh7-1b-gpu-279|32||cpu=32,gres/gpu:mi210=8,gres/gpu=8,mem=64G,node=1",
        )
        self.assertEqual(failed["state"], "FAILED")
        self.assertEqual(failed["exit_code"], "1:0")
        self.assertEqual(failed["elapsed"], "00:06:03")
        self.assertEqual(failed["elapsed_seconds"], 363)
        self.assertEqual(failed["max_rss"], "67027324K")
        self.assertEqual(failed["failure_phase"], "vae_load_and_source_encode")
        self.assertEqual(
            failed["failure_reason"],
            "live loaded VAE encode callable is not owned by its pinned source bytes",
        )
        self.assertIs(failed["load_config_decorator_gate_passed"], True)
        self.assertIs(failed["vae_loaded"], True)
        self.assertEqual(failed["failed_rank_count"], 8)
        self.assertIs(failed["rank0_failure_propagated_to_all_ranks"], True)
        self.assertIs(failed["failed_before_denoise"], True)
        self.assertIs(failed["failed_before_output"], True)
        self.assertEqual(failed["run_root_mode"], 0o700)
        self.assertEqual(failed["run_root_exact_member_count"], 0)
        self.assertIs(failed["mp4_present"], False)
        self.assertIs(failed["receipt_present"], False)
        self.assertIs(failed["committed_marker_present"], False)
        self.assertEqual(
            failed["run_log_sha256"],
            "d2912478b6ec6214d38b70af6be548cb7364fa53cc1327e3f4ce3dfcee1cb87e",
        )
        self.assertEqual(failed["run_log_size"], 56412)
        self.assertEqual(failed["run_log_mode"], 0o600)
        self.assertEqual(failed["run_log_nlink"], 1)
        self.assertEqual(
            failed["intent_sha256"],
            "6b4f5b118af5a3c982fcdf78fbff9fbb35732a4dd6cab5c267d4e5b02ebff068",
        )
        self.assertEqual(failed["intent_size"], 2250)
        self.assertEqual(failed["intent_mode"], 0o444)
        self.assertEqual(failed["intent_nlink"], 1)
        self.assertEqual(
            failed["screen_log_sha256"],
            "b0db1836e2133b86f87622a504b8b26fa74a64bcccfa2a8b491179d5f84aad79",
        )
        self.assertEqual(failed["screen_log_size"], 29909)
        self.assertEqual(failed["screen_log_mode"], 0o644)
        self.assertEqual(failed["screen_log_nlink"], 1)
        self.assertIs(failed["screen_exited"], True)
        self.assertIs(failed["controller_status_present"], True)
        self.assertEqual(
            failed["controller_status_sha256"],
            "516cf0d74be8ebed6d5298da38ce8541240cc72b46b0c370d7b81010f8768f08",
        )
        self.assertEqual(failed["controller_status_size"], 821)
        self.assertEqual(failed["controller_status_mode"], 0o444)
        self.assertEqual(failed["controller_status_nlink"], 1)
        self.assertEqual(
            failed["controller_status_schema"],
            "bernini-action-edit-level-b-p2-controller-status-v2",
        )
        self.assertEqual(failed["controller_status_child_exit_code"], 1)
        self.assertEqual(
            failed["parent_state_before"],
            "RUNNING|auh7-1b-gpu-[246-248,279]|gres/gpu:mi210:8",
        )
        self.assertEqual(failed["parent_state_after"], failed["parent_state_before"])
        self.assertIs(failed["controller_status_automatic_relaunch_authorized"], False)
        self.assertIs(failed["controller_status_parent_control_authorized"], False)
        self.assertIs(failed["controller_status_fix_verified_live"], True)
        self.assertIs(failed["parent_untouched"], True)
        self.assertIs(failed["gpus_released"], True)
        self.assertEqual(failed["gpu_process_count_after"], 0)
        self.assertEqual(failed["gpu_utilization_percent_after"], 0)
        self.assertEqual(failed["gpu_vram_bytes_after"], 0)
        self.assertIs(failed["automatic_retry_performed"], False)
        self.assertEqual(
            failed["release_manifest_sha256"],
            "7cdfc7213d7edd68b580bf736cb0645ff1a41d4f3d70c55890a312961ce68b06",
        )
        self.assertEqual(
            failed["renderer_sha256"],
            "b07ec9603e4820b4b9cb52fe7c8994591e3e5e30afb71b504086b0d259c06d68",
        )
        self.assertEqual(
            failed["controller_sha256"],
            "0d80a83508ebd8a5881d8ca92bc6f4fd125e40c63cd98c7802e53b2a27be17b9",
        )
        self.assertEqual(
            failed["superseded_deployment_pins_sha256"],
            "bec8eb16a2bf3eecba00c6e37564f40900e402e3de164901d384542a821ccf6b",
        )
        replacement = value["replacement"]
        self.assertEqual(replacement["tag"], "fresh-world8-level-b-p2-00435-v3")
        self.assertIs(replacement["fresh_roots_required"], True)
        self.assertIs(replacement["reuse_v2_attempt_or_run"], False)
        self.assertIs(replacement["automatic_retry_of_v2"], False)
        self.assertEqual(replacement["controller_sha256"], sha256(CONTROLLER_PATH))
        self.assertIn("preserve", replacement["controller_status_fix"])
        self.assertIn("encode", replacement["renderer_fix"])
        self.assertIn("apply_forward_hook", replacement["renderer_fix"])
        for root in self.core["roots"].values():
            self.assertIn("fresh-world8-level-b-p2-00435-v3", root)
            self.assertNotIn("fresh-world8-level-b-p2-00435-v2", root)
        self.assertEqual(self.core["supersession"]["superseded_attempt"], failed)
        core_replacement = self.core["supersession"]["replacement"]
        self.assertNotIn("controller_sha256", core_replacement)
        self.assertEqual(
            core_replacement,
            {key: item for key, item in replacement.items() if key != "controller_sha256"},
        )

    def test_v3_explicitly_closes_v1_to_v2_to_v3_without_old_root_mutation(self):
        history = self.pins["supersession"]["prior_v1_to_v2"]
        v2_core = json.loads(V2_CORE_PATH.read_text())
        self.assertEqual(
            history["v2_launch_authority_core"],
            {
                "sha256": sha256(V2_CORE_PATH),
                "size": V2_CORE_PATH.stat().st_size,
            },
        )
        self.assertEqual(
            history["v2_deployment_pins"],
            {
                "sha256": "bec8eb16a2bf3eecba00c6e37564f40900e402e3de164901d384542a821ccf6b",
                "size": 14123,
            },
        )
        self.assertEqual(
            history["superseded_v1_attempt"],
            v2_core["supersession"]["superseded_attempt"],
        )
        self.assertEqual(
            history["v2_replacement"],
            v2_core["supersession"]["replacement"],
        )
        self.assertEqual(history["transition"], "v1-.370-to-v2-.371")
        for key in (
            "v1_roots_immutable",
            "v2_roots_immutable",
            "reuse_v1_attempt_or_run",
            "overwrite_v1_or_v2_roots",
            "automatic_retry_of_v1",
        ):
            expected = key.endswith("_immutable")
            self.assertIs(history[key], expected)
        self.assertEqual(
            self.core["supersession"]["prior_v1_to_v2"],
            history,
        )
        active = canonical_json_bytes({
            "roots": self.pins["roots"],
            "output": self.pins["output"],
            "launch_chain": self.pins["launch_chain"],
        }).decode("utf-8")
        self.assertNotIn("fresh-world8-level-b-p2-00435-v1", active)
        self.assertNotIn("fresh-world8-level-b-p2-00435-v2", active)

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
        scoped_closure = literal_assignment(
            RENDERER_PATH, "PINNED_CPU_STATIC_SCOPED_MODULE_CLOSURE"
        )
        self.assertEqual(len(scoped_closure), 52)
        expected_bernini_sources = {
            "bernini/cli.py":
                "26949fbf246003403ed0cca1ec1bbb62c2099fc9740bb17ba5a1e7c86fbc0edf",
            "bernini/io_utils.py":
                "233541373746f5d97e1cb3680d3c2a41d5d212b797eefb97693afa6e3ab5f30a",
            "bernini/pipeline.py":
                "c6acf05c01a637d9bce69e8160eb6eb4260ff4ec798fd990de8e5aa73999ab40",
            **{
                relative: source_sha
                for _module, prefix, relative, source_sha, _size in scoped_closure
                if prefix == "bernini"
            },
        }
        expected_veomni_sources = {
            relative: source_sha
            for _module, prefix, relative, source_sha, _size in scoped_closure
            if prefix == "veomni"
        }
        self.assertEqual(
            self.pins["runtime_authority"]["bernini_source_sha256"],
            expected_bernini_sources,
        )
        self.assertEqual(
            self.pins["runtime_authority"]["veomni_source_sha256"],
            expected_veomni_sources,
        )
        self.assertEqual(
            self.pins["runtime_authority"]["cpu_static_scoped_module_closure"],
            [list(row) for row in scoped_closure],
        )
        self.assertEqual(
            self.pins["runtime_authority"]["site_package_source_sha256"],
            literal_assignment(RENDERER_PATH, "PINNED_SITE_PACKAGE_SOURCE_HASHES"),
        )
        site_pins = self.pins["runtime_authority"]["site_package_source_sha256"]
        for relative, expected_sha in {
            "botocore/vendored/six.py":
                "4ce39f422ee71467ccac8bed76beb05f8c321c7f0ceda9279ae2dfa3670106b3",
            "diffusers/utils/accelerate_utils.py":
                "664a2938adbdffa42badd9083e27479ced3bf01f01b73cc54adb37ba5d9c3fc4",
            "diffusers/models/modeling_outputs.py":
                "5c7dec24edf83115ba52e5aaa8aa34e6656ac464811516b3a5aa7ff982f03b62",
            "diffusers/models/autoencoders/vae.py":
                "90f6db6ed05b3a6bd61ab1abefc0414ebacf730f89135e6c4b2155b52c001d72",
            "torch/utils/_contextlib.py":
                "cf7aa5b08f44974ba8c1d08cd71ef70ffd13d1c48a4931576eb235306bfa46b5",
            "torch/autograd/grad_mode.py":
                "a67ddb0da569646f5d3806e248c2093b6e2f75f0a6b4959ab966e119c1b28d6d",
            "torch/distributed/nn/jit/instantiator.py":
                "567d1314ee27ff0b3bd22e7c4d1157246469de25e7a3183d96debe167b193615",
            "torch/distributed/nn/api/remote_module.py":
                "f9bb2f5c5438791581d399e38a27606e123bdbeb3c6cb53683318a06060439c1",
            "numpy/core/__init__.py":
                "08db0ef806f8cb03365b3dc06ea58e1f78a0d6ae419e8f4fb1432b0aff87352e",
            "six.py":
                "c51c91f703d3d4b3696c923cb5fec213e05e75d9215393befac7f2fa6a3904df",
            "urllib3/util/connection.py":
                "2633bbdb69731e5ccb5cf4e4afd65605d86c7979cc5633126f50c92d5ad74a74",
        }.items():
            self.assertEqual(site_pins[relative], expected_sha)
        self.assertEqual(
            self.pins["runtime_authority"]["stdlib_socket_source"],
            {
                "path": str(bootstrap.STATIC_PREFLIGHT_STDLIB_SOCKET_PATH),
                "sha256": bootstrap.STATIC_PREFLIGHT_STDLIB_SOCKET_SHA256,
                "size": bootstrap.STATIC_PREFLIGHT_STDLIB_SOCKET_SIZE,
                "nlink": bootstrap.STATIC_PREFLIGHT_STDLIB_SOCKET_NLINK,
            },
        )
        self.assertEqual(
            self.pins["runtime_authority"]["bernini_root"],
            str(bootstrap.BERNINI_ROOT),
        )
        self.assertEqual(
            self.pins["runtime_authority"]["veomni_root"],
            str(bootstrap.VEOMNI_ROOT),
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
        calibration = tests["static_preflight_calibration_harness"]
        self.assertEqual(calibration["sha256"], sha256(CALIBRATION_PATH))
        self.assertEqual(calibration["size"], CALIBRATION_PATH.stat().st_size)
        self.assertIs(calibration["is_release_member"], False)
        self.assertIs(calibration["is_launch_member"], False)
        self.assertEqual(
            calibration["pending_bootstrap_sha256"],
            "f11a05e8c91a6cfc041a146b12e1853875329308bc1e85848e59b4b931643c69",
        )
        self.assertEqual(calibration["pending_bootstrap_size"], 57335)
        self.assertEqual(
            calibration["two_literal_calibration_source_sha256"],
            "4c4dc754460a71f542052c7de079b464e909aad59dffc654c2d5074231898759",
        )
        self.assertEqual(calibration["two_literal_calibration_source_size"], 57362)
        self.assertEqual(calibration["assignment_replacement_counts"], [1, 1])
        self.assertIs(calibration["function_body_modified"], False)
        self.assertEqual(calibration["additional_preloaded_modules"], ["_ast", "ast"])
        evidence_pin = tests["static_preflight_calibration_evidence"]
        self.assertEqual(evidence_pin["sha256"], sha256(CALIBRATION_EVIDENCE_PATH))
        self.assertEqual(evidence_pin["size"], CALIBRATION_EVIDENCE_PATH.stat().st_size)
        self.assertIs(evidence_pin["is_release_member"], False)
        self.assertIs(evidence_pin["is_launch_member"], False)

    def test_calibration_raw_mechanically_binds_stdout_base64_and_digest(self):
        evidence = json.loads(CALIBRATION_EVIDENCE_PATH.read_text())
        self.assertEqual(
            CALIBRATION_EVIDENCE_PATH.read_bytes(),
            canonical_json_bytes(evidence) + b"\n",
        )
        raw = base64.b64decode(evidence["canonical_stdout_base64"], validate=True)
        self.assertEqual(len(raw), evidence["canonical_stdout_size"])
        self.assertEqual(hashlib.sha256(raw).hexdigest(), evidence["canonical_stdout_sha256"])
        self.assertEqual(raw[-1:], b"}")
        self.assertIs(evidence["canonical_stdout_has_trailing_lf"], False)
        self.assertEqual(raw, canonical_json_bytes(json.loads(raw)))
        encoded = base64.b64encode(raw)
        self.assertEqual(len(encoded), 4 * ((len(raw) + 2) // 3))
        self.assertEqual(len(encoded), evidence["base64_size"])
        self.assertEqual(hashlib.sha256(encoded).hexdigest(), evidence["base64_sha256"])
        self.assertEqual(
            evidence["canonical_stdout_sha256"],
            self.core["static_preflight"]["stdout_sha256"],
        )
        self.assertEqual(
            evidence["base64_sha256"],
            self.core["static_preflight"]["combined_stdout_stderr_base64_sha256"],
        )
        result = json.loads(raw)
        digest = result.pop("preflight_digest")
        self.assertEqual(hashlib.sha256(canonical_json_bytes(result)).hexdigest(), digest)
        self.assertEqual(digest, evidence["canonical_stdout_internal_preflight_digest"])
        self.assertEqual(digest, self.core["static_preflight"]["internal_preflight_digest"])

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

    def test_every_shell_command_substitution_has_an_explicit_status_boundary(self):
        for path in (CONTROLLER_PATH, STEP_PATH, RANK_PATH):
            for lineno, line in enumerate(path.read_text().splitlines(), 1):
                if "$(" not in line:
                    continue
                self.assertRegex(
                    line,
                    r"^\s*if ! [A-Za-z_][A-Za-z0-9_]*=\"\$\(",
                    f"{path}:{lineno}: {line}",
                )
                self.assertNotRegex(line, r"\b(?:readonly|local)\b")

    def test_no_launcher_has_unbounded_retry_loop(self):
        for text in (self.controller, self.step, self.rank):
            self.assertNotIn("while true", text.lower())
            self.assertNotIn("until ", text.lower())
        self.assertEqual(self.controller.count("for poll in"), 1)

    def test_bootstrap_accepts_no_caller_semantic_arguments(self):
        source = self.bootstrap_text[self.bootstrap_text.index("def main") :]
        self.assertIn('values == ["run"]', source)
        self.assertIn('values == ["validate-product"]', source)
        self.assertIn('values == ["static-preflight"]', source)
        self.assertNotIn("argparse", source)

    def test_output_name_and_unique_tag_are_fixed(self):
        self.assertEqual(
            bootstrap.OUTPUT_MP4.name,
            "00435ad621c44fac_p2_seed2026080821_v3.mp4",
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
            self.assertIn("fresh-world8-level-b-p2-00435-v3", text)
        self.assertNotEqual(bootstrap.TAG, "fresh-world8-level-a-r2-p2-launchbound-v2")


if __name__ == "__main__":
    unittest.main()
