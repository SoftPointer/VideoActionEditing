#!/usr/bin/env python3
"""Local-only release tests for the exact18 R14 gpu280 holder port."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = METHOD_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import build_generic_action_fit40_r14_gpu280_port_v1 as port  # noqa: E402


FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures/generic_action_fit40_r14_gpu280_port_v1.json"
)
RUNTIME_SUITE_PATH = (
    Path(__file__).resolve().parent
    / "generic_action_fit40_r14_gpu280_runtime_suite_v1.py"
)
DEFAULT_FROZEN_ROOT = Path("/private/tmp/action-r14-isolated.Fgh3sC")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GenericActionFit40R14Gpu280PortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(FIXTURE_PATH.read_text(encoding="ascii"))
        root = Path(os.environ.get("R14_FROZEN_ROOT", str(DEFAULT_FROZEN_ROOT)))
        cls.source_archive = (
            root / ".r14-determinism-1/release.tar"
        ).resolve(strict=True)
        cls.source_manifest = (
            root / ".r14-determinism-1/release.manifest.json"
        ).resolve(strict=True)
        cls.temporary = tempfile.TemporaryDirectory(prefix="fit40-r14-gpu280-test.")
        cls.output_root = Path(cls.temporary.name).resolve(strict=True)
        cls.archive = cls.output_root / "source.tar"
        cls.manifest = cls.output_root / "source.manifest.json"
        cls.receipt = cls.output_root / "source.port-receipt.json"
        port.build(
            cls.source_archive,
            cls.source_manifest,
            cls.archive,
            cls.manifest,
            cls.receipt,
        )
        cls.extract_root = cls.output_root / "extract"
        cls.extract_root.mkdir(mode=0o700)
        with tarfile.open(cls.archive, mode="r:") as archive:
            for member in archive.getmembers():
                target = Path(member.name)
                if target.is_absolute() or ".." in target.parts or not member.isfile():
                    raise AssertionError("unsafe ported archive member")
            archive.extractall(cls.extract_root)
        cls.extracted_method_root = cls.extract_root / port.MEMBER_ROOT

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def _payloads(self, archive_path: Path | None = None) -> dict[str, bytes]:
        path = self.archive if archive_path is None else archive_path
        with tarfile.open(path, mode="r:") as archive:
            return {
                member.name.removeprefix(port.MEMBER_ROOT + "/"): archive.extractfile(
                    member
                ).read()
                for member in archive.getmembers()
            }

    def test_frozen_source_pair_is_exact_authority(self) -> None:
        source = self.fixture["source"]
        self.assertEqual(_sha(self.source_archive), source["archive_sha256"])
        self.assertEqual(_sha(self.source_manifest), source["manifest_sha256"])
        manifest = json.loads(self.source_manifest.read_text(encoding="ascii"))
        self.assertEqual(manifest["manifest_digest"], source["manifest_digest"])
        self.assertEqual(
            manifest["content_closure_sha1"], source["content_closure_sha1"]
        )
        self.assertEqual(manifest["file_count"], 18)

    def test_candidate_hashes_exact18_and_port_receipt_binding(self) -> None:
        expected = self.fixture["port"]
        self.assertEqual(_sha(self.archive), expected["archive_sha256"])
        self.assertEqual(_sha(self.manifest), expected["manifest_sha256"])
        manifest = json.loads(self.manifest.read_text(encoding="ascii"))
        receipt = json.loads(self.receipt.read_text(encoding="ascii"))
        self.assertEqual(manifest["file_count"], 18)
        self.assertEqual(manifest["manifest_digest"], expected["manifest_digest"])
        self.assertEqual(
            manifest["content_closure_sha1"], expected["content_closure_sha1"]
        )
        # The receipt's source object intentionally binds only cryptographic
        # release identity; fixture-only census fields are checked separately.
        for field in (
            "archive_sha256",
            "manifest_sha256",
            "manifest_digest",
            "content_closure_sha1",
            "file_count",
        ):
            self.assertEqual(receipt["source"][field], self.fixture["source"][field])
        for field in (
            "holder_job",
            "holder_node",
            "confirmation",
            "plan_id",
            "launcher_member",
            "archive_sha256",
            "manifest_sha256",
            "manifest_digest",
            "content_closure_sha1",
            "file_count",
        ):
            self.assertEqual(receipt["port"][field], expected[field])
        unsigned = dict(receipt)
        declared = unsigned.pop("receipt_digest")
        self.assertEqual(declared, port.object_sha256(unsigned))
        self.assertEqual(receipt["unchanged_member_count"], 14)
        self.assertEqual(len(receipt["transforms"]), 4)

    def test_manifest_authority_is_fit40_pending_review_only(self) -> None:
        source = json.loads(self.source_manifest.read_text(encoding="ascii"))
        candidate = json.loads(self.manifest.read_text(encoding="ascii"))
        mutable = {
            "files",
            "component_pins",
            "allowed_entrypoints",
            "content_closure_sha1",
            "manifest_digest",
        }
        self.assertEqual(
            {key: value for key, value in candidate.items() if key not in mutable},
            {key: value for key, value in source.items() if key not in mutable},
        )
        # release_generation names the unchanged embedded runtime contract.
        # The holder-port revision is independently bound by PORT_RECEIPT_SCHEMA
        # and PORT_PLAN_ID; changing this field would widen the frozen transform.
        self.assertEqual(candidate["release_generation"], "r14")
        self.assertEqual(
            port.PORT_RECEIPT_SCHEMA,
            "bernini-generic-action-fit40-r14-gpu280-port-release-v1",
        )
        authority = candidate["authority"]
        self.assertEqual(authority["analysis_split"], "fit")
        self.assertEqual(authority["candidate_count"], 40)
        self.assertEqual(authority["seed_cell_count"], 4)
        for field in (
            "confirmation_generation_authorized",
            "independent_full81_blind_review_present",
            "phi_materializer_present",
            "phi_v1_extraction_authorized",
            "p_or_o_manifest_materialization_authorized",
            "optimizer_created",
            "optimizer_authorized",
            "training_authorized",
        ):
            self.assertIs(authority[field], False, field)
        self.assertEqual(
            authority["generated_media_role"],
            "pending-external-review-authoring-media-only",
        )
        topology = candidate["topology"]
        self.assertEqual(topology["run_sp4_shard_process_count"], 4)
        self.assertEqual(topology["world4_model_invocation_count"], 40)
        self.assertTrue(topology["all_model_invocations_strictly_serial"])
        self.assertFalse(topology["rank_or_gpu_action_family_partition"])

    def test_exact_four_postimages_and_fourteen_unchanged_members(self) -> None:
        source_manifest = json.loads(self.source_manifest.read_text(encoding="ascii"))
        source_payloads, source_modes = port._load_source_archive(
            self.source_archive.read_bytes(), source_manifest
        )
        candidate_payloads = self._payloads()
        expected = self.fixture["transformed_members"]
        transformed_paths = set(expected)
        for path, values in expected.items():
            source_path = values.get("source_member", path)
            self.assertEqual(
                hashlib.sha256(source_payloads[source_path]).hexdigest(),
                values["preimage_sha256"],
            )
            self.assertEqual(
                hashlib.sha256(candidate_payloads[path]).hexdigest(),
                values["postimage_sha256"],
            )
        unchanged = set(candidate_payloads) - transformed_paths
        self.assertEqual(len(unchanged), 14)
        self.assertEqual(unchanged, set(source_payloads) - set(port.TRANSFORMS))
        for path in unchanged:
            self.assertEqual(candidate_payloads[path], source_payloads[path])
            candidate_row = next(
                row
                for row in json.loads(self.manifest.read_text(encoding="ascii"))["files"]
                if row["path"] == path
            )
            self.assertEqual(candidate_row["mode"], source_modes[path])

    def test_old_holder_node_launcher_tokens_are_zero(self) -> None:
        payloads = self._payloads()
        joined = b"\n".join(payloads.values())
        for token in (
            b"136141",
            b"auh7-1b-gpu-299",
            b"gpu299",
            b"auh_generic_action_data_prep_136141_world4_v1.sh",
        ):
            self.assertEqual(joined.count(token), 0, token)
        self.assertIn(b"136309", joined)
        self.assertIn(b"auh7-1b-gpu-280", joined)

    def test_generator_has_eight_job_bindings_plus_one_diagnostic(self) -> None:
        path = self.extracted_method_root / "tools/reserve4_fixed_generation_sp4_v1.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        exact = sorted(
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and node.value == "136309"
        )
        diagnostic = sorted(
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and "retained 136309 Slurm child" in node.value
        )
        expected = self.fixture["transformed_members"][
            "tools/reserve4_fixed_generation_sp4_v1.py"
        ]
        self.assertEqual(exact, expected["exact_job_binding_ast_lines"])
        self.assertEqual(diagnostic, [expected["diagnostic_ast_line"]])

    def test_controller_launcher_and_builder_port_pins(self) -> None:
        controller_path = self.extracted_method_root / "generic_action_data_prep_controller_v1.py"
        controller = _load_module(controller_path, "gpu280_port_controller")
        self.assertEqual(controller.HOLDER_JOB, 136309)
        self.assertEqual(controller.HOLDER_NODE, "auh7-1b-gpu-280")
        self.assertEqual(controller.LAUNCH_CONFIRMATION, port.PORT_CONFIRMATION)
        source = controller_path.read_text(encoding="utf-8")
        self.assertIn(f'"plan_id": "{port.PORT_PLAN_ID}"', source)
        self.assertIn(port.PORT_LAUNCHER, source)
        builder_path = self.extracted_method_root / "tools/build_generic_action_data_prep_release_v1.py"
        builder = _load_module(builder_path, "gpu280_port_builder")
        self.assertIn(port.PORT_LAUNCHER, builder.FILES_AND_MODES)
        self.assertNotIn(port.SOURCE_LAUNCHER, builder.FILES_AND_MODES)
        self.assertEqual(builder.ENTRYPOINTS[-1], port.PORT_LAUNCHER)
        launcher = (self.extracted_method_root / port.PORT_LAUNCHER).read_text(
            encoding="utf-8"
        )
        self.assertIn("readonly holder_job=136309", launcher)
        self.assertIn("readonly holder_node=auh7-1b-gpu-280", launcher)
        self.assertIn(f"readonly launch_confirmation={port.PORT_CONFIRMATION}", launcher)
        self.assertIn(f'launcher="${{method_root}}/{port.PORT_LAUNCHER}"', launcher)
        for forbidden in ("scancel", "scontrol release", "scontrol requeue"):
            self.assertNotIn(forbidden, launcher)

    def test_extracted_builder_self_audits_and_shell_is_valid(self) -> None:
        builder = self.extracted_method_root / "tools/build_generic_action_data_prep_release_v1.py"
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                str(builder),
                "audit",
                "--archive",
                str(self.archive),
                "--manifest",
                str(self.manifest),
                "--expected-archive-sha256",
                _sha(self.archive),
                "--expected-manifest-sha256",
                _sha(self.manifest),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        audited = json.loads(result.stdout)
        self.assertEqual(audited["manifest_digest"], self.fixture["port"]["manifest_digest"])
        subprocess.run(
            ["bash", "-n", str(self.extracted_method_root / port.PORT_LAUNCHER)],
            check=True,
        )

    def test_extracted_runtime_passes_frozen_r14_semantic_suite(self) -> None:
        suite = self.fixture["runtime_suite"]
        self.assertEqual(_sha(RUNTIME_SUITE_PATH), suite["port_sha256"])
        source = RUNTIME_SUITE_PATH.read_text(encoding="utf-8")
        for token in ("136141", "job_136141", "gpu299", "GPU299"):
            self.assertNotIn(token, source)
        self.assertIn("job_136310", source)
        self.assertIn(
            "bernini-generic-action-fit40-generation-136309-plan-v10", source
        )
        environment = {
            **os.environ,
            "GADP_GPU280_PORT_METHOD_ROOT": str(self.extracted_method_root),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
        result = subprocess.run(
            [sys.executable, "-B", str(RUNTIME_SUITE_PATH)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"runtime suite failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        self.assertIn(f"Ran {suite['test_count']} tests", result.stderr)
        self.assertIn("OK", result.stderr)
        subprocess.run(
            [
                "bash",
                "-n",
                str(
                    self.extracted_method_root
                    / "scripts/auh_generic_action_data_prep_rank_exec_v1.sh"
                ),
            ],
            check=True,
        )

    def test_all_python_members_compile(self) -> None:
        paths = sorted(self.extracted_method_root.rglob("*.py"))
        self.assertEqual(len(paths), 16)
        for path in paths:
            compile(path.read_bytes(), str(path), "exec", dont_inherit=True)

    def test_three_independent_builds_are_byte_identical(self) -> None:
        rows = []
        for index in range(3):
            root = self.output_root / f"determinism-{index}"
            root.mkdir(mode=0o700)
            archive = root / "source.tar"
            manifest = root / "source.manifest.json"
            receipt = root / "source.port-receipt.json"
            port.build(
                self.source_archive,
                self.source_manifest,
                archive,
                manifest,
                receipt,
            )
            rows.append((archive.read_bytes(), manifest.read_bytes(), receipt.read_bytes()))
        self.assertEqual(rows[0], rows[1])
        self.assertEqual(rows[1], rows[2])

    def test_tampered_source_pair_and_transform_preimage_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gpu280-hostile.") as temporary:
            root = Path(temporary).resolve(strict=True)
            bad_manifest = root / "source.manifest.json"
            raw = bytearray(self.source_manifest.read_bytes())
            raw[-2] = ord(" ")
            bad_manifest.write_bytes(raw)
            with self.assertRaisesRegex(port.PortReleaseError, "manifest SHA-256"):
                port.build(
                    self.source_archive,
                    bad_manifest,
                    root / "bad.tar",
                    root / "bad.manifest.json",
                    root / "bad.receipt.json",
                )
            source = json.loads(self.source_manifest.read_text(encoding="ascii"))
            payloads, modes = port._load_source_archive(
                self.source_archive.read_bytes(), source
            )
            hostile = dict(payloads)
            target = "tools/reserve4_fixed_generation_sp4_v1.py"
            hostile[target] = hostile[target].replace(b"136141", b"136140", 1)
            with self.assertRaisesRegex(port.PortReleaseError, "preimage differs"):
                port._transform_payloads(hostile, modes)


if __name__ == "__main__":
    unittest.main()
