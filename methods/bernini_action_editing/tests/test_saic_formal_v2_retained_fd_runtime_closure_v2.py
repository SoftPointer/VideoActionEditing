"""Static regression coverage for the r10 formal-v2 runtime dependency closure."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = METHOD_ROOT / "scripts/auh_canary_saic_formal_v2_retained_fd_world8_payload_v1.sh"
WRAPPER = METHOD_ROOT / "scripts/auh_canary_saic_formal_v2_retained_fd_world8_v1.sbatch"
SUBMITTER = METHOD_ROOT / "tools/submit_saic_formal_v2_retained_fd_world8_canary_v1.py"
POSTFLIGHT = METHOD_ROOT / "tools/postflight_saic_formal_v2_retained_fd_world8_canary_v1.py"
MATERIALIZER = (
    METHOD_ROOT
    / "tools/materialize_saic_formal_v2_retained_fd_world8_canary_release_v2.py"
)
PROBE_VALIDATOR = METHOD_ROOT / "probe_admission_binding_v1.py"
ARCHIVE_SHA = "3f6a713c762751b06723448b22e627ec6571eae502d7311811005db91812ee7b"
ARCHIVE_MANIFEST_SHA = "1f3c8af23f5b4d416cea04476900c5d479ad3000338746e11f0e655b995b0fcc"
RUNTIME_ORIGIN_SHA = "2e9360581b21b56e6998e1e5db8df98e4cc66acf95fbb7819baffd1161eb98ba"
STEM = "saic-formal-v2-retained-fd-world8-canary-96335bf5-fb3f1ac4-r10"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RetainedFDRuntimeClosureV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.submitter = load(SUBMITTER, "fd_canary_r10_submitter")
        cls.postflight = load(POSTFLIGHT, "fd_canary_r10_postflight")
        cls.materializer = load(MATERIALIZER, "fd_canary_r10_materializer")

    def test_hash_and_stem_pins_are_transitively_closed(self) -> None:
        payload_sha = sha(PAYLOAD)
        wrapper_sha = sha(WRAPPER)
        postflight_sha = sha(POSTFLIGHT)
        self.assertEqual(payload_sha, self.submitter.EXPECTED_PAYLOAD_SHA256)
        self.assertEqual(payload_sha, self.postflight.EXPECTED_PAYLOAD_SHA256)
        self.assertEqual(wrapper_sha, self.submitter.EXPECTED_WRAPPER_SHA256)
        self.assertEqual(wrapper_sha, self.postflight.EXPECTED_WRAPPER_SHA256)
        self.assertEqual(postflight_sha, self.submitter.EXPECTED_POSTFLIGHT_SHA256)
        self.assertEqual(self.submitter.RELEASE.name, STEM)
        self.assertEqual(self.postflight.STEM, STEM)
        self.assertEqual(self.materializer.STEM, STEM)
        self.assertEqual(self.materializer.EXPECTED["payload"][1], payload_sha)
        self.assertEqual(self.materializer.EXPECTED["wrapper"][1], wrapper_sha)
        self.assertEqual(self.materializer.EXPECTED["postflight"][1], postflight_sha)

    def test_source_archive_is_an_exact_release_input(self) -> None:
        self.assertEqual(self.submitter.EXPECTED_SOURCE_ARCHIVE_SHA256, ARCHIVE_SHA)
        self.assertEqual(self.postflight.EXPECTED_SOURCE_ARCHIVE_SHA256, ARCHIVE_SHA)
        self.assertEqual(self.materializer.EXPECTED["source_archive"][1], ARCHIVE_SHA)
        self.assertEqual(
            self.submitter.EXPECTED_SOURCE_ARCHIVE.name,
            "videoedit-saic-20c2193-methods.tar",
        )
        for source in (PAYLOAD, WRAPPER, SUBMITTER, POSTFLIGHT, MATERIALIZER):
            text = source.read_text(encoding="utf-8")
            self.assertIn("source_archive", text, source.name)
            self.assertIn(ARCHIVE_SHA, text, source.name)

    def test_materializer_validator_import_cannot_pollute_release_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / PROBE_VALIDATOR.name
            shutil.copy2(PROBE_VALIDATOR, copied)
            self.materializer.load_validator(copied)
            self.assertFalse((Path(temporary) / "__pycache__").exists())

    def test_payload_extracts_before_runtime_help_and_removes_scratch(self) -> None:
        source = PAYLOAD.read_text(encoding="utf-8")
        extraction = source.index("with tarfile.open")
        runtime_import = source.index('runtime_import="${source_tree}')
        torchrun = source.index('worker --runtime "${runtime_import}"')
        cleanup = source.index("shutil.rmtree(scratch)")
        publication = source.index('operational-evidence.json", module.seal(core)')
        self.assertLess(extraction, runtime_import)
        self.assertLess(runtime_import, torchrun)
        self.assertLess(cleanup, publication)
        self.assertIn('bundle.pax_headers.get("comment")', source)
        self.assertIn("member.issym()", source)
        self.assertIn("member.islnk()", source)
        self.assertIn("os.O_NOFOLLOW", source)

    def test_archive_retained_fd_and_runtime_origin_evidence_are_closed(self) -> None:
        payload = PAYLOAD.read_text(encoding="utf-8")
        wrapper = WRAPPER.read_text(encoding="utf-8")
        postflight = POSTFLIGHT.read_text(encoding="utf-8")
        for source in (payload, wrapper, postflight):
            self.assertIn(ARCHIVE_MANIFEST_SHA, source)
            self.assertIn(RUNTIME_ORIGIN_SHA, source)
        self.assertIn("SAIC_FV2_FD_CANARY_SOURCE_ARCHIVE_FD", wrapper)
        self.assertIn("SAIC_FV2_FD_CANARY_STAGE0_SOURCE_ARCHIVE_FD_NUMBER", payload)
        self.assertIn("source_archive_read_from_stage0_retained_fd", payload)
        self.assertIn('== 7 ]] ||', wrapper)
        self.assertIn('== 7 ]] ||', payload)
        self.assertIn('relative.as_posix() != name', payload)
        self.assertIn('name in normalized', payload)
        self.assertIn('member_types.get(ancestor) != "directory"', payload)
        self.assertIn('observed != member_types', payload)
        self.assertIn('"mode": stat.S_IMODE(info.st_mode)', payload)
        self.assertIn('"size": info.st_size', payload)
        self.assertIn('"extracted_tree_manifest_source": "actual_lstat_after_extraction"', payload)
        self.assertNotIn('if extracted_sha != manifest_sha:', payload)
        self.assertIn('env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1', payload)
        canonical_name = (
            'canonical_module_name = "generate_saic_pure_t2v_event_bank_topup_v2"'
        )
        canonical_spec = (
            "spec = importlib.util.spec_from_file_location(canonical_module_name, runtime)"
        )
        registration = "sys.modules[canonical_module_name] = module"
        execution = "spec.loader.exec_module(module)"
        failure_cleanup = "del sys.modules[canonical_module_name]"
        self.assertIn(canonical_name, payload)
        self.assertIn(canonical_spec, payload)
        self.assertIn(registration, payload)
        self.assertIn(failure_cleanup, payload)
        self.assertNotIn('"_sealed_runtime_origin_checker"', payload)
        self.assertLess(payload.index(canonical_name), payload.index(canonical_spec))
        self.assertLess(payload.index(registration), payload.index(execution))
        self.assertLess(payload.index(execution), payload.index("for module_name, imported"))
        self.assertIn('[[ "${status}" == 86 ]]', payload)
        self.assertLess(
            payload.index('[[ "${status}" == 86 ]]'),
            payload.index('admit-collision'),
        )

    def test_shell_entry_points_parse(self) -> None:
        for source in (PAYLOAD, WRAPPER):
            checked = subprocess.run(
                ["bash", "-n", str(source)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(checked.returncode, 0, (source, checked.stderr))


if __name__ == "__main__":
    unittest.main()
