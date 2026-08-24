import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = METHOD_ROOT / "tools" / "build_self_generated_action_preservation_v2_release.py"
VERIFIED_RUNTIME_PATH = METHOD_ROOT / "action_preservation_verified_release_v1.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("apv2_release_builder", BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_verified_runtime():
    spec = importlib.util.spec_from_file_location(
        "apv2_release_verified_runtime_contract", VERIFIED_RUNTIME_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_builder()
verified_runtime = load_verified_runtime()


def replace_release_bytes(path: Path, raw: bytes, mode: int) -> None:
    path.chmod(0o600)
    path.write_bytes(raw)
    path.chmod(mode)


def resign(value: dict, digest_field: str) -> None:
    unsigned = dict(value)
    unsigned.pop(digest_field, None)
    value[digest_field] = builder.sha256(builder.canonical(unsigned))


def fully_resign_manifest_and_envelope(release: Path, mutate) -> None:
    manifest_path = release / "source.manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    mutate(manifest)
    resign(manifest, "manifest_digest")
    manifest_raw = builder.canonical(manifest) + b"\n"
    replace_release_bytes(manifest_path, manifest_raw, 0o444)

    envelope_path = release / "deployment-envelope.json"
    envelope = json.loads(envelope_path.read_bytes())
    manifest_ref = envelope["source_manifest"]
    manifest_ref["sha256"] = builder.sha256(manifest_raw)
    manifest_ref["manifest_digest"] = manifest["manifest_digest"]
    manifest_ref["content_revision"] = manifest["content_revision"]
    manifest_ref["file_count"] = manifest["file_count"]
    resign(envelope, "envelope_digest")
    replace_release_bytes(
        envelope_path, builder.canonical(envelope) + b"\n", 0o444
    )


def fully_resign_envelope(release: Path, mutate) -> None:
    envelope_path = release / "deployment-envelope.json"
    envelope = json.loads(envelope_path.read_bytes())
    mutate(envelope)
    resign(envelope, "envelope_digest")
    replace_release_bytes(
        envelope_path, builder.canonical(envelope) + b"\n", 0o444
    )


class ReleaseTests(unittest.TestCase):
    def test_builder_pins_every_runtime_member(self):
        self.assertEqual(
            set(builder.FILES_AND_MODES),
            set(builder.EXPECTED_SHA256),
        )
        self.assertEqual(set(builder.FILES_AND_MODES), set(builder.EXPECTED_SIZE))
        self.assertEqual(len(builder.FILES_AND_MODES), 9)
        for relative, expected in builder.EXPECTED_SHA256.items():
            raw = (METHOD_ROOT / relative).read_bytes()
            self.assertEqual(
                hashlib.sha256(raw).hexdigest(),
                expected,
            )
            self.assertEqual(len(raw), builder.EXPECTED_SIZE[relative])
        self.assertEqual(
            hashlib.sha256(
                (METHOD_ROOT / "scripts" / builder.DETACHED_CONTROLLER).read_bytes()
            ).hexdigest(),
            builder.DETACHED_CONTROLLER_SHA256,
        )

    def test_fixed_ustar_header_matches_verified_runtime_exactly(self):
        row = builder.expected_file_rows()[0]
        name = f"{builder.MEMBER_ROOT}/{row['path']}"
        builder_header = builder.fixed_ustar_header(
            name, size=row["size"], mode=row["mode"]
        )
        runtime_header = verified_runtime.fixed_ustar_header(
            name, size=row["size"], mode=row["mode"]
        )
        self.assertEqual(builder.ARCHIVE_FORMAT, verified_runtime.ARCHIVE_FORMAT)
        self.assertEqual(builder_header, runtime_header)
        self.assertEqual(len(builder_header), 512)
        self.assertEqual(builder_header[329:345], b"0000000\0" * 2)
        checksum_header = bytearray(builder_header)
        declared_checksum = int(checksum_header[148:154], 8)
        checksum_header[148:156] = b" " * 8
        self.assertEqual(declared_checksum, sum(checksum_header))
        self.assertNotIn(".tobuf(", BUILDER_PATH.read_text(encoding="utf-8"))
        self.assertNotIn(".tobuf(", VERIFIED_RUNTIME_PATH.read_text(encoding="utf-8"))

    def test_python310_null_device_field_variant_is_rejected_for_exact9(self):
        with tempfile.TemporaryDirectory() as directory:
            release = Path(directory) / "release"
            builder.build(release)
            manifest = json.loads((release / "source.manifest.json").read_bytes())
            hostile = bytearray((release / "source.tar").read_bytes())
            offset = 0
            for row in manifest["files"]:
                header = bytearray(
                    hostile[offset : offset + builder.FIXED_USTAR_BLOCK_SIZE]
                )
                header[329:345] = b"\0" * 16
                header[148:156] = b" " * 8
                checksum = sum(header)
                header[148:156] = f"{checksum:06o}\0 ".encode("ascii")
                hostile[offset : offset + builder.FIXED_USTAR_BLOCK_SIZE] = header
                blocks = (
                    row["size"] + builder.FIXED_USTAR_BLOCK_SIZE - 1
                ) // builder.FIXED_USTAR_BLOCK_SIZE
                offset += builder.FIXED_USTAR_BLOCK_SIZE * (1 + blocks)

            with tarfile.open(fileobj=io.BytesIO(hostile), mode="r:") as archive:
                self.assertEqual(len(archive.getmembers()), 9)
            with self.assertRaisesRegex(
                verified_runtime.ActionPreservationVerifiedReleaseError,
                "canonical USTAR",
            ):
                verified_runtime.verify_archive_snapshot(bytes(hostile), manifest)

    def test_two_fresh_builds_are_byte_identical_and_auditable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            first_result = builder.build(first)
            second_result = builder.build(second)
            self.assertTrue(first_result["static_audit_go"])
            self.assertTrue(second_result["static_audit_go"])
            self.assertEqual(
                {path.name: path.read_bytes() for path in first.iterdir()},
                {path.name: path.read_bytes() for path in second.iterdir()},
            )
            self.assertTrue(builder.audit(first, against_workspace=True)["static_audit_go"])

    def test_manifest_forbids_loss_promotion_and_requires_blind_review(self):
        with tempfile.TemporaryDirectory() as directory:
            release = Path(directory) / "release"
            builder.build(release)
            manifest = json.loads((release / "source.manifest.json").read_bytes())
            authority = manifest["authority"]
            self.assertTrue(authority["training_loss_promotion_forbidden"])
            self.assertFalse(authority["decoded_identity_background_camera_claim_authorized"])
            self.assertTrue(authority["blind_full_video_review_required_for_promotion"])
            self.assertFalse(authority["scientific_claim_authorized"])
            self.assertEqual(authority["checkpoint_steps"], [0, 5, 10, 20])
            self.assertEqual(authority["arms"], list(builder.ARMS))
            isolated = authority["isolated_frozen_runtime"]
            self.assertEqual(isolated["python_flags"], ["-I", "-S", "-B", "-c"])
            self.assertEqual(
                isolated["site_packages"], builder.FROZEN_SITE_PACKAGES
            )
            self.assertTrue(
                isolated["site_packages_added_only_after_full_release_capture"]
            )
            self.assertTrue(isolated["automatic_site_initialization_disabled"])
            self.assertEqual(
                isolated["torchrun_launcher"],
                {
                    "path": builder.TORCHRUN_PATH,
                    "sha256": builder.TORCHRUN_SHA256,
                    "size": builder.TORCHRUN_SIZE,
                    "uid": 2012,
                    "gid": 2000,
                    "mode": "0644",
                    "link_count": 1,
                },
            )
            self.assertTrue(
                isolated["torchrun_same_fd_double_read_full_identity_required"]
            )
            self.assertTrue(isolated["torchrun_executed_from_captured_source"])
            self.assertTrue(isolated["rank_python_is_root_verified_bootstrap"])
            self.assertEqual(
                manifest["allowed_entrypoints"],
                [
                    "scripts/auh_run_self_generated_action_preservation_v2.sh",
                    "audit_self_generated_action_preservation_v2.py",
                    "action_preservation_verified_release_v1.py",
                    "action_preservation_completion_publisher_v1.py",
                    builder.DETACHED_CONTROLLER,
                ],
            )
            self.assertEqual(
                manifest["component_sha256"]["verified_release"],
                builder.EXPECTED_SHA256[
                    "action_preservation_verified_release_v1.py"
                ],
            )
            self.assertEqual(
                manifest["component_sha256"]["completion_publisher"],
                builder.EXPECTED_SHA256[
                    "action_preservation_completion_publisher_v1.py"
                ],
            )

    def test_fully_resigned_scientific_claim_overclaim_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            release = Path(directory) / "release"
            builder.build(release)
            fully_resign_manifest_and_envelope(
                release,
                lambda manifest: manifest["authority"].__setitem__(
                    "scientific_claim_authorized", True
                ),
            )
            with self.assertRaises(builder.ReleaseError):
                builder.audit(release, against_workspace=False)

    def test_fully_resigned_manifest_field_and_value_hostiles_are_rejected(self):
        cases = (
            "top-extra",
            "top-missing",
            "authority-extra",
            "authority-missing",
            "authority-semantic-overclaim",
            "component-extra",
            "component-missing",
            "component-value",
            "verified-component-value",
            "publisher-component-missing",
            "file-extra",
            "file-missing",
            "file-mode-value",
            "file-sha-value",
        )
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                release = Path(directory) / "release"
                builder.build(release)

                def mutate(manifest):
                    if case == "top-extra":
                        manifest["future_authority"] = False
                    elif case == "top-missing":
                        manifest.pop("exact_member_closure")
                    elif case == "authority-extra":
                        manifest["authority"]["optimizer_authorized"] = False
                    elif case == "authority-missing":
                        manifest["authority"].pop("training_loss_promotion_forbidden")
                    elif case == "authority-semantic-overclaim":
                        manifest["authority"][
                            "decoded_identity_background_camera_claim_authorized"
                        ] = True
                    elif case == "component-extra":
                        manifest["component_sha256"]["optimizer"] = "0" * 64
                    elif case == "component-missing":
                        manifest["component_sha256"].pop("auditor")
                    elif case == "component-value":
                        manifest["component_sha256"]["trainer"] = "0" * 64
                    elif case == "verified-component-value":
                        manifest["component_sha256"]["verified_release"] = "0" * 64
                    elif case == "publisher-component-missing":
                        manifest["component_sha256"].pop("completion_publisher")
                    elif case == "file-extra":
                        manifest["files"][0]["future"] = False
                    elif case == "file-missing":
                        manifest["files"][0].pop("size")
                    elif case == "file-mode-value":
                        manifest["files"][0]["mode"] = 0o555
                    elif case == "file-sha-value":
                        manifest["files"][0]["sha256"] = "0" * 64
                    else:  # pragma: no cover - case closure above
                        self.fail(case)
                    if case.startswith("file-"):
                        manifest["content_revision"] = builder.content_revision(
                            manifest["files"]
                        )

                fully_resign_manifest_and_envelope(release, mutate)
                with self.assertRaises(builder.ReleaseError):
                    builder.audit(release, against_workspace=False)

    def test_fully_resigned_envelope_field_and_value_hostiles_are_rejected(self):
        cases = (
            "top-extra",
            "top-missing",
            "archive-extra",
            "archive-missing",
            "archive-value",
            "manifest-extra",
            "manifest-missing",
            "manifest-value",
            "controller-extra",
            "controller-missing",
            "controller-value",
            "entry-order",
            "semantic-overclaim",
        )
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                release = Path(directory) / "release"
                builder.build(release)

                def mutate(envelope):
                    if case == "top-extra":
                        envelope["future_deployment_authority"] = False
                    elif case == "top-missing":
                        envelope.pop("fresh_experiment_root_required")
                    elif case == "archive-extra":
                        envelope["source_archive"]["future"] = False
                    elif case == "archive-missing":
                        envelope["source_archive"].pop("mode")
                    elif case == "archive-value":
                        envelope["source_archive"]["basename"] = "renamed.tar"
                    elif case == "manifest-extra":
                        envelope["source_manifest"]["future"] = False
                    elif case == "manifest-missing":
                        envelope["source_manifest"].pop("content_revision")
                    elif case == "manifest-value":
                        envelope["source_manifest"]["mode"] = 0o555
                    elif case == "controller-extra":
                        envelope["detached_controller"]["future"] = False
                    elif case == "controller-missing":
                        envelope["detached_controller"].pop("sha256")
                    elif case == "controller-value":
                        envelope["detached_controller"]["sha256"] = "0" * 64
                    elif case == "entry-order":
                        envelope["remote_release_exact_entries"].reverse()
                    elif case == "semantic-overclaim":
                        envelope["automatic_scientific_promotion_authorized"] = True
                    else:  # pragma: no cover - case closure above
                        self.fail(case)

                fully_resign_envelope(release, mutate)
                with self.assertRaises(builder.ReleaseError):
                    builder.audit(release, against_workspace=False)

    def test_fully_resigned_nonzero_tar_payload_after_eof_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            release = Path(directory) / "release"
            builder.build(release)
            archive_path = release / "source.tar"
            extended = archive_path.read_bytes() + (b"hostile-after-eof" * 32)
            self.assertNotEqual(len(extended) % 512, 0)
            extended += b"X" * (512 - len(extended) % 512)
            replace_release_bytes(archive_path, extended, 0o444)

            fully_resign_envelope(
                release,
                lambda envelope: envelope["source_archive"].__setitem__(
                    "sha256", builder.sha256(extended)
                ),
            )
            with self.assertRaises(builder.ReleaseError):
                builder.audit(release, against_workspace=False)

    def test_archive_is_exact_regular_ustar_runtime_closure(self):
        with tempfile.TemporaryDirectory() as directory:
            release = Path(directory) / "release"
            builder.build(release)
            with tarfile.open(release / "source.tar", mode="r:") as archive:
                members = archive.getmembers()
            self.assertEqual(
                [member.name for member in members],
                [f"{builder.MEMBER_ROOT}/{name}" for name in sorted(builder.FILES_AND_MODES)],
            )
            for member in members:
                self.assertTrue(member.isfile())
                self.assertFalse(member.linkname)
                self.assertEqual((member.uid, member.gid, member.mtime), (0, 0, 0))

    def test_shell_entrypoints_parse_and_have_no_parent_destructive_commands(self):
        scripts = [
            METHOD_ROOT / "scripts" / builder.DETACHED_CONTROLLER,
            METHOD_ROOT / "scripts" / "auh_run_self_generated_action_preservation_v2.sh",
        ]
        for script in scripts:
            completed = subprocess.run(
                ["bash", "-n", str(script)], capture_output=True, text=True, check=False
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
        controller = scripts[0].read_text()
        for forbidden in ("scancel", "scontrol release", "scontrol requeue"):
            self.assertNotIn(forbidden, controller)
        self.assertIn("no retry was attempted", controller)
        self.assertIn("sealed training final replay failed", controller)
        self.assertIn(
            "--target action_preservation_completion_publisher_v1.py", controller
        )
        self.assertIn(
            "readonly ssh_sha="
            "3a9c5d143150f0b2816ab1a5a7c58a9f970280b061f617abee54d2834a498b53",
            controller,
        )
        self.assertNotIn(
            "3a9c5d143150f0b2816ab1a5a7c58a9f970280b061f617abee54d2834a498b53a",
            controller,
        )

    def test_node_runner_is_exact_v2_twenty_step_contract(self):
        runner = (
            METHOD_ROOT / "scripts" / "auh_run_self_generated_action_preservation_v2.sh"
        ).read_text()
        self.assertIn("--objective-family preservation_v2", runner)
        self.assertIn("--slots 5 --limit-cells 0 --max-steps 20", runner)
        self.assertIn("v2 canary", runner)
        self.assertNotIn("20260817", runner)
        self.assertIn("20260818", runner)
        self.assertNotIn("-m torch.distributed.run", runner)
        self.assertIn('"-I","-S","-B","-c",torchrun_bootstrap', runner)

    def test_controller_maps_exact_two_sp4_islands_per_holder(self):
        controller = (
            METHOD_ROOT / "scripts" / builder.DETACHED_CONTROLLER
        ).read_text()
        self.assertIn("arm_jobs=(136719 136719 136141 136141 136309 136309 136140 136140)", controller)
        self.assertIn(
            "arm_groups=(0,1,2,3 4,5,6,7 0,1,2,3 4,5,6,7 0,1,2,3 4,5,6,7 0,1,2,3 4,5,6,7)",
            controller,
        )
        self.assertIn("holder_preflight_after_owned_step", controller)
        self.assertIn("ACTION_PRESERVATION_EXPECTED_CACHE_SHA256", controller)


if __name__ == "__main__":
    unittest.main()
