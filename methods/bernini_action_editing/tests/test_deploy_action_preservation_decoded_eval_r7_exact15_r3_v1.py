from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import deploy_action_preservation_decoded_eval_r7_exact15_r3_v1 as deploy


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def run_program(source: bytes) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-"],
        input=source, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=False,
    )


def helper_source(
    *, bundle: Path, work: Path, marker: Path,
    sentinel: Path | None = None, replace_work: bool = False,
) -> bytes:
    paths = {
        "BUNDLE_ROOT": str(bundle),
        "WORK_ROOT": str(work),
        "DEPLOYMENT_REQUEST_PATH": str(work / "deployment-request.json"),
        "MATERIALIZED_RELEASE_ROOT": str(work / "materialized-release"),
        "CONTROLLER_AUTHORITY_PATH": str(work / "controller-authority.json"),
        "DEPLOYMENT_RECEIPT_PATH": str(work / "deployment-receipt.json"),
        "SOURCE_SPEC_PATH": str(work / "source-runtime-spec.json"),
        "SOURCE_SPEC_AUTHORITY_PATH": str(work / "source-spec-authority.json"),
    }
    sentinel_block = ""
    if sentinel is not None:
        sentinel_source = (
            "from pathlib import Path\n"
            f"Path({str(sentinel)!r}).write_text('SENTINEL_EXECUTED')\n"
        )
        sentinel_block = f"""
    os.chmod(BUNDLE_ROOT, 0o755)
    os.unlink(__file__)
    with open(__file__, "wb") as stream:
        stream.write({sentinel_source.encode('utf-8')!r})
    os.chmod(__file__, 0o444)
    os.chmod(BUNDLE_ROOT, 0o555)
"""
    replace_block = ""
    if replace_work:
        replace_block = f"""
    displaced = Path(str(WORK_ROOT) + ".displaced")
    os.rename(WORK_ROOT, displaced)
    os.mkdir(WORK_ROOT, 0o700)
    replacement = canonical(request) + b"\\n"
    fd = os.open(DEPLOYMENT_REQUEST_PATH, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o444)
    try:
        os.write(fd, replacement)
        os.fchmod(fd, 0o444)
        os.fsync(fd)
    finally:
        os.close(fd)
"""
    return (f'''from pathlib import Path
import hashlib,json,os,stat
BUNDLE_ROOT=Path({paths["BUNDLE_ROOT"]!r})
WORK_ROOT=Path({paths["WORK_ROOT"]!r})
DEPLOYMENT_REQUEST_PATH=Path({paths["DEPLOYMENT_REQUEST_PATH"]!r})
MATERIALIZED_RELEASE_ROOT=Path({paths["MATERIALIZED_RELEASE_ROOT"]!r})
CONTROLLER_AUTHORITY_PATH=Path({paths["CONTROLLER_AUTHORITY_PATH"]!r})
DEPLOYMENT_RECEIPT_PATH=Path({paths["DEPLOYMENT_RECEIPT_PATH"]!r})
SOURCE_SPEC_PATH=Path({paths["SOURCE_SPEC_PATH"]!r})
SOURCE_SPEC_AUTHORITY_PATH=Path({paths["SOURCE_SPEC_AUTHORITY_PATH"]!r})
def canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def identity(value):return {{"device":value.st_dev,"inode":value.st_ino,"uid":value.st_uid,"gid":value.st_gid,"mode":value.st_mode,"nlink":value.st_nlink,"rdev":value.st_rdev,"size":value.st_size,"blocks":getattr(value,"st_blocks",0),"mtime_ns":value.st_mtime_ns,"ctime_ns":value.st_ctime_ns}}
def view(path,entries,value):return {{"path":str(path),"mode":0o700,"entries":entries,"identity":identity(value),"parent_identity":identity(path.parent.stat()),"retained_parent_fd":True,"retained_root_fd":True}}
def main(argv):
    if argv != ["phase-a"]: return 91
    Path({str(marker)!r}).write_text("ORIGINAL_EXECUTED")
{sentinel_block.rstrip()}
    os.mkdir(WORK_ROOT, 0o700)
    created=WORK_ROOT.stat()
    initial=view(WORK_ROOT,[],created)
    authority={{"schema_version":"bernini-action-preservation-decoded-eval-work-root-authority-v1","path":str(WORK_ROOT),"parent_path":str(WORK_ROOT.parent),"creation_identity":identity(created),"immutable_identity":{{key:identity(created)[key] for key in ("device","inode","uid","gid","mode","rdev")}},"parent_immutable_identity":{{key:identity(WORK_ROOT.parent.stat())[key] for key in ("device","inode","uid","gid","mode","rdev")}},"initial_entries":[],"retained_parent_fd_through_request_publication":True,"retained_root_fd_through_request_publication":True}}
    authority["authority_digest"]=digest(authority)
    request={{"work_root_authority":authority}}
    raw=canonical(request)+b"\\n"
    fd=os.open(DEPLOYMENT_REQUEST_PATH,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o444)
    try:
        os.write(fd,raw);os.fchmod(fd,0o444);os.fsync(fd)
    finally:os.close(fd)
    after=WORK_ROOT.stat()
    result={{"status":"R7_EXACT15_R3_PHASE_A_REQUEST_PREPARED_NOT_EXECUTED","deployment_request":{{"path":str(DEPLOYMENT_REQUEST_PATH),"sha256":hashlib.sha256(raw).hexdigest(),"size":len(raw),"mode":0o444}},"request_digest":hashlib.sha256(raw).hexdigest(),"work_root_authority":authority,"work_root_initial":initial,"work_root_after_request":view(WORK_ROOT,["deployment-request.json"],after),"phase_a_expected_final_entries":["controller-authority.json","deployment-receipt.json","deployment-request.json","materialized-release"],"controller_argv":["/usr/bin/python3.10","-I","-S","-B","-c","captured-controller"],"controller_bootstrap_source_sha256":"0"*64,"remote_process_executed":False,"gpu_used":False}}
{replace_block.rstrip()}
    print(canonical(result).decode("utf-8"))
    return 0
''').encode("utf-8")


class Fixture:
    def __init__(
        self, root: Path, *, replace_helper: bool = False,
        replace_work: bool = False,
    ) -> None:
        self.root = root
        self.sources = root / "sources"
        self.sources.mkdir()
        self.bundle = root / "bundle"
        self.work = root / "work"
        self.authorization = root / "phase-a-authorization.json"
        self.marker = root / "original-marker"
        self.sentinel = root / "sentinel-marker"
        helper = helper_source(
            bundle=self.bundle, work=self.work, marker=self.marker,
            sentinel=self.sentinel if replace_helper else None,
            replace_work=replace_work,
        )
        rows: dict[str, dict] = {}
        for index, (role, remote) in enumerate(
            sorted(deploy.EXPECTED_REMOTE_PATHS.items())
        ):
            path = self.sources / f"{index:02d}-{role}.bin"
            raw = helper if role == "prepare_helper" else (
                f"fixture:{role}:{remote}\n".encode("utf-8")
            )
            path.write_bytes(raw)
            path.chmod(0o444)
            rows[role] = {
                "source_path": str(path), "source_mode": 0o444,
                "remote_path": remote, "remote_mode": 0o444,
                "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw),
            }
        self.plan = deploy.build_plan(
            bundle_root=self.bundle, work_root=self.work,
            phase_a_authorization_receipt_path=self.authorization,
            expected_uid=os.getuid(), expected_gid=os.getgid(),
            phase_a_status=(
                "R7_EXACT15_R3_PHASE_A_REQUEST_PREPARED_NOT_EXECUTED"
            ),
            files=rows,
            bundle_root_final_nlink=(
                2 + len(deploy.EXPECTED_TOP_LEVEL)
                if sys.platform == "darwin" else 3
            ),
            release_directory_final_nlink=(
                2 + len(deploy.EXPECTED_RELEASE_ENTRIES)
                if sys.platform == "darwin" else 2
            ),
            phase_a_work_final_nlink=(3 if sys.platform == "darwin" else 2),
        )
        self.authority = self.plan["authority_digest"]

    def upload_program(self) -> bytes:
        payloads = deploy.capture_local_payloads(
            self.plan, expected_authority_digest=self.authority
        )
        return deploy.render_upload_program(
            self.plan, expected_authority_digest=self.authority,
            payloads=payloads,
        )

    def author_program(self) -> bytes:
        helper = self.plan["files"]["prepare_helper"]
        return deploy.render_phase_a_author_program(
            self.plan, expected_authority_digest=self.authority,
            helper_sha256=helper["sha256"], helper_size=helper["size"],
            helper_mode=helper["remote_mode"],
        )


class Exact8DeploymentTests(unittest.TestCase):
    def test_real_rendered_upload_and_captured_helper_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary).resolve())
            upload = run_program(fixture.upload_program())
            self.assertEqual(upload.returncode, 0, upload.stderr.decode())
            upload_receipt = json.loads(upload.stdout)
            self.assertEqual(upload_receipt["exact8_file_count"], 8)
            self.assertEqual(stat.S_IMODE(fixture.bundle.stat().st_mode), 0o555)
            self.assertEqual(
                sorted(path.name for path in fixture.bundle.iterdir()),
                sorted(deploy.EXPECTED_TOP_LEVEL),
            )
            author = run_program(fixture.author_program())
            self.assertEqual(author.returncode, 0, author.stderr.decode())
            self.assertEqual(
                json.loads(author.stdout)["status"],
                "PHASE_A_AUTHORIZED_NOT_EXECUTED",
            )
            self.assertEqual(fixture.marker.read_text(), "ORIGINAL_EXECUTED")
            self.assertFalse(fixture.sentinel.exists())
            receipt = json.loads(fixture.authorization.read_bytes())
            self.assertTrue(
                receipt["helper_executed_from_same_fd_captured_bytes"]
            )
            self.assertFalse(receipt["controller_invocation"]["executed"])
            self.assertFalse(receipt["gpu_used"])

    def test_partial_collision_and_double_upload_fail_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary).resolve())
            fixture.bundle.mkdir(mode=0o700)
            sentinel = fixture.bundle / "partial"
            sentinel.write_bytes(b"DO-NOT-OVERWRITE")
            failed = run_program(fixture.upload_program())
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual(sentinel.read_bytes(), b"DO-NOT-OVERWRITE")
            self.assertEqual(list(fixture.bundle.iterdir()), [sentinel])

        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary).resolve())
            first = run_program(fixture.upload_program())
            self.assertEqual(first.returncode, 0, first.stderr.decode())
            snapshot = {
                str(path.relative_to(fixture.bundle)): path.read_bytes()
                for path in fixture.bundle.rglob("*") if path.is_file()
            }
            second = run_program(fixture.upload_program())
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(
                snapshot,
                {
                    str(path.relative_to(fixture.bundle)): path.read_bytes()
                    for path in fixture.bundle.rglob("*") if path.is_file()
                },
            )

    def test_work_root_extra_and_double_author_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary).resolve())
            self.assertEqual(run_program(fixture.upload_program()).returncode, 0)
            fixture.work.mkdir(mode=0o700)
            extra = fixture.work / "extra"
            extra.write_bytes(b"extra")
            failed = run_program(fixture.author_program())
            self.assertNotEqual(failed.returncode, 0)
            self.assertFalse(fixture.marker.exists())
            self.assertEqual(extra.read_bytes(), b"extra")
            self.assertFalse(fixture.authorization.exists())

        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary).resolve())
            self.assertEqual(run_program(fixture.upload_program()).returncode, 0)
            first = run_program(fixture.author_program())
            self.assertEqual(first.returncode, 0, first.stderr.decode())
            before = fixture.authorization.read_bytes()
            second = run_program(fixture.author_program())
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(fixture.authorization.read_bytes(), before)

    def test_helper_path_replacement_runs_captured_original_not_sentinel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary).resolve(), replace_helper=True)
            self.assertEqual(run_program(fixture.upload_program()).returncode, 0)
            author = run_program(fixture.author_program())
            self.assertNotEqual(author.returncode, 0)
            self.assertEqual(fixture.marker.read_text(), "ORIGINAL_EXECUTED")
            self.assertFalse(fixture.sentinel.exists())
            self.assertFalse(fixture.authorization.exists())

    def test_work_root_rename_replacement_barrier_rejects_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary).resolve(), replace_work=True)
            self.assertEqual(run_program(fixture.upload_program()).returncode, 0)
            author = run_program(fixture.author_program())
            self.assertNotEqual(author.returncode, 0)
            self.assertTrue(Path(str(fixture.work) + ".displaced").is_dir())
            self.assertFalse(fixture.authorization.exists())

    def test_fully_resigned_payload_cannot_replace_external_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary).resolve())
            original_authority = fixture.authority
            altered = json.loads(json.dumps(fixture.plan))
            role = "manifest"
            path = Path(altered["files"][role]["source_path"])
            path.chmod(0o644)
            path.write_bytes(b"fully resigned attacker payload")
            path.chmod(0o444)
            raw = path.read_bytes()
            altered["files"][role]["sha256"] = hashlib.sha256(raw).hexdigest()
            altered["files"][role]["size"] = len(raw)
            unsigned = dict(altered)
            unsigned.pop("authority_digest")
            altered["authority_digest"] = deploy.object_sha256(unsigned)
            with self.assertRaisesRegex(
                deploy.Exact8DeploymentError, "external authority digest"
            ):
                deploy.capture_local_payloads(
                    altered, expected_authority_digest=original_authority
                )

    def test_symlink_and_hardlink_sources_fail_same_fd_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary).resolve())
            role = "manifest"
            original = Path(fixture.plan["files"][role]["source_path"])
            hardlink = original.with_name("hardlink")
            os.link(original, hardlink)
            with self.assertRaisesRegex(
                deploy.Exact8DeploymentError, "physical identity"
            ):
                deploy.capture_local_payloads(
                    fixture.plan,
                    expected_authority_digest=fixture.authority,
                )

        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary).resolve())
            role = "manifest"
            original = Path(fixture.plan["files"][role]["source_path"])
            target = original.with_name("target")
            original.rename(target)
            original.symlink_to(target)
            with self.assertRaises((deploy.Exact8DeploymentError, OSError)):
                deploy.capture_local_payloads(
                    fixture.plan,
                    expected_authority_digest=fixture.authority,
                )

    def test_remote_extra_symlink_and_hardlink_fail_before_helper(self) -> None:
        def uploaded(root: Path) -> Fixture:
            fixture = Fixture(root)
            result = run_program(fixture.upload_program())
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            return fixture

        with tempfile.TemporaryDirectory() as temporary:
            fixture = uploaded(Path(temporary).resolve())
            fixture.bundle.chmod(0o755)
            (fixture.bundle / "EXTRA").write_bytes(b"extra")
            fixture.bundle.chmod(0o555)
            self.assertNotEqual(run_program(fixture.author_program()).returncode, 0)
            self.assertFalse(fixture.marker.exists())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            fixture = uploaded(root)
            helper = fixture.bundle / deploy.EXPECTED_REMOTE_PATHS["prepare_helper"]
            held = root / "held-helper"
            fixture.bundle.chmod(0o755)
            helper.rename(held)
            helper.symlink_to(held)
            fixture.bundle.chmod(0o555)
            self.assertNotEqual(run_program(fixture.author_program()).returncode, 0)
            self.assertFalse(fixture.marker.exists())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            fixture = uploaded(root)
            helper = fixture.bundle / deploy.EXPECTED_REMOTE_PATHS["prepare_helper"]
            second_link = root / "second-helper-link"
            os.link(helper, second_link)
            self.assertEqual(helper.stat().st_nlink, 2)
            self.assertNotEqual(run_program(fixture.author_program()).returncode, 0)
            self.assertFalse(fixture.marker.exists())


if __name__ == "__main__":
    unittest.main()
