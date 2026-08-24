from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = METHOD_ROOT / "tools" / "build_elal3_c1_decode_release_v1.py"
LAUNCHER_PATH = METHOD_ROOT / "scripts" / "auh_run_elal3_c1_decode_release_v1.sh"
CHECKPOINT_MANIFEST_PATH = (
    METHOD_ROOT / "audits" / "bernini_r13_ff4c5d4_checkpoint.sha256"
)
DECODER_PATH = METHOD_ROOT / "decode_elal3_c1_simulator_oracle_q_v1.py"
SPEC = importlib.util.spec_from_file_location("build_elal3_c1_decode_release_v1", BUILDER_PATH)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def inline_python_snippets() -> list[str]:
    source = LAUNCHER_PATH.read_text(encoding="utf-8")
    marker = "<<'PY'\n"
    snippets: list[str] = []
    offset = 0
    while True:
        begin = source.find(marker, offset)
        if begin < 0:
            return snippets
        begin += len(marker)
        end = source.find("\nPY\n", begin)
        if end < 0:
            raise AssertionError("unterminated inline Python")
        snippets.append(source[begin:end])
        offset = end + 4


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def write_receipt(path: Path, receipt: dict[str, object]) -> None:
    unsigned = dict(receipt)
    unsigned.pop("receipt_digest", None)
    receipt["receipt_digest"] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    path.write_bytes(canonical_json_bytes(receipt) + b"\n")


class ELAL3C1DecodeReleaseTests(unittest.TestCase):
    def test_production_builder_binds_final_decoder_and_checkpoint_manifest(self) -> None:
        self.assertNotEqual(builder.DECODER_SHA256, builder.PENDING_DECODER_SHA256)
        self.assertEqual(
            hashlib.sha256(DECODER_PATH.read_bytes()).hexdigest(),
            builder.DECODER_SHA256,
        )
        archive, manifest = builder.build_payload(
            DECODER_PATH.resolve(), CHECKPOINT_MANIFEST_PATH.resolve()
        )
        self.assertEqual(
            hashlib.sha256(archive).hexdigest(), manifest["archive_sha256"]
        )
        self.assertEqual(
            manifest["checkpoint_content_manifest"]["sha256"],
            builder.CHECKPOINT_MANIFEST_SHA256,
        )

    def test_deterministic_exact_two_member_ustar_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            source = parent / "decoder.py"
            source.write_text("VALUE = 7\n", encoding="ascii")
            pin = hashlib.sha256(source.read_bytes()).hexdigest()
            first = parent / "first"
            second = parent / "second"
            a = builder.publish(
                source, CHECKPOINT_MANIFEST_PATH, first, decoder_sha256=pin
            )
            b = builder.publish(
                source, CHECKPOINT_MANIFEST_PATH, second, decoder_sha256=pin
            )
            self.assertEqual((first / "decode-source.tar").read_bytes(), (second / "decode-source.tar").read_bytes())
            self.assertEqual((first / "decode-manifest.json").read_bytes(), (second / "decode-manifest.json").read_bytes())
            self.assertEqual(a["archive_sha256"], b["archive_sha256"])
            with tarfile.open(first / "decode-source.tar", "r:") as archive:
                members = archive.getmembers()
                self.assertEqual(
                    [row.name for row in members],
                    sorted(
                        [
                            builder.CHECKPOINT_MANIFEST_MEMBER,
                            builder.DECODER_MEMBER,
                        ]
                    ),
                )
                expected_payloads = {
                    builder.CHECKPOINT_MANIFEST_MEMBER: CHECKPOINT_MANIFEST_PATH.read_bytes(),
                    builder.DECODER_MEMBER: source.read_bytes(),
                }
                for row in members:
                    self.assertTrue(row.isreg())
                    self.assertEqual(
                        (row.mode, row.uid, row.gid, row.mtime),
                        (0o444, 0, 0, 0),
                    )
                    self.assertEqual(
                        archive.extractfile(row).read(), expected_payloads[row.name]
                    )
            manifest_raw = (first / "decode-manifest.json").read_bytes()
            manifest = json.loads(manifest_raw)
            unsigned = dict(manifest)
            stored = unsigned.pop("manifest_digest")
            self.assertEqual(stored, hashlib.sha256(builder.canonical_json_bytes(unsigned)).hexdigest())
            self.assertEqual(manifest_raw, builder.canonical_json_bytes(manifest) + b"\n")
            self.assertEqual(
                manifest["files"],
                [
                    {
                        "path": member,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "size": len(payload),
                        "archive_mode": 0o444,
                    }
                    for member, payload in sorted(expected_payloads.items())
                ],
            )
            self.assertEqual(
                manifest["checkpoint_content_manifest"],
                {
                    "member": builder.CHECKPOINT_MANIFEST_MEMBER,
                    "sha256": builder.CHECKPOINT_MANIFEST_SHA256,
                    "size": builder.CHECKPOINT_MANIFEST_SIZE,
                    "row_count": builder.CHECKPOINT_MANIFEST_ROW_COUNT,
                },
            )
            self.assertEqual(manifest["training_release"], builder.TRAINING_RELEASE)
            self.assertEqual(
                manifest["training_artifacts_by_seed"],
                builder.TRAINING_ARTIFACTS_BY_SEED,
            )
            self.assertEqual(manifest["runtime"], builder.RUNTIME)
            self.assertEqual(manifest["authority_bindings"], builder.AUTHORITY_BINDINGS)
            self.assertTrue(all(manifest[key] is False for key in builder.FORBIDDEN_CLAIMS))
            self.assertEqual((first.stat().st_mode & 0o777), 0o555)
            self.assertEqual((first / "decode-source.tar").stat().st_mode & 0o777, 0o444)
            self.assertEqual((first / "decode-manifest.json").stat().st_mode & 0o777, 0o444)

    def test_source_mismatch_and_output_overwrite_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            source = parent / "decoder.py"
            source.write_text("VALUE = 9\n", encoding="ascii")
            pin = hashlib.sha256(source.read_bytes()).hexdigest()
            with self.assertRaisesRegex(builder.ELAL3C1DecodeReleaseError, "source SHA-256 differs"):
                builder.build_payload(
                    source,
                    CHECKPOINT_MANIFEST_PATH,
                    decoder_sha256="0" * 64,
                )
            output = parent / "release"
            builder.publish(
                source, CHECKPOINT_MANIFEST_PATH, output, decoder_sha256=pin
            )
            with self.assertRaisesRegex(builder.ELAL3C1DecodeReleaseError, "fresh absolute"):
                builder.publish(
                    source,
                    CHECKPOINT_MANIFEST_PATH,
                    output,
                    decoder_sha256=pin,
                )

    def test_checkpoint_content_manifest_substitution_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary).resolve() / "decoder.py"
            source.write_text("VALUE = 11\n", encoding="ascii")
            pin = hashlib.sha256(source.read_bytes()).hexdigest()
            with self.assertRaisesRegex(
                builder.ELAL3C1DecodeReleaseError,
                "checkpoint content manifest bytes differ",
            ):
                builder.build_payload(
                    source,
                    source,
                    decoder_sha256=pin,
                )

    def test_cross_seed_holder_exchange_is_rejected_before_signing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary).resolve() / "decoder.py"
            source.write_text("VALUE = 10\n", encoding="ascii")
            pin = hashlib.sha256(source.read_bytes()).hexdigest()
            row17 = builder.TRAINING_ARTIFACTS_BY_SEED["20260817"]
            row18 = builder.TRAINING_ARTIFACTS_BY_SEED["20260818"]
            original17 = (row17["holder_job_id"], row17["holder_node"])
            original18 = (row18["holder_job_id"], row18["holder_node"])
            try:
                row17["holder_job_id"], row17["holder_node"] = original18
                row18["holder_job_id"], row18["holder_node"] = original17
                with self.assertRaisesRegex(
                    builder.ELAL3C1DecodeReleaseError,
                    "training artifact placement differs",
                ):
                    builder.build_payload(
                        source,
                        CHECKPOINT_MANIFEST_PATH,
                        decoder_sha256=pin,
                    )
            finally:
                row17["holder_job_id"], row17["holder_node"] = original17
                row18["holder_job_id"], row18["holder_node"] = original18

    def test_launcher_syntax_and_final_literal_gate(self) -> None:
        syntax = subprocess.run(
            ["bash", "-n", str(LAUNCHER_PATH)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        run = subprocess.run(
            ["bash", str(LAUNCHER_PATH)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(run.returncode, 2)
        self.assertIn("ELAL3_C1_DECODE_SOURCE_ARCHIVE is required", run.stderr)
        launcher = LAUNCHER_PATH.read_text(encoding="utf-8")
        for digest in (
            builder.DECODER_SHA256,
            builder.CHECKPOINT_MANIFEST_SHA256,
            "ee0ab30d9afa17ef5b92b6d0425cbf7c5c0ebaf6cb09e93a6d4165e9021c6119",
            "f850c2470792a9fd0e9c844d1574781d4002abf0f5238690f3842e4f4362b5c2",
        ):
            self.assertIn(digest, launcher)

    def test_launcher_binds_world4_exact3_training_and_backlinks(self) -> None:
        source = LAUNCHER_PATH.read_text(encoding="utf-8")
        for literal in (
            "141620:auh7-1b-gpu-226",
            "141618:auh7-1b-gpu-249",
            "141619:auh7-1b-gpu-257",
            "--nproc-per-node=4",
            "c1-oracle-train-release-r3",
            "elal3_c1_node226_seed20260817_r3/elal3_c1_ten_step_overfit",
            "elal3_c1_node249_seed20260818_r3/elal3_c1_ten_step_overfit",
            "elal3_c1_node257_seed20260819_r3/elal3_c1_ten_step_overfit",
            builder.TRAINING_RELEASE["manifest_sha256"],
            "--decode-release-manifest",
            "--expected-decode-release-manifest-sha256",
            "--decode-launcher",
            "--expected-decode-launcher-sha256",
            "--checkpoint-content-manifest",
            "--expected-checkpoint-content-manifest-sha256",
            builder.CHECKPOINT_MANIFEST_MEMBER,
            builder.CHECKPOINT_MANIFEST_SHA256,
            "rank_${rank}",
            "MIOPEN_USER_DB_PATH",
            "--num-inference-steps 40",
            '--sampling-seed "${sampling_seed}"',
        ):
            self.assertIn(literal, source)
        for seed, artifacts in builder.TRAINING_ARTIFACTS_BY_SEED.items():
            self.assertIn(seed, source)
            for value in artifacts.values():
                self.assertIn(str(value), source)

    def test_all_inline_python_is_syntax_valid(self) -> None:
        snippets = inline_python_snippets()
        self.assertGreaterEqual(len(snippets), 3)
        for index, snippet in enumerate(snippets):
            try:
                ast.parse(snippet)
            except SyntaxError as error:
                self.fail(f"inline Python {index} is invalid: {error}")

    def test_postflight_accepts_exact9_and_rejects_backlink_or_filename_drift(self) -> None:
        postflight = inline_python_snippets()[-1]
        order = [
            "source",
            "gt_target",
            "appearance_anchor",
            "frozen_base",
            "step0_correct_q",
            "trained_correct_q",
            "trained_zero_q",
            "trained_phase_reverse_q",
            "trained_role_swap_q",
        ]
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            root = parent / "decode"
            root.mkdir()
            manifest = parent / "decode-manifest.json"
            launcher = parent / "launcher.sh"
            checkpoint_manifest = parent / "checkpoint-content.sha256"
            checkpoint_root = parent / "checkpoint"
            checkpoint_root.mkdir()
            manifest.write_bytes(b"manifest\n")
            launcher.write_bytes(b"launcher\n")
            checkpoint_manifest.write_bytes(b"x" * builder.CHECKPOINT_MANIFEST_SIZE)
            manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
            launcher_sha = hashlib.sha256(launcher.read_bytes()).hexdigest()
            checkpoint_manifest_sha = hashlib.sha256(
                checkpoint_manifest.read_bytes()
            ).hexdigest()
            media: list[dict[str, object]] = []
            for index, key in enumerate(order):
                relative = f"{index:02d}_{key}.mp4"
                payload = f"media-{index}-{key}\n".encode("ascii")
                (root / relative).write_bytes(payload)
                media.append(
                    {
                        "key": key,
                        "relative_path": relative,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "size": len(payload),
                        "frame_count": 81,
                        "fps": 25.0,
                    }
                )
            html_payload = b"<!doctype html><title>exact9</title>\n"
            (root / "index.html").write_bytes(html_payload)
            receipt: dict[str, object] = {
                "schema_version": "bernini-elal3-c1-simulator-oracle-q-decode-receipt-v3",
                "method": "bernini-elal3-c1-simulator-oracle-q-checkpoint-decode-v3",
                "status": "SIMULATOR_ORACLE_Q_EXACT9_REVIEW_READY",
                "elal_branches_teacher_forced_simulator_oracle_q": True,
                "frozen_base_has_no_elal_q_input": True,
                "source_instruction_inference": False,
                "formal_c1_authorized": False,
                "exact160_authorized": False,
                "real_video_data": False,
                "scientific_claim_authorized": False,
                "action_encoder_qualified": False,
                "decode_release": {
                    "manifest_path": str(manifest),
                    "manifest_sha256": manifest_sha,
                    "launcher_path": str(launcher),
                    "launcher_sha256": launcher_sha,
                },
                "checkpoint_content_authority": {
                    "manifest_path": str(checkpoint_manifest),
                    "manifest_sha256": checkpoint_manifest_sha,
                    "manifest_size": builder.CHECKPOINT_MANIFEST_SIZE,
                    "row_count": builder.CHECKPOINT_MANIFEST_ROW_COUNT,
                    "ordered_manifest_rows_sha256": "1" * 64,
                    "pre_load_world4_replay": {
                        "stage": "decoder_checkpoint_pre_load",
                        "checkpoint_root": str(checkpoint_root),
                        "checkpoint_content_manifest_sha256": checkpoint_manifest_sha,
                        "row_count": builder.CHECKPOINT_MANIFEST_ROW_COUNT,
                        "content_rows_sha256": "2" * 64,
                        "exact23_full_stable_rehash_by_rank_zero": True,
                        "world_size": 4,
                        "world4_broadcast_identity_verified": True,
                        "world4_rank_receipt_digest_consensus": True,
                        "ordered_world4_rank_receipt_digests": ["3" * 64] * 4,
                    },
                    "final_pre_publish_world4_replay": {
                        "stage": "decoder_checkpoint_final_pre_publish",
                        "checkpoint_root": str(checkpoint_root),
                        "checkpoint_content_manifest_sha256": checkpoint_manifest_sha,
                        "row_count": builder.CHECKPOINT_MANIFEST_ROW_COUNT,
                        "content_rows_sha256": "2" * 64,
                        "exact23_full_stable_rehash_by_rank_zero": True,
                        "world_size": 4,
                        "world4_broadcast_identity_verified": True,
                        "world4_rank_receipt_digest_consensus": True,
                        "ordered_world4_rank_receipt_digests": ["4" * 64] * 4,
                    },
                    "exact23_unchanged_across_runtime": True,
                },
                "media": media,
                "html": {
                    "relative_path": "index.html",
                    "sha256": hashlib.sha256(html_payload).hexdigest(),
                },
            }
            receipt_path = root / "DECODE_RECEIPT.json"

            def run(candidate: dict[str, object]) -> subprocess.CompletedProcess[str]:
                write_receipt(receipt_path, candidate)
                return subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        "-B",
                        "-",
                        str(root),
                        str(manifest),
                        manifest_sha,
                        str(launcher),
                        launcher_sha,
                        str(checkpoint_manifest),
                        checkpoint_manifest_sha,
                        str(checkpoint_root),
                    ],
                    input=postflight,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

            accepted = run(receipt)
            self.assertEqual(accepted.returncode, 0, accepted.stderr)

            bad_backlink = json.loads(json.dumps(receipt))
            bad_backlink["decode_release"]["manifest_sha256"] = "0" * 64
            rejected_backlink = run(bad_backlink)
            self.assertNotEqual(rejected_backlink.returncode, 0)
            self.assertIn("release/launcher backlink differs", rejected_backlink.stderr)

            bad_filename = json.loads(json.dumps(receipt))
            bad_filename["media"][0]["relative_path"] = "00_trained_role_swap_q.mp4"
            rejected_filename = run(bad_filename)
            self.assertNotEqual(rejected_filename.returncode, 0)
            self.assertIn("media filename differs", rejected_filename.stderr)

            bad_checkpoint_replay = json.loads(json.dumps(receipt))
            bad_checkpoint_replay["checkpoint_content_authority"][
                "final_pre_publish_world4_replay"
            ]["content_rows_sha256"] = "5" * 64
            rejected_checkpoint_replay = run(bad_checkpoint_replay)
            self.assertNotEqual(rejected_checkpoint_replay.returncode, 0)
            self.assertIn(
                "checkpoint content changed across runtime",
                rejected_checkpoint_replay.stderr,
            )

            manifest.write_bytes(b"manifest changed after decode\n")
            rejected_terminal_drift = run(receipt)
            self.assertNotEqual(rejected_terminal_drift.returncode, 0)
            self.assertIn("decode manifest terminal digest differs", rejected_terminal_drift.stderr)

        launcher_source = LAUNCHER_PATH.read_text(encoding="utf-8")
        self.assertIn("receipt.get(\"decode_release\")", launcher_source)
        self.assertNotIn("receipt.get(\"decode_release_manifest\")", launcher_source)
        self.assertNotIn("receipt.get(\"decode_launcher\")", launcher_source)
        self.assertIn('f"{index:02d}_{order[index]}.mp4"', launcher_source)


if __name__ == "__main__":
    unittest.main()
