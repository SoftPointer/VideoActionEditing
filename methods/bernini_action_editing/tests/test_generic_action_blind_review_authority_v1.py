from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


METHOD_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = METHOD_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import build_generic_action_phi_v1_authority_release_v1 as release  # noqa: E402
import generic_action_blind_review_authority_v1 as authority  # noqa: E402


AUTHORING = METHOD_ROOT / "assets/pair_v5_t2v_calibration_first8_authoring_v1.json"
POPULATION = METHOD_ROOT / "assets/mosaic_event_population_compact6_topup20_v1.json"
PHASE_LABELS = ["onset", "transition", "terminal"] + ["hold"] * 18


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: dict) -> None:
    path.write_bytes(authority.canonical_json_bytes(value) + b"\n")


def _probe() -> dict:
    return {
        "decoder": "tools.materialize_vae._decode_exact_video",
        "decoder_source_sha256": "d" * 64,
        "all_integer_frames_0_through_80_decoded": True,
        "frame_count": 81, "fps": 25, "height": 32, "width": 48,
        "channels": 3, "dtype": "uint8",
    }


class BlindReviewAuthorityTests(unittest.TestCase):
    def _release(self, root: Path) -> tuple[Path, str]:
        archive = root / "authority-overlay.tar"
        manifest = root / "authority-overlay.json"
        built = release.build(METHOD_ROOT.resolve(), archive, manifest)
        return manifest, built["manifest_sha256"]

    def _reviewer(self, root: Path) -> tuple[Path, Path, Ed25519PrivateKey]:
        private = Ed25519PrivateKey.generate()
        public_raw = private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        tool = root / "external-reviewer-tool.py"
        tool.write_bytes(b"external blind full81 review tool fixture v1\n")
        unsigned = {
            "schema_version": authority.REVIEWER_AUTHORITY_SCHEMA,
            "authority_id": "reviewer-authority-test-0001",
            "reviewer_id": "external-reviewer-fixture",
            "review_method": "human_blind_video_review_v1",
            "reviewer_tool_source_sha256": _sha(tool),
            "verification_key": {
                "algorithm": "ed25519",
                "public_key_raw_hex": public_raw.hex(),
                "public_key_sha256": hashlib.sha256(public_raw).hexdigest(),
            },
            "independent_of_generation_runner": True,
            "independent_of_packet_builder": True,
            "independent_of_phi_runner": True,
            "private_key_embedded": False,
            "response_key_override_allowed": False,
            "signed_execution_credential_required": True,
        }
        reviewer = {**unsigned, "authority_digest": authority.object_sha256(unsigned)}
        path = root / "reviewer-authority.json"
        _json(path, reviewer)
        return path, tool, private

    def _generation(self, root: Path) -> tuple[list[dict], dict]:
        authoring = json.loads(AUTHORING.read_text())
        population = json.loads(POPULATION.read_text())
        expected, _ = authority._population_context(authoring, population)
        indexed = {}
        for ordinal, row in enumerate(expected):
            media = root / f"media-{ordinal:03d}.mp4"
            media.write_bytes(f"synthetic-test-media-{ordinal:03d}".encode())
            media_sha = _sha(media)
            receipt_path = root / f"receipt-{ordinal:03d}.json"
            receipt_path.write_text("fixture\n")
            receipt = {
                "root_spec_raw_sha256": row["root_spec_raw_sha256"],
                "candidate": {
                    "candidate_id": row["candidate_id"],
                    "semantic_branch": row["branch"],
                    "analysis_split": row["analysis_split"],
                    "seed": row["seed"],
                    "calibration_group_id": f"cell-{row['source_iid']}-s{row['seed']}",
                },
                "artifacts": {"mp4": {"path": str(media), "sha256": media_sha}},
                "_file_sha256": _sha(receipt_path),
            }
            indexed[row["candidate_id"]] = (receipt_path, receipt)
        return expected, indexed

    def _packet(self, root: Path, indexed: dict) -> tuple[Path, Path, dict, Path, str, Ed25519PrivateKey]:
        release_manifest, release_sha = self._release(root)
        reviewer_path, reviewer_tool, private = self._reviewer(root)
        key = root / "blind.key"
        key.write_bytes(bytes(range(32)))
        packet_dir = root / "packet"
        private_map = root / "private-map.json"
        with mock.patch.object(authority, "_scan_generation", return_value=indexed), mock.patch.object(authority, "_probe_full81", return_value=_probe()):
            packet = authority.build_packet(
                authoring_path=AUTHORING, population_path=POPULATION,
                generation_roots=[root], blind_key_path=key,
                packet_builder_execution_id="packet-builder-test-0001",
                reviewer_authority_path=reviewer_path,
                expected_reviewer_authority_sha256=_sha(reviewer_path),
                reviewer_tool_source_artifact=reviewer_tool,
                authority_release_manifest=release_manifest,
                expected_authority_release_manifest_sha256=release_sha,
                public_output_dir=packet_dir, private_map_output=private_map,
                gap_output=root / "packet-gap.json",
            )
        return packet_dir / "packet-manifest.json", private_map, dict(packet), release_manifest, release_sha, private

    def _response(
        self, root: Path, packet_path: Path, private_path: Path,
        packet: dict, signer: Ed25519PrivateKey,
        *, tool_sha_override: str | None = None,
    ) -> Path:
        private = json.loads(private_path.read_text())
        hidden = {row["opaque_id"]: row for row in private["rows"]}
        rows = []
        for public in packet["rows"]:
            branch = hidden[public["opaque_id"]]["branch"]
            rows.append({
                "opaque_id": public["opaque_id"], "media_sha256": public["media_sha256"],
                "entire_exact81_video_viewed": True, "frame_count": 81, "fps": 25,
                "technical_quality_pass": True, "observed_semantic_class": branch,
                "independent_axes": {axis: axis == branch for axis in authority.AXIS_FIELDS},
                "phase_labels": PHASE_LABELS,
            })
        response_unsigned = {
            "schema_version": authority.RESPONSE_SCHEMA,
            "packet_manifest_file_sha256": _sha(packet_path),
            "packet_digest": packet["packet_digest"],
            "rows": rows,
            "sealed_before_phi_extraction": True,
        }
        response_digest = authority.object_sha256(response_unsigned)
        fixed = packet["reviewer_authority"]
        credential_unsigned = {
            "schema_version": authority.EXECUTION_CREDENTIAL_SCHEMA,
            "signature_algorithm": "ed25519",
            "reviewer_authority_file_sha256": fixed["file_sha256"],
            "reviewer_authority_digest": fixed["authority_digest"],
            "packet_manifest_file_sha256": _sha(packet_path),
            "packet_digest": packet["packet_digest"],
            "reviewer_tool_source_sha256": tool_sha_override or fixed["reviewer_tool_source_sha256"],
            "review_execution_id": "external-review-test-0002",
            "response_digest": response_digest,
        }
        credential = {
            **credential_unsigned,
            "signature_hex": signer.sign(authority.canonical_json_bytes(credential_unsigned)).hex(),
        }
        response = {
            **response_unsigned, "response_digest": response_digest,
            "execution_credential": credential,
        }
        path = root / "response.json"
        _json(path, response)
        return path

    def _resign(self, path: Path, packet: dict, packet_path: Path, signer: Ed25519PrivateKey) -> None:
        value = json.loads(path.read_text())
        unsigned = {key: item for key, item in value.items() if key not in {"response_digest", "execution_credential"}}
        digest = authority.object_sha256(unsigned)
        credential = dict(value["execution_credential"])
        credential["response_digest"] = digest
        credential_unsigned = {key: item for key, item in credential.items() if key != "signature_hex"}
        credential["signature_hex"] = signer.sign(authority.canonical_json_bytes(credential_unsigned)).hex()
        value["response_digest"] = digest
        value["execution_credential"] = credential
        self.assertEqual(value["packet_manifest_file_sha256"], _sha(packet_path))
        self.assertEqual(value["packet_digest"], packet["packet_digest"])
        _json(path, value)

    def _ingest(self, root: Path, packet_path: Path, private_path: Path, response: Path, release_manifest: Path, release_sha: str):
        return authority.ingest_external_review(
            packet_manifest=packet_path, expected_packet_sha256=_sha(packet_path),
            private_map=private_path, expected_private_map_sha256=_sha(private_path),
            external_response=response, expected_response_sha256=_sha(response),
            authority_release_manifest=release_manifest,
            expected_authority_release_manifest_sha256=release_sha,
            output_dir=root / "receipts", authority_output=root / "authority.json",
            gap_output=root / "review-gap.json",
        )

    def test_core4_only_is_exactly_half_and_fails_before_packet(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            expected, indexed = self._generation(root)
            core4 = {key: value for key, value in indexed.items() if value[1]["root_spec_raw_sha256"] in {authority.PROFILE_PINS["core4-v2"]["seed1"], authority.PROFILE_PINS["core4-v2"]["seed2"]}}
            self.assertEqual(len(core4), 80)
            release_manifest, release_sha = self._release(root)
            reviewer_path, reviewer_tool, _ = self._reviewer(root)
            key = root / "blind.key"; key.write_bytes(b"k" * 32)
            with mock.patch.object(authority, "_scan_generation", return_value=core4), self.assertRaisesRegex(authority.BlindReviewAuthorityError, "full-first8 generation closure"):
                authority.build_packet(authoring_path=AUTHORING, population_path=POPULATION, generation_roots=[root], blind_key_path=key, packet_builder_execution_id="packet-builder-test-0001", reviewer_authority_path=reviewer_path, expected_reviewer_authority_sha256=_sha(reviewer_path), reviewer_tool_source_artifact=reviewer_tool, authority_release_manifest=release_manifest, expected_authority_release_manifest_sha256=release_sha, public_output_dir=root / "packet", private_map_output=root / "private.json", gap_output=root / "gap.json")
            gap = json.loads((root / "gap.json").read_text())
            self.assertEqual((gap["existing_core4_expected"], gap["reserve4_expected"], gap["full_first8_expected"]), (80, 80, 160))
            self.assertEqual(gap["observed_expected_candidate_count"], 80)
            self.assertEqual(len(gap["missing_candidate_ids"]), 80)
            self.assertFalse((root / "packet").exists())
            self.assertEqual(len(expected), 160)

    def test_full160_signed_external_ingestion_and_full_decode_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            expected, indexed = self._generation(root)
            packet_path, private_path, packet, release_manifest, release_sha, signer = self._packet(root, indexed)
            public_bytes = packet_path.read_bytes() + (root / "packet" / "blind-review.html").read_bytes()
            self.assertEqual(packet["scope_counts"], {"core4": 80, "reserve4": 80, "fit": 80, "confirmation": 80})
            self.assertEqual(packet["row_count"], 160)
            for row in expected:
                self.assertNotIn(row["candidate_id"].encode(), public_bytes)
            response = self._response(root, packet_path, private_path, packet, signer)
            with mock.patch.object(authority, "_probe_full81", return_value=_probe()) as decode:
                value = self._ingest(root, packet_path, private_path, response, release_manifest, release_sha)
                replayed = authority.load_authority(
                    root / "authority.json", _sha(root / "authority.json"),
                    authority_release_manifest=release_manifest,
                    expected_authority_release_manifest_sha256=release_sha,
                )
            self.assertGreaterEqual(decode.call_count, 320)
            self.assertEqual(value["row_count"], 160)
            self.assertEqual(len(value["rows"]), 160)
            self.assertEqual(sum(row["analysis_split"] == "fit" for row in value["rows"]), 80)
            self.assertEqual(replayed["authority_digest"], value["authority_digest"])
            self.assertEqual(value["reviewer"]["reviewer_tool_source_sha256"], packet["reviewer_authority"]["reviewer_tool_source_sha256"])
            self.assertFalse(value["same_runner_self_certification"])

    def test_self_signed_response_with_unregistered_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            _, indexed = self._generation(root)
            packet_path, private_path, packet, release_manifest, release_sha, _ = self._packet(root, indexed)
            response = self._response(root, packet_path, private_path, packet, Ed25519PrivateKey.generate())
            with mock.patch.object(authority, "_probe_full81", return_value=_probe()), self.assertRaisesRegex(authority.BlindReviewAuthorityError, "signature verification failed"):
                self._ingest(root, packet_path, private_path, response, release_manifest, release_sha)

    def test_random_response_reported_tool_hash_is_rejected_even_when_signed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            _, indexed = self._generation(root)
            packet_path, private_path, packet, release_manifest, release_sha, signer = self._packet(root, indexed)
            response = self._response(root, packet_path, private_path, packet, signer, tool_sha_override="f" * 64)
            with mock.patch.object(authority, "_probe_full81", return_value=_probe()), self.assertRaisesRegex(authority.BlindReviewAuthorityError, "credential binding differs"):
                self._ingest(root, packet_path, private_path, response, release_manifest, release_sha)

    def test_unprovisioned_reviewer_template_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            _, indexed = self._generation(root)
            release_manifest, release_sha = self._release(root)
            template = root / "reviewer-template.json"; _json(template, dict(authority.reviewer_authority_template()))
            tool = root / "tool.py"; tool.write_text("fixture\n")
            key = root / "blind.key"; key.write_bytes(b"k" * 32)
            with mock.patch.object(authority, "_scan_generation", return_value=indexed), self.assertRaises(authority.BlindReviewAuthorityError):
                authority.build_packet(authoring_path=AUTHORING, population_path=POPULATION, generation_roots=[root], blind_key_path=key, packet_builder_execution_id="packet-builder-test-0001", reviewer_authority_path=template, expected_reviewer_authority_sha256=_sha(template), reviewer_tool_source_artifact=tool, authority_release_manifest=release_manifest, expected_authority_release_manifest_sha256=release_sha, public_output_dir=root / "packet", private_map_output=root / "private.json", gap_output=root / "gap.json")
            self.assertIsNone(authority.reviewer_authority_template()["verification_key"]["public_key_raw_hex"])
            self.assertFalse((root / "packet").exists())

    def test_camera_and_appearance_axes_cannot_be_mixed_or_weighted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            _, indexed = self._generation(root)
            packet_path, private_path, packet, release_manifest, release_sha, signer = self._packet(root, indexed)
            response = self._response(root, packet_path, private_path, packet, signer)
            value = json.loads(response.read_text())
            value["rows"][0]["observed_semantic_class"] = "camera_only"
            value["rows"][0]["independent_axes"] = {axis: axis in {"camera_only", "appearance_only"} for axis in authority.AXIS_FIELDS}
            _json(response, value); self._resign(response, packet, packet_path, signer)
            with mock.patch.object(authority, "_probe_full81", return_value=_probe()), self.assertRaisesRegex(authority.BlindReviewAuthorityError, "mixed or weighted"):
                self._ingest(root, packet_path, private_path, response, release_manifest, release_sha)

    def test_one_failed_quality_row_emits_gap_and_no_partial_authority(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            _, indexed = self._generation(root)
            packet_path, private_path, packet, release_manifest, release_sha, signer = self._packet(root, indexed)
            response = self._response(root, packet_path, private_path, packet, signer)
            value = json.loads(response.read_text()); value["rows"][0]["technical_quality_pass"] = False
            _json(response, value); self._resign(response, packet, packet_path, signer)
            with mock.patch.object(authority, "_probe_full81", return_value=_probe()), self.assertRaisesRegex(authority.BlindReviewAuthorityError, "failed/unjudgeable/mismatched"):
                self._ingest(root, packet_path, private_path, response, release_manifest, release_sha)
            gap = json.loads((root / "review-gap.json").read_text())
            self.assertEqual(len(gap["failed_review_opaque_ids"]), 1)
            self.assertFalse((root / "receipts").exists())
            self.assertFalse((root / "authority.json").exists())

    def test_release_closure_failure_precedes_generation_scan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            key = root / "blind.key"; key.write_bytes(b"k" * 32)
            with mock.patch.object(authority.authority_release, "validate_installed_closure", side_effect=release.PhiAuthorityReleaseError("closure-first")), mock.patch.object(authority, "_scan_generation") as scan, self.assertRaisesRegex(authority.BlindReviewAuthorityError, "closure-first"):
                authority.build_packet(authoring_path=AUTHORING, population_path=POPULATION, generation_roots=[root], blind_key_path=key, packet_builder_execution_id="packet-builder-test-0001", reviewer_authority_path=root / "missing.json", expected_reviewer_authority_sha256="0" * 64, reviewer_tool_source_artifact=root / "missing.py", authority_release_manifest=root / "missing-release.json", expected_authority_release_manifest_sha256="0" * 64, public_output_dir=root / "packet", private_map_output=root / "private.json", gap_output=root / "gap.json")
            scan.assert_not_called()

    def test_release_closure_failure_precedes_ingest_and_authority_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with mock.patch.object(authority.authority_release, "validate_installed_closure", side_effect=release.PhiAuthorityReleaseError("closure-first")), mock.patch.object(authority, "_validate_packet") as packet_replay:
                with self.assertRaisesRegex(authority.BlindReviewAuthorityError, "closure-first"):
                    authority.ingest_external_review(packet_manifest=root / "missing-packet.json", expected_packet_sha256="0" * 64, private_map=root / "missing-private.json", expected_private_map_sha256="0" * 64, external_response=root / "missing-response.json", expected_response_sha256="0" * 64, authority_release_manifest=root / "missing-release.json", expected_authority_release_manifest_sha256="0" * 64, output_dir=root / "receipts", authority_output=root / "authority.json", gap_output=root / "gap.json")
                with self.assertRaisesRegex(authority.BlindReviewAuthorityError, "closure-first"):
                    authority.load_authority(root / "missing-authority.json", "0" * 64, authority_release_manifest=root / "missing-release.json", expected_authority_release_manifest_sha256="0" * 64)
            packet_replay.assert_not_called()


if __name__ == "__main__":
    unittest.main()
